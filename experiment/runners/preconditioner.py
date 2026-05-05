"""Spectral preconditioner experiment (the algorithmic payoff).

The kBM ODE dG/dt = -2(K E G + G E K) is kernel-preconditioned. Theorem 6
shows modes in K's bottom eigenspace evolve at rate ~ lambda^2, so hard
triplets there are frozen.

To remove the bias, we precondition the embedding-side gradient:

  Standard PyTorch: dL/d_theta = J^T @ vec(dL/dF)
  Preconditioned:   dL/d_theta = J^T @ vec(K^{-1} @ dL/dF)

The K^{-1} operates on the data axis of dL/dF (an N x M matrix). Under
the kBM derivation, this exactly cancels the K-prefactor in the Gram
matrix dynamics:

  dG/dt = -2 K [K^{-1} E] G + ... = -2 E G + ...

So the resulting effective dynamics are the unbiased BM flow.

Predictions (compared to vanilla NTK training):
  (a) Spectral bias of |G_tilde[i,i] - G_tilde[i,i](0)| vs lambda_i is
      flattened (slope drops from ~2 to ~0).
  (b) Bottom-eigenspace triplets get fit faster.
  (c) Final triplet-satisfaction rate is higher with the same number of
      epochs.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from kbm.kernels import standard_ntk_kernel
from kbm.data import generate_synthetic_data, generate_fixed_triplets
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model


def train_preconditioned(
    model: torch.nn.Module,
    X: np.ndarray,
    triplets,
    K: np.ndarray,
    n_epochs: int,
    lr: float,
    margin: float,
    record_every: int,
    eps: float = 1e-3,
    device: str = "cpu",
    alpha: float = 1.0,
):
    """Custom training loop applying $K^{-\\alpha}$ to the embedding-side gradient.

    For alpha=1 this is the canonical kBM correction. For alpha=0 the routine
    reduces to vanilla training (preconditioner = I); intermediate alphas
    sweep the strength of the correction.
    """
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    K_reg = K_t + eps * torch.eye(K_t.shape[0], device=device)
    if abs(alpha - 1.0) < 1e-9:
        K_pre = torch.linalg.solve(K_reg, torch.eye(K_reg.shape[0], device=device))
    elif abs(alpha) < 1e-9:
        K_pre = torch.eye(K_reg.shape[0], device=device)
    else:
        # K^{-alpha} via eigendecomposition (K is symmetric PSD).
        evals, evecs = torch.linalg.eigh(K_reg)
        K_pre = evecs @ torch.diag(evals.clamp(min=eps).pow(-alpha)) @ evecs.T
    K_inv = K_pre  # variable name kept for downstream usage
    model.to(device)

    triplet_arr = np.asarray(triplets, dtype=np.int64)
    a_idx = torch.as_tensor(triplet_arr[:, 0], device=device)
    p_idx = torch.as_tensor(triplet_arr[:, 1], device=device)
    n_idx = torch.as_tensor(triplet_arr[:, 2], device=device)
    margin_t = torch.tensor(float(margin), device=device)

    epochs_recorded, G_history, losses = [], [], []

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        F = model(Xt)
        F.retain_grad()
        G = F @ F.T
        diag = torch.diagonal(G)
        d_ap = diag[a_idx] + diag[p_idx] - 2.0 * G[a_idx, p_idx]
        d_an = diag[a_idx] + diag[n_idx] - 2.0 * G[a_idx, n_idx]
        per_t = torch.clamp(d_ap - d_an + margin_t, min=0.0)
        loss = per_t.sum()

        # Get dL/dF directly without touching parameters yet.
        grad_F = torch.autograd.grad(loss, F, create_graph=False, retain_graph=False)[0]
        # Apply K^{-1} to the data axis (N x M -> N x M).
        grad_F_precond = K_inv @ grad_F

        # Now propagate the modified upstream gradient through the model.
        F2 = model(Xt)
        F2.backward(grad_F_precond)
        opt.step()

        if epoch % record_every == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                F_ = model(Xt)
                G_ = (F_ @ F_.T).cpu().numpy()
            epochs_recorded.append(epoch)
            G_history.append(G_)
            losses.append(float(loss.item()))

    return {
        "epochs": np.array(epochs_recorded),
        "G_history": G_history,
        "losses": np.array(losses),
    }


@dataclass
class PreconditionerConfig:
    N: int = 50
    d_in: int = 20
    rank_true: int = 5
    depth: int = 1
    width: int = 1000
    n_triplets_factor: int = 8
    n_epochs: int = 4000
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 100
    eps: float = 1e-3


def _triplet_satisfactions_by_mode(
    G: np.ndarray, triplets, U: np.ndarray, eigvals_K: np.ndarray, margin: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """For each triplet, decompose its loss-gradient direction in K's
    eigenbasis and report which spectral mode dominates.

    Returns (per_triplet_satisfaction, per_triplet_dominant_mode).
    """
    N = G.shape[0]
    diag = np.diag(G)
    sats = []
    dom_modes = []
    triplet_arr = np.asarray(triplets)
    for a, p, n in triplet_arr:
        d_ap = diag[a] + diag[p] - 2.0 * G[a, p]
        d_an = diag[a] + diag[n] - 2.0 * G[a, n]
        sats.append(1 if d_ap - d_an + margin <= 0 else 0)
        # Each triplet's E-contribution in matrix form (rank-2 update).
        E_t = np.zeros((N, N))
        E_t[p, p] += 1; E_t[n, n] -= 1
        E_t[a, p] -= 1; E_t[p, a] -= 1
        E_t[a, n] += 1; E_t[n, a] += 1
        # Project to K's eigenbasis and find dominant mode.
        Et = U.T @ E_t @ U
        per_mode = np.abs(np.diag(Et)) * eigvals_K  # weighted by lambda
        dom_modes.append(int(np.argmax(per_mode)))
    return np.array(sats), np.array(dom_modes)


def run_one(seed: int, cfg: PreconditionerConfig, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, _, _ = generate_synthetic_data(cfg.N, cfg.d_in, cfg.rank_true, seed=seed)
    triplets = generate_fixed_triplets(cfg.N, cfg.N * cfg.n_triplets_factor, seed=seed)

    K = standard_ntk_kernel(X)
    eigvals_K, U = np.linalg.eigh(K)
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]; U = U[:, order]

    # Initial state shared between the two runs
    torch.manual_seed(seed); np.random.seed(seed)
    model_init = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    F0 = embeddings_from_model(model_init, X, device="cpu")
    G0 = F0 @ F0.T
    G0_modes = np.diag(U.T @ G0 @ U)

    # Vanilla training
    torch.manual_seed(seed); np.random.seed(seed)
    model_v = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    out_v = train_network(
        model_v, X, triplets,
        TrainConfig(n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
                    record_every=cfg.record_every),
        device=device,
    )

    # Preconditioned training: K^{-1} applied to embedding-side gradient
    torch.manual_seed(seed); np.random.seed(seed)
    model_p = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    out_p = train_preconditioned(
        model_p, X, triplets, K,
        n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
        record_every=cfg.record_every, eps=cfg.eps, device=device,
    )

    # Project both trajectories into K's eigenbasis (diagonal entries).
    def proj_diag(hist):
        return np.stack([np.diag(U.T @ G @ U) for G in hist], axis=0)

    G_modes_v = proj_diag(out_v["G_history"])  # (T, N)
    G_modes_p = proj_diag(out_p["G_history"])
    delta_v = G_modes_v - G0_modes[None, :]
    delta_p = G_modes_p - G0_modes[None, :]

    # Per-mode satisfaction at end of training, binned by K-eigenmode rank.
    G_v_final = out_v["G_history"][-1]
    G_p_final = out_p["G_history"][-1]
    sats_v, dom_v = _triplet_satisfactions_by_mode(G_v_final, triplets, U, eigvals_K, cfg.margin)
    sats_p, dom_p = _triplet_satisfactions_by_mode(G_p_final, triplets, U, eigvals_K, cfg.margin)
    n_bins = 5
    bin_edges = np.linspace(0, cfg.N, n_bins + 1).astype(int)
    sat_by_bin_v, sat_by_bin_p = [], []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask_v = (dom_v >= lo) & (dom_v < hi)
        mask_p = (dom_p >= lo) & (dom_p < hi)
        sat_by_bin_v.append(float(sats_v[mask_v].mean()) if mask_v.any() else float('nan'))
        sat_by_bin_p.append(float(sats_p[mask_p].mean()) if mask_p.any() else float('nan'))

    return {
        "seed": seed,
        "epochs": out_v["epochs"].tolist(),
        "vanilla_loss": out_v["losses"].tolist(),
        "precond_loss": out_p["losses"].tolist(),
        "eigvals_K": eigvals_K.tolist(),
        "G0_modes": G0_modes.tolist(),
        "delta_vanilla": delta_v.tolist(),
        "delta_precond": delta_p.tolist(),
        "sat_by_bin_vanilla": sat_by_bin_v,
        "sat_by_bin_precond": sat_by_bin_p,
        "overall_sat_vanilla": float(sats_v.mean()),
        "overall_sat_precond": float(sats_p.mean()),
        "config": asdict(cfg),
    }


def run_sweep(cfg: PreconditionerConfig, seeds: list[int], out_path: Path, device: str = "cpu") -> list[dict]:
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
