"""Bottom-eigenspace target experiment — the cleanest possible test.

We construct a target Gram matrix G* whose mass is concentrated in K's
bottom eigenspace (small-lambda modes). Theorem 7 predicts: under
vanilla training, G(t) cannot reach G* on bounded horizons because
bottom-eigenspace projections are λ²-suppressed; under K^{-1}-corrected
training, the suppression is removed and G(t) can reach G*.

Effect size should be dramatic: vanilla loss plateaus near the
initialization residual; corrected loss decreases substantially.

Triplets are sampled from G*: for triplet (a, p, n), require
G*_pp + G*_aa - 2 G*_ap < G*_aa + G*_nn - 2 G*_an, i.e., positive
closer than negative under G*. The training task is to fit a Gram
matrix that satisfies these triplets, which by construction requires
moving into K's bottom eigenspace.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from kbm.kernels import standard_ntk_kernel
from kbm.data import generate_synthetic_data
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model
from runners.preconditioner import train_preconditioned


@dataclass
class BottomTargetConfig:
    N: int = 60
    d_in: int = 20
    rank_true: int = 5
    depth: int = 1
    width: int = 1000
    n_triplets: int = 600
    n_epochs: int = 4000
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 100
    eps: float = 1e-3
    bottom_modes: int = 10  # how many bottom eigenmodes to load the target on


def _construct_bottom_target(K: np.ndarray, n_modes: int, scale: float = 5.0) -> np.ndarray:
    """Build G* whose support is exactly K's bottom n_modes eigenspace.

    G* = U_bottom diag(d) U_bottom^T where U_bottom holds the eigenvectors
    of K with the smallest eigenvalues. d is positive so G* is PSD.
    """
    eigvals, U = np.linalg.eigh(K)
    order = np.argsort(eigvals)  # ascending
    bottom = U[:, order[:n_modes]]
    d = scale * np.linspace(2.0, 1.0, n_modes)
    return bottom @ np.diag(d) @ bottom.T


def _sample_triplets_from_target(G_star: np.ndarray, n_triplets: int, seed: int):
    """Triplets where positive is closer than negative under G*'s induced
    distance: d*_ap = G*_pp + G*_aa - 2 G*_ap < d*_an."""
    rng = np.random.default_rng(seed)
    N = G_star.shape[0]
    diag = np.diag(G_star)
    triplets = []
    attempts = 0
    while len(triplets) < n_triplets and attempts < n_triplets * 50:
        a, p, n = rng.choice(N, size=3, replace=False)
        d_ap = diag[p] + diag[a] - 2.0 * G_star[a, p]
        d_an = diag[a] + diag[n] - 2.0 * G_star[a, n]
        if d_ap < d_an:
            triplets.append((int(a), int(p), int(n)))
        attempts += 1
    return triplets


def _frob_residual(G: np.ndarray, G_star: np.ndarray) -> float:
    return float(np.linalg.norm(G - G_star, ord="fro"))


def _modewise_overlap(G: np.ndarray, G_star: np.ndarray, U: np.ndarray) -> dict:
    """Fraction of G's mass aligned with G*'s support eigenspace."""
    Gtilde = U.T @ G @ U
    Gstar_t = U.T @ G_star @ U
    # Both diagonal in U's basis (target by construction; G approximately)
    return {
        "diag_dot": float(np.dot(np.diag(Gtilde), np.diag(Gstar_t))),
        "norm_ratio": float(np.linalg.norm(Gtilde) / (np.linalg.norm(Gstar_t) + 1e-9)),
    }


def run_one(seed: int, cfg: BottomTargetConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    X, _, _ = generate_synthetic_data(cfg.N, cfg.d_in, cfg.rank_true, seed=seed)
    K = standard_ntk_kernel(X)
    eigvals_K, U = np.linalg.eigh(K)
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]; U = U[:, order]

    G_star = _construct_bottom_target(K, cfg.bottom_modes)
    triplets = _sample_triplets_from_target(G_star, cfg.n_triplets, seed)
    if len(triplets) < cfg.n_triplets // 2:
        # Bottom-eigenspace target may be hard to satisfy; fall back to
        # whatever triplets we found.
        print(f"  warning: only sampled {len(triplets)} triplets", flush=True)

    torch.manual_seed(seed); np.random.seed(seed)
    m_init = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    F0 = embeddings_from_model(m_init, X, device="cpu")
    G0 = F0 @ F0.T

    # Vanilla
    torch.manual_seed(seed); np.random.seed(seed)
    m_v = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    out_v = train_network(
        m_v, X, triplets,
        TrainConfig(n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
                    record_every=cfg.record_every),
        device=device,
    )

    # Corrected
    torch.manual_seed(seed); np.random.seed(seed)
    m_p = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    out_p = train_preconditioned(
        m_p, X, triplets, K,
        n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
        record_every=cfg.record_every, eps=cfg.eps, device=device,
    )

    Gv_final = out_v["G_history"][-1]
    Gp_final = out_p["G_history"][-1]

    return {
        "seed": seed,
        "n_triplets_sampled": len(triplets),
        "frob_residual_init": _frob_residual(G0, G_star),
        "frob_residual_vanilla": _frob_residual(Gv_final, G_star),
        "frob_residual_precond": _frob_residual(Gp_final, G_star),
        "frac_residual_vanilla": _frob_residual(Gv_final, G_star) / _frob_residual(G0, G_star),
        "frac_residual_precond": _frob_residual(Gp_final, G_star) / _frob_residual(G0, G_star),
        "vanilla_final_loss": float(out_v["losses"][-1]),
        "precond_final_loss": float(out_p["losses"][-1]),
        "lambda_max": float(eigvals_K[0]),
        "lambda_min": float(eigvals_K[-1]),
        "lambda_at_bottom_target": [float(eigvals_K[-(i + 1)]) for i in range(cfg.bottom_modes)],
        "config": asdict(cfg),
    }


def run_sweep(cfg: BottomTargetConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
