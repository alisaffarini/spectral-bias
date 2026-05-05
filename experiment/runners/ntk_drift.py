"""Empirical NTK drift experiment: does the spectral bias signature
persist as the empirical NTK evolves during finite-width training?

A reviewer concern is that the end-to-end CNN result (Section~4.7)
uses K^emp at initialization only; literature shows the empirical NTK
can drift substantially during finite-width training. We probe this
directly: train the CNN end-to-end, recompute K^emp at multiple
checkpoints, and at each checkpoint fit the spectral slope against the
*current* K^emp (the theoretically-correct object).

If the spectral-bias signature persists across checkpoints, the
phenomenon is not an artifact of the strict lazy regime. If it
dissolves, that bounds the kBM theory's empirical applicability.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from runners.cifar10_end2end import (
    SmallCNN, End2EndConfig, _load_cifar10_subset, _make_triplets,
    _empirical_ntk, _train_cnn,
)


@dataclass
class DriftConfig:
    N: int = 80               # smaller than end2end's 100 to keep NTK runs tractable
    n_classes: int = 10
    M_out: int = 32
    n_triplets: int = 480
    n_epochs: int = 1500
    lr: float = 1e-6
    margin: float = 1.0
    eps: float = 1e-2
    data_root: str = "./data"
    n_checkpoints: int = 5      # epochs at 0, T/4, T/2, 3T/4, T
    n_seeds: int = 3            # NTK computation is expensive; small seed count


def _fit_slope(eigvals: np.ndarray, deltas: np.ndarray) -> float:
    eigvals = np.asarray(eigvals); deltas = np.abs(np.asarray(deltas))
    mask = (eigvals > 1e-3) & (deltas > 1e-8)
    if mask.sum() < 5:
        return float("nan")
    return float(np.polyfit(np.log10(eigvals[mask]),
                             np.log10(deltas[mask]), 1)[0])


def run_one(seed: int, cfg: DriftConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    X, labels = _load_cifar10_subset(End2EndConfig(N=cfg.N, M_out=cfg.M_out,
                                                   data_root=cfg.data_root), seed)
    triplets = _make_triplets(labels, cfg.n_triplets, seed)
    Xt = X.to(device)

    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN(M_out=cfg.M_out).to(device)

    # Snapshot the trained model at evenly-spaced checkpoints by manually
    # stepping the training loop and saving state_dicts.
    checkpoints = np.linspace(0, cfg.n_epochs - 1, cfg.n_checkpoints).astype(int)
    saved_states = {}
    saved_G = {}

    triplet_arr = np.asarray(triplets, dtype=np.int64)
    a_idx = torch.as_tensor(triplet_arr[:, 0], device=device)
    p_idx = torch.as_tensor(triplet_arr[:, 1], device=device)
    n_idx = torch.as_tensor(triplet_arr[:, 2], device=device)
    margin_t = torch.tensor(float(cfg.margin), device=device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    # Save G(0) for displacement reference.
    with torch.no_grad():
        F0 = model(Xt)
        G0 = (F0 @ F0.T).cpu().numpy()

    for epoch in range(cfg.n_epochs):
        if epoch in checkpoints:
            saved_states[int(epoch)] = {k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()}
            with torch.no_grad():
                F_ = model(Xt)
                saved_G[int(epoch)] = (F_ @ F_.T).cpu().numpy()
        model.train()
        opt.zero_grad()
        F = model(Xt)
        G = F @ F.T
        diag = torch.diagonal(G)
        d_ap = diag[a_idx] + diag[p_idx] - 2.0 * G[a_idx, p_idx]
        d_an = diag[a_idx] + diag[n_idx] - 2.0 * G[a_idx, n_idx]
        per_t = torch.clamp(d_ap - d_an + margin_t, min=0.0)
        loss = per_t.sum()
        loss.backward()
        opt.step()

    if int(cfg.n_epochs - 1) not in saved_states:
        saved_states[int(cfg.n_epochs - 1)] = {k: v.detach().cpu().clone()
                                                for k, v in model.state_dict().items()}
        with torch.no_grad():
            F_ = model(Xt)
            saved_G[int(cfg.n_epochs - 1)] = (F_ @ F_.T).cpu().numpy()

    # At each checkpoint, recompute K^emp and fit slope of (G(t) - G(0))
    # in K^emp(t)'s eigenbasis.
    per_checkpoint = []
    for ep in sorted(saved_states.keys()):
        model_at_t = SmallCNN(M_out=cfg.M_out).to(device)
        model_at_t.load_state_dict(saved_states[ep])
        print(f"[seed {seed}, epoch {ep}] computing empirical NTK...", flush=True)
        K_t = _empirical_ntk(model_at_t, X, device=device)
        eigvals, U = np.linalg.eigh(K_t)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]; U = U[:, order]

        G_t = saved_G[ep]
        delta_modes = np.diag(U.T @ (G_t - G0) @ U)
        slope = _fit_slope(eigvals, delta_modes)

        per_checkpoint.append({
            "epoch": int(ep),
            "slope_at_current_K": slope,
            "lambda_max": float(eigvals[0]),
            "lambda_min": float(eigvals[-1]),
        })
        print(f"  slope (vs current K^emp) = {slope:.3f}", flush=True)

    return {
        "seed": seed,
        "checkpoints": per_checkpoint,
        "config": asdict(cfg),
    }


def run_sweep(cfg: DriftConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
