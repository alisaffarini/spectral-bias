"""Spectral implicit-bias experiment (the payoff).

Theorem (informal): under kBM dynamics dG/dt = -2(K E G + G E K), if we
diagonalize K = U Lambda U^T and write E_tilde = U^T E U, G_tilde = U^T G U,
then in directions where Lambda has small eigenvalue the gradient force
KEG + GEK is small (suppressed by Lambda), so the corresponding components of
G evolve slowly. Components in the top-eigenvalue subspace evolve fast.

This experiment verifies the prediction:
  1. Diagonalize K = U Lambda U^T at initialization.
  2. Track the projected error |U^T (G(t) - G*) U|_diag mode-by-mode.
  3. Check the per-mode decay rate scales with Lambda_i (faster modes have
     larger Lambda_i).

If the prediction holds, the kBM theory yields a non-trivial implicit-bias
result: the network only fits triplet structure that lives in K's top
eigenspace at the rate one would naively expect from gradient flow; structure
in the bottom eigenspace is effectively frozen.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from kbm.kernels import standard_ntk_kernel, depth_L_ntk_kernel
from kbm.data import generate_synthetic_data, generate_fixed_triplets
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model


@dataclass
class SpectralConfig:
    N: int = 30
    d_in: int = 20
    rank_true: int = 5
    depth: int = 1
    width: int = 1000
    n_triplets_factor: int = 8
    n_epochs: int = 4000
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 50


def run_one(seed: int, cfg: SpectralConfig, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, _, _ = generate_synthetic_data(cfg.N, cfg.d_in, cfg.rank_true, seed=seed)
    triplets = generate_fixed_triplets(cfg.N, cfg.N * cfg.n_triplets_factor, seed=seed)

    widths = [cfg.width] * cfg.depth
    model = MetricNetwork(cfg.d_in, widths)

    if cfg.depth == 1:
        K = standard_ntk_kernel(X)
    else:
        K = depth_L_ntk_kernel(X, cfg.depth)
    eigvals_K, U = np.linalg.eigh(K)
    # Sort descending so index 0 is the top mode.
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]
    U = U[:, order]

    train_cfg = TrainConfig(
        n_epochs=cfg.n_epochs,
        lr=cfg.lr,
        margin=cfg.margin,
        record_every=cfg.record_every,
    )
    F0 = embeddings_from_model(model, X, device="cpu")
    G0 = F0 @ F0.T
    out = train_network(model, X, triplets, train_cfg, device=device)

    # Track mode-wise diag entries of U^T G(t) U through training.
    G_modes_history = []
    for G_t in out["G_history"]:
        Gt = U.T @ G_t @ U
        G_modes_history.append(np.diag(Gt))
    G_modes_history = np.stack(G_modes_history, axis=0)  # (T, N)
    G0_modes = np.diag(U.T @ G0 @ U)

    # Per-mode displacement from initial state, normalized by initial.
    delta = G_modes_history - G0_modes[None, :]

    return {
        "seed": seed,
        "epochs": out["epochs"].tolist(),
        "eigvals_K": eigvals_K.tolist(),
        "G_modes_history": G_modes_history.tolist(),
        "G0_modes": G0_modes.tolist(),
        "delta": delta.tolist(),
        "losses": out["losses"].tolist(),
        "config": asdict(cfg),
    }


def run_sweep(cfg: SpectralConfig, seeds: list[int], out_path: Path, device: str = "cpu") -> list[dict]:
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
