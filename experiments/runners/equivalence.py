"""Equivalence experiment: network vs kBM ODE from matched initial state.

This is the core empirical content supporting Theorem 1 (kBM equivalence) and
its depth-L extension. We track G_network(t) and G_ode(t) starting from the
same initial Gram matrix and report the relative Frobenius error trajectory.
Multi-seed; multi-depth; multi-N.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from kbm.kernels import standard_ntk_kernel, depth_L_ntk_kernel
from kbm.data import generate_synthetic_data, generate_fixed_triplets
from kbm.dynamics import run_kbm_ode
from kbm.network import MetricNetwork, train_network, embeddings_from_model, TrainConfig
from kbm.metrics import relative_frob


@dataclass
class EquivConfig:
    N: int = 10
    d_in: int = 20
    rank_true: int = 3
    depth: int = 1
    width: int = 1000
    n_triplets_factor: int = 5
    n_epochs: int = 200
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 4


def run_one(seed: int, cfg: EquivConfig, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, _, _ = generate_synthetic_data(cfg.N, cfg.d_in, cfg.rank_true, seed=seed)
    triplets = generate_fixed_triplets(cfg.N, cfg.N * cfg.n_triplets_factor, seed=seed)

    widths = [cfg.width] * cfg.depth
    model = MetricNetwork(cfg.d_in, widths)

    F0 = embeddings_from_model(model, X, device="cpu")
    G0 = F0 @ F0.T

    if cfg.depth == 1:
        K = standard_ntk_kernel(X)
    else:
        K = depth_L_ntk_kernel(X, cfg.depth)

    train_cfg = TrainConfig(
        n_epochs=cfg.n_epochs,
        lr=cfg.lr,
        margin=cfg.margin,
        record_every=cfg.record_every,
    )
    out = train_network(model, X, triplets, train_cfg, device=device)

    n_record = len(out["epochs"])
    times, ode_history, ode_losses = run_kbm_ode(
        G0, K, triplets,
        total_time=cfg.n_epochs * cfg.lr,
        n_steps=cfg.n_epochs,
        margin=cfg.margin,
        record_every=cfg.record_every,
    )
    n_record_eff = min(n_record, len(ode_history))

    rel_errors = []
    for k in range(n_record_eff):
        rel_errors.append(relative_frob(out["G_history"][k], ode_history[k]))

    return {
        "seed": seed,
        "epochs": out["epochs"][:n_record_eff].tolist(),
        "rel_errors": rel_errors,
        "network_loss": out["losses"][:n_record_eff].tolist(),
        "ode_loss": ode_losses[:n_record_eff].tolist(),
        "config": asdict(cfg),
    }


def run_sweep(cfg: EquivConfig, seeds: list[int], out_path: Path, device: str = "cpu") -> list[dict]:
    results = []
    for s in seeds:
        results.append(run_one(s, cfg, device=device))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
