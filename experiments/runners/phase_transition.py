"""Phase transition in lambda for nuclear-norm regularized training.

Honest framing: in our setup the effective rank drops sharply between two
regimes; we report the empirical transition point with multi-seed CIs.
We do NOT claim a universal critical point unless we can derive lambda*
from problem structure.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from kbm.data import generate_synthetic_data, generate_fixed_triplets
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model
from kbm.metrics import effective_rank


@dataclass
class PhaseConfig:
    N: int = 10
    d_in: int = 20
    rank_true: int = 3
    width: int = 200
    n_epochs: int = 4000
    lr: float = 1e-4
    margin: float = 1.0
    lambdas: tuple[float, ...] = ()  # filled in __post_init__
    record_every: int = 1000


DEFAULT_LAMBDAS = tuple(np.logspace(-3, 1, 25).tolist())


def run_one(seed: int, cfg: PhaseConfig, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    lambdas = cfg.lambdas if cfg.lambdas else DEFAULT_LAMBDAS

    X, _, _ = generate_synthetic_data(cfg.N, cfg.d_in, cfg.rank_true, seed=seed)
    triplets = generate_fixed_triplets(cfg.N, cfg.N * 5, seed=seed)

    ranks, losses = [], []
    for lam in lambdas:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MetricNetwork(cfg.d_in, [cfg.width])
        train_cfg = TrainConfig(
            n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
            record_every=cfg.record_every, nuclear_reg=float(lam),
        )
        out = train_network(model, X, triplets, train_cfg, device=device)
        F_final = embeddings_from_model(model, X, device=device)
        G_final = F_final @ F_final.T
        ranks.append(effective_rank(G_final))
        losses.append(float(out["losses"][-1]))

    return {
        "seed": seed,
        "lambdas": list(lambdas),
        "ranks": ranks,
        "losses": losses,
        "config": asdict(cfg),
    }


def run_sweep(cfg: PhaseConfig, seeds: list[int], out_path: Path, device: str = "cpu") -> list[dict]:
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
