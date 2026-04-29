"""Rank vs width: networks learn rank = N when M >= N (constraint-optimal).

This is a cleaned-up version of the original experiment 7, with multi-seed and
proper CIs. The result remains: triplet loss is rank-agnostic, so when
network capacity allows, the network occupies the full N-dim space.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from kbm.data import generate_synthetic_data, triplets_from_target
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model
from kbm.metrics import effective_rank


@dataclass
class RankConfig:
    N: int = 10
    d_in: int = 20
    rank_true: int = 3
    M_values: tuple[int, ...] = (5, 8, 10, 20, 50, 100, 200)
    n_epochs: int = 4000
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 1000


def run_one(seed: int, cfg: RankConfig, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, G_target, _ = generate_synthetic_data(cfg.N, cfg.d_in, cfg.rank_true, seed=seed)
    triplets = triplets_from_target(G_target, max_triplets=cfg.N * 5)

    ranks, losses, sats = [], [], []
    for M in cfg.M_values:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MetricNetwork(cfg.d_in, [M])
        train_cfg = TrainConfig(
            n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
            record_every=cfg.record_every,
        )
        out = train_network(model, X, triplets, train_cfg, device=device)
        F_final = embeddings_from_model(model, X, device=device)
        G_final = F_final @ F_final.T
        rk = effective_rank(G_final)
        final_loss = float(out["losses"][-1])
        n_sat = 0
        for a, p, n in triplets[:50]:
            d_ap = G_final[a, a] + G_final[p, p] - 2 * G_final[a, p]
            d_an = G_final[a, a] + G_final[n, n] - 2 * G_final[a, n]
            if d_ap - d_an + cfg.margin <= 0:
                n_sat += 1
        sats.append(n_sat / min(50, len(triplets)))
        ranks.append(rk)
        losses.append(final_loss)

    return {
        "seed": seed,
        "M_values": list(cfg.M_values),
        "ranks": ranks,
        "losses": losses,
        "satisfactions": sats,
        "config": asdict(cfg),
    }


def run_sweep(cfg: RankConfig, seeds: list[int], out_path: Path, device: str = "cpu") -> list[dict]:
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
