"""Alpha-sweep with retrieval R@K: K^{-alpha} for alpha in {0, 0.5, 1, 1.5, 2}
on hard CIFAR-10 retrieval. Closes the 'is alpha=1 empirically optimal on the
end-task?' question -- the slope-only sweep in app:alpha leaves alpha=0.5 as the
slope-flatness winner, which is reviewer-bait."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from kbm.kernels import depth_L_ntk_kernel
from kbm.network import MetricNetwork, embeddings_from_model
from runners.preconditioner import train_preconditioned
from runners.recall_eval import recall_at_k, _make_triplets, _split_train_test
from runners.recall_scaled import _load_subset, ScaledRecallConfig


@dataclass
class AlphaRetrievalConfig:
    N_train: int = 4000
    N_test: int = 1000
    n_classes: int = 10
    feature_dim: int = 256
    depth: int = 2
    width: int = 1024
    n_triplets: int = 12000
    n_epochs: int = 2000
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 200
    eps: float = 1e-2
    Ks: tuple[int, ...] = (1, 5, 10, 20)
    data_root: str = "./data"
    triplet_difficulty: str = "hard"
    backbone: str = "resnet50"
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)


def run_one(seed: int, cfg: AlphaRetrievalConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    scaled_cfg = ScaledRecallConfig(
        N_train=cfg.N_train, N_test=cfg.N_test, n_classes=cfg.n_classes,
        feature_dim=cfg.feature_dim, depth=cfg.depth, width=cfg.width,
        n_triplets=cfg.n_triplets, n_epochs=cfg.n_epochs, lr=cfg.lr,
        margin=cfg.margin, record_every=cfg.record_every, eps=cfg.eps,
        data_root=cfg.data_root, triplet_difficulty=cfg.triplet_difficulty,
        backbone=cfg.backbone,
    )
    Z_all, lbl_all = _load_subset(scaled_cfg, seed, device)
    Z_train, lbl_train, Z_test, lbl_test = _split_train_test(
        Z_all, lbl_all, cfg.N_train, cfg.N_test, seed,
    )
    triplets = _make_triplets(lbl_train, cfg.n_triplets, seed,
                               difficulty=cfg.triplet_difficulty, Z=Z_train)
    print(f"  computing depth-{cfg.depth} NTK on N={cfg.N_train}...", flush=True)
    K = depth_L_ntk_kernel(Z_train, cfg.depth)

    out = {"seed": seed, "Ks": list(cfg.Ks), "config": asdict(cfg), "per_alpha": {}}
    for alpha in cfg.alphas:
        torch.manual_seed(seed); np.random.seed(seed)
        model = MetricNetwork(cfg.feature_dim, [cfg.width] * cfg.depth)
        train_preconditioned(
            model, Z_train, triplets, K,
            n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
            record_every=cfg.record_every, eps=cfg.eps, device=device,
            alpha=alpha,
        )
        model.to("cpu")
        F_tr = embeddings_from_model(model, Z_train, device="cpu")
        F_te = embeddings_from_model(model, Z_test, device="cpu")
        rec = {f"recall@{k}": recall_at_k(F_te, lbl_test, F_tr, lbl_train, k)
               for k in cfg.Ks}
        out["per_alpha"][f"{alpha}"] = rec
        print(f"  alpha={alpha}: " + "  ".join(f"R@{k}={rec[f'recall@{k}']:.3f}" for k in cfg.Ks), flush=True)
    return out


def run_sweep(cfg: AlphaRetrievalConfig, seeds: list[int], out_path: Path,
              device: str = "cuda"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {r["seed"] for r in results}
        seeds = [s for s in seeds if s not in done]
        print(f"  resume: {len(done)} already done, running {len(seeds)} more", flush=True)
    for s in seeds:
        print(f"[alpha-retrieval seed={s}]", flush=True)
        r = run_one(s, cfg, device=device)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  wrote partial -> {out_path} ({len(results)} seeds)", flush=True)
    return results
