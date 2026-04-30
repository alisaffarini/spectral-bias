"""Comparison with Razin-Cohen rank bias.

Razin & Cohen (2020) characterize the implicit bias of deep matrix
factorization as a *low-rank* bias: gradient descent with infinitesimal
init drives the effective rank of the solution downward over time.

Our spectral implicit bias (Theorem 6) is *along K's eigenbasis*: bottom-
eigenvalue modes are frozen, top-eigenvalue modes evolve. These two biases
are conceptually distinct -- one operates on rank, the other on a kernel-
defined directional structure.

This runner makes the distinction concrete by tracking BOTH metrics on
the same training trajectory:

  effective_rank(t)         -- the Razin-Cohen-style metric, count of
                               singular values of G(t) above 1% of the
                               largest. Reflects how much the solution
                               is concentrated in a low-rank subspace.

  spectral_displacement(t)  -- our metric, the per-mode displacement
                               |G_tilde[i,i](t) - G_tilde[i,i](0)| as a
                               function of lambda_i.

The point: vanilla NTK metric learning produces TRIVIAL rank bias (rank
quickly hits N) but DRAMATIC spectral bias (slope ~ 2 in lambda). So our
mechanism is genuinely distinct from theirs, and complementary.
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
from kbm.metrics import effective_rank
from runners.preconditioner import train_preconditioned


@dataclass
class RankSpectralConfig:
    N: int = 50
    d_in: int = 20
    rank_true: int = 5
    depth: int = 1
    width: int = 1000
    n_triplets: int = 400
    n_epochs: int = 2000
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 50
    eps: float = 1e-2


def run_one(seed: int, cfg: RankSpectralConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    X, _, _ = generate_synthetic_data(cfg.N, cfg.d_in, cfg.rank_true, seed=seed)
    triplets = generate_fixed_triplets(cfg.N, cfg.n_triplets, seed=seed)

    K = standard_ntk_kernel(X)
    eigvals_K, U = np.linalg.eigh(K)
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]; U = U[:, order]

    torch.manual_seed(seed); np.random.seed(seed)
    model_init = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    F0 = embeddings_from_model(model_init, X, device="cpu")
    G0 = F0 @ F0.T
    G0_modes = np.diag(U.T @ G0 @ U)

    # Vanilla
    torch.manual_seed(seed); np.random.seed(seed)
    model_v = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    out_v = train_network(
        model_v, X, triplets,
        TrainConfig(n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
                    record_every=cfg.record_every),
        device=device,
    )

    # Preconditioned
    torch.manual_seed(seed); np.random.seed(seed)
    model_p = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
    out_p = train_preconditioned(
        model_p, X, triplets, K,
        n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
        record_every=cfg.record_every, eps=cfg.eps, device=device,
    )

    # Track effective rank trajectory (Razin-Cohen metric)
    rank_v = [effective_rank(G) for G in out_v["G_history"]]
    rank_p = [effective_rank(G) for G in out_p["G_history"]]

    # Track spectral-displacement trajectory (our metric)
    def spectral_norm_diff(hist):
        return [float(np.linalg.norm(np.diag(U.T @ G @ U) - G0_modes))
                for G in hist]
    spec_v = spectral_norm_diff(out_v["G_history"])
    spec_p = spectral_norm_diff(out_p["G_history"])

    # Final-time per-mode displacement (for slope fits)
    def proj(hist):
        return np.stack([np.diag(U.T @ G @ U) for G in hist], axis=0)
    delta_v = proj(out_v["G_history"]) - G0_modes[None, :]
    delta_p = proj(out_p["G_history"]) - G0_modes[None, :]

    return {
        "seed": seed,
        "epochs": out_v["epochs"].tolist(),
        "eigvals_K": eigvals_K.tolist(),
        "G0_modes": G0_modes.tolist(),
        "rank_vanilla": rank_v,
        "rank_precond": rank_p,
        "spectral_diff_vanilla": spec_v,
        "spectral_diff_precond": spec_p,
        "delta_vanilla": delta_v.tolist(),
        "delta_precond": delta_p.tolist(),
        "vanilla_loss": out_v["losses"].tolist(),
        "precond_loss": out_p["losses"].tolist(),
        "config": asdict(cfg),
    }


def run_sweep(cfg: RankSpectralConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
