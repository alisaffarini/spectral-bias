"""Defense experiments addressing reviewer audit concerns.

Three blocks:
  (1) Additional retrieval seeds (5 -> 10) for both easy and hard regimes,
      enabling Bonferroni-corrected paired t-tests across the two regimes.
  (2) Preconditioner alpha-sweep K^{-alpha} for alpha in {0.0, 0.5, 1.0,
      1.5, 2.0} on the synthetic depth-1, N=50 setup. Confirms K^{-1}
      specifically (not generic preconditioning) is what cancels the bias.
  (3) Semi-hard retrieval (embedding-space hard-negative mining,
      FaceNet-style) to verify the asymmetric prediction survives a more
      standard hardness operationalization.

Outputs JSON in `results/`. Re-run from the experiment directory:
    py run_defense.py --block all --device cuda
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from runners.recall_eval import RecallConfig, run_one as run_recall_one
from runners.preconditioner import (
    PreconditionerConfig,
    train_preconditioned,
)
from kbm.data import generate_synthetic_data, generate_fixed_triplets
from kbm.kernels import standard_ntk_kernel
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model


# ---------- Block 1: extra retrieval seeds ----------

def block_more_seeds(out_dir: Path, extra_seeds: list[int], device: str):
    for difficulty in ("easy", "hard"):
        cfg = RecallConfig(triplet_difficulty=difficulty)
        results = []
        for s in extra_seeds:
            print(f"[recall {difficulty}] seed={s}", flush=True)
            results.append(run_recall_one(s, cfg, device=device))
        out_path = out_dir / f"recall_{difficulty}_extra_seeds.json"
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  -> {out_path}")


# ---------- Block 2: K^{-alpha} sweep ----------

def block_alpha_sweep(out_dir: Path, seeds: list[int], device: str):
    cfg = PreconditionerConfig(N=50, depth=1)
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        X, _, _ = generate_synthetic_data(cfg.N, cfg.d_in, cfg.rank_true, seed=s)
        triplets = generate_fixed_triplets(cfg.N, cfg.N * cfg.n_triplets_factor, seed=s)
        K = standard_ntk_kernel(X)
        eigvals_K, U = np.linalg.eigh(K)
        order = np.argsort(eigvals_K)[::-1]
        eigvals_K = eigvals_K[order]; U = U[:, order]

        torch.manual_seed(s); np.random.seed(s)
        m_init = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
        F0 = embeddings_from_model(m_init, X, device="cpu")
        G0_modes = np.diag(U.T @ (F0 @ F0.T) @ U)

        per_alpha = {}
        for a in alphas:
            torch.manual_seed(s); np.random.seed(s)
            m = MetricNetwork(cfg.d_in, [cfg.width] * cfg.depth)
            if a == 0.0:
                # Vanilla
                out = train_network(
                    m, X, triplets,
                    TrainConfig(n_epochs=cfg.n_epochs, lr=cfg.lr,
                                margin=cfg.margin, record_every=cfg.record_every),
                    device=device,
                )
            else:
                out = train_preconditioned(
                    m, X, triplets, K,
                    n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
                    record_every=cfg.record_every, eps=cfg.eps,
                    device=device, alpha=a,
                )
            G_final = out["G_history"][-1]
            modes = np.diag(U.T @ G_final @ U)
            delta_final = np.abs(modes - G0_modes)
            # Power-law slope on log-log
            mask = (eigvals_K > 1e-3) & (delta_final > 1e-8)
            if mask.sum() >= 5:
                slope = np.polyfit(np.log10(eigvals_K[mask]),
                                    np.log10(delta_final[mask]), 1)[0]
            else:
                slope = float('nan')
            per_alpha[f"{a:.1f}"] = {
                "slope": float(slope),
                "delta_final": delta_final.tolist(),
                "final_loss": float(out["losses"][-1]),
            }
        results.append({
            "seed": s,
            "alphas": alphas,
            "eigvals_K": eigvals_K.tolist(),
            "per_alpha": per_alpha,
            "config": asdict(cfg),
        })
        print(f"[alpha-sweep] seed={s} done", flush=True)

    out_path = out_dir / "alpha_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  -> {out_path}")


# ---------- Block 3: Semi-hard retrieval ----------

def block_semi_hard(out_dir: Path, seeds: list[int], device: str):
    cfg = RecallConfig(triplet_difficulty="semi_hard")
    results = []
    for s in seeds:
        print(f"[recall semi_hard] seed={s}", flush=True)
        results.append(run_recall_one(s, cfg, device=device))
    out_path = out_dir / "recall_semi_hard.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  -> {out_path}")


# ---------- Driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block", default="all",
                    choices=("all", "seeds", "alpha", "semi_hard"))
    ap.add_argument("--extra_seeds", default="5,6,7,8,9")
    ap.add_argument("--alpha_seeds", default="0,1,2,3,4")
    ap.add_argument("--semi_hard_seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_seeds = [int(s) for s in args.extra_seeds.split(",")]
    alpha_seeds = [int(s) for s in args.alpha_seeds.split(",")]
    semi_hard_seeds = [int(s) for s in args.semi_hard_seeds.split(",")]

    if args.block in ("all", "seeds"):
        block_more_seeds(out_dir, extra_seeds, args.device)
    if args.block in ("all", "alpha"):
        block_alpha_sweep(out_dir, alpha_seeds, args.device)
    if args.block in ("all", "semi_hard"):
        block_semi_hard(out_dir, semi_hard_seeds, args.device)
    print("defense experiments done.")


if __name__ == "__main__":
    main()
