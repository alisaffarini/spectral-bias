"""Tier 1: scaled CIFAR-10 retrieval (10 seeds × {easy, hard}, N_train=4000).

Designed to run on the A100 pod. Streams results incrementally to JSON
files so partial progress survives crashes / timeouts.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.recall_scaled import ScaledRecallConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True, choices=("easy", "hard"))
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--start_seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--n_epochs", type=int, default=2000)
    args = ap.parse_args()

    cfg = ScaledRecallConfig(
        triplet_difficulty=args.difficulty,
        N_train=args.n_train,
        N_test=args.n_test,
        n_epochs=args.n_epochs,
    )
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    out_path = RES / f"recall_scaled_cifar10_{args.difficulty}.json"
    print(f"[scaled-cifar10 {args.difficulty}] N_train={args.n_train} -> {out_path}",
          flush=True)
    run_sweep(cfg, seeds, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
