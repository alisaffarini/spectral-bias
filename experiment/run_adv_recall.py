"""Tier 2: adversarial-robust retrieval --- does K^{-1} training produce
models more robust to PGD attacks?

CLI args for eps numerator (eps = eps_num/255) and n_seeds, so we can do a
multi-budget sweep without code duplication.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.adversarial_recall import AdvRecallConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps_num", type=int, default=8,
                    help="PGD eps numerator (eps = eps_num/255)")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--out_tag", type=str, default="",
                    help="Suffix for output JSON path; default uses eps in name")
    args = ap.parse_args()
    cfg = AdvRecallConfig(
        pgd_eps=args.eps_num / 255.0,
        pgd_alpha=max(args.eps_num // 4, 1) / 255.0,
    )
    seeds = list(range(args.n_seeds))
    tag = args.out_tag if args.out_tag else f"_eps{args.eps_num}_255"
    if args.eps_num == 8 and not args.out_tag:
        out_path = RES / "recall_adversarial.json"
    else:
        out_path = RES / f"recall_adversarial{tag}.json"
    print(f"[adv-recall eps={args.eps_num}/255] -> {out_path}", flush=True)
    run_sweep(cfg, seeds, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
