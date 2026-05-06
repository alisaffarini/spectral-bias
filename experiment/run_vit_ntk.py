"""Tier 3: empirical NTK on a small ViT --- does the slope-<=2 prediction
extend to attention-based architectures? CLI args allow swapping dataset
and tuning eps relative to the spectrum range.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.transformer_ntk import ViTNTKConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "fmnist"])
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--eps_rel", type=float, default=0.0,
                    help="If > 0: eps = eps_rel * lambda_max (use 1e-3 for ViT)")
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()
    cfg = ViTNTKConfig(
        n_seeds=args.n_seeds, dataset=args.dataset,
        eps=args.eps, eps_relative_to_lambda_max=args.eps_rel,
    )
    seeds = list(range(args.n_seeds))
    if args.out_tag:
        out_path = RES / f"vit_ntk_{args.out_tag}.json"
    elif args.dataset != "cifar10" or args.eps_rel > 0:
        tag = args.dataset
        if args.eps_rel > 0:
            tag += f"_epsrel{args.eps_rel}"
        out_path = RES / f"vit_ntk_{tag}.json"
    else:
        out_path = RES / "vit_ntk.json"
    print(f"[vit-ntk dataset={args.dataset} eps={args.eps} eps_rel={args.eps_rel}] -> {out_path}", flush=True)
    run_sweep(cfg, seeds, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
