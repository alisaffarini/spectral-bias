"""Run 5 more seeds (5-9) on the three direction-positive datasets to
push toward Bonferroni significance.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from runners.finegrained_retrieval import FineGrainedConfig, run_one as run_fg
from runners.cub200 import CUBConfig, run_one as run_cub
from runners.recall_cifar100 import CIFAR100RecallConfig, run_one as run_cifar100

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def extend(out_path: Path, new_results):
    if out_path.exists():
        existing = json.load(open(out_path))
    else:
        existing = []
    existing.extend(new_results)
    with open(out_path, "w") as f:
        json.dump(existing, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=("aircraft", "cub", "cifar100"))
    ap.add_argument("--difficulty", required=True, choices=("easy", "hard"))
    ap.add_argument("--start_seed", type=int, default=5)
    ap.add_argument("--n_extra", type=int, default=5)
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n_extra))
    new_results = []
    for s in seeds:
        if args.dataset == "aircraft":
            cfg = FineGrainedConfig(dataset="aircraft",
                                     triplet_difficulty=args.difficulty,
                                     data_root="./data")
            r = run_fg(s, cfg, device="cuda")
        elif args.dataset == "cub":
            cfg = CUBConfig(triplet_difficulty=args.difficulty)
            r = run_cub(s, cfg, device="cuda")
        elif args.dataset == "cifar100":
            cfg = CIFAR100RecallConfig(triplet_difficulty=args.difficulty)
            r = run_cifar100(s, cfg, device="cuda")
        new_results.append(r)
        print(f"  seed {s} done", flush=True)

    out_path = RES / f"recall_{args.dataset}_{args.difficulty}.json"
    extend(out_path, new_results)
    print(f"appended {len(new_results)} seeds -> {out_path}")


if __name__ == "__main__":
    main()
