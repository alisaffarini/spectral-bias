"""A100 utility runner: high-seed-count sweeps across high-class-count
retrieval datasets, with incremental JSON writes so partial results
survive disconnects.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _run(dataset: str, difficulty: str, seeds: list[int], out_path: Path):
    if dataset == "cub":
        from runners.cub200 import CUBConfig, run_one
        cfg = CUBConfig(triplet_difficulty=difficulty)
    elif dataset == "cifar100":
        from runners.recall_cifar100 import CIFAR100RecallConfig, run_one
        cfg = CIFAR100RecallConfig(triplet_difficulty=difficulty)
    elif dataset in ("pet", "aircraft", "flowers"):
        from runners.finegrained_retrieval import FineGrainedConfig, run_one
        cfg = FineGrainedConfig(dataset=dataset, triplet_difficulty=difficulty)
    else:
        raise ValueError(f"unknown dataset {dataset}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        done = {r["seed"] for r in existing}
        seeds = [s for s in seeds if s not in done]
        results = existing
        print(f"  resume: {len(done)} seeds already done, running {len(seeds)} more",
              flush=True)
    else:
        results = []

    for s in seeds:
        print(f"[{dataset}/{difficulty} seed={s}]", flush=True)
        r = run_one(s, cfg, device="cuda")
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  wrote {len(results)} seeds -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=("cub", "cifar100", "pet", "aircraft", "flowers"))
    ap.add_argument("--difficulty", required=True, choices=("easy", "hard"))
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--start_seed", type=int, default=0)
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    out_path = RES / f"recall_{args.dataset}_{args.difficulty}.json"
    print(f"[a100 {args.dataset}/{args.difficulty}] -> {out_path}", flush=True)
    _run(args.dataset, args.difficulty, seeds, out_path)
    print("done.")


if __name__ == "__main__":
    main()
