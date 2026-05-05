"""Drive fine-grained retrieval defense (Pet / Aircraft / Flowers)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.finegrained_retrieval import FineGrainedConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pet", "aircraft", "flowers", "cars"))
    ap.add_argument("--difficulty", required=True, choices=("easy", "hard"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data-root", default="./data")
    args = ap.parse_args()

    cfg = FineGrainedConfig(dataset=args.dataset, triplet_difficulty=args.difficulty,
                             data_root=args.data_root)
    out_path = RES / f"recall_{args.dataset}_{args.difficulty}.json"
    seeds = list(range(args.seeds))
    print(f"[{args.dataset} {args.difficulty}] -> {out_path}", flush=True)
    run_sweep(cfg, seeds, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
