"""+5 seeds on a fine-grained dataset, appended to existing JSON."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.finegrained_retrieval import FineGrainedConfig, run_one as run_fg

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--difficulty", required=True)
    ap.add_argument("--start_seed", type=int, default=5)
    ap.add_argument("--n_extra", type=int, default=5)
    ap.add_argument("--data-root", default="./data2")
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n_extra))
    cfg = FineGrainedConfig(dataset=args.dataset,
                             triplet_difficulty=args.difficulty,
                             data_root=args.data_root)
    new_results = []
    for s in seeds:
        r = run_fg(s, cfg, device="cuda")
        new_results.append(r)
        print(f"  seed {s} done", flush=True)

    out_path = RES / f"recall_{args.dataset}_{args.difficulty}.json"
    if out_path.exists():
        existing = json.load(open(out_path))
    else:
        existing = []
    existing.extend(new_results)
    with open(out_path, "w") as f:
        json.dump(existing, f)
    print(f"appended {len(new_results)} seeds -> {out_path}")


if __name__ == "__main__":
    main()
