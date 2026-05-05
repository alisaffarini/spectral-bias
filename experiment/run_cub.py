"""Drive CUB-200 retrieval defense experiment (5 seeds, easy + hard)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.cub200 import CUBConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEEDS = list(range(5))


def main():
    for difficulty in ("easy", "hard"):
        cfg = CUBConfig(triplet_difficulty=difficulty)
        out_path = RES / f"recall_cub_{difficulty}.json"
        print(f"[cub {difficulty}] -> {out_path}", flush=True)
        run_sweep(cfg, SEEDS, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
