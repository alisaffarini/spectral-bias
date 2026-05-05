"""Drive Fashion-MNIST retrieval defense experiment (10 seeds, easy + hard)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.recall_fashion import FashionRecallConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEEDS = list(range(10))


def main():
    for difficulty in ("easy", "hard"):
        cfg = FashionRecallConfig(triplet_difficulty=difficulty)
        out_path = RES / f"recall_fashion_{difficulty}.json"
        print(f"[fashion {difficulty}] -> {out_path}", flush=True)
        run_sweep(cfg, SEEDS, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
