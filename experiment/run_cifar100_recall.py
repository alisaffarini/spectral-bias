"""Drive CIFAR-100 retrieval defense experiment (5 seeds, easy + hard)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.recall_cifar100 import CIFAR100RecallConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEEDS = list(range(5))


def main():
    for difficulty in ("easy", "hard"):
        cfg = CIFAR100RecallConfig(triplet_difficulty=difficulty)
        out_path = RES / f"recall_cifar100_{difficulty}.json"
        print(f"[cifar100 {difficulty}] -> {out_path}", flush=True)
        run_sweep(cfg, SEEDS, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
