"""Long-tailed CIFAR-10 retrieval defense (5 seeds)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.longtail_cifar import LongTailConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEEDS = list(range(5))


def main():
    cfg = LongTailConfig(triplet_difficulty="hard")
    out_path = RES / "recall_longtail_cifar10.json"
    print(f"[longtail-cifar10 hard] -> {out_path}", flush=True)
    run_sweep(cfg, SEEDS, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
