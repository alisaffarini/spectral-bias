"""Tier 2: adversarial-robust retrieval — does K^{-1} training produce
models more robust to PGD attacks?
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.adversarial_recall import AdvRecallConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    cfg = AdvRecallConfig()
    seeds = list(range(5))
    out_path = RES / "recall_adversarial.json"
    print(f"[adv-recall] -> {out_path}", flush=True)
    run_sweep(cfg, seeds, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
