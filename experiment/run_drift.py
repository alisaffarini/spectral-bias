"""Run NTK drift experiment: 3 seeds, 5 checkpoints per seed."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.ntk_drift import DriftConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    cfg = DriftConfig()
    out_path = RES / "ntk_drift.json"
    seeds = list(range(cfg.n_seeds))
    print(f"[drift] -> {out_path}", flush=True)
    run_sweep(cfg, seeds, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
