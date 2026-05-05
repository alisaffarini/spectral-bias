"""Drive bottom-eigenspace target experiment (5 seeds)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.bottom_target import BottomTargetConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    cfg = BottomTargetConfig()
    out_path = RES / "bottom_target.json"
    seeds = list(range(5))
    print(f"[bottom-target] -> {out_path}", flush=True)
    run_sweep(cfg, seeds, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
