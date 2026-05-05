"""Tier 3: empirical NTK on a small ViT — does the slope-<=2 prediction
extend to attention-based architectures?
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.transformer_ntk import ViTNTKConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    cfg = ViTNTKConfig()
    seeds = list(range(cfg.n_seeds))
    out_path = RES / "vit_ntk.json"
    print(f"[vit-ntk] -> {out_path}", flush=True)
    run_sweep(cfg, seeds, out_path, device="cuda")
    print("done.")


if __name__ == "__main__":
    main()
