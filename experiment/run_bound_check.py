"""Bound-verification experiment.

Theorem 7 says displacement scales as λ_i^2 with finite-N residual
O(T λ_i N^{-1/2}). The empirical slope of |G̃_ii(T) - G̃_ii(0)| vs λ_i in
log-log should approach 2 from above (or near 2) as N grows.

We run spectral_bias at N ∈ {30, 100, 300, 1000, 3000}, depth 1, and
extract the fitted slope per N to plot a slope-vs-N phase curve.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from runners.spectral_bias import SpectralConfig, run_sweep

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]


def fit_slope(eigvals: np.ndarray, deltas: np.ndarray) -> float:
    """Fit log-log slope of |delta| vs lambda for the top portion of the
    spectrum, where the leading-order theorem applies."""
    eigvals = np.asarray(eigvals); deltas = np.abs(np.asarray(deltas))
    mask = (eigvals > 1e-3) & (deltas > 1e-8)
    return float(np.polyfit(np.log10(eigvals[mask]),
                             np.log10(deltas[mask]), 1)[0])


def main():
    Ns = [30, 100, 300, 1000, 3000]
    rows = []
    for N in Ns:
        out_path = RES / f"spectral_bias_N{N}_L1.json"
        if not out_path.exists():
            print(f"[run] N={N} L=1 -> {out_path}", flush=True)
            cfg = SpectralConfig(N=N, depth=1, n_epochs=4000)
            run_sweep(cfg, SEEDS, out_path, device="cuda")
        else:
            print(f"[skip] {out_path} exists", flush=True)

        with open(out_path) as f:
            data = json.load(f)
        slopes = []
        for r in data:
            eigvals = np.asarray(r["eigvals_K"])
            delta = np.abs(np.asarray(r["delta"])[-1])
            slopes.append(fit_slope(eigvals, delta))
        slopes = np.array(slopes)
        rows.append({
            "N": N,
            "slope_mean": float(slopes.mean()),
            "slope_std": float(slopes.std()),
            "slopes": slopes.tolist(),
        })
        print(f"  slope = {slopes.mean():.3f} ± {slopes.std():.3f}", flush=True)

    out = RES / "bound_check.json"
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
