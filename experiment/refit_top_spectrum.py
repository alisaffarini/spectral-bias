"""Refit spectral slopes restricted to the top half of the spectrum,
where Theorem 7's leading-order λ^2 prediction is theoretically expected
to hold (the O(N^{-1/2}) residual and mode-mixing with bottom-spectrum
modes both come from small-λ modes).

This is a physics-motivated fit restriction, not data-dredging: the
theorem applies most cleanly where (A1) is approximately diagonal,
which is the top of the spectrum.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def fit_slope(eigvals: np.ndarray, deltas: np.ndarray, frac: float = 0.5) -> float:
    """Fit log-log slope on the top `frac` of the spectrum (by eigenvalue)."""
    eigvals = np.asarray(eigvals); deltas = np.abs(np.asarray(deltas))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; deltas = deltas[order]
    n_top = max(5, int(len(eigvals) * frac))
    eig_top = eigvals[:n_top]; del_top = deltas[:n_top]
    mask = (eig_top > 1e-3) & (del_top > 1e-8)
    if mask.sum() < 5:
        return float("nan")
    return float(np.polyfit(np.log10(eig_top[mask]),
                             np.log10(del_top[mask]), 1)[0])


print("Top-half-of-spectrum slope refits (depth-1, synthetic):\n")
print(f"{'N':>6}  {'top-50%':>14}  {'top-25%':>14}  {'full spectrum':>16}")
for N in [30, 100, 300, 1000, 3000]:
    p = RES / f"spectral_bias_N{N}_L1.json"
    if not p.exists():
        continue
    data = json.load(open(p))
    s50, s25, sfull = [], [], []
    for r in data:
        eigs = np.asarray(r["eigvals_K"])
        delta = np.abs(np.asarray(r["delta"])[-1])
        s50.append(fit_slope(eigs, delta, frac=0.5))
        s25.append(fit_slope(eigs, delta, frac=0.25))
        sfull.append(fit_slope(eigs, delta, frac=1.0))
    s50 = np.array(s50); s25 = np.array(s25); sfull = np.array(sfull)
    print(f"  {N:>4}  {s50.mean():.3f}±{s50.std():.3f}    "
          f"{s25.mean():.3f}±{s25.std():.3f}    "
          f"{sfull.mean():.3f}±{sfull.std():.3f}")
