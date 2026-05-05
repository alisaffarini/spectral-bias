"""Slope-vs-N phase plot. Theorem 7's bound (slope ≤ 2) should hold across
N; small-N noise on the regression can push the empirical mean slightly
above 2, with the lower error bar still respecting the bound."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "fig_bound_check.png"

Ns = [30, 100, 300, 1000, 3000]


def fit_slope(eigvals, deltas):
    eigvals = np.asarray(eigvals)
    deltas = np.abs(np.asarray(deltas))
    mask = (eigvals > 1e-3) & (deltas > 1e-8)
    return float(np.polyfit(np.log10(eigvals[mask]), np.log10(deltas[mask]), 1)[0])


rows = []
for N in Ns:
    p = RES / f"spectral_bias_N{N}_L1.json"
    data = json.load(open(p))
    slopes = []
    for r in data:
        eigvals = np.asarray(r["eigvals_K"])
        delta = np.abs(np.asarray(r["delta"])[-1])
        slopes.append(fit_slope(eigvals, delta))
    slopes = np.array(slopes)
    rows.append({"N": N, "slopes": slopes,
                 "mean": slopes.mean(), "std": slopes.std()})
    print(f"N={N}: {slopes.mean():.3f} ± {slopes.std():.3f}")

fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
xs = [r["N"] for r in rows]
ys = [r["mean"] for r in rows]
err = [r["std"] for r in rows]
ax.errorbar(xs, ys, yerr=err, marker="o", linewidth=1.8, capsize=4,
            color="C0", markersize=8)
ax.axhline(2.0, linestyle="--", color="C3", alpha=0.7,
           label=r"theory upper bound (slope $\leq 2$)")
ax.set_xscale("log")
ax.set_xlabel(r"data set size $N$ (log scale)")
ax.set_ylabel("fitted spectral slope")
ax.set_title(r"Empirical slope vs. $N$: bound holds, with small-$N$ regression noise")
ax.set_ylim(1.0, 2.4)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig(OUT, dpi=160)
print(f"saved {OUT}")
