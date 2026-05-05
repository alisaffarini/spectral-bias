"""Plot K^{-alpha} sweep slope vs alpha. The U-shape rules out the
generic-preconditioner explanation for the K^{-1} gain."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "results" / "alpha_sweep.json"
OUT = ROOT / "fig_alpha_sweep.png"

data = json.load(SRC.open())
alphas = data[0]["alphas"]
slopes = np.array([
    [r["per_alpha"][f"{a:.1f}"]["slope"] for a in alphas] for r in data
])
mean = slopes.mean(0); std = slopes.std(0)

fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
ax.errorbar(alphas, mean, yerr=std, marker="o", linewidth=1.8,
            capsize=4, color="C0", markersize=7)
ax.axhline(2.0, linestyle=":", color="gray", alpha=0.5, label=r"theory bound (slope $\leq 2$)")
ax.axvline(1.0, linestyle="--", color="C1", alpha=0.5,
           label=r"canonical kBM correction $\alpha=1$")
ax.set_xlabel(r"preconditioner exponent $\alpha$ in $K^{-\alpha}$")
ax.set_ylabel("fitted spectral slope")
ax.set_title("Slope vs preconditioner strength: $K^{-1}$ specifically cancels the bias")
ax.set_ylim(0, 2.4)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig(OUT, dpi=160)
print(f"saved {OUT} (alpha 0 -> 2 slope: {mean[0]:.2f} -> {mean[-1]:.2f})")
