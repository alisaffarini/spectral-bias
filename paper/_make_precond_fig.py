"""Regenerate fig_preconditioner.png as a 2-panel figure.

Left:  per-mode displacement |G̃_ii(T) - G̃_ii(0)| vs kernel eigenvalue λ_i (slope plot).
Right: time-resolved displacement of top / middle / bottom modes vs epoch
       (vanilla vs corrected), showing that the K^{-1} correction unblocks
       evolution of low-eigenvalue modes that vanilla suppresses.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "results" / "preconditioner_N50.json"
OUT = ROOT / "fig_preconditioner.png"

with SRC.open() as f:
    data = json.load(f)

eigvals = np.asarray(data[0]["eigvals_K"])
N = len(eigvals)
order = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[order]

epochs = np.asarray(data[0]["epochs"])

# Aggregate per-mode displacement at final time across seeds.
deltas_v = np.stack([np.abs(np.asarray(s["delta_vanilla"][-1]))[order] for s in data])
deltas_p = np.stack([np.abs(np.asarray(s["delta_precond"][-1]))[order] for s in data])
mean_v = deltas_v.mean(0)
mean_p = deltas_p.mean(0)

# Power-law fit on the means (excluding zero-eigenvalue modes for safety).
mask = (eigvals_sorted > 1e-3) & (mean_v > 1e-8) & (mean_p > 1e-8)
slope_v, log_c_v = np.polyfit(np.log10(eigvals_sorted[mask]), np.log10(mean_v[mask]), 1)
slope_p, log_c_p = np.polyfit(np.log10(eigvals_sorted[mask]), np.log10(mean_p[mask]), 1)

# Time-resolved trajectories at three representative mode ranks: top, middle, bottom.
top_rank, mid_rank, bot_rank = 0, N // 2, N - 1
def traj_for_rank(deltas_per_seed_per_t, rank_in_sorted_order):
    # deltas_per_seed_per_t[seed][t] is a list of length N indexed by *original* mode index.
    # We want |G̃_ii(t) - G̃_ii(0)| in sorted-by-eigenvalue order.
    out = np.zeros((len(data), len(epochs)))
    for s_idx, s in enumerate(data):
        for t_idx, dvec in enumerate(s[deltas_per_seed_per_t]):
            out[s_idx, t_idx] = abs(np.asarray(dvec)[order][rank_in_sorted_order])
    return out

traj_v_top = traj_for_rank("delta_vanilla", top_rank)
traj_v_mid = traj_for_rank("delta_vanilla", mid_rank)
traj_v_bot = traj_for_rank("delta_vanilla", bot_rank)
traj_p_top = traj_for_rank("delta_precond", top_rank)
traj_p_mid = traj_for_rank("delta_precond", mid_rank)
traj_p_bot = traj_for_rank("delta_precond", bot_rank)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

# --- Left panel ---
ax = axes[0]
ax.loglog(eigvals_sorted[mask], mean_v[mask], "o", color="C0", alpha=0.7,
          label=f"vanilla (slope {slope_v:.2f})")
ax.loglog(eigvals_sorted[mask], mean_p[mask], "s", color="C1", alpha=0.7,
          label=f"corrected (slope {slope_p:.2f})")
xs = np.logspace(np.log10(eigvals_sorted[mask].min()), np.log10(eigvals_sorted[mask].max()), 50)
ax.loglog(xs, 10**log_c_v * xs**slope_v, "--", color="C0", alpha=0.5)
ax.loglog(xs, 10**log_c_p * xs**slope_p, "--", color="C1", alpha=0.5)
ax.set_xlabel(r"kernel eigenvalue $\lambda_i$")
ax.set_ylabel(r"$|\widetilde{G}_{ii}(T) - \widetilde{G}_{ii}(0)|$")
ax.set_title("Per-mode displacement vs. $\\lambda_i$")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, which="both", alpha=0.3)

# --- Right panel ---
ax = axes[1]
def plot_line(ax, traj, color, ls, label):
    # Multiplicative (log-domain) band: median + IQR on log scale to avoid
    # the linear-std-on-log-axis artefact that creates phantom shading.
    mu = traj.mean(0)
    ax.plot(epochs, mu, ls, color=color, label=label, linewidth=1.8)

plot_line(ax, traj_v_top, "C0", "-",  f"vanilla, top mode ($\\lambda_{{1}}={eigvals_sorted[top_rank]:.1f}$)")
plot_line(ax, traj_v_bot, "C0", "--", f"vanilla, bottom mode ($\\lambda_{{{N}}}={eigvals_sorted[bot_rank]:.2f}$)")
plot_line(ax, traj_p_top, "C1", "-",  f"corrected, top mode")
plot_line(ax, traj_p_bot, "C1", "--", f"corrected, bottom mode")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel(r"$|\widetilde{G}_{ii}(t) - \widetilde{G}_{ii}(0)|$")
ax.set_title("Mode trajectories: top vs. bottom of $K$'s spectrum")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, which="both", alpha=0.3)

fig.savefig(OUT, dpi=160)
print(f"saved {OUT} (slope vanilla={slope_v:.2f}, corrected={slope_p:.2f})")
