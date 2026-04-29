"""Generate publication figures from JSON outputs in results_v2/raw."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RAW = ROOT / "results_v2" / "raw"
OUT = ROOT / "paper"  # save figures next to the .tex
OUT.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def _load(path):
    with open(path) as f:
        return json.load(f)


def _mean_std(arr_list):
    arr = np.stack([np.asarray(a) for a in arr_list], axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


# -----------------------------------------------------------------------------
# Figure: Equivalence (network vs ODE) across depth and N.
# -----------------------------------------------------------------------------
def fig_equivalence():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    Ns = [10, 30]
    Ls = [1, 2, 3]
    for r, N in enumerate(Ns):
        for c, L in enumerate(Ls):
            path = RAW / f"equivalence_N{N}_L{L}.json"
            if not path.exists():
                axes[r, c].text(0.5, 0.5, "missing", transform=axes[r, c].transAxes,
                                ha="center", va="center")
                continue
            data = _load(path)
            epochs = np.array(data[0]["epochs"])
            errors = np.stack([np.array(d["rel_errors"]) * 100 for d in data], axis=0)
            mean = errors.mean(axis=0)
            std = errors.std(axis=0)
            axes[r, c].plot(epochs, mean, lw=1.5, label=f"N={N}, L={L}")
            axes[r, c].fill_between(epochs, mean - std, mean + std, alpha=0.25)
            axes[r, c].set_yscale("log")
            axes[r, c].set_title(f"N={N}, depth L={L}")
            axes[r, c].set_xlabel("epoch")
            if c == 0:
                axes[r, c].set_ylabel(r"$\|G_{net}-G_{ode}\|_F / \|G_{net}\|_F$ (%)")
    fig.suptitle("Network vs.\\ kBM ODE relative error -- mean +/- std over 5 seeds")
    fig.tight_layout()
    fig.savefig(OUT / "fig_equivalence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure: Spectral implicit bias.
# -----------------------------------------------------------------------------
def fig_spectral_bias():
    """Three-panel figure: N=30, N=100, N=1000 each with L=1 and L=2 overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    Ns = [30, 100, 1000]
    colors = {1: "tab:blue", 2: "tab:orange"}
    for c, N in enumerate(Ns):
        for L in [1, 2]:
            path = RAW / f"spectral_bias_N{N}_L{L}.json"
            if not path.exists():
                continue
            data = _load(path)
            eigvals_K = np.array(data[0]["eigvals_K"])
            deltas = np.stack([np.array(d["delta"]) for d in data], axis=0)
            final = np.abs(deltas[:, -1, :])
            m = final.mean(axis=0); s = final.std(axis=0)

            axes[c].errorbar(eigvals_K, m, yerr=s, fmt="o", capsize=2,
                             lw=0.8, ms=3, alpha=0.6, color=colors[L])
            mask = (eigvals_K > 1e-6) & (m > 1e-6)
            if mask.sum() >= 4:
                # Per-seed slopes for honest reporting.
                slopes = []
                for d in data:
                    fi = np.abs(np.array(d["delta"])[-1])
                    mi = (eigvals_K > 1e-6) & (fi > 1e-6)
                    if mi.sum() >= 4:
                        slopes.append(np.polyfit(np.log(eigvals_K[mi]), np.log(fi[mi]), 1)[0])
                slope_mean = np.mean(slopes); slope_std = np.std(slopes)
                p = np.polyfit(np.log(eigvals_K[mask]), np.log(m[mask]), 1)
                xs = np.geomspace(eigvals_K[mask].min(), eigvals_K[mask].max(), 50)
                axes[c].plot(xs, np.exp(p[1]) * xs ** p[0], "--", color=colors[L],
                             label=fr"$L={L}$: slope $= {slope_mean:.2f} \pm {slope_std:.2f}$")
        axes[c].set_xscale("log")
        axes[c].set_yscale("log")
        axes[c].set_xlabel(r"kernel eigenvalue $\lambda_i$")
        if c == 0:
            axes[c].set_ylabel(r"$|\widetilde G_{ii}(T) - \widetilde G_{ii}(0)|$")
        axes[c].set_title(f"N = {N}")
        axes[c].legend(loc="upper left", fontsize=8)

    fig.suptitle(r"Per-mode displacement vs.\ kernel eigenvalue -- mean $\pm$ std over 5 seeds")
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectral_bias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure: Rank vs width.
# -----------------------------------------------------------------------------
def fig_rank_vs_width():
    path = RAW / "rank_vs_width.json"
    if not path.exists():
        return
    data = _load(path)
    Ms = np.array(data[0]["M_values"])
    ranks = np.stack([np.array(d["ranks"]) for d in data], axis=0)
    rm = ranks.mean(axis=0); rs = ranks.std(axis=0)
    losses = np.stack([np.array(d["losses"]) for d in data], axis=0)
    lm = losses.mean(axis=0); ls = losses.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    N_DATA = data[0]["config"].get("N", 10)
    axes[0].errorbar(Ms, rm, yerr=rs, fmt="o-", capsize=3)
    axes[0].axhline(N_DATA, color="red", ls="--", label=f"N = {N_DATA}")
    axes[0].set_xlabel("network width M")
    axes[0].set_ylabel("effective rank")
    axes[0].legend()
    axes[0].set_xscale("log")

    axes[1].errorbar(Ms, lm, yerr=ls, fmt="s-", color="green", capsize=3)
    axes[1].set_xlabel("network width M")
    axes[1].set_ylabel("final triplet loss")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    fig.suptitle("Rank and loss vs.\\ network width -- mean +/- std over 5 seeds")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rank_vs_width.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure: Phase transition.
# -----------------------------------------------------------------------------
def fig_phase():
    path = RAW / "phase_transition.json"
    if not path.exists():
        return
    data = _load(path)
    lambdas = np.array(data[0]["lambdas"])
    ranks = np.stack([np.array(d["ranks"]) for d in data], axis=0)
    rm = ranks.mean(axis=0); rs = ranks.std(axis=0)
    # Trim to the regime where the relative-eigenvalue rank metric is reliable
    # (G is not collapsed to numerical noise). Past lambda ~ 30 the regularizer
    # drives G to zero and the ratio-based rank is dominated by noise.
    mask = lambdas <= 30.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(lambdas[mask], rm[mask], yerr=rs[mask], fmt="o-", capsize=3, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"nuclear-norm regularization strength $\lambda$")
    ax.set_ylabel("effective rank")
    ax.set_title("Rank vs.\\ regularization (mean +/- std over 5 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_phase.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure: Fashion-MNIST spectral bias (if available).
# -----------------------------------------------------------------------------
def fig_fashion():
    path = RAW / "fashion_mnist.json"
    if not path.exists():
        # Make a placeholder so the LaTeX compiles.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Fashion-MNIST experiment pending\n(scaffolding ready, run separately)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        fig.savefig(OUT / "fig_fashion_spectral.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    data = _load(path)
    eigvals_K = np.array(data[0]["eigvals_K"])
    G_modes = np.stack(
        [np.array(d["G_modes_history"])[-1] - np.array(d["G0_modes"]) for d in data],
        axis=0,
    )
    final = np.abs(G_modes)
    m = final.mean(axis=0); s = final.std(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(eigvals_K, m, yerr=s, fmt="o", capsize=3, lw=1.0, ms=3, alpha=0.7)
    mask = (eigvals_K > 1e-3) & (m > 1e-4)
    if mask.sum() >= 5:
        p = np.polyfit(np.log(eigvals_K[mask]), np.log(m[mask]), 1)
        xs = np.geomspace(eigvals_K[mask].min(), eigvals_K[mask].max(), 50)
        ax.plot(xs, np.exp(p[1]) * xs ** p[0], "--", color="gray",
                label=f"slope = {p[0]:.2f}")
        ax.legend(loc="upper left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"kernel eigenvalue $\lambda_i$")
    ax.set_ylabel(r"$|\widetilde G_{ii}(T) - \widetilde G_{ii}(0)|$")
    ax.set_title("Fashion-MNIST: per-mode displacement vs.\\ kernel eigenvalue")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fashion_spectral.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_preconditioner():
    path = RAW / "preconditioner_N50.json"
    if not path.exists():
        return
    data = _load(path)
    eigvals_K = np.array(data[0]["eigvals_K"])
    deltas_v = np.stack([np.array(d["delta_vanilla"])[-1] for d in data], axis=0)
    deltas_p = np.stack([np.array(d["delta_precond"])[-1] for d in data], axis=0)
    sat_v = np.stack([np.array(d["sat_by_bin_vanilla"]) for d in data], axis=0)
    sat_p = np.stack([np.array(d["sat_by_bin_precond"]) for d in data], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left panel: per-mode displacement scaling, vanilla vs preconditioned
    ax = axes[0]
    final_v = np.abs(deltas_v); final_p = np.abs(deltas_p)
    mv = final_v.mean(axis=0); sv = final_v.std(axis=0)
    mp = final_p.mean(axis=0); sp = final_p.std(axis=0)
    ax.errorbar(eigvals_K, mv, yerr=sv, fmt="o", capsize=2, lw=1, ms=4, alpha=0.7,
                color="tab:blue", label="vanilla")
    ax.errorbar(eigvals_K, mp, yerr=sp, fmt="s", capsize=2, lw=1, ms=4, alpha=0.7,
                color="tab:orange", label="preconditioned")
    # Slopes
    slopes_v, slopes_p = [], []
    for d in data:
        dv = np.abs(np.array(d["delta_vanilla"])[-1]); dp = np.abs(np.array(d["delta_precond"])[-1])
        mvi = (eigvals_K > 1e-6) & (dv > 1e-6); mpi = (eigvals_K > 1e-6) & (dp > 1e-6)
        if mvi.sum() >= 4: slopes_v.append(np.polyfit(np.log(eigvals_K[mvi]), np.log(dv[mvi]), 1)[0])
        if mpi.sum() >= 4: slopes_p.append(np.polyfit(np.log(eigvals_K[mpi]), np.log(dp[mpi]), 1)[0])
    sv_m = np.mean(slopes_v); sv_s = np.std(slopes_v)
    sp_m = np.mean(slopes_p); sp_s = np.std(slopes_p)
    for slope, color in [(sv_m, "tab:blue"), (sp_m, "tab:orange")]:
        mask = (eigvals_K > 1e-6) & (mv > 1e-6)
        if mask.sum() >= 3:
            xs = np.geomspace(eigvals_K[mask].min(), eigvals_K[mask].max(), 50)
            anchor = mv[mask].mean() if color == "tab:blue" else mp[mask].mean()
            anchor_x = eigvals_K[mask].mean()
            const = anchor / (anchor_x ** slope)
            ax.plot(xs, const * xs ** slope, "--", color=color, alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"kernel eigenvalue $\lambda_i$")
    ax.set_ylabel(r"$|\widetilde G_{ii}(T) - \widetilde G_{ii}(0)|$")
    ax.legend(loc="upper left",
              title=fr"vanilla: ${sv_m:.2f} \pm {sv_s:.2f}$" + "\n"
                    + fr"precond: ${sp_m:.2f} \pm {sp_s:.2f}$",
              fontsize=9, title_fontsize=9)
    ax.set_title("Per-mode displacement scaling")

    # Right panel: triplet satisfaction by mode-bin
    ax = axes[1]
    n_bins = sat_v.shape[1]
    x = np.arange(n_bins)
    width = 0.35
    sat_v_m = np.nanmean(sat_v, axis=0); sat_v_s = np.nanstd(sat_v, axis=0)
    sat_p_m = np.nanmean(sat_p, axis=0); sat_p_s = np.nanstd(sat_p, axis=0)
    ax.bar(x - width/2, sat_v_m, width, yerr=sat_v_s, capsize=3, alpha=0.7,
           color="tab:blue", label="vanilla")
    ax.bar(x + width/2, sat_p_m, width, yerr=sat_p_s, capsize=3, alpha=0.7,
           color="tab:orange", label="preconditioned")
    ax.set_xlabel(r"K-eigenvalue mode bin (top $\to$ bottom)")
    ax.set_ylabel("triplet satisfaction rate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bin {i+1}" for i in range(n_bins)])
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Triplet satisfaction by spectral bin")
    ax.set_ylim(0, 1.05)

    fig.suptitle(r"Spectral preconditioner: kBM theory $\to$ algorithm (mean $\pm$ std, 5 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_preconditioner.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_equivalence()
    fig_spectral_bias()
    fig_rank_vs_width()
    fig_phase()
    fig_fashion()
    fig_preconditioner()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
