"""Aggregate all JSON results and emit LaTeX appendix tables for maxinfo.tex.

Writes individual snippet files into paper/_maxinfo_snippets/, then maxinfo.tex
\\input's them. Keeping computation here (numpy) means we can fit log-log slopes
correctly and reproducibly without polluting maxinfo.tex with raw numbers.
"""
import json
import os
import math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(ROOT, "..", "results"))
OUT = os.path.join(ROOT, "_maxinfo_snippets")
os.makedirs(OUT, exist_ok=True)


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def fit_loglog(eigvals, disp):
    eigvals = np.asarray(eigvals, dtype=float)
    disp = np.asarray(disp, dtype=float)
    mask = (eigvals > 1e-12) & (disp > 1e-12) & np.isfinite(eigvals) & np.isfinite(disp)
    if mask.sum() < 2:
        return float("nan"), float("nan"), 0, (float("nan"), float("nan"))
    x = np.log(eigvals[mask])
    y = np.log(disp[mask])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    eig_range = (float(eigvals[mask].min()), float(eigvals[mask].max()))
    return float(slope), float(r2), int(mask.sum()), eig_range


def fmt(x, p=4):
    if isinstance(x, float) and not math.isfinite(x):
        return "n/a"
    return f"{x:.{p}f}"


def fmt_sci(x, p=3):
    if isinstance(x, float) and not math.isfinite(x):
        return "n/a"
    return f"{x:.{p}e}"


# ===================== APPENDIX A: per-seed spectral-bias slopes =====================
def appendix_A():
    lines = []
    lines.append(r"\section{Full per-seed spectral-bias slopes}")
    lines.append(r"\label{app:per-seed-slopes}")
    lines.append("")
    lines.append(
        r"For each of the six synthetic configurations $(N, L)$ with "
        r"$N \in \{30, 100, 1000\}$, $L \in \{1, 2\}$, we recompute the "
        r"log-log slope of $|\widetilde{G}_{ii}(T) - \widetilde{G}_{ii}(0)|$ "
        r"versus $\lambda_i$ at the final recorded epoch directly from the JSON "
        r"per seed. Slopes are obtained by ordinary least squares "
        r"(\texttt{numpy.polyfit}) on the masked log-data; modes with "
        r"non-positive eigenvalue or non-positive displacement are dropped. "
        r"$R^2$ is the coefficient of determination of the same regression."
    )
    lines.append("")
    aggregate_rows = []
    for L in (1, 2):
        for N in (30, 100, 1000):
            fname = f"spectral_bias_N{N}_L{L}.json"
            data = load(fname)
            lines.append(rf"\subsection{{$N={N}$, $L={L}$}}")
            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(rf"\caption{{Per-seed slope, $R^2$, eigenvalue range, "
                          rf"and number of modes used for $N={N}$, $L={L}$ "
                          rf"(\texttt{{spectral\_bias\_N{N}\_L{L}.json}}).}}")
            lines.append(rf"\label{{tab:slopeA-N{N}-L{L}}}")
            lines.append(r"\begin{tabular}{c c c c c c}")
            lines.append(r"\toprule")
            lines.append(r"seed & slope & $R^2$ & $\lambda_{\min}$ & $\lambda_{\max}$ & modes \\")
            lines.append(r"\midrule")
            slopes_list = []
            for seed_dict in data:
                eigvals = seed_dict["eigvals_K"]
                G_final = np.array(seed_dict["G_modes_history"][-1])
                G0 = np.array(seed_dict["G0_modes"])
                disp = np.abs(G_final - G0)
                slope, r2, n, (lo, hi) = fit_loglog(eigvals, disp)
                slopes_list.append(slope)
                lines.append(
                    f"{seed_dict['seed']} & {fmt(slope, 4)} & {fmt(r2, 4)} & "
                    f"{fmt_sci(lo, 3)} & {fmt_sci(hi, 3)} & {n} \\\\"
                )
            arr = np.array(slopes_list, dtype=float)
            mu, sd = float(np.mean(arr)), float(np.std(arr, ddof=0))
            lines.append(r"\midrule")
            lines.append(rf"\textbf{{mean $\pm$ std}} & {fmt(mu, 4)} $\pm$ {fmt(sd, 4)} & "
                         rf"\multicolumn{{4}}{{l}}{{}} \\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")
            aggregate_rows.append((N, L, mu, sd, slopes_list))
    # summary table
    lines.append(r"\subsection*{Aggregate summary}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean $\pm$ standard deviation of fitted log-log "
                 r"slopes over 5 seeds, all six configurations.}")
    lines.append(r"\label{tab:slope-aggregate}")
    lines.append(r"\begin{tabular}{c c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $L$ & seed 0 & seed 1 & seed 2 & seed 3 & seed 4 & mean $\pm$ std \\")
    lines.append(r"\midrule")
    for N, L, mu, sd, slopes in aggregate_rows:
        cells = " & ".join(fmt(s, 4) for s in slopes)
        lines.append(rf"{N} & {L} & {cells} & {fmt(mu, 4)} $\pm$ {fmt(sd, 4)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX B: equivalence trajectories =====================
def appendix_B():
    lines = []
    lines.append(r"\section{Equivalence error trajectories}")
    lines.append(r"\label{app:equiv-traj}")
    lines.append("")
    lines.append(
        r"Per-seed network-vs-ODE relative Frobenius error trajectories for "
        r"each of the six $(N, L)$ equivalence configurations with "
        r"$N \in \{10, 30\}$, $L \in \{1, 2, 3\}$. We tabulate the recorded "
        r"epochs at five time-points (the first epoch, the 25\%, 50\%, "
        r"75\% recorded indices, and the final epoch) together with the "
        r"network triplet loss and the ODE-side reference loss at the same "
        r"time-stamps."
    )
    lines.append("")
    for N in (10, 30):
        for L in (1, 2, 3):
            fname = f"equivalence_N{N}_L{L}.json"
            data = load(fname)
            lines.append(rf"\subsection{{$N={N}$, $L={L}$}}")
            for seed_dict in data:
                seed = seed_dict["seed"]
                epochs = seed_dict["epochs"]
                rel = seed_dict["rel_errors"]
                netl = seed_dict["network_loss"]
                odel = seed_dict["ode_loss"]
                T = len(epochs)
                idxs = sorted({0, T // 4, T // 2, (3 * T) // 4, T - 1})
                lines.append(r"\begin{table}[H]")
                lines.append(r"\centering")
                lines.append(rf"\caption{{Seed {seed}: equivalence trajectory "
                             rf"for $N={N}$, $L={L}$. \texttt{{rel\_err}} is "
                             rf"$\|G_{{\mathrm{{net}}}} - G_{{\mathrm{{ode}}}}\|_F"
                             rf" / \|G_{{\mathrm{{net}}}}\|_F$.}}")
                lines.append(rf"\label{{tab:equiv-N{N}-L{L}-s{seed}}}")
                lines.append(r"\begin{tabular}{c c c c c}")
                lines.append(r"\toprule")
                lines.append(r"index & epoch & rel\_err & net loss & ODE loss \\")
                lines.append(r"\midrule")
                for i in idxs:
                    lines.append(
                        f"{i} & {epochs[i]} & {fmt(rel[i], 6)} & "
                        f"{fmt(netl[i], 4)} & {fmt(odel[i], 4)} \\\\"
                    )
                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")
                lines.append(r"\end{table}")
                lines.append("")
            # final-epoch summary across seeds
            finals = [s["rel_errors"][-1] for s in data]
            mu, sd = float(np.mean(finals)), float(np.std(finals))
            lines.append(rf"Final relative Frobenius error, $N={N}$, $L={L}$, "
                         rf"5 seeds: mean $\pm$ std = {fmt(mu, 4)} $\pm$ {fmt(sd, 4)}.")
            lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX C: preconditioner per-seed =====================
def appendix_C():
    lines = []
    lines.append(r"\section{Per-seed preconditioner effects}")
    lines.append(r"\label{app:precond-perseed}")
    lines.append("")
    lines.append(
        r"Per-seed log-log slopes for the spectral preconditioner experiments. "
        r"For each seed and each variant (vanilla / preconditioned) we fit "
        r"$\log |\widetilde{G}_{ii}(T) - \widetilde{G}_{ii}(0)|$ against "
        r"$\log \lambda_i$ at the final recorded epoch."
    )
    lines.append("")

    def precond_table(fname, title, label):
        data = load(fname)
        rows_v, rows_p = [], []
        for seed_dict in data:
            seed = seed_dict["seed"]
            eigvals = seed_dict["eigvals_K"]
            dv_final = np.array(seed_dict["delta_vanilla"][-1])
            dp_final = np.array(seed_dict["delta_precond"][-1])
            sv, rv, nv, (loV, hiV) = fit_loglog(eigvals, np.abs(dv_final))
            sp, rp, np_, (loP, hiP) = fit_loglog(eigvals, np.abs(dp_final))
            rows_v.append((seed, sv, rv, nv, loV, hiV))
            rows_p.append((seed, sp, rp, np_, loP, hiP))
        out = []
        out.append(r"\begin{table}[H]")
        out.append(r"\centering")
        out.append(rf"\caption{{{title}}}")
        out.append(rf"\label{{tab:{label}}}")
        out.append(r"\begin{tabular}{c c c c c c c}")
        out.append(r"\toprule")
        out.append(r"seed & variant & slope & $R^2$ & $\lambda_{\min}$ & $\lambda_{\max}$ & modes \\")
        out.append(r"\midrule")
        for (seed, sv, rv, nv, loV, hiV), (_, sp, rp, np_, loP, hiP) in zip(rows_v, rows_p):
            out.append(f"{seed} & vanilla & {fmt(sv, 4)} & {fmt(rv, 4)} & "
                       f"{fmt_sci(loV, 3)} & {fmt_sci(hiV, 3)} & {nv} \\\\")
            out.append(f"{seed} & precond & {fmt(sp, 4)} & {fmt(rp, 4)} & "
                       f"{fmt_sci(loP, 3)} & {fmt_sci(hiP, 3)} & {np_} \\\\")
        sv_arr = np.array([r[1] for r in rows_v])
        sp_arr = np.array([r[1] for r in rows_p])
        out.append(r"\midrule")
        out.append(rf"\multicolumn{{2}}{{l}}{{vanilla mean $\pm$ std}} & "
                   rf"{fmt(float(sv_arr.mean()), 4)} $\pm$ {fmt(float(sv_arr.std()), 4)} & & & & \\")
        out.append(rf"\multicolumn{{2}}{{l}}{{precond mean $\pm$ std}} & "
                   rf"{fmt(float(sp_arr.mean()), 4)} $\pm$ {fmt(float(sp_arr.std()), 4)} & & & & \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{table}")
        out.append("")
        return "\n".join(out)

    lines.append(r"\subsection{Synthetic depth-1, $N=50$ (\texttt{preconditioner\_N50.json})}")
    lines.append(precond_table(
        "preconditioner_N50.json",
        r"Per-seed slopes, synthetic $N=50$, depth $1$.",
        "precond-N50"))

    lines.append(r"\subsection{CIFAR-10 head-only (\texttt{cifar10.json})}")
    lines.append(precond_table(
        "cifar10.json",
        r"Per-seed slopes, frozen-feature CIFAR-10 with depth-2 head.",
        "precond-cifar10"))

    lines.append(r"\subsection{CIFAR-10 end-to-end CNN (\texttt{cifar10\_end2end.json})}")
    lines.append(precond_table(
        "cifar10_end2end.json",
        r"Per-seed slopes, end-to-end CNN trained on CIFAR-10 from scratch.",
        "precond-cifar10-e2e"))

    # Also include triplet-satisfaction overall and per-bin from preconditioner_N50
    lines.append(r"\subsection{Triplet satisfaction (synthetic $N=50$)}")
    data = load("preconditioner_N50.json")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-seed overall triplet satisfaction and per-bin "
                 r"satisfaction for synthetic $N=50$.}")
    lines.append(r"\label{tab:precond-sat}")
    lines.append(r"\begin{tabular}{c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"seed & overall vanilla & overall precond & bin0 (vanilla / precond) & bin1 (vanilla / precond) \\")
    lines.append(r"\midrule")
    for seed_dict in data:
        seed = seed_dict["seed"]
        ov = seed_dict["overall_sat_vanilla"]
        op = seed_dict["overall_sat_precond"]
        sb_v = seed_dict["sat_by_bin_vanilla"]
        sb_p = seed_dict["sat_by_bin_precond"]
        def fnan(x):
            try:
                if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                    return "n/a"
            except Exception:
                return "n/a"
            return f"{x:.4f}"
        b0 = f"{fnan(sb_v[0])} / {fnan(sb_p[0])}"
        b1 = f"{fnan(sb_v[1])} / {fnan(sb_p[1])}" if len(sb_v) > 1 else "n/a"
        lines.append(f"{seed} & {fmt(ov, 4)} & {fmt(op, 4)} & {b0} & {b1} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX D: Recall@K =====================
def appendix_D():
    lines = []
    lines.append(r"\section{Recall@K: full per-seed curves}")
    lines.append(r"\label{app:recall}")
    lines.append("")
    lines.append(
        r"Held-out CIFAR-10 retrieval metrics for both triplet-difficulty "
        r"regimes. For each of $K \in \{1, 5, 10, 20\}$ we report all five "
        r"seeds' vanilla and preconditioned recall, the per-seed "
        r"difference $\Delta = \text{precond} - \text{vanilla}$, and a "
        r"paired $t$-statistic / two-sided $p$-value over the five paired "
        r"differences."
    )
    lines.append("")

    def paired_t(diffs):
        diffs = np.asarray(diffs, dtype=float)
        n = diffs.size
        if n < 2:
            return float("nan"), float("nan")
        mean = diffs.mean()
        sd = diffs.std(ddof=1)
        if sd == 0:
            return float("inf"), 0.0
        t = mean / (sd / math.sqrt(n))
        # two-sided p via student-t survival; approximate via scipy if available
        try:
            from scipy import stats
            p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1))
        except Exception:
            # rough normal approximation
            from math import erfc
            p = erfc(abs(t) / math.sqrt(2))
        return float(t), float(p)

    for tag, fname, label in (
        ("Easy triplets", "recall_easy.json", "recall-easy"),
        ("Hard triplets", "recall_hard.json", "recall-hard"),
    ):
        lines.append(rf"\subsection{{{tag}}}")
        data = load(fname)
        Ks = data[0]["Ks"]
        for K in Ks:
            key = f"recall@{K}"
            v_list = [s["vanilla"][key] for s in data]
            p_list = [s["precond"][key] for s in data]
            d_list = [p - v for v, p in zip(v_list, p_list)]
            t, p = paired_t(d_list)
            mu_v, sd_v = float(np.mean(v_list)), float(np.std(v_list))
            mu_p, sd_p = float(np.mean(p_list)), float(np.std(p_list))
            mu_d, sd_d = float(np.mean(d_list)), float(np.std(d_list))
            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(rf"\caption{{{tag}, recall@{K} per seed.}}")
            lines.append(rf"\label{{tab:{label}-K{K}}}")
            lines.append(r"\begin{tabular}{c c c c}")
            lines.append(r"\toprule")
            lines.append(r"seed & vanilla & precond & $\Delta$ \\")
            lines.append(r"\midrule")
            for s, v, pp, d in zip(data, v_list, p_list, d_list):
                lines.append(f"{s['seed']} & {fmt(v, 4)} & {fmt(pp, 4)} & {fmt(d, 4)} \\\\")
            lines.append(r"\midrule")
            lines.append(rf"mean $\pm$ std & {fmt(mu_v, 4)} $\pm$ {fmt(sd_v, 4)} & "
                         rf"{fmt(mu_p, 4)} $\pm$ {fmt(sd_p, 4)} & "
                         rf"{fmt(mu_d, 4)} $\pm$ {fmt(sd_d, 4)} \\")
            lines.append(rf"paired $t$ / $p$ & \multicolumn{{3}}{{l}}{{$t = "
                         rf"{fmt(t, 3)}$, two-sided $p = {fmt(p, 4)}$}} \\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX E: phase transition =====================
def appendix_E():
    lines = []
    lines.append(r"\section{Phase-transition full trajectory}")
    lines.append(r"\label{app:phase}")
    lines.append("")
    lines.append(
        r"Effective rank and triplet loss as a function of the nuclear-norm "
        r"regularization strength $\lambda$, sweeping $\lambda$ across 25 "
        r"log-spaced points. We tabulate every $\lambda$ value and report "
        r"the per-seed rank and loss, plus the cross-seed mean $\pm$ std."
    )
    lines.append("")
    data = load("phase_transition.json")
    lambdas = data[0]["lambdas"]
    lines.append(r"\begin{longtable}{c c c c c c c c}")
    lines.append(r"\caption{Phase-transition sweep; per-seed rank and loss.}\\")
    lines.append(r"\label{tab:phase-traj}\\")
    lines.append(r"\toprule")
    lines.append(r"$\lambda$ & seed 0 (rank/loss) & seed 1 & seed 2 & seed 3 & seed 4 & rank mean $\pm$ std & loss mean $\pm$ std \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"$\lambda$ & seed 0 (rank/loss) & seed 1 & seed 2 & seed 3 & seed 4 & rank mean $\pm$ std & loss mean $\pm$ std \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")
    for j, lam in enumerate(lambdas):
        ranks = [s["ranks"][j] for s in data]
        losses = [s["losses"][j] for s in data]
        cells = " & ".join(f"{r}/{l:.2f}" for r, l in zip(ranks, losses))
        ru, rs = float(np.mean(ranks)), float(np.std(ranks))
        lu, ls = float(np.mean(losses)), float(np.std(losses))
        lines.append(f"{lam:.4f} & {cells} & {fmt(ru, 2)} $\\pm$ {fmt(rs, 2)} & "
                     f"{fmt(lu, 3)} $\\pm$ {fmt(ls, 3)} \\\\")
    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX F: rank vs width =====================
def appendix_F():
    lines = []
    lines.append(r"\section{Rank-vs-width: full sweep}")
    lines.append(r"\label{app:rank-width}")
    lines.append("")
    lines.append(
        r"For each network width $M$ in the sweep, the learned effective "
        r"rank, final triplet loss, and triplet-satisfaction rate. "
        r"$N = 10$ for this experiment, so $M \ge N$ regimes test the "
        r"capacity-saturated case."
    )
    lines.append("")
    data = load("rank_vs_width.json")
    Ms = data[0]["M_values"]
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Rank, loss, satisfaction by width and seed.}")
    lines.append(r"\label{tab:rank-width-perseed}")
    lines.append(r"\begin{tabular}{c | " + "c " * len(data) + "| c c}")
    lines.append(r"\toprule")
    lines.append(r"$M$ & " + " & ".join(f"seed {s['seed']}" for s in data) +
                 r" & rank mean $\pm$ std & loss mean $\pm$ std \\")
    lines.append(r"\midrule")
    for j, M in enumerate(Ms):
        ranks = [s["ranks"][j] for s in data]
        losses = [s["losses"][j] for s in data]
        cells = " & ".join(f"{ranks[k]}/{losses[k]:.2f}" for k in range(len(data)))
        ru, rs = float(np.mean(ranks)), float(np.std(ranks))
        lu, ls = float(np.mean(losses)), float(np.std(losses))
        lines.append(f"{M} & {cells} & {fmt(ru, 2)} $\\pm$ {fmt(rs, 2)} & "
                     f"{fmt(lu, 3)} $\\pm$ {fmt(ls, 3)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # also satisfaction
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Triplet satisfaction by width and seed.}")
    lines.append(r"\label{tab:rank-width-sat}")
    lines.append(r"\begin{tabular}{c | " + "c " * len(data) + "| c}")
    lines.append(r"\toprule")
    lines.append(r"$M$ & " + " & ".join(f"seed {s['seed']}" for s in data) +
                 r" & sat mean $\pm$ std \\")
    lines.append(r"\midrule")
    for j, M in enumerate(Ms):
        sats = [s["satisfactions"][j] for s in data]
        cells = " & ".join(f"{sats[k]:.3f}" for k in range(len(data)))
        u, sd = float(np.mean(sats)), float(np.std(sats))
        lines.append(f"{M} & {cells} & {fmt(u, 3)} $\\pm$ {fmt(sd, 3)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX G: rank vs spectral =====================
def appendix_G():
    lines = []
    lines.append(r"\section{Rank-vs-spectral trajectory}")
    lines.append(r"\label{app:rank-spec}")
    lines.append("")
    lines.append(
        r"Per-epoch rank and Frobenius-norm spectral displacement on the "
        r"same trajectory. The synthetic experiment uses $N = 50$, depth "
        r"$1$, width $1000$. \texttt{spectral\_diff} columns are "
        r"$\| \widetilde{G}(t) - \widetilde{G}(0) \|_F$ for the indicated "
        r"variant."
    )
    lines.append("")
    data = load("rank_vs_spectral.json")
    epochs = data[0]["epochs"]
    # Tabulate every 4th epoch (~10 rows)
    idxs = list(range(0, len(epochs), max(1, len(epochs) // 10)))
    if (len(epochs) - 1) not in idxs:
        idxs.append(len(epochs) - 1)
    for seed_dict in data:
        seed = seed_dict["seed"]
        lines.append(rf"\subsection*{{Seed {seed}}}")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{Seed {seed}: rank trajectory and spectral "
                     rf"displacement, $N = 50$ depth-1.}}")
        lines.append(rf"\label{{tab:ranspec-s{seed}}}")
        lines.append(r"\begin{tabular}{c c c c c c c c}")
        lines.append(r"\toprule")
        lines.append(r"epoch & rank V & rank P & spec\_diff V & spec\_diff P & loss V & loss P \\")
        lines.append(r"\midrule")
        for i in idxs:
            lines.append(
                f"{epochs[i]} & {seed_dict['rank_vanilla'][i]} & "
                f"{seed_dict['rank_precond'][i]} & "
                f"{fmt(seed_dict['spectral_diff_vanilla'][i], 4)} & "
                f"{fmt(seed_dict['spectral_diff_precond'][i], 4)} & "
                f"{fmt(seed_dict['vanilla_loss'][i], 3)} & "
                f"{fmt(seed_dict['precond_loss'][i], 3)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
    # cross-seed summary at final epoch
    lines.append(r"\subsection*{Cross-seed summary at final epoch}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Final-epoch cross-seed summary.}")
    lines.append(r"\label{tab:ranspec-summary}")
    lines.append(r"\begin{tabular}{l c c}")
    lines.append(r"\toprule")
    lines.append(r"metric & vanilla & precond \\")
    lines.append(r"\midrule")
    for key, label in (
        ("rank_vanilla", "rank V"),
        ("rank_precond", "rank P"),
    ):
        pass
    rv = [s["rank_vanilla"][-1] for s in data]
    rp = [s["rank_precond"][-1] for s in data]
    sv = [s["spectral_diff_vanilla"][-1] for s in data]
    sp = [s["spectral_diff_precond"][-1] for s in data]
    lv = [s["vanilla_loss"][-1] for s in data]
    lp = [s["precond_loss"][-1] for s in data]
    lines.append(rf"effective rank & {fmt(np.mean(rv),2)} $\pm$ {fmt(np.std(rv),2)} & "
                 rf"{fmt(np.mean(rp),2)} $\pm$ {fmt(np.std(rp),2)} \\")
    lines.append(rf"spectral displacement & {fmt(np.mean(sv),3)} $\pm$ {fmt(np.std(sv),3)} & "
                 rf"{fmt(np.mean(sp),3)} $\pm$ {fmt(np.std(sp),3)} \\")
    lines.append(rf"final loss & {fmt(np.mean(lv),3)} $\pm$ {fmt(np.std(lv),3)} & "
                 rf"{fmt(np.mean(lp),3)} $\pm$ {fmt(np.std(lp),3)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX H: NTK eigenvalue spectra =====================
def appendix_H():
    lines = []
    lines.append(r"\section{NTK eigenvalue spectra}")
    lines.append(r"\label{app:eigvals}")
    lines.append("")
    lines.append(
        r"For each configuration where the empirical NTK was diagonalized, "
        r"we list the leading and trailing eigenvalues at the standard seed "
        r"($\text{seed}=0$). Eigenvalues are sorted in decreasing order. "
        r"For $N \le 20$ we list all eigenvalues; otherwise we list the top "
        r"10 and bottom 5 with the count of intermediate eigenvalues."
    )
    lines.append("")
    configs = []
    for L in (1, 2):
        for N in (30, 100, 1000):
            configs.append((f"spectral_bias_N{N}_L{L}.json", f"$N={N}$, $L={L}$ (synthetic)"))
    configs.append(("fashion_mnist.json", "Fashion-MNIST $N = 200$ depth-2"))
    configs.append(("cifar10.json", "CIFAR-10 head-only $N = 400$ depth-2"))
    configs.append(("cifar10_end2end.json", "CIFAR-10 end-to-end CNN $N = 100$"))
    configs.append(("preconditioner_N50.json", "synthetic $N=50$ depth-1"))
    for fname, label in configs:
        data = load(fname)
        eigvals = sorted(data[0]["eigvals_K"], reverse=True)
        N_total = len(eigvals)
        lines.append(rf"\subsection*{{{label} \quad (n = {N_total})}}")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{Eigenvalue spectrum for {label}.}}")
        lines.append(r"\begin{tabular}{c c}")
        lines.append(r"\toprule")
        lines.append(r"index (sorted desc) & $\lambda$ \\")
        lines.append(r"\midrule")
        if N_total <= 20:
            for i, lam in enumerate(eigvals):
                lines.append(f"{i} & {fmt_sci(lam, 4)} \\\\")
        else:
            for i in range(10):
                lines.append(f"{i} & {fmt_sci(eigvals[i], 4)} \\\\")
            lines.append(rf"\multicolumn{{2}}{{c}}{{$\cdots$ ({N_total - 15} intermediate eigenvalues) $\cdots$}} \\")
            for i in range(N_total - 5, N_total):
                lines.append(f"{i} & {fmt_sci(eigvals[i], 4)} \\\\")
        lines.append(r"\midrule")
        lam = np.array(eigvals)
        lam_pos = lam[lam > 0]
        cond = float(lam_pos.max() / lam_pos.min()) if lam_pos.size > 0 else float("nan")
        eff_rank = float((lam.sum() ** 2) / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else float("nan")
        lines.append(rf"$\lambda_{{\max}}/\lambda_{{\min, +}}$ & {fmt_sci(cond, 3)} \\")
        lines.append(rf"effective rank $\mathrm{{tr}}(K)^2/\|K\|_F^2$ & {fmt(eff_rank, 3)} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX I: per-mode displacement tables =====================
def appendix_I():
    lines = []
    lines.append(r"\section{Per-mode displacement tables}")
    lines.append(r"\label{app:per-mode}")
    lines.append("")
    lines.append(
        r"For the smaller configurations where listing every mode is "
        r"feasible, we report $\lambda_i$ alongside the per-seed final "
        r"displacement $|\widetilde{G}_{ii}(T) - \widetilde{G}_{ii}(0)|$. "
        r"Larger configurations are summarized by listing the top-10 and "
        r"bottom-10 modes only, since the same information is captured by "
        r"the slope/$R^2$ of Appendix~\ref{app:per-seed-slopes}."
    )
    lines.append("")
    # Smaller configs first - N=30 L=1 and L=2 listed in full
    for L in (1, 2):
        N = 30
        fname = f"spectral_bias_N{N}_L{L}.json"
        data = load(fname)
        lines.append(rf"\subsection{{$N={N}$, $L={L}$ (full)}}")
        eig0 = data[0]["eigvals_K"]
        lines.append(r"\begin{longtable}{c c c c c c c}")
        lines.append(rf"\caption{{Per-mode displacement, $N={N}$, $L={L}$. "
                     rf"Columns are seed-0 eigenvalues then "
                     rf"$|\widetilde{{G}}_{{ii}}(T)-\widetilde{{G}}_{{ii}}(0)|$ "
                     rf"for each seed.}}\\")

        lines.append(rf"\label{{tab:permode-N{N}-L{L}}}\\")
        lines.append(r"\toprule")
        lines.append(r"i & $\lambda_i$(s0) & disp s0 & disp s1 & disp s2 & disp s3 & disp s4 \\")
        lines.append(r"\midrule")
        lines.append(r"\endfirsthead")
        lines.append(r"\toprule")
        lines.append(r"i & $\lambda_i$(s0) & disp s0 & disp s1 & disp s2 & disp s3 & disp s4 \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")
        lines.append(r"\bottomrule")
        lines.append(r"\endfoot")
        for i in range(len(eig0)):
            cells = []
            for s in data:
                gf = s["G_modes_history"][-1][i]
                g0 = s["G0_modes"][i]
                cells.append(fmt_sci(abs(gf - g0), 3))
            lines.append(f"{i} & {fmt_sci(eig0[i], 4)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\end{longtable}")
        lines.append("")

    # Bigger configs: top-10 / bottom-10 only
    for N, L in [(100, 1), (100, 2), (1000, 1), (1000, 2)]:
        fname = f"spectral_bias_N{N}_L{L}.json"
        data = load(fname)
        lines.append(rf"\subsection{{$N={N}$, $L={L}$ (top/bottom 10)}}")
        eig0 = data[0]["eigvals_K"]
        # use sorted-desc indices on seed 0
        order = np.argsort(-np.array(eig0))
        sel = list(order[:10]) + list(order[-10:])
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{$N={N}$, $L={L}$: top-10 and bottom-10 eigenmodes by seed-0 ordering.}}")
        lines.append(rf"\label{{tab:permode-N{N}-L{L}}}")
        lines.append(r"\begin{tabular}{c c c c c c c c}")
        lines.append(r"\toprule")
        lines.append(r"rank & idx & $\lambda$(s0) & disp s0 & disp s1 & disp s2 & disp s3 & disp s4 \\")
        lines.append(r"\midrule")
        for r, i in enumerate(sel[:10]):
            cells = []
            for s in data:
                gf = s["G_modes_history"][-1][i]
                g0 = s["G0_modes"][i]
                cells.append(fmt_sci(abs(gf - g0), 3))
            lines.append(f"top {r} & {i} & {fmt_sci(eig0[i], 4)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        for r, i in enumerate(sel[10:]):
            cells = []
            for s in data:
                gf = s["G_modes_history"][-1][i]
                g0 = s["G0_modes"][i]
                cells.append(fmt_sci(abs(gf - g0), 3))
            lines.append(f"bot {r} & {i} & {fmt_sci(eig0[i], 4)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
    return "\n".join(lines)


# ===================== APPENDIX J: end-to-end CNN training details =====================
def appendix_J():
    lines = []
    lines.append(r"\section{End-to-end CNN training details}")
    lines.append(r"\label{app:e2e}")
    lines.append("")
    data = load("cifar10_end2end.json")
    cfg = data[0]["config"]
    lines.append(r"\subsection*{Configuration (from JSON)}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{End-to-end CIFAR-10 CNN training configuration.}")
    lines.append(r"\label{tab:e2e-config}")
    lines.append(r"\begin{tabular}{l c}")
    lines.append(r"\toprule")
    lines.append(r"key & value \\")
    lines.append(r"\midrule")
    for k, v in cfg.items():
        if isinstance(v, list):
            v_str = ", ".join(str(x) for x in v[:8]) + ("..." if len(v) > 8 else "")
        else:
            v_str = str(v)
        lines.append(rf"\texttt{{{k.replace('_', '\\_')}}} & {v_str} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\subsection*{Per-epoch losses, all seeds}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{End-to-end CIFAR-10 CNN per-epoch losses.}")
    lines.append(r"\label{tab:e2e-losses}")
    epochs = data[0]["epochs"]
    lines.append(r"\begin{tabular}{c | " + "c c " * len(data) + "}")
    lines.append(r"\toprule")
    header = "epoch"
    for s in data:
        header += rf" & van s{s['seed']} & pre s{s['seed']}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for i, ep in enumerate(epochs):
        row = f"{ep}"
        for s in data:
            row += f" & {fmt(s['vanilla_loss'][i], 3)} & {fmt(s['precond_loss'][i], 3)}"
        lines.append(row + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # final slopes per seed
    lines.append(r"\subsection*{Per-seed final slopes}")
    rows = []
    for s in data:
        eig = s["eigvals_K"]
        dv = np.array(s["delta_vanilla"][-1])
        dp = np.array(s["delta_precond"][-1])
        sv, rv, nv, _ = fit_loglog(eig, np.abs(dv))
        sp, rp, npc, _ = fit_loglog(eig, np.abs(dp))
        rows.append((s["seed"], sv, rv, nv, sp, rp, npc))
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{End-to-end CIFAR-10 CNN: per-seed slopes and $R^2$.}")
    lines.append(r"\label{tab:e2e-slopes}")
    lines.append(r"\begin{tabular}{c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"seed & vanilla slope & vanilla $R^2$ & van modes & precond slope & precond $R^2$ & pre modes \\")
    lines.append(r"\midrule")
    for seed, sv, rv, nv, sp, rp, npc in rows:
        lines.append(f"{seed} & {fmt(sv, 4)} & {fmt(rv, 4)} & {nv} & {fmt(sp, 4)} & {fmt(rp, 4)} & {npc} \\\\")
    sv_arr = np.array([r[1] for r in rows])
    sp_arr = np.array([r[4] for r in rows])
    lines.append(r"\midrule")
    lines.append(rf"mean $\pm$ std & {fmt(float(sv_arr.mean()),4)} $\pm$ {fmt(float(sv_arr.std()),4)} & "
                 rf"& & {fmt(float(sp_arr.mean()),4)} $\pm$ {fmt(float(sp_arr.std()),4)} & & \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    snippets = {
        "A": appendix_A(),
        "B": appendix_B(),
        "C": appendix_C(),
        "D": appendix_D(),
        "E": appendix_E(),
        "F": appendix_F(),
        "G": appendix_G(),
        "H": appendix_H(),
        "I": appendix_I(),
        "J": appendix_J(),
    }
    for k, v in snippets.items():
        path = os.path.join(OUT, f"appendix_{k}.tex")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(v)
        print(f"wrote {path} ({len(v)} chars)")


if __name__ == "__main__":
    main()
