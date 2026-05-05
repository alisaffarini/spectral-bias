"""Final 10-seed summary for scaled CIFAR-10 retrieval (hard + easy).
Reports mean +/- std, t-stat, raw p, Bonferroni-corrected p across 4 R@K endpoints."""
import json
from pathlib import Path
from math import sqrt
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent

def t_to_p(t, df):
    # Two-sided p-value for Student's t.
    # Series expansion / scipy unavailable; use a simple lookup approximation.
    # For df=8 or df=9, common critical values:
    #   df=9 two-sided: t=2.262 -> p=0.05; t=3.250 -> p=0.01
    # Linearly interpolate in log-p.
    import math
    if t < 0: t = -t
    table = {9: [(0.0, 1.0), (1.833, 0.10), (2.262, 0.05), (2.821, 0.02),
                  (3.250, 0.01), (4.781, 0.001)],
             8: [(0.0, 1.0), (1.860, 0.10), (2.306, 0.05), (2.896, 0.02),
                  (3.355, 0.01), (5.041, 0.001)]}
    pts = table.get(df, table[9])
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            # interpolate in -log p
            lp = math.log(p0) + (math.log(p1) - math.log(p0)) * (t - t0) / (t1 - t0)
            return math.exp(lp)
    return pts[-1][1]


def summarize(json_path: Path, label: str, n_endpoints: int = 4):
    r = json.load(open(json_path))
    n = len(r)
    ks = r[0]['Ks']
    print(f"\n=== {label} ({n} seeds) ===")
    print(f"  {'K':>3}  {'vanilla':>13}  {'precond':>13}  {'delta(pp)':>10}  {'t':>5}  {'raw_p':>7}  {'Bonf*4':>8}")
    for k in ks:
        v = [s['vanilla'][f'recall@{k}'] for s in r]
        p = [s['precond'][f'recall@{k}'] for s in r]
        diffs = [pi - vi for vi, pi in zip(v, p)]
        d_mean = mean(diffs); d_std = stdev(diffs) if n > 1 else 0.0
        t = d_mean / (d_std / sqrt(n)) if d_std > 0 else float('inf')
        raw_p = t_to_p(t, n - 1)
        bonf = min(1.0, raw_p * n_endpoints)
        sig = '***' if bonf < 0.001 else ('**' if bonf < 0.01 else ('*' if bonf < 0.05 else ''))
        print(f"  {k:>3}  {mean(v):.4f}+/-{stdev(v):.3f}  {mean(p):.4f}+/-{stdev(p):.3f}  "
              f"{d_mean*100:+8.2f}  {t:5.2f}  {raw_p:7.4f}  {bonf:7.4f} {sig}")


if __name__ == '__main__':
    summarize(ROOT / 'recall_scaled_cifar10_hard.json', 'SCALED HARD (N_train=4000)')
    summarize(ROOT / 'recall_scaled_cifar10_easy.json', 'SCALED EASY (N_train=4000)')
