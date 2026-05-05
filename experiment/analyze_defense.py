"""Aggregate defense-experiment results: Bonferroni-corrected paired tests
on combined seeds, alpha-sweep U-shape statistics, semi-hard retrieval.

Run after `run_defense.py` produces JSONs in `../results/`.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load(path):
    with open(path) as f:
        return json.load(f)


def combined_recall(difficulty: str):
    """Combine original 5 seeds + 5 new seeds for a paired t-test with
    Bonferroni correction across the four R@K endpoints."""
    base = load(RES / f"recall_{difficulty}.json")
    extra_path = RES / f"recall_{difficulty}_extra_seeds.json"
    extra = load(extra_path) if extra_path.exists() else []
    all_runs = base + extra
    Ks = all_runs[0]["Ks"]
    print(f"\n=== Recall ({difficulty}) — {len(all_runs)} seeds ===")
    rows = []
    for K in Ks:
        v = np.array([r["vanilla"][f"recall@{K}"] for r in all_runs])
        p = np.array([r["precond"][f"recall@{K}"] for r in all_runs])
        delta = p - v
        t, pval = stats.ttest_rel(p, v)
        # Bonferroni across the 4 K's = multiply by 4 (capped at 1).
        pval_bonf = min(1.0, pval * len(Ks))
        rows.append((K, v.mean(), v.std(), p.mean(), p.std(),
                     delta.mean(), delta.std(), pval, pval_bonf))
        print(f"  R@{K:>2}: vanilla {v.mean():.3f}±{v.std():.3f}  "
              f"precond {p.mean():.3f}±{p.std():.3f}  "
              f"Δ={delta.mean():+.4f}  raw p={pval:.4f}  "
              f"Bonferroni p={pval_bonf:.4f}")
    return rows


def alpha_sweep_summary():
    path = RES / "alpha_sweep.json"
    if not path.exists():
        print("\n[alpha_sweep] not yet present"); return None
    data = load(path)
    alphas = data[0]["alphas"]
    by_alpha = defaultdict(list)
    for r in data:
        for a in alphas:
            by_alpha[a].append(r["per_alpha"][f"{a:.1f}"]["slope"])
    print(f"\n=== Alpha sweep — {len(data)} seeds ===")
    print("alpha   slope (mean ± std)")
    for a in alphas:
        s = np.array(by_alpha[a])
        print(f"  {a:>3.1f}   {s.mean():.3f} ± {s.std():.3f}")
    return by_alpha


def semi_hard_summary():
    path = RES / "recall_semi_hard.json"
    if not path.exists():
        print("\n[semi-hard] not yet present"); return None
    runs = load(path)
    Ks = runs[0]["Ks"]
    print(f"\n=== Recall (semi-hard, embedding-space) — {len(runs)} seeds ===")
    for K in Ks:
        v = np.array([r["vanilla"][f"recall@{K}"] for r in runs])
        p = np.array([r["precond"][f"recall@{K}"] for r in runs])
        delta = p - v
        t, pval = stats.ttest_rel(p, v)
        pval_bonf = min(1.0, pval * len(Ks))
        print(f"  R@{K:>2}: vanilla {v.mean():.3f}±{v.std():.3f}  "
              f"precond {p.mean():.3f}±{p.std():.3f}  "
              f"Δ={delta.mean():+.4f}  raw p={pval:.4f}  "
              f"Bonferroni p={pval_bonf:.4f}")


if __name__ == "__main__":
    for diff in ("easy", "hard"):
        try:
            combined_recall(diff)
        except FileNotFoundError as e:
            print(f"[recall {diff}] missing: {e}")
    alpha_sweep_summary()
    semi_hard_summary()
