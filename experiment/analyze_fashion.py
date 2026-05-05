"""Analyze Fashion-MNIST retrieval defense results (10 seeds, easy+hard)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def report(diff: str):
    runs = json.load(open(RES / f"recall_fashion_{diff}.json"))
    Ks = runs[0]["Ks"]
    print(f"\n=== Fashion-MNIST Recall ({diff}) — {len(runs)} seeds ===")
    for K in Ks:
        v = np.array([r["vanilla"][f"recall@{K}"] for r in runs])
        p = np.array([r["precond"][f"recall@{K}"] for r in runs])
        delta = p - v
        t, pval = stats.ttest_rel(p, v)
        pval_bonf = min(1.0, pval * len(Ks))
        print(f"  R@{K:>2}: vanilla {v.mean():.3f}±{v.std():.3f}  "
              f"precond {p.mean():.3f}±{p.std():.3f}  "
              f"Δ={delta.mean():+.4f}  raw p={pval:.4f}  Bonferroni p={pval_bonf:.4f}")


report("easy")
report("hard")
