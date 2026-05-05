"""Analyze long-tailed CIFAR-10 retrieval: per-class-frequency-band Recall@K."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
runs = json.load(open(RES / "recall_longtail_cifar10.json"))
n = len(runs)
counts = runs[0]["class_counts"]
print(f"Long-tailed CIFAR-10, {n} seeds. Class counts: {counts}")
print(f"Head 3 = {counts[:3]}, middle 4 = {counts[3:7]}, tail 3 = {counts[7:]}\n")

# Aggregate per-class recall across seeds
def stratified(method, K, band):
    """band in {'head','middle','tail','overall'}"""
    if band == "head":   classes = list(range(3))
    elif band == "middle": classes = list(range(3, 7))
    elif band == "tail": classes = list(range(7, 10))
    elif band == "overall": classes = list(range(10))
    vals = []
    for r in runs:
        per = r[method]["per_class"][str(K)]
        vals.append(np.mean([per[str(c)] for c in classes]))
    return np.array(vals)


for band in ("head", "middle", "tail"):
    print(f"=== {band.upper()} CLASSES ===")
    for K in (1, 5, 10, 20):
        v = stratified("vanilla", K, band)
        p = stratified("precond", K, band)
        delta = p - v
        t, pval = stats.ttest_rel(p, v)
        pval_b = min(1.0, pval * 4)
        print(f"  R@{K:>2}: vanilla {v.mean():.3f}±{v.std():.3f}  "
              f"precond {p.mean():.3f}±{p.std():.3f}  "
              f"Δ={delta.mean():+.4f}  p={pval:.4f}  Bonf={pval_b:.4f}")
    print()
