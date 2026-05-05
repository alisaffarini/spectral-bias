"""Aggregate Recall@K results across datasets with Bonferroni correction."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def report_recall(path: Path, label: str):
    if not path.exists():
        print(f"[{label}] missing"); return
    runs = json.load(open(path))
    Ks = runs[0]["Ks"]
    print(f"\n=== {label} — {len(runs)} seeds ===")
    for K in Ks:
        v = np.array([r["vanilla"][f"recall@{K}"] for r in runs])
        p = np.array([r["precond"][f"recall@{K}"] for r in runs])
        delta = p - v
        t, pval = stats.ttest_rel(p, v)
        pval_b = min(1.0, pval * len(Ks))
        print(f"  R@{K:>2}: vanilla {v.mean():.3f}±{v.std():.3f}  "
              f"precond {p.mean():.3f}±{p.std():.3f}  "
              f"Δ={delta.mean():+.4f}  p={pval:.4f}  Bonferroni={pval_b:.4f}")


def report_drift(path: Path):
    if not path.exists():
        print("\n[drift] missing"); return
    runs = json.load(open(path))
    print(f"\n=== NTK drift — {len(runs)} seeds ===")
    by_epoch = {}
    for r in runs:
        for ck in r["checkpoints"]:
            by_epoch.setdefault(ck["epoch"], []).append(ck["slope_at_current_K"])
    for ep in sorted(by_epoch):
        vals = np.array(by_epoch[ep])
        print(f"  epoch {ep:>5}: slope {vals.mean():.3f} ± {vals.std():.3f}  (n={len(vals)})")


for diff in ("easy", "hard"):
    report_recall(RES / f"recall_cub_{diff}.json", f"CUB-200 ({diff})")
for diff in ("easy", "hard"):
    report_recall(RES / f"recall_cifar100_{diff}.json", f"CIFAR-100 ({diff})")
for diff in ("easy", "hard"):
    report_recall(RES / f"recall_aircraft_{diff}.json", f"FGVC-Aircraft ({diff})")
for diff in ("easy", "hard"):
    report_recall(RES / f"recall_pet_{diff}.json", f"Oxford-IIIT Pet ({diff})")
for diff in ("easy", "hard"):
    report_recall(RES / f"recall_flowers_{diff}.json", f"Flowers-102 ({diff})")
report_drift(RES / "ntk_drift.json")
