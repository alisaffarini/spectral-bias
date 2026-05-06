"""Empirically verify approximate diagonality of $\\widetilde E_0$
in $K$'s eigenbasis -- the load-bearing assumption for Cor 8.

For each seed: build $Z$ via ResNet-50 features (same protocol as scaled CIFAR-10),
form the depth-2 NTK $K$, diagonalize as $K = U\\Lambda U^\\top$,
compute $E_0 = \\partial L/\\partial G$ at initialization on triplet hinge loss
(pure data-side gradient, no head training), and project
$\\widetilde E_0 = U^\\top E_0 U$.

Report: fraction of Frobenius mass on the diagonal of $\\widetilde E_0$,
ratio of max off-diagonal entry to max diagonal entry, and per-mode
diagonal-vs-row-norm ratio (the quantity actually used in the
Cauchy-Schwarz step of the Thm 7 proof).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision

from kbm.kernels import depth_L_ntk_kernel
from runners.recall_eval import _make_triplets


@dataclass
class EtildeConfig:
    N_train: int = 4000
    n_classes: int = 10
    feature_dim: int = 256
    depth: int = 2
    n_triplets: int = 12000
    margin: float = 1.0
    triplet_difficulty: str = "hard"
    backbone: str = "resnet50"
    data_root: str = "./data"


def _make_backbone(name: str, device: str):
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        m = torchvision.models.resnet50(weights=weights)
    else:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Identity()
    return m.eval().to(device)


def _load_Z(cfg: EtildeConfig, seed: int, device: str):
    rng = np.random.default_rng(seed)
    ds = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True)
    targets = np.array(ds.targets)
    per_class = cfg.N_train // cfg.n_classes
    idxs, labels = [], []
    for c in range(cfg.n_classes):
        cand = np.where(targets == c)[0]
        chosen = rng.choice(cand, size=per_class, replace=False)
        idxs.extend(chosen.tolist())
        labels.extend([c] * per_class)
    arr = ds.data[np.asarray(idxs)]
    raw = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    labels = np.array(labels)
    backbone = _make_backbone(cfg.backbone, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats_list = []
    bsz = 128
    with torch.no_grad():
        for i in range(0, len(raw), bsz):
            b = raw[i:i + bsz].to(device, non_blocking=True).float() / 255.0
            b = torch.nn.functional.interpolate(b, size=224, mode="bilinear", align_corners=False)
            b = (b - mean) / std
            feats_list.append(backbone(b).cpu())
    feats = torch.cat(feats_list, dim=0).numpy()
    fc = feats - feats.mean(axis=0, keepdims=True)
    U_, S_, Vt_ = np.linalg.svd(fc, full_matrices=False)
    Z = U_[:, :cfg.feature_dim] * S_[:cfg.feature_dim]
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z, labels


def _compute_E0(Z, labels, triplets, margin):
    """Triplet hinge loss gradient w.r.t. G at G = ZZ^T (linear-head init).

    L = sum_t max(0, d(a,p) - d(a,n) + margin)
    where d(i,j) = G_{ii} + G_{jj} - 2 G_{ij}.
    Active triplets contribute +2 to G_{a,n}, -2 to G_{a,p}, +1 to (G_{a,a},
    G_{p,p}), -1 to G_{n,n}. We only need entries that are non-zero in E.
    """
    N = Z.shape[0]
    G = Z @ Z.T
    E = np.zeros((N, N), dtype=np.float64)
    active = 0
    for (a, p, n) in triplets:
        d_ap = G[a, a] + G[p, p] - 2 * G[a, p]
        d_an = G[a, a] + G[n, n] - 2 * G[a, n]
        if d_ap - d_an + margin > 0:
            active += 1
            E[a, p] += -1; E[p, a] += -1
            E[a, n] += +1; E[n, a] += +1
            E[a, a] += +1 - 1
            E[p, p] += +1
            E[n, n] += -1
    return E, active


def _diagonality_metrics(E_tilde, eigvals=None):
    diag = np.diag(E_tilde)
    off = E_tilde - np.diag(diag)
    fro_full = np.linalg.norm(E_tilde, 'fro')
    fro_diag = np.linalg.norm(diag)
    fro_off = np.linalg.norm(off, 'fro')
    diag_frac = (fro_diag ** 2) / (fro_full ** 2 + 1e-30)
    max_diag = float(np.max(np.abs(diag))) if diag.size else 0.0
    max_off = float(np.max(np.abs(off)))
    row_norms = np.linalg.norm(E_tilde, axis=1)
    op_norm = float(np.linalg.norm(E_tilde, 2))
    out = {
        "diag_frac_fro2": float(diag_frac),
        "off_to_diag_fro_ratio": float(fro_off / (fro_diag + 1e-30)),
        "max_off_over_max_diag": float(max_off / (max_diag + 1e-30)),
        "max_diag": max_diag,
        "max_off": max_off,
        "op_norm": op_norm,
        "median_row_norm_vs_op_norm": float(np.median(row_norms) / (op_norm + 1e-30)),
    }
    # === The actual (A3) form Cor 8 needs: |E_tilde[i,j]| <= c_E * lambda_i ===
    # For each off-diagonal (i,j) with i!=j, compute |E_tilde[i,j]| / lambda_i,
    # and check whether the maximum is bounded across the spectrum.
    if eigvals is not None:
        N = E_tilde.shape[0]
        lam = np.asarray(eigvals).astype(np.float64)
        # avoid division by zero / very small lam at the bottom -- cap by a
        # tiny floor lambda_floor = 1e-3 * lambda_max so we don't spuriously
        # blow up the ratio for the trailing eigenvalues.
        lam_floor = max(1e-3 * float(lam.max()), 1e-12)
        lam_safe = np.maximum(lam, lam_floor)
        # ratio[i,j] = |E_tilde[i,j]| / lambda_i
        absE = np.abs(E_tilde)
        np.fill_diagonal(absE, 0.0)  # only off-diagonals
        ratio_per_ij = absE / lam_safe[:, None]
        # Look at this ratio split by row-index quartiles (top/middle/bottom of spectrum)
        q = np.array_split(np.arange(N), 4)
        per_quartile = []
        for k, idx in enumerate(q):
            sub = ratio_per_ij[idx]
            per_quartile.append({
                "quartile": k,
                "lambda_range": [float(lam[idx[0]]), float(lam[idx[-1]])],
                "max_ratio": float(np.max(sub)),
                "mean_ratio": float(np.mean(sub)),
                "p99_ratio": float(np.percentile(sub, 99)),
            })
        out["A3_max_ratio_global"] = float(np.max(ratio_per_ij))
        out["A3_p99_ratio_global"] = float(np.percentile(ratio_per_ij, 99))
        out["A3_mean_ratio_global"] = float(np.mean(ratio_per_ij))
        out["A3_per_quartile"] = per_quartile
    return out


def run_one(seed: int, cfg: EtildeConfig, device: str = "cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"[etilde seed={seed}] loading Z...", flush=True)
    Z, labels = _load_Z(cfg, seed, device)
    print(f"  Z shape: {Z.shape}", flush=True)
    triplets = _make_triplets(labels, cfg.n_triplets, seed,
                               difficulty=cfg.triplet_difficulty, Z=Z)
    print(f"  computing depth-{cfg.depth} NTK...", flush=True)
    K = depth_L_ntk_kernel(Z, cfg.depth)
    eigvals, U = np.linalg.eigh(K)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; U = U[:, order]
    print(f"  computing E_0 = dL/dG and projecting to K's eigenbasis...", flush=True)
    E0, n_active = _compute_E0(Z, labels, triplets, cfg.margin)
    E_tilde = U.T @ E0 @ U
    metrics = _diagonality_metrics(E_tilde, eigvals=eigvals)
    metrics["seed"] = seed
    metrics["n_active_triplets"] = n_active
    metrics["n_total_triplets"] = len(triplets)
    metrics["lambda_max"] = float(eigvals[0])
    metrics["lambda_min"] = float(eigvals[-1])
    print(f"  diag_frac (fro^2) = {metrics['diag_frac_fro2']:.4f}", flush=True)
    print(f"  off/diag fro ratio = {metrics['off_to_diag_fro_ratio']:.4f}", flush=True)
    print(f"  max_off/max_diag = {metrics['max_off_over_max_diag']:.4f}", flush=True)
    return metrics


def run_sweep(cfg: EtildeConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {r["seed"] for r in results}
        seeds = [s for s in seeds if s not in done]
    else:
        results = []
    for s in seeds:
        results.append(run_one(s, cfg, device=device))
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  wrote partial -> {out_path} ({len(results)} seeds)", flush=True)
    return results
