"""CIFAR-10 metric-learning experiment.

Uses pretrained ResNet-18 to extract penultimate-layer features, then trains
a 2-layer ReLU head with triplet loss on a subset. Shows:
  (i)  Spectral-bias phenomenon transfers to CIFAR-10 features.
  (ii) The preconditioner from runners/preconditioner.py flattens it on
       real data.

We use a fixed feature extractor so the N x d input matrix is well-defined
and the depth-L NTK on it can be computed. The 2-layer head is trained
end-to-end.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from kbm.kernels import depth_L_ntk_kernel
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model
from runners.preconditioner import train_preconditioned


@dataclass
class CIFARConfig:
    N: int = 400
    n_classes: int = 10
    feature_dim: int = 128
    depth: int = 2
    width: int = 1024
    n_triplets: int = 2000
    n_epochs: int = 1500
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 100
    eps: float = 1e-2
    data_root: str = "./data"


def _resnet_features(images: torch.Tensor, device: str) -> torch.Tensor:
    """Extract penultimate features via pretrained ResNet-18."""
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    with torch.no_grad():
        feats = model(images.to(device))
    return feats.cpu()


def _load_subset(cfg: CIFARConfig, seed: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tfm = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=tfm,
    )
    targets = np.array(ds.targets)
    per_class = cfg.N // cfg.n_classes
    idxs, labels = [], []
    for c in range(cfg.n_classes):
        candidates = np.where(targets == c)[0]
        chosen = rng.choice(candidates, size=per_class, replace=False)
        idxs.extend(chosen.tolist())
        labels.extend([c] * per_class)

    images = torch.stack([ds[i][0] for i in idxs])
    labels = np.array(labels)

    # Process in batches to fit on 8GB GPU.
    feats = []
    bsz = 64
    for i in range(0, len(images), bsz):
        batch = images[i:i + bsz]
        feats.append(_resnet_features(batch, device))
    feats = torch.cat(feats, dim=0).numpy()  # (N, 512)

    feats_centered = feats - feats.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(feats_centered, full_matrices=False)
    Z = U[:, :cfg.feature_dim] * S[:cfg.feature_dim]
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z, labels


def _make_triplets(labels: np.ndarray, n_triplets: int, seed: int):
    rng = np.random.default_rng(seed)
    by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    classes = list(by_class.keys())
    triplets = []
    while len(triplets) < n_triplets:
        c_pos = rng.choice(classes)
        c_neg = rng.choice([c for c in classes if c != c_pos])
        a, p = rng.choice(by_class[c_pos], 2, replace=False)
        n = rng.choice(by_class[c_neg])
        triplets.append((int(a), int(p), int(n)))
    return triplets


def run_one(seed: int, cfg: CIFARConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Z, labels = _load_subset(cfg, seed, device)
    triplets = _make_triplets(labels, cfg.n_triplets, seed)

    K = depth_L_ntk_kernel(Z, cfg.depth)
    eigvals_K, U = np.linalg.eigh(K)
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]; U = U[:, order]

    # Vanilla
    torch.manual_seed(seed); np.random.seed(seed)
    model_v = MetricNetwork(cfg.feature_dim, [cfg.width] * cfg.depth)
    F0 = embeddings_from_model(model_v, Z, device="cpu")
    G0 = F0 @ F0.T
    G0_modes = np.diag(U.T @ G0 @ U)

    out_v = train_network(
        model_v, Z, triplets,
        TrainConfig(n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
                    record_every=cfg.record_every),
        device=device,
    )

    # Preconditioned
    torch.manual_seed(seed); np.random.seed(seed)
    model_p = MetricNetwork(cfg.feature_dim, [cfg.width] * cfg.depth)
    out_p = train_preconditioned(
        model_p, Z, triplets, K,
        n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
        record_every=cfg.record_every, eps=cfg.eps, device=device,
    )

    def proj_diag(hist):
        return np.stack([np.diag(U.T @ G @ U) for G in hist], axis=0)

    G_modes_v = proj_diag(out_v["G_history"])
    G_modes_p = proj_diag(out_p["G_history"])
    delta_v = G_modes_v - G0_modes[None, :]
    delta_p = G_modes_p - G0_modes[None, :]

    return {
        "seed": seed,
        "epochs": out_v["epochs"].tolist(),
        "vanilla_loss": out_v["losses"].tolist(),
        "precond_loss": out_p["losses"].tolist(),
        "eigvals_K": eigvals_K.tolist(),
        "G0_modes": G0_modes.tolist(),
        "delta_vanilla": delta_v.tolist(),
        "delta_precond": delta_p.tolist(),
        "config": asdict(cfg),
    }


def run_sweep(cfg: CIFARConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
