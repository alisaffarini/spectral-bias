"""Fashion-MNIST triplet-loss experiment to verify the spectral-bias prediction
in a non-synthetic setting.

Setup:
  - Take a fixed subsample of size N from Fashion-MNIST.
  - Use raw pixel features (or PCA to d_in dims) as inputs to a 2-layer
    fully-connected ReLU embedder.
  - Form triplets from class labels (positive = same class, negative = diff).
  - Compute the depth-2 NTK kernel K on the inputs.
  - Train; track G(t) and project residual error into K's eigenbasis.
  - Verify: top-K-eigenvalue components of (G(t) - G(0)) decay fast; bottom
    components stay near zero.

Why this works as a demonstration: the prediction is a property of the kBM
flow that depends only on K being a generic kernel. It should hold on
Fashion-MNIST in the NTK regime exactly as in the synthetic setting.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms

from kbm.kernels import depth_L_ntk_kernel
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model


@dataclass
class FashionConfig:
    N: int = 200
    d_in: int = 64
    depth: int = 2
    width: int = 1024
    classes: tuple[int, ...] = (0, 1, 2, 3)
    n_epochs: int = 2000
    lr: float = 5e-5
    margin: float = 1.0
    record_every: int = 50
    n_triplets: int = 1000
    data_root: str = "./data"


def _load_subset(cfg: FashionConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.FashionMNIST(
        root=cfg.data_root, train=True, download=True, transform=tfm,
    )
    targets = np.array(ds.targets)
    keep_classes = set(cfg.classes)

    per_class = cfg.N // len(cfg.classes)
    idxs = []
    labels = []
    for c in cfg.classes:
        candidate = np.where(targets == c)[0]
        chosen = rng.choice(candidate, size=per_class, replace=False)
        idxs.extend(chosen.tolist())
        labels.extend([c] * per_class)
    idxs = np.array(idxs)
    labels = np.array(labels)

    images = np.stack([np.asarray(ds[i][0]).reshape(-1) for i in idxs], axis=0)
    images = images.astype(np.float32) / images.max()
    # Reduce to d_in via PCA on this batch (deterministic given seed).
    images_centered = images - images.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(images_centered, full_matrices=False)
    Z = U[:, : cfg.d_in] * S[: cfg.d_in]
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z, labels


def _make_triplets(labels: np.ndarray, n_triplets: int, seed: int) -> list[tuple[int, int, int]]:
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


def run_one(seed: int, cfg: FashionConfig, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Z, labels = _load_subset(cfg, seed)
    triplets = _make_triplets(labels, cfg.n_triplets, seed)

    K = depth_L_ntk_kernel(Z, cfg.depth)
    eigvals_K, U = np.linalg.eigh(K)
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]
    U = U[:, order]

    widths = [cfg.width] * cfg.depth
    model = MetricNetwork(cfg.d_in, widths)

    train_cfg = TrainConfig(
        n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
        record_every=cfg.record_every,
    )
    F0 = embeddings_from_model(model, Z, device="cpu")
    G0 = F0 @ F0.T
    out = train_network(model, Z, triplets, train_cfg, device=device)

    G_modes_history = []
    for G_t in out["G_history"]:
        G_modes_history.append(np.diag(U.T @ G_t @ U).tolist())
    G0_modes = np.diag(U.T @ G0 @ U).tolist()

    return {
        "seed": seed,
        "epochs": out["epochs"].tolist(),
        "eigvals_K": eigvals_K.tolist(),
        "G_modes_history": G_modes_history,
        "G0_modes": G0_modes,
        "losses": out["losses"].tolist(),
        "config": asdict(cfg),
    }


def run_sweep(cfg: FashionConfig, seeds: list[int], out_path: Path, device: str = "cpu") -> list[dict]:
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
