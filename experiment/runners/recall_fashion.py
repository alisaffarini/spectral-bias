"""Retrieval defense on a second dataset (Fashion-MNIST).

Mirrors `recall_eval.py` but on Fashion-MNIST features. Used to verify
the asymmetric (hard / easy) Recall@K pattern is not specific to
CIFAR-10. Inputs are PCA-reduced raw pixels (d_in=64), as in
Appendix C; the depth-2 head is unchanged.
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
from runners.preconditioner import train_preconditioned
from runners.recall_eval import recall_at_k, _make_triplets


@dataclass
class FashionRecallConfig:
    N_train: int = 400
    N_test: int = 200
    classes: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    feature_dim: int = 64
    depth: int = 2
    width: int = 1024
    n_triplets: int = 2000
    n_epochs: int = 1500
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 100
    eps: float = 1e-2
    Ks: tuple[int, ...] = (1, 5, 10, 20)
    data_root: str = "./data"
    triplet_difficulty: str = "hard"  # 'easy' or 'hard'


def _load_fashion_subset(cfg: FashionRecallConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.FashionMNIST(
        root=cfg.data_root, train=True, download=True, transform=tfm,
    )
    targets = np.array(ds.targets)
    per_class = (cfg.N_train + cfg.N_test) // len(cfg.classes)
    idxs, labels = [], []
    for c in cfg.classes:
        candidate = np.where(targets == c)[0]
        chosen = rng.choice(candidate, size=per_class, replace=False)
        idxs.extend(chosen.tolist())
        labels.extend([c] * per_class)
    idxs = np.array(idxs); labels = np.array(labels)

    images = np.stack([np.asarray(ds[i][0]).reshape(-1) for i in idxs], axis=0)
    images = images.astype(np.float32) / images.max()
    images_centered = images - images.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(images_centered, full_matrices=False)
    Z = U[:, : cfg.feature_dim] * S[: cfg.feature_dim]
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z, labels


def _split_train_test(Z, labels, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    train_idx, test_idx = [], []
    per_class_train = n_train // len(classes)
    per_class_test = n_test // len(classes)
    for c in classes:
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        train_idx.extend(idxs[:per_class_train].tolist())
        test_idx.extend(idxs[per_class_train:per_class_train + per_class_test].tolist())
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)
    return Z[train_idx], labels[train_idx], Z[test_idx], labels[test_idx]


def run_one(seed: int, cfg: FashionRecallConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    Z_all, lbl_all = _load_fashion_subset(cfg, seed)
    Z_train, lbl_train, Z_test, lbl_test = _split_train_test(
        Z_all, lbl_all, cfg.N_train, cfg.N_test, seed,
    )
    triplets = _make_triplets(lbl_train, cfg.n_triplets, seed,
                               difficulty=cfg.triplet_difficulty, Z=Z_train)

    K = depth_L_ntk_kernel(Z_train, cfg.depth)

    def train_and_eval(method: str):
        torch.manual_seed(seed); np.random.seed(seed)
        model = MetricNetwork(cfg.feature_dim, [cfg.width] * cfg.depth)
        if method == "vanilla":
            train_network(
                model, Z_train, triplets,
                TrainConfig(n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
                            record_every=cfg.record_every),
                device=device,
            )
        else:
            train_preconditioned(
                model, Z_train, triplets, K,
                n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
                record_every=cfg.record_every, eps=cfg.eps, device=device,
            )
        model.to("cpu")
        F_train = embeddings_from_model(model, Z_train, device="cpu")
        F_test = embeddings_from_model(model, Z_test, device="cpu")
        return {f"recall@{k}": recall_at_k(F_test, lbl_test, F_train, lbl_train, k)
                for k in cfg.Ks}

    return {
        "seed": seed,
        "vanilla": train_and_eval("vanilla"),
        "precond": train_and_eval("precond"),
        "Ks": list(cfg.Ks),
        "config": asdict(cfg),
    }


def run_sweep(cfg: FashionRecallConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
