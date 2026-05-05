"""Generic fine-grained metric-learning retrieval runner.

Supports several torchvision-native fine-grained datasets that are
standard in the metric-learning literature: Oxford-IIIT Pet (37
classes), FGVC Aircraft (100 classes), Flowers-102 (102 classes),
StanfordCars (196 classes). The pipeline is identical for each ---
ResNet-50 features + 2-layer ReLU head + triplet loss --- so the
asymmetric prediction (no easy R@K gain, hard R@K gain) can be tested
across multiple non-CIFAR datasets with one runner.
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
from runners.recall_eval import recall_at_k, _make_triplets


@dataclass
class FineGrainedConfig:
    dataset: str = "pet"   # 'pet' | 'aircraft' | 'flowers' | 'cars'
    N_train: int = 600
    N_test: int = 300
    feature_dim: int = 128
    depth: int = 2
    width: int = 1024
    n_triplets: int = 3000
    n_epochs: int = 1500
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 100
    eps: float = 1e-2
    Ks: tuple[int, ...] = (1, 5, 10, 20)
    data_root: str = "./data"
    triplet_difficulty: str = "hard"


def _load_dataset(name: str, data_root: str):
    """Return (PIL_dataset, label_array)."""
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if name == "pet":
        ds = torchvision.datasets.OxfordIIITPet(
            data_root, split="trainval", download=True, transform=tfm,
        )
        labels = np.array(ds._labels)
    elif name == "aircraft":
        ds = torchvision.datasets.FGVCAircraft(
            data_root, split="trainval", download=True, transform=tfm,
        )
        labels = np.array(ds._labels)
    elif name == "flowers":
        ds = torchvision.datasets.Flowers102(
            data_root, split="train", download=True, transform=tfm,
        )
        labels = np.array(ds._labels)
    elif name == "cars":
        ds = torchvision.datasets.StanfordCars(
            data_root, split="train", download=True, transform=tfm,
        )
        labels = np.array([s[1] for s in ds._samples])
    else:
        raise ValueError(name)
    return ds, labels


def _resnet50_features(images: torch.Tensor, device: str) -> torch.Tensor:
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    with torch.no_grad():
        feats = model(images.to(device))
    return feats.cpu()


def _load_subset(cfg: FineGrainedConfig, seed: int, device: str):
    rng = np.random.default_rng(seed)
    ds, labels = _load_dataset(cfg.dataset, cfg.data_root)
    classes = np.unique(labels)
    n_classes = len(classes)
    per_class_total = (cfg.N_train + cfg.N_test) // n_classes
    if per_class_total < 2:
        # Many classes, few samples each — use 2 per class minimum
        per_class_total = 2

    chosen_idx, chosen_lbl = [], []
    for c in classes:
        c_idxs = np.where(labels == c)[0]
        if len(c_idxs) >= per_class_total:
            sel = rng.choice(c_idxs, size=per_class_total, replace=False)
        else:
            sel = rng.choice(c_idxs, size=per_class_total, replace=True)
        chosen_idx.extend(sel.tolist())
        chosen_lbl.extend([int(c)] * per_class_total)

    images = []
    for idx in chosen_idx:
        images.append(ds[idx][0])
    images = torch.stack(images, dim=0)
    feats = []
    bsz = 64
    for i in range(0, len(images), bsz):
        feats.append(_resnet50_features(images[i:i + bsz], device))
    feats = torch.cat(feats, dim=0).numpy()
    feats_centered = feats - feats.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(feats_centered, full_matrices=False)
    Z = U[:, :cfg.feature_dim] * S[:cfg.feature_dim]
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z, np.array(chosen_lbl)


def _split_train_test(Z, labels, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    train_idx, test_idx = [], []
    for c in classes:
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        n = len(idxs)
        split = max(1, n * 2 // 3)
        train_idx.extend(idxs[:split].tolist())
        test_idx.extend(idxs[split:].tolist())
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)
    return Z[train_idx], labels[train_idx], Z[test_idx], labels[test_idx]


def run_one(seed: int, cfg: FineGrainedConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    Z_all, lbl_all = _load_subset(cfg, seed, device)
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
        "dataset": cfg.dataset,
        "vanilla": train_and_eval("vanilla"),
        "precond": train_and_eval("precond"),
        "Ks": list(cfg.Ks),
        "config": asdict(cfg),
    }


def run_sweep(cfg: FineGrainedConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
