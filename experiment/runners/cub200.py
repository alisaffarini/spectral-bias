"""CUB-200-2011 metric-learning retrieval — the canonical fine-grained
metric-learning benchmark.

Downloads the dataset on first run from the Caltech mirror, extracts
ResNet-50 penultimate features, splits per-class into train + test,
trains both vanilla and K^{-1}-corrected heads with triplet loss, and
reports Recall@K.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import tarfile
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from kbm.kernels import depth_L_ntk_kernel
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model
from runners.preconditioner import train_preconditioned
from runners.recall_eval import recall_at_k, _make_triplets


CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


@dataclass
class CUBConfig:
    N_train: int = 600     # 6 per class × 100 train classes
    N_test: int = 300      # 3 per class × 100 test classes (different classes)
    n_classes_train: int = 100
    n_classes_test: int = 100
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


def _download_cub(data_root: Path):
    target = data_root / "CUB_200_2011"
    if target.exists() and (target / "images.txt").exists():
        return target
    data_root.mkdir(parents=True, exist_ok=True)
    tgz_path = data_root / "CUB_200_2011.tgz"
    if not tgz_path.exists():
        print(f"  downloading CUB-200 from {CUB_URL} (1.2 GB)...", flush=True)
        urllib.request.urlretrieve(CUB_URL, tgz_path)
    print(f"  extracting {tgz_path}...", flush=True)
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(data_root)
    return target


def _load_image_paths(cub_dir: Path) -> tuple[list[Path], np.ndarray]:
    """Return list of image paths and class labels (1..200)."""
    images_txt = cub_dir / "images.txt"
    labels_txt = cub_dir / "image_class_labels.txt"
    img_root = cub_dir / "images"
    paths = {}
    with open(images_txt) as f:
        for line in f:
            idx, rel = line.strip().split(" ", 1)
            paths[int(idx)] = img_root / rel
    labels = {}
    with open(labels_txt) as f:
        for line in f:
            idx, lab = line.strip().split(" ", 1)
            labels[int(idx)] = int(lab)
    ordered_idx = sorted(paths.keys())
    path_list = [paths[i] for i in ordered_idx]
    lab_arr = np.array([labels[i] for i in ordered_idx])
    return path_list, lab_arr


def _resnet50_features(images: torch.Tensor, device: str) -> torch.Tensor:
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    with torch.no_grad():
        feats = model(images.to(device))
    return feats.cpu()


def _load_subset(cfg: CUBConfig, seed: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Sample `n_classes_train + n_classes_test` distinct classes from CUB-200,
    extract ResNet-50 features, PCA-reduce to feature_dim, and unit-normalise."""
    rng = np.random.default_rng(seed)
    cub_dir = _download_cub(Path(cfg.data_root))
    paths, labels = _load_image_paths(cub_dir)

    classes_all = np.unique(labels)  # 1..200
    n_classes = cfg.n_classes_train + cfg.n_classes_test
    chosen_classes = rng.choice(classes_all, size=n_classes, replace=False)

    per_class_train = cfg.N_train // cfg.n_classes_train
    per_class_test = cfg.N_test // cfg.n_classes_test
    per_class = per_class_train + per_class_test

    chosen_idx = []
    chosen_labels = []
    for c in chosen_classes:
        c_idxs = np.where(labels == c)[0]
        if len(c_idxs) >= per_class:
            sel = rng.choice(c_idxs, size=per_class, replace=False)
        else:
            sel = rng.choice(c_idxs, size=per_class, replace=True)
        chosen_idx.extend(sel.tolist())
        chosen_labels.extend([int(c)] * per_class)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for idx in chosen_idx:
        img = Image.open(paths[idx]).convert("RGB")
        images.append(tfm(img))
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
    return Z, np.array(chosen_labels)


def _split_train_test(Z, labels, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    train_idx, test_idx = [], []
    per_class_train = n_train // len(classes) * 2  # we sampled per_class items; split 2:1 ish
    # Re-derive: half the items per class go to train, half to test (using config ratios).
    # Use the cfg implicit ratios below.
    train_idx, test_idx = [], []
    for c in classes:
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        # split: 2/3 to train, 1/3 to test for this protocol
        n = len(idxs)
        split = max(1, n * 2 // 3)
        train_idx.extend(idxs[:split].tolist())
        test_idx.extend(idxs[split:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    return Z[train_idx], labels[train_idx], Z[test_idx], labels[test_idx]


def run_one(seed: int, cfg: CUBConfig, device: str = "cuda") -> dict:
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
        "vanilla": train_and_eval("vanilla"),
        "precond": train_and_eval("precond"),
        "Ks": list(cfg.Ks),
        "config": asdict(cfg),
    }


def run_sweep(cfg: CUBConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
