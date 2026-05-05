"""Scaled retrieval — N_train high enough that bottom-eigenspace gradient
signal is structured rather than noise-dominated. Mirrors recall_eval.py
but at 10× the previous scale; designed for A100.
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
class ScaledRecallConfig:
    N_train: int = 4000     # 400 per class × 10 classes
    N_test: int = 1000      # 100 per class × 10 classes
    n_classes: int = 10
    feature_dim: int = 256
    depth: int = 2
    width: int = 1024
    n_triplets: int = 12000
    n_epochs: int = 2000
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 200
    eps: float = 1e-2
    Ks: tuple[int, ...] = (1, 5, 10, 20)
    data_root: str = "./data"
    triplet_difficulty: str = "hard"
    backbone: str = "resnet50"


def _make_backbone(name: str, device: str) -> nn.Module:
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
    else:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Identity()
    return model.eval().to(device)


def _load_subset(cfg: ScaledRecallConfig, seed: int, device: str):
    rng = np.random.default_rng(seed)
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=tfm,
    )
    targets = np.array(ds.targets)
    per_class = (cfg.N_train + cfg.N_test) // cfg.n_classes
    idxs, labels = [], []
    for c in range(cfg.n_classes):
        candidates = np.where(targets == c)[0]
        chosen = rng.choice(candidates, size=per_class, replace=False)
        idxs.extend(chosen.tolist())
        labels.extend([c] * per_class)
    print(f"  loading {len(idxs)} CIFAR-10 images at 224x224...", flush=True)
    images = torch.stack([ds[i][0] for i in idxs])
    labels = np.array(labels)
    print(f"  extracting {cfg.backbone} features (bs=128) on {device}...", flush=True)
    backbone = _make_backbone(cfg.backbone, device)
    feats_list = []
    bsz = 128
    with torch.no_grad():
        for i in range(0, len(images), bsz):
            batch = images[i:i + bsz].to(device, non_blocking=True)
            feats_list.append(backbone(batch).cpu())
    del backbone
    torch.cuda.empty_cache()
    feats = torch.cat(feats_list, dim=0).numpy()
    print(f"  features shape: {feats.shape}; computing PCA basis...", flush=True)
    feats_centered = feats - feats.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(feats_centered, full_matrices=False)
    Z = U[:, :cfg.feature_dim] * S[:cfg.feature_dim]
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
    return Z[train_idx], labels[train_idx], Z[test_idx], labels[test_idx]


def run_one(seed: int, cfg: ScaledRecallConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    Z_all, lbl_all = _load_subset(cfg, seed, device)
    Z_train, lbl_train, Z_test, lbl_test = _split_train_test(
        Z_all, lbl_all, cfg.N_train, cfg.N_test, seed,
    )
    triplets = _make_triplets(lbl_train, cfg.n_triplets, seed,
                               difficulty=cfg.triplet_difficulty, Z=Z_train)
    print(f"  computing depth-{cfg.depth} NTK on N={cfg.N_train}...", flush=True)
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


def run_sweep(cfg: ScaledRecallConfig, seeds: list[int], out_path: Path,
              device: str = "cuda"):
    """Stream-friendly: write incrementally so partial results survive crashes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {r["seed"] for r in results}
        seeds = [s for s in seeds if s not in done]
        print(f"  resuming: {len(done)} seeds already done, running {len(seeds)}", flush=True)
    for s in seeds:
        print(f"[scaled-cifar10 {cfg.triplet_difficulty} seed={s}]", flush=True)
        r = run_one(s, cfg, device=device)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  wrote partial -> {out_path} ({len(results)} seeds)", flush=True)
    return results
