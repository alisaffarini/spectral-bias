"""Long-tailed CIFAR-10 retrieval — practical impact angle.

Class frequencies follow an exponential decay so the head class has
N_max samples and the tail class has N_min samples. Tail classes
naturally occupy low-eigenvalue modes of the kernel (they have less
mass in the empirical NTK), so K^{-1}-corrected training should help
tail-class retrieval while leaving head-class retrieval largely
unchanged.

We report Recall@K stratified by class frequency band: head (top 3
classes), middle (next 4), tail (bottom 3).
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
class LongTailConfig:
    n_classes: int = 10
    head_count: int = 80    # samples per head class
    tail_count: int = 8     # samples per tail class
    n_test_per_class: int = 30
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


def _resnet_features(images: torch.Tensor, device: str) -> torch.Tensor:
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    with torch.no_grad():
        feats = model(images.to(device))
    return feats.cpu()


def _exponential_class_counts(n_classes: int, head: int, tail: int) -> list[int]:
    """Exponential decay from `head` to `tail` over n_classes."""
    if head == tail:
        return [head] * n_classes
    rates = np.exp(np.linspace(np.log(head), np.log(tail), n_classes))
    return [int(round(r)) for r in rates]


def _load_longtail_subset(cfg: LongTailConfig, seed: int, device: str):
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
    counts = _exponential_class_counts(cfg.n_classes, cfg.head_count, cfg.tail_count)
    train_idx, train_lbl, test_idx, test_lbl = [], [], [], []
    for c in range(cfg.n_classes):
        candidates = np.where(targets == c)[0]
        rng.shuffle(candidates)
        train_idx.extend(candidates[:counts[c]].tolist())
        train_lbl.extend([c] * counts[c])
        test_idx.extend(
            candidates[counts[c]:counts[c] + cfg.n_test_per_class].tolist()
        )
        test_lbl.extend([c] * cfg.n_test_per_class)
    all_idx = train_idx + test_idx
    images = torch.stack([ds[i][0] for i in all_idx])

    feats = []
    bsz = 64
    for i in range(0, len(images), bsz):
        feats.append(_resnet_features(images[i:i + bsz], device))
    feats = torch.cat(feats, dim=0).numpy()
    feats_centered = feats - feats.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(feats_centered, full_matrices=False)
    Z = U[:, :cfg.feature_dim] * S[:cfg.feature_dim]
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    n_train = len(train_idx)
    return (Z[:n_train], np.array(train_lbl),
            Z[n_train:], np.array(test_lbl), counts)


def _recall_per_class(test_emb, test_labels, train_emb, train_labels, k):
    d = (np.sum(test_emb ** 2, axis=1, keepdims=True)
         + np.sum(train_emb ** 2, axis=1)[None, :]
         - 2.0 * test_emb @ train_emb.T)
    nn_idx = np.argsort(d, axis=1)[:, :k]
    nn_labels = train_labels[nn_idx]
    hits = (nn_labels == test_labels[:, None]).any(axis=1)
    per_class = {}
    for c in np.unique(test_labels):
        mask = test_labels == c
        per_class[int(c)] = float(hits[mask].mean())
    return per_class


def run_one(seed: int, cfg: LongTailConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    Z_train, lbl_train, Z_test, lbl_test, counts = _load_longtail_subset(
        cfg, seed, device,
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
        per_class_at_k = {}
        for k in cfg.Ks:
            per_class_at_k[k] = _recall_per_class(F_test, lbl_test, F_train,
                                                   lbl_train, k)
        overall = {f"recall@{k}": recall_at_k(F_test, lbl_test, F_train,
                                               lbl_train, k) for k in cfg.Ks}
        return {"per_class": per_class_at_k, "overall": overall}

    return {
        "seed": seed,
        "class_counts": counts,
        "vanilla": train_and_eval("vanilla"),
        "precond": train_and_eval("precond"),
        "Ks": list(cfg.Ks),
        "config": asdict(cfg),
    }


def run_sweep(cfg: LongTailConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
