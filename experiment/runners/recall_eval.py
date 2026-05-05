"""Recall@K evaluation for vanilla vs preconditioned metric learning.

Reviewer's first question: does the spectral preconditioner actually improve
test-time retrieval performance? This runner produces the answer.

Setup:
  - CIFAR-10 features from frozen ResNet-18, PCA to d_in.
  - Split per-class images into train / test halves.
  - Train both methods on train triplets; evaluate retrieval on test images.
  - Recall@K = fraction of test queries whose nearest K training points
    include at least one same-class neighbor.

The key claim to test: the slope-flattening of the preconditioner translates
into measurable retrieval improvement, particularly when the test geometry
loads on K's bottom eigenspace (which vanilla cannot fit).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch

from kbm.kernels import depth_L_ntk_kernel
from kbm.network import MetricNetwork, train_network, TrainConfig, embeddings_from_model
from runners.preconditioner import train_preconditioned
from runners.cifar10 import _load_subset, CIFARConfig


@dataclass
class RecallConfig:
    N_train: int = 400
    N_test: int = 200
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
    Ks: tuple[int, ...] = (1, 5, 10, 20)
    data_root: str = "./data"
    # 'easy' = random-class negative; 'hard' = nearest-different-class negative
    triplet_difficulty: str = "easy"


def recall_at_k(test_emb: np.ndarray, test_labels: np.ndarray,
                train_emb: np.ndarray, train_labels: np.ndarray, k: int) -> float:
    # squared L2 distance from each test point to each train point
    d = np.sum(test_emb ** 2, axis=1, keepdims=True) + \
        np.sum(train_emb ** 2, axis=1)[None, :] - \
        2.0 * test_emb @ train_emb.T
    nn_idx = np.argsort(d, axis=1)[:, :k]  # (N_test, k) indices into train
    nn_labels = train_labels[nn_idx]       # (N_test, k)
    hits = (nn_labels == test_labels[:, None]).any(axis=1)
    return float(hits.mean())


def _split_train_test(Z: np.ndarray, labels: np.ndarray,
                      n_train: int, n_test: int, seed: int):
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


def _make_triplets(labels: np.ndarray, n_triplets: int, seed: int,
                   difficulty: str = "easy", Z: np.ndarray | None = None,
                   embeddings: np.ndarray | None = None):
    """Form (anchor, positive, negative) triplets.

    difficulty='easy': negative is uniformly random from a different class.
    difficulty='hard': negative is the nearest different-class point in
                       INPUT space (Z), so the loss-relevant gradient
                       loads on bottom eigenspace of K.
    difficulty='semi_hard': negative is the nearest different-class point
                       in EMBEDDING space (uses `embeddings`); this matches
                       the standard FaceNet-style operationalization, where
                       hardness is defined in the model's own representation.
    """
    rng = np.random.default_rng(seed)
    by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    classes = list(by_class.keys())

    if difficulty == "hard":
        if Z is None:
            raise ValueError("hard triplets require Z")
        d2 = np.sum(Z ** 2, axis=1, keepdims=True) + np.sum(Z ** 2, axis=1)[None, :] \
             - 2.0 * Z @ Z.T
        np.fill_diagonal(d2, np.inf)
    elif difficulty == "semi_hard":
        if embeddings is None:
            raise ValueError("semi_hard triplets require embeddings")
        d2 = np.sum(embeddings ** 2, axis=1, keepdims=True) + \
             np.sum(embeddings ** 2, axis=1)[None, :] - \
             2.0 * embeddings @ embeddings.T
        np.fill_diagonal(d2, np.inf)

    triplets = []
    while len(triplets) < n_triplets:
        c_pos = rng.choice(classes)
        a, p = rng.choice(by_class[c_pos], 2, replace=False)
        if difficulty == "easy":
            c_neg = rng.choice([c for c in classes if c != c_pos])
            n = rng.choice(by_class[c_neg])
        else:  # hard or semi_hard
            other = np.concatenate([by_class[c] for c in classes if c != c_pos])
            n = int(other[np.argmin(d2[a, other])])
        triplets.append((int(a), int(p), int(n)))
    return triplets


def run_one(seed: int, cfg: RecallConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    # Load enough samples for train + test, then split.
    cifar_cfg = CIFARConfig(N=cfg.N_train + cfg.N_test, n_classes=cfg.n_classes,
                            feature_dim=cfg.feature_dim, data_root=cfg.data_root)
    Z_all, labels_all = _load_subset(cifar_cfg, seed, device)
    Z_train, lbl_train, Z_test, lbl_test = _split_train_test(
        Z_all, labels_all, cfg.N_train, cfg.N_test, seed,
    )

    # For semi-hard mining, briefly warm up a vanilla model on easy
    # triplets to obtain embeddings, then mine negatives in embedding space.
    semi_hard_emb = None
    if cfg.triplet_difficulty == "semi_hard":
        easy_triplets = _make_triplets(lbl_train, cfg.n_triplets, seed,
                                        difficulty="easy", Z=Z_train)
        torch.manual_seed(seed); np.random.seed(seed)
        warm = MetricNetwork(cfg.feature_dim, [cfg.width] * cfg.depth)
        train_network(
            warm, Z_train, easy_triplets,
            TrainConfig(n_epochs=max(50, cfg.n_epochs // 10),
                        lr=cfg.lr, margin=cfg.margin,
                        record_every=10 ** 9),
            device=device,
        )
        warm.to("cpu")
        semi_hard_emb = embeddings_from_model(warm, Z_train, device="cpu")

    triplets = _make_triplets(lbl_train, cfg.n_triplets, seed,
                               difficulty=cfg.triplet_difficulty,
                               Z=Z_train, embeddings=semi_hard_emb)

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
        model.to("cpu")  # move back so embeddings_from_model("cpu") works
        F_train = embeddings_from_model(model, Z_train, device="cpu")
        F_test = embeddings_from_model(model, Z_test, device="cpu")
        recalls = {f"recall@{k}": recall_at_k(F_test, lbl_test, F_train, lbl_train, k)
                   for k in cfg.Ks}
        return recalls

    vanilla = train_and_eval("vanilla")
    precond = train_and_eval("precond")
    return {
        "seed": seed,
        "vanilla": vanilla,
        "precond": precond,
        "Ks": list(cfg.Ks),
        "config": asdict(cfg),
    }


def run_sweep(cfg: RecallConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
