"""Adversarial-robust retrieval (Tier 2): does K^{-1} training
produce models that are more robust to adversarial perturbations
of the test set?

Theory motivation: adversarial perturbations target small-margin
directions in feature space, which empirically align with bottom
eigenmodes of K. K^{-1}-trained models have moved further along
those bottom eigenmodes during training, so the embedding-space
"robustness margin" should be larger.

Pipeline: train both vanilla and K^{-1}-corrected on CIFAR-10
(small N, ResNet-18 features); on the test set, apply PGD
perturbation in PIXEL space (then re-extract features) and measure
clean vs adversarial Recall@K.
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
class AdvRecallConfig:
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
    triplet_difficulty: str = "hard"
    pgd_eps: float = 8.0 / 255.0
    pgd_alpha: float = 2.0 / 255.0
    pgd_steps: int = 10
    dataset: str = "cifar10"


def _resnet18(device):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Identity()
    return m.eval().to(device)


def _features(images, model, device):
    with torch.no_grad():
        return model(images.to(device)).cpu()


def _load_cifar10_imgs(cfg: AdvRecallConfig, seed: int):
    """Fast loader: pull selected indices from the underlying numpy array
    (avoids the PIL per-image path), then resize as a single batched torch op.
    Supports both CIFAR-10 and Fashion-MNIST via cfg.dataset."""
    rng = np.random.default_rng(seed)
    if cfg.dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10(cfg.data_root, train=True, download=True)
        targets = np.array(ds.targets)
        per_class = (cfg.N_train + cfg.N_test) // cfg.n_classes
        idxs, labels = [], []
        for c in range(cfg.n_classes):
            cand = np.where(targets == c)[0]
            chosen = rng.choice(cand, size=per_class, replace=False)
            idxs.extend(chosen.tolist()); labels.extend([c] * per_class)
        arr = ds.data[np.asarray(idxs)]                                   # (B,32,32,3) uint8
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0      # (B,3,32,32)
    elif cfg.dataset == "fmnist":
        ds = torchvision.datasets.FashionMNIST(cfg.data_root, train=True, download=True)
        targets = np.array(ds.targets)
        per_class = (cfg.N_train + cfg.N_test) // cfg.n_classes
        idxs, labels = [], []
        for c in range(cfg.n_classes):
            cand = np.where(targets == c)[0]
            chosen = rng.choice(cand, size=per_class, replace=False)
            idxs.extend(chosen.tolist()); labels.extend([c] * per_class)
        arr = ds.data[np.asarray(idxs)].numpy()                            # (B,28,28) uint8
        t = torch.from_numpy(arr).unsqueeze(1).float() / 255.0             # (B,1,28,28)
        t = t.repeat(1, 3, 1, 1)                                           # (B,3,28,28)
    else:
        raise ValueError(f"unknown dataset {cfg.dataset}")
    images = torch.nn.functional.interpolate(t, size=224, mode="bilinear", align_corners=False)
    return images, np.array(labels)


def _normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class HeadOnly(nn.Module):
    """Wraps frozen ResNet-18 + a trainable metric head; PGD attacks
    the input pixel-space embedding."""
    def __init__(self, backbone, metric_head, pca_basis, pca_mean):
        super().__init__()
        self.backbone = backbone
        self.head = metric_head
        self.register_buffer("pca", torch.as_tensor(pca_basis, dtype=torch.float32))
        self.register_buffer("pca_mean", torch.as_tensor(pca_mean, dtype=torch.float32))

    def forward(self, x):
        feats = self.backbone(_normalize(x))
        f = feats - self.pca_mean
        z = f @ self.pca
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return self.head(z)


def _pgd_attack(images, labels, full_model, cfg, device):
    """Pixel-space PGD attack: maximize cross-class L2 distance to anchor.

    For each test image, push it away from its own class's centroid in
    embedding space. Standard untargeted attack on the embedding
    similarity to true class.
    """
    full_model.eval().to(device)
    images = images.to(device)
    # Compute clean centroids per class
    with torch.no_grad():
        Z = full_model(images)
    classes = np.unique(labels)
    centroids = torch.stack([Z[labels == c].mean(0) for c in classes]).to(device)
    label_to_idx = {int(c): i for i, c in enumerate(classes)}
    label_idx = torch.tensor([label_to_idx[int(l)] for l in labels], device=device)

    delta = torch.zeros_like(images, requires_grad=True)
    for step in range(cfg.pgd_steps):
        x_adv = (images + delta).clamp(0, 1)
        z = full_model(x_adv)
        z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        c_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
        sim = z_norm @ c_norm.T
        true_sim = sim[torch.arange(len(images), device=device), label_idx]
        loss = -true_sim.sum()  # maximize distance from own class
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta + cfg.pgd_alpha * grad.sign()).detach()
        delta = delta.clamp(-cfg.pgd_eps, cfg.pgd_eps)
        delta.requires_grad_()
    return (images + delta.detach()).clamp(0, 1).cpu()


def run_one(seed: int, cfg: AdvRecallConfig, device: str = "cuda") -> dict:
    import time as _t
    t0 = _t.time()
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"  [t={_t.time()-t0:.1f}s] loading CIFAR-10 images...", flush=True)
    images, labels = _load_cifar10_imgs(cfg, seed)
    print(f"  [t={_t.time()-t0:.1f}s] images loaded shape={tuple(images.shape)}", flush=True)
    # Stratified per-class split: every class appears in both train and test.
    per_class_train = cfg.N_train // cfg.n_classes
    per_class_test = cfg.N_test // cfg.n_classes
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in range(cfg.n_classes):
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        train_idx.extend(idxs[:per_class_train].tolist())
        test_idx.extend(idxs[per_class_train:per_class_train + per_class_test].tolist())
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)
    train_imgs, train_lbl = images[train_idx], labels[train_idx]
    test_imgs, test_lbl = images[test_idx], labels[test_idx]

    print(f"  [t={_t.time()-t0:.1f}s] loading ResNet-18 backbone on {device}...", flush=True)
    backbone = _resnet18(device)
    print(f"  [t={_t.time()-t0:.1f}s] extracting train features...", flush=True)
    train_feats = []
    bsz = 64
    for i in range(0, len(train_imgs), bsz):
        train_feats.append(backbone(_normalize(train_imgs[i:i + bsz].to(device))).detach().cpu())
    train_feats = torch.cat(train_feats, dim=0).numpy()
    print(f"  [t={_t.time()-t0:.1f}s] features {train_feats.shape}; PCA + NTK...", flush=True)
    feats_centered = train_feats - train_feats.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(feats_centered, full_matrices=False)
    pca_basis = Vt[: cfg.feature_dim].T  # (512, feature_dim)
    pca_mean = train_feats.mean(axis=0, keepdims=True)
    Z_train = (train_feats - pca_mean) @ pca_basis
    Z_train = Z_train / np.linalg.norm(Z_train, axis=1, keepdims=True)

    triplets = _make_triplets(train_lbl, cfg.n_triplets, seed,
                               difficulty=cfg.triplet_difficulty, Z=Z_train)
    K = depth_L_ntk_kernel(Z_train, cfg.depth)
    print(f"  [t={_t.time()-t0:.1f}s] NTK done, training heads...", flush=True)

    def make_head_and_train(method):
        torch.manual_seed(seed); np.random.seed(seed)
        head = MetricNetwork(cfg.feature_dim, [cfg.width] * cfg.depth)
        if method == "vanilla":
            train_network(head, Z_train, triplets,
                           TrainConfig(n_epochs=cfg.n_epochs, lr=cfg.lr,
                                       margin=cfg.margin, record_every=cfg.record_every),
                           device=device)
        else:
            train_preconditioned(head, Z_train, triplets, K,
                                  n_epochs=cfg.n_epochs, lr=cfg.lr, margin=cfg.margin,
                                  record_every=cfg.record_every, eps=cfg.eps, device=device)
        head.to(device)
        full = HeadOnly(backbone, head, pca_basis, pca_mean.flatten())
        full.to(device)
        return full

    out = {"seed": seed, "Ks": list(cfg.Ks), "config": asdict(cfg)}
    for method in ("vanilla", "precond"):
        full = make_head_and_train(method)
        # Clean recall
        with torch.no_grad():
            Z_te_clean = full(test_imgs.to(device)).cpu().numpy()
            Z_tr = full(train_imgs.to(device)).cpu().numpy()
        clean_recall = {f"recall@{k}": recall_at_k(Z_te_clean, test_lbl, Z_tr, train_lbl, k)
                        for k in cfg.Ks}

        # Adversarial recall
        adv_test = _pgd_attack(test_imgs, test_lbl, full, cfg, device)
        with torch.no_grad():
            Z_te_adv = full(adv_test.to(device)).cpu().numpy()
        adv_recall = {f"recall@{k}": recall_at_k(Z_te_adv, test_lbl, Z_tr, train_lbl, k)
                      for k in cfg.Ks}
        out[method] = {"clean": clean_recall, "adv": adv_recall}
        print(f"  [{method}] clean R@10={clean_recall['recall@10']:.3f}, "
              f"adv R@10={adv_recall['recall@10']:.3f}", flush=True)
    return out


def run_sweep(cfg: AdvRecallConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {r["seed"] for r in results}
        seeds = [s for s in seeds if s not in done]
    else:
        results = []
    for s in seeds:
        print(f"[adv-recall seed={s}]", flush=True)
        results.append(run_one(s, cfg, device=device))
        with open(out_path, "w") as f:
            json.dump(results, f)
    return results
