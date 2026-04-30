"""End-to-end CIFAR-10 metric learning with a small CNN trained from scratch.

This addresses the reviewer concern that the frozen-features setup
(runners/cifar10.py) is too easy. Here we train a small CNN end-to-end on
CIFAR-10 with triplet loss and verify:

  (i)  The kBM equivalence and spectral-bias prediction hold for the CNN.
  (ii) The K^{-1}-preconditioner flattens the bias end-to-end.

Since the CNN-NTK has no convenient closed form for our small-batch, small-N
setup, we compute the *empirical* NTK at initialization via the Jacobian:

  K_emp[i, j] = (1/M) sum_m <d F_m(x_i)/d theta, d F_m(x_j)/d theta>

The kBM theory predicts dG/dt = -2(K_emp E G + G E K_emp), and the
spectral-bias prediction (Theorem 6) refers to K_emp's spectrum. Empirical
NTK is the right object for finite-width networks anyway.
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

from kbm.network import train_network, TrainConfig, embeddings_from_model
from runners.preconditioner import train_preconditioned


class SmallCNN(nn.Module):
    """3-conv-block ReLU CNN with a linear projection head for triplet loss.

    Compact enough that the empirical NTK is computable at N <= 200 with
    Jacobian-based methods on an 8GB GPU.
    """

    def __init__(self, M_out: int = 64, channels: tuple[int, int, int] = (32, 64, 64)):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(c1, c2, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(c2, c3, 3, padding=1, bias=False), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B, c3, 1, 1)
            nn.Flatten(),
        )
        self.head = nn.Linear(c3, M_out, bias=False)
        # Keep init scale modest so triplet loss is stable.
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p, nonlinearity="relu")

    def forward(self, x):
        return torch.relu(self.head(self.features(x)))


@dataclass
class End2EndConfig:
    N: int = 100
    n_classes: int = 10
    M_out: int = 32
    n_triplets: int = 600
    n_epochs: int = 1500
    lr: float = 1e-6
    margin: float = 1.0
    record_every: int = 100
    eps: float = 1e-2
    data_root: str = "./data"


def _empirical_ntk(model: nn.Module, X: torch.Tensor, device: str) -> np.ndarray:
    """Compute the embedding-to-embedding empirical NTK at the data.

    K_emp[i, j] = (1/M) sum_m <d F_m(x_i)/d theta, d F_m(x_j)/d theta>.

    For each output feature m, we backprop a one-hot to get dF_m/dtheta
    per data point, then take the Gram. Loop over m to keep memory bounded.
    """
    model.eval().to(device)
    X = X.to(device)
    N = X.shape[0]
    F = model(X)
    M = F.shape[1]
    K = torch.zeros(N, N, device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    for m_idx in range(M):
        # Gradient of sum_i F[i, m_idx] wrt parameters: gives a matrix of
        # per-sample-per-param gradients. Use jacrev-style trick via repeated
        # backward.
        # We'd like J[i] = dF[i, m_idx]/dtheta as a flat vector per i.
        # Then K_emp[i,j] += <J[i], J[j]> / M.
        # Doing this in a single forward is costly; we do per-sample backward.
        J_rows = []
        for i in range(N):
            model.zero_grad(set_to_none=True)
            F_i = model(X[i:i + 1])
            grads = torch.autograd.grad(F_i[0, m_idx], params, retain_graph=False)
            J_rows.append(torch.cat([g.flatten() for g in grads]))
        J = torch.stack(J_rows, dim=0)  # (N, |theta|)
        K += J @ J.T
    return (K / float(M)).detach().cpu().numpy()


def _load_cifar10_subset(cfg: End2EndConfig, seed: int):
    rng = np.random.default_rng(seed)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = torchvision.datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=tfm)
    targets = np.array(ds.targets)
    per_class = cfg.N // cfg.n_classes
    idxs, labels = [], []
    for c in range(cfg.n_classes):
        cand = np.where(targets == c)[0]
        chosen = rng.choice(cand, size=per_class, replace=False)
        idxs.extend(chosen.tolist()); labels.extend([c] * per_class)
    images = torch.stack([ds[i][0] for i in idxs])  # (N, 3, 32, 32)
    return images, np.array(labels)


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


def _train_cnn(model: nn.Module, X: torch.Tensor, triplets, cfg: End2EndConfig,
               device: str, K: np.ndarray | None = None):
    """Train CNN end-to-end. If K is provided, apply K^{-1} preconditioner."""
    model.to(device)
    X = X.to(device)
    triplet_arr = np.asarray(triplets, dtype=np.int64)
    a_idx = torch.as_tensor(triplet_arr[:, 0], device=device)
    p_idx = torch.as_tensor(triplet_arr[:, 1], device=device)
    n_idx = torch.as_tensor(triplet_arr[:, 2], device=device)
    margin_t = torch.tensor(float(cfg.margin), device=device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    if K is not None:
        K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
        K_inv = torch.linalg.solve(
            K_t + cfg.eps * torch.eye(K_t.shape[0], device=device),
            torch.eye(K_t.shape[0], device=device),
        )
    else:
        K_inv = None

    epochs_recorded, G_history, losses = [], [], []
    for epoch in range(cfg.n_epochs):
        model.train()
        opt.zero_grad()
        F = model(X)
        F.retain_grad()
        G = F @ F.T
        diag = torch.diagonal(G)
        d_ap = diag[a_idx] + diag[p_idx] - 2.0 * G[a_idx, p_idx]
        d_an = diag[a_idx] + diag[n_idx] - 2.0 * G[a_idx, n_idx]
        per_t = torch.clamp(d_ap - d_an + margin_t, min=0.0)
        loss = per_t.sum()

        if K_inv is None:
            loss.backward()
        else:
            grad_F = torch.autograd.grad(loss, F, create_graph=False)[0]
            F2 = model(X)
            F2.backward(K_inv @ grad_F)
        opt.step()

        if epoch % cfg.record_every == 0 or epoch == cfg.n_epochs - 1:
            with torch.no_grad():
                F_ = model(X)
                G_ = (F_ @ F_.T).cpu().numpy()
            epochs_recorded.append(epoch)
            G_history.append(G_)
            losses.append(float(loss.item()))
    return {
        "epochs": np.array(epochs_recorded),
        "G_history": G_history,
        "losses": np.array(losses),
    }


def run_one(seed: int, cfg: End2EndConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    X, labels = _load_cifar10_subset(cfg, seed)
    triplets = _make_triplets(labels, cfg.n_triplets, seed)

    # Empirical NTK at initialization
    torch.manual_seed(seed); np.random.seed(seed)
    model_for_ntk = SmallCNN(M_out=cfg.M_out)
    print(f"[seed {seed}] computing empirical NTK...", flush=True)
    K = _empirical_ntk(model_for_ntk, X, device=device)
    eigvals_K, U = np.linalg.eigh(K)
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]; U = U[:, order]

    # Initial G (same model -> same init)
    F0 = model_for_ntk(X.to(device)).detach().cpu().numpy()
    G0 = F0 @ F0.T
    G0_modes = np.diag(U.T @ G0 @ U)

    # Vanilla
    torch.manual_seed(seed); np.random.seed(seed)
    model_v = SmallCNN(M_out=cfg.M_out)
    out_v = _train_cnn(model_v, X, triplets, cfg, device=device, K=None)

    # Preconditioned
    torch.manual_seed(seed); np.random.seed(seed)
    model_p = SmallCNN(M_out=cfg.M_out)
    out_p = _train_cnn(model_p, X, triplets, cfg, device=device, K=K)

    def proj(hist):
        return np.stack([np.diag(U.T @ G @ U) for G in hist], axis=0)
    delta_v = proj(out_v["G_history"]) - G0_modes[None, :]
    delta_p = proj(out_p["G_history"]) - G0_modes[None, :]

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


def run_sweep(cfg: End2EndConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    results = [run_one(s, cfg, device=device) for s in seeds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    return results
