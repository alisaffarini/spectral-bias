"""Empirical NTK on a small transformer (Tier 3): does the slope-<=2
spectral-bias prediction extend to attention-based architectures?

We use a 2-layer ViT-Tiny on CIFAR-10 patches (no pretraining), compute
the empirical NTK K^emb at initialization via per-output-feature
Jacobian backprop, and apply the same kBM analysis as the CNN
end-to-end run (Section 4.7 of the paper). N is kept small (200) so
that the NTK is tractable; the transformer is large enough that GPU
saturates.
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


class SmallViT(nn.Module):
    """Compact ViT — patches of CIFAR-10 with a 2-layer transformer trunk."""
    def __init__(self, patch: int = 4, dim: int = 128, depth: int = 2,
                 heads: int = 4, mlp_dim: int = 256, M_out: int = 32,
                 image_size: int = 32, channels: int = 3):
        super().__init__()
        self.patch = patch
        n_patches = (image_size // patch) ** 2
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch, bias=False)
        self.pos = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(dim, M_out, bias=False)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.proj(x)            # (B, dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, dim)
        x = x + self.pos
        x = self.tr(x)              # (B, n_patches, dim)
        x = x.mean(dim=1)           # global average pool
        return self.head(x)         # (B, M_out)


@dataclass
class ViTNTKConfig:
    N: int = 200
    n_classes: int = 10
    M_out: int = 16
    patch: int = 4
    dim: int = 128
    depth: int = 2
    heads: int = 4
    mlp_dim: int = 256
    n_triplets: int = 1200
    n_epochs: int = 1000
    lr: float = 1e-6
    margin: float = 1.0
    record_every: int = 100
    eps: float = 1e-2
    data_root: str = "./data"
    n_seeds: int = 3


def _load_cifar10(cfg: ViTNTKConfig, seed: int):
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
        idxs.extend(chosen.tolist())
        labels.extend([c] * per_class)
    images = torch.stack([ds[i][0] for i in idxs])
    return images, np.array(labels)


def _make_triplets(labels, n_triplets, seed):
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


def _empirical_ntk(model: nn.Module, X: torch.Tensor, device: str) -> np.ndarray:
    """Compute the embedding-to-embedding empirical NTK at the data."""
    model.eval().to(device)
    X = X.to(device)
    N = X.shape[0]
    F = model(X)
    M = F.shape[1]
    K = torch.zeros(N, N, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    for m_idx in range(M):
        J_rows = []
        for i in range(N):
            model.zero_grad(set_to_none=True)
            F_i = model(X[i:i + 1])
            grads = torch.autograd.grad(F_i[0, m_idx], params, retain_graph=False)
            J_rows.append(torch.cat([g.flatten() for g in grads]))
        J = torch.stack(J_rows, dim=0)
        K += J @ J.T
    return (K / float(M)).detach().cpu().numpy()


def _train(model, X, triplets, cfg, device, K=None):
    model.to(device)
    X = X.to(device)
    triplet_arr = np.asarray(triplets, dtype=np.int64)
    a = torch.as_tensor(triplet_arr[:, 0], device=device)
    p = torch.as_tensor(triplet_arr[:, 1], device=device)
    n = torch.as_tensor(triplet_arr[:, 2], device=device)
    margin = torch.tensor(float(cfg.margin), device=device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    if K is not None:
        K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
        K_inv = torch.linalg.solve(K_t + cfg.eps * torch.eye(K_t.shape[0], device=device),
                                   torch.eye(K_t.shape[0], device=device))
    else:
        K_inv = None

    epochs_rec, G_hist, losses = [], [], []
    for ep in range(cfg.n_epochs):
        model.train()
        opt.zero_grad()
        F = model(X)
        F.retain_grad()
        G = F @ F.T
        diag = torch.diagonal(G)
        d_ap = diag[a] + diag[p] - 2.0 * G[a, p]
        d_an = diag[a] + diag[n] - 2.0 * G[a, n]
        per_t = torch.clamp(d_ap - d_an + margin, min=0.0)
        loss = per_t.sum()

        if K_inv is None:
            loss.backward()
        else:
            grad_F = torch.autograd.grad(loss, F, create_graph=False)[0]
            F2 = model(X)
            F2.backward(K_inv @ grad_F)
        opt.step()

        if ep % cfg.record_every == 0 or ep == cfg.n_epochs - 1:
            with torch.no_grad():
                F_ = model(X)
                G_ = (F_ @ F_.T).cpu().numpy()
            epochs_rec.append(ep)
            G_hist.append(G_)
            losses.append(float(loss.item()))
    return {"epochs": np.array(epochs_rec), "G_history": G_hist, "losses": np.array(losses)}


def run_one(seed: int, cfg: ViTNTKConfig, device: str = "cuda") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    X, labels = _load_cifar10(cfg, seed)
    triplets = _make_triplets(labels, cfg.n_triplets, seed)

    torch.manual_seed(seed); np.random.seed(seed)
    m_init = SmallViT(patch=cfg.patch, dim=cfg.dim, depth=cfg.depth, heads=cfg.heads,
                       mlp_dim=cfg.mlp_dim, M_out=cfg.M_out)
    print(f"  computing empirical ViT NTK (N={cfg.N}, params={sum(p.numel() for p in m_init.parameters())/1e6:.2f}M)...",
          flush=True)
    K = _empirical_ntk(m_init, X, device=device)
    eigvals_K, U = np.linalg.eigh(K)
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]; U = U[:, order]

    F0 = m_init(X.to(device)).detach().cpu().numpy()
    G0 = F0 @ F0.T
    G0_modes = np.diag(U.T @ G0 @ U)

    torch.manual_seed(seed); np.random.seed(seed)
    m_v = SmallViT(patch=cfg.patch, dim=cfg.dim, depth=cfg.depth, heads=cfg.heads,
                    mlp_dim=cfg.mlp_dim, M_out=cfg.M_out)
    out_v = _train(m_v, X, triplets, cfg, device=device, K=None)

    torch.manual_seed(seed); np.random.seed(seed)
    m_p = SmallViT(patch=cfg.patch, dim=cfg.dim, depth=cfg.depth, heads=cfg.heads,
                    mlp_dim=cfg.mlp_dim, M_out=cfg.M_out)
    out_p = _train(m_p, X, triplets, cfg, device=device, K=K)

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


def run_sweep(cfg: ViTNTKConfig, seeds: list[int], out_path: Path, device: str = "cuda"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {r["seed"] for r in results}
        seeds = [s for s in seeds if s not in done]
    else:
        results = []
    for s in seeds:
        print(f"[vit-ntk seed={s}]", flush=True)
        results.append(run_one(s, cfg, device=device))
        with open(out_path, "w") as f:
            json.dump(results, f)
    return results
