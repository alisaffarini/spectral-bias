"""Multi-layer ReLU networks under NTK parameterization, and trainers."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MetricNetwork(nn.Module):
    """Depth-L fully-connected ReLU network with NTK parameterization.

    NTK parameterization: weights initialized as N(0, 1/d_in) per fan-in, no
    explicit normalization at output. Output is the embedding F (N x M_L).
    """

    def __init__(self, d_in: int, widths: list[int]):
        super().__init__()
        if len(widths) < 1:
            raise ValueError("must have at least one hidden layer")
        self.widths = widths
        self.depth = len(widths)
        self.layers = nn.ModuleList()
        prev = d_in
        for w in widths:
            lin = nn.Linear(prev, w, bias=False)
            with torch.no_grad():
                lin.weight.normal_(0.0, 1.0 / np.sqrt(prev))
            self.layers.append(lin)
            prev = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.depth - 1:
                h = torch.relu(h)
        return torch.relu(h)


def embeddings_from_model(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        F = model(Xt)
    return F.cpu().numpy()


@dataclass
class TrainConfig:
    n_epochs: int = 5000
    lr: float = 1e-4
    margin: float = 1.0
    record_every: int = 100
    weight_decay: float = 0.0
    optimizer: str = "sgd"  # "sgd" or "adam"
    nuclear_reg: float = 0.0


def train_network(
    model: nn.Module,
    X: np.ndarray,
    triplets,
    config: TrainConfig,
    device: str = "cpu",
):
    """Train and record G(t) at intervals. Returns dict of recorded trajectories."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    model.to(device)

    if config.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"unknown optimizer {config.optimizer}")

    epochs_recorded = []
    G_history = []
    losses = []

    triplet_arr = np.array(triplets, dtype=np.int64)
    a_idx = torch.as_tensor(triplet_arr[:, 0], device=device)
    p_idx = torch.as_tensor(triplet_arr[:, 1], device=device)
    n_idx = torch.as_tensor(triplet_arr[:, 2], device=device)

    margin_t = torch.tensor(float(config.margin), device=device)

    for epoch in range(config.n_epochs):
        model.train()
        opt.zero_grad()
        F = model(Xt)
        G = F @ F.T
        diag = torch.diagonal(G)
        d_ap = diag[a_idx] + diag[p_idx] - 2.0 * G[a_idx, p_idx]
        d_an = diag[a_idx] + diag[n_idx] - 2.0 * G[a_idx, n_idx]
        per_t = torch.clamp(d_ap - d_an + margin_t, min=0.0)
        loss = per_t.sum()
        if config.nuclear_reg > 0.0:
            G_svals = torch.linalg.svdvals(G)
            loss = loss + config.nuclear_reg * G_svals.sum()
        loss.backward()
        opt.step()

        if epoch % config.record_every == 0 or epoch == config.n_epochs - 1:
            with torch.no_grad():
                F_ = model(Xt)
                G_ = (F_ @ F_.T).cpu().numpy()
            epochs_recorded.append(epoch)
            G_history.append(G_)
            losses.append(float(loss.item()))

    return {
        "epochs": np.array(epochs_recorded),
        "G_history": G_history,
        "losses": np.array(losses),
    }
