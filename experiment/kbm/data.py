"""Synthetic data and triplet generation."""
from __future__ import annotations
import numpy as np


def generate_synthetic_data(
    N: int, d_in: int, rank_true: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d_in))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    F_target = rng.standard_normal((N, rank_true))
    G_target = F_target @ F_target.T
    diag_G = np.diag(G_target)
    D_target_sq = diag_G[:, None] + diag_G[None, :] - 2 * G_target
    return X, G_target, D_target_sq


def generate_fixed_triplets(N: int, num_triplets: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    triplets = []
    for _ in range(num_triplets):
        idx = rng.choice(N, 3, replace=False)
        triplets.append((int(idx[0]), int(idx[1]), int(idx[2])))
    return triplets


def triplets_from_target(G_target: np.ndarray, max_triplets: int = 5000):
    """Triplets consistent with the target Gram matrix's distance ordering."""
    N = G_target.shape[0]
    diag = np.diag(G_target)
    triplets = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                d_ij = diag[i] + diag[j] - 2 * G_target[i, j]
                d_ik = diag[i] + diag[k] - 2 * G_target[i, k]
                d_jk = diag[j] + diag[k] - 2 * G_target[j, k]
                # for each pair, the closer of the two becomes positive
                if d_ij < d_ik:
                    triplets.append((i, j, k))
                elif d_ik < d_ij:
                    triplets.append((i, k, j))
                if d_ij < d_jk:
                    triplets.append((j, i, k))
                elif d_jk < d_ij:
                    triplets.append((j, k, i))
    if len(triplets) > max_triplets:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(triplets), max_triplets, replace=False)
        triplets = [triplets[i] for i in idx]
    return triplets
