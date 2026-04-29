"""Triplet loss in Gram-matrix coordinates and its gradient E = dL/dG.

Vectorized via numpy.add.at so the ODE step stays cheap at large N and many
triplets. Functionally equivalent to the original Python-loop reference,
which is kept around for unit testing under `triplet_gradient_E_reference`.
"""
from __future__ import annotations
import numpy as np


def triplet_loss_value(G: np.ndarray, triplets, margin: float = 1.0) -> float:
    if not isinstance(triplets, np.ndarray):
        triplets = np.asarray(triplets, dtype=np.int64)
    a, p, n = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    diag = np.diag(G)
    d_ap = diag[a] + diag[p] - 2.0 * G[a, p]
    d_an = diag[a] + diag[n] - 2.0 * G[a, n]
    raw = d_ap - d_an + margin
    return float(np.maximum(raw, 0.0).sum())


def triplet_gradient_E(G: np.ndarray, triplets, margin: float = 1.0) -> np.ndarray:
    if not isinstance(triplets, np.ndarray):
        triplets = np.asarray(triplets, dtype=np.int64)
    a, p, n = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    diag = np.diag(G)
    d_ap = diag[a] + diag[p] - 2.0 * G[a, p]
    d_an = diag[a] + diag[n] - 2.0 * G[a, n]
    active = (d_ap - d_an + margin) > 0.0
    if not active.any():
        return np.zeros_like(G)
    a_act, p_act, n_act = a[active], p[active], n[active]

    N = G.shape[0]
    E = np.zeros((N, N))
    # Diagonal contributions: E[p,p] += 1, E[n,n] -= 1
    np.add.at(E, (p_act, p_act), 1.0)
    np.add.at(E, (n_act, n_act), -1.0)
    # Off-diagonal symmetric contributions:
    # E[a,p] -= 1, E[p,a] -= 1, E[a,n] += 1, E[n,a] += 1
    np.add.at(E, (a_act, p_act), -1.0)
    np.add.at(E, (p_act, a_act), -1.0)
    np.add.at(E, (a_act, n_act), 1.0)
    np.add.at(E, (n_act, a_act), 1.0)
    return E


def triplet_gradient_E_reference(G: np.ndarray, triplets, margin: float = 1.0) -> np.ndarray:
    """Slow reference implementation used only for unit tests."""
    N = G.shape[0]
    E = np.zeros((N, N))
    diag = np.diag(G)
    for a, p, n in triplets:
        d_ap = diag[a] + diag[p] - 2 * G[a, p]
        d_an = diag[a] + diag[n] - 2 * G[a, n]
        if d_ap - d_an + margin > 0:
            E[p, p] += 1
            E[n, n] -= 1
            E[a, p] -= 1
            E[p, a] -= 1
            E[a, n] += 1
            E[n, a] += 1
    return E
