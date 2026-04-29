"""Common scalar metrics used across experiments."""
from __future__ import annotations
import numpy as np
from scipy.linalg import svd


def effective_rank(G: np.ndarray, threshold: float = 0.01) -> int:
    s = svd(G, compute_uv=False)
    if len(s) == 0 or s[0] <= 0:
        return 0
    return int(np.sum(s / s[0] > threshold))


def factorization_error(G: np.ndarray, F: np.ndarray) -> float:
    G_from_F = F @ F.T
    denom = np.linalg.norm(G, "fro")
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(G - G_from_F, "fro") / denom)


def nuclear_norm(G: np.ndarray) -> float:
    return float(svd(G, compute_uv=False).sum())


def relative_frob(A: np.ndarray, B: np.ndarray) -> float:
    denom = np.linalg.norm(A, "fro")
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(A - B, "fro") / denom)


def project_E_on_K_eigenbasis(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return |U^T E U| diagonal: per-eigendirection magnitude of E.

    This gives the per-mode 'force' driving G's evolution in the kBM ODE
    decomposed in K's eigenbasis.
    """
    eigvals, U = np.linalg.eigh(K)
    Et = U.T @ E @ U
    return np.abs(np.diag(Et))


def k_eigvals(K: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(K)
