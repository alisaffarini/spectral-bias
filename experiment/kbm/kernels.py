"""NTK kernels: single-layer (Jacot 2018, Cho-Saul 2009 arc-cosine) and depth-L recursive."""
import numpy as np


def _arc_cosine_kappa1(cos_theta: np.ndarray) -> np.ndarray:
    """First-order arc-cosine kernel for ReLU activations under unit-norm inputs.

    kappa_1(u, v) = (1/pi) * (sin(theta) + (pi - theta) * cos(theta)) * ||u|| ||v||
    Here we assume the inputs to this layer have unit norm so ||u||=||v||=1.
    """
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return (1.0 / np.pi) * (np.sin(theta) + (np.pi - theta) * cos_theta)


def _arc_cosine_kappa1_dot(cos_theta: np.ndarray) -> np.ndarray:
    """Derivative-flavored arc-cosine kernel: kappa_0 in Cho-Saul, used for NTK recursion.

    kappa_0(u, v) = (1/pi) * (pi - theta), the activation-derivative kernel for ReLU.
    """
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return (1.0 / np.pi) * (np.pi - theta)


def standard_ntk_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Single-layer ReLU NTK on unit-norm inputs.

    The standard scalar NTK used in the original kBM paper is the sum
    Sigma_1 + Sigma_0 * <x,y>, where Sigma_1 = kappa_1(<x,y>) is the
    feature-feature kernel and Sigma_0 = kappa_0(<x,y>) is the
    derivative-derivative kernel (Jacot et al. 2018, eq. 6, single hidden
    layer with ReLU). For unit-norm inputs this collapses to a function of
    cos theta only.
    """
    if Y is None:
        Y = X
    cos_theta = X @ Y.T
    sigma_1 = _arc_cosine_kappa1(cos_theta)
    sigma_0 = _arc_cosine_kappa1_dot(cos_theta)
    return sigma_1 + sigma_0 * cos_theta


def depth_L_ntk_kernel(X: np.ndarray, L: int, Y: np.ndarray | None = None) -> np.ndarray:
    """Depth-L fully-connected ReLU NTK via the Jacot recursion.

    Recursion (with unit-norm inputs):
        Sigma^{(0)}(x, y) = <x, y>
        Sigma^{(l)}     = kappa_1(rho^{(l-1)})    where rho^{(l-1)} is the
                          correlation derived from Sigma^{(l-1)}
        Sigma_dot^{(l)} = kappa_0(rho^{(l-1)})

        Theta^{(1)}     = Sigma^{(1)} + Sigma_dot^{(1)} <x, y>
        Theta^{(l)}     = Sigma^{(l)} + Sigma_dot^{(l)} * Theta^{(l-1)}, l >= 2

    Returns Theta^{(L)} (N x M).
    """
    if L < 1:
        raise ValueError("depth must be >= 1")
    if Y is None:
        Y = X
    if L == 1:
        return standard_ntk_kernel(X, Y)

    # Build the correlation matrix rho^{(l)} from Sigma^{(l)} using the diagonals.
    Sigma = X @ Y.T
    Sigma_xx = np.einsum('ij,ij->i', X, X)  # diagonal of X X^T
    Sigma_yy = np.einsum('ij,ij->i', Y, Y)
    diag_x = Sigma_xx
    diag_y = Sigma_yy

    # Layer-1 NTK pieces
    rho = Sigma / np.sqrt(np.outer(diag_x, diag_y) + 1e-30)
    rho = np.clip(rho, -1.0, 1.0)
    Sigma_next = np.sqrt(np.outer(diag_x, diag_y)) * _arc_cosine_kappa1(rho)
    Sigma_dot = _arc_cosine_kappa1_dot(rho)
    Theta = Sigma_next + Sigma_dot * Sigma

    # Diagonal of Sigma_next is needed for next-layer correlation
    diag_x = np.diag(Sigma_next) if X is Y else (
        np.sqrt(diag_x * diag_x) * _arc_cosine_kappa1(np.ones_like(diag_x))
    )
    diag_y = diag_x  # in the symmetric case

    Sigma = Sigma_next

    for _layer in range(2, L + 1):
        rho = Sigma / np.sqrt(np.outer(diag_x, diag_y) + 1e-30)
        rho = np.clip(rho, -1.0, 1.0)
        Sigma_next = np.sqrt(np.outer(diag_x, diag_y)) * _arc_cosine_kappa1(rho)
        Sigma_dot = _arc_cosine_kappa1_dot(rho)
        Theta = Sigma_next + Sigma_dot * Theta
        diag_x = np.diag(Sigma_next)
        diag_y = diag_x
        Sigma = Sigma_next

    return Theta


def normalize_kernel(K: np.ndarray) -> np.ndarray:
    """Diagonal-normalize K to a correlation kernel for cleaner spectra."""
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return K / np.outer(d, d)
