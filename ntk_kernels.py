"""
NTK Kernel Functions for Metric Learning Experiments
"""
import numpy as np


def _relu_ntk_integral(cos_theta):
    """
    Helper function for the standard scalar NTK kernel (Ntk_1) for ReLU.
    κ(x, y) = (1/2π) √(1 - cos²θ) + (π - θ)/2π cos θ
    where θ = arccos(cos θ).
    """
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return (1 / (2 * np.pi)) * (np.sin(theta) + cos_theta * (np.pi - theta))


def standard_ntk_kernel(X, Y=None):
    """
    Calculates the standard scalar NTK kernel matrix K(x, y) for a single-layer ReLU network.
    Assumes inputs are normalized (||x|| = 1).
    
    Args:
        X: Input data matrix (N x d), rows should be normalized
        Y: Optional second data matrix (M x d). If None, uses X.
    
    Returns:
        K: NTK kernel matrix (N x M)
    """
    if Y is None:
        Y = X
    cos_theta = X @ Y.T
    K = _relu_ntk_integral(cos_theta)
    return K


def generate_synthetic_data(N, d_in, rank_true, seed=42):
    """
    Generates synthetic data X and a target Gram matrix G_target.
    
    Args:
        N: Number of data points
        d_in: Input dimension
        rank_true: True rank of target Gram matrix
        seed: Random seed for reproducibility
    
    Returns:
        X: Normalized data matrix (N x d_in)
        G_target: Target Gram matrix (N x N) with rank = rank_true
        D_target_sq: Target squared distance matrix (N x N)
    """
    np.random.seed(seed)
    
    # 1. Generate data X (normalized)
    X = np.random.randn(N, d_in)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # 2. Generate target Gram matrix G_target (rank-constrained)
    F_target = np.random.randn(N, rank_true)
    G_target = F_target @ F_target.T
    
    # 3. The target pairwise squared distances D_target_sq
    diag_G = np.diag(G_target)
    D_target_sq = diag_G[:, None] + diag_G[None, :] - 2 * G_target
    
    return X, G_target, D_target_sq



