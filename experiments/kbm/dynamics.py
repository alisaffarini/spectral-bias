"""kBM ODE integrator: dG/dt = -2 (K E G + G E K)."""
from __future__ import annotations
import numpy as np
from .loss import triplet_loss_value, triplet_gradient_E


def run_kbm_ode(
    G_init: np.ndarray,
    K: np.ndarray,
    triplets,
    total_time: float,
    n_steps: int,
    margin: float = 1.0,
    record_every: int = 1,
):
    """Integrate dG/dt = -2 (K E(G) G + G E(G) K) using RK4.

    Returns (times, G_history, loss_history).
    """
    N = G_init.shape[0]
    dt = total_time / n_steps
    G = G_init.copy()

    def rhs(G_):
        E = triplet_gradient_E(G_, triplets, margin=margin)
        return -2.0 * (K @ E @ G_ + G_ @ E @ K)

    times = [0.0]
    history = [G.copy()]
    losses = [triplet_loss_value(G, triplets, margin=margin)]

    for step in range(n_steps):
        k1 = rhs(G)
        k2 = rhs(G + 0.5 * dt * k1)
        k3 = rhs(G + 0.5 * dt * k2)
        k4 = rhs(G + dt * k3)
        G = G + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Force symmetry to suppress numerical drift.
        G = 0.5 * (G + G.T)

        if (step + 1) % record_every == 0 or step == n_steps - 1:
            times.append((step + 1) * dt)
            history.append(G.copy())
            losses.append(triplet_loss_value(G, triplets, margin=margin))

    return np.array(times), history, np.array(losses)
