# run_006 -- Spectral preconditioner (algorithmic payoff)

**Date:** 2026-04-29
**Device:** CUDA (RTX 3070 Ti Laptop)
**Wall time:** ~60 seconds

## Why

The slope-2 spectral bias is descriptive. To claim the kBM viewpoint yields actionable predictions, we use it to derive a training-procedure modification and verify the modification has the predicted effect (flatter spectral bias).

## Modification

In standard PyTorch backprop, `dL/dtheta = J^T @ vec(dL/dF)` where `F ∈ R^{N×M}` is the embedding and `J` is the network Jacobian. We replace with:

```
dL/dtheta = J^T @ vec(K^{-1} @ dL/dF)
```

`K^{-1}` is applied to the data axis of the embedding-side gradient. In the NTK regime this exactly cancels the `K`-prefactor in the kBM dynamics:

```
dG/dt = -2(K [K^{-1} E] G + G [K^{-1} E] K) = -2(E G + G E)
```

so the resulting Gram-matrix dynamics are the un-preconditioned BM flow.

## Implementation

`experiment/runners/preconditioner.py::train_preconditioned`:

1. Forward pass to F (with `retain_grad`).
2. `torch.autograd.grad(loss, F)` to get `dL/dF` directly.
3. Multiply: `grad_F_precond = K_inv @ grad_F`.
4. Forward pass again: `F2 = model(X)`.
5. `F2.backward(grad_F_precond)` -- arbitrary upstream gradient.
6. `optimizer.step()`.

Adds one matrix-solve per step. K_inv is computed once at startup.

## Command

```python
from runners.preconditioner import PreconditionerConfig, run_sweep
from pathlib import Path
cfg = PreconditionerConfig(N=50, n_epochs=2000, record_every=100,
                           width=1000, lr=1e-4, eps=1e-2)
run_sweep(cfg, list(range(5)), Path('../results/preconditioner_N50.json'),
          device='cuda')
```

## Outputs

`results/preconditioner_N50.json`

## Headline numbers (5 seeds)

- Vanilla slope: **1.99 +/- 0.11** (matches Theorem 6 prediction)
- Preconditioned slope: **0.74 +/- 0.10** (2.7x flatter)
- Both reach ~83% triplet satisfaction; the preconditioner spreads work across the spectrum rather than focusing on top eigenmodes.
