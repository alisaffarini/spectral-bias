# run_003 -- Fashion-MNIST validation on GPU

**Date:** 2026-04-28
**Device:** CUDA (RTX 3070 Ti Laptop, 8 GB VRAM)
**Wall time:** ~3 minutes

## Command (Python equivalent)

```python
from runners.fashion_mnist import FashionConfig, run_sweep
from pathlib import Path
cfg = FashionConfig(N=200, depth=2, n_epochs=2000, n_triplets=1000,
                    width=1024, classes=(0, 1, 2, 3))
run_sweep(cfg, list(range(5)), Path('../results/fashion_mnist.json'), device='cuda')
```

## Setup

- 200 Fashion-MNIST images, 4 classes (T-shirt, Trouser, Pullover, Dress), 50 per class.
- Inputs centered then PCA-reduced to 64 dims, unit-normalized.
- Network: 2-layer fully-connected ReLU, width 1024.
- 1000 triplets per seed sampled from class labels (anchor-positive same class, negative different).
- Depth-2 NTK computed analytically (Cho-Saul recursion).
- Tracking: `G_tilde(t) = U^T G(t) U` projected into K's eigenbasis at every recorded epoch.

## Outputs

```
results/fashion_mnist.json   (5 seeds, 200 modes per seed, 21 recorded epochs)
```

## Headline numbers

- Per-mode displacement vs kernel eigenvalue: power-law slope **1.70** across 200 modes spanning ~3 orders of magnitude in eigenvalue.
- Confirms the spectral-bias prediction transfers to real image data with class structure.
