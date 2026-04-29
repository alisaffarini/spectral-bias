# run_007 -- CIFAR-10 with ResNet-18 features

**Date:** 2026-04-29
**Device:** CUDA (RTX 3070 Ti Laptop)
**Wall time:** ~75 seconds (5 seeds)

## Why

Real-data validation that goes beyond Fashion-MNIST. Reviewers will expect at least one image-classification benchmark for a paper claiming relevance to neural metric learning. CIFAR-10 with frozen ImageNet-pretrained features is the cheapest credible setup: it captures real visual statistics, the network we train on top is small enough to study cleanly, and the depth-2 NTK on the frozen features is well-defined.

## Setup

- Frozen ImageNet-pretrained ResNet-18 (`torchvision.models.resnet18(weights=DEFAULT)`).
- Feature extraction: replace the final FC layer with `Identity()`, run the test images through, take the 512-dim penultimate output.
- 400 CIFAR-10 images, 40 per class across all 10 classes.
- PCA-reduce features to 128 dims, unit-normalize.
- Train a 2-layer ReLU head (depth-2, width 1024) with triplet loss for 1500 epochs.
- 1000 triplets per seed sampled from class labels.
- Run both vanilla and preconditioned training (per run_006).

## Command

```python
from runners.cifar10 import CIFARConfig, run_sweep
from pathlib import Path
cfg = CIFARConfig(N=400, n_epochs=1500, n_triplets=2000, feature_dim=128)
run_sweep(cfg, list(range(5)), Path('../results/cifar10.json'), device='cuda')
```

## Outputs

`results/cifar10.json` (5 seeds, 400 modes each, 16 recorded epochs)

## Headline numbers (5 seeds)

- Vanilla slope: **2.058 +/- 0.011** -- matches synthetic slope-2 prediction.
- Preconditioned slope: **1.589 +/- 0.016** -- 23% reduction in spectral bias.

Std of 0.01 across seeds rules out seed noise; the phenomenon is reproducible on real CIFAR-10 features.

## Discrepancy with synth

Preconditioner reduces slope 2.06 -> 1.59 on CIFAR (23% reduction) vs 1.99 -> 0.74 on synth (63% reduction). The depth-2 NTK on PCA-reduced ResNet features has heavier-tailed spectrum than the synthetic depth-1 NTK, so the eps regularization in K^{-1} (eps=1e-2) clips more of the relevant range, and the preconditioner is less aggressive in proportion. Lowering eps further trades robustness for flatness; we kept the synthetic-validated value.
