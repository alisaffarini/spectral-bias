# run_009 -- End-to-end CIFAR-10 with small CNN

**Date:** 2026-04-30
**Device:** CUDA (RTX 3070 Ti Laptop)
**Wall time:** 207.7s (5 seeds, ~40s/seed)

## Why

The CIFAR-10 experiment in run_007 used frozen ResNet-18 features, leaving the reviewer concern that the spectral-bias phenomenon is an artifact of the fixed-feature setup. This run trains a small CNN from scratch end-to-end with triplet loss on CIFAR-10 to rule that out.

## Setup

- 100 CIFAR-10 images, 10 per class, 32x32 normalized.
- Small CNN: 3 conv blocks (32-32-64 channels), kernel 3, ReLU, max-pool, adaptive avg pool, 32-dim linear head with ReLU. ~104k params.
- Training: 1200 epochs, lr=1e-6 (Kaiming-init scale), 400 random-class triplets per seed.
- Empirical NTK at initialization: K_emp[i,j] = (1/M) sum_m <dF_m(x_i)/dtheta, dF_m(x_j)/dtheta> via per-feature backward passes. ~25s per seed for the NTK computation.

## Outputs

`results/cifar10_end2end.json` (5 seeds, 100 modes per seed)

## Headline numbers

- Vanilla slope: **1.15 +/- 0.07**
- Preconditioned slope: **0.97 +/- 0.11**
- Slope reduction: 16% (smaller than synth's 63% because the empirical CNN-NTK has heavier-tailed spectrum: lambda_max/lambda_min ratio ~ 10^4 here vs ~ 10^2 on synth).

## Interpretation

The kBM equivalence and spectral-bias prediction are NOT artifacts of the frozen-feature setup. End-to-end CIFAR-10 with a CNN from scratch shows the same monotone scaling of per-mode displacement with kernel eigenvalue, and the K^{-1}-preconditioner reduces the slope as predicted. The slope is somewhat flatter than the slope-2 synthetic prediction because the empirical NTK's heavier-tailed spectrum makes the eps-regularization in K^{-1} more aggressive in proportion.
