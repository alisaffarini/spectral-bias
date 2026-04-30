# run_008 -- Recall@K test-time evaluation

**Date:** 2026-04-30
**Device:** CUDA (RTX 3070 Ti Laptop)
**Wall time:** 75.1s (easy) + 90.9s (hard) = ~3 minutes total

## Why

The first reviewer question for an "algorithmic-payoff" claim is "does it improve test performance?" The slope-flattening result (run_006) is mechanistic, not behavioral. We need a held-out test-set metric to close the loop.

## Setup

- 600 CIFAR-10 images, 60 per class, ResNet-18 frozen features, PCA to 128 dims, unit-normalized.
- Split: 400 train images, 200 test images (per-class balanced).
- Train both methods on 2000 triplets formed from training labels only.
- Evaluate retrieval: for each test image, find the K nearest training-set images in embedding space; hit if any of them shares the test image's class.
- Two triplet-sampling regimes:
  - `easy`: random different-class negative.
  - `hard`: nearest different-class negative in input (PCA-feature) space. By construction, the loss gradient for hard triplets loads on K's bottom eigenspace.

## Outputs

`results/recall_easy.json`, `results/recall_hard.json` (5 seeds each).

## Headline numbers

```
                  Vanilla              Preconditioned       Diff             paired t-test
EASY triplets
recall@5         0.910 +/- 0.019      0.916 +/- 0.012      +0.006           NS
recall@10        0.952 +/- 0.017      0.954 +/- 0.013      +0.002           NS
recall@20        0.977 +/- 0.012      0.981 +/- 0.012      +0.004           NS

HARD triplets
recall@5         0.875 +/- 0.019      0.897 +/- 0.023      +0.022 +/- 0.011  t=3.92, p=0.017
recall@10        0.919 +/- 0.014      0.951 +/- 0.015      +0.032 +/- 0.011  t=5.94, p=0.004
recall@20        0.955 +/- 0.010      0.977 +/- 0.015      +0.022 +/- 0.012  t=3.77, p=0.020
```

## Interpretation

Result matches theory. On easy triplets the gradient loads on K's top eigenspace, vanilla fits naturally, and the methods tie. On hard triplets the gradient is in K's bottom eigenspace (frozen for vanilla per Theorem 6); the preconditioner unfreezes those modes and gains 2-3 percentage points at recall@5/10/20, statistically significant via paired t-test (p < 0.02 for all three K values, p = 0.004 for recall@10).
