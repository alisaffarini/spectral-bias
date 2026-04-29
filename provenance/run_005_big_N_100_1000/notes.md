# run_005 -- Big-N spectral-bias sweep (N=100, N=1000)

**Date:** 2026-04-29
**Device:** CUDA (RTX 3070 Ti Laptop)
**Wall time:** ~3 minutes total

## Why

The slope-2 phenomenon needs to hold at scale. N=30 left room for "small-N artifact" critique. We push to N=100 and N=1000.

## Commands (Python)

```python
from runners.spectral_bias import SpectralConfig, run_sweep
from pathlib import Path
seeds = list(range(5))
for L in [1, 2]:
    cfg = SpectralConfig(N=100, depth=L, n_epochs=2000, record_every=200,
                         width=2000, n_triplets_factor=8)
    run_sweep(cfg, seeds, Path(f'../results/spectral_bias_N100_L{L}.json'),
              device='cuda')
for L in [1, 2]:
    cfg = SpectralConfig(N=1000, depth=L, n_epochs=1000, record_every=200,
                         width=2000, n_triplets_factor=4)
    run_sweep(cfg, seeds, Path(f'../results/spectral_bias_N1000_L{L}.json'),
              device='cuda')
```

## Outputs

```
results/spectral_bias_N100_L{1,2}.json
results/spectral_bias_N1000_L{1,2}.json
```

## Headline numbers (5 seeds)

| | L=1 | L=2 |
|---|---|---|
| N=100 | 1.49 +/- 0.03 | 1.94 +/- 0.03 |
| N=1000 | **1.38 +/- 0.001** | **1.60 +/- 0.002** |

Std at N=1000 is ~0.001 across seeds, ruling out seed noise as the explanation. The slope-2-ish phenomenon is robust to N at least up to 1000 modes.

## Per-seed timing on RTX 3070 Ti

- N=100, L=1: 19.7s; L=2: 25.5s
- N=1000, L=1: 33.6s; L=2: 73.5s
