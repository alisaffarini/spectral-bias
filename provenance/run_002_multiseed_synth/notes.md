# run_002 -- Synthetic multi-seed sweep

**Date:** 2026-04-28
**Device:** CPU (sweep dominated by Python overhead, GPU adds no benefit at this scale)
**Wall time:** ~5 minutes

## Command

```bash
cd experiment
python run_all.py --scale medium --seeds 5 --include equiv,spectral,rank,phase --device cpu
```

## Configurations

| Sweep | Configs |
|---|---|
| Equivalence (network vs kBM ODE) | N in {10, 30}, L in {1, 2, 3}, M=1000, 200 epochs |
| Spectral bias | N=30, L in {1, 2}, M=1000, 4000 epochs |
| Rank vs width | N=10, M in {5, 8, 10, 20, 50, 100, 200}, 800 epochs |
| Phase transition | N=10, 25 lambda values, 800 epochs (later replaced -- see run_004) |

## Outputs

```
results/equivalence_N{10,30}_L{1,2,3}.json   (6 files)
results/spectral_bias_N30_L{1,2}.json        (2 files)
results/rank_vs_width.json
results/phase_transition.json   (superseded by run_004)
```

## Headline numbers

- **Equivalence:** initial relative Frobenius error <0.1% across all configurations, growing to <15% by 200 epochs (kernel drift, expected).
- **Spectral bias slope (N=30):** L=1 = 1.99 +/- 0.12, L=2 = 1.66 +/- 0.22 over 5 seeds.
- **Rank vs width:** clean saturation at rank=N for M >= N=10. Std=0 for M >= 50.
