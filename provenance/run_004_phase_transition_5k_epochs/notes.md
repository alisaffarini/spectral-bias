# run_004 -- Phase transition rerun with 5000 epochs

**Date:** 2026-04-28
**Device:** CPU
**Wall time:** ~8 minutes

## Why a rerun

The medium-scale sweep in run_002 used 800 epochs, which was insufficient for the nuclear-norm regularizer to drive rank down. The rank stayed at 10 across the entire lambda range. We rerun with 5000 epochs and a wider lambda range (logspace(-1, 2) instead of logspace(-3, 1)).

## Command (Python)

```python
from runners.phase_transition import PhaseConfig, run_sweep
from pathlib import Path
import numpy as np
cfg = PhaseConfig(n_epochs=5000, lr=1e-4, width=200, record_every=2000,
                  lambdas=tuple(np.logspace(-1, 2, 25).tolist()))
run_sweep(cfg, list(range(5)), Path('../results/phase_transition.json'))
```

## Outputs

`results/phase_transition.json` (overwrites the run_002 file)

## Findings

- Clean phase-transition shape: rank stays at 10 for lambda < 1, drops sharply over lambda in [2, 25], hits minimum effective rank ~2 at lambda ~20.
- **Numerical artifact at lambda > 30:** the regularizer drives G toward zero. With G nearly zero, the relative-eigenvalue rank metric (count of singular values above 1% of the largest) becomes noise-dominated and bounces back up. The figure (`paper/fig_phase.png`) trims at lambda <= 30.
- We frame this in the paper as an empirical observation, not a phase-transition theorem.
