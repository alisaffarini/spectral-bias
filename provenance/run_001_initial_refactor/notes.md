# run_001 -- Initial refactor

**Date:** 2026-04-28

## What

Refactored the December 2025 single-file project (`run_final_paper_experiments.py`, ~1380 lines) into a modular layout under `experiment/kbm/` (library) and `experiment/runners/` (one runner per experiment, each producing JSON to `results/`). Vectorized the triplet gradient via `numpy.add.at` so the kBM ODE step stays cheap at large N.

## Verification

Single-seed test of the equivalence runner: N=10, depth=1, 100 epochs. Initial relative Frobenius error 0.092%, growing to 8.3% by epoch 100. Consistent with the December project's reported 0.084% initial / 7.27% final.

The vectorized triplet gradient was checked against the original Python-loop reference implementation: `np.allclose` with max diff 0.

## Outputs

No JSON yet -- this run was infrastructure-only. Set up the orchestrator (`run_all.py`) with `--scale {small, medium, large}` and `--include` flags so subsequent runs are reproducible from a single command.
