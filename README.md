# Kernelized Burer-Monteiro Dynamics

A spectral theory of implicit bias for neural metric learning.

**Authors:** Ali Saffarini, Hemmy Kalam (Harvard College)

This repo contains the LaTeX source, code, and raw experimental results for
the paper "Kernelized Burer-Monteiro Dynamics: A Spectral Theory of
Implicit Bias for Neural Metric Learning."

## Layout

```
paper/                     LaTeX source + figures
  main.tex                 paper source
  references.bib           bibliography
  SCOPE.md                 working doc, locks contribution scope (not part of paper)
  fig_*.png                publication figures (generated from raw results)
  main.pdf                 compiled PDF

experiments/               experimental code (Python)
  kbm/                     library: kernels, dynamics, network, metrics
  runners/                 one runner per experiment, each yields JSON
  run_all.py               top-level orchestrator (multi-seed sweep)
  make_figures.py          regenerates paper figures from JSON

results_v2/raw/            raw multi-seed JSON outputs (5 seeds each)

results/                   legacy figures from the original Dec 2025 project version
paper_submission.zip       legacy submission archive (untouched)
ntk_kernels.py             legacy single-file NTK kernels (superseded by kbm/)
run_final_paper_experiments.py   legacy experiment script (superseded by experiments/)
```

## Reproducing the experiments

```bash
cd experiments
# Sanity: small-scale run, 3 seeds, ~3 minutes
python run_all.py --scale small --seeds 3 --include equiv,spectral,rank

# Paper figures: medium-scale, 5 seeds, ~10 minutes
python run_all.py --scale medium --seeds 5 --include equiv,spectral,rank,phase
python run_all.py --scale medium --seeds 5 --include fashion --device cuda

python make_figures.py
```

The orchestrator writes JSON to `../results_v2/raw/` and figures land in
`../paper/`.

## Compiling the paper

```bash
cd paper
tectonic -X compile main.tex
```

## Theorem at a glance

Gradient flow on a depth-`L` ReLU embedder with twice-differentiable loss
on `G = F F^T`, in the NTK regime, evolves as

```
dG/dt = -2 (K E G + G E K),    E = dL/dG
```

where `K` is the depth-`L` NTK at the data. In `K`'s eigenbasis this
becomes `dG_tilde/dt = -2 (Lambda E_tilde G_tilde + G_tilde E_tilde Lambda)`,
so each mode evolves at a rate proportional to its kernel eigenvalue. This
is the spectral implicit bias: bottom-eigenvalue modes are essentially
frozen on bounded time intervals.
