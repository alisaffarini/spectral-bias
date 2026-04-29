# Kernelized Burer-Monteiro Dynamics: A Spectral Theory of Implicit Bias for Neural Metric Learning

**Authors:** Anonymous Authors, Anonymous Institution

## Key Results

- **Equivalence theorem (Theorem 1, 2).** Under gradient flow with the standard NTK approximation, a depth-L ReLU embedder trained with any twice-differentiable loss on its Gram matrix `G = FF^T` evolves as `dG/dt = -2 (K E G + G E K)`, the kernel-preconditioned Burer-Monteiro flow. When `K = I` this reduces to classical BM dynamics.
- **Spectral implicit-bias theorem (Theorem 4, 6).** In `K`'s eigenbasis the kBM flow's mode `(i,j)` evolves at rate `lambda_i + lambda_j`. Modes with `lambda_i = 0` are exactly frozen; under NTK-regime hypotheses the per-mode displacement satisfies `|G_tilde_ii(T) - G_tilde_ii(0)| <= 4 c_max T E lambda_i^2 + O(T^2)`, a quadratic-in-`lambda` bound matching empirical slopes.
- **Empirical validation across 6 configurations (5 seeds each):**
  - Synth N=30, L=1: slope 2.19 +/- 0.12
  - Synth N=30, L=2: slope 1.66 +/- 0.22
  - Synth N=100, L=1/L=2: 1.49 +/- 0.03 / 1.94 +/- 0.03
  - Synth N=1000, L=1/L=2: 1.38 +/- 0.001 / 1.60 +/- 0.002
  - Fashion-MNIST L=2: 1.70
  - **CIFAR-10 (ResNet-18 features, depth-2 head): 2.06 +/- 0.01**
- **Algorithmic payoff: spectral preconditioner.** A one-line modification to backprop (apply `K^{-1}` to the embedding-side gradient) flattens the spectral bias from slope `1.99 +/- 0.11` to `0.74 +/- 0.10` on synthetic, and `2.06 +/- 0.01` to `1.59 +/- 0.01` on CIFAR-10.
- **Funk-Hecke proof of approximate eigen-alignment** (Proposition 5) in the appendix justifies the slope-2 refinement using arc-cosine kernel decomposition on `S^{d-1}`.

## Datasets and seeds

| Setup | Scale | Seeds |
|---|---|---|
| Synthetic Gaussian on `S^{d-1}` | N in {30, 100, 1000}, L in {1, 2} | 5 (0-4) |
| Fashion-MNIST | N=200, depth-2 head, 4 classes | 5 (0-4) |
| CIFAR-10 | N=400, ResNet-18 frozen features, depth-2 head, 10 classes | 5 (0-4) |
| Spectral preconditioner | N=50, depth-1 head | 5 (0-4) |

## Structure

```
paper/        LaTeX source, figures, refs.bib, compiled PDF, scope doc
results/      Multi-seed JSON outputs, one file per experiment configuration
experiment/   Python code: kbm/ library + runners/ + run_all.py + make_figures.py
provenance/   Per-run records of what was executed and what came out
results_legacy/  December 2025 project-version figures (preserved)
```

Legacy artifacts (untouched, preserved for diff): `ntk_kernels.py`, `run_final_paper_experiments.py`, `paper_submission.zip`.

## Reproducing the experiments

```bash
pip install -r requirements.txt
cd experiment

# Synthetic + spectral-bias sweep, 5 seeds, ~10 minutes on CPU
python run_all.py --scale medium --seeds 5 --include equiv,spectral,rank,phase

# Fashion-MNIST + CIFAR-10 (GPU)
python -c "from runners.fashion_mnist import FashionConfig, run_sweep; from pathlib import Path; run_sweep(FashionConfig(n_epochs=2000), list(range(5)), Path('../results/fashion_mnist.json'), device='cuda')"
python -c "from runners.cifar10 import CIFARConfig, run_sweep; from pathlib import Path; run_sweep(CIFARConfig(), list(range(5)), Path('../results/cifar10.json'), device='cuda')"
python -c "from runners.preconditioner import PreconditionerConfig, run_sweep; from pathlib import Path; run_sweep(PreconditionerConfig(), list(range(5)), Path('../results/preconditioner_N50.json'), device='cuda')"

# Big-N spectral-bias (GPU, ~3 minutes total)
python -c "from runners.spectral_bias import SpectralConfig, run_sweep; from pathlib import Path
for N in [100, 1000]:
    for L in [1, 2]:
        run_sweep(SpectralConfig(N=N, depth=L, n_epochs=2000 if N==100 else 1000), list(range(5)), Path(f'../results/spectral_bias_N{N}_L{L}.json'), device='cuda')"

# Generate all paper figures from JSON
python make_figures.py
```

## Compiling the paper

```bash
cd paper
tectonic -X compile main.tex   # produces main.pdf
```

## Theorem at a glance

Gradient flow on a depth-`L` ReLU embedder with twice-differentiable loss on `G = FF^T`, in the NTK regime, evolves as `dG/dt = -2 (K E G + G E K)` where `E = dL/dG` and `K` is the depth-`L` NTK at the data.

In `K = U Lambda U^T` eigenbasis this becomes `dG_tilde/dt = -2 (Lambda E_tilde G_tilde + G_tilde E_tilde Lambda)`. Each mode evolves at a rate set by the corresponding kernel eigenvalue: bottom-eigenvalue modes are essentially frozen on bounded time intervals. This is the spectral implicit bias.

## Citation

Target venue: NeurIPS 2026 (anonymous submission).
