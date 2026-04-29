# Scope and Claims (working doc, not part of paper)

This document locks the contribution boundary so the paper stops drifting between
"empirical observations" and "theory result."

## The single thesis

> Neural metric learning under NTK dynamics is a kernel-preconditioned
> Burer-Monteiro flow on the Gram matrix, and the preconditioner -- the NTK at
> initialization -- shapes which metric structures the network learns first and
> which it cannot learn at all.

Two parts:

1. **Equivalence (existing core).** Under gradient flow with NTK, the Gram
   matrix evolves as `dG/dt = -2(K E G + G E K)`. When `K = I` this is
   classical BM dynamics. So neural metric learning is kernelized BM in
   non-linear feature space.

2. **Spectral implicit bias (new payoff).** The kBM flow's convergence rate in
   each spectral direction is set by the corresponding eigenvalue of `K`.
   Components of the loss gradient `E` aligned with high-`K`-eigenvalue
   directions decay fast; components in low-eigenvalue directions are stuck.
   This gives a *predictive* theorem about which triplets the network learns
   to satisfy and in what order.

This second piece is what turns the paper from "we observed an equivalence"
into "the equivalence buys us a non-trivial prediction."

## What is in scope

- Single-layer ReLU + extension to depth-`L` fully-connected ReLU. Both with
  isotropic-NTK assumption stated as an explicit hypothesis (not a hidden
  approximation).
- Triplet loss; setup generalizes to any twice-differentiable loss on `G`.
- Synthetic experiments at `N in {10, 100, 1000}` with 5+ seeds.
- A small real-data experiment (MNIST or Fashion-MNIST embedding via triplet
  loss) at the scale a 2-layer MLP can handle on a 3070 Ti.
- Spectral-bias experiment: project gradient onto K-eigenspaces, track decay,
  compare to theory.
- Phase-transition section: rewritten as "in our setup" with multi-seed CIs;
  no universal-critical-point claim unless we can derive `lambda*` from the
  problem.

## What is out of scope (explicit limitations)

- Architectures beyond fully-connected (CNNs, transformers): future work.
- Beyond isotropic NTK (matrix-valued NTK, anisotropic): explicit limitation
  with one-paragraph discussion.
- Feature-learning regime / lazy-vs-rich training: cited (Chizat-Bach,
  Mei-Misiakiewicz-Montanari) but not analyzed; we restrict to the NTK
  regime explicitly.
- Generalization bounds: not claimed in this paper. Optimization-side only.
  Future work points to Recht-Fazel-Parrilo.

## Demoted from "contributions"

These are stated as remarks or as numerical sanity, not contributions:

- `G = F F^T`. True by construction once we define `G := F F^T`.
- `kBM(K=I) = sBM`. True by inspection once both ODEs are written down.
- `||G||_* = ||F||_F^2 = tr(G)` for PSD `G = F F^T`. Two-line lemma.
- `K_init` vs `G_final` spectral correlation rho ~ 1. Two sorted decaying
  spectra correlate by default; replaced with the spectral-bias experiment.

## New contributions list (paper-facing)

1. Theorem (kBM equivalence): Under isotropic NTK, gradient flow on a feature
   network yields `dG/dt = -2(K E G + G E K)`; when `K = I` this reduces to
   BM dynamics (Burer-Monteiro 2003).
2. Theorem (depth-L extension): The kBM equivalence holds for fully-connected
   depth-L ReLU networks with the depth-L NTK substituted for `K`, under the
   standard NTK regime.
3. Theorem (spectral implicit bias): Decomposing `E(t)` in the eigenbasis of
   `K`, each component decays at rate proportional to the corresponding
   eigenvalue. Triplets whose error sits in the bottom eigenspace of `K`
   are not learnable in the NTK regime without explicit kernel modification.
4. Numerical illustration: validates the equivalence (synthetic, 5+ seeds, up
   to N=1000), the depth-L extension (L=1,2,3), and the spectral-bias
   prediction (network and ODE agree on directional decay rates).
5. Real-data illustration: triplet learning on Fashion-MNIST exhibits the
   spectral-bias prediction: confused-pair errors live in `K`'s bottom
   eigenspace.

## Target venues

- Primary: AISTATS 2027 (deadline ~Oct 2026).
- Backup: NeurIPS 2026 Math-of-ML or DeepMath workshop (Sept-Oct 2026).
- Fallback: TMLR (rolling).

## What we will NOT do this revision

- No CUB-200 / Cars-196 / SOP. Out of scope at this iteration; would need
  full image pipeline and ImageNet-pretrained backbones to be competitive,
  which doesn't add to the theory's claims and would dilute the
  contribution. MNIST-class is enough to demonstrate non-synthetic
  applicability.
- No comparison with proxy/contrastive baselines. The paper is a theory
  paper; baseline tables would invite the wrong comparison.
- No claim about feature learning beyond NTK. Explicitly out of scope.
