# NeurIPS 2026 OpenReview Submission — kBM paper

Copy-paste sheet for the abstract registration. Today (2026-05-05) is the
abstract-registration deadline; the PDF can keep being updated until the
full-paper deadline. **Four fields lock at submission and cannot be edited
later** (flagged with 🔒 below): Reviewer Nomination, Primary Area,
Secondary Area, Contribution Type. Everything else (PDF, abstract text,
title, license, LLM disclosures) remains editable.

## Track: Main Track

NeurIPS 2026 has three tracks:
- **Main Track** ← submit here
- Evaluations and Datasets Track (separate CFP, not applicable)
- Position Papers Track (separate CFP, not applicable)

Plus separate Reproducibility and Competitions calls (not applicable).
This is a theoretical paper with experiments — exactly the Main Track's
intended scope.

---

## Pre-submit checklist (verify before clicking submit)

- [ ] OpenReview profile complete: affiliations under "Education & Career
      History" cover the last 3 years, all email addresses added, DBLP
      imported if available. Incomplete profiles can trigger desk
      rejection (Handbook §OpenReview Setup).
- [ ] PDF anonymized — `main.pdf` already says "Anonymous Author(s)";
      do NOT upload a de-anonymized variant.
- [ ] PDF includes the checklist — confirmed: `main.tex` line 1165
      `\input{checklist.tex}`, rendered into the compiled
      `kbm-spectral-bias/paper/main.pdf` (built 2026-05-05).
- [ ] Final PDF page count: 9-page main body + references + appendix +
      checklist (~20 pages total per project memory).
- [ ] Confirm exact OpenReview ID format from your profile URL
      (`~Firstname_Lastname[n]`). The placeholder used below is
      `~Ali_Saffarini1` — replace if your real ID differs.
- [ ] Supplementary code/data: NOT required today; can be uploaded as a
      separate ZIP up to the full-paper deadline. When you do upload,
      the GitHub link must be anonymized (Handbook §Double-blind
      Reviewing). Your note about not having anonymized the GitHub yet is
      fine — that's a later step.

---

## Title*

```
Kernel-Preconditioned Burer–Monteiro Flow: A Spectral Implicit Bias for Wide Networks Trained with Triplet Loss
```

(Verbatim from `kbm-spectral-bias/paper/main.tex` line 33. Uses an en-dash
between "Burer" and "Monteiro" — OpenReview renders Unicode fine; you can
also enter it as `Burer--Monteiro` and it will display correctly.)

---

## Authors*

- **Ali Saffarini** — `~Ali_Saffarini1`

Solo author. The handbook is explicit that author lists cannot be added
to or removed from after the submission deadline (only reordered), so
since you're solo this is settled.

---

## TL;DR

```
Wide-network metric learning is a kernel-preconditioned Burer–Monteiro flow with a quadratic spectral implicit bias; we prove a slope-≤2 power-law bound and derive a one-line K⁻¹ correction that improves hard-negative retrieval.
```

(228 characters, within the OpenReview 250-char limit. Captures all
three contributions: the equivalence theorem, the slope-$\leq 2$
bound, and the algorithmic K$^{-1}$ correction. Optional field, but
worth filling — reviewers and ACs scan TL;DRs during paper-bidding.)

---

## Abstract*

Adapted from `main.tex` lines 47–70 (the high-level rewrite from
2026-05-04, no equations/numbers, per reviewer feedback). Anonymous,
~250 words. **OpenReview only processes `$...$` math, not LaTeX
text-mode syntax**, so the version below replaces:
- LaTeX `---` → Unicode em-dash `—` (otherwise renders as literal `---`)
- LaTeX `Burer--Monteiro` → Unicode en-dash `Burer–Monteiro`
  (otherwise renders as literal `Burer--Monteiro`)
- `slope-$\leq 2$` → `slope $\leq 2$` (drops the hyphen, reads as
  "slope ≤ 2" instead of the awkward "slope-≤ 2" rendering)

The asterisks around `*kernel-preconditioned matrix flow*` and
`*quadratic*` render as italics in OpenReview markdown — keep as-is.

```
Neural metric learning trains an embedding network with a triplet or contrastive loss, but which similarity geometries gradient descent can actually fit on a finite training horizon is poorly understood. We address this by bridging two theoretical lenses — the neural tangent kernel (NTK) for wide-network gradient dynamics, and the Burer–Monteiro factorisation for low-rank semidefinite optimization — in a way that has not been combined for primal feature-space training: gradient flow on a wide embedding network induces a closed-form *kernel-preconditioned matrix flow* on the Gram matrix of the embeddings, with the NTK at initialization as the preconditioner. The resulting dynamics exhibit a *quadratic* spectral implicit bias — per-mode displacement of the Gram matrix scales as the square of the corresponding NTK eigenvalue, distinct from the linear-in-eigenvalue rates known for scalar NTK regression — which pins down the effective hypothesis class on bounded training horizons: target structures aligned with the kernel's top eigenspace are reachable, while those concentrated in its bottom eigenspace remain near initialization. We verify the slope $\leq 2$ bound across synthetic data, Fashion-MNIST, frozen-feature CIFAR-10, and an end-to-end CNN trained from scratch. The same analysis prescribes a one-line algorithmic correction — applying the inverse kernel to the embedding-side gradient — which improves hard-negative retrieval but not easy retrieval, exactly the regime where the theory predicts an effect.
```

---

## PDF*

Upload: `kbm-spectral-bias/paper/main.pdf` (built 2026-05-05).
Already anonymized, already includes the NeurIPS checklist on its final pages.

**Checklist Confirmation\***: ✅ check the box ("I confirm that I have
included a paper checklist in the paper PDF").

---

## 🔒 Contribution Type* (LOCKS at submission)

**Choice: `Theory`**

NeurIPS 2026 contribution types (verified from the official
neurips.cc blog post on contribution types and the Main Track
Handbook):
1. **General** — most submissions
2. **Theory** — "main contribution is via theoretical analyses and proofs"
3. **Use-Inspired** — real-world application framing
4. **Concept & Feasibility** — high-risk/high-reward, preliminary results
5. **Negative Results**

Reasoning for picking **Theory**:
- Three formal theorems are the headline (kBM equivalence; depth-L
  extension; quadratic spectral implicit bias), plus a corollary on
  effective rank. Match for "main contribution is via theoretical
  analyses and proofs."
- Experiments verify a theory-derived prediction (slope-$\leq 2$ across
  multiple regimes; asymmetric Recall@$K$ under K$^{-1}$), not a
  benchmark race.
- Reviewers under "Theory" are given theory-specific criteria
  (correctness, non-triviality, scope of theorems) rather than being
  asked for SOTA-level empirical performance.
- Picking "General" would risk reviewers expecting CUB/Cars/SOP-scale
  benchmark comparisons, which is the wrong framing for this paper.

---

## 🔒 Primary Area* (LOCKS at submission)

NeurIPS 2026 uses a **flat list of 21 area categories** (verified from
the official Call for Papers at neurips.cc/Conferences/2026/CallForPapers).
There is **no nested "Deep Learning → Theory" subcategory**. The
relevant top-level options for this paper are:

- **Theory**
- **Deep learning**
- **Optimization**
- **Generalization and multi-task learning**
- **General machine learning: core contributions in supervised and unsupervised methods**

**Choice: `Theory`** (priority 1).

Reasoning:
- The headline contribution is three formal theorems (kBM equivalence,
  depth-$L$ extension, quadratic spectral implicit bias) plus a
  corollary on effective rank. This is the only category that signals
  "theorem paper" to ACs and routes the submission to theory reviewers.
- Picking "Deep learning" would pool the paper with empirical
  deep-learning submissions where reviewers would expect SOTA-level
  benchmark gains — wrong framing.
- Picking "Optimization" would route to optimization theorists who
  may not recognise the NTK/wide-network angle.

---

## 🔒 Secondary Area* (LOCKS at submission)

Same flat list of 21 categories.

**Choice: `Deep learning`** (priority 1).

Reasoning:
- The kBM equivalence is specifically a wide-network NTK result;
  "Deep learning" pulls a reviewer who recognises NTK, Burer-Monteiro,
  and implicit-bias work.
- Backup second-area choice if you want optimization expertise instead:
  **`Optimization`** — the Burer-Monteiro side is a legit
  optimization-dynamics insight, and a BM/SDP-aware reviewer would
  recognise the primal-Gram kBM contribution.

If the form requires a third area, **`Generalization and multi-task
learning`** is the closest match for the effective-hypothesis-class
corollary.

---

## 🔒 Reviewer Nomination* (LOCKS at submission)

```
~Ali_Saffarini1
```

(Replace with your exact OpenReview ID if it differs — copy from your
profile URL.)

Reasoning: solo-author paper. The handbook's bar is "at least 2
first-author OR at least 5 co-authored peer-reviewed publications". If
you don't yet meet that bar, the handbook's escape clause covers you:
"Only if no qualified reviewer exists in the author list, nominate the
best-qualified author for consideration by the PC chairs or the E&D
track chairs." With one author, that's you regardless. The PCs handle
the qualification check on their end.

---

## Financial Support (optional)

```
~Ali_Saffarini1
```

If you'd need financial support to attend and present at NeurIPS 2026
should the paper be accepted. Leave blank otherwise. (You can still
fill out the separate Financial Aid application later — this field is
just the per-paper nomination.)

---

## Responsible Reviewing*

✅ check the box ("We acknowledge the responsible reviewing
obligations as authors").

---

## Academic Integrity*

✅ check the box ("I acknowledge that I have read the NeurIPS Handbook
and commit to adhering to all policies...").

---

## LLM Usage* (confidential — for PC stats only, not shown to reviewers)

The exact dropdown options aren't visible, but typical NeurIPS LLM
disclosure choices include things like: writing/editing assistance,
code generation, idea generation, literature search, none. **Be honest
and select all that apply.** Per the handbook, basic editing/spell-check
and basic code assistance do NOT need to be documented in the paper
itself, but this form is for PC analytics — answer accurately.

For your case (Claude Code throughout the paper-prep cycle):
- ✅ writing / editing assistance
- ✅ code generation / assistance
- ✅ math/proof checking (audit pass) if such a category exists

**Other LLM Usage** (free-text, optional): if you want to be precise,
you can write something like:

```
Used Claude Code (Anthropic) throughout for: drafting and revising prose, sub-agent math audits of the proofs, code generation for experiments, and figure preparation. All theorems, proofs, and experimental claims were verified by the author.
```

---

## LLM Experiment (opt-in)

**Recommendation: leave UNCHECKED.**

The paper is math-heavy. Opting in means some of your reviewers may use
a custom LLM-assisted reviewing interface, and dense theorem/proof
content is a known weak spot for current LLM assistants. Default
(unchecked) keeps reviews human-driven.

(Tradeoff to be aware of: if you opt in, you may also become eligible
to participate as a reviewer in the experiment. Not a big factor for a
solo first submission.)

---

## Ready For LLM Feedback (Google PAT)

**Recommendation: ✅ check the box** if the program is still accepting
requests.

This is *free, author-only feedback* from Google's Paper Assistant
Tool. It is NOT shared with reviewers. The form note says "The program
ends on the abstract deadline May 4th (Anywhere on Earth). High demand
may create processing delays" — so the queue may be closed already, but
checking the box costs nothing if it isn't.

---

## License*

**Recommendation: `CC BY 4.0`** (Creative Commons Attribution 4.0).

Standard, permissive, NeurIPS-compatible. If the dropdown has multiple
CC variants, CC BY 4.0 is the default. Avoid CC BY-NC unless you have
a specific reason to restrict commercial reuse.

---

## Declaration*

✅ check the box ("I confirm that the above information is accurate").

---

## Readers* / Signatures*

These auto-populate based on your authorship. Nothing to fill in
manually — the form sets these for you.

---

## Notes for after submission (full-paper deadline window)

The PDF and most metadata can be updated until the full-paper deadline.
Things you can still iterate on:

- Refine the paper PDF (continue gap-filling, polish proofs).
- Add supplementary material (anonymized code ZIP, ≤100 MB).
- Update the title or abstract (minor edits only — they should not
  substantially differ from what was submitted today).
- Update LLM Usage / Other LLM Usage as needed.

Things you **cannot** change after today's submission:
- Reviewer Nomination
- Primary Area
- Secondary Area
- Contribution Type
- Author list (only reordering allowed; no additions or removals)

---

*Last updated 2026-05-05 by author for NeurIPS 2026 abstract-registration submission.*
