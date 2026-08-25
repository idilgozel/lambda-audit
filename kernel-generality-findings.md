# Kernel generality: the thresholds are not a 2-D artifact

Written 2026-08-21. Deterministic computation only — no seeds, no training, no
stochastic data; every number regenerates from
`vendor/gleam/kernel_general.py`, which refuses to print a multi-mode value
unless it first reproduces the frozen single-mode tables of `prereg_e10.py`
(it does: k_AC(λ) to relative 6×10⁻⁷ or better at all eight grid points, S to
6×10⁻⁸, grid λ_half to 4×10⁻⁸).

## Why

The audit's experiments all run on the single-exponential member of the GLE
family — state dimension 2 — and the natural objection is that the phenomena
are artifacts of a two-dimensional linear-quadratic toy. The derivations never
used the dimension: `lstd_lambda.py`'s resolvent identities and the S-index of
`prereg_e7_bias.py` are written over an arbitrary closed loop, and only the
module that builds the closed loop pinned N = 1. `kernel_general.py` rebuilds
the same three exact objects — the LSTD(λ) fixed point, the expected GAE(λ)
gradient, and the off-manifold stability index — for any Prony memory kernel
K(τ) = Σᵢ cᵢ e^(−γᵢτ), i.e. hidden dimension N and non-Markovian observed
dynamics for N ≥ 2.

## Result

Five kernels, K(0) = 0.3 held fixed, h = 0.05, σ_e = 0.3:

| kernel | state dim | τ_K | k_AC(0.8) | k_AC(1.0) | collapse | λ_half | λ* |
|---|---|---|---|---|---|---|---|
| single (paper) | 2 | 1.000 | 22.427 | 1.507 | 14.9× | 0.9838 | 0.9759 |
| 2-mode balanced | 3 | 1.450 | 21.651 | 1.525 | 14.2× | 0.9833 | 0.9809 |
| 2-mode slow-tail | 3 | 1.250 | 20.911 | 1.406 | 14.9× | 0.9644 | 0.9686 |
| 3-mode broad | 4 | 1.556 | 21.724 | 1.528 | 14.2× | 0.9835 | 0.9822 |
| single fast | 2 | 0.500 | 20.077 | 1.267 | 15.8× | 0.9614 | 0.9010 |

**What is universal.** Every kernel, at every state dimension computed, shows
the identical structure: an inflated aliased equilibrium below a knee, a
14–16× collapse of the learning target across the λ grid, a cost-law
threshold λ_half, and a stability threshold λ*. The confound and both of its
critical values are properties of the memory kernel, not of the 2-D instance
the experiments happen to sample.

**What is not universal — two honest surprises, reported as computed.**

1. The file's header predicted, before the sweep ran, that λ_half would order
   by the kernel correlation time τ_K = ∫K/K(0). It does not: the slow-tail
   kernel (τ_K = 1.25) has λ_half = 0.9644, below both the single-mode kernel
   at τ_K = 1.0 and the balanced kernel at 1.45. Within the single-mode
   family, 1/γ orders everything (that is P10.2, which passed); across kernel
   *shapes*, no single timescale scalar does. The effective timescale the
   critic responds to is a property of the whole kernel, not its first moment.
   This sharpens, and bounds, the paper's τ-based reasoning — and it is
   consistent with how E12's τ-from-episode-length bridge failed.

2. The two-threshold ordering λ* < λ_half — true in all four experimental
   instances and the premise of P10.3 — is **not kernel-universal either**:
   the slow-tail kernel inverts it (λ* = 0.9686 > λ_half = 0.9644). The
   stable-but-biased window that exists throughout the single-mode family can
   close and invert for kernels with a slow minority mode. P10.3's measured
   failure (the gap did not widen with γh) sits more comfortably beside this
   than beside the original claim.

## What this licenses in the paper

- "The phenomenon is a property of learning under any Prony memory kernel;
  experiments sample its minimal member" — stated as computation, with the
  regenerating script released. State dimensions 2–4 shown; the formulas
  carry no dimension restriction.
- It does NOT license claims beyond linear-Gaussian dynamics with a scalar
  observation and quadratic cost. The nonlinearity question is separate and
  is addressed by the E13 robustness run (its preregistration, when frozen,
  governs).
- Together with the bath-universality computation in the parent GLEAM project
  (lemma-a-findings.md, 2026-07-23: equilibrium location moves 3.4× across
  distribution-matched baths while cost at equilibrium stays within 1–7% of
  r·Θ·K(0)), the theory now separates its own predictions into
  distribution-robust (cost levels, thresholds) and fragile (equilibrium
  locations) classes — a split the audit's pass/fail record matches: every
  failed location prediction (P6.1, P11.1, P10.1), every passed cost/threshold
  prediction (P6.2, P11.3, and the ≤1.05% cost-map agreement in all 32 E10
  cells).

## Regenerating

    python3 vendor/gleam/kernel_general.py

Requires numpy/scipy (run in an environment with both, e.g. conda base; the
validation gate aborts the sweep if the frozen tables do not reproduce).
