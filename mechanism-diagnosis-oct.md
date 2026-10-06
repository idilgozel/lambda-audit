# Mechanism (S-index) misses, re-examined with the exact machinery — 2026-10-05

Status: **post hoc, no confirmatory weight.** Written after P10.1/P10.3 failed as registered
(`prereg_e7_bias.py --score`). Nothing here rescues a verdict; it records what the E7 data say
about why the registered location missed, so the paper's one-paragraph treatment of the
demoted mechanism is accurate.

## What the E7 cells actually look like (mlp, from scratch, 5 seeds, IQM)

| instance | λ | gain IQM (min–max) | E = J/J_ml | late evals with cost > 2 J_ml, per seed |
|---|---|---|---|---|
| g1_h0.02 | 0.8 | 3.4 (1.3–9.2) | 2.78 | 0.68 0.65 0.55 0.40 0.22 |
| | 0.9 | 4.9 (1.1–5.7) | 3.10 | 0.73 0.87 0.52 0.89 0.43 |
| | 0.95 | 4.1 (1.1–8.3) | 1.23 | 0.00 0.57 0.30 0.71 0.21 |
| | 0.98 | 14.9 (7.7–29.4) | 2.17 | 0.00 0.04 0.21 0.24 1.00 |
| | 0.99 | 19.7 (4.4–31.9) | 1.46 | 0 0 0 0.10 0 |
| | 0.995 | 2.9 (2.0–24.1) | 1.06 | all 0 |
| g1_h0.05 | 0.8–0.95 | 4.1–5.9 | 2.6–7.0 | 0.36–0.77 |
| | 0.98 | 4.2 | 1.30 | all 0 |
| g2_h0.05 | 0.8, 0.9 | 10.0, 7.5 | 1.97, 2.04 | 0.28–0.72 |
| | 0.95 | 1.9 | 1.04 | all 0 |
| g4_h0.05 | 0.8 | 2.2 | 1.15 | all 0 |

Predicted plateau gains k_AC(λ) at these cells are 16–60. Three facts follow.

1. **Below the knee the from-scratch mlp never sits at k_AC.** Gains are 2–10 against predicted
   15–60, exactly the approach instability `prereg_phase2.py` recorded before E6 ran (gains of
   10–19 destabilising before 20.6). The cost is nonetheless far above the plateau (E = 2.6–7
   against a predicted plateau of 1.35) because the runs thrash: this is why P7.3 (plateau
   level) failed, and why P7.1 (the mlp's own knee) failed as well -- the mlp's grid lambda_half
   reads 0.868-0.928 against 0.958-0.9966 predicted, because the thrashing cost rather than
   the plateau sets the midpoint (`prereg_phase2.py` output, 2026-10-05 re-run). The knee was
   located only by the linear arm started at k_AC (P11.3); see e17-pilot-findings.md for what
   that arm can and cannot resolve.
2. **The intermittency statistic is contaminated where the plateau itself exceeds 2 J_ml.**
   On g2_h0.05 at λ = 0.8/0.9 the deployed E is 1.97–2.04, so "cost > 2 J_ml" is the plateau's
   own noise, not an excursion; the measured onset (0.9, 0.95) there coincides with the cost
   collapse (λ½ = 0.958) and does not independently measure off-manifold escape. The criterion
   was frozen with an absolute multiple; a multiple of the cell's own median would have been
   the right instrument. Reported, not rescored.
3. **Evaluating S at the measured gain does not rescue P10.1.** S(λ, k_meas) agrees with the
   intermittency flag on 25/32 cells; the seven disagreements are all intermittent cells whose
   IQM gain (4–10) sits where the single-bias direction is *stable*. Either another off-manifold
   direction goes first — the asymmetry the prereg itself anticipated ("if some other direction
   is more fragile … the mechanism is incomplete, not wrong") — or the instability is a
   transient of the climb toward the plateau rather than a property of any fixed point. The
   data cannot separate these; the mechanism stays demoted.

## Two observations the fixed-point theory does make exact (see knee-formula-findings.md)

- g1_h0.02 is first-order (bistable in λ ∈ [0.99529, 0.99655]). At λ = 0.995, just below the
  window, 4 of 5 from-scratch seeds sit on the *low* branch (gain 2.0–3.5, E 1.06) while the
  largest-root theory says 25.5 (E 1.33); one seed sits at 24.1. The grid-matched λ½
  estimator is blind to this because the midpoint falls between 0.99 (E 1.46) and 0.995, so
  P11.3's pass on this instance stands but says nothing about the branch structure. From-scratch
  training appears to find the low branch somewhat below the fixed-point window.
- The S index was evaluated at k_AC(λ), the largest root. For a bistable instance the b-stability
  of the *lower* branch is what a from-scratch run experiences; this does not change the P10.1
  verdict (g1_h0.02's onset is below the bracket either way) and is noted for completeness.

## Consequence for the paper

Keep §3.5 as a suggestive calculation, one paragraph; move the S auxiliaries to the appendix.
The sentence to add, if any: the location prediction assumed training sits at the plateau
equilibrium, and the data show it does not — the mechanism's onset is a statement about a
state the learner does not visit.
