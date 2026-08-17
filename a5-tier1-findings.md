# A-5 Tier 1 findings: the abort criterion fired, and it pointed somewhere

Date: 2026-08-13. Executed by hand per the scope in
a5-stochastic-stability-scope.md. Verdict up front: the Kramers-escape-in-k
story is falsified by exact computation, the pre-registered abort criterion
is honoured, and Tier 2 as designed does not run. The failure localises the
true mechanism: the below-horizon instability does not live on the gain
axis at all.

## 1. Machinery validated first

- lstd_lambda reproduces Props 1 to 3 exactly at lambda 0 (theta2 13.169491,
  ghat -0.143898, both to the last digit).
- k_eq(lambda) reproduces the round-3 record: 24.794 / 20.547 / 17.671 /
  2.197 / 1.549 at lambda 0 / 0.9 / 0.95 / 0.99 / 0.999.
- The Monte Carlo estimator's mean matches gae_gradient to 1.1% away from
  equilibrium and is statistically zero at the equilibria.

## 2. The landscape (instance g1_h0.05, s_e = 0.09)

Divergence edge k_max = 39.9975, lambda independent.

| lambda | k_eq | slope at k_eq | barrier to edge |
|---|---|---|---|
| 0.8 | 22.43 | 1.6e-04 | 0.669 |
| 0.9 | 20.55 | 1.2e-04 | 0.758 |
| 0.95 | 17.67 | 7.4e-05 | 0.809 |
| 0.98 | 11.39 | 3.4e-05 | 0.843 |
| 0.985 | 7.21 | 1.4e-05 | 0.849 |
| 0.99 | 2.20 | 5.5e-04 | 0.856 |
| 0.995 | 1.76 | 1.4e-03 | 0.864 |
| 0.999 | 1.55 | 2.1e-03 | 0.871 |

Two facts. The barrier is nearly flat in lambda, so the potential alone
cannot produce a sharp stability transition. And the local restoring force
is softest exactly at the knee (1.4e-05 at 0.985, 150x weaker than at
0.999), a critical-slowing-down signature worth keeping for the paper.

The noise is strongly anisotropic, as conjectured: D falls three orders of
magnitude from k = 2 to k = 30 because the state variance collapses under
high gain, then blows up near the divergence edge. At the equilibria, D is
260x LARGER at lambda 0.99 (k_eq 2.2) than at 0.9 (k_eq 20.5).

## 3. The abort check failed under every noise model

Kramers action Phi = integral of 2 ghat / D from k_eq to the edge, computed
under three treatments of the noise:

| lambda | flat D | anisotropic D | advantage-normalised |
|---|---|---|---|
| 0.8 | 47.3 | 108.2 | 15.6 |
| 0.9 | 31.8 | 53.9 | 12.3 |
| 0.95 | 20.9 | 34.7 | 11.9 |
| 0.99 | 8.2 | 12.5 | 9.4 |
| 0.999 | 3.1 | 4.8 | 7.4 |

Escape resistance RISES as lambda falls, in all three models. The data
demands the opposite: instability at 0.8 to 0.95, stability at 0.99 and
above. Wrong direction, everywhere.

Independent second falsification: at lambda 0.9 under the measured
anisotropic noise, Phi down from the equilibrium to low gain is 13.5 against
Phi up to the edge of 53.9. Diffusion along the gain axis should fall toward
LOW gain four times more easily than it escapes upward. The observed
excursions are toward catastrophic cost.

Per the scope's pre-registered rule, this kills the story: the below-horizon
instability is NOT noise-driven escape of the gain coordinate over a
quasi-potential barrier.

## 4. Where the mechanism actually lives

The decisive number was already in E6. Ratio of measured deployed cost to
the analytic cost of the run's own fitted gain:

| lambda | per-seed ratios |
|---|---|
| 0.8 | 2.81, 4.21, 7.33, 7.83, 8.05 |
| 0.9 | 1.19, 1.22, 1.79, 3.63, 3.83 |
| 0.95 | 1.01, 2.92, 3.93, 4.37, 14.69 |
| 0.98 | 1.08, 1.09, 1.20, 1.24, 1.27 |
| 0.99 to 1.0 | 0.97 to 1.02, every seed |

Above the horizon the trained policy IS its linear fit, to two percent.
Below it, two to fifteen times the cost lives in policy directions the gain
coordinate does not see. The 1D reduction to k is valid above the knee and
invalid below it, which is precisely why the Kramers calculation had to
fail. The instability is off-manifold.

## 5. Two surviving hypotheses, and the experiment that separates them

**H_A, actor-critic oscillation.** The averaged analysis holds the critic at
its fixed point. In PPO the critic lags, and below the horizon theta2 is
large, so the coupled policy-critic loop can be oscillatory-unstable even
where the reduced 1D flow is stable. Predicts recurrent departure and
partial return, which matches the observed shape.

**H_B, off-manifold policy instability.** The aliased critic destabilises
policy directions outside the pure gain, such as the action bias and the
MLP's nonlinear components. The ratio table is direct evidence that the
damage lives there. At b = 0 the expected bias update vanishes by parity;
whether it is stable is exactly computable with the existing Wick machinery
extended to nonzero means.

**The discriminating experiment costs an afternoon.** Train a pure linear
gain policy, u = -k v with no bias and no hidden layer, same PPO, same
instance, lambda 0.9 against 0.99, a few seeds at modest budget. H_B says
the below-horizon excursions vanish, because the policy cannot leave the
manifold and the gain axis alone is stable by section 3. H_A says they
persist, because the oscillation lives in the policy-critic loop. Either
outcome converts a falsified story into a located mechanism.

## 6. What survives for the paper regardless

- The intermittency diagnostic and the E6 dissociation are phenomenology and
  stand independent of mechanism.
- The paper can now state, from exact computation: one natural explanation,
  diffusion of the gain over a shallow basin, is quantitatively ruled out.
  Referees will propose it; the answer is already in hand.
- The critical softening at the knee (section 2) is a clean, reportable
  signature on its own.
- Language rule holds: fixed point of the averaged update, never
  "converges to", and below the horizon "training does not converge and its
  quality is stopping-time dependent".

## 7. Timebox status and ask

Tier 1 consumed roughly one day of the one-week box and ended at the abort
gate, as designed. Proposed Tier 1b, bounded at two to three days: the
linear-policy discriminating run, then whichever of H_A (two-timescale
linearisation, three coupled equations) or H_B (bias-direction stability via
Wick with nonzero means) the run selects. Both are exactly analysable in
this model. Awaiting sign-off before spending it.
