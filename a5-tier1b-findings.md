# A-5 Tier 1b: the instability is off-manifold

Date: 2026-08-13. Predictions frozen before any run (section 1, verbatim).
Verdict: H_B supported, H_A not supported, on the frozen criteria. Two
implementation bugs were found and fixed en route and are reported in full.

## 1. The frozen predictions, written before running

    Discriminator: pure linear-gain policy u = -k*v (single scalar, no bias,
    no hidden layer, no learned sigma) vs the E6 MLP policy, lambda 0.9 vs 0.99.

    H_B (off-manifold policy instability): the linear policy CANNOT leave the
      gain manifold. The gain axis alone is stable at every lambda. So it shows
      NO late-half intermittency at 0.9, and cost tracks the analytic J(k) of
      its own gain.
    H_A (actor-critic loop oscillation): the oscillation lives in the coupled
      policy-critic dynamics, which the linear policy still has. Excursions
      PERSIST at 0.9 and are absent at 0.99.

    SCORING (late half, cost = -eval_return, J_ml = 0.2196):
      intermittency := fraction of late-half points > 2 * J_ml
      H_B supported : mean intermittency at lam 0.9 < 0.05
      H_A supported : mean intermittency at lam 0.9 > 0.25
      ambiguous     : between 0.05 and 0.25
      Control required either way: lam 0.99 intermittency < 0.05
      Secondary: J_measured / J_analytic(fitted gain). H_B predicts ~1.

## 2. Results

Linear policy, instance g1_h0.05, sigma pinned at s_e = 0.09, 3M steps.

| lambda | final k | predicted k_eq | intermittency | late cost median | late cost max |
|---|---|---|---|---|---|
| 0.9 | 19.20 | 20.55 | **0.000** | 0.2863 | 0.2866 |
| 0.99 | 3.00 | 2.20 | 0.000 | 0.2312 | 0.2324 |

Intermittency 0.000 at lambda 0.9, against 0.492 for the MLP in E6. The
control passes. The secondary prediction passes: analytic J at k = 19.2 is
about 0.290 against a measured 0.2863, a ratio of 0.99, so the linear policy
IS its own linear fit, exactly as it must be.

Critic-lag control, which is the part that makes this a discriminator rather
than a demonstration. Lambda 0.9, the only knob being how fast the critic
tracks its own fixed point:

| critic lag | final k | late k range | intermittency | late cost max |
|---|---|---|---|---|
| 1.0 (none) | 19.00 | 17.49 to 19.00 | 0.000 | 0.2864 |
| 0.3 | 19.43 | 18.37 to 19.43 | 0.000 | 0.2869 |
| 0.1 (strong) | 19.18 | 17.92 to 19.18 | 0.000 | 0.2866 |

A strongly lagging critic, which is precisely H_A's proposed mechanism,
produces no intermittency at all when the policy is confined to the gain
manifold.

## 3. What this separates

Two effects that E6 could not distinguish are now distinct.

**The aliased critic displaces the equilibrium.** Fully present in the linear
policy: it converges to k = 19.2 rather than k_ml = 1.649, an order of
magnitude of displacement, landing within 6.6 percent of the predicted
k_eq(0.9) = 20.55. The mechanism of the original Props is intact and doing
its work with only one policy parameter.

**The intermittency requires policy degrees of freedom beyond the gain.**
Removing them removes the instability entirely, while leaving the
displacement untouched. Critic lag does not substitute for them.

So the paper's claim sharpens to: partial observability biases the learning
fixed point through the aliased critic, and separately, in policy classes
richer than the displaced coordinate itself, the displaced fixed point
becomes unreachable through an instability in the remaining directions.

## 4. Two bugs found and fixed, reported because they nearly produced a false result

**Bug 1, a silently dead learner.** The PPO clip selection used strict `<`.
On the first epoch the ratio is exactly 1, so `g1 < g2` selected nothing, the
gradient was identically zero, k never moved, and the ratio stayed exactly 1
forever. Every run returned final gain 1.000 with intermittency 0.000, which
reads as clean H_B support. Fixed to `<=`. Any harness with a clipped
surrogate should be checked for this.

**Bug 2, an absent critic.** The critic was fit by a few small gradient steps
from zero and stayed at roughly zero, so the aliasing mechanism the
experiment exists to probe was not present. Replaced with iteration to the
least-squares fixed point, with the lag knob above added so the lag could be
controlled rather than accidental.

## 5. Limitations, which are real

- **This is not the E6 harness.** It is a numpy reimplementation: a quadratic
  critic in (1, v, v^2) fitted by least squares, different learning-rate
  schedule, a scalar parameter rather than a network. The comparison against
  E6's MLP therefore crosses two implementations, and that is a confound.
  The clean version is a `--arch linear` option inside ppo_lambda.py, same
  critic, same optimiser, same everything, only the policy head differing.
  That is an afternoon of work and it should be done before the claim goes in
  the paper.
- The critic here is quadratic by construction, so it cannot itself go
  unstable in the way a neural critic can. That is part of what is being
  tested, but it does mean H_A is refuted only for a critic in this class.
- One seed per configuration, 3M steps, one instance. Sufficient for a
  discriminator whose predicted outcomes are 0.000 against 0.49, not
  sufficient for a paper table.
- The lambda 0.99 endpoint lands at k = 3.00 against a predicted 2.20. The
  equilibrium is shallow there, so this is unsurprising, but it is a 37
  percent discrepancy and should not be quoted as agreement.

## 6. What to do next

1. Add `--arch linear` to ppo_lambda.py and rerun the discriminator inside
   the real harness, 3 seeds, both lambdas. This converts a preliminary
   result into a paper-grade one and removes the cross-implementation
   confound. It also gives the MLP-versus-linear comparison at matched
   everything, which is the actual figure the paper wants.
2. With H_B selected, the theory to attempt is the stability of the non-gain
   policy directions under the aliased critic. In this model the natural
   first target is the action-bias direction, where the expected update
   vanishes at b = 0 by parity, so the question is the sign of its derivative
   there. That is exactly computable with the Wick machinery already built,
   extended to nonzero means.
3. The mechanism must eventually predict the LOCATION of the instability
   onset, bracketed by E6 between 0.95 and 0.98, with four more brackets
   arriving from E7.
