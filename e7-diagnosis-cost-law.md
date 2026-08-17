# E7 diagnosis: what actually contradicts the cost law, and what does not

Date: 2026-08-14, written the day E7 landed and was scored. P7.1, P7.2 and
P7.3 FAILED as registered. Those verdicts stand and will be reported. This
document is the diagnosis: every claim in it is computed from the landed data
or from the exact machinery, and each analysis is labelled registered or
post-hoc. The registered instrument failed; the question is what the failure
is evidence OF.

## 1. The registered failure, restated precisely

P7.1 asked the measured half-collapse of excess cost to land within 0.01 of
lambda_half on four instances. Measured: 0.928 vs 0.997, 0.868 vs 0.985,
0.926 vs 0.958, 0.863 vs 0.873. One of four inside tolerance. P7.2's ordering
failed because the 0.02 and 0.10 instances came out reversed. P7.3's plateau
came out 42 to 181 percent off on three instances and 3.6 percent on one.

## 2. The decomposition, post-hoc, labelled as such

P7 conflates two claims that the E6 record had already separated:

  (a) the COST LAW: the function J(k), including its plateau, and the map
      lambda -> k_AC(lambda) -> J(k_AC);
  (b) the CONVERGENCE ASSUMPTION: 15M steps of PPO delivers the policy TO
      k_AC(lambda).

E7's data refutes (b), not (a), and it does so in a way (a) survives being
evaluated on. Two facts, computed per cell:

**The below-horizon runs never arrive.** In every below-knee cell of every
instance, all five seeds stall far short of k_AC, tightly clustered: 6 to 50
percent of the target (g1_h0.02 at 0.8: gains 1.3 to 9.2 against k_AC 60;
g4 at 0.8: gains 2.09 to 2.29 against 14.5). This is the same
optimiser-limited transport measured directly in the linear-arch pilot on
2026-08-14: the expected gradient along the gain is orders of magnitude too
small to cover the distance in the budget.

**Evaluated at the gain each run actually reached, the law holds.** For all
160 E7+E6 coupled runs, compute J_measured / J_analytic(k_measured), the
"policy is its own linear fit" ratio, with J_analytic from the exact machinery
(calibrated: reproduces every frozen J_ml to 1.00000):

| cells | n | median ratio | range |
|---|---|---|---|
| no intermittency | 23 | 0.994 | 0.991 to 1.200 |
| intermittent | 9 | 1.699 | 1.056 to 7.325 |

22 of 23 clean cells lie within 10 percent of the law. The one exception at
1.20 is g1_h0.05 lambda 0.98, the documented regime-2 cell, elevated but
stable, consistent across seeds (1.08 to 1.27). Every cell that misses the
law badly is a cell with off-manifold excursions, and the excess is the
excursions, which is the paper's other subject.

So nothing in E7 contradicts J(k). What E7 contradicts is the assumption that
the training endpoint sits at k_AC below the horizon, an assumption already
falsified twice in this project's record (round-2 approach instability; A-5
off-manifold), and which the registered estimator inherited silently.

**Why the estimator then failed.** measured_lambda_half anchors its midpoint
on the max and min cells. The max cell below the knee is set by intermittency
contamination, which is non-monotone in lambda (g1_h0.05 excess by lambda:
7.03, 2.62, 4.40, 1.30, ...). The estimator located the first large dip, not
the half-collapse. Reconstructed by hand, its reported values are reproduced
exactly by this mechanism on every instance. And g4's "pass" is not evidence
for the law either: its whole excess range is 1.15 to 1.00, so there was
almost no dynamic range in which to misplace the midpoint. P7.1 was not a
well-posed measurement on this grid. That is a design failure of E7, and it is
reported as one.

## 3. The mechanism, scored on E7

The frozen bracket predictions (prereg_e7_bias.py, frozen 2026-08-14 before
E7 existed) score: g2_h0.05 HIT (0.9, 0.95), g4_h0.05 HIT (no onset anywhere,
observed zero intermittency in all 8 cells), g1_h0.02 MISS (predicted
(0.995, 0.999), observed (0.98, 0.99)). Two of three prospective. The
ordering of measured onsets in gamma*h holds, and every onset sits strictly
below its instance's lambda_half, the frozen two-threshold structure.

Post-hoc, labelled: the dynamically correct question is not the sign of S at
k_AC, which the runs never reach, but whether the climb path toward k_AC
passes through the unstable region. Scoring S_min = min over k in
[k_ml, k_AC] of S(lambda, k) against observed intermittency, all 32 cells:

**30 of 32 concordant.** Both misses are the two cells with the smallest
predicted growth rates among all path-unstable cells (|S_min| = 1.0e-4 and
1.8e-5), and the first of them shows intermittency 0.020, nonzero but under
the 0.05 threshold: rare excursions, exactly what a marginal rate predicts at
finite time. Among path-unstable cells, log |S_min| correlates 0.70 with the
observed intermittency fraction. The g1_h0.02 bracket miss dissolves under
the same lens: its S at the measured gains is negative (unstable) at 0.98 and
marginal at 0.99, matching what was seen.

## 4. E10, the experiment P7 should have been

Frozen today in prereg_e10.py, before any E10 row exists, with scoring in the
file. Two arms, same harness, same seeds, policy placed AT k_AC by
construction so no reachability excuse survives:

- **linear** (u = -k v, gain-initialised at k_AC): P11.1 the equilibrium holds
  (all 32 cells within 15 percent), P11.2 the cost law (all 32 cells within
  10 percent of J(k_AC)), P11.3 lambda_half within 0.01 of the GRID-MATCHED
  analytic target per instance, targets frozen: 0.996742 / 0.983837 /
  0.961432 / 0.855871. Grid-matching separates measurement error from
  interpolation bias, which section 2 shows is not ignorable on this grid.
**Reachability, measured directly rather than inferred.** The linear arm's
design is forced by a pilot run on 2026-08-14 in this harness, g1_h0.05,
lambda 0.9, gain initialised at zero: the gain reaches 0.026 after 200k steps,
0.149 after 1M, and 0.218 after 1M with the gradient scale raised twentyfold.
The target is k_AC(0.9) = 20.547. A one-parameter policy under Adam moves its
gain far more slowly than an MLP whose effective gain is a sum over 128
weights, and scaling buys a factor of 1.5 rather than 20. Starting from zero
therefore cannot test the cost law below the horizon in any feasible budget,
in this harness or in E7's. Initialised AT k_AC instead, the same policy holds
it: gain 20.468 after 1M steps (drift 0.4 percent) with J_rate 0.2796 against
an analytic 0.292074, and at lambda 0.99, gain 2.122 with J_rate 0.2110
against 0.222898. That is why E10 places the policy at the equilibrium by
construction: it converts an untestable claim into a testable one.

- **linearb** (u = -k v + b, bias KICKED to b0 = 1.0): perturbation-decay
  reads the bias eigenvalue sign directly. 29 scored cells: 8 predicted
  unstable (|b| stays above 0.5), 21 predicted stable (decays below 0.5),
  3 excluded by a frozen rate floor |S| >= 1e-4. Pilot, g1_h0.05: at
  lambda 0.9 the kick GROWS, 1.00 -> 1.38 by 3M steps; at 0.995 it decays,
  1.00 -> 0.65 by 1M. g4 predicts stable everywhere; any instability there is
  a hard failure.

192 tasks, about 45 minutes each. Submission: `bash myriad/deploy.sh ucapgoz
--e10`. The branch runs the freeze and clean checks before submitting.

## 5. What the paper says if E10 lands, and if it does not

If P11.1-P11.4 pass: the audit's arc becomes "a registered instrument failed;
the failure was diagnosed to a specific invalid assumption; the law and the
mechanism were then confirmed prospectively in-domain, including a predicted
28-cell stability pattern and a predicted quantitative curve on four
instances." That is a stronger paper than P7 passing quietly would have been,
and every step of it is dated and checkable.

If P11.2 or P11.3 fail: the cost law is wrong somewhere reachability cannot
excuse, and the paper reports that. If P11.4 fails beyond its allowance: the
bias-direction mechanism is not the driver at k_AC and Tier 2 is demoted to a
suggestive calculation. The registered failures of P7 are reported in the
paper in full either way.
