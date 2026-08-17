# The seed-5 anomaly is node heterogeneity, not a mapping bug

Date: 2026-08-14. Written after E9 landed and before E7 was resubmitted.
Verdict up front: the task-to-(config, seed) mapping is correct. The
non-reproducing runs are explained completely by which Myriad node class they
landed on, and no registered verdict depends on them.

## 1. What was observed

E9 re-ran 40 PositionOnlyCartPoleEasy cells that the pilot had already run at
identical hyperparameters: arch in {mlp, gru128}, lambda in {0.9, 0.95, 0.99,
1.0}, seeds 1 to 5. Comparing per cell, 33 reproduced to the last bit and 7 did
not, and all 7 sat at seed 5 or seed 4. Clustering that sharp usually means an
index offset, so the first hypothesis was that one of the two runs had a
seed-to-task mapping shifted by one.

## 2. That hypothesis is wrong

A mapping offset cannot leave 33 of 40 cells bitwise identical. The decisive
evidence is elsewhere: the seven non-reproducing runs are exactly the seven
fastest pilot runs, once wallclock is normalised within architecture. That
normalisation is required, not cosmetic: a gru128 run takes about eight times
an mlp run, so raw seconds are not comparable across the two and the seven
fastest in raw seconds are simply the seven mlp runs.

| arch | reproduced | pilot wallclock, as a fraction of that arch's median |
|---|---|---|
| mlp | yes, 18 runs | 0.861x to 1.225x |
| mlp | no, 2 runs | 0.661x and 0.716x |
| gru128 | yes, 15 runs | 0.930x to 1.167x |
| gru128 | no, 5 runs | 0.607x to 0.712x |

Within each architecture the two groups do not overlap and there are no ties.

Reading the host line out of each job log closes it. Every one of the seven ran
on a `node-t00a-*` host in the pilot. Every one of the 33 that reproduced ran
on `node-d00a-*` or `node-d00b-*`, and so did all 160 E9 runs. The
correspondence is 7 for 7 and 33 for 33, with no exceptions in either
direction. Under random assignment, drawing exactly those seven out of forty
has probability 1 in 18,643,560.

The mechanism is ordinary: a different CPU generation gives a different
vectorised reduction order in torch and BLAS, the last bits of the gradient
differ, and in a chaotic optimisation that separates the trajectories over 15M
steps. Nothing was shuffled and nothing was mislabelled.

Two further facts fall out of the same table. The `d00a` and `d00b` groups are
interchangeable: gru128 lambda 0.99 seed 1 ran on `d00b-021` in the pilot and
`d00a-216` in E9 and reproduced bitwise. So the reproducibility boundary is the
letter class, not the number after it. And it is the pilot that is
heterogeneous, not E9: the pilot put 38 of 240 tasks on t00a, E9 put 0 of 160.

## 3. Size of the effect, and whether anything depends on it

Across the seven, mean difference in deployed return -0.0010, largest single
difference 0.0185, signs mixed. Two of the seven differ only in `final_return`
and `gap_value` and are bitwise identical in `deployed_return`, so over the five
that actually moved the mean is -0.0014; both figures are quoted because the
first is the one that belongs next to a claim about deployed return.

For scale, the across-seed standard deviation of deployed return within a
PositionOnlyCartPoleEasy mlp cell runs from 0.0026 to 0.0164 across the eight
lambdas, median 0.0098. So the largest node-swap difference is about 1.9 seed
standard deviations and the typical one is far below one. It is
within-condition variance, not a between-condition bias.

Tested directly rather than argued. Substituting the pilot's t00a values back
into E9 and re-running the frozen scorer leaves every verdict unchanged:

| | as landed | with t00a values substituted |
|---|---|---|
| P9.1 Easy | -0.0213, CI [-0.0288, -0.0140] | -0.0234, CI [-0.0307, -0.0168] |
| P9.1 Hard | -0.0060, CI [-0.0095, -0.0018] | unchanged |
| P9.1 | PASS | PASS |
| P9.2, six pairwise | all contain zero | all contain zero |
| P9.3 pooled ratio | 42.56 | 42.63 |

## 4. Where this does bite, and what was done about it

E6 ran 61 of its 80 tasks on t00a; E9 ran none. So the E6 numbers and the E9
numbers are not bitwise comparable to each other, and when E7 lands it will
almost certainly be d class while E6, which supplies P7.1's fourth instance,
is mostly t class.

This is a limitation to state, not a defect to repair, for three reasons. The
E6 effects are enormous next to the perturbation measured here: intermittency
0.49 against 0.00, cost ratios of 2 to 15. P7.1 estimates each instance's knee
from that instance's own lambda sweep, so the comparison that matters is
within instance and within class. And node class does not track the arm inside
E6: coupled and decoupled cells at matched lambda draw from the same mixture,
so the dissociation, which is E6's centrepiece, is a contrast between two
similarly contaminated groups.

Pinning E7 to one class with `-ac allow=` was considered and rejected: it
changes a submission path that works, for a bias that has been measured and is
smaller than seed noise, at the cost of queue time on the critical path.

What was done instead: `myriad/collect.sh` now reads the host out of every job
log and prints the class distribution per job family, flagging any family that
spans classes. Provenance is recorded rather than controlled, which is the
honest version and is what the reproducibility section will say.

## 5. The scorer defect found on the way

`prereg_phase2.py` could not score E9 at all. Amendment 1's duplicate guard
keyed on (env, lambda, seed) on the reasoning that env carries the arm. That
holds for E6 and E7, which are mlp only, and fails for E9, which runs mlp and
gru128 in one env: every legitimate E9 cell looked like a duplicate, so the
guard fired on all 160 rows and scoring exited before computing anything.

Fixed by adding arch to the key, and recorded as Amendment 2, dated and
disclosed as post-data because it is. The disclosure matters more than the
change: the guard's return value is read in exactly one place, a truth test
that either raises SystemExit or does nothing, so the edit can only decide
whether scoring runs, never what it returns. Amendment 1 had already put that
property in writing before any E9 data existed.

With scoring restored, the registered E9 verdicts are P9.1 PASS, P9.2 PASS,
P9.3 PASS. E6 stands as recorded: P6.1 FAIL, P6.2, P6.3 and P6.4 PASS.

## 6. The wrong constant has turned up a third time, now inside the data

Found while verifying the above, not previously recorded. Every one of the 80
GLEAM rows in results_phase2.jsonl carries

    ref_k_AC = 24.793797   ref_J_AC = 0.296390

and carries the same pair at every lambda from 0.8 to 1.0. Those are the TD(0)
values. The correct lambda-dependent references, which the preregistration
holds and uses, run 22.427, 20.547, 17.671, 11.395, 2.197, 1.756, 1.549, 1.507.
The other four reference fields, `ref_k_ml`, `ref_J_ml`, `ref_J_opt` and
`ref_rThK0`, match the frozen baselines exactly.

This is the third appearance of one number in a slot it does not belong in. The
preregistration header already records the first two: phase2-design.md quotes
k_AC = 24.79 for lambda 0.9, and the same value sits in the gleam_bench
baseline table as the TD(0) equilibrium, which is where it is correct. The
harness evidently writes the baseline-table entry into a field named for the
lambda-dependent quantity.

No verdict moves. Scoring never reads any `ref_*` field; every P6, P7 and P9
statistic takes its references from K_AC_LAMBDA and J_AC_LAMBDA inside the
preregistration, which are right. What it does mean is that the raw rows
self-describe with a reference the preregistration contradicts.

Deliberately NOT fixed in the harness before E7. E6's rows cannot be re-run, so
correcting ppo_lambda.py now would make E7's 120 rows disagree with E6's 80 in
a field that neither is scored on, trading a uniform documented quirk for a
silent inconsistency inside one results file. The fix is one line of
documentation at release: `ref_k_AC` and `ref_J_AC` in the raw rows are
k_AC(lambda=0) and J at that gain, taken from the baselines table, and are not
the reference for the row's own lambda. The released tool should either say
that or drop the two fields.

## 7. One cosmetic wart, left alone deliberately

`score_e6`'s header prints "E6 rows: 240 / expected 80" because it counts every
row in the file rather than E6's. It is a display line; `cell()` filters
correctly and no statistic reads it. Editing a frozen file to tidy a print
statement costs more credibility than the wart does, so it stays and is
recorded here instead.
