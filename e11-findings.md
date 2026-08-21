# E11: the gates fired, and the design error they exposed was mine

Date: 2026-08-21. Registered verdict first, because it stands regardless of
everything that follows.

    G11a  mlp IQM at lambda 1.0  = -0.4412   (needed >= -0.35)   FAIL
    G11b  gru IQM at lambda 1.0  = -0.4455   (needed >= +0.02)   FAIL
    P12.1, P12.2, P12.3  NOT SCORED (capability gates failed)
    P13.1 (prereg_e8.py)  PASS: 50 specificity runs, 0 flagged

E11 makes no claim. The gates were written to stop a dead instrument
masquerading as a negative result, and that is exactly what they did.

## 1. What the runs actually show

BattleshipEasy, 8x8, 5 seeds, 15M steps, deployed return by cell. Scripted
anchors: uniform random -0.4447, no-repeat +0.0568, no-repeat+targeting
+0.4135.

| lambda | mlp IQM | gru128 IQM | measured memory gap |
|---|---|---|---|
| 0.8 | -0.4042 | -0.3046 | +0.0996 |
| 0.9 | -0.4283 | **-0.2699** | **+0.1584** |
| 0.95 | -0.4267 | -0.3508 | +0.0759 |
| 0.98 | -0.4513 | -0.4067 | +0.0446 |
| 0.99 | -0.4287 | -0.4342 | -0.0055 |
| 0.995 | -0.4356 | -0.4380 | -0.0024 |
| 0.999 | -0.4326 | -0.4400 | -0.0074 |
| 1.0 | -0.4412 | -0.4455 | -0.0043 |

Neither architecture approaches competent play. The mlp sits at the uniform
floor at every lambda. The gru is meaningfully above the floor at low lambda
and collapses onto it by lambda 0.99.

## 2. The design error, which is mine and is documented rather than buried

**The gate thresholds were set above the published state of the art.** POPGym's
own Table 3 reports BattleshipEasy MMER from -0.427 (DNC) to -0.226 (PosMLP),
with most of their thirteen architectures clustering between -0.29 and -0.38.
G11b demanded gru >= +0.02. No published architecture reaches that on this
task. The gate was unachievable by construction, and I set it by reasoning
from our own scripted anchors instead of from the benchmark's reported
results.

**The environment was the wrong choice for the hypothesis.** The POPGym paper
states directly: "Memory models do not noticeably outperform the MLP in many
game environments, such as Autoencode or Battleship." E11 existed to test the
POSITIVE side of the scope law, on a task where memory matters. Their own
results say Battleship is not such a task. This was checkable in one lookup
before a single run.

That check is the project's own standing standard. lambda-audit-phase0.md
records a verified lambda table with "every entry checked in source code or
the paper itself". The standard existed; it was applied to lambda and not to
the environment choice.

**The capability probe ran in the wrong order.** The pilot's design submits a
2-task capability job (gate G1c) BEFORE the 240-task array. E11 put its
capability gates after the full 80-run grid. Same instrument, wrong position,
119 core-hours spent to learn what 2 runs would have shown.

## 3. What the data is nonetheless worth, labelled post-hoc

**It replicates POPGym.** Our gru reaches -0.2699 (IQM) and -0.2013 (best
seed) against their best reported -0.226 and their typical -0.29 to -0.38, in
a completely independent single-file implementation with a natively
reimplemented, bitwise-verified environment. That is an unplanned external
replication of their Battleship numbers.

**The measured memory advantage is strongly lambda-dependent.** The gap runs
+0.158 at lambda 0.9 and -0.004 at lambda 1.0 — POPGym's own reference
setting. Their reported "memory does not help on Battleship" is measured at
exactly the lambda where our data shows the gap vanishing.

The registered statistic P12.3 (gap(0.95) - gap(1.0)) has point estimate
**+0.0802**, in the predicted direction. IT IS NOT SCORED AND IS NOT CLAIMED.
The gates failed; the frozen scoring refuses; relaxing a gate after seeing the
data is the precise failure mode this project exists to avoid. It is recorded
here only so the decision about what to do next is made on the evidence.

**Caveat on the direction.** Low lambda helping here is most likely the
ordinary variance benefit of bootstrapping on a 64-step sparse-reward task
with a 64-action space, not the aliased-critic bias mechanism the audit is
about. Both produce lambda-dependent architecture rankings; only one is this
paper's mechanism. Any use of this data must say so.

## 4. P13.1, passed but weakened

The diagnostic flagged 0 of 50 specificity runs, as registered, and read
exactly 0.000 on every below-knee mlp cell. Registered PASS. But the runs are
flat by construction here — nothing learned enough to develop the dynamics the
tool discriminates — so this is a weak confirmation of specificity, not the
test intended. Reported as such.

## 5. What would have to be true for a retry to be worth it

A replacement (E12) needs all three, verified before submission:

1. A CLOSED-LOOP POPGym environment (action-dependent hidden state), so it
   tests the scope law's positive side.
2. One where POPGym's own Table 3 shows memory architectures BEATING the MLP,
   read from the table, not assumed.
3. Gates calibrated to their published range for that environment, plus a
   2-run capability probe submitted and read BEFORE the grid.

If no POPGym environment satisfies (1) and (2) together, the honest conclusion
is that the positive-side third-party test is not available in this benchmark
suite within budget, and the paper says so in limitations.
