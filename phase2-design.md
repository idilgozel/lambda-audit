# Phase 2 design and preregistration protocol

Date: 2026-08-12. Supersedes the Phase 2 sketch in lambda-audit-plan.md section 5.
Decision taken 2026-08-12: the audit and the theory paper MERGE into one paper.
The arXiv posting is that paper's preprint. Phase 2 builds its empirical arm.

## 1. What went wrong in the pilot, with evidence

The pilot returned a null on the flagship gate. That null is not evidence
against the thesis. It is evidence that the grid could not test it. Three
defects, each now measured rather than asserted.

**Defect 1: no action coupling in two of three environments.** In
RepeatPrevious the deck deals the next card regardless of the answer, so
Q(s,a) - Q(s,a') = r(s,a) - r(s,a') exactly. Every critic bias cancels from
the advantage, at any lambda and any degree of aliasing. The learning-channel
mechanism needs the closed loop: a biased critic misplacing the fixed point
of a policy that feeds back into the state. Two thirds of the grid had no
channel. This is provable, not a story.

**Defect 2: ceiling saturation.** 26 of 48 cells sit above 0.995 deployed
return. Over half the grid physically cannot show graded degradation. E4
measured excess cost against a known optimum, which is unbounded above and
therefore sensitive. RepeatPrevious is near binary: you learn the k-back
mapping or you do not.

**Defect 3: the gate read the wrong cell.** G2 measured the memory-versus-MLP
gap on Medium, an action-independent environment. The one cell in the grid
that could speak was not what the gate was reading.

**And the effect appeared anyway, where it was permitted.**
PositionOnlyCartPole is the only closed-loop environment, its critical
horizon is 0.88, and the MLP is the only architecture there with headroom.
That cell reads 0.2360 at lambda 0.9 against 0.2565, 0.2567 and 0.2568 at
0.95, 0.99 and 1.0. Flat above, one step down below. Difference -0.0204,
95% bootstrap CI [-0.0319, -0.0077], P(negative) = 0.999. The gru128
instability sits in the same cell and nowhere else. This is exploratory, not
a registered result, and Phase 2 registers it properly.

## 2. Instrument choice

The GLEAM environment (gleam_bench.py, GLEBench) removes all three defects at
once and is the only instrument available that does.

- Closed loop. The action u enters the state update directly.
- Tunable critical horizon. The bath correlation time is set by gamma and the
  step by h, and lambda_crit is approximately 1 - O(gamma*h), so the
  transition can be driven through the lambda grid by design.
- Exact analytic baselines, so the metric is excess cost against a true
  optimum with no ceiling.
- Already validated. E4 ran 40 clean tasks on it and found lambda 0.9 giving
  0 of 5 restored at J = 0.3455 against lambda 0.99 giving 4 of 5 at 0.2315.

Effect sizes available, computed from the exact baselines today:

| c | gamma | h | gamma*h | J_opt | k_ml | J_ml | k_AC | J_AC | k_AC/k_ml | J_AC/J_ml |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 1.0 | 0.05 | 0.050 | 0.1989 | 1.649 | 0.2196 | 24.794 | 0.2964 | 15.0x | 1.350 |
| 0.3 | 2.0 | 0.05 | 0.100 | 0.1552 | 1.347 | 0.1720 | 23.920 | 0.2959 | 17.8x | 1.720 |
| 0.3 | 1.0 | 0.02 | 0.020 | 0.1973 | 1.664 | 0.2175 | 63.506 | 0.2987 | 38.2x | 1.373 |
| 0.3 | 4.0 | 0.05 | 0.200 | 0.1044 | 1.158 | 0.1130 | 22.006 | 0.2874 | 19.0x | 2.543 |
| 0.3 | 1.0 | 0.10 | 0.100 | 0.2017 | 1.621 | 0.2233 | 11.873 | 0.2919 |  7.3x | 1.307 |

The predicted excess cost is 30% to 150%, against a largest measured pilot
effect of 8%. rThK0 is exactly 0.3000 for every instance, which is the cost
ceiling law and doubles as a cross-check on any implementation.

## 3. Lambda grid

The pilot grid was too coarse near 1. E4 located the transition inside
[0.98, 0.99] with lambda_half = 0.98541. Phase 2 uses:

    lambda in {0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0}

Eight values, four of them above 0.97, so the transition is resolved rather
than straddled.

## 4. Experiments

### E6, the minimal pair. This is the kill test and runs first.

Instance: c = 0.3, gamma = 1.0, h = 0.05, theta = 1, q = r = 1, rho = 0.05.
Two arms, identical in every respect except one term in the state update.

- **Coupled**: s <- F s + G u + noise. The standard environment.
- **Decoupled**: s <- F s + noise. The action still incurs its r u^2 cost and
  the agent still sees only v, but the action does not move the state.

Grid: 2 arms x 8 lambda x 5 seeds = 80 runs, 15M steps each.

Frozen predictions, generated from the closed forms before any run:

- P6.1 Coupled, lambda <= 0.95: learned gain converges toward k_AC(lambda)
  from lstd_lambda.py, and J lands within 15% of J(k_AC). At lambda 0.9 the
  reference values are k_AC = 24.79 and J_AC = 0.2964.
- P6.2 Coupled, lambda -> 1: gain falls toward k_ml = 1.649 and J toward
  J_ml = 0.2196. Excess cost over the below-knee arm is at least 25%.
- P6.3 Decoupled, all lambda: learned gain converges to 0 and stays there.
  No lambda dependence beyond Monte Carlo noise. This is the preregistered
  replication of the pilot's null, on the same environment, which removes the
  post-hoc character of the RepeatPrevious explanation.
- P6.4 The arm difference in final gain at lambda 0.9 exceeds the arm
  difference at lambda 1.0 by at least a factor of 3.

Kill criterion: if the coupled arm shows no lambda dependence in gain or cost
at 5 seeds, the learning-channel thesis fails its cleanest test and Phase 2
stops. That verdict is accepted and published.

### E7, the quantitative law. Runs only if E6 passes.

Four instances placing the critical horizon at different points:
gamma*h in {0.02, 0.05, 0.10, 0.20}, via (gamma, h) = (1, 0.02), (1, 0.05),
(2, 0.05), (4, 0.05). Coupled arm only.

Grid: 4 instances x 8 lambda x 5 seeds = 160 runs.

Frozen predictions:

- P7.1 For each instance the measured half-collapse point of excess cost
  falls within 0.01 of the lambda_half computed from lstd_lambda.py.
- P7.2 lambda_half is monotone decreasing in gamma*h.
- P7.3 The plateau of J below the knee lies within 10% of r*theta*K(0) =
  0.3000 for every instance, independently of where the knee sits.

P7.1 is the claim nobody else can make. It is a prediction of a location,
generated before the runs, not a fitted curve.

### E8, the diagnostic, rebuilt against ground truth.

No new runs. The pilot's gap_lambda failed because std(G_MC) collapses on
solved deterministic tasks, so it measured solvedness. The replacement is
built and validated on E6 and E7 data, where the true value function and the
true theta2 are known analytically.

Candidates, all architecture blind: normalized by return range rather than
return std; unnormalized mean lambda-return gap; direct estimate of critic
curvature theta2 against its analytic value; measured policy gain against
k_ml. Selection is exploratory. The winner is frozen before any external
benchmark run and cannot be confirmed on the data used to select it.

### E9, corroboration on a familiar benchmark.

PositionOnlyCartPole, where the pilot already showed a signal. A reader
trusts a known benchmark more than an environment we built.

Grid: {mlp, gru128} x {Easy, Hard} x 8 lambda x 5 seeds = 160 runs. Hard has
a 600-step cap, which triples the headroom that saturated the Easy cells.

Frozen predictions:

- P9.1 The MLP shows a step down in return below lambda 0.9 on both variants,
  replicating the pilot's exploratory finding at 5 fresh seeds.
- P9.2 The step is absent above the critical horizon, so returns at 0.95,
  0.99 and 1.0 are statistically indistinguishable.
- P9.3 gru128 instability, measured as across-seed variance, is highest at the
  lowest lambda.

### Deferred to Phase 3

Velocity-only MuJoCo Walker, POPGym's closed-loop families (Labyrinth,
Battleship, Minesweeper, Concentration) and Memory Gym's GRU versus TrXL
comparison. Admission requires all three preconditions: action coupled,
graded reward, and headroom.

## 5. Preregistration protocol

1. `prereg_phase2.py` is written and frozen BEFORE the E6 array is submitted.
   It contains every constant above, the closed-form reference values
   generated by running lstd_lambda.py and gleam_bench.py, and a no-sentinel
   guard that refuses to score while any value is unset.
2. `deploy.sh` runs that guard as a pre-flight and refuses to ship otherwise.
3. E7 predictions are frozen before E7 runs, not before E6, since E7 only
   happens conditional on E6.
4. Any amendment is dated, justified, and made before the relevant data
   exists. The pilot's best-memory-arch amendment is the model to follow.
5. The pilot's frozen record and its two failed gates stand unchanged. They
   are reported in the paper as registered.

## 6. Implementation constraints

These exist because dependency problems cost days on the cluster.

- **No scipy on Myriad.** gleam_bench.py uses expm, solve_discrete_lyapunov
  and solve_discrete_are. All of those are needed only to build the constant
  matrices F, G, Sd, Lw, L0 and the analytic baselines. Compute them locally,
  serialise them, and ship them. The cluster steps the environment with pure
  numpy.
- **No new cluster dependencies at all.** torch and numpy from the UCL module,
  as now.
- **Bitwise verification.** The native GLEAM environment must reproduce
  GLEBench step for step under identical seeds, the same protocol that
  validated envs_native.py against popgym.
- **Continuous actions.** ppo_lambda.py is discrete only. Phase 2 needs a
  Gaussian policy head with state-independent log_std, and a `--fix-sigma`
  option, since E4 showed entropy collapse acts as a brake that confounds the
  lambda reading.
- Cost accounting: report J as a rate, matching gleam_bench conventions, so
  the analytic baselines are directly comparable.

## 7. Scale

E6 is 80 runs, E7 is 160, E9 is 160. Roughly 400 runs at 15M steps. The
GLEAM state is two dimensional and the networks are small, so expect
per-run times near the pilot's mlp figure of about 22 minutes, giving
roughly 150 core-hours in total. Three arrays through the existing pipeline.
