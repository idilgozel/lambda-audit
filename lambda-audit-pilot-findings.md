# Pilot findings: the lambda audit, round 1

Date: 2026-08-12. Data: 240 grid runs plus 2 capability runs, job lam_pilot
(119762), all tasks completed, all curves saved. Gates scored by the frozen
prereg_lambda_pilot.py, unmodified.

## 1. Verdicts, verbatim

    G1a: min memory IQM 1.000 (>= 0.9), mlp -0.498 (<= 0.0) -> PASS
    G1b: memory-mlp gap 0.741 (>= 0.1) -> PASS
    G1c: transition gap at 1.0 = 1.492 (>= 0.5) -> PASS
    G2 : gap at 0.9 = 1.495 (< 0.373) -> FAIL
    G3 : balanced accuracy 0.476 -> FAIL

The sanity gates all pass. The published Easy separation reproduces exactly
(memory architectures at 1.000, MLP at chance). The harness is not in
question. The flagship prediction and the blind diagnostic both failed. Per
the pre-commitment in the prereg, this gets reported as is.

## 2. The headline result

stack64 solved RepeatPreviousMedium at every lambda. Twenty runs, four
lambda values, all at 0.999 or better. The memory advantage over MLP is
1.495 at lambda 0.9 and 1.492 at 1.0. It is lambda invariant to three
decimal places. The registered prediction said it would collapse below the
knee. It did not. The prediction is refuted on this environment.

Secondary observations, all consistent with one explanation given below:

- gru128 fails Medium at every lambda (about -0.31 to -0.44). Its k=32
  failure is architecture, not lambda, matching the published POPGym GRU.
- The only lambda sensitivity in the entire grid sits in the one borderline
  closed-loop cell. PositionOnlyCartPole at 0.9, knee 0.88: MLP drops about
  9% relative there and nowhere else, and gru128's single unstable seed
  (0.53) is there and nowhere else.
- gap_lambda is exactly 0.000 in all 62 lambda=1.0 rows. The identity holds
  on cluster hardware. Data integrity is clean.

## 3. Why the mechanism did not bite here

RepeatPrevious transitions are action independent. The deck deals the next
card whatever the agent answers. When actions do not affect the state, the
next-state value term is identical across actions, so any critic bias
cancels out of every action comparison in expectation. This holds for any
lambda and any degree of aliasing. The learning-channel mechanism requires a
closed loop: a biased critic misplacing the fixed point of a policy that
feeds back into the state. That is where the GLE results (E4, the stack32
capture) live. RepeatPrevious has no loop. It is a prediction task, and for
prediction tasks the theory, correctly applied, predicts the null we
observed. Lambda can change gradient variance and learning speed there, but
it cannot create a wrong fixed point.

The design error, in one sentence: no cell in this grid was both closed-loop
and below the knee. The knee arithmetic was done carefully and the
control-coupling precondition was not checked. That precondition is now the
first row of any future environment selection table.

## 4. Why the diagnostic failed

gap_lambda = mean(G_lambda - G_MC) / std(G_MC). The denominator is the
problem. On solved, near-deterministic tasks the spread of Monte Carlo
returns collapses, so small critic imperfections blow the ratio up. stack8
on Easy at 0.9 scores 0.999 on the task and 0.52 on the statistic. On failed
tasks the reward stream is near random, the denominator is large, and the
ratio shrinks. So the statistic mostly measured solvedness and reward
determinism, not the bootstrap horizon. The threshold was calibrated on
undertrained smoke runs, which made it look discriminative when it was not.
Any replacement must be validated across the solved and failed regimes
before it is frozen, and it cannot be re-frozen on this dataset for
confirmatory use.

## 5. What survives, reframed

The audit was two sided by design, and the negative side is a real finding:

1. POPGym's recall-style tasks are lambda robust. Rankings measured at
   lambda 1.0 transfer to 0.9 unchanged. For this family of benchmarks the
   fixed-lambda practice is vindicated, not indicted.
2. The confound claim narrows to control-coupled partial observability,
   which is where the closed-form theory lived all along. The E4 and round-2
   evidence for lambda gating comes from a control task. The whisper at
   cartpole 0.9 points the same way.

The narrowed claim is sharper and more useful than the broad one: lambda is
a confound for memory comparisons when actions influence the hidden state,
and not when the task is pure recall. That scope law is itself the audit's
product, and it is testable.

## 6. Phase 2 targets (closed loop, below knee, in that order of priority)

1. Velocity-only MuJoCo Walker across lambda. Closed loop, knee computable
   from measured velocity autocorrelation, E2 infrastructure exists, and it
   is a documented community practice, not our construction.
2. POPGym's own closed-loop families: navigation (LabyrinthEscape and
   LabyrinthExplore), Battleship, Minesweeper, Concentration. Persistent
   hidden state means very long correlation times, so common lambdas sit
   far below the knee. This keeps the audit inside the reference benchmark.
   Feasibility of native reimplementation to be assessed per env.
3. Memory Gym's GRU vs TrXL comparison, run at their 0.95 and at 1.0.
   Closed loop and below knee, and it is the published comparison itself.

Before any Phase 2 run: a new prereg file, with the control-coupling
precondition stated, a fresh P2 for closed-loop environments, and a
diagnostic redesigned and validated on this pilot's data as a training set,
frozen before any new below-knee run.

## 7. Status of the frozen record

prereg_lambda_pilot.py remains untouched. The pilot verdicts stand as
registered: three sanity passes, two substantive fails. The dataset (242
rows, 242 curve files) is the audit's first public artifact and supports
both the robustness finding and the diagnostic post-mortem.
