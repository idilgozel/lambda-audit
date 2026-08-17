# E6 curves: the below-horizon regime is intermittent, not divergent

Date: 2026-08-13. Analysis of the coupled arm's deployed-evaluation
trajectories, 15M steps, instance GLEAM_g1_h0.05. This CORRECTS a
characterisation in e6-findings.md section 3 and supplies a much better
diagnostic candidate for E8.

## 1. What the correction is

e6-findings.md says the below-horizon policies "climb toward the biased fixed
point, destabilize on approach, and never arrive". That was inferred from
final values, specifically measured J sitting 4 to 6 times the analytic J at
the fitted gain. The inference of instability was right. The shape was wrong.

The trajectories do not climb once and break. They make RECURRENT EXCURSIONS.
The cost departs by a factor of 4 to 30, partially returns, and departs again,
for the entire run. Examples from lambda 0.9, deployed cost sampled through
training:

    seed 3:  0.278, 0.279, 0.342, 3.188, 0.273
    seed 2:  0.294, 0.382, 0.976, 0.380, 0.351
    seed 4:  0.328, 0.515, 0.279, 1.937, 0.470

Nothing converges. The reported final value is a SNAPSHOT of a wandering
process, taken wherever the 15M-step boundary happens to fall.

## 2. This explains three things that were open

- Why P6.1 failed. It required convergence to k_AC(lambda). Below the horizon
  there is no convergence to anything, so the prediction was testing for a
  state the system never enters.
- Why below-horizon excess cost is non-monotone in lambda, with 0.9 reading
  lower than both 0.8 and 0.95. That is not ordinary seed noise. It is where
  in the excursion cycle each run terminated.
- Why the seed spread at 0.95 runs from 0.23 to 4.38. Same cause.

## 3. Quantified, and the separation is total

Fraction of deployed-eval points exceeding twice J_ml, split into the first
and second half of training. Early excursions are learning transient. Late
excursions are the mechanism.

| lambda | late-half excursion fraction (per seed) | mean |
|---|---|---|
| 0.9 | 0.592, 0.489, 0.495, 0.511, 0.375 | 0.492 |
| 0.95 | 0.533, 0.370, 0.451, 0.560, 0.359 | 0.454 |
| 0.99 | 0.000, 0.000, 0.000 | 0.000 |
| 1.0 | 0.000, 0.000 | 0.000 |

Ten below-horizon runs all at or above 0.359. Five above-horizon runs all
exactly zero. No overlap whatsoever. Above the horizon the late-half median
sits at J_ml and the late maximum never reaches 0.27. Below it, the process
is outside twice J_ml for roughly half of the second half of training.

Early-half fractions are nonzero everywhere, including 0.126 and 0.186 at
lambda 1.0, which is the learning transient. The early-late split is doing
real work and should be kept.

## 4. A self-normalising version, which is what could ship

The table above needs J_ml, which an external benchmark does not have.
Replacing it with the run's OWN late-half median removes that dependency:

| lambda | fraction above 2x own late median | mean |
|---|---|---|
| 0.9 | 0.152, 0.446, 0.337, 0.255, 0.293 | 0.297 |
| 0.95 | 0.364, 0.212, 0.179, 0.261, 0.190 | 0.241 |
| 0.99 | 0.000, 0.000, 0.000 | 0.000 |
| 1.0 | 0.000, 0.000 | 0.000 |

Still separates completely, window (0.0000, 0.1522). It is per-run, needs no
analytic baseline, no privileged information and no multiple seeds. Those are
exactly the properties the shipped diagnostic needs and that no candidate in
the current E8 table has together.

Not yet a candidate for freezing. It must first be run against BOTH negative
sets: the pilot's action-independent cells, and the closed-loop cells above
the horizon where the solvedness pathology lives. Its behaviour on discrete
environments with different reward scales is untested.

## 5. What the paper should now claim

Stronger and more useful than the equilibrium framing:

> Above the critical bootstrap horizon, learning converges, and it converges
> to the location the closed form predicts, within 5 to 7 percent. Below it,
> learning does not converge at all. It enters a regime of recurrent
> instability in which the policy repeatedly departs from and partially
> returns toward the stable region, with excursions of 4 to 30 times the
> optimal cost persisting for the entire training budget.

That is a sharper practical warning than a biased equilibrium. A practitioner
below the horizon does not get a worse controller. They get a controller
whose quality depends on when they stopped training.

## 6. Consequence for E7

P7.1 locates the transition by half-collapse of final excess cost. Below the
horizon those finals are draws from a wandering process, so that estimator
carries structural noise beyond seed variation. P7.1 is frozen and will be
scored exactly as registered.

Alongside it, report an exploratory transition locator based on onset of
late-half intermittency. On E6 that measure is binary at the knee, so it may
locate lambda_half far more sharply. Exploratory, reported as such, and it
cannot alter the frozen scoring.
