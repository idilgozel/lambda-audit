# A-5 Tier 2: the mechanism, and where it says the instability starts

Date: 2026-08-14. This closes the gap Tier 1b left open. Tier 1 falsified the
gain-axis escape story, Tier 1b located the instability off the gain manifold
but could not say what drives it or where it begins. Both are now answered in
closed form, validated against Monte Carlo, and the prediction for the three
unrun E7 instances is frozen in `prereg_e7_bias.py` before those runs exist.

## 1. The result in one line

Give the policy an action bias, `u = -k v + b + eta`. The aliased critic
destabilises `b = 0` below a critical lambda, and the destabilising term carries
the factor `(1 - lambda)/lambda` — exactly the bootstrap weight. On the E6
instance the crossing sits at **lambda\* = 0.9759**, inside the bracket E6
measured for the intermittency onset, `(0.95, 0.98)`, and strictly below that
instance's cost-law `lambda_half = 0.985382`.

Two distinct critical lambdas, from one calculation, in the right order, at the
right places.

## 2. The calculation

At `b = 0` the expected bias update vanishes by parity, so `b = 0` is a fixed
point of the averaged bias dynamics and the question is its stability.

Turning on `b` shifts the stationary mean and leaves the covariance alone, so
with `A_k` the closed loop,

    v = vtil + mu_v b,   mu_v = e1' (I - A_k)^-1 G0
    u = util + (1 - k mu_v) b,   util = -k vtil + eta

The system at `-b` is the mirror image of the system at `b`, so the critic obeys
`V_{-b}(v) = V_b(-v)`: `theta0` and `theta2` are even in `b`, `theta1` is odd.
To first order the ONLY change in the critic is a term linear in `vtil`, with
coefficient `kappa = 2 theta2 mu_v + dtheta1/db`. The O(b) parts are then

    chat_t = 2h [ Q mu_v vtil_t + R (1 - k mu_v) util_t ]
    r_t    = chat_t + beta kappa vtil_{t+1} - kappa vtil_t
    H_t    = sum_l (beta lambda)^l r_{t+l}

`kappa` is pinned without ever solving the critic. LSTD(lambda) at its fixed
point satisfies `E[psi(v_t) Ahat_t] = 0` for each feature. Projecting on the odd
feature and keeping O(b) gives `E[vtil_t H_t] = 0`, and the conditions from the
even features `1` and `v^2` are satisfied identically by parity, so this single
scalar equation determines `kappa`:

    kappa = 2 h alpha X / (X - beta X1),   alpha = Q mu_v - k R (1 - k mu_v)

with `X = sum_l (beta lambda)^l x_l`, `X1 = sum_l (beta lambda)^l x_{l+1}`,
`x_l = e1' A_k^l Sig e1`.

The gradient's response follows. `eta_t` is independent of the state at `t` and
enters the future only through `G0`, so `E[vtil_{t+m} eta_t] = s_e e1' A_k^{m-1}
G0` for `m >= 1` and zero at `m = 0`, and every sum collapses to a resolvent:

    Phi     = beta lambda e1' (I - beta lambda A_k)^-1 G0
    S(lam)  = 2hR(1 - k mu_v) + 2h alpha Phi + kappa Phi (1 - lambda)/lambda
    d ghat_b / db = -S

**`b = 0` is stable iff `S > 0`.**

The three terms are worth naming. The first is the direct cost of a nonzero mean
action, always stabilising, and it is proportional to `(1 - k mu_v)`, which
becomes small exactly when the gain is large. The second is the true-return
response. The third is the aliased critic's contribution, and it is the whole
story: it vanishes identically at `lambda = 1`, as it must, because at
`lambda = 1` the baseline cannot bias the gradient; and its weight
`(1 - lambda)/lambda` grows without bound as `lambda` falls.

So the mechanism is not that the critic is wrong. It is that the critic's
wrongness is admitted into the gradient in proportion to how much the estimator
bootstraps.

## 3. Validation, done before any prediction was extracted

The Monte Carlo was validated first, then used to validate the closed form.

Its LSTD(lambda) estimator reproduces `lstd_lambda.py`'s exact `theta2` to
within 0.3 percent at `lambda` in {0, 0.9, 0.95, 0.99} and `k` in {5, 20.5}, and
its advantage pipeline reproduces the exact `gae_gradient` to within 3 percent
away from the equilibria.

Then, closed form against a Monte Carlo central difference of `E[ghat_b]`, with
the critic refit at its LSTD(lambda) fixed point at each `b` and common random
numbers across `+b` and `-b`:

| instance | lambda | k | exact -S | MC slope |
|---|---|---|---|---|
| g1 h0.05 | 0.80 | 5.00 | +0.001792 | +0.001772 |
| g1 h0.05 | 0.90 | 5.00 | -0.000893 | -0.000915 |
| g1 h0.05 | 0.90 | 20.50 | +0.000938 | +0.000935 |
| g1 h0.05 | 0.95 | 10.00 | +0.000463 | +0.000451 |
| g1 h0.05 | 0.99 | 2.20 | -0.018337 | -0.018301 |
| g1 h0.05 | 1.00 | 5.00 | -0.003892 | -0.003938 |
| g2 h0.05 | 0.85 | 18.00 | +0.000163 | +0.000165 |
| g2 h0.05 | 0.95 | 8.00 | -0.001411 | -0.001412 |
| g4 h0.05 | 0.80 | 14.50 | -0.000443 | -0.000443 |
| g4 h0.05 | 0.90 | 2.20 | -0.026481 | -0.026434 |

Both signs, three instances, agreement within 1 to 3 percent throughout.

**One bug, found and fixed, reported because it silently removed the effect.**
The first simulator wrote the state update as `s' = A_k s + G0 b + w`, which
looks right and is wrong: the full action is `u = -k v + b + eta`, so the
correct update is `s' = F s + G0 u + w` and the `G0 eta` term had been dropped.
With exploration noise unable to enter the state, `E[v_{t+m} eta_t] = 0`, the
aliased critic has nothing to bite on, and the entire mechanism disappears. It
presented as a 16 percent error in `theta2` at `k = 5` and a factor of 2.2 at
`k = 20.5`, which is how it was caught: the Monte Carlo was checked against the
exact `theta2` before it was trusted for anything else.

**One near-miss worth recording.** An early attempt to read `kappa` out of a
simulation by fitting the critic at `b = 0.05` and computing
`theta1/b + 2 theta2 mu_v` disagreed with the closed form by a factor of 2.5.
The two terms are `-6.91` and `+6.68`; `kappa` is `-0.23`. It is a difference of
large numbers and a one percent error in either term moves it by thirty. The
resolution was to stop testing `kappa` in isolation and test the quantity that
matters, `S`, where no such cancellation occurs. The numerator and denominator
of `kappa` were then confirmed separately against Monte Carlo moments, to 0.4
percent each.

## 4. What it predicts, evaluated at the equilibrium the policy actually reaches

Stability has to be assessed at the gain the learner sits at, `k_AC(lambda)`.

| lambda | k_AC | S | direct | critic | verdict |
|---|---|---|---|---|---|
| 0.900 | 20.547 | -0.000935 | +0.000359 | -0.001293 | unstable |
| 0.950 | 17.671 | -0.000525 | +0.000405 | -0.000930 | unstable |
| 0.970 | 14.676 | -0.000187 | +0.000540 | -0.000727 | unstable |
| 0.980 | 11.395 | +0.000230 | +0.000851 | -0.000621 | stable |
| 0.990 | 2.197 | +0.018381 | +0.017086 | +0.001295 | stable |
| 1.000 | 1.507 | +0.033067 | +0.033067 | 0 | stable |

Crossing at **lambda\* = 0.975864**.

E6's measured onset bracket is `(0.95, 0.98)`. The prediction lands inside it.
The frozen `lambda_half` for the same instance is `0.985382`, so the predicted
instability onset sits strictly below the cost-law threshold, which is the
two-threshold structure `e7-predata-two-thresholds.md` recorded as an
expectation and which now has a derivation.

This instance is a POSTDICTION and carries no evidential weight: its bracket was
known. The weight is in the next section.

## 5. The frozen prediction for E7

Generated by executing the calculator, not copied from anywhere, and frozen in
`prereg_e7_bias.py` on 2026-08-14 while Myriad held zero `lam_e7` logs and
`results_phase2.jsonl` held zero rows for all three instances. The freeze is
checked mechanically, not asserted:

    python3 prereg_e7_bias.py --check-clean results/results/results_phase2.jsonl

| instance | gamma*h | lambda_half | lambda* | predicted onset bracket | status |
|---|---|---|---|---|---|
| GLEAM_g1_h0.02 | 0.02 | 0.996552 | 0.995594 | (0.995, 0.999) | PREDICTION |
| GLEAM_g1_h0.05 | 0.05 | 0.985382 | 0.975864 | (0.95, 0.98) | postdiction |
| GLEAM_g2_h0.05 | 0.10 | 0.957977 | 0.900964 | (0.9, 0.95) | PREDICTION |
| GLEAM_g4_h0.05 | 0.20 | 0.873138 | none | no onset anywhere | PREDICTION |

Three distinct bracket answers and one "nothing anywhere", on a grid of eight
lambdas. The gap `lambda_half - lambda*` is predicted to widen with `gamma*h`:
0.00096, 0.00952, 0.05701. That gap is the width of the stable-but-biased
regime, which `e7-predata-two-thresholds.md` called X4 and could only expect;
it is now a number.

The `g4_h0.05` prediction deserves a note. The theory says its bias direction is
stable at every lambda down to 0.70. That is independently what X3 expected from
a different argument — its knee is so low that most of the grid sits above the
horizon — and it doubles as the false-positive test for the shipped diagnostic.
Two arguments, written at different times from different starting points,
agreeing on the one instance where the answer is "nothing happens".

## 6. How a miss is to be read, fixed in advance

This is the stability of ONE off-manifold direction. The agent is an MLP with
others. If another direction is more fragile it destabilises at a higher lambda,
so the observed onset would sit ABOVE lambda\*.

- Onset inside the predicted bracket: the mechanism is identified.
- Onset above the bracket: the bias direction is real but is not the first to
  go. The mechanism is incomplete, not wrong, and lambda\* is reported as a
  lower bound.
- Onset below the bracket, or no onset where one is predicted, or any onset on
  g4_h0.05: hard failure. Reported as a failure, not reinterpreted.

## 7. What this changes for the paper

Tier 1b's own section 6 listed three things standing between it and a
paper-grade claim. This closes the second and sets up the third:

1. `--arch linear` inside `ppo_lambda.py`, to remove the cross-implementation
   confound between the numpy discriminator and the E6 MLP. STILL OPEN. It is an
   afternoon, and it should be done before the claim goes in the paper.
2. The theory for the non-gain directions. DONE, in closed form, validated, with
   the action-bias direction computed exactly as the scope proposed.
3. A mechanism that predicts the LOCATION of the onset. DONE and frozen, with
   three prospective instances arriving from E7.

The paper's claim can now read: partial observability biases the learning fixed
point through the aliased critic, and the same aliased critic destabilises the
policy directions orthogonal to the biased coordinate, with a strength
proportional to the bootstrap weight `(1 - lambda)`, so that below a computable
lambda the biased fixed point is unreachable. Both thresholds come out of one
calculation and they are different numbers.
