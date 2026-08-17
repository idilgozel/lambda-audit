# A-5 scope: stochastic stability of the aliased fixed point

Date: 2026-08-13. Item A-5 was parked in arxiv-plan.md with a one-week
timebox and a language-repair fallback. E6's intermittency result makes it
empirically central, with 80 runs behind it. This scopes the attempt.

## 1. What the data now demands an explanation for

E6, coupled arm, instance g1_h0.05. Late-half fraction of deployed-eval
points above twice J_ml:

    lambda 0.9  : 0.592, 0.489, 0.495, 0.511, 0.375
    lambda 0.95 : 0.533, 0.370, 0.451, 0.560, 0.359
    lambda 0.99 : 0.000, 0.000, 0.000
    lambda 1.0  : 0.000, 0.000

Below the horizon the iterate departs and partially returns for the whole
run, one to two distinct excursions per late half, amplitudes of four to
thirty times optimal cost. Above it, nothing. The averaged theory predicts a
stable fixed point at every lambda, so the averaged theory alone cannot
explain this. That gap is A-5.

## 2. The landscape, computed 2026-08-13 (TD(0), s_e = 0.09)

    k_ml = 1.649      k_eq = 24.794      divergence edge k_max = 39.998
    ghat'(k_eq) = +2.087e-04, positive, so STABLE under k <- k - alpha*ghat
    restoring force at k_eq is 0.38% of that at k_ml: basin ~266x shallower
    headroom to the divergence edge: 15.2 gain units, 61% of k_eq
    quasi-potential, k_eq to divergence edge : 1.093
    quasi-potential, k_eq to k_ml           : 0.283
    |ghat| stays under 5% of its maximum across k in [0.6, 39.4]

A stable fixed point with a very weak restoring force, sitting on a nearly
flat plateau, with a hard divergence boundary close above. Classic Kramers
with a low barrier.

One thing that does NOT yet fit and must be resolved in Tier 1. The
quasi-potential to the divergence edge (1.093) is about four times the
quasi-potential back down to k_ml (0.283), so a naive isotropic-noise
argument says the iterate should fall toward k_ml more readily than it
climbs toward divergence. The observed excursions are toward HIGH cost. The
likely resolution is that the gradient noise is strongly state dependent,
since its scale follows the state variance, which itself collapses as the
gain rises. If so, the escape problem is anisotropic and the barrier must be
weighted by the local noise intensity, not read off the potential alone.
Resolving this is the first real test of the story.

## 3. Target proposition

> For the linear GLE model at pinned exploration variance and step size
> alpha, the fixed point k_eq(lambda) of the averaged update is locally
> asymptotically stable for every lambda. It is NOT stochastically stable
> below a critical lambda: the expected time to a large excursion is finite
> and short relative to a practical training budget, and the excursion rate
> grows as lambda falls, diverging as the confining barrier collapses.

The practical corollary is what the paper wants: below the critical horizon,
a practitioner does not obtain a worse controller, they obtain one whose
quality depends on when they stopped training.

## 4. Tiers, with a one-week timebox on 1 and 2 combined

**Tier 1, two to three days, high confidence.** All closed form or
numerically exact from existing machinery.
- ghat'(k_eq(lambda)) as a function of lambda, via lstd_lambda.py. Predict
  the restoring force weakens as lambda falls.
- k_max(lambda), the divergence edge, and the gap k_max - k_eq(lambda).
- The gradient-noise intensity D(k) at the equilibrium, from the same Wick
  machinery that produced ghat. This is what resolves the direction puzzle
  in section 2.
- Deliverable: barrier-over-noise as a function of lambda, plotted against
  the measured excursion fractions above.
- ABORT CRITERION: if the barrier-over-noise ratio does not fall with
  lambda, the story is wrong. Stop here and report the numerical landscape
  as phenomenology. This is a one-day check and it comes first.

**Tier 2, three to four days, moderate confidence.**
- Kramers rate for the escape, giving expected excursion frequency as
  exp(-2 Delta(lambda) / (alpha D)).
- Test against E6's measured excursion counts, one to two per late half
  below the horizon and zero above.
- A predicted critical lambda from barrier collapse, compared against the
  frozen lambda_half of 0.985382 and the empirical 0.977.

**Tier 3, stretch, strictly timeboxed, low confidence.** A rigorous
statement with remainder bounds. This needs stochastic approximation with
state-dependent noise. Kushner and Yin, and Borkar, are the standard
references. If it resists inside the timebox, state Tiers 1 and 2 as a
verified quantitative account and mark rigour as future work. That is the
same fallback pattern used for Lemma A.

## 5. Why this is worth the week

- It converts the paper's second finding from phenomenology into a result.
  Without it, "below the horizon learning is unstable" is an observation.
  With it, the instability has a rate with a formula.
- It uses machinery already built and validated in this project, so the
  marginal cost is low.
- It closes a referee's most obvious objection: the averaged theory predicts
  a stable equilibrium, so why does the experiment not show one.
- The same Kramers structure appears in the risk-engine work, so the thesis
  gains a genuine methodological through-line rather than two unrelated
  applications.

## 6. Sequencing

Tier 1 runs now, in parallel with the E7 and E9 queues, since it is pure
computation on existing closed forms and needs no cluster. Tier 2 starts
only if Tier 1's abort criterion is cleared. Neither touches any frozen
preregistration, and neither can alter E7 or E9 scoring.
