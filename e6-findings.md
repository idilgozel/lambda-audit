# E6 findings: the minimal pair

Date: 2026-08-13. Job lam_e6 (127988), 80 of 80 rows, scored by the frozen
prereg_phase2.py. Verdicts as registered: P6.1 FAIL, P6.2 PASS, P6.3 PASS,
P6.4 PASS. Kill criterion not triggered.

## 1. The headline

The dissociation exists and is now preregistered evidence, not narrative.
One term in the state update separates the two arms.

- Decoupled arm: gain at most 0.0079 in magnitude, spread 0.0091 across all
  eight lambdas, cost pinned at 1.0. No lambda dependence. This replicates
  the pilot's null as a frozen prediction on the same instrument that shows
  the positive effect. The action-coupling scope law is confirmed on both
  sides.
- Coupled arm: cost moves by a factor of six across the same lambda grid.
  Arm difference in gain at 0.9 is 3.52 times the difference at 1.0, against
  a registered threshold of 3.

## 2. Above the critical horizon the law is quantitative

| lambda | gain measured | gain predicted | J measured | J at fitted gain |
|---|---|---|---|---|
| 0.99 | 2.301 | 2.20 | 0.2233 | 0.2239 |
| 0.995 | 1.865 | ~1.8 | 0.2191 | 0.2203 |
| 0.999 | 1.665 | 1.55 | 0.2182 | 0.2196 |
| 1.0 | 1.623 | k_ml = 1.649 | 0.2182 | 0.2196 |

Fixed-point locations within 5 to 7 percent. Measured cost matches the
analytic cost of the fitted gain to under 1 percent, so the learned policies
are clean linear controllers here. At lambda 1.0 the agent recovers the
optimal memoryless gain to 1.6 percent and its cost to 0.6 percent.

## 3. Below the critical horizon: unstable, and worse than predicted

P6.1 required convergence to k_AC(lambda) with cost within 15 percent of
J(k_AC). It failed, in exactly the mode the pre-data note anticipated from
the round-2 record, written before this data existed.

| lambda | gain (range) | J measured | J at fitted gain | J predicted at k_AC |
|---|---|---|---|---|
| 0.8 | 5.0 (3.0 to 9.7) | 1.461 | 0.251 | 0.294 |
| 0.9 | 6.3 (4.1 to 9.9) | 0.597 | 0.259 | 0.292 |
| 0.95 | 4.5 (0.6 to 8.5) | 1.501 | 0.247 | 0.289 |

Measured cost is four to six times the analytic cost at the fitted gain, so
these policies are not linear controllers at a wrong equilibrium. They climb
toward the biased fixed point, destabilize on approach, and never arrive.
The seed spread at 0.95 runs from 0.23 to 4.38, the escape-or-blow-up
bimodality already visible in the round-2 curves.

Refined claim, which the paper should carry: the equilibrium location law is
verified where the fixed point is stochastically stable. Below the critical
horizon the fixed point exists but is not reachable at practical step sizes,
and realized cost exceeds even the biased equilibrium's prediction. The
instability is not a nuisance to the theory. It is the stronger practical
warning, and it is the empirical content of the stochastic-stability item
(A-5) the arXiv plan had parked.

## 4. The transition sits where the closed form says

Excess cost over J_ml by lambda: 5.65, 1.72, 5.84, 0.30, 0.02, 0.00, -0.01,
-0.01 at 0.8 through 1.0. The collapse concentrates between 0.95 and 0.99.
Interpolated half-collapse is near 0.977 against the frozen model value
0.98541. P7.1's window is plus or minus 0.01. E7 tests this properly across
four instances with the knee at four different locations.

Below-knee excess is non-monotone (0.9 reads lower than 0.8 and 0.95). At
five seeds with bimodal instability this is expected ordering noise, and the
curves should be shown rather than smoothed over.

## 5. Small flags, so they are on record

- The decoupled arm reads 0.992 to 0.999, marginally under the exact floor
  of 1.0. Expectation cannot be below 1.0. A 200k-step window estimates
  Var(v) with about 1.4 percent noise, so this is finite-window wobble, the
  same size as the calibration test's seed spread. No gate reads it.
- J at lambda 1.0 (0.2182) sits 0.6 percent below J_ml, same cause.

## 6. Decisions taken 2026-08-13

1. E7 launches. The frozen conditionality for E7 was the kill criterion,
   which did not trigger. P6.1's failure is a finding, not a stop.
2. Duplicate-row hazard closed before launch: configs_e7.txt includes the
   g1_h0.05 instance E6 already ran at the same seeds. E6's coupled arm IS
   that instance's data. E7 ships only the three new instances.
3. E9 scoring is frozen as a dated amendment before any E9 data exists, then
   E9 launches in parallel.
4. E8 diagnostic work begins on E6 data as its training set, exploratory,
   with the winner to be frozen before any external benchmark run.
