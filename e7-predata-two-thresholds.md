# Pre-data note: two critical lambdas, three regimes

Date: 2026-08-13, written while E7 and E9 are queued and UNREAD. No E7 or E9
data has been collected or seen. P7 and P9 scoring is frozen and unaffected
by this note. These are exploratory expectations, stated in advance so the
three-regime test on E7 is prospective rather than retrofitted.

## 1. What E6 showed, assembled from registered and post-hoc pieces

Three regimes on the coupled arm, instance g1_h0.05:

1. Converged, lambda at or above 0.99. Gains land at the predicted
   equilibria within 5 to 7 percent. Measured cost matches the analytic cost
   of the fitted gain to about 1 percent. Registered, part of E6's record.
2. Stable but biased, lambda near 0.98. No intermittency at any threshold
   (late-half maxima within 15 percent of medians, verified independently).
   Cost elevated 20 to 40 percent, and 4 of 5 seeds sit inside the plus or
   minus 15 percent band around J(k_AC(0.98)) = 0.2781, mean 0.2849. The
   aliased-equilibrium law holds here. POST-HOC for E6: P6.1 did not
   register 0.98, so this regime's confirmation awaits E7.
3. Unstable, lambda at or below 0.95. Recurrent off-manifold excursions,
   cost exceeds every equilibrium prediction, quality is stopping-time
   dependent. Registered as P6.1's failure plus the curves analysis.

Consequence: P6.1 failed because its registered set {0.8, 0.9, 0.95} lies
entirely in regime 3, outside the equilibrium law's domain of validity. The
domain boundary, the instability onset, is a SECOND critical lambda. On this
instance it sits between 0.95 and 0.98, while the cost-law lambda_half is
0.98538. The two thresholds are distinct.

## 2. Expectations for E7, stated before its data is read

- X1. On every instance, the intermittency onset (E_selfabs switching from
  zero to positive, boundary cells treated as in the pilot precedent) sits
  AT OR BELOW that instance's frozen lambda_half. Onset above lambda_half on
  any instance breaks the picture.
- X2. On every instance, cells between the onset and lambda_half, if the
  grid resolves any, show measured J inside the 15 percent band around
  J(k_AC(lambda)) with no intermittency. This is regime 2 confirmed
  prospectively at fresh instances.
- X3. The g4_h0.05 instance has the lowest knee, so most of its grid is
  above the horizon. Its cells at or above the knee read E_selfabs zero.
  This is the onset detector's false-positive test, and it is the same
  condition as Code's third tripwire, reread through the two-threshold lens.
- X4. P7.1's half-collapse estimator and the intermittency onset are
  estimators of DIFFERENT quantities and are expected to disagree by
  roughly the width of regime 2. Their gap per instance is a measurement of
  that width, not an error in either.

These are exploratory. They alter no frozen scoring. If they fail, they fail
in the open.

## 3. Consequence for the shipped diagnostic

The tool detects the instability onset, not lambda_half. It is SILENT in
regime 2, where cost is elevated but stable and reproducible. That silent
band is a documented scope limitation: a practitioner at 0.98 on this
instance has a 30 percent excess-cost problem the intermittency tool will
not flag. The bias indicator for regime 2 is a separate, later question.
D_gaplam is a natural candidate for it, which is a better use of that
statistic than below-horizon detection, where its window died at 1.14x.

## 4. Consequence for A-5 Tier 1b

The discriminating experiment (pure linear-gain policy at 0.9 versus 0.99)
now has a sharper target. Whatever mechanism it selects must predict the
LOCATION of the instability onset, the second critical lambda, not just its
existence. The onset bracket from E6, between 0.95 and 0.98, and the four
brackets E7 will supply per instance, are the quantitative targets any
mechanism theory has to hit.
