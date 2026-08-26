# E13 findings: the threshold survives breaking LQG exactness

Written 2026-08-21, after scoring. Prereg frozen the same day (commit
`12ccb86`, before any run; `--check-clean` passed at run start). All 72 runs
completed; scored by the frozen `prereg_e13.py`, verdicts quoted from its
output.

## Verdicts, as registered

    G13   (instrument, nl = 0):   PASS
    P16.1 (nl = 2.24, rho = 0.10): PASS
    P16.2 (nl = 11.2, rho = 0.50): PASS

## The numbers

| arm | measured lambda_half | vs linearized 0.983837 | vs pilot (pinned gains) | tolerance |
|---|---|---|---|---|
| nl = 0 (control) | 0.983819 | 0.000018 | 0.00001 | 0.005 |
| nl = 2.24 | 0.985529 | 0.00169 | 0.0000005 | 0.005 |
| nl = 11.2 | 0.985542 | 0.00171 | 0.00005 | 0.005 |

Profile structure J(0.8)/J(1.0): 1.335 (control) -> 1.818 (mild) -> 3.375
(strong); the floor cost moves 0.218 -> 0.160 -> 0.086, a factor of 2.5,
while the below-knee plateau moves by under 0.2 percent.

## What this establishes

The cost-law threshold's LOCATION is invariant under a cubic damping that
changes the cost landscape it lives in by a factor of 2.5. The linearized
theory locates the measured threshold within 0.0017 at a perturbation equal
to half the dominant balance at the above-knee equilibrium — and the reason
is structural, not accidental: high gain suppresses v, so the same
nonlinearity that reshapes the above-knee floor is inert on the inflated
branch (E[v^2] differs by ~100x between branches), and lambda_half is set by
where the equilibrium collapses, not by the cost values on either side.

This closes the linearity prong of the "2-D LQG toy" objection with
experimental, preregistered, prospective evidence. The other two prongs are
already closed by computation: kernel/dimension generality
(`kernel-generality-findings.md`: same structure at state dims 2-4, any Prony
kernel, port validated to 1e-7) and drive-distribution robustness of the cost
law (parent project, `lemma-a-findings.md`: cost at equilibrium within 1-7
percent across distribution-matched baths whose equilibrium locations move
3.4x). Together: the results this paper rests on (thresholds and cost levels)
are the ones that survive leaving the exactly-solvable family in every
direction we tested; the results that failed (equilibrium locations) are the
ones the theory itself marks fragile.

## Two unscored observations, recorded as exploratory

1. The trained strong-nl floor (0.0861) sits BELOW the pinned-gain pilot's
   (0.1077): under heavy cubic damping the E10-style downward gain drift
   lands in a cheaper region than the linearized equilibrium — the drift that
   costs accuracy on the linear family pays under this nonlinearity. Not
   scored, not claimed; noted because it is visible in the released rows.
2. Agreement with the pilot's pinned-gain lambda_half (5e-7 to 5e-5) is far
   tighter than with the linearized target (1.7e-3). The pilot's
   deployed-J-at-pinned-gains computation is evidently the right effective
   theory for the trained measurement; the linearized value is the frozen,
   registered target and the one the verdicts use.

## Provenance

Runs executed locally (Idils-MacBook-Pro-3.local, torch 2.9.1, 6 workers,
0.86 h wall-clock; recorded in results/logs/e13_host.txt), not on Myriad —
disclosed in the prereg before any run, with the nl = 0 control arm
calibrating this host and budget: its lambda_half reproduces the frozen
linearized value to 1.8e-5, which is the instrument-validity statement the
G13 gate exists to make. Environment map (4-substep cubic), harness dispatch
fix, and the pre-freeze pilot are documented in prereg_e13.py's docstring;
the four defects caught before freezing (integrator overflow, tolerance
admitting a grid-step shift, argparse rejection, silent episodic-evaluator
dispatch) are recorded there rather than here.
