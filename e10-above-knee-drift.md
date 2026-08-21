# E10 above-knee drift: diagnosis of the P11.1/P11.2 failures

Written 2026-08-21, after E10 was scored. P11.1 and P11.2 remain FAIL as
registered; nothing here rescores them. This document explains what failed,
verifies the explanation numerically, and bounds what the failure does and
does not touch. Every number is computed by `analysis/e10_drift_diagnosis.py`
from `results/results/results_e10.jsonl` and the constants frozen in
`prereg_e10.py`.

## What failed, precisely

P11.1 required every linear cell to hold its registered equilibrium,
|final_gain / k_AC(λ) − 1| ≤ 0.15, in all 32 cells. Nineteen cells broke the
tolerance. P11.2 required |J_rate / J_AC(λ) − 1| ≤ 0.10 in all 32 cells; four
broke it (g1_h0.02 at λ ∈ {0.999, 1.0}: 14.1%; g2_h0.05 at 0.99: 10.0%;
g4_h0.05 at 0.9: 10.9%).

## Four verified facts

**1. The drift is systematically downward — in every run, everywhere.**
All 96 linear runs (32 cells × 3 seeds), passing cells included, finished
below their cell's k_AC(λ). Under any symmetric scatter around the
equilibrium this has probability 2⁻⁹⁶. Within failing cells the three seeds
agree to well within the drift (spread/drift 0.3–0.6). This is a
deterministic bias, not diffusion.

**2. The cost map is intact — including in every failing cell.**
Evaluating the analytic cost at the *measured* gain, J_measured /
J_analytic(k_measured) lies in [0.9895, 1.0105] across all 32 cells, median
0.9909. The four P11.2 violations are therefore entirely inherited: they are
P11.1's positional errors pushed through a cost map that is itself correct to
≈1%. (P11.2 compares against J at the *registered* gain; the ≥10% excesses
appear exactly where the drift moved k far enough that J(k_obs) ≠ J(k_AC).)
This reproduces, in-domain and on-manifold, the E7 decomposition (median
J_measured / J_analytic(k_measured) = 0.994 on uncontaminated cells).

**3. Failure aligns with the branch of the equilibrium map, not with λ.**
The aliased equilibrium map has an inflated branch (k_AC ≈ 11–60, below the
instance's knee) and a collapsed branch (k_AC ≈ 1.1–2.2, above it). All 18
collapsed-branch cells fail; 13 of 14 inflated-branch cells pass. The single
inflated-branch failure, g4_h0.05 at λ = 0.8 (22.5% vs 15% tolerance), is the
cell nearest its instance's fold (knee at 0.856) and shows the anomalously
soft restoring stiffness expected there — 10× lower than its instance
siblings. The failure is not "high λ": g1_h0.02 passes at λ = 0.995 with 0.3%
drift and fails at 0.999, because that is where *its* branch collapses.

**4. The absolute displacement is the same order everywhere; the relative
tolerance is what localises failure.** Passing cells undershoot by 0.08–2.6
in absolute gain units; failing collapsed-branch cells by 0.2–1.3. What
changes across the knee is the denominator: against k_AC ≈ 11–60 such a
displacement is 0.3–11%; against k_AC ≈ 1.1–2.2 it is 26–59%.

## Reading

The parsimonious account consistent with all 32 cells is: both frozen maps
are intact (the cost map verified to ≤1.05% at every measured gain; the
equilibrium map's values reproduced as roots of the exact GAE(λ) gradient to
|ĝ| ≤ 6×10⁻⁹, a provenance check), and the PPO estimator carries a small,
systematically negative bias relative to that exact gradient — visible as a
0.3–11% undershoot even in every passing cell — whose *relative* effect is
amplified wherever the equilibrium is small or the branch is near its fold.

The preregistration's fixed reading — "P11.1/P11.2 failing anywhere means the
equilibrium map or the cost function is wrong at that cell" — anticipated
arrival failure and was written to close the reachability excuse. E10 did
close it: the policy was placed at k_AC and the large-k_AC cells held it. What
the design did not separate is map error from estimator bias in the holding
dynamics. The registered verdicts stand; the decomposition above is what they
mean.

### Candidates excluded, with the test that excluded them

Discount mismatch: the harness discount equals exp(−ρh) bitwise, per
instance (0.9975031223974601 for h = 0.05; 0.999000499833375 for h = 0.02) —
the theory's β is already the training discount. GAE tail truncation at
num_steps = 128: the tail weight (βλ)¹²⁸ is 0.463 in a cell that passes with
0.3% drift (g1_h0.02, λ = 0.995) and 0.201 in one that fails at 58.8%
(g1_h0.05, λ = 0.99) — no alignment. Entropy bonus: σ is fixed, so policy
entropy is independent of the gain and contributes no gradient.

Not adjudicated here, and named as candidates only: critic lag (the harness
critic is a network trained by SGD, not the exact LSTD(λ) fixed point the
gradient theory assumes), the clipped-surrogate asymmetry, and Adam
preconditioning at small gradient scale. Distinguishing these is future work;
none of the audit's claims depends on which it is.

## The same softness explains the P11.4 misses

P11.4's three misses (18/21 stable cells vs the registered 19) are the three
scored cells with the smallest positive S: g1_h0.05 λ = 0.98 (S = +2.3×10⁻⁴),
g2_h0.05 λ = 0.95 (+6.0×10⁻⁴), g4_h0.05 λ = 0.8 (+4.4×10⁻⁴). Each is the
scored cell immediately below its instance's knee — precisely where S crosses
zero and the theory itself puts the restoring rate near zero. The instance
with no miss, g1_h0.02, is the one whose knee-adjacent cells (|S| = 7.7×10⁻⁵,
1.8×10⁻⁵) fell under the registered floor and were excluded — the floor did
there exactly what it was meant to do everywhere.

The outcome data show the floor was calibrated on the wrong side: growth
resolved down to |S| = 1.3×10⁻⁴ (all 8 unstable cells read correctly), but
decay resolved only at S ≥ 1.1×10⁻², with perfect separation between the two
groups. Growth is self-amplifying and aided by diffusion; decay must beat the
stationary noise floor of the bias coordinate under a weak restoring force.
The pilot calibrated the floor on a growth cell (|S| ≈ 9×10⁻⁴ resolved within
3M steps) and applied it symmetrically. At the 15M budget the stable-side
resolvability threshold lies somewhere in (6.0×10⁻⁴, 1.1×10⁻²). The minimal
reading of the three misses is "unresolved at budget"; a genuine sign error
at |S| < 10⁻³ cannot be excluded on these data. g4_h0.05 λ = 0.8 appears in
both failure lists — near the fold, everything is soft.

## What this does and does not touch

P11.3 — λ_half within 0.01 of the grid-matched analytic target on all four
instances, the audit's strongest result — is estimated from the measured cost
profile J(k_obs(λ)), and fact 2 shows that profile sits on the analytic cost
map to ≈1% everywhere. The quantitative law rests on the map that survived,
not on the holding accuracy that failed. Conversely, any future use of this
harness to test *point* predictions of equilibrium location on the collapsed
branch needs either a bias-corrected estimator or tolerances set in absolute
gain units; a relative gate at small k_AC measures the harness, not the map.

## Regenerating

    python3 analysis/e10_drift_diagnosis.py results/results/results_e10.jsonl

The script asserts the discount identity, reproduces the frozen k_AC table as
gradient roots, and prints the per-cell table behind every claim above.
