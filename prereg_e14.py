"""Preregistration E14: PositionOnlyPendulum, a second in-scope external test.

FROZEN 2026-09-11, BEFORE ANY E14 RUN EXISTS. Verified mechanically:
    python3 prereg_e14.py --check-clean results/results/results_e14.jsonl
The deploy script runs this file as a pre-flight; a sentinel left in place
(FROZEN_ON or FIX_SIGMA None) stops the launch.

THE CLAIM, and why one more environment is worth 160 runs
---------------------------------------------------------
The paper's scope proposition says the aliased-critic confound needs a hidden
coordinate the ACTION can steer. The only external environment on which that
condition holds and the predictions were scored is PositionOnlyCartPole (E9,
P9.1-P9.3, all PASS). The two POPGym games (E11, E12) do not satisfy the
condition and, as the proposition says, show nothing. One in-scope external
result is thin. E14 REPLICATES E9's registered structure on independent
dynamics: the same three predictions, the same harness, the same lambda grid,
a different physical system. It is a replication, not a new claim, and it is
scored whichever way it comes out.

WHY THIS ENVIRONMENT, verified before submission
------------------------------------------------
E12's error was choosing an environment from the wrong table and placing its
knee from an episode length. Both checks are done here, in writing.

1. STEERED HIDDEN STATE. The observation is [cos theta, sin theta]; the
   angular velocity theta_dot is hidden and every observation aliases every
   theta_dot. The torque u enters theta_dot directly,
       theta_dot' = theta_dot + (15 sin theta + 3 u) dt,   dt = 0.05,
   i.e. delta theta_dot = 0.15 u per step. This is the scope proposition's
   positive side by construction, not by argument: the hidden coordinate is
   the one the action moves. (In E11/E12 the latent was fixed at reset and
   actions only revealed it.)

2. MEMORY DEMONSTRABLY HELPS, from POPGym's own results table (the paper calls
   this environment "Stateless Pendulum"; the code calls it
   PositionOnlyPendulum; ICLR 2023, appendix table, MMER mean (sd)):
       Easy    MLP 0.448 (0.005)   GRU 0.913 (0.002)   gap +0.465
       Medium  MLP 0.455 (0.025)   GRU 0.884 (0.002)   gap +0.429
       Hard    MLP 0.477 (0.030)   GRU 0.828 (0.001)   gap +0.351
   Both arms are far off the floor and the memory advantage is the largest of
   any environment this audit has used. A memoryless policy CANNOT solve the
   task even in principle: any policy u(theta) is a conservative torque field,
   so it cannot damp the swing, only reshape the potential. The MLP-to-GRU gap
   is therefore the value of estimating theta_dot from history, which is
   exactly the quantity the aliased critic is claimed to distort.

3. THE GRID STRADDLES THE KNEE, placed from a RELAXATION TIME. The hidden
   coordinate's intrinsic timescale is the swing, not the episode. Small
   oscillation about the hanging point obeys theta'' = -(3g/2l) theta, so
   omega_0 = sqrt(15) = 3.873 rad/s and the period is 1.622 s = 32.4 steps.
   Along a swing the autocorrelation of theta_dot crosses zero at a quarter
   period (8.1 steps) and reaches its negative extremum at a half period
   (16.2 steps); a full rotation at max_speed 8 rad/s takes 15.7 steps.
   Memoryless closed loops are conservative (point 1 above), so no damping
   shortens these times. The bootstrap horizon H = (1 - gamma lambda)^-1
   sees the hidden velocity reverse once it exceeds the half period and sees
   the whole swing once it exceeds the period:
       H = 16.2 (half period)  ->  lambda = (1 - 1/H)/gamma = 0.948
       H = 32.4 (full period)  ->  lambda = 0.979
   The knee is therefore BRACKETED in [0.948, 0.979], not pinned. In the
   testbed the half-collapse sits at a horizon of 1.5-4.5 correlation times,
   so a bracket, not a point, is what the theory licenses here. The grid is
   scored against the bracket, never against a point inside it:
       BELOW     {0.8, 0.9}                      H = 4.8, 9.2: under the half
                                                 period (0.9 is just past the
                                                 quarter period, 8.1 steps)
       BOUNDARY  {0.95}                          H = 16.8, the half period:
                                                 inside the bracket, REPORTED,
                                                 NOT SCORED
       ABOVE     {0.98, 0.99, 0.995, 0.999, 1.0} H >= 51, above the full period
   This differs from E9's grouping, which put 0.95 in the above group on the
   strength of a pilot bracket rather than a derivation. E14 registers the
   derived grouping and says so. If the step is still present at 0.98 the
   bracket is wrong and P17.2 FAILS as registered.

ENVIRONMENT VERIFICATION
------------------------
Reimplemented natively in harness/envs_native.py (PositionOnlyPendulum
Easy/Medium/Hard) and verified BITWISE against popgym before any run:
14,400 steps of (observation, reward, terminated, truncated) AND the hidden
state (theta, theta_dot) identical across 8 seeds x 4 episodes per variant,
with actions drawn wider than the torque bound so the clip is exercised, and
the unseeded-reset RNG continuation identical (harness/test_env_equivalence.py).

HARNESS: FIRST CONTINUOUS-ACTION EXTERNAL ENVIRONMENT
-----------------------------------------------------
The policy head is the harness's existing Gaussian (used for every GLEAM
instance) with sigma PINNED at FIX_SIGMA rather than learned, for the reason
the paper gives in-domain: entropy collapse would otherwise be an
uncontrolled brake on the lambda reading, differing between arms. Deployed
evaluation uses the policy MEAN with the declared clip, the convention of
every continuous run in the audit, over 1000 episodes (E9 used 100; see
P17.2 for why); the discrete path is byte-identical to before
(three-environment regression, rows and curves identical). Every other
hyperparameter is E9's: gamma 0.99, 15M steps, 16 envs x 128 steps,
lr 3e-4, clip 0.2, 4 epochs, ent 0.01.

PRE-FREEZE SMOKE RUNS, DISCLOSED. To choose FIX_SIGMA and to confirm the new
head trains at all, mlp and gru128 were run at lambda = 1.0 ONLY, seed 101
(outside the registered seeds), on the Easy variant, in a scratch directory
that is not a results file: mlp at sigma in {0.3, 0.5, 1.0, learned} for 15M
steps, gru128 at sigma 0.5 for 5M steps. No run at any lambda below 1.0
existed when this file was frozen, so nothing that discriminates the
predictions was seen. The chosen value and the smoke numbers are recorded at
FIX_SIGMA below.

SCRIPTED ANCHORS (frozen; 20,000 episodes, seeds 2,000,000+, action rng 42,
analysis/e14_anchors.py; mean / IQM)
    Easy   uniform torque +0.2433 / +0.2589   zero torque +0.2494 / +0.2485
           bang-bang toward the top from the angle alone +0.3521 / +0.3116
    Hard   uniform torque +0.2403 / +0.2526   zero torque +0.2493 / +0.2490
           bang-bang toward the top from the angle alone +0.3586 / +0.3079
The bang-bang policy is the strongest thing a policy can do from the angle
alone without an estimate of theta_dot; it is the memoryless heuristic the
MLP arm is measured against, as E12's zero-count expansion was.

CAPABILITY GATES, per variant, scored first, from a 4-run PROBE that is
submitted and read BEFORE the grid (mlp and gru128 at lambda 1.0, seed 1, on
Easy and Hard). Deployed return of the final policy mean, IQM over the seeds
present (at the probe, the single value).
    G14a  mlp IQM at lambda 1.0 >= 0.30   off the floor (0.25) and learning;
                                          POPGym MLP 0.45-0.48
    G14b  gru IQM at lambda 1.0 >= 0.60   clear of the memoryless heuristic
                                          (0.31-0.35) and of POPGym's MLP, so
                                          a real memory advantage exists;
                                          POPGym GRU 0.83-0.91
    G14c  gap at lambda 1.0 >= 0.15       POPGym's gap is 0.35-0.47
A variant that fails any gate has its predictions NOT SCORED on that variant
and its grid is not submitted. Predictions are scored over the variants that
clear their gates and PASS only if they pass on every such variant; if no
variant clears, P17.1-P17.3 are NOT SCORED and E14 reports its gates. The
gates are re-read on the full grid (IQM over five seeds); a variant that
clears on the probe's single seed and fails on five is NOT SCORED, and both
readings are reported. A gate is not a finding.

THE REGISTERED PREDICTIONS (P9.1-P9.3 restated on the derived grouping, with
E12's hardened scoring). Seed-level bootstrap over the 5 seeds, 10,000
draws, seed 20260911, IQM over seeds within each cell. Scoring requires the
complete grid for every gated-in variant: 2 archs x 8 lambdas x 5 seeds, and
refuses an incomplete or duplicated cell rather than scoring around it.

P17.1  THE STEP BELOW THE KNEE. mlp's mean cell IQM over BELOW minus its mean
       cell IQM over ABOVE is negative with a 95% CI excluding zero, on every
       gated-in variant. No effect size is fixed, for the reason P9.1 gave.

P17.2  NO STEP ABOVE THE KNEE. On every gated-in variant, both of:
       (a) the mlp's mean cell IQM over {0.98, 0.99, 0.995, 0.999} is
           statistically EQUIVALENT to its lambda 1.0 cell: TOST, the 95%
           bootstrap interval of the difference lies entirely inside
           +/- 0.05;
       (b) no single above-knee cell's IQM differs from the lambda 1.0 cell
           by more than 0.05 in magnitude (point estimate), so a step hiding
           at one cell cannot be diluted away by the pooling in (a).
       0.05 is E12's margin; here it is 11-14% of POPGym's published memory
       advantage and 2.4x the step E9 measured on CartPoleEasy. A "CI
       contains zero" criterion was not used, for the reason recorded in
       prereg_e12.py: it becomes harder to satisfy as data improve. A
       per-cell Bonferroni TOST was written first and REJECTED before
       freezing on a power simulation (analysis/test_prereg_e14.py --power,
       pass rate under TRUE equivalence, 5 seeds, 40 replicates):
           seed sd   0.010  0.015  0.020  0.030  0.040
           per-cell   1.00   0.68   0.12   0.00   0.00
           registered 1.00   1.00   0.97   0.42   0.10
       i.e. the per-cell form fails the registered pattern on noise alone at
       any realistic spread. The registered form (a)+(b) resolves at a seed
       sd of 0.02 (E9's CartPole mlp spread was about 0.01; POPGym reports
       0.005-0.030 for this environment's MLP) and is a registered risk
       above that: a FAIL from spread alone is still reported as a FAIL.
       To hold the evaluation component of the spread down, E14 deploys with
       1000 final evaluation episodes rather than E9's 100 (a config-line
       setting; the training path is untouched).

P17.3  MEMORY ARM VARIANCE. gru128's across-seed variance of deployed return
       is highest at the lowest lambdas: pooled over gated-in variants,
       mean variance over {0.8, 0.9} divided by mean variance over
       {0.99, 1.0} >= 2.0, AND the ordering holds within each gated-in
       variant separately. Identical to P9.3.

REPORTED, NOT SCORED: the 0.95 boundary cell (mlp IQM against 1.0, with CI),
so the reader can see where inside the bracket the step sits.

HOW A MISS IS READ, fixed in advance. P17.1 failing: the memoryless step
below the knee does not replicate on pendulum dynamics; the E9 result stands
alone and the paper says so. P17.1 passing with P17.2 failing: the lambda
dependence is not threshold-like on this system (a trend, or a step at the
wrong place); reported. P17.3 failing: the low-lambda variance signature of
the memory arm (E9's pooled factor 42.6) is not general; reported. A
variant failing its gates: not a finding; named in the paper as not scored.

AMENDMENTS. None at freeze. Any edit after the first submission is an
amendment, dated and disclosed here; one made after data exist is labelled
POST-DATA. Verdict-neutral repairs (guards, messages) are permitted; a
constant, threshold, grouping or comparison is not.
"""
import json
import math
import random
import sys

# ---------------------------------------------------------------------------
# Design constants. FROZEN. Sentinels (None) stop the pre-flight.
# ---------------------------------------------------------------------------

FROZEN_ON = "2026-09-11"
ENVS = ("PositionOnlyPendulumEasy", "PositionOnlyPendulumHard")
SHORT = {"PositionOnlyPendulumEasy": "Easy", "PositionOnlyPendulumHard": "Hard"}
ARCHS = ("mlp", "gru128")
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
SEEDS = (1, 2, 3, 4, 5)
GAMMA = 0.99
TOTAL_STEPS = 15_000_000

# The exploration pin for the Gaussian head. Chosen from the pre-freeze smoke
# runs at lambda 1.0 only (see the header); their numbers are in
# FIX_SIGMA_SMOKE below (deployed return of the policy mean; the 20-episode
# mid-training evaluations of the same policies read 0.47-0.50 for every mlp
# because the first twenty evaluation seeds happen to start near the top,
# which is why E14 deploys with 1000 evaluation episodes).
FIX_SIGMA = 0.5
# Deployed return, policy mean, Easy, lambda 1.0, 15M steps unless noted;
# (n) is the number of evaluation episodes. Runs live in scratch, not in a
# results file. Training time per run on the smoke machine: mlp ~18 min,
# gru128 ~2 h at 15M (5M shown).
FIX_SIGMA_SMOKE = {
    ("mlp", 0.3, 101): (0.3931, 100),
    ("mlp", 0.5, 101): (0.3862, 100),
    ("mlp", 0.5, 102): (0.3658, 1000),
    ("mlp", 0.5, 103): (0.3484, 1000),
    ("mlp", 1.0, 101): (0.3867, 100),
    ("mlp", "learned", 101): (None, 0),         # killed at 7.3M steps by a
                                                # machine suspend; by then the
                                                # learned sigma had grown to
                                                # 2.2, larger than the torque
                                                # range, under the entropy
                                                # bonus -- the reason for a pin
    ("gru128", 0.5, 101): (0.8772, 100),        # 5M steps
}
# Reading: the mlp lands at 0.35-0.39 for every sigma, i.e. sigma does not
# decide whether the memoryless arm learns; 0.5 is kept because gru128 was
# verified at it and it is a quarter of the torque range. Scripted anchors
# are 0.25 (floor) and 0.31-0.35 (angle-only bang-bang); POPGym's MLP MMER,
# a max over training epochs, is 0.448. gru128 at 0.877 after 5M steps
# clears G14b by 0.28 and puts the gap at ~0.5 against G14c's 0.15.

# Knee bracket from the swing period (header, point 3). Not a scored constant;
# it determines the grouping below and is checked for consistency with it.
OMEGA0 = math.sqrt(15.0)                     # rad/s, small swing at the bottom
PERIOD_STEPS = 2 * math.pi / OMEGA0 / 0.05   # 32.4 steps
KNEE_BRACKET = ((1 - 2 / PERIOD_STEPS) / GAMMA,   # half period: 0.948
                (1 - 1 / PERIOD_STEPS) / GAMMA)   # full period: 0.979
BELOW = [0.8, 0.9]
BOUNDARY = [0.95]
ABOVE = [0.98, 0.99, 0.995, 0.999, 1.0]
REF_LAM = 1.0

# Gates (per variant).
G14A_MIN, G14B_MIN, G14C_MIN = 0.30, 0.60, 0.15

# Bootstrap.
BOOT_N, BOOT_SEED, CI = 10_000, 20260911, 95.0

# P17.2: pooled TOST margin and level, plus the per-cell point bound.
P172_CI = 95.0
P172_DELTA = 0.05
P172_POOL = [0.98, 0.99, 0.995, 0.999]
FINAL_EVAL_EPISODES = 1000

# P17.3, identical to P9.3.
P173_LOW = [0.8, 0.9]
P173_HIGH = [0.99, 1.0]
P173_MIN_RATIO = 2.0

ANCHORS = {
    "PositionOnlyPendulumEasy": {"uniform": 0.2589, "zero": 0.2485, "bangbang": 0.3116},
    "PositionOnlyPendulumHard": {"uniform": 0.2526, "zero": 0.2490, "bangbang": 0.3079},
}
POPGYM = {
    "PositionOnlyPendulumEasy": {"mlp": 0.448, "gru": 0.913},
    "PositionOnlyPendulumHard": {"mlp": 0.477, "gru": 0.828},
}


def _assert_frozen():
    frozen = [FROZEN_ON, FIX_SIGMA, G14A_MIN, G14B_MIN, G14C_MIN, P172_DELTA,
              P173_MIN_RATIO, BOOT_SEED]
    if any(v is None for v in frozen):
        raise SystemExit("PREREG NOT FROZEN. A sentinel is still None. "
                         "Do not launch, do not score.")
    if not (isinstance(FROZEN_ON, str) and len(FROZEN_ON) == 10):
        raise SystemExit("PREREG NOT FROZEN: FROZEN_ON is not a date.")
    if not (0.0 < float(FIX_SIGMA) <= 2.0):
        raise SystemExit("PREREG INCONSISTENT: FIX_SIGMA outside (0, 2].")
    lo, hi = KNEE_BRACKET
    if not (max(BELOW) < lo < BOUNDARY[0] < hi < min(ABOVE)):
        raise SystemExit("PREREG INCONSISTENT: grouping does not straddle the "
                         f"knee bracket [{lo:.3f}, {hi:.3f}].")
    if sorted(BELOW + BOUNDARY + ABOVE) != LAMBDAS or REF_LAM not in ABOVE:
        raise SystemExit("PREREG INCONSISTENT: lambda grouping.")
    if set(P173_LOW) - set(BELOW) or set(P173_HIGH) - set(ABOVE):
        raise SystemExit("PREREG INCONSISTENT: P17.3 groups.")
    if sorted(P172_POOL + [REF_LAM]) != sorted(ABOVE):
        raise SystemExit("PREREG INCONSISTENT: P17.2 pool must be ABOVE minus 1.0.")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def cell(rows, env, arch, lam):
    return [float(r["deployed_return"]) for r in rows
            if r.get("env") == env and r.get("arch") == arch
            and r.get("seed") in SEEDS
            and math.isclose(float(r.get("lam", -1)), lam)
            and r.get("deployed_return") is not None
            and math.isfinite(float(r["deployed_return"]))]


def iqm(xs):
    xs = sorted(xs)
    n = len(xs)
    lo, hi = n // 4, n - n // 4
    m = xs[lo:hi] if hi > lo else xs
    return sum(m) / len(m)


def _variance(xs):
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _ci(draws, level):
    draws.sort()
    return (draws[int(len(draws) * (100 - level) / 200)],
            draws[int(len(draws) * (1 - (100 - level) / 200)) - 1])


def _boot(rng, spec, level=CI):
    """spec: list of (cell, weight). CI of sum_i weight_i * IQM(resample_i),
    resampling seeds within each cell independently."""
    out = []
    for _ in range(BOOT_N):
        tot = 0.0
        for c, w in spec:
            tot += w * iqm([c[rng.randrange(len(c))] for _ in c])
        out.append(tot)
    return _ci(out, level)


def duplicate_cells(rows):
    counts = {}
    for r in rows:
        if r.get("env") not in ENVS:
            continue
        try:
            k = (r.get("env"), r.get("arch"), round(float(r.get("lam")), 6),
                 int(r.get("seed")))
        except (TypeError, ValueError):
            continue
        counts[k] = counts.get(k, 0) + 1
    return {k: n for k, n in counts.items() if n > 1}


def check_clean(path):
    _assert_frozen()
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"CLEAN: {path} does not exist. Freeze is legitimate.")
        return 0
    n = sum(1 for r in rows if r.get("env") in ENVS)
    print(f"{'NOT CLEAN' if n else 'CLEAN'}: {n} E14 rows in {path}.")
    return 2 if n else 0


def _v(b):
    return "NOT SCORED" if b is None else ("PASS" if b else "FAIL")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_gates(rows, verbose=True):
    """Per-variant gates from the lambda 1.0 cells. Returns (verdicts, gated_in)."""
    v, gated_in = {}, []
    for env in ENVS:
        s = SHORT[env]
        m1, g1 = cell(rows, env, "mlp", REF_LAM), cell(rows, env, "gru128", REF_LAM)
        if not m1 or not g1:
            if verbose:
                print(f"{s}: MISSING lambda=1.0 cell(s) "
                      f"(mlp {len(m1)}, gru128 {len(g1)} seeds); gates unreadable.")
            continue
        im, ig = iqm(m1), iqm(g1)
        ga, gb, gc = im >= G14A_MIN, ig >= G14B_MIN, (ig - im) >= G14C_MIN
        if verbose:
            print(f"{s}: G14a mlp(1.0) IQM {im:+.4f} over {len(m1)} seed(s) "
                  f"(>= {G14A_MIN}) -> {_v(ga)}   [POPGym MLP {POPGYM[env]['mlp']}, "
                  f"floor {ANCHORS[env]['uniform']}]")
            print(f"{s}: G14b gru(1.0) IQM {ig:+.4f} over {len(g1)} seed(s) "
                  f"(>= {G14B_MIN}) -> {_v(gb)}   [POPGym GRU {POPGYM[env]['gru']}, "
                  f"memoryless heuristic {ANCHORS[env]['bangbang']}]")
            print(f"{s}: G14c gap(1.0)     {ig - im:+.4f} (>= {G14C_MIN}) -> {_v(gc)}"
                  f"   [POPGym gap {POPGYM[env]['gru'] - POPGYM[env]['mlp']:+.3f}]")
        v[f"G14a-{s}"], v[f"G14b-{s}"], v[f"G14c-{s}"] = ga, gb, gc
        if ga and gb and gc:
            gated_in.append(env)
    return v, gated_in


def incomplete_cells(rows, envs):
    """(env, arch, lambda, n_seeds) for every cell that is not exactly full."""
    out = []
    for env in envs:
        for arch in ARCHS:
            for lam in LAMBDAS:
                n = len(cell(rows, env, arch, lam))
                if n != len(SEEDS):
                    out.append((env, arch, lam, n))
    return out


def score(path):
    _assert_frozen()
    rows = load(path)

    dups = duplicate_cells(rows)
    if dups:
        print(f"REFUSING TO SCORE: {len(dups)} duplicated (env, arch, lambda, "
              f"seed) cells, e.g. {sorted(dups)[0]}.")
        print("Separate the extra rows; do not score a file that double-counts "
              "a seed.")
        return 2

    print("=" * 74)
    print(" E14 capability gates (per variant, lambda 1.0)")
    print("=" * 74)
    v, gated_in = score_gates(rows)
    if not v:
        print("No lambda=1.0 cells at all; nothing to read. Is this the right file?")
        return 2
    if not gated_in:
        print("\nP17.1, P17.2, P17.3 NOT SCORED: no variant cleared its gates.")
        print("VERDICTS:", {k: _v(b) for k, b in v.items()})
        return 0
    not_in = [SHORT[e] for e in ENVS if e not in gated_in]
    print(f"\nGated in: {[SHORT[e] for e in gated_in]}"
          + (f"; NOT scored (gate failed): {not_in}" if not_in else ""))

    # Data sufficiency. The gates are readable at the probe stage; the
    # predictions are not scored until every gated-in variant's grid is
    # complete, cell by cell. Partial data are named, never scored around.
    inc = incomplete_cells(rows, gated_in)
    if inc:
        probe_stage = all(n <= 1 for *_, n in inc) and all(
            len(cell(rows, e, a, l)) == 0 for e in gated_in for a in ARCHS
            for l in LAMBDAS if l != REF_LAM)
        print("\nP17 NOT SCORABLE YET: " + (
            "probe stage, the grid has not been collected."
            if probe_stage else
            f"{len(inc)} cell(s) are not complete ({len(SEEDS)} seeds each):"))
        if not probe_stage:
            for env, arch, lam, n in inc:
                print(f"    {SHORT[env]:5} {arch:7} lambda={lam:<6} {n}/{len(SEEDS)} seeds")
        for env in gated_in:
            print(f"Next: bash myriad/deploy.sh <ucl_username> --e14{SHORT[env].lower()}")
        print("VERDICTS:", {k: _v(b) for k, b in v.items()})
        return 0

    print("\n" + "=" * 74)
    print(f" E14 predictions (knee bracket [{KNEE_BRACKET[0]:.3f}, "
          f"{KNEE_BRACKET[1]:.3f}]; below {BELOW}, above {ABOVE})")
    print("=" * 74)

    # -- P17.1 --------------------------------------------------------------
    ok1 = True
    for env in gated_in:
        low = [cell(rows, env, "mlp", l) for l in BELOW]
        high = [cell(rows, env, "mlp", l) for l in ABOVE]
        rng = random.Random(BOOT_SEED)
        spec = [(c, 1.0 / len(low)) for c in low] + [(c, -1.0 / len(high)) for c in high]
        lo, hi = _boot(rng, spec)
        point = (sum(iqm(c) for c in low) / len(low)
                 - sum(iqm(c) for c in high) / len(high))
        good = point < 0 and hi < 0
        ok1 = ok1 and good
        print(f"P17.1 {SHORT[env]}: mlp below-above step {point:+.4f}, {CI:.0f}% CI "
              f"[{lo:+.4f}, {hi:+.4f}] -> "
              f"{'negative, excludes zero' if good else 'DOES NOT QUALIFY'}")
    v["P17.1"] = ok1
    print(f"P17.1: {_v(ok1)}")

    # -- P17.2 --------------------------------------------------------------
    ok2 = True
    for env in gated_in:
        m1 = cell(rows, env, "mlp", REF_LAM)
        pool = [cell(rows, env, "mlp", l) for l in P172_POOL]
        rng = random.Random(BOOT_SEED + 10)
        spec = [(c, 1.0 / len(pool)) for c in pool] + [(m1, -1.0)]
        lo, hi = _boot(rng, spec, level=P172_CI)
        point = sum(iqm(c) for c in pool) / len(pool) - iqm(m1)
        same = (lo > -P172_DELTA) and (hi < P172_DELTA)          # TOST, (a)
        worst = max(abs(iqm(c) - iqm(m1)) for c in pool)         # (b)
        bounded = worst <= P172_DELTA
        good = same and bounded
        ok2 = ok2 and good
        print(f"  P17.2 {SHORT[env]} (a) pooled {P172_POOL} vs 1.0: {point:+.4f}, "
              f"{P172_CI:.0f}% CI [{lo:+.4f}, {hi:+.4f}] within +/-{P172_DELTA} "
              f"-> {'EQUIVALENT' if same else 'NOT equivalent'}")
        for l, c in zip(P172_POOL, pool):
            print(f"  P17.2 {SHORT[env]} (b) lambda {l:<6} vs 1.0: {iqm(c) - iqm(m1):+.4f}")
        print(f"  P17.2 {SHORT[env]} (b) largest |cell - 1.0| {worst:.4f} "
              f"(<= {P172_DELTA}) -> {'bounded' if bounded else 'EXCEEDED'}")
    v["P17.2"] = ok2
    print(f"P17.2: {_v(ok2)}")

    # -- P17.3 --------------------------------------------------------------
    per_variant_ok, lo_all, hi_all = True, [], []
    for env in gated_in:
        lo_v = [_variance(cell(rows, env, "gru128", l)) for l in P173_LOW]
        hi_v = [_variance(cell(rows, env, "gru128", l)) for l in P173_HIGH]
        lo_m, hi_m = sum(lo_v) / len(lo_v), sum(hi_v) / len(hi_v)
        lo_all.extend(lo_v)
        hi_all.extend(hi_v)
        ratio = lo_m / hi_m if hi_m > 0 else float("inf")
        good = ratio > 1.0
        per_variant_ok = per_variant_ok and good
        print(f"  P17.3 {SHORT[env]}: gru128 variance low {lo_m:.6f} vs high {hi_m:.6f}, "
              f"ratio {ratio:.2f} ({'ordered' if good else 'NOT ORDERED'})")
    pooled = ((sum(lo_all) / len(lo_all)) / (sum(hi_all) / len(hi_all))
              if sum(hi_all) > 0 else float("inf"))
    ok3 = bool(pooled >= P173_MIN_RATIO and per_variant_ok)
    v["P17.3"] = ok3
    print(f"  P17.3 pooled ratio {pooled:.2f} (>= {P173_MIN_RATIO}), ordering in "
          f"every variant: {per_variant_ok}")
    print(f"P17.3: {_v(ok3)}")

    # -- boundary cell, reported not scored ---------------------------------
    for env in gated_in:
        m1 = cell(rows, env, "mlp", REF_LAM)
        for lam in BOUNDARY:
            c = cell(rows, env, "mlp", lam)
            rng = random.Random(BOOT_SEED + 100)
            lo, hi = _boot(rng, [(c, 1.0), (m1, -1.0)])
            print(f"  boundary {SHORT[env]} lambda {lam} vs 1.0 (mlp): "
                  f"{iqm(c) - iqm(m1):+.4f}, {CI:.0f}% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"[reported, not scored]")

    print("\nVERDICTS:", {k: _v(b) for k, b in v.items()})
    return 0


def main(argv):
    _assert_frozen()
    if len(argv) > 2 and argv[1] == "--check-clean":
        return check_clean(argv[2])
    if len(argv) > 1 and argv[1] != "--check-clean":
        return score(argv[1])
    print(f"E14 prereg frozen {FROZEN_ON}. {', '.join(SHORT.values())}; "
          f"knee bracket [{KNEE_BRACKET[0]:.3f}, {KNEE_BRACKET[1]:.3f}]; "
          f"below {BELOW}, boundary {BOUNDARY} (reported), above {ABOVE}; "
          f"sigma pinned {FIX_SIGMA}. Gates G14a/b/c per variant from the "
          f"4-run probe, then P17.1 step, P17.2 no step above, P17.3 variance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
