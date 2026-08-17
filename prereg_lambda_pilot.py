"""Preregistration for the lambda-audit pilot (Phase 1 of lambda-audit-plan.md).

FROZEN 2026-08-08, before any run of the pilot grid anywhere. No sentinel
TODOs remain. Edits after the first array submission are amendments and must
be dated, justified, and made before any below-knee data exists.

The freeze is mechanical: ship_to_myriad.sh runs this file as a pre-flight
and refuses to ship if _assert_frozen() fails (E4 lesson: frozen values live
in an executable step, never in a comment).

Usage after runs land:
    python prereg_lambda_pilot.py results/results_lambda_pilot.jsonl
"""

import json
import math
import sys

# ---------------------------------------------------------------------------
# Design constants. FROZEN 2026-08-08.
# ---------------------------------------------------------------------------

# Grid: 3 envs x 4 architectures x 4 lambdas x 5 seeds = 240 runs, 15M steps.
# RepeatPreviousEasy was added to the original 2-env pilot on 2026-08-08,
# BEFORE any pilot run, for two reasons: it is the crisp published G1 anchor
# (six models at exactly 1.000 vs MLP at -0.320), and it is a third control
# arm (its knee at lambda ~ 0.76 sits below the whole grid, so the theory
# predicts NO ranking movement there; if rankings scramble on Easy at 0.9,
# the knee placement is refuted, not just decorated).

ENVS = {
    # env_id: (hidden timescale in steps, predicted knee lambda, role)
    "RepeatPreviousEasy": (4, 0.76, "control_above_knee_memory_matters"),
    "RepeatPreviousMedium": (32, 0.978, "transition"),
    "PositionOnlyCartPoleEasy": (8, 0.88, "control_above_knee_borderline"),
}
TRANSITION_ENV = "RepeatPreviousMedium"

ARCHS = ["mlp", "stack8", "stack64", "gru128"]
LAMBDAS = [0.9, 0.95, 0.99, 1.0]
SEEDS = [1, 2, 3, 4, 5]
TOTAL_STEPS = 15_000_000
GAMMA = 0.99

# Memory-capable per env: window >= hidden timescale, or recurrent.
MEMORY_CAPABLE = {
    "RepeatPreviousEasy": ["stack8", "stack64", "gru128"],   # k=4
    "RepeatPreviousMedium": ["stack64", "gru128"],           # k=32
    "PositionOnlyCartPoleEasy": ["stack8", "stack64", "gru128"],  # tau~8
}

# ---------------------------------------------------------------------------
# Published anchors (POPGym paper, Table 3, arXiv:2303.01859v3, MMER mu+-sigma,
# 3 trials, their regime: lambda=1.0, batch 65,536, lr 5e-5, 15M steps).
# Extracted 2026-08-08 from the full-results table. Fr.Stack is their frame
# stacking baseline; PositionOnlyCartPole was named StatelessCartPole.
# ---------------------------------------------------------------------------

G1_PUBLISHED_MMER = {
    "RepeatPreviousEasy": {
        "mlp": (-0.320, 0.007), "gru": (1.000, 0.000), "frstack": (1.000, 0.000),
        "lstm": (1.000, 0.000), "posmlp": (-0.336, 0.014), "s4d": (-0.473, 0.003),
    },
    "StatelessCartPoleEasy": {
        "mlp": (0.722, 0.001), "gru": (1.000, 0.000), "frstack": (1.000, 0.000),
        "lstm": (1.000, 0.000), "posmlp": (0.967, 0.006), "s4d": (0.514, 0.076),
    },
    # Context, NOT a G1 anchor: at their hyperparameters (229 gradient
    # updates) Medium is essentially unsolved by everything except LMU.
    # Our regime is ~7300 updates, so solvability at lambda=1.0 is open and
    # is gated by G1c below, checked BEFORE the main array.
    "RepeatPreviousMedium_context": {
        "mlp": (-0.486, 0.001), "gru": (-0.315, 0.017), "frstack": (-0.484, 0.003),
        "lstm": (-0.284, 0.024), "lmu": (0.789, 0.288),
    },
}

# ---------------------------------------------------------------------------
# Gates. All values FROZEN 2026-08-08. Rationale recorded per gate.
# IQM = interquartile mean of deployed_return over the 5 seeds of a cell.
# ---------------------------------------------------------------------------

# G1a, sanity on the published anchor env. At lambda=1.0 on RepeatPreviousEasy:
# every memory-capable arch IQM >= 0.90 (published: 1.000; our smoke run hit
# 0.990 in 600k steps) and mlp IQM <= 0.0 (published: -0.320).
G1A_MEMORY_MIN = 0.90
G1A_MLP_MAX = 0.0

# G1b, sanity on the second control. At lambda=1.0 on PositionOnlyCartPoleEasy:
# min over memory-capable IQMs minus mlp IQM >= 0.10. Published gap is 0.278
# (1.000 vs 0.722); half of it allows for harness deltas while still
# requiring a real separation.
G1B_MIN_GAP = 0.10

# G1c, capability of the transition env in OUR regime. At lambda=1.0 on
# RepeatPreviousMedium: MEM(1.0) - IQM(mlp, 1.0) >= 0.5 (G1_MIN_GAP), where
# MEM(lam) = max over {stack64, gru128} of the cell IQM. If a memory arch
# solves (~1.0) and mlp sits at chance (~-0.49) the gap is ~1.5, so 0.5 is a
# forgiving floor that still means "memory clearly works here at lambda=1".
#
# AMENDED PRE-LAUNCH 2026-08-08, before any below-knee run anywhere: the gate
# was first written as gru128-specific. Sandbox probes (seed 101, disclosed,
# probe/ dir) then showed stack64 solves Medium at lambda=1.0 (0.997 in 4M
# steps) while gru128 is still at chance at 800k, so a gru-specific gate
# could fail on trainer speed while the flagship claim is architecture
# generic. Gating on the best memory-capable arch is also the conservative
# choice for G2: if EITHER memory arch keeps its advantage at 0.9, G2 fails.
G1_MIN_GAP = 0.5
G2_MEMORY_ARCHS = ["stack64", "gru128"]

# G1c pre-array protocol: before the 240-task array, run gru128 and stack64
# on Medium at lambda=1.0, seed 101 (configs_capability.txt, SEED_OFFSET=100).
# If NEITHER reaches deployed_return >= 0.5, invoke the contingency: the
# transition env becomes RepeatPreviousK16 (k=16, knee ~ 0.947, already in
# the harness registry), this file is amended and recommitted BEFORE any
# below-knee run. That keeps every below-knee cell pre-registered.
G1C_CAPABILITY_MIN = 0.5

# G2, the flagship gate. On the transition env, with gap(lam) =
# MEM(lam) - IQM(mlp, lam): PASS iff gap(0.9) < 0.25 * gap(1.0), conditional
# on G1c passing. Rationale: lambda=0.9 has GAE horizon ~9 steps against a
# hidden timescale of 32, deeply below the knee, and the closed-form law
# predicts near-total collapse of the memory advantage, not a mild dent.
# Requiring 75% of the gap to vanish is strict enough to mean something and
# loose enough to tolerate residual learning inside 15M steps. rliable
# stratified-bootstrap CIs are computed in the analysis notebook and
# reported alongside; the frozen gate is this point rule.
G2_SHRINK_FRACTION = 0.25

# G3, the blind diagnostic. Statistic: gap_lambda = mean(G_lam - G_mc)/std(G_mc)
# from the final evaluation, chosen over gap_value because it is identically
# zero at lambda=1 by construction (no false positives at the top of the
# grid, and a built-in implementation check) and because at the TD(lambda)
# fixed point the two coincide anyway. Threshold 0.25: double the largest
# above-knee value observed in harness validation (|-0.12|, cartpole at
# 0.95) and below every below-knee smoke value observed (|-0.28| to |-0.39|
# at only 60k steps). Classification rule: a run is labeled below-knee by
# knee arithmetic alone (transition env at lambda in {0.9, 0.95}); the
# transition env at 0.99 is EXCLUDED as boundary (knee 0.978 sits between);
# everything else is above-knee. Note the honest edge: cartpole at 0.9 is
# labeled above-knee with its knee at 0.88, barely. PASS iff balanced
# accuracy >= 0.80 at the frozen threshold.
G3_STATISTIC = "gap_lambda"
G3_FLAG_THRESHOLD = 0.25
G3_MIN_BALANCED_ACCURACY = 0.80
G3_EXCLUDED_CELLS = [(TRANSITION_ENV, 0.99)]

PREDICTIONS = """
P1 (G1): at lambda=1.0, Easy separates exactly as published (memory ~1.0,
   mlp at chance); cartpole separates (memory ~1.0, mlp below); Medium
   separates in our regime pending G1c.
P2 (G2, flagship): on the transition env at lambda in {0.9, 0.95}, all
   architectures compress toward the memoryless level, INCLUDING gru128 and
   stack64 (input sufficiency does not rescue below the knee; the stack32
   torch result predicts this).
P3: Easy and cartpole rankings are approximately lambda-invariant across the
   grid. If Easy scrambles at 0.9, the knee PLACEMENT is refuted.
P4 (G3): gap_lambda classifies below-knee vs above-knee cells blind to
   architecture at balanced accuracy >= 0.8.
"""

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _assert_frozen():
    frozen = [G1A_MEMORY_MIN, G1A_MLP_MAX, G1B_MIN_GAP, G1_MIN_GAP,
              G1C_CAPABILITY_MIN, G2_SHRINK_FRACTION, G3_STATISTIC,
              G3_FLAG_THRESHOLD, G3_MIN_BALANCED_ACCURACY]
    if any(v is None for v in frozen):
        raise SystemExit("PREREG NOT FROZEN. Do not launch, do not score.")
    for env, d in G1_PUBLISHED_MMER.items():
        if any(v is None or v[0] is None for v in d.values()):
            raise SystemExit(f"PREREG NOT FROZEN: anchors missing for {env}.")


def iqm(xs):
    xs = sorted(xs)
    n = len(xs)
    lo, hi = n // 4, n - n // 4
    mid = xs[lo:hi] if hi > lo else xs
    return sum(mid) / len(mid)


def load_results(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cell(rows, env, arch, lam, field="deployed_return"):
    return [r[field] for r in rows
            if r["env"] == env and r["arch"] == arch
            and math.isclose(r["lam"], lam) and r["seed"] in SEEDS]


def score_gates(path):
    _assert_frozen()
    rows = load_results(path)
    expected = len(ENVS) * len(ARCHS) * len(LAMBDAS) * len(SEEDS)
    n_grid = len([r for r in rows if r.get("seed") in SEEDS])
    print(f"grid rows: {n_grid} / expected {expected}")
    verdicts = {}

    # G1a
    easy_mem = min(iqm(cell(rows, "RepeatPreviousEasy", a, 1.0))
                   for a in MEMORY_CAPABLE["RepeatPreviousEasy"])
    easy_mlp = iqm(cell(rows, "RepeatPreviousEasy", "mlp", 1.0))
    verdicts["G1a"] = easy_mem >= G1A_MEMORY_MIN and easy_mlp <= G1A_MLP_MAX
    print(f"G1a: min memory IQM {easy_mem:.3f} (>= {G1A_MEMORY_MIN}), "
          f"mlp {easy_mlp:.3f} (<= {G1A_MLP_MAX}) -> "
          f"{'PASS' if verdicts['G1a'] else 'FAIL'}")

    # G1b
    cp_mem = min(iqm(cell(rows, "PositionOnlyCartPoleEasy", a, 1.0))
                 for a in MEMORY_CAPABLE["PositionOnlyCartPoleEasy"])
    cp_mlp = iqm(cell(rows, "PositionOnlyCartPoleEasy", "mlp", 1.0))
    verdicts["G1b"] = (cp_mem - cp_mlp) >= G1B_MIN_GAP
    print(f"G1b: memory-mlp gap {cp_mem - cp_mlp:.3f} (>= {G1B_MIN_GAP}) -> "
          f"{'PASS' if verdicts['G1b'] else 'FAIL'}")

    # G1c and G2 on the transition env, memory side = best of the
    # memory-capable architectures (see the dated amendment above)
    def mem(lam):
        return max(iqm(cell(rows, TRANSITION_ENV, a, lam))
                   for a in G2_MEMORY_ARCHS)

    gap10 = mem(1.0) - iqm(cell(rows, TRANSITION_ENV, "mlp", 1.0))
    gap09 = mem(0.9) - iqm(cell(rows, TRANSITION_ENV, "mlp", 0.9))
    verdicts["G1c"] = gap10 >= G1_MIN_GAP
    print(f"G1c: transition gap at 1.0 = {gap10:.3f} (>= {G1_MIN_GAP}) -> "
          f"{'PASS' if verdicts['G1c'] else 'FAIL'}")
    if verdicts["G1c"]:
        verdicts["G2"] = gap09 < G2_SHRINK_FRACTION * gap10
        print(f"G2 : gap at 0.9 = {gap09:.3f} (< {G2_SHRINK_FRACTION} x "
              f"{gap10:.3f} = {G2_SHRINK_FRACTION * gap10:.3f}) -> "
              f"{'PASS' if verdicts['G2'] else 'FAIL'}")
    else:
        verdicts["G2"] = None
        print("G2 : NOT SCORED, G1c failed (transition env uninformative)")

    # G3, blind classification
    tp = fn = tn = fp = 0
    for r in rows:
        if r.get("seed") not in SEEDS or r.get(G3_STATISTIC) is None:
            continue
        if (r["env"], r["lam"]) in [(e, l) for e, l in G3_EXCLUDED_CELLS]:
            continue
        below = r["env"] == TRANSITION_ENV and r["lam"] in (0.9, 0.95)
        flagged = abs(r[G3_STATISTIC]) > G3_FLAG_THRESHOLD
        if below and flagged:
            tp += 1
        elif below:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    if tp + fn > 0 and tn + fp > 0:
        bal = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        verdicts["G3"] = bal >= G3_MIN_BALANCED_ACCURACY
        print(f"G3 : balanced accuracy {bal:.3f} "
              f"(tp {tp} fn {fn} tn {tn} fp {fp}) -> "
              f"{'PASS' if verdicts['G3'] else 'FAIL'}")
    else:
        verdicts["G3"] = None
        print("G3 : NOT SCORED, missing cells")

    print("VERDICTS:", verdicts)
    return verdicts


if __name__ == "__main__":
    if len(sys.argv) != 2:
        _assert_frozen()
        print("Prereg is FROZEN and self-consistent.")
        print(PREDICTIONS)
    else:
        score_gates(sys.argv[1])
