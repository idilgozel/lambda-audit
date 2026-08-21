"""Preregistration E12: are POPGym's published scores lambda=1.0 scores?

FROZEN 2026-08-21, BEFORE ANY E12 RUN EXISTS. Verified mechanically:
    python3 prereg_e12.py --check-clean results/results/results_e12.jsonl

THE CLAIM, and why it is stronger than "one more environment works"
--------------------------------------------------------------------
POPGym ranked thirteen architectures at GAE lambda = 1.0 and reported, per
environment, how much memory is worth. Common practice uses lambda = 0.95.
If the measured value of memory differs between those two settings, then the
published numbers are lambda = 1.0 numbers, and an architecture decision made
from them is conditional on a hyperparameter the benchmark did not vary. That
is a statement about existing practice, not about our testbed.

E12 tests it on MineSweeperMedium, chosen against their Table 3 rather than
assumed. E11's failure was choosing Battleship, where their own paper says
"memory models do not noticeably outperform the MLP"; that check is now part
of the design and is recorded here.

WHY THIS ENVIRONMENT, verified before submission
------------------------------------------------
1. CLOSED LOOP. Clicking reveals tiles, so the information state is
   action-dependent: the scope law's positive side. (RepeatPrevious, the
   pilot's open-loop control, is the negative side and already scored: G2
   FAILED as registered, i.e. no effect where the law says none.)
2. MEMORY DEMONSTRABLY HELPS, from POPGym Table 3:
       MineSweeperMedium   MLP -0.158 (sd 0.006)   GRU -0.006 (sd 0.013)
   a +0.152 advantage, with both architectures far off the floor.
3. THE GRID STRADDLES THE KNEE. Hidden state persists all episode, so
   tau = max_episode_length = 30 steps. With gamma = 0.99 the bootstrap
   horizon equals tau at
       lambda_knee = (1 - 1/30)/0.99 = 0.97643
   Below: {0.8, 0.9, 0.95}. Above: {0.98, 0.99, 0.995, 0.999, 1.0}.
   The community default 0.95 is BELOW the knee; the benchmark's own 1.0 is
   above it. That placement is the whole experiment, and MineSweeperEasy was
   rejected for failing it (tau = 14, knee 0.938, so 0.95 sits above).
   ConcentrationEasy was rejected too: tau = 104 puts the ENTIRE grid below
   its knee, leaving no above-knee reference cell.

ENVIRONMENT VERIFICATION
------------------------
Reimplemented natively in harness/envs_native.py and verified BITWISE against
popgym before any run: 200 seeded boards and neighbour-count grids identical,
1,218 steps of (obs, reward, terminated, truncated) identical, unseeded-reset
RNG continuation identical (harness/test_minesweeper_equivalence.py).
Adaptations, ours: MultiDiscrete([6,6]) flattened to Discrete(36) (bijection);
previous action folded into the observation exactly as their
obs_requires_prev_action flag and PreviousAction wrapper intend, giving
[onehot7(count), onehot6(prev_row), onehot6(prev_col)].

SCRIPTED ANCHORS (frozen; 20,000 episodes, episode seeds 2,000,000+, action
rng 42, analysis/e12_baselines.py):
    uniform random (with repeats)            mean -0.4011   IQM -0.4281
    no-repeat                                mean -0.3904   IQM -0.4230
    no-repeat + zero-count safe expansion    mean -0.1350   IQM -0.1523

Recorded because it interprets the gap: the third policy is MEMORYLESS (it
uses only the current observation's zero-count) and lands at -0.152, which is
POPGym's MLP score to within noise. The MLP-to-GRU gap is therefore the value
of accumulating the revealed map ACROSS the episode -- exactly the quantity
whose measurement lambda is claimed to distort.

CAPABILITY GATES, scored first, and a 2-run PROBE is submitted and read
BEFORE the 80-run grid. E11 ran its gates after the grid and spent 119
core-hours to learn what 2 runs would have shown; the pilot's own protocol
(a 2-task capability job before the 240-task array) is restored here.

  G12a  mlp IQM at lambda 1.0 >= -0.30   (floor is -0.428, their MLP -0.158;
        this only asks that the memoryless arm is off the floor and learning)
  G12b  gru IQM at lambda 1.0 >= -0.10   (must clear the MEMORYLESS-heuristic
        plateau at -0.152, so a genuine memory advantage exists to modulate)
  G12c  gap at lambda 1.0, IQM_gru - IQM_mlp >= 0.05  (their gap is 0.152;
        without a gap at the benchmark's own setting there is nothing to vary)

If any gate fails, ALL of P15 is NOT SCORED and E12 reports the gate, exactly
as E11 did. A gate is not a finding.

THE REGISTERED PREDICTIONS
--------------------------
Seed-level bootstrap over 5 seeds, 10,000 draws, seed 20260821, 95% CI. IQM
over seeds. gap(l) := IQM_gru(l) - IQM_mlp(l).

P15.1  PRACTICAL HEADLINE. The measured memory advantage is larger at the
       community default than at the benchmark's own setting:
       gap(0.95) - gap(1.0) > 0, CI excluding zero.

P15.2  MECHANISM DISCRIMINATOR. The change is driven by the MEMORYLESS arm:
       (a) IQM_mlp(0.95) - IQM_mlp(1.0) < 0, CI excluding zero;
       (b) IQM_gru(0.95) - IQM_gru(1.0) has CI containing zero.
       This is what separates the aliased-critic mechanism from an ordinary
       bias-variance effect. Lower lambda reduces gradient variance, which
       would help BOTH arms and help the memory arm MORE, widening the gap
       with the mlp flat or improving. The registered claim is the opposite
       pattern: the mlp specifically degrades. On BattleshipEasy (E11) the
       observed pattern was the variance one -- gru moved, mlp sat at the
       floor -- which is why P15.2 exists and why P15.1 alone is not enough.

P15.3  THRESHOLD LOCATION. The mlp degradation onset bracket contains the
       computed knee 0.97643: mlp IQM at 0.98 and above is statistically
       EQUIVALENT to its value at 1.0 within a pre-specified margin, and at
       0.95 it is lower (that part is P15.2a, which P15.3 requires). Scored by
       TOST: the Bonferroni-corrected 98.75% interval for each of the four
       above-knee cells must lie entirely inside +/- 0.05, one third of
       POPGym's published memory advantage here. A "CI contains zero"
       criterion was rejected before freezing because it becomes harder to
       satisfy as data quality improves; both that flaw and the
       multiple-comparison inflation were found by running this scorer on
       synthetic data drawn from the predicted pattern.

AMENDMENT 1, 2026-08-21, POST-DATA, DISCLOSED. Scored thresholds unchanged.
--------------------------------------------------------------------------
State when written: results_e12.jsonl held ONE row, the probe's mlp arm at
lambda 1.0 seed 1 (deployed_return -0.14639285714285719); the gru probe task
was still in the queue. No gate had been read and no P15 cell existed.

Defect: score() read the gates, and on the PASS path went straight into P15.1,
which bootstraps lambda=0.95 cells. During the probe stage those cells do not
exist, so _boot divided by zero and the scorer crashed -- on the success path
only. A gate FAILURE printed correctly; a gate PASS traced back. Found by
running this scorer against three synthetic probe states (mlp-only, all-pass,
gru-fails) before the gru task returned.

Fix: a data-sufficiency guard between the gates and P15.1, in the same form as
the existing "MISSING lambda=1.0 cells" guard above it. If the lambda=0.95
cells are absent, print that the gates passed and that P15 awaits the grid,
then return. No constant, no threshold, no comparison and no verdict is
touched; a run of this file on the complete 80-run grid produces byte-identical
output to the frozen version. The amendment converts a crash into a message.

HOW A MISS IS READ, fixed in advance. P15.1 failing: the measured value of
memory does not depend on lambda here, and the practical claim does not
generalise to this benchmark; reported. P15.1 passing with P15.2 failing: the
gap moves, but through the memory arm, so it is the ordinary variance effect
and NOT this paper's mechanism -- reported as such, and the paper must not
cite it as mechanism evidence. P15.3 failing with P15.1 and P15.2 passing:
the effect is real and the predicted location is wrong.
"""
import json, math, random, sys

FROZEN_ON = "2026-08-21"
ENV = "MineSweeperMedium"
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
SEEDS = (1, 2, 3, 4, 5)
TAU = 30
GAMMA = 0.99
LAMBDA_KNEE = 0.97643
DEFAULT_LAM, REF_LAM = 0.95, 1.0
ABOVE_KNEE = [0.98, 0.99, 0.995, 0.999, 1.0]
G12A_MIN, G12B_MIN, G12C_MIN = -0.30, -0.10, 0.05
BOOT_N, BOOT_SEED, CI = 10_000, 20260821, 95.0
# P15.3 asserts EQUIVALENCE (above-knee cells match lambda=1.0), and a
# "CI contains zero" criterion is the wrong instrument for that: it gets
# HARDER to satisfy as the data improve, because tighter intervals exclude
# zero on ever-smaller real fluctuations. Verified on synthetic data before
# freezing -- cells drawn from an IDENTICAL distribution were flagged as
# differing, first two of four at 95%, then one of four even after Bonferroni.
# Replaced with TOST: the interval must lie ENTIRELY inside +/- P153_DELTA.
# That is monotone in data quality, which an equivalence test must be.
# P153_DELTA = 0.05 is one third of POPGym's published memory advantage on
# this environment (0.152), pre-specified from their effect size, not ours.
P153_CI = 98.75
P153_DELTA = 0.05
ANCHORS = {"uniform": -0.4281, "no_repeat": -0.4230, "safe_expansion": -0.1523}
POPGYM = {"mlp": -0.158, "gru": -0.006}


def _assert_frozen():
    if not (FROZEN_ON == "2026-08-21" and ENV == "MineSweeperMedium"
            and abs(LAMBDA_KNEE - (1 - 1 / TAU) / GAMMA) < 1e-4
            and DEFAULT_LAM < LAMBDA_KNEE < REF_LAM):
        raise SystemExit("PREREG NOT FROZEN OR TAMPERED. Do not score.")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def cell(rows, arch, lam):
    return [float(r["deployed_return"]) for r in rows
            if r.get("env") == ENV and r.get("arch") == arch
            and r.get("seed") in SEEDS
            and math.isclose(float(r.get("lam", -1)), lam)
            and r.get("deployed_return") is not None]


def iqm(xs):
    xs = sorted(xs); n = len(xs); lo, hi = n // 4, n - n // 4
    m = xs[lo:hi] if hi > lo else xs
    return sum(m) / len(m)


def _ci(draws, level=CI):
    draws.sort()
    return (draws[int(len(draws) * (100 - level) / 200)],
            draws[int(len(draws) * (1 - (100 - level) / 200)) - 1])


def _boot(rng, spec, level=CI):
    """spec: list of (cells, sign). Returns CI of the signed sum of IQMs."""
    out = []
    for _ in range(BOOT_N):
        tot = 0.0
        for c, sgn in spec:
            tot += sgn * iqm([c[rng.randrange(len(c))] for _ in c])
        out.append(tot)
    return _ci(out, level)


def check_clean(path):
    _assert_frozen()
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"CLEAN: {path} does not exist. Freeze is legitimate.")
        return 0
    n = sum(1 for r in rows if r.get("env") == ENV)
    print(f"{'NOT CLEAN' if n else 'CLEAN'}: {n} {ENV} rows in {path}.")
    return 2 if n else 0


def score(path):
    _assert_frozen()
    rows = load(path)
    m1, g1 = cell(rows, "mlp", REF_LAM), cell(rows, "gru128", REF_LAM)
    if not m1 or not g1:
        print("MISSING lambda=1.0 cells; cannot score gates."); return 2
    im1, ig1 = iqm(m1), iqm(g1)
    ga, gb, gc = im1 >= G12A_MIN, ig1 >= G12B_MIN, (ig1 - im1) >= G12C_MIN
    print(f"G12a mlp(1.0) IQM {im1:+.4f} (>= {G12A_MIN}) -> {'PASS' if ga else 'FAIL'}"
          f"    [POPGym MLP {POPGYM['mlp']}, floor {ANCHORS['uniform']}]")
    print(f"G12b gru(1.0) IQM {ig1:+.4f} (>= {G12B_MIN}) -> {'PASS' if gb else 'FAIL'}"
          f"    [POPGym GRU {POPGYM['gru']}, memoryless plateau {ANCHORS['safe_expansion']}]")
    print(f"G12c gap(1.0)     {ig1-im1:+.4f} (>= {G12C_MIN}) -> {'PASS' if gc else 'FAIL'}"
          f"    [POPGym gap {POPGYM['gru']-POPGYM['mlp']:+.3f}]")
    v = {"G12a": ga, "G12b": gb, "G12c": gc}
    if not (ga and gb and gc):
        print("\nP15.1, P15.2, P15.3 NOT SCORED (a capability gate failed).")
        print("VERDICTS:", {k: ('PASS' if b else 'FAIL') for k, b in v.items()})
        return 0

    m95, g95 = cell(rows, "mlp", DEFAULT_LAM), cell(rows, "gru128", DEFAULT_LAM)
    # Amendment 1 (2026-08-21, post-data, disclosed above). The gates are read
    # at the PROBE stage, where no lambda=0.95 cell exists yet; without this
    # the all-gates-pass path fell into _boot on an empty cell and crashed.
    # Verdict-neutral: on the full grid this branch is never taken.
    if not m95 or not g95:
        print(f"\nAll gates PASS. P15 NOT SCORABLE YET: no lambda="
              f"{DEFAULT_LAM} cells in this file, i.e. the 80-run grid has not "
              f"been collected.")
        print("Next: bash myriad/deploy.sh <ucl_username> --e12")
        print("VERDICTS:", {k: ('PASS' if b else 'FAIL') for k, b in v.items()})
        return 0
    rng = random.Random(BOOT_SEED)
    lo, hi = _boot(rng, [(g95, 1), (m95, -1), (g1, -1), (m1, 1)])
    point = (iqm(g95) - iqm(m95)) - (ig1 - im1)
    ok1 = point > 0 and lo > 0
    print(f"\nP15.1 gap(0.95) {iqm(g95)-iqm(m95):+.4f} vs gap(1.0) {ig1-im1:+.4f}"
          f" -> inflation {point:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] -> "
          f"{'PASS' if ok1 else 'FAIL'}")
    v["P15.1"] = ok1

    rng = random.Random(BOOT_SEED + 1)
    lo_m, hi_m = _boot(rng, [(m95, 1), (m1, -1)])
    pm = iqm(m95) - im1
    a_ok = pm < 0 and hi_m < 0
    rng = random.Random(BOOT_SEED + 2)
    lo_g, hi_g = _boot(rng, [(g95, 1), (g1, -1)])
    pg = iqm(g95) - ig1
    b_ok = lo_g <= 0 <= hi_g
    print(f"P15.2a mlp(0.95)-mlp(1.0) {pm:+.4f}, CI [{lo_m:+.4f}, {hi_m:+.4f}]"
          f" (negative, excludes 0) -> {'PASS' if a_ok else 'FAIL'}")
    print(f"P15.2b gru(0.95)-gru(1.0) {pg:+.4f}, CI [{lo_g:+.4f}, {hi_g:+.4f}]"
          f" (contains 0) -> {'PASS' if b_ok else 'FAIL'}")
    v["P15.2"] = a_ok and b_ok
    print(f"P15.2: {'PASS' if v['P15.2'] else 'FAIL'}")

    detail, ok3 = [], a_ok
    for lam in ABOVE_KNEE:
        if lam == REF_LAM:
            continue
        c = cell(rows, "mlp", lam)
        if not c:
            ok3 = False; detail.append(f"lam {lam}: MISSING"); continue
        rng = random.Random(BOOT_SEED + 10 + int(lam * 1000))
        l2, h2 = _boot(rng, [(c, 1), (m1, -1)], level=P153_CI)
        same = (l2 > -P153_DELTA) and (h2 < P153_DELTA)   # TOST
        ok3 = ok3 and same
        detail.append(f"lam {lam:<6} vs 1.0: {iqm(c)-im1:+.4f} CI [{l2:+.4f},"
                      f" {h2:+.4f}] within +/-{P153_DELTA} "
                      f"{'EQUIVALENT' if same else 'NOT equivalent'}")
    for d in detail:
        print(f"  P15.3 {d}")
    v["P15.3"] = ok3
    print(f"P15.3: {'PASS' if ok3 else 'FAIL'}  (knee at {LAMBDA_KNEE})")
    print("\nVERDICTS:", {k: ('PASS' if b else 'FAIL') for k, b in v.items()})
    return 0


def main(argv):
    _assert_frozen()
    if len(argv) > 2 and argv[1] == "--check-clean":
        return check_clean(argv[2])
    if len(argv) > 1 and argv[1] != "--check-clean":
        return score(argv[1])
    print(f"E12 prereg frozen {FROZEN_ON}. {ENV}, tau={TAU}, knee="
          f"{LAMBDA_KNEE} (0.95 below, 1.0 above). Gates G12a/b/c, then "
          f"P15.1 practical, P15.2 mechanism discriminator, P15.3 location.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
