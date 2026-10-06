"""Preregistration E19: the critic swap. Does PPO's actor go to the aliased fixed
point when its critic is the one the theory assumes?

FROZEN 2026-10-05, BEFORE ANY E19 RUN EXISTS. Verified mechanically:
    python3 prereg_e19.py --check-clean results/results/results_e19.jsonl

WHY THIS EXPERIMENT
-------------------
e17-pilot-findings.md: E10's one-parameter linear arm, placed anywhere on
GLEAM_g1_h0.02 and trained by PPO with its own MLP critic, drifts DOWN -- toward
under-control -- at every lambda and from every start, including starts where
the exact aliased gradient ghat of identity (I) points up and the start AT the
theory's equilibrium k_AC(lambda) itself (continuing training: 57.8 -> 48.2,
two seeds). Fitting the trained critic along the deployed trajectory shows why:
it carries none of the quadratic curvature theta2 (fitted +0.07..+0.13 against
-199..-405) through which, in identity (I), the whole displacement acts. With
the exact LSTD(lambda) value -theta2(k, lambda) v^2 substituted for the critic's
values in the GAE (--oracle-critic lstd; actor, sampling, surrogate, optimiser
untouched), the same arm at lambda = 0.9 moved TOWARD k_AC = 57.8 from 10
(-> 24.0), from 30 (-> 41.6) and from 80 (-> 66.9), seed 101. E19 registers
that contrast with seeds, two instances and two lambdas.

DESIGN (frozen)
---------------
Instances GLEAM_g1_h0.02 and GLEAM_g1_h0.05; coupled arm; --arch linear;
--fix-sigma 0.3; gamma = beta per instance; continuing training
(--gleam-episode-len 10000000); 15M steps; Adam, lr 3e-4, gain scale 1 (E10's
arm). Factors: lambda in {0.9, 0.95}; start gain k0 in {0.5 k_AC, 1.5 k_AC}
(k_AC from the frozen prereg_phase2 table); critic in {oracle (--oracle-critic
lstd), default}; seeds 1..3. 2 x 2 x 2 x 2 x 3 = 48 runs, results_e19.jsonl.
Every row records linear_gain_init, gleam_episode_len, oracle_critic (oracle rows
only) and, with --log-critic-fit 1, the quadratic fit of the network critic
along the deployed trajectory (critic_fit; diagnostic, not scored). drift :=
final_gain - k0. Cell statistic: median over the 3 seeds. D_MIN = 1.0 gain unit.
Runs locally (analysis/run_e19.py) after E18, on the harness carrying the five
inert-by-default patches (analysis/apply_*_patch.py), whose default path is
bitwise identical to the shipped harness.

CAPABILITY GATE, per (instance, lambda), scored first.
  G19  The arm moves: at least one of the four (critic, start) cells has
       |median drift| >= D_MIN. A (instance, lambda) whose arm does not move a
       gain unit in 15M steps under either critic is NOT SCORED (the pilots'
       mobility was ~10 units at lambda = 0.9; it falls with the horizon).

THE REGISTERED PREDICTIONS (over the gated-in (instance, lambda) cells; all of
them must hold for a PASS)
  P23.1  ATTRACTION. The oracle arm moves toward k_AC from both sides:
         median drift from the below-start >= +D_MIN and from the above-start
         <= -D_MIN.
  P23.2  THE DEFAULT CRITIC DOES NOT. From the below-start the default arm's
         median drift is <= -D_MIN (away from k_AC, toward under-control). The
         above-start is not scored for the default arm: both critics move down
         from there and the contrast is in the magnitude, not the sign.
  P23.3  APPROACH WITHOUT OVERSHOOT. The oracle arm's end gains lie strictly
         between their starts and the fixed point: k0_below < median end gain
         from below < k_AC(lambda) < median end gain from above < k0_above. (A
         pair of endpoints that merely stayed on their own sides would satisfy a
         plain bracket; this asks for movement toward k_AC that stops short of
         crossing it.)

HOW A MISS IS READ. P23.1 failing: the exact critic is not sufficient for the
actor to follow identity (I) -- the surrogate, the sampling or the finite rollout
also matter, and the paper cannot claim the fixed point is realised by any PPO
learner. P23.2 failing (default arm moves up): the pilots' downward drift was
seed- or cell-specific and e17-pilot-findings.md overstates it. P23.3 failing
with P23.1 passing: the oracle arm overshoots k_AC in 15M steps; the location is
then not bounded by the endpoints and the attraction is reported qualitatively.

AMENDMENTS. None at freeze.
"""
import json
import math
import os
import sys

FROZEN_ON = "2026-10-05"
INSTANCES = ["GLEAM_g1_h0.02", "GLEAM_g1_h0.05"]
ARCH = "linear"
FIX_SIGMA = 0.3
EPISODE_LEN = 10_000_000
LAMBDAS = [0.9, 0.95]
SEEDS = (1, 2, 3)
TOTAL_STEPS = 15_000_000
BETA = {"GLEAM_g1_h0.05": 0.9975031223974601, "GLEAM_g1_h0.02": 0.999000499833375}
# prereg_phase2.K_AC_LAMBDA at lambda 0.9 and 0.95 (frozen 2026-08-12; re-checked at scoring)
K_AC = {"GLEAM_g1_h0.02": {0.9: 57.809930, 0.95: 54.142926},
        "GLEAM_g1_h0.05": {0.9: 20.546902, 0.95: 17.670634}}
START_FACTORS = {"below": 0.5, "above": 1.5}
CRITICS = ("oracle", "default")
D_MIN = 1.0

ROOT = os.path.dirname(os.path.abspath(__file__))


def k0_of(inst, lam, side):
    return round(K_AC[inst][lam] * START_FACTORS[side], 6)


def _assert_frozen():
    for v in (FROZEN_ON, D_MIN, EPISODE_LEN):
        if v is None:
            raise SystemExit("PREREG NOT FROZEN. Do not launch, do not score.")
    if FROZEN_ON != "2026-10-05" or ARCH != "linear" or FIX_SIGMA != 0.3 or EPISODE_LEN != 10_000_000:
        raise SystemExit("PREREG TAMPERED. Do not score.")
    if EPISODE_LEN <= TOTAL_STEPS // 16:
        raise SystemExit("PREREG INCONSISTENT: episode length must exceed the per-environment step budget.")
    try:
        sys.path.insert(0, ROOT)
        import prereg_phase2 as P2
        for inst in INSTANCES:
            for lam in LAMBDAS:
                if abs(P2.K_AC_LAMBDA[inst][P2.LAMBDAS.index(lam)] - K_AC[inst][lam]) > 1e-6:
                    raise SystemExit("PREREG INCONSISTENT with prereg_phase2.K_AC_LAMBDA.")
            if abs(P2.BETA[inst] - BETA[inst]) > 1e-12:
                raise SystemExit("PREREG INCONSISTENT with prereg_phase2.BETA.")
    except ImportError:
        pass


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def critic_of(r):
    return "oracle" if r.get("oracle_critic") == "lstd" else "default"


def rows_of(rows):
    return [r for r in rows if r.get("env") in INSTANCES and r.get("arch") == ARCH and r.get("seed") in SEEDS
            and r.get("fix_sigma") == FIX_SIGMA and r.get("gleam_episode_len") == EPISODE_LEN
            and r.get("linear_gain_init") is not None and not r.get("linear_sgd_lr")]


def cell(rows, inst, lam, side, critic):
    k0 = k0_of(inst, lam, side)
    out = []
    for r in rows_of(rows):
        if r["env"] == inst and math.isclose(float(r["lam"]), lam) and math.isclose(float(r["linear_gain_init"]), k0, abs_tol=1e-6) \
           and critic_of(r) == critic:
            fg = r.get("final_gain")
            if fg is not None and math.isfinite(float(fg)):
                out.append(float(fg))
    return out


def median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def duplicate_cells(rows):
    counts = {}
    for r in rows_of(rows):
        k = (r["env"], round(float(r["lam"]), 6), round(float(r["linear_gain_init"]), 6), critic_of(r), int(r["seed"]))
        counts[k] = counts.get(k, 0) + 1
    return {k: n for k, n in counts.items() if n > 1}


def check_clean(path):
    _assert_frozen()
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"CLEAN: {path} does not exist. Freeze is legitimate.")
        return 0
    n = len(rows_of(rows))
    print(f"{'NOT CLEAN' if n else 'CLEAN'}: {n} E19 rows in {path}.")
    return 2 if n else 0


def _v(b):
    return "NOT SCORED" if b is None else ("PASS" if b else "FAIL")


def score(path):
    _assert_frozen()
    rows = load(path)
    dups = duplicate_cells(rows)
    if dups:
        print(f"REFUSING TO SCORE: {len(dups)} duplicated cells, e.g. {sorted(dups)[0]}.")
        return 2
    v = {}
    print("=" * 78)
    print(" E19 critic swap: oracle LSTD(lambda) critic vs PPO's default critic, linear arm, continuing training")
    print("=" * 78)
    gated = []
    for inst in INSTANCES:
        for lam in LAMBDAS:
            cells = {(side, critic): cell(rows, inst, lam, side, critic) for side in START_FACTORS for critic in CRITICS}
            incomplete = [(s, c, len(x)) for (s, c), x in cells.items() if len(x) != len(SEEDS)]
            if incomplete:
                print(f"{inst} lambda {lam}: NOT SCORABLE YET; seeds per (start, critic): " + ", ".join(f"{s}/{c}: {n}" for s, c, n in incomplete))
                continue
            drifts = {key: median([x - k0_of(inst, lam, key[0]) for x in xs]) for key, xs in cells.items()}
            moved = max(abs(d) for d in drifts.values())
            g = moved >= D_MIN
            v[f"G19-{inst}-{lam}"] = g
            print(f"{inst} lambda {lam} (k_AC {K_AC[inst][lam]:.2f}; starts {k0_of(inst, lam, 'below'):.2f} / {k0_of(inst, lam, 'above'):.2f}): "
                  f"G19 max |median drift| {moved:.2f} (>= {D_MIN}) -> {_v(g)}")
            for (side, critic), d in sorted(drifts.items()):
                ends = sorted(round(x, 2) for x in cells[(side, critic)])
                print(f"    {critic:8} from {side:5}: median drift {d:+7.2f}; end gains {ends}")
            if g:
                gated.append((inst, lam, drifts, cells))
    if not v:
        print("VERDICTS:", {})
        return 0
    if not gated:
        print("\nP23.1-P23.3 NOT SCORED: no (instance, lambda) cleared its gate.")
        print("VERDICTS:", {k: _v(b) for k, b in v.items()})
        return 0
    ok1 = ok2 = ok3 = True
    print()
    for inst, lam, drifts, cells in gated:
        a1 = drifts[("below", "oracle")] >= D_MIN and drifts[("above", "oracle")] <= -D_MIN
        a2 = drifts[("below", "default")] <= -D_MIN
        kac = K_AC[inst][lam]
        lo_end = median(cells[("below", "oracle")]); hi_end = median(cells[("above", "oracle")])
        a3 = k0_of(inst, lam, "below") < lo_end < kac < hi_end < k0_of(inst, lam, "above")
        ok1, ok2, ok3 = ok1 and a1, ok2 and a2, ok3 and a3
        print(f"  {inst} lambda {lam}: P23.1 attraction {'ok' if a1 else 'NOT ok'} "
              f"(oracle below {drifts[('below', 'oracle')]:+.2f}, above {drifts[('above', 'oracle')]:+.2f}); "
              f"P23.2 default below {drifts[('below', 'default')]:+.2f} {'ok' if a2 else 'NOT ok'}; "
              f"P23.3 approach {k0_of(inst, lam, 'below'):.2f} < {lo_end:.2f} < {kac:.2f} < {hi_end:.2f} < {k0_of(inst, lam, 'above'):.2f} {'ok' if a3 else 'NOT ok'}")
    v["P23.1"], v["P23.2"], v["P23.3"] = ok1, ok2, ok3
    # diagnostic: the network critic's fitted curvature where recorded
    fits = [(r["env"], r["lam"], critic_of(r), r["critic_fit"][2]) for r in rows_of(rows) if r.get("critic_fit")]
    if fits:
        for inst in INSTANCES:
            for lam in LAMBDAS:
                cs = [c for e, l, cr, c in fits if e == inst and math.isclose(l, lam) and cr == "default"]
                if cs:
                    print(f"  diagnostic {inst} lambda {lam}: default critic fitted curvature c, median {median(cs):+.3f} "
                          f"(range {min(cs):+.3f}..{max(cs):+.3f}); the LSTD critic's is -theta2, O(1e2)")
    print(f"P23.1: {_v(ok1)}   P23.2: {_v(ok2)}   P23.3: {_v(ok3)}")
    print("\nVERDICTS:", {k: _v(b) for k, b in v.items()})
    return 0


def main(argv):
    _assert_frozen()
    if len(argv) > 2 and argv[1] == "--check-clean":
        return check_clean(argv[2])
    if len(argv) > 1:
        return score(argv[1])
    print(f"E19 prereg frozen {FROZEN_ON}. Critic swap on {INSTANCES}, lambda {LAMBDAS}, starts 0.5/1.5 k_AC, "
          f"oracle vs default critic, {len(SEEDS)} seeds, continuing training. Gate G19 (arm moves >= {D_MIN}); "
          f"P23.1 attraction, P23.2 default drifts away from below, P23.3 oracle endpoints approach k_AC without overshoot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
