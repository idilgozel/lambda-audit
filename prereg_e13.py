"""Preregistration E13: does the cost-law threshold survive breaking LQG exactness?

DRAFT 2026-08-21, submitted for approval. FREEZE RULE: this file is frozen the
moment it is committed with results/results/results_e13.jsonl absent; verify:

    python3 prereg_e13.py --check-clean results/results/results_e13.jsonl

WHY E13 EXISTS
--------------
Every experimental confirmation of the cost-law threshold lambda_half sits on
the exactly-solvable linear-quadratic family, and "artifact of a 2-D LQG toy"
is the paper's most predictable objection. kernel-generality-findings.md
answers the dimension half by exact computation (the structure holds at state
dimensions 2-4, any Prony kernel). E13 answers the linearity half by
experiment: add a cubic damping the theory cannot absorb, and test whether the
LINEARIZED theory still locates the measured threshold.

This is a robustness test of a claim that already passed (P11.3, prospective,
four instances), not a retry of one that failed. Whatever it returns is
reported.

THE ENVIRONMENT MAP, exact
--------------------------
GLEAM_g1_h0.05 with, after the linear ZOH update of every step,

    for _ in range(4): v -= (h/4) * nl * v**3

The four substeps are part of the frozen definition. The pre-freeze pilot
found the single-step map OVERFLOWS at nl = 11.2: it is unstable once
|v| > sqrt(2/(h*nl)) ~ 1.9, which exploration noise reaches. Substepping
doubles that threshold; a 2M-sample scan at u = -k v + eta, sigma = 0.3, at
k in {1.51, 2.20, 22.43}, was finite for both nl arms. Implemented
identically in harness/gleam_native.py (env ids GLEAM_g1_h0.05_nl2.24,
GLEAM_g1_h0.05_nl11.2) and in the pilot simulator.

WHY THESE nl VALUES
-------------------
Scale set by the physics, not convenience: rho(nl) := nl * E[v^2] / k at the
above-knee equilibrium (k = k_AC(1.0) = 1.5066, E[v^2] = 0.0673, the
largest-amplitude point of the grid, hence the worst case). rho = 0.10 gives
nl = 2.2394 and rho = 0.50 gives nl = 11.1969; frozen at the piloted values
2.24 and 11.2. The perturbation is deliberately asymmetric across the knee:
at the inflated equilibrium (k ~ 22) E[v^2] = 0.00058, so the same nl
perturbs the below-knee branch ~100x less. A cubic that bent BOTH branches
equally would test integration accuracy; one that bites exactly where the
collapse lands tests the threshold.

PILOT, RECORDED BEFORE FREEZING (deployed J at gains PINNED to the linearized
k_AC(lambda); 2.3M samples per cell; analysis code in this file's history):

    nl      0.8     0.9     0.95    0.98    0.99    0.995   0.999   1.0     lam_half
    0.0     .2938   .2918   .2885   .2778   .2225   .2194   .2195   .2197   0.98383
    2.24    .2937   .2917   .2883   .2771   .1927   .1771   .1689   .1672   0.98553
    11.2    .2933   .2912   .2875   .2745   .1421   .1205   .1099   .1077   0.98559

Three facts the predictions below are built on. (i) The pinned-gain pipeline
reproduces the frozen linearized lambda_half (0.98383 vs 0.983837). (ii) The
cubic deepens the floor (the collapse gets STRONGER: profile ratio 1.34 ->
1.76 -> 2.72) but leaves the plateau alone, so the threshold's location moves
only ~+0.002 even at a 50 percent perturbation. (iii) The lambda_half
estimator is invariant under scaling and shifting of the J profile, so no
nonlinear baseline (J_ml under nl, which is not exactly computable) is needed
anywhere: scoring uses raw deployed J.

THE EXPERIMENT
--------------
Arms nl in {0.0, 2.24, 11.2}; the 8-point lambda grid; seeds 1, 2, 3;
arch linear ONLY, gain initialised at the LINEARIZED k_AC(lambda) (table
below, copied from prereg_e10.py's frozen C); 1M steps; fix_sigma 0.3;
gamma_RL = 0.9975031223974601 (= exp(-rho*h), bitwise as E6/E7/E10).
72 runs. The nl = 0.0 arm is the instrument control: same budget, same host,
same pipeline, known truth.

E10's drift lesson, pre-committed so nobody diagnoses it post hoc: trained
gains sit BELOW k_AC (96/96 runs in E10), and measured J then sits a few
percent above the pinned-gain floors of the pilot table. lambda_half is
estimated from the profile SHAPE and E10 measured it to 0.00014 despite that
drift; the control arm gate absorbs whatever the 1M budget adds.

WHERE THIS RUNS, disclosed
--------------------------
Locally (analysis/run_e13.py), not on Myriad: cluster access is running out
and is reserved for completing the frozen E12 design. Consequences, stated:
no SGE job-id chronology for E13 (the freeze commit and the results file
mtime are the record), no node-class comparability with E6-E12 rows, and no
claim of cross-host bitwise reproducibility. The control arm calibrates the
actual host and budget; host and torch version are recorded in
results/logs/e13_host.txt.

THE REGISTERED PREDICTIONS
--------------------------
Cell value = median over seeds of deployed J_rate. Profile = the 8 cell
values against lambda. ratio := J(0.8)/J(1.0). lambda_half by the verbatim
prereg_e10 estimator on the raw profile.

G13   INSTRUMENT GATE (nl = 0.0): ratio >= 1.15 AND
      |lambda_half - 0.983837| <= 0.005.
      Fails -> P16 NOT SCORED: the 1M-step local instrument cannot resolve
      the threshold it is being asked to resolve, and that is an instrument
      statement, not a theory statement. A gate is not a finding.

P16.1 MILD (nl = 2.24, rho = 0.10): ratio >= 1.15 AND
      |lambda_half - 0.983837| <= 0.005.

P16.2 STRONG (nl = 11.2, rho = 0.50): ratio >= 1.15 AND
      |lambda_half - 0.983837| <= 0.005.

THE TOLERANCE IS 0.005, TIGHTENED FROM P11.3's 0.01 BEFORE ANY DATA EXISTED,
and the reason is recorded because it was found by testing this scorer on
synthetic profiles (the E12 lesson, applied at birth): with the collapse
sitting between grid points 0.98 and 0.99, a 0.01 band also PASSES a profile
whose entire collapse is delayed by one full grid step (lambda_half shift
+0.0089) - a qualitative change the test exists to catch. At 0.005 that
synthetic case fails, while the budget still carries ~3x margin over
everything expected: the pilot's true pinned-gain shift is +0.0017/+0.0018,
E10-style drift moved lambda_half by 0.00014 at 15M steps, the estimator is
exactly invariant under profile scaling (verified: a uniform 5 percent
elevation moves it by 0.0), and 2 percent per-cell J noise propagates to
~0.001 through the 0.98-0.99 crossing. G13 shares the same band, so an
instrument too noisy to hold it fails the GATE - not scored - rather than
producing a false theory verdict. No dose-ordering prediction is registered:
the pilot separates the two shifts by 0.00006, and predicting their order
would be gambling on noise.

HOW A MISS IS READ, fixed in advance
------------------------------------
G13 fail: budget/instrument insufficient; nothing about the theory.
P16.1 fail, structure present: lambda_half is NOT robust to mild
  nonlinearity; the paper's generality claim is bounded at exactly-LQG and
  says so. This is the expensive outcome and it is reported in the abstract.
P16.1 fail, structure absent (ratio < 1.15): stronger - the collapse itself
  needs linearity here.
P16.2 fail with P16.1 passing: robustness has a dose boundary between
  rho = 0.10 and 0.50; reported as the measured boundary, which is a
  scope statement, not a defeat.
PASS licenses exactly: "the threshold location survives cubic damping up to
a 50 percent perturbation of the dominant balance at the above-knee
equilibrium". It does NOT license "beyond LQG" in general, and the paper
must not say so.
"""
import json
import math
import sys

FROZEN_ON = "2026-08-21"
INSTANCE = "GLEAM_g1_h0.05"
NL_ARMS = (0.0, 2.24, 11.2)
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
SEEDS = (1, 2, 3)
STEPS = 1_000_000
GAMMA_RL = 0.9975031223974601
FIX_SIGMA = 0.3
# linearized k_AC(lambda), frozen 2026-08-14 in prereg_e10.py's C table
GAIN_INIT = [22.426983, 20.546902, 17.670634, 11.394604,
             2.197126, 1.756354, 1.548754, 1.506599]
LIN_LAMBDA_HALF = 0.983837
TOL = 0.005
RATIO_MIN = 1.15
# pilot record (pinned gains; design support, nothing is scored against it)
PILOT_LAMBDA_HALF = {0.0: 0.98383, 2.24: 0.98553, 11.2: 0.98559}


def _assert_frozen():
    ok = (FROZEN_ON == "2026-08-21" and len(LAMBDAS) == 8
          and len(GAIN_INIT) == 8 and NL_ARMS == (0.0, 2.24, 11.2)
          and abs(GAMMA_RL - math.exp(-0.05 * 0.05)) < 1e-15
          and TOL == 0.005 and RATIO_MIN == 1.15)
    if not ok:
        raise SystemExit("PREREG NOT FROZEN OR TAMPERED. Do not score.")


def env_id(nl):
    return INSTANCE if nl == 0.0 else f"{INSTANCE}_nl{nl}"


def _parse_nl(env):
    if env == INSTANCE:
        return 0.0
    if env.startswith(INSTANCE + "_nl"):
        try:
            return float(env[len(INSTANCE) + 3:])
        except ValueError:
            return None
    return None


def lambda_half(pts):
    """Verbatim from prereg_e10.py (frozen 2026-08-14)."""
    if len(pts) < 3:
        return None
    hi = max(v for _, v in pts); lo = min(v for _, v in pts)
    if hi - lo < 1e-9:
        return None
    mid = 0.5 * (hi + lo)
    for (l0, v0), (l1, v1) in zip(pts, pts[1:]):
        if (v0 - mid) * (v1 - mid) <= 0 and abs(v1 - v0) > 1e-12:
            return l0 + (mid - v0) * (l1 - l0) / (v1 - v0)
    return None


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def check_clean(path):
    _assert_frozen()
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"CLEAN: {path} does not exist. Freeze is legitimate.")
        return 0
    n = sum(1 for r in rows if _parse_nl(r.get("env", "")) is not None)
    print(f"{'NOT CLEAN' if n else 'CLEAN'}: {n} E13 rows in {path}.")
    return 2 if n else 0


def score(path):
    _assert_frozen()
    rows = [r for r in load(path)
            if _parse_nl(r.get("env", "")) is not None
            and r.get("arch") == "linear" and r.get("seed") in SEEDS]

    # duplicate refusal (E12's lesson, adopted at birth here)
    seen, dup = {}, []
    for r in rows:
        k = (r["env"], round(float(r["lam"]), 6), int(r["seed"]))
        if k in seen:
            dup.append(k)
        seen[k] = True
    if dup:
        print(f"REFUSING TO SCORE: {len(dup)} duplicated (env, lambda, seed) "
              f"cells, e.g. {dup[0]}.")
        print("Separate the extra rows first; scoring silently absorbing "
              "duplicates is how E12's reference cell was contaminated.")
        return 2

    def cell(nl, lam):
        return [float(r["J_rate"]) for r in rows
                if _parse_nl(r["env"]) == nl
                and math.isclose(float(r["lam"]), lam)]

    # coverage, and refusal to extrapolate an incomplete arm
    print(f"coverage: {len(rows)} / {len(NL_ARMS) * len(LAMBDAS) * len(SEEDS)}"
          f" rows")
    complete = {}
    for nl in NL_ARMS:
        missing = [(lam, s) for lam in LAMBDAS for s in SEEDS
                   if len([r for r in rows if _parse_nl(r["env"]) == nl
                           and math.isclose(float(r["lam"]), lam)
                           and r["seed"] == s]) == 0]
        complete[nl] = not missing
        if missing:
            print(f"  nl={nl}: INCOMPLETE, missing {missing}")
            print("  Missing runs are runs, not results: complete the arm, "
                  "do not score it partial.")

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    v = {}
    results = {}
    for nl in NL_ARMS:
        if not complete[nl]:
            results[nl] = None
            continue
        pts = [(lam, median(cell(nl, lam))) for lam in LAMBDAS]
        ratio = pts[0][1] / pts[-1][1]
        lh = lambda_half(pts)
        results[nl] = (pts, ratio, lh)
        print(f"\n  nl={nl}  " + " ".join(f"{J:.4f}" for _, J in pts))
        print(f"        ratio J(0.8)/J(1.0) = {ratio:.3f} (need >= {RATIO_MIN})"
              f"   lambda_half = {lh if lh else 'none'}"
              f"   vs linearized {LIN_LAMBDA_HALF} "
              f"(pilot pinned-gain: {PILOT_LAMBDA_HALF[nl]})")

    def verdict(nl):
        if results[nl] is None:
            return None
        _, ratio, lh = results[nl]
        return (ratio >= RATIO_MIN and lh is not None
                and abs(lh - LIN_LAMBDA_HALF) <= TOL)

    g13 = verdict(0.0)
    if g13 is None:
        print("\nG13 NOT SCORABLE: control arm incomplete."); return 2
    v["G13"] = g13
    print(f"\nG13 (instrument, nl=0): {'PASS' if g13 else 'FAIL'}")
    if not g13:
        print("P16.1, P16.2 NOT SCORED (instrument gate failed; the 1M local "
              "budget cannot\nresolve the threshold, and that is an "
              "instrument statement, not a theory one).")
        print("\nVERDICTS:", {k: ('PASS' if b else 'FAIL')
                              for k, b in v.items()})
        return 0
    for name, nl in (("P16.1", 2.24), ("P16.2", 11.2)):
        w = verdict(nl)
        if w is None:
            print(f"{name} NOT SCORABLE: nl={nl} arm incomplete.")
            continue
        v[name] = w
        print(f"{name} (nl={nl}): {'PASS' if w else 'FAIL'}")
    print("\nVERDICTS:", {k: ('PASS' if b else 'FAIL') for k, b in v.items()})
    return 0


def main(argv):
    _assert_frozen()
    if len(argv) > 2 and argv[1] == "--check-clean":
        return check_clean(argv[2])
    if len(argv) > 1 and argv[1] != "--check-clean":
        return score(argv[1])
    print(f"E13 prereg frozen {FROZEN_ON}. {INSTANCE} + cubic damping "
          f"nl in {NL_ARMS[1:]},\n4-substep map, linear arm at linearized "
          f"k_AC(lambda), {STEPS} steps, seeds {SEEDS}.\nG13 instrument gate, "
          f"then P16.1 (rho=0.10) and P16.2 (rho=0.50): lambda_half\nwithin "
          f"{TOL} of the linearized {LIN_LAMBDA_HALF}, structure ratio >= "
          f"{RATIO_MIN}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
