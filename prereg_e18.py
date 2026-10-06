"""Preregistration E18: E7 under the CONTINUING objective.

FROZEN 2026-10-05, BEFORE ANY E18 RUN EXISTS. Verified mechanically:
    python3 prereg_e18.py --check-clean results/results/results_e18.jsonl

WHY THIS EXPERIMENT
-------------------
The theory (identity I, the cost law, k_AC(lambda)) and the deployed evaluation
(evaluate_continuous: one unbroken trajectory, no resets) are statements about
the CONTINUING control problem. The training loop was not: GleamNative
truncates every 1024 steps and the vector wrapper then resets the state from
the UNCONTROLLED stationary distribution (Var v = 1). Each reset is followed by
a settling transient whose cost grows with the gain, so the objective PPO
actually optimised in E6, E7, E10 and E13 carries a gain-gradient the theory
does not have. Computed exactly from the shipped matrices (e17-pilot-findings.md):
on GLEAM_g1_h0.02 that gradient is +6e-4 to +1e-2 per step for k = 10..80,
against an aliasing gradient ghat of at most 5e-4; the equilibria of the
episodic objective sit at 1/2 to 1/20 of k_AC and its knee 0.03-0.05 lower in
lambda. Fifteen pre-freeze pilots (seed 101) of E10's one-parameter arm placed
away from k_AC drifted DOWN in every cell, including cells where the exact
continuing gradient points up; the same arm with resets switched off is the
pre-freeze check recorded below.

E7's registered predictions failed (P7.1, P7.2, P7.3 FAIL, prereg_phase2.py).
Part of that failure may be the objective mismatch rather than the learner. E18
asks exactly E7's questions of exactly E7's learner, with ONE change: training
is continuing (--gleam-episode-len 10000000, far above the 937,500 steps any one
of the 16 environments sees in 15M steps, so no reset ever happens). The
predictions below are E7's, with the grid-matched knee targets E10 froze.

DESIGN (frozen)
---------------
Four instances, coupled arm, --arch mlp, --fix-sigma 0.3, gamma = beta per
instance, 15M steps, 8 lambdas x 5 seeds x 4 instances = 160 runs (reduced to
the single instance GLEAM_g1_h0.05, 40 runs, by Amendment 2 below), results file
results_e18.jsonl, --gleam-episode-len 10000000 (recorded in every row as
`gleam_episode_len`). Deployed cost J_rate: policy mean, one 200,000-step
trajectory after a 1,000-step burn-in, exactly as E6/E7/E10. Estimators are
verbatim E7's (prereg_phase2.measured_lambda_half, iqm, the plateau mean).
Runs locally (analysis/run_e18.py), one machine, after E16 has exited and
analysis/apply_episode_len_patch.py has been applied (one harness version per
experiment; the patch is inert for every other experiment).

CAPABILITY GATE, per instance, scored first.
  G18  IQM deployed E at lambda = 1.0 <= 1.05. The memoryless mlp reached
       0.996-1.00 at lambda = 1 in E7 under the episodic objective; an instance
       whose mlp does not reach the memoryless floor under continuing training
       has not learned, and its predictions are NOT SCORED.

THE REGISTERED PREDICTIONS (IQM over seeds within cells; an instance is scored
only when all 40 of its cells are present once)
  P22.1  THE KNEE. Per gated-in instance, the grid-matched lambda_half measured
         from the mlp's deployed excess (J_rate / J_ml) with E7's estimator is
         within 0.01 of the grid-matched analytic target frozen in prereg_e10:
           GLEAM_g1_h0.02  0.996742     GLEAM_g1_h0.05  0.983837
           GLEAM_g2_h0.05  0.961432     GLEAM_g4_h0.05  0.855871
         (E7 under the episodic objective measured 0.92795, 0.86835, 0.92606,
         0.86332.)
  P22.2  THE ORDER. The measured knees decrease with gamma*h across the gated-in
         instances (E7's P7.2).
  P22.3  THE PLATEAU. Per gated-in instance, the mean over below-knee cells
         (lambda < the continuum lambda_half) of IQM J_rate / J_AC(lambda) is
         within 10% of 1, J_AC(lambda) being the frozen analytic cost of the
         fixed point (prereg_e10 table). E7's P7.3 compared the plateau with
         the ceiling law r*Theta*K0 = 0.30, which the fixed point itself only
         approaches (J_AC(0.8) is 0.298, 0.294, 0.289, 0.259 on the four
         instances), so the instance-specific target is used here and the 0.30
         comparison is reported as context. (E7 measured plateau J 0.4276,
         0.8423, 0.2893, 0.1303.)

HOW A MISS IS READ. P22.1 passing where P7.1 failed: the quantitative law holds
in-domain for the mlp and E7's failure was the objective, not the learner; the
paper's in-domain evidence is then this experiment, and E7 is reported as the
episodic control. P22.1 failing in the same direction as E7 (knee far below the
target, below-knee gains far below k_AC): the from-scratch mlp does not reach
the aliased equilibrium under either objective, and the thrashing account of
mechanism-diagnosis-oct.md stands. P22.3 failing with P22.1 passing: the knee
location is right and the plateau level is not, as in E7 -- the cost spent
off-manifold, not the equilibrium, sets the level. A gate failing: continuing
training changes learnability itself and the comparison with E7 is void for
that instance.

AMENDMENTS.
  1. 2026-10-05, PRE-DATA (no E18 row exists). Every run also passes
     --log-critic-fit 1 (analysis/apply_critic_fit_patch.py), which records in
     the row the least-squares quadratic fit of the critic along the deployed
     trajectory (`critic_fit`, `v_var`). Diagnostic only: no gate or prediction
     reads it. Motivation: the E17 pilots suggest PPO's critic carries far less
     curvature than the LSTD(lambda) critic of identity (I), which would remove
     the aliasing displacement at its source; E18's rows will show whether the
     mlp's critic does the same.
  2. 2026-10-05, PRE-DATA (no E18 row exists). Scope reduced to ONE instance,
     GLEAM_g1_h0.05 (the paper's instance): 8 lambdas x 5 seeds = 40 runs.
     Reason: the pre-freeze pilot of the mlp under continuing training on this
     instance (seed 101, results/pilots_e17/pilots_e18_mlp.jsonl) reproduced E7
     at the decisive cell -- lambda 0.9: gain 4.61, E = 2.16 (E7 episodic: gain
     4.1-5.9, E 2.6; k_AC 20.5, E_AC 1.33); lambda 1.0: gain 1.646 = k_ml,
     E = 1.02 -- so the objective mismatch does not explain E7's failure and a
     four-instance grid is not warranted. The single-instance run registers the
     continuing-objective control with five seeds. P22.2 (order across
     instances) is therefore NOT SCORED; P22.1 and P22.3 are unchanged.
"""
import json
import math
import os
import sys

FROZEN_ON = "2026-10-05"
INSTANCES = ["GLEAM_g1_h0.05"]          # Amendment 2 (pre-data): one instance
ALL_INSTANCES = ["GLEAM_g1_h0.02", "GLEAM_g1_h0.05", "GLEAM_g2_h0.05", "GLEAM_g4_h0.05"]
ARCH = "mlp"
FIX_SIGMA = 0.3
EPISODE_LEN = 10_000_000
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
SEEDS = (1, 2, 3, 4, 5)
TOTAL_STEPS = 15_000_000
BETA = {"GLEAM_g1_h0.05": 0.9975031223974601, "GLEAM_g1_h0.02": 0.999000499833375,
        "GLEAM_g2_h0.05": 0.9975031223974601, "GLEAM_g4_h0.05": 0.9975031223974601}
GAMMA_H = {"GLEAM_g1_h0.02": 0.02, "GLEAM_g1_h0.05": 0.05, "GLEAM_g2_h0.05": 0.10, "GLEAM_g4_h0.05": 0.20}
J_ML = {"GLEAM_g1_h0.05": 0.219595, "GLEAM_g1_h0.02": 0.217471, "GLEAM_g2_h0.05": 0.172023, "GLEAM_g4_h0.05": 0.112991}
LAMBDA_HALF_CONTINUUM = {"GLEAM_g1_h0.05": 0.985382, "GLEAM_g1_h0.02": 0.996552,
                         "GLEAM_g2_h0.05": 0.957977, "GLEAM_g4_h0.05": 0.873138}
LAMBDA_HALF_GRID = {"GLEAM_g1_h0.02": 0.996742, "GLEAM_g1_h0.05": 0.983837,
                    "GLEAM_g2_h0.05": 0.961432, "GLEAM_g4_h0.05": 0.855871}
E7_MEASURED = {"GLEAM_g1_h0.02": 0.92795, "GLEAM_g1_h0.05": 0.86835, "GLEAM_g2_h0.05": 0.92606, "GLEAM_g4_h0.05": 0.86332}
G18_MAX_E1 = 1.05
P221_TOL = 0.01
P223_LAW, P223_TOL = 0.300000, 0.10
LOG_CRITIC_FIT = 1          # Amendment 1: diagnostic row fields, not scored
# prereg_e10 C table, J_AC(lambda) on the grid LAMBDAS (frozen 2026-08-14; re-checked against prereg_e10 at scoring)
J_AC = {"GLEAM_g1_h0.02": [0.298108, 0.297731, 0.297106, 0.295609, 0.293492, 0.288892, 0.217472, 0.217731],
        "GLEAM_g1_h0.05": [0.294022, 0.292073, 0.288756, 0.278065, 0.222898, 0.219773, 0.219782, 0.219987],
        "GLEAM_g2_h0.05": [0.289001, 0.282376, 0.265346, 0.173934, 0.172195, 0.172023, 0.17215, 0.172215],
        "GLEAM_g4_h0.05": [0.259228, 0.12836, 0.115142, 0.113132, 0.112993, 0.113008, 0.113058, 0.113076]}

ROOT = os.path.dirname(os.path.abspath(__file__))


def _assert_frozen():
    for v in (FROZEN_ON, G18_MAX_E1, P221_TOL, P223_TOL, EPISODE_LEN):
        if v is None:
            raise SystemExit("PREREG NOT FROZEN. Do not launch, do not score.")
    if FROZEN_ON != "2026-10-05" or ARCH != "mlp" or FIX_SIGMA != 0.3 or EPISODE_LEN != 10_000_000:
        raise SystemExit("PREREG TAMPERED. Do not score.")
    if EPISODE_LEN <= TOTAL_STEPS // 16:
        raise SystemExit("PREREG INCONSISTENT: episode length must exceed the per-environment step budget.")
    try:
        sys.path.insert(0, ROOT)
        import prereg_phase2 as P2
        for n in INSTANCES:
            if abs(P2.LAMBDA_HALF[n] - LAMBDA_HALF_CONTINUUM[n]) > 1e-9 or abs(P2.BASELINES[n]["J_ml"] - J_ML[n]) > 1e-9 \
               or abs(P2.BETA[n] - BETA[n]) > 1e-12 or P2.LAMBDAS != LAMBDAS:
                raise SystemExit("PREREG INCONSISTENT with prereg_phase2 constants.")
        import prereg_e10 as P10
        for n in INSTANCES:
            if abs(P10.C[n]["grid_lambda_half"] - LAMBDA_HALF_GRID[n]) > 1e-9:
                raise SystemExit("PREREG INCONSISTENT with prereg_e10 grid-matched knees.")
            if any(abs(a - b) > 1e-6 for a, b in zip(P10.C[n]["J_AC"], J_AC[n])):
                raise SystemExit("PREREG INCONSISTENT with prereg_e10 J_AC table.")
    except ImportError:
        pass


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def rows_of(rows, inst):
    return [r for r in rows if r.get("env") == inst and r.get("arch") == ARCH and r.get("seed") in SEEDS
            and r.get("fix_sigma") == FIX_SIGMA and r.get("gleam_episode_len") == EPISODE_LEN]


def cell(rows, inst, lam, field):
    out = []
    for r in rows_of(rows, inst):
        if math.isclose(float(r.get("lam", -1)), lam):
            v = r.get(field)
            if v is not None and math.isfinite(float(v)):
                out.append(float(v))
    return out


def iqm(xs):
    """Verbatim prereg_phase2.iqm."""
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    lo, hi = n // 4, n - n // 4
    m = xs[lo:hi] if hi > lo else xs
    return sum(m) / len(m)


def measured_lambda_half(rows, inst):
    """Verbatim port of prereg_phase2.measured_lambda_half (excess_cost, IQM per cell,
    midpoint between the largest and smallest cell values, linear interpolation)."""
    pts = []
    for lam in LAMBDAS:
        c = cell(rows, inst, lam, "excess_cost")
        if c:
            pts.append((lam, iqm(c)))
    if len(pts) < 3:
        return None
    hi = max(x for _, x in pts); lo = min(x for _, x in pts)
    if hi - lo < 1e-9:
        return None
    mid = 0.5 * (hi + lo)
    for (l0, v0), (l1, v1) in zip(pts, pts[1:]):
        if (v0 - mid) * (v1 - mid) <= 0 and abs(v1 - v0) > 1e-12:
            return l0 + (mid - v0) * (l1 - l0) / (v1 - v0)
    return None


def duplicate_cells(rows):
    counts = {}
    for inst in INSTANCES:
        for r in rows_of(rows, inst):
            k = (inst, round(float(r.get("lam", -1)), 6), int(r.get("seed")))
            counts[k] = counts.get(k, 0) + 1
    return {k: n for k, n in counts.items() if n > 1}


def check_clean(path):
    _assert_frozen()
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"CLEAN: {path} does not exist. Freeze is legitimate.")
        return 0
    n = sum(len(rows_of(rows, i)) for i in INSTANCES)
    print(f"{'NOT CLEAN' if n else 'CLEAN'}: {n} E18 rows in {path}.")
    return 2 if n else 0


def _v(b):
    return "NOT SCORED" if b is None else ("PASS" if b else "FAIL")


def score(path):
    _assert_frozen()
    rows = load(path)
    dups = duplicate_cells(rows)
    if dups:
        print(f"REFUSING TO SCORE: {len(dups)} duplicated (instance, lambda, seed) cells, e.g. {sorted(dups)[0]}.")
        return 2
    v = {}
    print("=" * 74)
    print(" E18: E7 under the continuing objective (no resets), mlp from scratch")
    print("=" * 74)
    complete, gated = [], []
    for inst in INSTANCES:
        missing = [(l, len(cell(rows, inst, l, "J_rate"))) for l in LAMBDAS if len(cell(rows, inst, l, "J_rate")) != len(SEEDS)]
        if missing:
            if any(n for _, n in missing) or len(missing) < len(LAMBDAS):
                print(f"{inst}: NOT SCORABLE YET, incomplete cells (lambda, seeds present): "
                      + ", ".join(f"({l}, {n})" for l, n in missing))
            else:
                print(f"{inst}: no rows yet.")
            continue
        complete.append(inst)
        e1 = iqm(cell(rows, inst, 1.0, "J_rate")) / J_ML[inst]
        g = e1 <= G18_MAX_E1
        v[f"G18-{inst}"] = g
        print(f"{inst}: G18 E(1.0) IQM {e1:.4f} (<= {G18_MAX_E1}) -> {_v(g)}")
        if g:
            gated.append(inst)
    if not complete:
        print("VERDICTS:", {})
        return 0
    if not gated:
        print("\nP22.1, P22.2, P22.3 NOT SCORED: no instance cleared its gate.")
        print("VERDICTS:", {k: _v(b) for k, b in v.items()})
        return 0
    print(f"\nGated in: {gated}" + (f"; not scored: {[i for i in complete if i not in gated]}" if len(gated) < len(complete) else ""))
    halves = {}
    ok1 = True
    for inst in gated:
        mh = measured_lambda_half(rows, inst)
        halves[inst] = mh
        if mh is None:
            ok1 = False
            print(f"  P22.1 {inst}: knee not measurable (flat profile)")
            continue
        good = abs(mh - LAMBDA_HALF_GRID[inst]) <= P221_TOL
        ok1 = ok1 and good
        print(f"  P22.1 {inst}: measured {mh:.5f} vs grid-matched {LAMBDA_HALF_GRID[inst]:.6f}, |diff| {abs(mh - LAMBDA_HALF_GRID[inst]):.5f} "
              f"({'within' if good else 'OUTSIDE'} {P221_TOL})   [continuum {LAMBDA_HALF_CONTINUUM[inst]}; E7 episodic {E7_MEASURED[inst]}]")
    v["P22.1"] = ok1
    order = sorted(gated, key=lambda n: GAMMA_H[n])
    seq = [halves.get(n) for n in order]
    if len(order) < 2 or any(x is None for x in seq):
        v["P22.2"] = None
        print("  P22.2: NOT SCORED (fewer than two measurable knees)")
    else:
        v["P22.2"] = all(a > b for a, b in zip(seq, seq[1:]))
        print("  P22.2 " + ", ".join(f"gamma*h {GAMMA_H[n]:.2f}: {halves[n]:.5f}" for n in order) + f" -> {_v(v['P22.2'])}")
    ok3 = True
    for inst in gated:
        below = [l for l in LAMBDAS if l < LAMBDA_HALF_CONTINUUM[inst]]
        vals = [iqm(cell(rows, inst, l, "J_rate")) for l in below]
        ratios = [vv / J_AC[inst][LAMBDAS.index(l)] for vv, l in zip(vals, below)]
        ratio = sum(ratios) / len(ratios)
        good = abs(ratio - 1.0) <= P223_TOL
        ok3 = ok3 and good
        plateau = sum(vals) / len(vals)
        print(f"  P22.3 {inst}: plateau J_meas/J_AC mean {ratio:.4f} over lambda {below} ({'within' if good else 'OUTSIDE'} {int(100 * P223_TOL)}%)"
              f"   [plateau J {plateau:.4f}; ceiling law {P223_LAW}: {100 * abs(plateau - P223_LAW) / P223_LAW:.1f}% off, context]")
    v["P22.3"] = ok3
    for inst in gated:
        E = {l: iqm(cell(rows, inst, l, "J_rate")) / J_ML[inst] for l in LAMBDAS}
        G = {l: iqm(cell(rows, inst, l, "final_gain")) for l in LAMBDAS}
        print(f"  {inst} E by lambda: " + ", ".join(f"{l}: {E[l]:.3f}" for l in LAMBDAS))
        print(f"  {inst} gain IQM by lambda: " + ", ".join(f"{l}: {G[l]:.2f}" for l in LAMBDAS))
    print(f"P22.1: {_v(v['P22.1'])}   P22.2: {_v(v['P22.2'])}   P22.3: {_v(v['P22.3'])}")
    print("\nVERDICTS:", {k: _v(b) for k, b in v.items()})
    return 0


def main(argv):
    _assert_frozen()
    if len(argv) > 2 and argv[1] == "--check-clean":
        return check_clean(argv[2])
    if len(argv) > 1:
        return score(argv[1])
    print(f"E18 prereg frozen {FROZEN_ON}. E7 under continuing training (episode length {EPISODE_LEN}), "
          f"{len(INSTANCES)} instance(s) {INSTANCES} x {len(LAMBDAS)} lambdas x {len(SEEDS)} seeds, mlp, sigma {FIX_SIGMA}. "
          f"Gate G18 per instance (E(1.0) <= {G18_MAX_E1}); P22.1 knee within {P221_TOL} of the grid-matched target, "
          f"P22.2 order by gamma*h, P22.3 below-knee cost within {int(100 * P223_TOL)}% of J_AC(lambda).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
