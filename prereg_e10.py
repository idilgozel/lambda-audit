"""Preregistration E10: the cost law in its domain, and the mechanism, tested clean.

FROZEN 2026-08-14, BEFORE ANY E10 RUN EXISTS ANYWHERE. Verified mechanically:

    python3 prereg_e10.py --check-clean results/results/results_e10.jsonl

refuses if E10 rows already exist. Scoring is IN THIS FILE (score mode), so no
judgement call survives contact with the data.

WHY E10 EXISTS
--------------
E7's P7.1-P7.3 FAILED as registered, and the failure is diagnosed, not excused:
below the horizon, PPO's mlp policy never reaches the aliased equilibrium
k_AC(lambda) (all five seeds stall at 6 to 50 percent of it, every instance),
and where it goes intermittent the measured cost is contaminated by
off-manifold excursions. The registered estimator then anchored on
contaminated cells. Decomposed on the same data: in the 23 cells with no
intermittency, J_measured / J_analytic(k_measured) has median 0.994 and 22/23
lie within 10 percent. The cost law was never contradicted; the assumption
that training ARRIVES at k_AC below the horizon is what failed, and the Tier 2
mechanism (bias-direction instability, S(lambda,k)) predicts which cells fail:
30/32 by the path test, with both misses at predicted rates |S| <= 1e-4.

E10 therefore places a policy AT k_AC by construction and tests both claims
prospectively, in the same harness as E6/E7, with the policy class the theory
is actually about:

  arm "linear"  u = -k v          gain initialised at k_AC(lambda). Cannot
                                  leave the gain manifold. Tests the cost law.
  arm "linearb" u = -k v + b      additionally, the bias is KICKED to b0 = 1.0
                                  at t = 0. Tests the mechanism: b is predicted
                                  to decay where S > 0 and persist/grow where
                                  S < 0. Perturbation-decay reads the
                                  eigenvalue sign directly instead of waiting
                                  for stochastic escape from b = 0.

PILOT SUPPORT (g1_h0.05, 1 seed, recorded before freezing; this instance's
lambda* is a postdiction and is burned for prospective purposes anyway):
  linear,  1M steps, init k_AC(0.9):  gain 20.47 held (drift 0.4%), J ratio 0.96
  linearb, kick b0=1.0, lam 0.9  (S=-9.3e-4): b = 0.94 at 1M, 1.38 at 3M  GROWS
  linearb, kick b0=1.0, lam 0.995 (S=+2.7e-2): b = 0.65 at 1M              DECAYS
Threshold b0/2 = 0.5 at 15M steps follows: decay at the observed rate reaches
<< 0.5 within budget; persistence/growth stays >= b0.

HARNESS IDENTITY at freeze: ppo_lambda.py md5 recorded in FREEZE_MD5 below.
mlp/gru128 paths verified bitwise-identical to the E6/E7 binary (rows and
curves, timing columns excluded) after the linear/linearb additions.

RESULTS FILE: results_e10.jsonl. NEVER results_phase2.jsonl; the harness
refuses (cell() in prereg_phase2.py ignores arch, so a linear row would
contaminate E6's registered cells).

THE REGISTERED PREDICTIONS
--------------------------
Cell = (instance, lambda), median over seeds 1..3. Grid LAMBDAS below.

P11.1 VALIDITY. Every linear cell holds its equilibrium:
      |final_gain / k_AC(lambda) - 1| <= 0.15 in all 32 cells.
P11.2 COST LAW. Every linear cell pays the predicted cost:
      |J_rate / J_AC(lambda) - 1| <= 0.10 in all 32 cells.
P11.3 THE QUANTITATIVE LAW, P7.1 done in-domain. Per instance, lambda_half
      measured from the linear arm's excess (J_rate / J_ml) with the SAME
      estimator as prereg_phase2.measured_lambda_half falls within 0.01 of the
      GRID-MATCHED analytic target (the same estimator applied to the analytic
      J_AC/J_ml on the same 8-point grid; freezing the grid-matched value
      separates measurement error from interpolation bias):
        GLEAM_g1_h0.02  0.996742     GLEAM_g1_h0.05  0.983837
        GLEAM_g2_h0.05  0.961432     GLEAM_g4_h0.05  0.855871
P11.4 MECHANISM. linearb cells are scored by the frozen sign of S(lambda,k_AC)
      with a rate floor: cells with |S| < 1e-4 are EXCLUDED as too slow to
      resolve at 15M steps (three cells, named in EXCLUDED below; the floor is
      set by the pilot, which resolved |S| ~ 9e-4 within 3M).
        S < 0 (8 cells)  -> |final_bias| >= 0.5   (persisted or grew)
        S > 0 (21 cells) -> |final_bias| <  0.5   (decayed)
      PASS: >= 7 of 8 unstable cells read unstable AND >= 19 of 21 stable
      cells read stable. Every miss is reported by name either way.
      g4_h0.05 predicts stable at ALL lambdas; any instability there is a
      hard failure of the mechanism, exactly as in prereg_e7_bias.py.

HOW A MISS IS READ, fixed in advance: P11.1/P11.2 failing anywhere means the
equilibrium map or the cost function is wrong at that cell; there is no
reachability excuse left, that is the point of the design. P11.3 failing means
the law's shape is wrong, not just a point. P11.4 misses beyond the allowance
falsify the bias-direction mechanism as the driver at k_AC.
"""
import json, math, sys

FROZEN_ON = "2026-08-14"
FREEZE_MD5 = "ae3ad1f0db5d83e0878734db75f82287"
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
SEEDS = (1, 2, 3)
B0 = 1.0
B_THRESH = 0.5
GAIN_TOL = 0.15
J_TOL = 0.10
LHALF_TOL = 0.01
S_FLOOR = 1e-4
UNSTABLE_MIN = 7   # of 8
STABLE_MIN = 19    # of 21

C = json.loads(r'''{"GLEAM_g1_h0.02": {"gamma": 1.0, "h": 0.02, "J_ml": 0.217471, "k_AC": [60.071834, 57.80993, 54.142926, 46.176014, 37.359041, 25.550589, 1.671172, 1.54499], "J_AC": [0.298108, 0.297731, 0.297106, 0.295609, 0.293492, 0.288892, 0.217472, 0.217731], "S_at_kAC": [-0.0003007, -0.0002628, -0.0002089, -0.0001304, -7.71e-05, -1.8e-05, 0.0113614, 0.0126882], "grid_lambda_half": 0.996742, "continuum_lambda_half": 0.996552}, "GLEAM_g1_h0.05": {"gamma": 1.0, "h": 0.05, "J_ml": 0.219595, "k_AC": [22.426983, 20.546902, 17.670634, 11.394604, 2.197126, 1.756354, 1.548754, 1.506599], "J_AC": [0.294022, 0.292073, 0.288756, 0.278065, 0.222898, 0.219773, 0.219782, 0.219987], "S_at_kAC": [-0.0012998, -0.0009345, -0.0005254, 0.0002301, 0.0183811, 0.0265833, 0.0318702, 0.0330674], "grid_lambda_half": 0.983837, "continuum_lambda_half": 0.985382}, "GLEAM_g2_h0.05": {"gamma": 2.0, "h": 0.05, "J_ml": 0.172023, "k_AC": [20.07708, 16.82564, 11.046191, 1.646322, 1.42903, 1.343181, 1.281927, 1.26748], "J_AC": [0.289001, 0.282376, 0.265346, 0.173934, 0.172195, 0.172023, 0.17215, 0.172215], "S_at_kAC": [-0.000264, -4.4e-06, 0.000602, 0.035298, 0.0429266, 0.0463756, 0.0489777, 0.049607], "grid_lambda_half": 0.961432, "continuum_lambda_half": 0.957977}, "GLEAM_g4_h0.05": {"gamma": 4.0, "h": 0.05, "J_ml": 0.112991, "k_AC": [14.515075, 2.175412, 1.453986, 1.226369, 1.164381, 1.135285, 1.112818, 1.107307], "J_AC": [0.259228, 0.12836, 0.115142, 0.113132, 0.112993, 0.113008, 0.113058, 0.113076], "S_at_kAC": [0.0004416, 0.0270735, 0.0503611, 0.0621631, 0.0658115, 0.0675828, 0.0689743, 0.0693187], "grid_lambda_half": 0.855871, "continuum_lambda_half": 0.873138}}''')

GRID_LAMBDA_HALF = {k: v["grid_lambda_half"] for k, v in C.items()}
EXCLUDED = [("GLEAM_g1_h0.02", 0.99), ("GLEAM_g1_h0.02", 0.995),
            ("GLEAM_g2_h0.05", 0.9)]


def _assert_frozen():
    for env, d in C.items():
        for key in ("k_AC", "J_AC", "S_at_kAC"):
            if len(d[key]) != len(LAMBDAS):
                raise SystemExit(f"NOT FROZEN: {env} {key}")
    for env, lam in EXCLUDED:
        s = C[env]["S_at_kAC"][LAMBDAS.index(lam)]
        if abs(s) >= S_FLOOR:
            raise SystemExit(f"INCONSISTENT: excluded cell {env} {lam} has |S| >= floor")
    n_uns = sum(1 for env, d in C.items() for i, s in enumerate(d["S_at_kAC"])
                if s < 0 and abs(s) >= S_FLOOR)
    n_sta = sum(1 for env, d in C.items() for i, s in enumerate(d["S_at_kAC"])
                if s > 0 and abs(s) >= S_FLOOR)
    if (n_uns, n_sta) != (8, 21):
        raise SystemExit(f"INCONSISTENT: expected 8 unstable / 21 stable scored "
                         f"cells, constants give {n_uns} / {n_sta}")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def cell(rows, env, arch, lam, field):
    out = [float(r[field]) for r in rows
           if r.get("env") == env and r.get("arch") == arch
           and r.get("seed") in SEEDS
           and math.isclose(float(r.get("lam", -1)), lam)
           and r.get(field) is not None and math.isfinite(float(r[field]))]
    return out


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def lambda_half(pts):
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


def check_clean(path):
    _assert_frozen()
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"CLEAN: {path} does not exist. Freeze is legitimate.")
        return 0
    n = sum(1 for r in rows if r.get("arch") in ("linear", "linearb"))
    if n:
        print(f"NOT CLEAN: {n} linear/linearb rows already in {path}.")
        return 2
    print(f"CLEAN: no E10 rows in {path}.")
    return 0


def score(path):
    _assert_frozen()
    rows = load(path)
    v = {}
    print("=" * 74); print(" P11.1 validity / P11.2 cost law (linear arm)"); print("=" * 74)
    ok1 = ok2 = True
    for env, d in C.items():
        for i, lam in enumerate(LAMBDAS):
            g = med(cell(rows, env, "linear", lam, "final_gain"))
            J = med(cell(rows, env, "linear", lam, "J_rate"))
            if g is None or J is None:
                ok1 = ok2 = False
                print(f"  {env:16} lam {lam:<6} MISSING"); continue
            r1 = abs(g / d["k_AC"][i] - 1); r2 = abs(J / d["J_AC"][i] - 1)
            b1 = r1 <= GAIN_TOL; b2 = r2 <= J_TOL
            ok1 &= b1; ok2 &= b2
            flag = "" if (b1 and b2) else "   <<<"
            print(f"  {env:16} lam {lam:<6} gain {g:8.3f}/{d['k_AC'][i]:8.3f} "
                  f"({100*r1:4.1f}%) J {J:7.4f}/{d['J_AC'][i]:7.4f} ({100*r2:4.1f}%){flag}")
    v["P11.1"], v["P11.2"] = ok1, ok2
    print(f"P11.1: {'PASS' if ok1 else 'FAIL'}   P11.2: {'PASS' if ok2 else 'FAIL'}")

    print("=" * 74); print(" P11.3 lambda_half, in-domain (linear arm)"); print("=" * 74)
    ok3 = True
    for env, d in C.items():
        pts = []
        for lam in LAMBDAS:
            J = med(cell(rows, env, "linear", lam, "J_rate"))
            if J is not None:
                pts.append((lam, J / d["J_ml"]))
        lh = lambda_half(pts)
        tgt = GRID_LAMBDA_HALF[env]
        good = lh is not None and abs(lh - tgt) <= LHALF_TOL
        ok3 &= good
        print(f"  {env:16} measured {lh if lh else float('nan'):.6f} vs grid target "
              f"{tgt:.6f}  |diff| {abs(lh-tgt) if lh else float('nan'):.5f} "
              f"{'within' if good else 'OUTSIDE'} {LHALF_TOL}")
    v["P11.3"] = ok3
    print(f"P11.3: {'PASS' if ok3 else 'FAIL'}")

    print("=" * 74); print(" P11.4 mechanism (linearb arm, kicked bias)"); print("=" * 74)
    hit_u = tot_u = hit_s = tot_s = 0
    for env, d in C.items():
        for i, lam in enumerate(LAMBDAS):
            s = d["S_at_kAC"][i]
            if abs(s) < S_FLOOR:
                print(f"  {env:16} lam {lam:<6} EXCLUDED (|S| {abs(s):.1e} < floor)")
                continue
            b = med([abs(x) for x in cell(rows, env, "linearb", lam, "final_bias")])
            if b is None:
                print(f"  {env:16} lam {lam:<6} MISSING"); continue
            pred_unstable = s < 0
            read_unstable = b >= B_THRESH
            ok = pred_unstable == read_unstable
            if pred_unstable: tot_u += 1; hit_u += ok
            else: tot_s += 1; hit_s += ok
            print(f"  {env:16} lam {lam:<6} S {s:+9.6f} predict "
                  f"{'UNSTABLE' if pred_unstable else 'stable':8} |b| {b:7.3f} "
                  f"-> {'UNSTABLE' if read_unstable else 'stable':8} "
                  f"{'ok' if ok else 'MISS <<<'}")
    ok4 = hit_u >= UNSTABLE_MIN and hit_s >= STABLE_MIN and tot_u == 8 and tot_s == 21
    v["P11.4"] = ok4
    print(f"  unstable cells: {hit_u}/{tot_u} (need >= {UNSTABLE_MIN}/8)   "
          f"stable cells: {hit_s}/{tot_s} (need >= {STABLE_MIN}/21)")
    print(f"P11.4: {'PASS' if ok4 else 'FAIL'}")
    print(); print("VERDICTS:", {k: ('PASS' if b else 'FAIL') for k, b in v.items()})
    return 0


def main(argv):
    _assert_frozen()
    if len(argv) > 2 and argv[1] == "--check-clean":
        return check_clean(argv[2])
    if len(argv) > 1 and argv[1] != "--check-clean":
        return score(argv[1])
    print(f"E10 prereg frozen {FROZEN_ON}. 32 linear cells, 29 scored linearb "
          f"cells (8 unstable / 21 stable predicted), 3 excluded by the "
          f"{S_FLOOR} rate floor: {EXCLUDED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
