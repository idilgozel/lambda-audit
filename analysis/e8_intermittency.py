#!/usr/bin/env python3
"""E8: intermittency candidates against a corrected negative set.

EXPLORATORY THROUGHOUT. Nothing here is frozen. E8 freezes after E7 lands.

Supersedes e8_candidates.py. Two changes.

1. The negative set was incomplete. It held only action-independent cells. A
   deployed detector must also stay silent on CLOSED-LOOP tasks run ABOVE
   their horizon, the ordinary practical case. Cartpole sits above its horizon
   of 0.88 at every lambda the pilot ran.

2. Two intermittency candidates from the curves alone, per run, needing
   neither multiple seeds nor privileged information, plus a sign-safe third
   that this evaluation forced into existence.

The decisive number is the WINDOW, the gap between the worst negative and the
weakest positive. Every candidate in the previous table ordered E6 perfectly
and that told us nothing.

Run:  python3 analysis/e8_intermittency.py
"""

import csv
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import prereg_phase2 as P  # noqa: E402

RES = os.path.join(ROOT, "results", "results")
E6 = os.path.join(RES, "results_phase2.jsonl")
PILOT = os.path.join(RES, "results_lambda_pilot.jsonl")
INSTANCE = P.E6_INSTANCE
KNEE = P.LAMBDA_HALF[INSTANCE]
J_ML = P.BASELINES[INSTANCE]["J_ml"]

MULT, SPLIT = 2.0, 0.5
CARTPOLE_KNEE = 0.88
OPEN_LOOP = ("RepeatPrevious",)
CLOSED_PILOT = ("PositionOnlyCartPole",)

# Boundary cell. lambda 0.98 sits 0.005 BELOW the predicted knee of 0.985382,
# inside the width of the transition itself. prereg_lambda_pilot.py set the
# precedent of excluding the cell that straddles the knee (G3_EXCLUDED_CELLS).
# Both labellings are reported, because which one you use changes the answer.
BOUNDARY_LAMBDA = 0.98


def costs_of(path):
    """Cost through training. eval_return is -J_rate on GLEAM and a plain
    return on the discrete tasks, so cost = -eval_return in both. The sign
    convention is not incidental; it is the whole story in section 4."""
    xs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get("eval_return", "")
            if v not in ("", None):
                xs.append(-float(v))
    return xs


def curve_path(env, arch, lam, seed):
    return os.path.join(RES, f"curves_{env}_{arch}_lam{lam}_s{seed}.csv")


def late(xs, split=SPLIT):
    return xs[int(len(xs) * split):]


def i_abs(xs, base, mult=MULT, split=SPLIT):
    L = late(xs, split)
    if not L or base is None:
        return float("nan")
    return sum(1 for c in L if c > mult * base) / len(L)


def i_self(xs, mult=MULT, split=SPLIT):
    L = late(xs, split)
    if not L:
        return float("nan")
    m = st.median(L)
    return sum(1 for c in L if c > mult * m) / len(L)


def i_selfabs(xs, mult=MULT, split=SPLIT):
    """Sign-safe: excursion measured in units of |median|, c > m+(mult-1)|m|.
    Identical to i_self when the median cost is positive."""
    L = late(xs, split)
    if not L:
        return float("nan")
    m = st.median(L)
    return sum(1 for c in L if c > m + (mult - 1.0) * abs(m)) / len(L)


def inventory():
    runs = []
    for line in open(E6):
        if not line.strip():
            continue
        r = json.loads(line)
        p = curve_path(r["env"], r["arch"], r["lam"], r["seed"])
        if os.path.exists(p):
            runs.append(dict(row=r, curve=p, env=r["env"], arch=r["arch"],
                             lam=float(r["lam"]), seed=r["seed"],
                             arm=r["arm"], fam="E6"))
    for line in open(PILOT):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("seed") not in P.SEEDS:
            continue
        p = curve_path(r["env"], r["arch"], r["lam"], r["seed"])
        if os.path.exists(p):
            runs.append(dict(row=r, curve=p, env=r["env"], arch=r["arch"],
                             lam=float(r["lam"]), seed=r["seed"],
                             arm="n/a", fam="pilot"))
    return runs


def knee_of(env):
    """Each instance has its own critical horizon. Using g1_h0.05's knee for
    all four, as the first version did, mislabels every E7 cell."""
    return P.LAMBDA_HALF.get(env.replace("_dec", ""), KNEE)


def boundary_lambda(env):
    """The grid point immediately below that instance's knee is the boundary
    cell, excluded on the same precedent the pilot set. It is 0.98 only for
    g1_h0.05; for the others it is wherever the knee falls."""
    k = knee_of(env)
    below = [l for l in P.LAMBDAS if l < k]
    return below[-1] if below else None


def label(run, strict):
    """POSITIVE / NEG_A / NEG_B / NEG_B_E6 / None."""
    if run["fam"] == "E6":
        if run["arm"] == "decoupled":
            return "NEG_A"
        k = knee_of(run["env"])
        if run["lam"] == boundary_lambda(run["env"]):
            return "POSITIVE" if strict else None
        return "POSITIVE" if run["lam"] < k else "NEG_B_E6"
    if run["env"].startswith(OPEN_LOOP):
        return "NEG_A"
    if run["env"].startswith(CLOSED_PILOT):
        return "NEG_B" if run["lam"] >= CARTPOLE_KNEE else None
    return None


def values(run):
    xs = costs_of(run["curve"])
    base = J_ML if run["fam"] == "E6" else None
    gl = run["row"].get("gap_lambda")
    return {
        "D_gaplam": abs(float(gl)) if gl is not None else float("nan"),
        "E_abs": i_abs(xs, base),
        "E_self": i_self(xs),
        "E_selfabs": i_selfabs(xs),
    }


def win(pos, neg):
    pos = [x for x in pos if math.isfinite(x)]
    neg = [x for x in neg if math.isfinite(x)]
    if not pos or not neg:
        return None
    lo, hi = max(neg), min(pos)
    return lo, hi, (hi / lo if lo > 0 else float("inf")), hi > lo


def show(w):
    if w is None:
        return f"{'not evaluable':>24}"
    lo, hi, ratio, ok = w
    s = f"({lo:.3f}, {hi:.3f})"
    return f"{s:>16}{ratio:7.2f}x" if ok else f"{s:>16}  CLOSED"


CANDS = ["D_gaplam", "E_abs", "E_self", "E_selfabs"]


def cell_medians(runs, cand):
    """Group runs into cells and return the per-cell median of a candidate."""
    g = {}
    for r in runs:
        k = (r["env"], r["arch"], r["lam"])
        v = r["v"][cand]
        if math.isfinite(v):
            g.setdefault(k, []).append(v)
    return {k: st.median(v) for k, v in g.items()}


def report(runs, strict):
    for r in runs:
        r["cls"] = label(r, strict)
    pos = [r for r in runs if r["cls"] == "POSITIVE"]
    nA = [r for r in runs if r["cls"] == "NEG_A"]
    nB = [r for r in runs if r["cls"] == "NEG_B"]
    nBe = [r for r in runs if r["cls"] == "NEG_B_E6"]

    tag = ("STRICT, boundary lambda 0.98 counted as a positive" if strict
           else "BOUNDARY EXCLUDED, lambda 0.98 dropped")
    print(f"\n{'=' * 78}\n {tag}\n{'=' * 78}")
    print(f"  positives {len(pos):3d}   NEG-A {len(nA):3d}   "
          f"NEG-B cartpole {len(nB):3d}   NEG-B E6-above {len(nBe):3d}")
    print(f"\n  per run windows")
    print(f"  {'candidate':<11}{'vs NEG-A':>24}{'vs NEG-B':>24}"
          f"{'vs A+B':>24}{'vs A+B+E6above':>24}")
    for c in CANDS:
        pv = [r["v"][c] for r in pos]
        print(f"  {c:<11}"
              f"{show(win(pv, [r['v'][c] for r in nA]))}"
              f"{show(win(pv, [r['v'][c] for r in nB]))}"
              f"{show(win(pv, [r['v'][c] for r in nA + nB]))}"
              f"{show(win(pv, [r['v'][c] for r in nA + nB + nBe]))}")

    print(f"\n  per cell median windows (the basis of the reference numbers)")
    print(f"  {'candidate':<11}{'vs NEG-A':>24}{'vs NEG-B':>24}"
          f"{'vs A+B':>24}{'vs A+B+E6above':>24}")
    for c in CANDS:
        pm = list(cell_medians(pos, c).values())
        print(f"  {c:<11}"
              f"{show(win(pm, list(cell_medians(nA, c).values())))}"
              f"{show(win(pm, list(cell_medians(nB, c).values())))}"
              f"{show(win(pm, list(cell_medians(nA + nB, c).values())))}"
              f"{show(win(pm, list(cell_medians(nA + nB + nBe, c).values())))}")
    return pos, nA, nB, nBe


def worst_negatives(runs):
    print(f"\n{'=' * 78}\n THE LIMITATION THAT SURVIVES\n{'=' * 78}")
    print("  A cell that SOLVES its task and still scores high is the")
    print("  solvedness pathology that killed G3. It is not cured.")
    for c in CANDS:
        cm = cell_medians([r for r in runs if r["cls"] in ("NEG_A", "NEG_B")], c)
        if not cm:
            continue
        (env, arch, lam), v = max(cm.items(), key=lambda t: t[1])
        pr = [r["row"].get("deployed_return") for r in runs
              if (r["env"], r["arch"], r["lam"]) == (env, arch, lam)]
        pr = [x for x in pr if x is not None]
        dep = f"{st.median(pr):.4f}" if pr else "n/a"
        print(f"  {c:<11} worst negative cell median {v:.4f}  "
              f"{env} {arch} lam={lam}, deployed_return {dep}")


def decoupled_check(runs):
    print(f"\n{'=' * 78}\n DISQUALIFYING CHECK: the E6 decoupled arm\n{'=' * 78}")
    print("  It is closed-loop-free by construction. Any candidate that fires")
    print("  here is detecting something other than the mechanism.")
    dec = [r for r in runs if r["fam"] == "E6" and r["arm"] == "decoupled"]
    for c in ("E_abs", "E_self", "E_selfabs"):
        v = [r["v"][c] for r in dec if math.isfinite(r["v"][c])]
        flag = "  <-- DISQUALIFIED" if max(v) > 0.05 else ""
        print(f"  {c:<11} max {max(v):.4f}  mean {st.mean(v):.4f}  "
              f"over {len(v)} runs{flag}")


def sign_problem(runs):
    print(f"\n{'=' * 78}\n SIGN CONVENTION, and why E_self cannot ship\n{'=' * 78}")
    disc = [r for r in runs if r["fam"] == "pilot"]
    neg = [r for r in disc if st.median(late(costs_of(r["curve"]))) < 0]
    print(f"  pilot runs whose late-half median COST is negative: "
          f"{len(neg)} of {len(disc)}")
    print("  On those, 2 x median moves DOWNWARD, so 'cost above twice the")
    print("  median' is satisfied by nearly every point. E_self is not noisy")
    print("  there, it is inverted, and it reads 1.000 on solved tasks.")
    for r in neg[:3]:
        m = st.median(late(costs_of(r["curve"])))
        print(f"    {r['env']:24} {r['arch']:8} lam={r['lam']:<5} "
              f"median cost {m:+.4f}  E_self={r['v']['E_self']:.3f}  "
              f"E_selfabs={r['v']['E_selfabs']:.3f}")
    print("  E_abs is not computable here at all: it needs J_ml, an analytic")
    print("  optimum no external benchmark supplies.")


def sensitivity(runs):
    print(f"\n{'=' * 78}\n SENSITIVITY. A threshold that works at one setting "
          f"is useless.\n{'=' * 78}")
    for r in runs:
        r["cls"] = label(r, strict=False)
    pos = [r for r in runs if r["cls"] == "POSITIVE"]
    negs = [r for r in runs if r["cls"] in ("NEG_A", "NEG_B", "NEG_B_E6")]
    print(f"\n  E_selfabs, boundary excluded, all negatives incl. E6 above")
    print(f"  {'mult':>5} " + "".join(f"{f'last {int(100*(1-s))}%':>22}"
                                      for s in (0.7, 0.5, 0.3)))
    for mult in (1.5, 2.0, 3.0, 5.0):
        cells_ = []
        for split in (0.7, 0.5, 0.3):
            pv = [i_selfabs(costs_of(r["curve"]), mult, split) for r in pos]
            nv = [i_selfabs(costs_of(r["curve"]), mult, split) for r in negs]
            cells_.append(show(win(pv, nv)))
        print(f"  {mult:>5} " + "".join(c.replace(" ", "", 0) for c in cells_))


def locator(runs):
    print(f"\n{'=' * 78}\n EXPLORATORY TRANSITION LOCATOR, ready for E7"
          f"\n{'=' * 78}")
    print("  P7.1 is frozen and is scored exactly as registered, by half")
    print("  collapse of final excess cost. This is a SEPARATE, EXPLORATORY")
    print("  estimator: the lambda at which late-half intermittency switches")
    print("  off. On E6 the measure is binary, so it brackets sharply.")
    for inst in sorted(set(r["env"] for r in runs
                           if r["fam"] == "E6" and r["arm"] == "coupled")):
        pts = []
        for lam in P.LAMBDAS:
            v = [i_selfabs(costs_of(r["curve"]))
                 for r in runs if r["env"] == inst and r["lam"] == lam
                 and r["arm"] == "coupled"]
            if v:
                pts.append((lam, st.median(v)))
        lo, hi = locate(pts)
        pred = P.LAMBDA_HALF.get(inst)
        print(f"\n  {inst}")
        print("    " + "  ".join(f"{l}:{v:.3f}" for l, v in pts))
        if lo is None:
            print("    no transition bracketed")
        else:
            mid = 0.5 * (lo + hi)
            extra = (f", frozen prediction {pred:.5f}, "
                     f"|midpoint - predicted| = {abs(mid - pred):.5f}"
                     if pred else "")
            print(f"    intermittency switches off between {lo} and {hi}, "
                  f"midpoint {mid:.4f}{extra}")


def locate(pts, thresh=0.05):
    """Bracket the lambda where the median measure falls below thresh."""
    for (l0, v0), (l1, v1) in zip(pts, pts[1:]):
        if v0 > thresh >= v1:
            return l0, l1
    return None, None


def main():
    runs = inventory()
    for r in runs:
        r["v"] = values(r)
    print("=" * 78)
    print(" E8 INTERMITTENCY, corrected negative set. EXPLORATORY, nothing "
          "frozen.")
    print("=" * 78)
    print(f"  knee {KNEE:.6f}   J_ml {J_ML:.6f}   2*J_ml {2 * J_ML:.4f}   "
          f"mult {MULT}  split {SPLIT}")
    report(runs, strict=True)
    pos, nA, nB, nBe = report(runs, strict=False)
    worst_negatives(runs)
    decoupled_check(runs)
    sign_problem(runs)
    sensitivity(runs)
    locator(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
