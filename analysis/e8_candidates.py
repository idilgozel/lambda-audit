#!/usr/bin/env python3
"""E8: candidate below-horizon detectors, evaluated on E6 as a training set.

EXPLORATORY. Nothing here is frozen, nothing here scores a registered
prediction, and the winner cannot be confirmed on the data used to select it.

The pilot's gap_lambda failed because std(G_MC) collapses on solved
deterministic tasks, so the statistic measured solvedness rather than the
bootstrap horizon. Any replacement has to clear two bars, not one:

  SEPARATION    it must split below-horizon from above-horizon cells on E6,
                where the label is known from the frozen lambda_half.
  NO FALSE FIRE it must stay quiet on action-independent cells. That is the
                trap that killed G3, and it is tested here on two negative
                controls: the E6 DECOUPLED arm, which is continuous and
                open loop by construction, and the pilot's RepeatPrevious
                cells, which are action independent for a different reason.

Run:  python3 analysis/e8_candidates.py
"""

import json
import math
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))

import prereg_phase2 as P  # noqa: E402

try:
    import gleam_native as native
    from gleam_bench import GLEBench
    HAVE_EXACT = True
except Exception:
    HAVE_EXACT = False

E6_PATH = os.path.join(ROOT, "results", "results", "results_phase2.jsonl")
PILOT_PATH = os.path.join(ROOT, "results", "results", "results_lambda_pilot.jsonl")
INSTANCE = P.E6_INSTANCE
KNEE = P.LAMBDA_HALF[INSTANCE]          # 0.985382, frozen before any E6 run

# Action-independent pilot environments. RepeatPrevious deals the next card
# whatever the agent answers, so no critic bias can reach the advantage.
OPEN_LOOP_PILOT = ["RepeatPreviousEasy", "RepeatPreviousMedium"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def exact_J(k):
    p = native.instance_params(INSTANCE)
    env = GLEBench(cs=(p["c"],), gammas=(p["gamma_bath"],), theta=p["theta"],
                   h=p["h"], q=p["q"], r=p["r"], rho=p["rho"], n_envs=1, seed=0)
    return float(env.J_gain(k))


def fit_envelope(pairs, nbins=6):
    """Model-free lower envelope of cost against gain, in log-log.

    Blind to the label. Bins by gain, takes the cheapest run in each bin, and
    fits a quadratic through those. The cheapest run at a given gain is the one
    behaving like a clean linear controller, so the envelope approximates the
    analytic J(k) curve without knowing it.
    """
    ks = np.array([k for k, _ in pairs])
    js = np.array([j for _, j in pairs])
    lk, lj = np.log(ks), np.log(js)
    edges = np.quantile(lk, np.linspace(0, 1, nbins + 1))
    px, py = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lk >= a) & (lk <= b)
        if m.sum() >= 2:
            i = np.argmin(lj[m])
            px.append(lk[m][i])
            py.append(lj[m][i])
    if len(px) < 3:
        return None
    c = np.polyfit(np.array(px), np.array(py), 2)
    lo, hi = float(np.exp(min(px))), float(np.exp(max(px)))

    def f(k):
        # Refuse to extrapolate. A quadratic in log-log fitted over gains of
        # order 1 to 10 says nothing at gain 0.002, and pretending otherwise
        # manufactured a false "quiet" reading on the decoupled arm.
        if not (lo <= k <= hi):
            return float("nan")
        return float(np.exp(np.polyval(c, np.log(k))))

    f.support = (lo, hi)
    return f


def cells(rows, key_env, arch=None, field_gain="final_gain", field_J="J_rate"):
    """Group by lambda. Returns {lam: [row, ...]}."""
    out = {}
    for r in rows:
        if r.get("env") != key_env:
            continue
        if arch and r.get("arch") != arch:
            continue
        out.setdefault(float(r["lam"]), []).append(r)
    return out


def stats_for(cell_rows, envelope):
    """The candidate statistics for one cell, from logged quantities only."""
    J = [r["J_rate"] for r in cell_rows if r.get("J_rate") is not None]
    k = [r["final_gain"] for r in cell_rows if r.get("final_gain") is not None]
    gl = [abs(r["gap_lambda"]) for r in cell_rows
          if r.get("gap_lambda") is not None]
    out = {}
    if HAVE_EXACT and J and k:
        out["A_oracle"] = st.median([j / exact_J(max(kk, 1e-6))
                                     for j, kk in zip(J, k)])
    if envelope and J and k:
        vals = [j / envelope(kk) for j, kk in zip(J, k)
                if math.isfinite(envelope(kk))]
        out["B_proxy"] = st.median(vals) if vals else float("nan")
    if len(J) >= 2:
        out["C_bimodal"] = max(J) / min(J)
    if gl:
        out["D_gaplam"] = st.median(gl)
    return out


def separation(below, above):
    """Cleanly separable if the ranges do not overlap. Returns (auc, margin)."""
    if not below or not above:
        return float("nan"), float("nan")
    wins = sum(1 for b in below for a in above if b > a)
    ties = sum(1 for b in below for a in above if b == a)
    auc = (wins + 0.5 * ties) / (len(below) * len(above))
    margin = min(below) / max(above) if max(above) > 0 else float("inf")
    return auc, margin


def main():
    e6 = load(E6_PATH)
    coupled = [r for r in e6 if r.get("arm") == "coupled"]
    decoupled = [r for r in e6 if r.get("arm") == "decoupled"]

    envelope = fit_envelope([(r["final_gain"], r["J_rate"]) for r in coupled
                             if r["final_gain"] > 0.05])

    print("=" * 78)
    print(" E8 candidate detectors, trained on E6. EXPLORATORY, nothing frozen.")
    print("=" * 78)
    print(f"\n  label from the frozen lambda_half = {KNEE:.6f}")
    print(f"  below horizon: lambda < {KNEE:.3f}   above: lambda > {KNEE:.3f}")
    print(f"  exact analytic J available: {HAVE_EXACT}")

    cc = cells(coupled, INSTANCE)
    rows_out = {}
    print(f"\n  {'lambda':>7} {'label':>6} {'A_oracle':>9} {'B_proxy':>9} "
          f"{'C_bimodal':>10} {'D_gaplam':>9}")
    for lam in sorted(cc):
        s = stats_for(cc[lam], envelope)
        lab = "below" if lam < KNEE else "above"
        rows_out[lam] = (lab, s)
        print(f"  {lam:7} {lab:>6} "
              f"{s.get('A_oracle', float('nan')):9.3f} "
              f"{s.get('B_proxy', float('nan')):9.3f} "
              f"{s.get('C_bimodal', float('nan')):10.3f} "
              f"{s.get('D_gaplam', float('nan')):9.3f}")

    print("\n  separation on E6 coupled (AUC 1.0 means perfectly ordered,")
    print("  margin > 1 means the ranges do not overlap at all):")
    print(f"    {'candidate':<12} {'AUC':>6} {'margin':>8}   verdict")
    summary = {}
    for cand in ("A_oracle", "B_proxy", "C_bimodal", "D_gaplam"):
        below = [s[cand] for lab, s in rows_out.values()
                 if lab == "below" and cand in s]
        above = [s[cand] for lab, s in rows_out.values()
                 if lab == "above" and cand in s]
        auc, margin = separation(below, above)
        summary[cand] = (auc, margin)
        verdict = ("separates cleanly" if margin > 1.0 else
                   "orders but overlaps" if auc >= 0.9 else "does not separate")
        print(f"    {cand:<12} {auc:6.2f} {margin:8.2f}   {verdict}")

    # ---- false positive check -------------------------------------------
    print("\n" + "=" * 78)
    print(" FALSE POSITIVE CHECK. These cells are action independent, so a")
    print(" below-horizon detector must stay QUIET on every one of them.")
    print("=" * 78)

    print("\n  negative control 1: the E6 DECOUPLED arm")
    print("  continuous and open loop by construction, so every candidate is")
    print("  computable, which makes this the sharpest available control.")
    dc = cells(decoupled, INSTANCE + "_dec")
    print(f"  {'lambda':>7} {'A_oracle':>9} {'B_proxy':>9} {'C_bimodal':>10} "
          f"{'D_gaplam':>9}")
    dec_stats = {c: [] for c in ("A_oracle", "B_proxy", "C_bimodal", "D_gaplam")}
    for lam in sorted(dc):
        s = stats_for(dc[lam], envelope)
        for c in dec_stats:
            if c in s:
                dec_stats[c].append(s[c])
        print(f"  {lam:7} {s.get('A_oracle', float('nan')):9.3f} "
              f"{s.get('B_proxy', float('nan')):9.3f} "
              f"{s.get('C_bimodal', float('nan')):10.3f} "
              f"{s.get('D_gaplam', float('nan')):9.3f}")

    print("\n  negative control 2: the pilot's action-independent cells")
    print("  RepeatPrevious deals the next card whatever the agent answers.")
    print("  Only the gap statistic is computable here: the discrete harness")
    print("  logs no cost rate and no linear gain, which is itself a finding.")
    pilot = load(PILOT_PATH)
    pilot_worst = 0.0
    for env in OPEN_LOOP_PILOT:
        for arch in ("mlp", "stack8", "stack64", "gru128"):
            vals = []
            for lam in (0.9, 0.95, 0.99, 1.0):
                cr = [r for r in pilot if r["env"] == env and r["arch"] == arch
                      and math.isclose(float(r["lam"]), lam)
                      and r.get("gap_lambda") is not None]
                if cr:
                    m = st.median(abs(r["gap_lambda"]) for r in cr)
                    vals.append((lam, m))
                    pilot_worst = max(pilot_worst, m)
            if vals:
                s = ", ".join(f"{l}:{v:.3f}" for l, v in vals)
                print(f"    D_gaplam  {env:22} {arch:8} {s}")
    print(f"\n    worst open-loop pilot cell: {pilot_worst:.3f}. Any threshold "
          f"at or below this false fires.")

    # ---- the decision -----------------------------------------------------
    print("\n" + "=" * 78)
    print(" READING")
    print("=" * 78)
    print("  A usable detector needs a threshold that sits ABOVE every")
    print("  negative control and BELOW every below-horizon cell. If that")
    print("  window is empty the candidate is dead, however good its AUC.\n")
    print(f"  {'candidate':<12} {'AUC':>5} {'neg max':>8} {'below min':>10} "
          f"{'window':>16}  verdict")
    for cand in ("A_oracle", "B_proxy", "C_bimodal", "D_gaplam"):
        auc, _ = summary[cand]
        quiet = [x for x in dec_stats[cand] if math.isfinite(x)]
        neg = max(quiet) if quiet else float("nan")
        if cand == "D_gaplam" and math.isfinite(pilot_worst):
            neg = max(neg, pilot_worst)      # pilot control applies to this one
        below = [s[cand] for lab, s in rows_out.values()
                 if lab == "below" and cand in s and math.isfinite(s[cand])]
        bmin = min(below) if below else float("nan")
        ok = math.isfinite(neg) and math.isfinite(bmin) and neg < bmin
        window = f"({neg:.3f}, {bmin:.3f})" if ok else "EMPTY"
        verdict = ("usable on this data" if ok else
                   "not evaluable" if not math.isfinite(neg) else "FALSE FIRES")
        print(f"  {cand:<12} {auc:5.2f} {neg:8.3f} {bmin:10.3f} "
              f"{window:>16}  {verdict}")
    print("\n  Exploratory. Selected on E6, so none of this is confirmatory,")
    print("  and the winner must be frozen before any external benchmark run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
