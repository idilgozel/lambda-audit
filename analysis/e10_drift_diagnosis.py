"""Post-hoc diagnosis of the E10 P11.1/P11.2 failures ("above-knee drift").

Written 2026-08-21, AFTER E10 was scored. This is a diagnosis of registered
failures, not a rescue: P11.1 and P11.2 remain FAIL as scored by prereg_e10.py.
The findings document is e10-above-knee-drift.md; every number quoted there is
computed here.

Tests, in order of evidential weight:

T-1  DIRECTION (seed level). Count runs with final_gain < k_AC(lambda).
     Systematic bias predicts all of them; symmetric diffusion predicts a
     binomial split.

T-2  COST MAP AT THE MEASURED GAIN. J_measured / J_analytic(k_measured) per
     cell, J_analytic = GLEBench.J_gain. If ~1 everywhere, P11.2's failures
     are P11.1's failures pushed through an intact cost map.

T-3  BRANCH ALIGNMENT. Classify cells by which branch of the aliased
     equilibrium map k_AC(lambda) sits on (inflated vs collapsed) and
     cross-tabulate with P11.1 pass/fail.

T-4  EXCLUSIONS. (a) discount mismatch: the harness discount must equal
     exp(-rho*h) per instance, bitwise; (b) GAE tail truncation: report
     (beta*lambda)^num_steps for passing and failing cells.

Descriptive (not evidence): the exact GAE(lambda) gradient at the observed
resting gain, ghat(k_obs). To linear order this is stiffness * displacement,
i.e. a reparameterisation of the drift, so it checks nothing — it only
expresses the drift in gradient units for future comparison against candidate
bias mechanisms.

Run from the repository root:
    python3 analysis/e10_drift_diagnosis.py results/results/results_e10.jsonl
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "vendor", "gleam"))
import theorem2_learning as T           # noqa: E402
import lstd_lambda as L                 # noqa: E402
from gleam_bench import GLEBench        # noqa: E402

LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
S_E = 0.09
GAIN_TOL = 0.15
NUM_STEPS = 128
RHO = 0.05

# k_AC(lambda) exactly as frozen in prereg_e10.py's C table.
C = json.loads(r'''{"GLEAM_g1_h0.02": {"gamma": 1.0, "h": 0.02, "k_AC": [60.071834, 57.80993, 54.142926, 46.176014, 37.359041, 25.550589, 1.671172, 1.54499], "knee": 0.996742}, "GLEAM_g1_h0.05": {"gamma": 1.0, "h": 0.05, "k_AC": [22.426983, 20.546902, 17.670634, 11.394604, 2.197126, 1.756354, 1.548754, 1.506599], "knee": 0.983837}, "GLEAM_g2_h0.05": {"gamma": 2.0, "h": 0.05, "k_AC": [20.07708, 16.82564, 11.046191, 1.646322, 1.42903, 1.343181, 1.281927, 1.26748], "knee": 0.961432}, "GLEAM_g4_h0.05": {"gamma": 4.0, "h": 0.05, "k_AC": [14.515075, 2.175412, 1.453986, 1.226369, 1.164381, 1.135285, 1.112818, 1.107307], "knee": 0.855871}}''')

BENCH_KW = {
    "GLEAM_g1_h0.02": dict(gammas=(1.0,), h=0.02),
    "GLEAM_g1_h0.05": dict(gammas=(1.0,), h=0.05),
    "GLEAM_g2_h0.05": dict(gammas=(2.0,), h=0.05),
    "GLEAM_g4_h0.05": dict(gammas=(4.0,), h=0.05),
}


def set_h(h):
    """theorem2_learning holds h/beta as module globals; repoint them."""
    T.H = h
    T.BETA = np.exp(-T.RHO * h)


def ghat(env, k, lam):
    d = C[env]
    set_h(d["h"])
    return L.gae_gradient(d["gamma"], k, S_E, lam)


def load(path):
    """(env, lam) -> per-seed [(final_gain, J_rate, gamma_rl)], linear arm."""
    cells = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("arch") != "linear":
            continue
        cells.setdefault((r["env"], r["lam"]), []).append(
            (r["final_gain"], r["J_rate"], r["gamma"]))
    return cells


def main(path):
    cells = load(path)
    benches = {env: GLEBench(**kw) for env, kw in BENCH_KW.items()}

    # Self-check: the frozen k_AC values are roots of the exact gradient.
    worst = max(abs(ghat(env, d["k_AC"][i], lam))
                for env, d in C.items() for i, lam in enumerate(LAMBDAS))
    print(f"self-check  max |ghat(k_AC)| over 32 cells: {worst:.2e}  "
          f"(frozen table reproduced)\n")

    # T-4a: discount identity, bitwise, per instance.
    print("T-4a discount identity (harness gamma_RL == exp(-rho*h), bitwise)")
    for env, d in C.items():
        want = float(np.exp(-RHO * d["h"]))
        got = {g for rows in
               (cells[(env, lam)] for lam in LAMBDAS)
               for _, _, g in rows}
        assert got == {want}, (env, got, want)
        print(f"  {env:>16}: gamma_RL = {want!r}  OK")

    # Per-cell table.
    print(f"\n{'cell':>18} {'lam':>6} {'k_AC':>8} {'seeds k_obs':>26} "
          f"{'drift%':>7} {'P11.1':>6} {'branch':>10} {'Jm/Jan':>7} "
          f"{'ghat(kobs)':>11}")
    rows = []
    for env, d in C.items():
        for i, lam in enumerate(LAMBDAS):
            kstar = d["k_AC"][i]
            gains = sorted(g for g, _, _ in cells[(env, lam)])
            g_med = float(np.median(gains))
            J_med = float(np.median([j for _, j, _ in cells[(env, lam)]]))
            drift = abs(g_med / kstar - 1)
            ok = drift <= GAIN_TOL
            branch = "inflated" if kstar > 5.0 else "collapsed"
            ratio = J_med / benches[env].J_gain(g_med)
            gh = ghat(env, g_med, lam)
            below = sum(1 for g in gains if g < kstar)
            rows.append(dict(env=env, lam=lam, kstar=kstar, kobs=g_med,
                             ok=ok, branch=branch, jratio=ratio,
                             n=len(gains), below=below))
            print(f"{env:>18} {lam:>6} {kstar:>8.3f} "
                  f"{str([round(g, 3) for g in gains]):>26} "
                  f"{100 * drift:>6.1f}% {'pass' if ok else 'FAIL':>6} "
                  f"{branch:>10} {ratio:>7.4f} {gh:>+11.2e}")

    # T-1: direction at seed level.
    n_runs = sum(r["n"] for r in rows)
    n_below = sum(r["below"] for r in rows)
    print(f"\nT-1 direction: {n_below}/{n_runs} runs finished below k_AC "
          f"(symmetric diffusion: p = 2^-{n_runs})")

    # T-2: cost map.
    jr = [r["jratio"] for r in rows]
    jf = [r["jratio"] for r in rows if not r["ok"]]
    print(f"T-2 cost map:  all cells median {np.median(jr):.4f}, "
          f"worst |ratio-1| {max(abs(x - 1) for x in jr):.4f}; "
          f"failing cells median {np.median(jf):.4f}, "
          f"worst {max(abs(x - 1) for x in jf):.4f}")

    # T-3: branch alignment.
    tab = {}
    for r in rows:
        tab[(r["branch"], r["ok"])] = tab.get((r["branch"], r["ok"]), 0) + 1
    print(f"T-3 branch:    collapsed  fail {tab.get(('collapsed', False), 0)} "
          f"pass {tab.get(('collapsed', True), 0)}   "
          f"inflated  fail {tab.get(('inflated', False), 0)} "
          f"pass {tab.get(('inflated', True), 0)}")
    exc = [r for r in rows if r["branch"] == "inflated" and not r["ok"]]
    for r in exc:
        print(f"               inflated-branch exception: {r['env']} "
              f"lam {r['lam']} (knee {C[r['env']]['knee']}; nearest-to-fold "
              f"cell of its instance)")

    # T-4b: tail weight excludes GAE truncation as the driver.
    print("T-4b tail weight (beta*lam)^128, pass vs fail:")
    for env, lam in [("GLEAM_g1_h0.02", 0.995), ("GLEAM_g1_h0.05", 0.99)]:
        d = C[env]
        w = (np.exp(-RHO * d["h"]) * lam) ** NUM_STEPS
        i = LAMBDAS.index(lam)
        r = next(x for x in rows if x["env"] == env and x["lam"] == lam)
        print(f"  {env:>16} lam {lam}: w = {w:.3f}  "
              f"({'pass' if r['ok'] else 'FAIL'}, "
              f"drift {abs(r['kobs'] / d['k_AC'][i] - 1) * 100:.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1])
