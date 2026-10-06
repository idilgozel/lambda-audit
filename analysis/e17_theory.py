"""E17 theory table: the exact averaged GAE(lambda) gradient ghat(k; lambda) of the
linear-Gaussian family (identity I, analysis/knee_formula.py) at the registered
(lambda, k0) cells of the drift-direction test, and the equilibrium structure
(roots of ghat in k with their stability) at each registered lambda.

    python3 analysis/e17_theory.py            # prints the table, writes analysis/e17_theory_table.json
    python3 analysis/e17_theory.py --lams 0.9930 0.9957 0.9975 --k0 3.5 10.5

Numbers are computed, never typed: prereg_e17.py embeds the output and re-derives
it at every scoring call (tolerance 1e-9) when vendor/gleam/kernel_general.py is
importable. Stability under the averaged update k <- k - a*ghat: stable iff
d ghat/dk > 0 (ghat crosses from negative to positive).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))
import kernel_general as kg  # noqa: E402
from gleam_bench import GLEBench  # noqa: E402

INSTANCE = dict(name="GLEAM_g1_h0.02", cs=(0.3,), gammas=(1.0,), h=0.02)


def roots(b, lam, s_e=kg.S_E, n=400, k_lo=0.2, k_hi=None):
    k_hi = k_hi or 1.9 / b.h
    ks = np.geomspace(k_lo, k_hi, n)
    vs = np.array([kg.ghat(b, k, lam, s_e) for k in ks])
    out = []
    for i in range(len(ks) - 1):
        a, c = vs[i], vs[i + 1]
        if np.isfinite(a) and np.isfinite(c) and np.sign(a) != np.sign(c):
            r = brentq(lambda k: kg.ghat(b, k, lam, s_e), ks[i], ks[i + 1], xtol=1e-9)
            out.append((float(r), bool(c > a)))
    return out


def table(lams, k0s):
    kg.validate()
    b = GLEBench(cs=INSTANCE["cs"], gammas=INSTANCE["gammas"], h=INSTANCE["h"])
    out = {"instance": INSTANCE["name"], "s_e": kg.S_E, "lams": list(lams), "k0s": list(k0s),
           "roots": {}, "ghat": {}}
    for lam in lams:
        out["roots"][f"{lam:.4f}"] = [{"k": round(r, 6), "stable": s} for r, s in roots(b, lam)]
        for k0 in k0s:
            out["ghat"][f"{lam:.4f}|{k0:g}"] = float(kg.ghat(b, k0, lam))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="+", default=[0.9930, 0.9957, 0.9966, 0.9975])
    ap.add_argument("--k0", type=float, nargs="+", default=[3.5, 10.5])
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "e17_theory_table.json"))
    a = ap.parse_args()
    t = table(a.lams, a.k0)
    print(f"{t['instance']}  s_e = {t['s_e']}")
    for lam in a.lams:
        rs = t["roots"][f"{lam:.4f}"]
        print(f"  lambda {lam:.4f}: equilibria " + ", ".join(f"{r['k']:.3f} ({'S' if r['stable'] else 'U'})" for r in rs))
        for k0 in a.k0:
            g = t["ghat"][f"{lam:.4f}|{k0:g}"]
            print(f"      k0 = {k0:5.1f}: ghat = {g:+.3e}  -> averaged update drifts {'DOWN' if g > 0 else 'UP'}")
    with open(a.out, "w") as f:
        json.dump(t, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
