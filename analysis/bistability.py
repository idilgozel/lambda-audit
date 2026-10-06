"""Multiplicity of actor-critic equilibria along lambda: continuous collapse or
first-order (bistable) transition?  (2026-10-05; see knee-formula-findings.md, section 4)

    python3 analysis/bistability.py              # the paper's instances + slices
    python3 analysis/bistability.py --boundary   # locate Pi* = Theta c/(h s_e) at the transition

For each instance, scan lambda and count the roots of ghat(k; lambda) in k,
classify each root's stability under the averaged update k <- k - a*ghat
(stable iff d ghat/dk > 0), and report the bistable window [lam_lo, lam_hi]
in which a low-gain and a high-gain stable equilibrium coexist.
"""
import os, sys, json, math
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))
import kernel_general as kg  # noqa: E402
from gleam_bench import GLEBench  # noqa: E402
from scipy.optimize import brentq  # noqa: E402


def roots(b, lam, s_e=kg.S_E, n=240):
    ks = np.geomspace(0.2, 1.9 / b.h, n)
    vs = np.array([kg.ghat(b, k, lam, s_e) for k in ks])
    out = []
    for i in range(len(ks) - 1):
        a, c = vs[i], vs[i + 1]
        if np.isfinite(a) and np.isfinite(c) and np.sign(a) != np.sign(c):
            r = brentq(lambda k: kg.ghat(b, k, lam, s_e), ks[i], ks[i + 1], xtol=1e-7)
            stable = c > a           # ghat crosses from - to +  => d ghat/dk > 0
            out.append((r, stable))
    return out


def window(b, s_e=kg.S_E, u_lo=0.3, u_hi=4.0, n=60):
    """Scan u = -log10(1 - lambda) in [u_lo, u_hi]; return the bistable window
    (lambda values where >= 2 stable roots exist) or None, plus the scan."""
    us = np.linspace(u_lo, u_hi, n)
    scan = []
    for u in us:
        lam = 1 - 10 ** (-u)
        rs = roots(b, lam, s_e)
        nst = sum(1 for _, s in rs if s)
        scan.append((lam, len(rs), nst, [round(r, 3) for r, _ in rs]))
    bist = [lam for lam, nr, nst, _ in scan if nst >= 2]
    if not bist:
        return None, scan
    # refine edges by bisection on the stable-root count
    def nstable(lam):
        return sum(1 for _, s in roots(b, lam, s_e) if s)
    i_lo = next(i for i, (lam, nr, nst, _) in enumerate(scan) if nst >= 2)
    i_hi = max(i for i, (lam, nr, nst, _) in enumerate(scan) if nst >= 2)
    def edge(l_a, l_b):
        for _ in range(30):
            m = 0.5 * (l_a + l_b)
            if nstable(m) >= 2:
                l_b = m
            else:
                l_a = m
        return 0.5 * (l_a + l_b)
    lam_lo = edge(scan[i_lo - 1][0], scan[i_lo][0]) if i_lo > 0 else scan[0][0]
    if i_hi + 1 < len(scan):
        # above the window: count drops back to 1
        l_a, l_b = scan[i_hi][0], scan[i_hi + 1][0]
        for _ in range(30):
            m = 0.5 * (l_a + l_b)
            if nstable(m) >= 2:
                l_a = m
            else:
                l_b = m
        lam_hi = 0.5 * (l_a + l_b)
    else:
        lam_hi = scan[-1][0]
    return (lam_lo, lam_hi), scan


def boundary():
    def cls(b, se=0.09):
        w, _ = window(b, s_e=se, n=40)
        return ("BISTABLE %s" % [round(float(x), 5) for x in w]) if w else "continuous"
    print("Pi = Theta c / (h s_e); rho = 0.05, q = r = 1 unless stated")
    for c in (0.3, 0.4, 0.46, 0.48, 0.5, 0.6, 0.8):
        print(f"  c={c:<5} h=0.05 s_e=0.09  Pi={c / (0.05 * 0.09):5.0f}:", cls(GLEBench(cs=(c,), gammas=(1.0,), h=0.05)), flush=True)
    for se in (0.05, 0.06, 0.07):
        print(f"  c=0.3   h=0.05 s_e={se:<5} Pi={0.3 / (0.05 * se):5.0f}:", cls(GLEBench(cs=(0.3,), gammas=(1.0,), h=0.05), se), flush=True)
    for h, c in ((0.1, 1.08), (0.1, 0.81), (0.02, 0.162), (0.02, 0.216)):
        print(f"  c={c:<5} h={h:<4} s_e=0.09  Pi={c / (h * 0.09):5.0f}:", cls(GLEBench(cs=(c,), gammas=(1.0,), h=h)), flush=True)
    for g in (0.4, 2.0, 4.0):
        for c in (0.3, 1.0):
            print(f"  gamma={g} c={c} Pi={c / (0.05 * 0.09):5.0f}:", cls(GLEBench(cs=(c,), gammas=(g,), h=0.05)), flush=True)
    for rho in (0.2, 0.005):
        for c in (0.405, 0.54):
            print(f"  rho={rho} c={c} Pi={c / (0.05 * 0.09):5.0f}:", cls(GLEBench(cs=(c,), gammas=(1.0,), h=0.05, rho=rho)), flush=True)
    for r in (0.3, 3.0):
        for c in (0.405, 0.54):
            print(f"  r={r} c={c} Pi={c / (0.05 * 0.09):5.0f}:", cls(GLEBench(cs=(c,), gammas=(1.0,), h=0.05, r=r)), flush=True)


if __name__ == "__main__":
    if "--boundary" in sys.argv:
        boundary()
        raise SystemExit(0)
    cases = [("GLEAM_g1_h0.02", dict(cs=(0.3,), gammas=(1.0,), h=0.02), 0.09),
             ("GLEAM_g1_h0.05", dict(cs=(0.3,), gammas=(1.0,), h=0.05), 0.09),
             ("GLEAM_g2_h0.05", dict(cs=(0.3,), gammas=(2.0,), h=0.05), 0.09),
             ("GLEAM_g4_h0.05", dict(cs=(0.3,), gammas=(4.0,), h=0.05), 0.09),
             ("c=1.0", dict(cs=(1.0,), gammas=(1.0,), h=0.05), 0.09),
             ("c=3.0", dict(cs=(3.0,), gammas=(1.0,), h=0.05), 0.09),
             ("c=0.1", dict(cs=(0.1,), gammas=(1.0,), h=0.05), 0.09),
             ("s_e=0.01", dict(cs=(0.3,), gammas=(1.0,), h=0.05), 0.01),
             ("s_e=0.03", dict(cs=(0.3,), gammas=(1.0,), h=0.05), 0.03),
             ("s_e=0.3", dict(cs=(0.3,), gammas=(1.0,), h=0.05), 0.3),
             ("theta=3", dict(cs=(0.3,), gammas=(1.0,), h=0.05, theta=3.0), 0.09),
             ("h=0.01", dict(cs=(0.3,), gammas=(1.0,), h=0.01), 0.09),
             ("h=0.1", dict(cs=(0.3,), gammas=(1.0,), h=0.1), 0.09),
             ("2-mode balanced", dict(cs=(0.15, 0.15), gammas=(0.4, 2.5), h=0.05), 0.09),
             ("3-mode broad", dict(cs=(0.1, 0.1, 0.1), gammas=(0.3, 1.0, 3.0), h=0.05), 0.09)]
    out = []
    for name, kw, se in cases:
        b = GLEBench(**kw)
        w, scan = window(b, s_e=se)
        if w is None:
            print(f"{name:>18}: continuous collapse (single stable equilibrium at every lambda scanned)")
        else:
            # roots at the window centre
            mid = 0.5 * (w[0] + w[1])
            rs = roots(b, mid, se)
            print(f"{name:>18}: BISTABLE for lambda in [{w[0]:.5f}, {w[1]:.5f}]  "
                  f"(width {w[1]-w[0]:.5f}); at {mid:.5f} roots {[(round(r,3), 'S' if s else 'U') for r, s in rs]}")
        out.append(dict(name=name, window=w, params=kw, s_e=se))
    json.dump(out, open(os.path.join(ROOT, "analysis", "bistability_table.json"), "w"), indent=1, default=float)
