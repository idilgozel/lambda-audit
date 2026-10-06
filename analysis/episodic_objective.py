"""The objective the harness TRAINS on versus the one the theory analyses.

GleamNative truncates every 1024 steps and the training wrapper resets from the
uncontrolled stationary distribution (Var v = 1), so each training episode begins
with a settling transient whose cost grows with the gain. This script computes,
exactly from the shipped matrices (harness/gleam_instances.npz: F, G, Lw, L0) and
the vendored kernel (vendor/gleam), for each instance and gain:

  * the per-step cost averaged over a 1024-step episode started from L0 L0^T,
    against the stationary per-step cost (the deployed evaluation's quantity);
  * the gain-gradient of that episodic cost against the true stationary gradient
    and the exact aliased gradient ghat(k; lambda) of identity (I);
  * the equilibria of the episodic objective, ghat + d(transient)/dk = 0, against
    k_AC(lambda).

Exploration variance s_e = 0.09 is included in both objectives, as in training.
See e17-pilot-findings.md section 4.

    python3 analysis/episodic_objective.py            -> table + analysis/episodic_objective_table.json
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import kernel_general as kg  # noqa: E402
from gleam_bench import GLEBench  # noqa: E402
import gleam_native as gn  # noqa: E402

S_E = 0.09
T = 1024
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]


class Episodic:
    def __init__(self, inst):
        d = gn._load()
        self.F, self.G, self.Lw, self.L0 = (d[f"{inst}__{k}"] for k in ("F", "G", "Lw", "L0"))
        p = gn.instance_params(inst)
        self.h, self.q, self.r = p["h"], p["q"], p["r"]
        self.bench = GLEBench(cs=(p["c"],), gammas=(p["gamma_bath"],), theta=p["theta"], h=p["h"])
        assert np.allclose(self.F, self.bench.F) and np.allclose(self.G, self.bench.G), "npz and vendor disagree"

    def _loop(self, k):
        Krow = np.zeros((1, 2)); Krow[0, 0] = k
        return self.F - self.G @ Krow, self.Lw @ self.Lw.T + S_E * (self.G @ self.G.T)

    def per_step(self, k, P):
        return self.h * (self.q * P[0, 0] + self.r * (k * k * P[0, 0] + S_E))

    def episodic(self, k):
        Fk, drive = self._loop(k)
        P = self.L0 @ self.L0.T
        tot = 0.0
        for _ in range(T):
            tot += self.per_step(k, P)
            P = Fk @ P @ Fk.T + drive
        return tot / T

    def stationary(self, k):
        Fk, drive = self._loop(k)
        return self.per_step(k, solve_discrete_lyapunov(Fk, drive))

    def transient_grad(self, k):
        dk = 1e-3 * k
        f = lambda kk: self.episodic(kk) - self.stationary(kk)
        return (f(k + dk) - f(k - dk)) / (2 * dk)

    def stationary_grad(self, k):
        dk = 1e-3 * k
        return (self.stationary(k + dk) - self.stationary(k - dk)) / (2 * dk)

    def episodic_equilibria(self, lam):
        f = lambda k: kg.ghat(self.bench, k, lam) + self.transient_grad(k)
        ks = np.geomspace(0.5, 1.9 / self.h, 120)
        vs = np.array([f(k) for k in ks])
        out = []
        for i in range(len(ks) - 1):
            if np.isfinite(vs[i]) and np.isfinite(vs[i + 1]) and np.sign(vs[i]) != np.sign(vs[i + 1]):
                out.append(float(brentq(f, ks[i], ks[i + 1], xtol=1e-6)))
        return out


def main():
    kg.validate()
    table = {}
    for inst in gn.instance_names():
        e = Episodic(inst)
        print(f"\n{inst}  (h = {e.h})")
        print("  k      episodic/stationary cost   d(transient)/dk   true dJ/dk     ghat(k;0.9)   ghat(k;0.995)")
        rows = []
        for k in (1.0, 1.6, 2.5, 4.0, 6.0, 10.0, 20.0, 30.0, 40.0, 57.8, 80.0):
            if k >= 1.9 / e.h:
                continue
            ratio = e.episodic(k) / e.stationary(k)
            tg, sg = e.transient_grad(k), e.stationary_grad(k)
            g09, g995 = kg.ghat(e.bench, k, 0.9), kg.ghat(e.bench, k, 0.995)
            rows.append(dict(k=k, ratio=ratio, transient_grad=tg, stationary_grad=sg, ghat_0p9=g09, ghat_0p995=g995))
            print(f"  {k:5.1f}  {ratio:8.3f}                 {tg:+.3e}        {sg:+.3e}    {g09:+.3e}    {g995:+.3e}")
        eq = {}
        print("  lambda   k_AC (continuing)   episodic-objective equilibria")
        for lam in LAMBDAS:
            kac = kg.k_AC(e.bench, lam)
            eqs = e.episodic_equilibria(lam)
            eq[str(lam)] = dict(k_AC=kac, episodic=eqs)
            print(f"  {lam:<7}  {kac:8.2f}            {[round(x, 2) for x in eqs]}")
        table[inst] = dict(rows=rows, equilibria=eq)
    out = os.path.join(ROOT, "analysis", "episodic_objective_table.json")
    json.dump(table, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
