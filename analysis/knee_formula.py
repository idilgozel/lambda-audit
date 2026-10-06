"""The knee, in closed form to first order: where lambda_half sits and why.

Written 2026-10-05. Pure deterministic computation on the exact objects of
vendor/gleam/kernel_general.py (which reproduces the frozen prereg_e10.py
tables); no training, no seeds. Run from the repository root:

    python3 analysis/knee_formula.py            # validate, then tables
    python3 analysis/knee_formula.py --slices   # + one-parameter slices (slow)

1. A COMPACT FORM OF THE EXACT GAE(lambda) GRADIENT
---------------------------------------------------
For u_t = -k v_t + eta_t, eta ~ N(0, s_e), and the observation-restricted
LSTD(lambda) critic on (1, v, v^2), the expected GAE(lambda) policy gradient is

    ghat(k; lam) = 2h [ r k Var(v)  -  (q + r k^2) Phi_k(beta lam) Lambda_k(lam) / s_e ]      (I)

with
    Phi_k(mu)      = sum_{j>=1} mu^j x_j y_j                 truncated kick-response sensitivity
    Lambda_k(lam)  = 1 / (1 - (1 - lam) Ntilde_k(lam))        critic aliasing amplification
    Ntilde_k(lam)  = sum_{j>=0} (beta lam)^j rho_j^2,  rho_j = x_j / x_0
    x_j = E[v_t v_{t+j}],  y_j = E[eta_t v_{t+j}] = s_e e1' F_k^{j-1} G  (j >= 1).

Derivation (four lines). The lambda-advantage is
    A_t = sum_j (bl)^j c_{t+j} - V(v_t) + beta(1-lam) sum_{j>=1} (bl)^{j-1} V(v_{t+j}),  bl = beta lam.
With score s_t = -v_t eta_t / s_e: E[s_t V(v_t)] = 0 (eta_t independent of v_t);
E[s_t c_t] = 2 h r k Var(v); for j >= 1, by Isserlis, E[s_t c_{t+j}] = -(2h(q + r k^2)/s_e) x_j y_j
and E[s_t V(v_{t+j})] = -(2 theta_2/s_e) x_j y_j. Summing,
    ghat = 2 h r k Var(v) - (2 Phi(bl)/s_e) [ h (q + r k^2) + theta_2 (1 - lam)/lam ].
The LSTD(lambda) fixed point is theta_2 = h (q + r k^2) lam Ntilde / (1 - (1-lam) Ntilde)
(the denominator sum_j (bl)^j (x_j^2 - beta x_{j+1}^2) telescopes to (x_0^2 - (1-lam) N)/lam),
so the bracket equals h (q + r k^2) / (1 - (1-lam) Ntilde) = h (q + r k^2) Lambda.  QED.

Everything the critic does to the gradient is the scalar Lambda_k(lam) >= 1; the
rest of the lambda dependence is the honest truncation Phi_k(beta lam) <= Phi_k(beta).

2. DIRECTION OF THE DISPLACEMENT
--------------------------------
Write B_k(lam) = Phi_k(beta lam) Lambda_k(lam) / s_e for the perceived benefit of
control per unit (q + r k^2). To first order in (1 - lam),
    B_k(lam) = B_k(1) [ 1 + (1 - lam) C_k + O((1-lam)^2) ],   C_k = Ntilde_k(1) - T_k,
    T_k = sum_j j beta^j x_j y_j / sum_j beta^j x_j y_j       (mean lag of the kick response).
Ntilde_k(1) is the discounted squared-autocorrelation time of the OBSERVATION;
T_k is the lag over which a control kick is still felt. When the hidden state
outlives the control response -- Ntilde_k(1) > T_k, true on every instance
below -- the aliased critic INFLATES the perceived benefit of control, and the
equilibrium gain is displaced UPWARD (over-control) for every lam < 1. The
paper's observed direction is therefore a theorem about timescales.

3. THE KNEE
-----------
At the gain-midpoint knee, k_AC(lam_K) = k_m := sqrt(k_0 k_1) by definition
(k_0 = TD(0) equilibrium, k_1 = lam=1 equilibrium), so lam_K is the scalar root
of  B_{k_m}(lam) = L(k_m),  L(k) := r k Var_k(v) / (q + r k^2).  No k_AC(lam)
curve is needed: two gains and one 1-D root.  Expanding B to first order,

    1 - lam_half  ~=  (R_m - 1) / C_m,      R_m = L(k_m) / B_{k_m}(1),                    (II)
    H_half = 1/(1 - beta lam_half)  ~=  1 / [ (R_m - 1)/C_m + (1 - beta) ].

R_m - 1 is the normalised TRUE policy gradient at the midpoint gain: how hard
the lam = 1 objective pulls k_m back toward k_1. It is small (0.06-0.5 on every
instance tested) because the memoryless cost surface J(k) is flat. C_m is the
critic's aliasing budget in steps, of order the hidden correlation time. So

    the knee horizon is the hidden correlation time DIVIDED BY the flatness of
    the memoryless cost landscape -- which is why H_half runs 1.5-6x the
    correlation time rather than tracking it, why lambda = 0.95 is unsafe for a
    20-step hidden coordinate, and why lambda_half is not ordered by kernel
    correlation time across kernel shapes (R_m is not).

Validation below: (II) against the exact gain-midpoint knee on the four paper
instances, five Prony kernels (state dimension 2-4) and 31 single-mode slices
in gamma, c, h, rho, s_e, Theta.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))
import kernel_general as kg  # noqa: E402
from gleam_bench import GLEBench  # noqa: E402

S_E = kg.S_E


# ---------------------------------------------------------------------------
# the pieces of identity (I)
# ---------------------------------------------------------------------------

def pieces(bench, k, lam, s_e=S_E):
    L = kg.loop(bench, k, s_e)
    if L is None:
        return None
    Fk, Sig, e0, g0, beta = L
    n = len(e0)
    bl = beta * lam
    m2 = Sig[0, 0]
    K2 = np.kron(Fk, Fk)
    ee = np.kron(e0, e0)
    v2 = np.kron(Sig @ e0, Sig @ e0)
    Ntilde = ee @ np.linalg.solve(np.eye(n * n) - bl * K2, v2) / m2 ** 2
    w = np.kron(Fk @ Sig @ e0, g0)
    Phi = bl * s_e * ee @ np.linalg.solve(np.eye(n * n) - bl * K2, w)
    if lam > 0:
        Lam = 1.0 / (1.0 - (1.0 - lam) * Ntilde)
        B = Phi * Lam / s_e
    else:                                   # lam -> 0 limit: TD(0)
        x1 = (Fk @ Sig)[0, 0]
        Lam = np.inf
        B = beta * x1 * g0[0] / (1 - beta * (x1 / m2) ** 2)
    qt = bench.q + bench.r * k * k
    Lk = bench.r * k * m2 / qt
    return dict(m2=m2, Ntilde=Ntilde, Lam=Lam, Phi=Phi, B=B, L=Lk,
                ghat=2 * bench.h * qt * (Lk - B), beta=beta)


def mean_lag(bench, k, s_e=S_E):
    """T_k = sum_j j beta^j x_j y_j / sum_j beta^j x_j y_j."""
    Fk, Sig, e0, g0, beta = kg.loop(bench, k, s_e)
    n = len(e0)
    K2 = np.kron(Fk, Fk)
    ee = np.kron(e0, e0)
    w = np.kron(Fk @ Sig @ e0, g0)
    R = np.linalg.solve(np.eye(n * n) - beta * K2, np.eye(n * n))
    Phi = beta * s_e * ee @ R @ w
    # sum_{j>=1} j mu^j K2^{j-1} = mu (I - mu K2)^-1 + mu^2 (I - mu K2)^-2 K2
    num = beta * s_e * ee @ (R + beta * R @ R @ K2) @ w
    return num / Phi


def validate_identity(tol=1e-9):
    worst = 0.0
    for cs, gs in [((0.3,), (1.0,)), ((0.15, 0.15), (0.4, 2.5)), ((0.1, 0.1, 0.1), (0.3, 1.0, 3.0))]:
        b = GLEBench(cs=cs, gammas=gs, h=0.05)
        for k in (0.5, 1.5, 3.0, 6.0, 12.0, 24.0):
            for lam in (0.3, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0):
                ref = kg.ghat(b, k, lam)
                got = pieces(b, k, lam)["ghat"]
                worst = max(worst, abs(got - ref) / max(abs(ref), 1e-12))
    return worst


# ---------------------------------------------------------------------------
# knees
# ---------------------------------------------------------------------------

def knee_exact(bench, s_e=S_E):
    """Continuous cost-midpoint knee lamE (E halfway between the TD(0) plateau
    and the lam=1 floor) and gain-midpoint knee lamK (k_AC = sqrt(k0 k1))."""
    _, J_ml = bench.best_memoryless()
    k0 = bench.ac_equilibrium(s_e)[0]
    k1 = kg.k_AC(bench, 1.0, s_e)
    E0, E1 = bench.J_gain(k0) / J_ml, bench.J_gain(k1) / J_ml
    midE, km = 0.5 * (E0 + E1), math.sqrt(k0 * k1)
    fE = lambda lam: bench.J_gain(kg.k_AC(bench, lam, s_e)) / J_ml - midE
    fK = lambda lam: kg.k_AC(bench, lam, s_e) - km
    lamE = brentq(fE, 0.2, 0.99999, xtol=1e-7)
    lamK = brentq(fK, 0.2, 0.99999, xtol=1e-7)
    return dict(k0=k0, k1=k1, km=km, E0=E0, E1=E1, lamE=lamE, lamK=lamK)


def knee_formula(bench, s_e=S_E, k0=None, k1=None):
    """lam_K by the fixed-gain root (exact reformulation) and by (II)."""
    k0 = bench.ac_equilibrium(s_e)[0] if k0 is None else k0
    k1 = kg.k_AC(bench, 1.0, s_e) if k1 is None else k1
    km = math.sqrt(k0 * k1)
    f = lambda lam: pieces(bench, km, lam, s_e)["B"] - pieces(bench, km, lam, s_e)["L"]
    lam_fixed = brentq(f, 1e-4, 1.0 - 1e-9, xtol=1e-9)
    p1 = pieces(bench, km, 1.0, s_e)
    R = p1["L"] / p1["B"]
    N1 = p1["Ntilde"]
    T = mean_lag(bench, km, s_e)
    C = N1 - T
    lam_asym = 1.0 - (R - 1.0) / C
    beta = p1["beta"]
    H_asym = 1.0 / ((R - 1.0) / C + (1.0 - beta))
    return dict(k0=k0, k1=k1, km=km, lam_fixed=lam_fixed, R=R, N1=N1, T=T, C=C,
                lam_asym=lam_asym, H_asym=H_asym, beta=beta)


INSTANCES = [("GLEAM_g1_h0.02", (0.3,), (1.0,), 0.02), ("GLEAM_g1_h0.05", (0.3,), (1.0,), 0.05),
             ("GLEAM_g2_h0.05", (0.3,), (2.0,), 0.05), ("GLEAM_g4_h0.05", (0.3,), (4.0,), 0.05)]
KERNELS = [("2-mode balanced", (0.15, 0.15), (0.4, 2.5), 0.05),
           ("2-mode slow-tail", (0.05, 0.25), (0.2, 2.0), 0.05),
           ("3-mode broad", (0.1, 0.1, 0.1), (0.3, 1.0, 3.0), 0.05)]
BASE = dict(g=1.0, c=0.3, h=0.05, rho=0.05, s_e=0.09, theta=1.0)
SLICES = {"gamma": ("g", (0.2, 0.4, 2.0, 4.0, 8.0)), "c": ("c", (0.03, 0.1, 1.0, 3.0)),
          "h": ("h", (0.01, 0.02, 0.1, 0.2)), "rho": ("rho", (0.005, 0.02, 0.2, 0.5)),
          "s_e": ("s_e", (0.01, 0.03, 0.3, 1.0)), "theta": ("theta", (0.1, 0.3, 3.0, 10.0))}


def row(name, bench, s_e=S_E, tauK=None):
    X = knee_exact(bench, s_e)
    F = knee_formula(bench, s_e, X["k0"], X["k1"])
    beta = F["beta"]
    H_E = 1.0 / (1.0 - beta * X["lamE"])
    tau = tauK if tauK is not None else kg.corr_time(bench) / bench.h
    print(f"{name:>18} {tau:6.1f} | {X['lamE']:8.5f} {X['lamK']:8.5f} {F['lam_fixed']:8.5f} {F['lam_asym']:9.5f} "
          f"| {F['R'] - 1:7.4f} {F['N1']:7.2f} {F['T']:6.2f} {F['C']:7.2f} | {H_E:7.1f} {F['H_asym']:7.1f} {H_E / tau:5.2f}")
    return dict(name=name, tau=tau, **X, **{k: v for k, v in F.items() if k not in X})


def header():
    print(f"{'instance':>18} {'tau_K':>6} | {'lamE':>8} {'lamK':>8} {'lam_fix':>8} {'lam_asym':>9} "
          f"| {'R_m-1':>7} {'N1':>7} {'T':>6} {'C':>7} | {'H_E':>7} {'H_asym':>7} {'H/tau':>5}")


def main(argv):
    worst = validate_identity()
    print(f"identity (I) vs kernel_general.ghat, 3 kernels x 6 gains x 8 lambdas: "
          f"max relative discrepancy {worst:.1e}  {'ok' if worst < 1e-9 else 'MISMATCH'}")
    if worst >= 1e-9:
        raise SystemExit("identity does not reproduce the exact gradient; nothing else is printed.")
    print("\nlamE: cost-midpoint knee (the paper's lambda_half definition, continuous).")
    print("lamK: gain-midpoint knee. lam_fix: exact fixed-gain root of B_{k_m} = L(k_m) (= lamK).")
    print("lam_asym: first-order formula (II). Columns R_m-1, N1, T, C evaluated at k_m.\n")
    out = []
    header()
    for name, cs, gs, h in INSTANCES:
        out.append(row(name, GLEBench(cs=cs, gammas=gs, h=h)))
    print("-- Prony kernels, state dimension 3-4")
    for name, cs, gs, h in KERNELS:
        out.append(row(name, GLEBench(cs=cs, gammas=gs, h=h)))
    if "--slices" in argv:
        print("-- one-parameter slices around GLEAM_g1_h0.05 (tau column = 1/(gamma h))")
        for sl, (key, vals) in SLICES.items():
            for v in vals:
                p = dict(BASE); p[key] = v
                b = GLEBench(cs=(p["c"],), gammas=(p["g"],), h=p["h"], rho=p["rho"], theta=p["theta"])
                r = row(f"{sl}={v}", b, s_e=p["s_e"], tauK=1.0 / (p["g"] * p["h"]))
                r.update(p)
                out.append(r)
    errs = [abs(r["lam_asym"] - r["lamK"]) for r in out]
    small = [abs(r["lam_asym"] - r["lamK"]) for r in out if 1 - r["lamK"] <= 0.05]
    print(f"\n|lam_asym - lamK|: max {max(errs):.4f} over {len(out)} instances; "
          f"max {max(small):.4f} over the {len(small)} with 1 - lamK <= 0.05.")
    print("Direction: C_m = N1 - T > 0 on every instance -> over-control for every lam < 1:",
          all(r["C"] > 0 for r in out))
    path = os.path.join(ROOT, "analysis", "knee_formula_table.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"table written to {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
