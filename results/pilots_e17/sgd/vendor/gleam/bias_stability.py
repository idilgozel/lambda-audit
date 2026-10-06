"""A-5 Tier 2: stability of the ACTION-BIAS direction under the aliased critic.

The policy is u = -k v + b + eta. At b = 0 the expected update for b vanishes by
parity, so b = 0 is a fixed point of the averaged b-dynamics. This module
computes the sign of d E[ghat_b] / db there, exactly, and validates it by Monte
Carlo.

DERIVATION (see bias-direction-stability.md for the full write-up).

Closed loop A_k = F - k G0 e1'. Turning on b shifts the stationary mean and
leaves the covariance alone: m = M b with M = (I - A_k)^-1 G0, so

    v = vtil + mu_v b,   mu_v = e1' (I - A_k)^-1 G0
    u = util + (1 - k mu_v) b,   util = -k vtil + eta

By the symmetry V_{-b}(v) = V_b(-v), the critic's theta0 and theta2 are even in
b and theta1 is odd, so to first order the ONLY critic change is a term linear
in vtil with coefficient kappa = 2 theta2 mu_v + dtheta1/db.

The O(b) part of the TD error is then

    r_t = chat_t + beta kappa vtil_{t+1} - kappa vtil_t,
    chat_t = 2h [ Q mu_v vtil_t + R (1 - k mu_v) util_t ]

and the O(b) part of the GAE advantage is H_t = sum_l (beta lam)^l r_{t+l}.

kappa is pinned WITHOUT solving the critic, by the LSTD(lambda) fixed-point
condition projected on the odd feature: E[vtil_t H_t] = 0, which gives

    kappa = 2 h alpha X / (X - beta X1),   alpha = Q mu_v - k R (1 - k mu_v)
    X  = sum_l (beta lam)^l x_l,   X1 = sum_l (beta lam)^l x_{l+1},
    x_l = e1' A_k^l Sig e1

The gradient derivative is E[H_t eta_t] / s_e, and since eta_t is independent of
the state at t and enters the future through G0 only,

    E[vtil_{t+m} eta_t] = s_e e1' A_k^{m-1} G0  (m >= 1),  0 at m = 0

so the whole sum collapses to resolvents:

    S(lam) = 2hR(1 - k mu_v) + 2h alpha Phi + kappa Phi (1 - lam)/lam
    Phi = beta lam e1' (I - beta lam A_k)^-1 G0

and d ghat_b / db = -S. So b = 0 is STABLE iff S > 0.

The third term is the aliased critic's contribution. It carries the factor
(1 - lam)/lam, which is exactly the bootstrap weight: it vanishes at lam = 1
and grows without bound as lam falls. That factor is the mechanism.
"""
import numpy as np
import theorem2_learning as T

E1 = np.array([1.0, 0.0])
I2 = np.eye(2)


def bias_stability(g, k, lam, s_e=T.S_E, beta=None):
    """Exact S(lam, k). b = 0 is stable in the bias direction iff S > 0."""
    beta = T.BETA if beta is None else beta
    F, G, Sd, Fk, Sig = T.closed_loop(g, k, s_e)
    G0 = G[:, 0]
    bl = beta * lam

    mu_v = float(E1 @ np.linalg.solve(I2 - Fk, G0))
    Phi = float(bl * (E1 @ np.linalg.solve(I2 - bl * Fk, G0)))
    Xv = np.linalg.solve(I2 - bl * Fk, Sig @ E1)
    X = float(E1 @ Xv)
    X1 = float(E1 @ (Fk @ Xv))

    alpha = T.Q * mu_v - k * T.R * (1.0 - k * mu_v)
    kappa = 2.0 * T.H * alpha * X / (X - beta * X1)

    critic_term = kappa * Phi * (1.0 - lam) / lam if lam > 0 else np.inf
    S = 2.0 * T.H * T.R * (1.0 - k * mu_v) + 2.0 * T.H * alpha * Phi + critic_term
    return S, dict(mu_v=mu_v, Phi=Phi, alpha=alpha, kappa=kappa,
                   direct=2.0 * T.H * T.R * (1.0 - k * mu_v) + 2.0 * T.H * alpha * Phi,
                   critic=critic_term)


# ---------------------------------------------------------------------------
# Monte Carlo validation. Nothing above is trusted until this agrees.
# ---------------------------------------------------------------------------

def _simulate(g, k, b, s_e, T_steps, n_env, seed):
    """Roll the closed loop with u = -k v + b + eta. Common random numbers."""
    F, G, Sd, Fk, Sig = T.closed_loop(g, k, s_e)
    G0 = G[:, 0]
    LW = np.linalg.cholesky(Sd + 1e-15 * I2)
    L0 = np.linalg.cholesky(Sig + 1e-15 * I2)
    rng = np.random.default_rng(seed)
    s = (L0 @ rng.standard_normal((2, n_env))).T
    V = np.empty((T_steps, n_env))
    U = np.empty_like(V)
    ETA = np.empty_like(V)
    C = np.empty_like(V)
    for t in range(T_steps):
        v = s[:, 0]
        eta = rng.standard_normal(n_env) * np.sqrt(s_e)
        u = -k * v + b + eta
        C[t] = T.H * (T.Q * v * v + T.R * u * u)
        V[t], U[t], ETA[t] = v, u, eta
        # s' = F s + G0 u + w, with the FULL u. Writing this as Fk s + G0 b + w
        # and forgetting the G0 eta term is the bug that silently removes the
        # entire mechanism: exploration noise then never enters the state, so
        # E[v_{t+m} eta_t] = 0 and the aliased critic has nothing to bite on.
        s = s @ F.T + np.outer(u, G0) \
            + (LW @ rng.standard_normal((2, n_env))).T
    return V, U, ETA, C


def _lstd_lambda(V, C, lam, beta):
    """LSTD(lambda) on features (1, v, v^2) from a trajectory, via traces."""
    Tn, N = V.shape
    Psi = np.stack([np.ones_like(V), V, V * V], -1)          # (T,N,3)
    bl = beta * lam
    z = np.zeros((N, 3))
    A = np.zeros((3, 3))
    c = np.zeros(3)
    n = 0
    for t in range(Tn - 1):
        z = bl * z + Psi[t]
        d = Psi[t] - beta * Psi[t + 1]                        # (N,3)
        A += z.T @ d
        c += z.T @ C[t]
        n += N
    return np.linalg.solve(A / n, c / n)


def _grad_b(V, U, ETA, C, theta, lam, beta, s_e):
    """MC estimate of ghat_b = -E[Ahat_t eta_t]/s_e with the given critic."""
    Psi = np.stack([np.ones_like(V), V, V * V], -1)
    val = Psi @ theta
    delta = C[:-1] + beta * val[1:] - val[:-1]
    acc = np.zeros(V.shape[1])
    A = np.zeros_like(delta)
    for t in range(delta.shape[0] - 1, -1, -1):
        acc = delta[t] + beta * lam * acc
        A[t] = acc
    return -float(np.mean(A * ETA[:-1])) / s_e


def mc_bias_slope(g, k, lam, eps=0.02, s_e=T.S_E, beta=None,
                  T_steps=4000, n_env=4000, seed=7):
    """Central difference of E[ghat_b] in b, at b = 0, common random numbers."""
    beta = T.BETA if beta is None else beta
    out = {}
    for sign in (+1, -1):
        b = sign * eps
        V, U, ETA, C = _simulate(g, k, b, s_e, T_steps, n_env, seed)
        theta = _lstd_lambda(V, C, lam, beta)
        out[sign] = (_grad_b(V, U, ETA, C, theta, lam, beta, s_e), theta)
    slope = (out[+1][0] - out[-1][0]) / (2 * eps)
    return slope, out


def predict_all(s_e=0.09):
    """Regenerate the frozen table in prereg_e7_bias.py. Prints, returns rows."""
    import importlib
    import lstd_lambda as L
    from scipy.optimize import brentq
    inst = [("GLEAM_g1_h0.02", 1.0, 0.02, 0.996552),
            ("GLEAM_g1_h0.05", 1.0, 0.05, 0.985382),
            ("GLEAM_g2_h0.05", 2.0, 0.05, 0.957977),
            ("GLEAM_g4_h0.05", 4.0, 0.05, 0.873138)]
    rows = []
    for name, g, h, lam_half in inst:
        T.H = h
        T.BETA = float(np.exp(-T.RHO * h))
        T._DM_CACHE.clear()
        importlib.reload(L)

        def keq(lam):
            r = L.k_eq_lambda(g, s_e, lam)
            return max(r) if r else float("nan")

        def f(lam):
            return bias_stability(g, keq(lam), lam, s_e, beta=T.BETA)[0]

        grid = np.linspace(0.70, 0.9995, 60)
        vals = []
        for lam in grid:
            try:
                vals.append(f(lam))
            except Exception:
                vals.append(np.nan)
        root, kk = float("nan"), float("nan")
        for i in range(len(grid) - 1):
            a, b = vals[i], vals[i + 1]
            if np.isfinite(a) and np.isfinite(b) and np.sign(a) != np.sign(b):
                root = brentq(f, grid[i], grid[i + 1], xtol=1e-6)
                kk = keq(root)
                break
        rows.append((name, g * h, lam_half, root, kk))
        ls = f"{root:.6f}" if np.isfinite(root) else "none"
        ks = f"{kk:.3f}" if np.isfinite(kk) else "-"
        gap = f"{lam_half - root:.6f}" if np.isfinite(root) else "-"
        print(f"{name:18}{g*h:7.2f}{lam_half:11.6f}{ls:>11}{ks:>11}{gap:>11}")
    return rows


if __name__ == "__main__":
    import sys
    if "--predict" in sys.argv:
        print(f"{'instance':18}{'g*h':>7}{'lam_half':>11}{'lam*':>11}"
              f"{'k_AC(lam*)':>11}{'gap':>11}")
        predict_all()
        raise SystemExit(0)
    g = 1.0
    s_e = T.S_E
    print(f"instance g1_h{T.H}, s_e = {s_e}, beta = {T.BETA:.6f}")
    print()
    print("VALIDATION: exact -S(lam) against a Monte Carlo slope of E[ghat_b]")
    print(f"{'lam':>6} {'k':>7} {'exact -S':>12} {'MC slope':>12} {'ratio':>8}")
    for lam, k in [(0.9, 5.0), (0.9, 20.5), (0.99, 2.2), (0.95, 10.0), (1.0, 5.0)]:
        S, _ = bias_stability(g, k, lam, s_e)
        mc, _ = mc_bias_slope(g, k, lam, s_e=s_e)
        print(f"{lam:6.2f} {k:7.2f} {-S:12.6f} {mc:12.6f} {mc / -S:8.3f}")
