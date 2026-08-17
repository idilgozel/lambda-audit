"""Tier 1b discriminator: pure linear-gain policy on the GLEAM instance.

Policy is a SINGLE scalar k, u = -k*v + eta, eta ~ N(0, s_e) with s_e PINNED,
exactly matching the theory's assumption. No bias, no hidden layer, no learned
sigma. The policy therefore cannot leave the gain manifold.

Everything else mirrors the E6 harness: GAE(lambda) advantages, a quadratic
critic in (1, v, v^2) fitted by the same bootstrapped targets, PPO-style
clipped surrogate, Adam, annealed lr, deployed evaluation on a single long
episode with burn-in discarded and policy mean (no exploration noise).

Numpy only. Instance g1_h0.05 constants come from gleam_bench.
"""
import numpy as np
from gleam_bench import GLEBench

E = GLEBench(cs=(0.3,), gammas=(1.0,), h=0.05, n_envs=1)
F, G, Sd = E.F, E.G, E.Sd
LW = np.linalg.cholesky(Sd + 1e-14 * np.eye(2))
L0 = np.linalg.cholesky(E.Sig0 + 1e-14 * np.eye(2))
H, Q, R = E.h, E.q, E.r
BETA = float(np.exp(-E.rho * E.h))
S_E = 0.09
SIG = np.sqrt(S_E)
J_ML = E.best_memoryless()[1]
G0 = G[:, 0]


def rollout(k, n_envs, T, rng, state=None):
    """Collect T steps across n_envs. Returns v, u, eta, cost, next state."""
    s = (L0 @ rng.standard_normal((2, n_envs))).T if state is None else state
    V = np.empty((T, n_envs)); U = np.empty_like(V)
    ETA = np.empty_like(V); C = np.empty_like(V)
    for t in range(T):
        v = s[:, 0]
        eta = rng.standard_normal(n_envs) * SIG
        u = -k * v + eta
        C[t] = H * (Q * v * v + R * u * u)
        V[t], U[t], ETA[t] = v, u, eta
        s = s @ F.T + np.outer(u, G0) + (LW @ rng.standard_normal((2, n_envs))).T
    return V, U, ETA, C, s


def fit_critic(V, C, lam, theta, lr_c, iters=8):
    """Quadratic critic on (1, v, v^2), fitted to bootstrapped lambda-returns.
    Same aliased observation as the agent: only v is visible."""
    T, N = V.shape
    for _ in range(iters):
        Phi = np.stack([np.ones_like(V), V, V * V], -1)      # (T,N,3)
        val = Phi @ theta
        delta = C[:-1] + BETA * val[1:] - val[:-1]
        acc = np.zeros(N); Gl = np.zeros_like(delta)
        for t in range(delta.shape[0] - 1, -1, -1):
            acc = delta[t] + BETA * lam * acc
            Gl[t] = acc
        target = val[:-1] + Gl
        err = val[:-1] - target
        grad = (Phi[:-1] * err[..., None]).reshape(-1, 3).mean(0)
        theta = theta - lr_c * grad
    return theta


def train(lam, seed, total_steps=6_000_000, n_envs=16, T=128,
          lr_a=3e-4, lr_c=1e-2, clip=0.2, epochs=4, eval_every=20):
    rng = np.random.default_rng(seed)
    k = 1.0
    theta = np.zeros(3)
    state = None
    n_updates = total_steps // (n_envs * T)
    curve = []
    for up in range(n_updates):
        frac = 1.0 - up / n_updates
        V, U, ETA, C, state = rollout(k, n_envs, T, rng, state)
        theta = fit_critic(V, C, lam, theta, lr_c * frac)
        Phi = np.stack([np.ones_like(V), V, V * V], -1)
        val = Phi @ theta
        delta = C[:-1] + BETA * val[1:] - val[:-1]
        acc = np.zeros(n_envs); A = np.zeros_like(delta)
        for t in range(delta.shape[0] - 1, -1, -1):
            acc = delta[t] + BETA * lam * acc
            A[t] = acc
        A = -(A - A.mean()) / (A.std() + 1e-8)   # cost -> ascend negative
        v, eta = V[:-1].ravel(), ETA[:-1].ravel()
        adv = A.ravel()
        logp_old = -0.5 * (eta / SIG) ** 2
        k_new = k
        for _ in range(epochs):
            mu = -k_new * v
            logp = -0.5 * ((v * (-k_new) + eta - mu) / SIG) ** 2  # = logp_old shape
            # d logp / dk for u fixed: u = -k_old v + eta ; logp = -(u + k_new v)^2/(2 s_e)
            uu = -k * v + eta
            logp = -0.5 * (uu + k_new * v) ** 2 / S_E
            ratio = np.exp(logp - logp_old)
            g1 = ratio * adv
            g2 = np.clip(ratio, 1 - clip, 1 + clip) * adv
            use = (g1 < g2)                       # min for maximisation
            dlogp = -(uu + k_new * v) * v / S_E
            grad = np.where(use, ratio * dlogp * adv, 0.0).mean()
            k_new = k_new + lr_a * frac * grad * 100.0
            k_new = float(np.clip(k_new, 0.01, 39.0))
        k = k_new
        if (up + 1) % eval_every == 0:
            curve.append((up + 1, (up + 1) * n_envs * T, k, deploy_cost(k)))
    return k, curve


def deploy_cost(k, steps=60_000, burn=1000, seed=99):
    """Policy MEAN, single long episode, burn-in discarded."""
    if k <= 0 or np.max(np.abs(np.linalg.eigvals(F - G @ np.array([[k, 0.0]])))) >= 1:
        return np.inf
    rng = np.random.default_rng(seed)
    s = (L0 @ rng.standard_normal((2, 1))).T
    tot = 0.0
    for t in range(burn + steps):
        v = s[:, 0]; u = -k * v
        c = H * (Q * v * v + R * u * u)
        if t >= burn:
            tot += c[0]
        s = s @ F.T + np.outer(u, G0) + (LW @ rng.standard_normal((2, 1))).T
    return tot / steps / H


if __name__ == "__main__":
    import json, sys
    lam = float(sys.argv[1]); seed = int(sys.argv[2])
    k, curve = train(lam, seed)
    print(json.dumps({"lam": lam, "seed": seed, "final_gain": k,
                      "curve": curve}))
