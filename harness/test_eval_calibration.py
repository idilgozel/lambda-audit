"""Prove the deployed evaluation estimator is unbiased against exact theory.

Nothing is trained here. The policy is PINNED to a known linear gain, the real
deployed evaluation path is run, and the measured cost rate is compared with
GLEBench.J_gain(k) computed analytically. If those agree, J_rate can be
compared with the frozen baselines in prereg_phase2.py. If they do not, it
cannot, and no Phase 2 number means anything.

The bias this guards against: GLEBench resets from the UNCONTROLLED stationary
covariance, Var(v) = 1.0, while the controlled stationary variance is 0.059 at
k = 1.649 and 0.00069 at k = 20.55. Averaging many short episodes folds a
settling transient into the cost. Over 1024 step episodes that inflates the
measurement by 1.088x, 1.270x and 2.415x at the three gains below.

That error is multiplicative, does not average out, and grows with the gain,
which lambda controls. So it is confounded with the effect under test and is
largest exactly where the predicted effect is largest. Hence the second
assertion here: the ratio must not trend with k. A flat ratio near 1 means the
estimator is measuring the environment. A ratio that climbs with k means it is
measuring the reset policy.

Requires scipy, for GLEBench. Runs locally only.

Run:  python3 harness/test_eval_calibration.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))

import gleam_native as native      # noqa: E402
import ppo_lambda as P             # noqa: E402

try:
    from gleam_bench import GLEBench  # noqa: E402
except ImportError:
    print("scipy or gleam_bench unavailable. This test runs locally only.")
    sys.exit(2)

INSTANCE = "GLEAM_g1_h0.05"
GAINS = [1.649, 5.99, 20.5469]
TOLERANCE = 0.02          # 2 percent, per the specification
SPREAD_TOLERANCE = 0.02   # ratios must agree with each other this closely
EVAL_STEPS = 200_000
EVAL_BURNIN = 1000


class PinnedAgent:
    """Quacks like the Agent that evaluate_continuous expects.

    Emits exactly u = -k v, deterministically, so the measured cost rate has a
    known analytic value. hidden is present because the evaluation path
    allocates a recurrent state whatever the architecture.
    """

    def __init__(self, k, hidden=128):
        self.k = float(k)
        self.hidden = hidden
        self.continuous = True

    def act(self, x, h, done, action=None, deterministic=False):
        # x is the stacked observation, one frame of one scalar for GLEAM
        v = x.reshape(-1)[-1]
        u = (-self.k * v).reshape(1, 1)
        zero = torch.zeros(1)
        return u, zero, zero, zero, h

    def value(self, x, h, done):
        return torch.zeros(1)


def analytic_J(instance, k):
    p = native.instance_params(instance)
    env = GLEBench(cs=(p["c"],), gammas=(p["gamma_bath"],), theta=p["theta"],
                   h=p["h"], q=p["q"], r=p["r"], rho=p["rho"], n_envs=1, seed=0)
    return float(env.J_gain(k))


def main():
    print("=" * 78)
    print(" Deployed evaluation calibration against exact theory")
    print("=" * 78)
    print(f"\n  instance {INSTANCE}, single trajectory, "
          f"{EVAL_STEPS:,} steps after a {EVAL_BURNIN} step burn-in")
    print(f"  policy pinned to u = -k v, no exploration noise\n")
    print(f"  {'k':>10} {'measured J':>12} {'analytic J':>12} {'ratio':>8} "
          f"{'gain read back':>15}")

    ratios, failures = [], []
    for k in GAINS:
        agent = PinnedAgent(k)
        mean_ret, _, _, ctrl = P.evaluate_continuous(
            agent, INSTANCE, stack=1, seed=1, gamma=0.9975031223974601,
            device=torch.device("cpu"), collect_gap=False, lam=None,
            eval_steps=EVAL_STEPS, eval_burnin=EVAL_BURNIN)
        measured = ctrl["J_rate"]
        exact = analytic_J(INSTANCE, k)
        ratio = measured / exact
        ratios.append(ratio)
        read_back = ctrl["final_gain"]
        print(f"  {k:10.4f} {measured:12.5f} {exact:12.5f} {ratio:8.4f} "
              f"{read_back:15.4f}")
        if abs(ratio - 1.0) > TOLERANCE:
            failures.append(
                f"k={k}: measured {measured:.5f} vs analytic {exact:.5f}, "
                f"ratio {ratio:.4f}, outside {int(100 * TOLERANCE)}%")
        # the gain regression must also recover the pinned gain
        if read_back is None or abs(read_back - k) / k > TOLERANCE:
            failures.append(
                f"k={k}: final_gain read back as {read_back}, expected {k}")

    spread = max(ratios) - min(ratios)
    print(f"\n  ratio spread across gains: {spread:.5f} "
          f"(must be <= {SPREAD_TOLERANCE})")
    print("  a ratio that climbs with k is the pathology being removed, "
          "since\n  lambda moves the gain and would confound the measurement "
          "with the effect")
    if spread > SPREAD_TOLERANCE:
        failures.append(f"ratio trends with k, spread {spread:.5f}")

    # For the record: what the old episodic estimator would have reported.
    print("\n  for the record, the episodic estimator this replaced:")
    p = native.instance_params(INSTANCE)
    env = GLEBench(cs=(p["c"],), gammas=(p["gamma_bath"],), theta=p["theta"],
                   h=p["h"], q=p["q"], r=p["r"], rho=p["rho"], n_envs=1, seed=0)
    from scipy.linalg import solve_discrete_lyapunov
    for k in GAINS:
        Krow = np.zeros((1, 2))
        Krow[0, 0] = k
        Fk = env.F - env.G @ Krow
        S = env.Sig0.copy()
        acc = []
        for _ in range(p["episode_len"]):
            acc.append(S[0, 0])
            S = Fk @ S @ Fk.T + env.Sd
        qt = 1.0 + k * k
        print(f"    k={k:8.4f}  would have read {qt * np.mean(acc):.5f} "
              f"against a true {analytic_J(INSTANCE, k):.5f}  "
              f"({qt * np.mean(acc) / analytic_J(INSTANCE, k):.3f}x)")

    if failures:
        print("\n  RESULT: FAILED. J_rate cannot be compared to the frozen "
              "baselines.")
        for f in failures:
            print(f"    {f}")
        return 1
    print(f"\n  RESULT: PASS. All {len(GAINS)} gains within "
          f"{int(100 * TOLERANCE)}% of exact theory, ratio flat in k.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
