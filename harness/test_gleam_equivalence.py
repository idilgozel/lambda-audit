"""Prove gleam_native.py is a bitwise drop-in for GLEBench.

Same protocol that licensed envs_native.py against popgym. Identical seeds,
identical action sequences, every observation and every cost compared bit for
bit, not approximately. This is what allows the cluster to run without scipy.

Both arms are covered:

  coupled     compared against GLEBench.step directly.
  decoupled   compared against a reference built from GLEBench's own matrices
              with the outer(u, G[:,0]) term removed and nothing else changed.
              That reference mirrors GLEBench.step expression for expression,
              so bitwise equality is a meaningful claim rather than an
              artefact of rewriting the arithmetic.

Requires scipy, since GLEBench needs it. Runs locally only.

Run:  python3 harness/test_gleam_equivalence.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))

import gleam_native as native  # noqa: E402

try:
    from gleam_bench import GLEBench  # noqa: E402
except ImportError:
    print("scipy or gleam_bench unavailable. This test runs locally only.")
    sys.exit(2)

STEPS_PER_CASE = 3000          # 4 instances x 2 arms x 3000 = 24,000 steps
ACTION_SCALE = 5.0


def bits(x):
    """Bitwise-comparable representation of a scalar or array."""
    a = np.asarray(x, dtype=np.float64)
    return (a.dtype.str, a.shape, a.tobytes())


class DecoupledReference:
    """GLEBench with the action term dropped from the state update only.

    Deliberately written as a copy of GLEBench.step with one term deleted, so
    the floating point operations occur in the same order. Anything else would
    make a bitwise assertion meaningless.
    """

    def __init__(self, core):
        self.c = core

    def reset(self):
        c = self.c
        c.s = (c.L0 @ c.rng.standard_normal((c.nx, c.n_envs))).T
        return c.s[:, 0:1].copy()

    def step(self, u):
        c = self.c
        u = np.asarray(u, float).reshape(-1)
        cost = c.h * (c.q * c.s[:, 0] ** 2 + c.r * u ** 2)
        noise = (c.Lw @ c.rng.standard_normal((c.nx, c.n_envs))).T
        c.s = c.s @ c.F.T + noise
        return c.s[:, 0:1].copy(), cost


def compare(instance, coupled, seed, n_steps):
    """Returns (steps_compared, first_mismatch or None)."""
    p = native.instance_params(instance)

    ref_core = GLEBench(cs=(p["c"],), gammas=(p["gamma_bath"],),
                        theta=p["theta"], h=p["h"], q=p["q"], r=p["r"],
                        rho=p["rho"], n_envs=1, seed=seed)
    ref = ref_core if coupled else DecoupledReference(ref_core)

    env = native.GleamNative(instance, coupled=coupled, seed=seed)

    # Constant matrices must be identical before anything is stepped. If the
    # npz drifted from the reference these would diverge and every later
    # comparison would be uninterpretable.
    for label, a, b in (("F", ref_core.F, env.F), ("G", ref_core.G, env.G),
                        ("Lw", ref_core.Lw, env.Lw), ("L0", ref_core.L0, env.L0)):
        if bits(a) != bits(b):
            return 0, f"constant matrix {label} differs before any step"

    o_ref = ref.reset()
    o_nat, _ = env.reset(seed=seed)
    if bits(o_ref[0]) != bits(o_nat):
        return 0, "reset observation differs"

    # Same action sequence for both, drawn from a stream independent of the
    # environments so it cannot mask an RNG desynchronisation.
    arng = np.random.default_rng(90210 + seed)
    compared = 0
    for t in range(n_steps):
        u = ACTION_SCALE * arng.standard_normal(1)
        o_ref, c_ref = ref.step(u)
        o_nat, rew, term, trunc, info = env.step(u)

        if bits(o_ref[0]) != bits(o_nat):
            return compared, f"observation differs at step {t}"
        if bits(c_ref[0]) != bits(info["cost"]):
            return compared, f"cost differs at step {t}"
        if bits(-c_ref[0]) != bits(rew):
            return compared, f"reward is not minus cost at step {t}"
        if term:
            return compared, f"native reported termination at step {t}"
        compared += 1

        # The native class truncates at episode_len and resets. GLEBench is a
        # continuing task with no boundary, so the reference must be reset in
        # the same place and in the same way to keep the streams aligned.
        if trunc:
            o_ref = ref.reset()
            o_nat, _ = env.reset()
            if bits(o_ref[0]) != bits(o_nat):
                return compared, f"reset observation differs after step {t}"
    return compared, None


def main():
    print("=" * 78)
    print(" gleam_native.py vs GLEBench, bitwise")
    print("=" * 78)
    print(f"\n  {STEPS_PER_CASE} steps per case, actions ~ "
          f"{ACTION_SCALE} * N(0,1), floats compared as raw bytes\n")

    total = 0
    failures = []
    for instance in native.instance_names():
        for coupled in (True, False):
            arm = "coupled" if coupled else "decoupled"
            n, err = compare(instance, coupled, seed=7, n_steps=STEPS_PER_CASE)
            total += n
            if err is None:
                print(f"    ok        {instance:16} {arm:10} {n:6d} steps identical")
            else:
                print(f"    MISMATCH  {instance:16} {arm:10} {err}")
                failures.append((instance, arm, err))

    print(f"\n  total steps compared: {total:,}")

    # The cost ceiling law, printed as the protocol requires.
    print("\n  cross-check, rThK0 must be 0.3000 for every instance:")
    bad_law = []
    for instance in native.instance_names():
        v = native.baselines(instance)["rThK0"]
        print(f"    {instance:16} rThK0 = {v:.4f}")
        if abs(v - 0.3000) > 5e-5:
            bad_law.append(instance)

    if failures or bad_law:
        print("\n  RESULT: FAILED. Nothing ships until this passes.")
        return 1
    print(f"\n  RESULT: PASS. {total:,} steps bitwise identical across "
          f"{len(native.instance_names())} instances and both arms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
