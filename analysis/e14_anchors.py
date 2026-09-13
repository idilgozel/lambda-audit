"""Scripted anchors for E14 (PositionOnlyPendulum), frozen with prereg_e14.py.

Three memoryless policies, none of which learns anything, so their scores
bracket what a learner must beat to count as "off the floor":

    uniform     torque ~ U(-2, 2) every step (the random floor)
    zero        torque = 0 (let it swing; the passive floor)
    bangbang    torque = -2 * sign(sin theta): push toward the top using the
                current angle only. This is the strongest thing a policy can
                do from [cos theta, sin theta] without an estimate of theta_dot
                (any function of the angle alone is a conservative torque
                field, so it cannot damp the swing). It is the memoryless
                heuristic the MLP arm is measured against.

Identical to analysis/e12_baselines.py in protocol: 20,000 episodes per
policy per variant, episode seeds 2,000,000 + i, action rng seed 42. Uses
harness/envs_native.py, which test_env_equivalence.py proves bitwise
identical to popgym.

Run:  python3 analysis/e14_anchors.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
import envs_native  # noqa: E402

N_EPISODES = 20_000
EPISODE_SEED0 = 2_000_000
ACTION_SEED = 42


def iqm(xs):
    xs = sorted(xs)
    n = len(xs)
    lo, hi = n // 4, n - n // 4
    m = xs[lo:hi] if hi > lo else xs
    return sum(m) / len(m)


def run(env_id, policy):
    rng = np.random.default_rng(ACTION_SEED)
    env = envs_native.make(env_id)
    rets = []
    for i in range(N_EPISODES):
        o, _ = env.reset(seed=EPISODE_SEED0 + i)
        total = 0.0
        while True:
            if policy == "uniform":
                a = np.array([rng.uniform(-2.0, 2.0)])
            elif policy == "zero":
                a = np.array([0.0])
            elif policy == "bangbang":
                a = np.array([-2.0 * np.sign(o[1]) if o[1] != 0 else 0.0])
            else:
                raise ValueError(policy)
            o, r, te, tr, _ = env.step(a)
            total += r
            if te or tr:
                break
        rets.append(total)
    return float(np.mean(rets)), float(iqm(rets)), float(np.std(rets))


def main():
    print(f"E14 anchors: {N_EPISODES} episodes, seeds {EPISODE_SEED0}+, "
          f"action rng {ACTION_SEED}")
    for env_id in ("PositionOnlyPendulumEasy", "PositionOnlyPendulumHard"):
        print(env_id)
        for pol in ("uniform", "zero", "bangbang"):
            mean, q, sd = run(env_id, pol)
            print(f"  {pol:9s}  mean {mean:+.4f}  IQM {q:+.4f}  sd {sd:.4f}")


if __name__ == "__main__":
    main()
