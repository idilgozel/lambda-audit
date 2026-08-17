"""Scripted reference policies for BattleshipEasy. Generates the frozen
anchors quoted in prereg_e11.py (20,000 episodes, seeds 1,000,000+, action
rng 42). Run: python3 analysis/e11_baselines.py"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
import envs_native


def run(policy, n_ep=20000, seed0=1_000_000):
    env = envs_native.make("BattleshipEasy")
    rng = np.random.default_rng(42)
    rets = np.empty(n_ep)
    for ep in range(n_ep):
        env.reset(seed=seed0 + ep)
        fired = np.zeros(64, bool); hits = []; tot = 0.0; done = False
        while not done:
            a = policy(rng, fired, hits)
            o, r, t, tr, _ = env.step(a)
            fired[a] = True
            if o[0] > 0.5:
                hits.append(a)
            tot += r; done = t or tr
        rets[ep] = tot
    return rets


def uniform(rng, fired, hits):
    return int(rng.integers(64))


def no_repeat(rng, fired, hits):
    u = np.flatnonzero(~fired); return int(u[rng.integers(len(u))])


def hunt_target(rng, fired, hits):
    cand = []
    for a in hits:
        r, c = divmod(a, 8)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8 and not fired[rr * 8 + cc]:
                cand.append(rr * 8 + cc)
    if cand:
        return int(cand[rng.integers(len(cand))])
    u = np.flatnonzero(~fired); return int(u[rng.integers(len(u))])


if __name__ == "__main__":
    for name, pol in [("uniform", uniform), ("no_repeat", no_repeat),
                      ("hunt_target", hunt_target)]:
        r = run(pol)
        iqm = np.mean(np.sort(r)[len(r) // 4:-len(r) // 4])
        print(f"{name:12} mean {r.mean():+.4f}  IQM {iqm:+.4f}")
