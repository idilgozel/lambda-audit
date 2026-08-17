"""Bitwise equivalence: envs_native.BattleshipEasy vs popgym's BattleshipEasy.

Run where popgym IS installed (laptop/container, not the cluster):
    python3 harness/test_battleship_equivalence.py
Asserts: 200 seeded boards identical, every step's (obs-bit, reward,
terminated, truncated) identical over full episodes, and RNG stream
continuation across unseeded resets identical. Mirrors
test_env_equivalence.py; part of the audit's reproducibility package.
"""
import numpy as np
import envs_native
from popgym.envs.battleship import BattleshipEasy as RefEnv


def main():
    nat = envs_native.make("BattleshipEasy")
    ref = RefEnv()
    rng = np.random.default_rng(20260818)
    steps = 0
    for seed in range(200):
        on, _ = nat.reset(seed=seed)
        orf, _ = ref.reset(seed=seed)
        assert np.array_equal(nat.board, ref.board), seed
        assert on[0] == float(orf)
        done = False
        while not done:
            r, c = int(rng.integers(8)), int(rng.integers(8))
            x1 = nat.step(r * 8 + c)
            x2 = ref.step(np.array([r, c]))
            assert x1[0][0] == float(x2[0]) and x1[1] == x2[1] and x1[2:4] == x2[2:4]
            steps += 1
            done = x1[2] or x1[3]
    nat.reset(seed=77); ref.reset(seed=77)
    for _ in range(5):
        nat.reset(); ref.reset()
        assert np.array_equal(nat.board, ref.board)
    print(f"PASS: bitwise identical over 200 boards, {steps} steps, "
          f"5 unseeded-reset continuations")


if __name__ == "__main__":
    main()
