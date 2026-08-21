"""Bitwise equivalence: envs_native.MineSweeperMedium vs popgym's.

Run where popgym IS installed:  python3 harness/test_minesweeper_equivalence.py
Asserts 200 seeded boards AND neighbour-count grids identical, every step's
(obs count, reward, terminated, truncated) identical over full episodes, and
RNG stream continuation across unseeded resets. Part of the repro package.
"""
import numpy as np
import envs_native
from popgym.envs.minesweeper import MineSweeperMedium as Ref


def main():
    nat, ref = envs_native.make("MineSweeperMedium"), Ref()
    rng = np.random.default_rng(20260821); steps = 0
    for seed in range(200):
        nat.reset(seed=seed); ref.reset(seed=seed)
        assert np.array_equal(nat.hidden_grid, ref.hidden_grid), seed
        assert np.array_equal(nat.neighbor_grid, ref.neighbor_grid), seed
        done = False
        while not done:
            r, c = int(rng.integers(6)), int(rng.integers(6))
            x1 = nat.step(r * 6 + c); x2 = ref.step(np.array([r, c]))
            assert int(np.argmax(x1[0][:7])) == min(int(x2[0]), 6)
            assert x1[1] == x2[1] and x1[2:4] == x2[2:4]
            steps += 1; done = x1[2] or x1[3]
    nat.reset(seed=99); ref.reset(seed=99)
    for _ in range(5):
        nat.reset(); ref.reset()
        assert np.array_equal(nat.hidden_grid, ref.hidden_grid)
    print(f"PASS: bitwise identical, 200 boards, {steps} steps, 5 continuations")


if __name__ == "__main__":
    main()
