"""Prove envs_native.py is a bitwise drop-in for the real POPGym environments.

Runs both implementations through identical seeds and identical action
sequences, comparing observations, rewards, terminated and truncated at every
single step. Floats are compared bitwise, not approximately.

Requires popgym (available in any normal environment). It is NOT needed on the
cluster; this test is what licenses running without it there.

Run:  python test_env_equivalence.py
"""

import sys

import numpy as np

import envs_native as native

try:
    from popgym.envs.repeat_previous import (
        RepeatPrevious as PgRepeatPrevious,
        RepeatPreviousEasy as PgEasy,
        RepeatPreviousMedium as PgMedium,
        RepeatPreviousHard as PgHard,
    )
    from popgym.envs.position_only_cartpole import (
        PositionOnlyCartPoleEasy as PgCPEasy,
        PositionOnlyCartPoleMedium as PgCPMedium,
        PositionOnlyCartPoleHard as PgCPHard,
    )
except ImportError:
    print("popgym not installed. This test must run where popgym is available.")
    sys.exit(2)


def bits(x):
    """Bitwise-comparable representation of a scalar or array."""
    a = np.asarray(x)
    return (a.dtype.str, a.shape, a.tobytes())


def compare_episode(env_a, env_b, seed, actions, label, first_reset_seeded=True):
    """Step both envs identically. Returns number of steps compared."""
    oa, _ = env_a.reset(seed=seed) if first_reset_seeded else env_a.reset()
    ob, _ = env_b.reset(seed=seed) if first_reset_seeded else env_b.reset()
    assert bits(oa) == bits(ob), (
        f"{label}: reset obs differ. popgym={oa!r} native={ob!r}")
    n = 0
    for a in actions:
        ra = env_a.step(a)
        rb = env_b.step(a)
        oa, rwa, ta, tra, _ = ra
        ob, rwb, tb, trb, _ = rb
        assert bits(oa) == bits(ob), (
            f"{label} step {n}: obs differ. popgym={oa!r} native={ob!r}")
        assert bits(np.float64(rwa)) == bits(np.float64(rwb)), (
            f"{label} step {n}: reward differ. popgym={rwa!r} native={rwb!r}")
        assert bool(ta) == bool(tb), (
            f"{label} step {n}: terminated differ {ta} vs {tb}")
        assert bool(tra) == bool(trb), (
            f"{label} step {n}: truncated differ {tra} vs {trb}")
        n += 1
        if ta or tb or tra or trb:
            break
    return n


def run_pair(make_pg, make_native, label, n_seeds=8, n_episodes=4):
    total = 0
    for seed in range(n_seeds):
        pg = make_pg()
        nv = make_native()
        rng = np.random.default_rng(1234 + seed)
        # episode 0 is seeded; later episodes reset WITHOUT a seed, which is
        # exactly what the training harness does, and exercises the fact that
        # the RNG stream must continue rather than restart
        for ep in range(n_episodes):
            n_actions = 2000
            acts = [int(rng.integers(pg.action_space.n)) for _ in range(n_actions)]
            steps = compare_episode(
                pg, nv, seed, acts, f"{label} seed={seed} ep={ep}",
                first_reset_seeded=(ep == 0))
            total += steps
    print(f"  OK  {label}: {total} steps compared bitwise across "
          f"{n_seeds} seeds x {n_episodes} episodes")
    return total


def main():
    print("Bitwise equivalence: popgym vs envs_native")
    grand = 0
    grand += run_pair(PgEasy, native.RepeatPreviousEasy, "RepeatPreviousEasy")
    grand += run_pair(PgMedium, native.RepeatPreviousMedium, "RepeatPreviousMedium")
    grand += run_pair(PgHard, native.RepeatPreviousHard, "RepeatPreviousHard")
    grand += run_pair(lambda: PgRepeatPrevious(num_decks=1, k=16),
                      lambda: native.RepeatPrevious(num_decks=1, k=16),
                      "RepeatPreviousK16")
    grand += run_pair(PgCPEasy, native.PositionOnlyCartPoleEasy,
                      "PositionOnlyCartPoleEasy")
    grand += run_pair(PgCPMedium, native.PositionOnlyCartPoleMedium,
                      "PositionOnlyCartPoleMedium")
    grand += run_pair(PgCPHard, native.PositionOnlyCartPoleHard,
                      "PositionOnlyCartPoleHard")

    # spaces must agree too
    for pg_cls, nv_cls, label in [
        (PgEasy, native.RepeatPreviousEasy, "RepeatPreviousEasy"),
        (PgCPEasy, native.PositionOnlyCartPoleEasy, "PositionOnlyCartPoleEasy"),
    ]:
        p, n = pg_cls(), nv_cls()
        assert int(p.action_space.n) == int(n.action_space.n), label
        pobs = p.observation_space
        nobs = n.observation_space
        if hasattr(pobs, "n"):
            assert int(pobs.n) == int(nobs.n), label
        else:
            assert tuple(pobs.shape) == tuple(nobs.shape), label
        assert p.max_episode_length == n.max_episode_length, label
    print(f"  OK  spaces and episode lengths agree")

    print(f"\nPASS. {grand} steps identical bit for bit. "
          f"envs_native is a verified drop-in for popgym.")


if __name__ == "__main__":
    main()
