#!/usr/bin/env python3
"""Generate the Phase 2 config files from the frozen preregistration.

The config files are derived, never hand written, so the lambda grid, the
discount and the exploration noise cannot drift away from prereg_phase2.py.

The discount is the one place this matters most. The analytic k_AC(lambda) is
computed at beta = exp(-rho*h), so the agent has to discount at that beta or it
is not being compared against its own prediction. beta differs between the
h = 0.05 and h = 0.02 instances, so it is written per line rather than left at
the harness default of 0.99.

Task mapping is the existing one, unchanged:
    IDX = (TASK-1) % NCFG + 1,  SEED = (TASK-1)/NCFG + 1

    configs_e6.txt   2 arms x 8 lambda        = 16 lines, qsub -t 1-80
    configs_e7.txt   4 instances x 8 lambda   = 32 lines, qsub -t 1-160
    configs_e9.txt   2 archs x 2 envs x 8 lam = 32 lines, qsub -t 1-160

Run:  python3 myriad/build_phase2_configs.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import prereg_phase2 as P  # noqa: E402

HERE = os.path.join(ROOT, "myriad")
STEPS = P.TOTAL_STEPS
RESULTS = "results_phase2.jsonl"


def gleam_line(env_id, instance, lam):
    beta = P.BETA[instance]
    return (f"--env {env_id} --arch mlp --gae-lambda {lam} "
            f"--gamma {beta!r} --fix-sigma {P.FIX_SIGMA} "
            f"--total-timesteps {STEPS} --results-name {RESULTS}")


def write(path, lines, expected):
    if len(lines) != expected:
        raise SystemExit(f"{path}: built {len(lines)} lines, expected {expected}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.relpath(path, ROOT):28} {len(lines):3d} lines  "
          f"-> qsub -t 1-{len(lines) * len(P.SEEDS)}")


def main():
    print("Phase 2 configs, derived from the frozen prereg\n")

    # E6, the minimal pair. Arms outer, lambda inner.
    e6 = []
    for arm in P.E6_ARMS:
        env_id = P.E6_INSTANCE + ("" if arm == "coupled" else "_dec")
        for lam in P.LAMBDAS:
            e6.append(gleam_line(env_id, P.E6_INSTANCE, lam))
    write(os.path.join(HERE, "configs_e6.txt"), e6,
          len(P.E6_ARMS) * len(P.LAMBDAS))

    # E7, the quantitative law. Coupled arm only, instances in gamma*h order.
    #
    # E6_INSTANCE is deliberately EXCLUDED. E6 already ran it, coupled, at
    # seeds 1 to 5, and those rows land in the same results file. Re-running it
    # would append a second (env, lam, seed) row for every cell and corrupt
    # every IQM that reads them. E6's coupled arm IS this instance's E7 data,
    # which is legitimate because the P7 predictions were frozen before E6 ran.
    e7_new = [n for n in P.E7_INSTANCES if n != P.E6_INSTANCE]
    e7 = []
    for inst in sorted(e7_new, key=lambda n: P.GAMMA_H[n]):
        for lam in P.LAMBDAS:
            e7.append(gleam_line(inst, inst, lam))
    write(os.path.join(HERE, "configs_e7.txt"), e7,
          len(e7_new) * len(P.LAMBDAS))
    print(f"    E7 excludes {P.E6_INSTANCE}, already run by E6 at seeds "
          f"{P.SEEDS[0]} to {P.SEEDS[-1]}")

    # E9, corroboration. Discrete environments, so no fix-sigma, and the
    # pilot's gamma of 0.99 since these are the pilot's own environments.
    e9 = []
    for arch in P.E9_ARCHS:
        for env in P.E9_ENVS:
            for lam in P.LAMBDAS:
                e9.append(f"--env {env} --arch {arch} --gae-lambda {lam} "
                          f"--gamma 0.99 --total-timesteps {STEPS} "
                          f"--results-name {RESULTS}")
    write(os.path.join(HERE, "configs_e9.txt"), e9,
          len(P.E9_ARCHS) * len(P.E9_ENVS) * len(P.LAMBDAS))

    print("\n  first and last line of each file:")
    for name in ("configs_e6.txt", "configs_e7.txt", "configs_e9.txt"):
        lines = open(os.path.join(HERE, name)).read().splitlines()
        print(f"\n  {name}")
        print(f"    1   {lines[0]}")
        print(f"    {len(lines):<3} {lines[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
