"""Separate E12's capability-probe rows from the 80-run grid rows.

Written 2026-08-21, after the grid landed. The probe (2 runs, mlp and gru128 at
lambda 1.0 seed 1) and the grid both wrote to results_e12.jsonl, and the grid's
task numbering re-runs exactly those two cells: with 16 config lines and
seed = (task-1)//16 + 1, task 8 is (mlp, 1.0, seed 1) and task 16 is
(gru128, 1.0, seed 1). ppo_lambda.py opens the results file in append mode, so
the file ends up with two copies of each.

Why that matters: lambda = 1.0 is the reference cell every E12 prediction is
measured against. Duplicated, it carries seed 1 twice, and prereg_e12.iqm()
trims the middle 4 of 6 instead of the middle 3 of 5. The bootstrap also
resamples len(cell) draws, so interval widths move too.

The two duplicate pairs are BITWISE IDENTICAL in deployed_return
(mlp -0.1463928571, gru128 -0.0545595238), differing only in wallclock_s. Both
the probe and the grid tasks ran on the d00a node class, so this is the
harness's determinism showing rather than a coincidence, and the separation
below is unambiguous: no judgement is exercised about which value to keep.

This script does NOT delete anything. It moves the probe's two rows to
results_e12_probe.jsonl, where they remain the record of the capability gates
that were read before the grid was submitted, and leaves the grid's own rows in
results_e12.jsonl.

    python3 analysis/split_e12_probe.py --dry-run     # default; shows the plan
    python3 analysis/split_e12_probe.py --apply
"""
import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "results")
MAIN = os.path.join(RES, "results_e12.jsonl")
PROBE = os.path.join(RES, "results_e12_probe.jsonl")

# The probe's configuration, from myriad/configs_e12_probe.txt.
PROBE_CELLS = {("mlp", 1.0, 1), ("gru128", 1.0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the files (default is a dry run)")
    a = ap.parse_args()

    if not os.path.isfile(MAIN):
        raise SystemExit(f"no {MAIN}")
    lines = [l for l in open(MAIN) if l.strip()]

    first, grid, probe = {}, [], []
    for l in lines:
        r = json.loads(l)
        key = (r["arch"], float(r["lam"]), int(r["seed"]))
        if key in PROBE_CELLS and key not in first:
            # First occurrence of a probe cell is the probe run: the probe
            # completed at 16:40Z and the grid started at ~17:00Z, so the
            # probe's row was appended first.
            first[key] = r
            probe.append(l)
        else:
            grid.append(l)

    print(f"input          {len(lines)} rows in results_e12.jsonl")
    print(f"probe rows     {len(probe)} -> results_e12_probe.jsonl")
    print(f"grid rows      {len(grid)} -> results_e12.jsonl")

    # Verify the separation is value-neutral before touching anything.
    problems = []
    for key, pr in first.items():
        twins = [json.loads(l) for l in grid
                 if (json.loads(l)["arch"], float(json.loads(l)["lam"]),
                     int(json.loads(l)["seed"])) == key]
        if not twins:
            problems.append(f"{key}: probe row has no grid counterpart; "
                            f"removing it would LOSE a registered cell")
        elif any(abs(t["deployed_return"] - pr["deployed_return"]) > 0 for t in twins):
            problems.append(
                f"{key}: probe {pr['deployed_return']!r} differs from grid "
                f"{[t['deployed_return'] for t in twins]!r} -- the separation is "
                f"NOT value-neutral and must be decided explicitly, not here")
    if problems:
        print("\nREFUSING:")
        for p in problems:
            print(f"  {p}")
        return 2
    print("\ncheck: every probe row has a bitwise-identical grid counterpart.")

    # Report remaining coverage so a short cell cannot pass unnoticed.
    seen = {}
    for l in grid:
        r = json.loads(l)
        seen.setdefault((r["arch"], float(r["lam"])), set()).add(int(r["seed"]))
    short = {k: sorted({1, 2, 3, 4, 5} - v) for k, v in seen.items()
             if {1, 2, 3, 4, 5} - v}
    if short:
        print("\nINCOMPLETE CELLS in the grid (registered design is seeds 1-5):")
        for (arch, lam), miss in sorted(short.items()):
            print(f"  {arch:8} lam {lam:<6} missing seed(s) {miss}")
        print("  These are runs that did not execute, not results. Re-running")
        print("  them completes the frozen design; it is not data selection.")

    if not a.apply:
        print("\nDRY RUN. Re-run with --apply to write.")
        return 0

    shutil.copy2(MAIN, MAIN + ".bak-presplit")
    open(PROBE, "w").writelines(probe)
    open(MAIN, "w").writelines(grid)
    print(f"\nwritten. Backup at {os.path.basename(MAIN)}.bak-presplit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
