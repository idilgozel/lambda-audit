"""Bitwise regression: does the CURRENT harness still reproduce PUBLISHED data?

Written 2026-08-21. Replaces the one-time before/after diff recorded in
MEMO-arch-linear.md section 6, which verified that adding the `linear` and
`linearb` policy heads left `mlp` and `gru128` bitwise unchanged. That check
was correct when it was run, but a memo recording a past comparison is weaker
than a comparison anyone can re-run, and `harness/ppo_lambda.py` has been
edited since the E6/E7/E9 data was written (harness mtime 2026-08-17,
results_phase2.jsonl mtime 2026-08-14). paper/03-method.md section 3.5 asserts
that "the registered E6/E7/E9 data remains reproducible from the released
file". This is the test of that sentence.

METHOD. Training is deterministic given (config, seed), and the harness writes
a curve row every update from the first update onward. So re-running a
published config for a SHORT budget must reproduce the PREFIX of the published
curve exactly. Comparing prefixes costs about a minute instead of the ~37
minutes a full 15M-step reproduction would take, and it is equally decisive:
any RNG shift, any change to initialisation order, any change to the update
maths moves update 1.

Columns `sps` (and, in the JSONL row, `wallclock_s`) are pure timing and are
excluded. MEMO-arch-linear.md section 6 records that an earlier version of the
check named only `wallclock_s` and the curve comparison then failed for no
reason; that lesson is encoded here rather than left in prose.

WHERE THIS MUST RUN. On Myriad, and on the same node CLASS that produced the
published curve. Cross-CPU-generation runs are not bitwise comparable — see
provenance-node-class.md and PROVENANCE.md. The test prints its host so the
class is part of the record. Run it anywhere else and a mismatch means "wrong
CPU", not "harness changed"; the test says so rather than letting the reader
guess.

    python3 harness/test_arch_regression.py                  # default pair
    python3 harness/test_arch_regression.py --updates 40
    python3 harness/test_arch_regression.py --run GLEAM_g1_h0.05_gru128_lam0.9_s1

Exit 0 = every checked run reproduced bitwise. Exit 1 = a mismatch, with the
first differing cell named. Exit 2 = could not test (curves absent, etc).
"""
import argparse
import csv
import os
import platform
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CURVES = os.path.join(ROOT, "results", "results")

# Timing-only columns. Excluded by design, not by convenience.
IGNORE = {"sps"}

# One memoryless and one recurrent published run. The recurrent one matters
# most: the GRU path is where an initialisation-order change would show up.
DEFAULT_RUNS = [
    "GLEAM_g1_h0.05_mlp_lam0.9_s1",
    "GLEAM_g1_h0.05_gru128_lam0.9_s1",
]


def parse_run(run):
    """'GLEAM_g1_h0.05_mlp_lam0.9_s1' -> (env, arch, lam, seed)."""
    if not run.startswith("GLEAM_"):
        raise SystemExit(f"only GLEAM runs are supported here, got {run}")
    body = run[len("GLEAM_"):]
    head, _, seed = body.rpartition("_s")
    head, _, lam = head.rpartition("_lam")
    inst, _, arch = head.rpartition("_")
    return f"GLEAM_{inst}", arch, lam, int(seed)


def read_curve(path, limit=None):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def compare(pub, new, n):
    """Return None if the first n rows agree on every non-timing column."""
    if len(new) < n:
        return f"replay produced only {len(new)} updates, needed {n}"
    for i in range(n):
        a, b = pub[i], new[i]
        for k in a:
            if k in IGNORE:
                continue
            if a[k] != b.get(k):
                return (f"update {a.get('update')}: column '{k}' "
                        f"published {a[k]!r} vs replay {b.get(k)!r}")
    return None


def host():
    try:
        return subprocess.run(["hostname", "-f"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or platform.node()
    except Exception:
        return platform.node()


def node_class(h):
    """'node-d00a-213.myriad...' -> 'd00a'. Anything else is not Myriad."""
    base = h.split(".")[0]
    parts = base.split("-")
    return parts[1] if len(parts) >= 3 and parts[0] == "node" else None


def replay(run, updates, outdir):
    env, arch, lam, seed = parse_run(run)
    # num_envs * num_steps = 2048 per update, matching the published header.
    steps = updates * 2048
    cmd = [sys.executable, os.path.join(HERE, "ppo_lambda.py"),
           "--env", env, "--arch", arch, "--gae-lambda", lam,
           "--seed", str(seed), "--total-timesteps", str(steps),
           "--fix-sigma", "0.3", "--torch-threads", "1",
           "--out-dir", outdir, "--results-name", "regression_scratch.jsonl"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"replay failed for {run}:\n{p.stderr[-2000:]}")
    return os.path.join(outdir, f"curves_{run}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=20,
                    help="how many published updates to reproduce (default 20)")
    ap.add_argument("--run", action="append", default=None,
                    help="published run name; repeatable. Default: one mlp, one gru128")
    a = ap.parse_args()
    runs = a.run or DEFAULT_RUNS

    h = host()
    cls = node_class(h)
    print(f"host        : {h}")
    print(f"node class  : {cls if cls else 'NOT A MYRIAD COMPUTE NODE'}")
    if cls is None:
        print("  Bitwise comparison against published curves is only meaningful")
        print("  on the node class that produced them. A mismatch here is not")
        print("  evidence the harness changed. See PROVENANCE.md.")
    print(f"torch       : ", end="", flush=True)
    try:
        import torch
        print(torch.__version__)
    except Exception as exc:
        print(f"unavailable ({exc})")
        return 2
    print(f"updates     : {a.updates}\n")

    failures, tested = [], 0
    with tempfile.TemporaryDirectory() as td:
        for run in runs:
            pub_path = os.path.join(CURVES, f"curves_{run}.csv")
            if not os.path.isfile(pub_path):
                print(f"  {run:44} SKIP (published curve absent; "
                      f"curves are gitignored, see PROVENANCE.md)")
                continue
            pub = read_curve(pub_path, a.updates)
            if len(pub) < a.updates:
                print(f"  {run:44} SKIP (published curve has {len(pub)} rows)")
                continue
            new = read_curve(replay(run, a.updates, td))
            why = compare(pub, new, a.updates)
            tested += 1
            if why:
                failures.append((run, why))
                print(f"  {run:44} MISMATCH")
                print(f"      {why}")
            else:
                print(f"  {run:44} IDENTICAL over {a.updates} updates "
                      f"(all columns but {sorted(IGNORE)})")

    print()
    if not tested:
        print("NOTHING TESTED. Published curves are not present on this machine.")
        return 2
    if failures:
        print(f"REGRESSION: {len(failures)} of {tested} runs did not reproduce.")
        if cls is None:
            print("Re-run on the originating Myriad node class before concluding"
                  " the harness changed.")
        return 1
    print(f"OK: {tested}/{tested} published runs reproduce bitwise from the "
          f"current harness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
