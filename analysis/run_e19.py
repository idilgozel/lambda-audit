"""Local runner for E19 (critic swap). Same discipline as run_e18.py: pre-flight
freeze check, --check-clean unless --resume, configs generated from the prereg's
constants, duplicate cells refused at the source, one torch thread per run,
per-run logs and a provenance record. Refuses to start unless the harness carries
the --gleam-episode-len, --log-critic-fit and --oracle-critic options.

    python3 analysis/run_e19.py [--workers N] [--resume] [--dry-run] [--status]
"""
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "harness")
RES_DIR = os.path.join(ROOT, "results", "results")
LOG_DIR = os.path.join(ROOT, "results", "logs")
RESULTS = os.path.join(RES_DIR, "results_e19.jsonl")
PY = (os.path.join(ROOT, ".venv", "bin", "python")
      if os.path.exists(os.path.join(ROOT, ".venv", "bin", "python")) else sys.executable)
sys.path.insert(0, ROOT)
import prereg_e19 as P  # noqa: E402

_lock = threading.Lock()


def say(*a):
    with _lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def run_name(inst, lam, side, critic, seed):
    return f"{inst}_linear_{critic}_lam{lam}_{side}_k{P.k0_of(inst, lam, side):.6f}_s{seed}"


def k0_arg(inst, lam, side):
    """Start gain as passed to the harness: full precision, round-trips to the registered cell.
    (Run 1 of 2026-10-06 used :g, six significant figures, and matched no cell; see
    results/results/e19_run1_unscorable/README.md.)"""
    k0 = P.k0_of(inst, lam, side)
    s = f"{k0:.6f}"
    assert abs(float(s) - k0) < 1e-7, (inst, lam, side, s, k0)
    return s


def configs():
    out = []
    for inst in P.INSTANCES:
        for lam in P.LAMBDAS:
            for side in P.START_FACTORS:
                for critic in P.CRITICS:
                    for seed in P.SEEDS:
                        args = (f"--env {inst} --arch linear --gae-lambda {lam} --gamma {P.BETA[inst]} --fix-sigma {P.FIX_SIGMA} "
                                f"--linear-gain-init {k0_arg(inst, lam, side)} --gleam-episode-len {P.EPISODE_LEN} "
                                f"--log-critic-fit 1 --total-timesteps {P.TOTAL_STEPS} --results-name results_e19.jsonl"
                                + (" --oracle-critic lstd" if critic == "oracle" else ""))
                        out.append(((inst, lam, side, critic), seed, args))
    return out


def done_cells():
    cells = set()
    if os.path.isfile(RESULTS):
        for r in P.rows_of(P.load(RESULTS)):
            cells.add((r["env"], round(float(r["lam"]), 6), round(float(r["linear_gain_init"]), 6), P.critic_of(r), int(r["seed"])))
    return cells


def key_of(key, seed):
    inst, lam, side, critic = key
    return (inst, round(lam, 6), round(P.k0_of(inst, lam, side), 6), critic, seed)


def run_one(key, seed, args, jobid, idx):
    inst, lam, side, critic = key
    log = os.path.join(LOG_DIR, f"lam_e19.o{jobid}.{idx}")
    cmd = [PY, "ppo_lambda.py"] + args.split() + ["--seed", str(seed), "--torch-threads", "1", "--out-dir", RES_DIR,
                                                 "--run-name", run_name(inst, lam, side, critic, seed)]
    t0 = time.time()
    with open(log, "w") as f:
        f.write(f"host      : {platform.node()} (local, {platform.machine()})\n")
        f.write(f"task      : {idx} -> {key}, seed {seed}\nargs      : {args}\n")
        f.write(f"starting  : {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=HARNESS, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"finished  : {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  rc={p.returncode}  wall={time.time() - t0:.0f}s\n")
    return idx, key, seed, p.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 4)))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    P._assert_frozen()
    src = open(os.path.join(HARNESS, "ppo_lambda.py")).read()
    if not all(k in src for k in ("gleam_episode_len", "log_critic_fit", "oracle_critic")):
        raise SystemExit("harness lacks --gleam-episode-len / --log-critic-fit / --oracle-critic; apply the analysis/apply_*_patch.py "
                         "patches first (after E16 has exited).")
    if a.status:
        print(f"E19: {len(done_cells())}/{len(configs())} cells present")
        r = subprocess.run([PY, os.path.join(ROOT, "prereg_e19.py"), RESULTS], capture_output=True, text=True)
        print(r.stdout + r.stderr); return 0
    if not a.resume and os.path.isfile(RESULTS) and done_cells():
        raise SystemExit("E19 rows already exist. Score them, or pass --resume to complete missing cells only.")
    os.makedirs(RES_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    have = done_cells()
    todo = [(i + 1, key, seed, args) for i, (key, seed, args) in enumerate(configs()) if key_of(key, seed) not in have]
    say(f"E19: {len(todo)} of {len(configs())} cells to run on {a.workers} workers (host {platform.node()})")
    if a.dry_run:
        for i, key, seed, args in todo[:4]:
            print("  ", i, key, seed, args)
        return 0
    prov = dict(written=dt.datetime.now(dt.timezone.utc).isoformat(), host=platform.node(),
                cpu=subprocess.run("sysctl -n machdep.cpu.brand_string", shell=True, capture_output=True, text=True).stdout.strip(),
                torch=subprocess.run([PY, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip(),
                numpy=subprocess.run([PY, "-c", "import numpy; print(numpy.__version__)"], capture_output=True, text=True).stdout.strip(),
                harness_md5=subprocess.run(f"md5 -q {HARNESS}/ppo_lambda.py 2>/dev/null || md5sum {HARNESS}/ppo_lambda.py", shell=True, capture_output=True, text=True).stdout.strip(),
                git=subprocess.run(f"git -C {ROOT} rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip(),
                dirty=subprocess.run(f"git -C {ROOT} status --short", shell=True, capture_output=True, text=True).stdout.strip())
    with open(os.path.join(LOG_DIR, "e19_provenance.json"), "a") as f:
        f.write(json.dumps(prov) + "\n")
    jobid = time.strftime("%Y%m%d%H%M")
    fails = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, key, seed, args, jobid, i) for i, key, seed, args in todo]
        n = 0
        for fu in as_completed(futs):
            idx, key, seed, rc, wall = fu.result()
            n += 1
            if rc != 0:
                fails += 1
                say(f"task {idx} ({key}, seed {seed}) FAILED rc={rc}; see logs/lam_e19.o{jobid}.{idx}")
            else:
                say(f"task {idx} ({key}, seed {seed}) done in {wall / 60:.1f} min [{n}/{len(todo)}]")
    say(f"finished, {fails} failure(s); scoring")
    r = subprocess.run([PY, os.path.join(ROOT, "prereg_e19.py"), RESULTS], capture_output=True, text=True)
    print(r.stdout + r.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
