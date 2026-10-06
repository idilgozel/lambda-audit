"""Local runner for E15 (learned exploration). Same discipline as run_e13.py /
run_e14.py: pre-flight freeze check, --check-clean unless --resume, configs
generated from the prereg's constants so runner and scorer cannot disagree,
duplicate cells refused at the source, one torch thread per run, per-run logs
and a provenance record.

    python3 analysis/run_e15.py [--workers N] [--resume] [--dry-run]
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
RESULTS = os.path.join(RES_DIR, "results_e15.jsonl")
PY = (os.path.join(ROOT, ".venv", "bin", "python")
      if os.path.exists(os.path.join(ROOT, ".venv", "bin", "python")) else sys.executable)
sys.path.insert(0, ROOT)
import prereg_e15 as P  # noqa: E402

_lock = threading.Lock()


def say(*a):
    with _lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def configs():
    out = []
    for lam in P.LAMBDAS:
        for seed in P.SEEDS:
            args = (f"--env {P.ENV} --arch {P.ARCH} --gae-lambda {lam} --gamma {P.GAMMA} "
                    f"--total-timesteps {P.TOTAL_STEPS} --results-name results_e15.jsonl")
            out.append((lam, seed, args))
    return out


def done_cells():
    cells = set()
    if os.path.isfile(RESULTS):
        for line in open(RESULTS):
            if line.strip():
                r = json.loads(line)
                if r.get("env") == P.ENV and r.get("arch") == P.ARCH and r.get("fix_sigma") is None:
                    cells.add((round(float(r["lam"]), 6), int(r["seed"])))
    return cells


def run_one(lam, seed, args, jobid, idx):
    log = os.path.join(LOG_DIR, f"lam_e15.o{jobid}.{idx}")
    cmd = [PY, "ppo_lambda.py"] + args.split() + ["--seed", str(seed), "--torch-threads", "1", "--out-dir", RES_DIR]
    t0 = time.time()
    with open(log, "w") as f:
        f.write(f"host      : {platform.node()} (local, {platform.machine()})\n")
        f.write(f"task      : {idx} -> lambda {lam}, seed {seed}\nargs      : {args}\n")
        f.write(f"starting  : {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=HARNESS, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"finished  : {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  rc={p.returncode}  wall={time.time() - t0:.0f}s\n")
    return idx, lam, seed, p.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(14, (os.cpu_count() or 4) - 4)))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    P._assert_frozen()
    if not a.resume and os.path.isfile(RESULTS) and done_cells():
        raise SystemExit("E15 rows already exist. Score them, or pass --resume to complete missing cells only.")
    # the harness must record the learned sigma
    src = open(os.path.join(HARNESS, "ppo_lambda.py")).read()
    if "final_sigma" not in src:
        raise SystemExit("harness/ppo_lambda.py does not record final_sigma; P18.3 needs it. Nothing runs.")
    os.makedirs(RES_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    have = done_cells()
    todo = [(i + 1, lam, seed, args) for i, (lam, seed, args) in enumerate(configs())
            if (round(lam, 6), seed) not in have]
    say(f"E15: {len(todo)} of {len(configs())} cells to run on {a.workers} workers (host {platform.node()})")
    if a.dry_run:
        for i, lam, seed, args in todo[:5]:
            print("  ", i, lam, seed, args)
        return 0
    prov = dict(written=dt.datetime.now(dt.timezone.utc).isoformat(), host=platform.node(),
                cpu=subprocess.run("sysctl -n machdep.cpu.brand_string", shell=True, capture_output=True, text=True).stdout.strip(),
                torch=subprocess.run([PY, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip(),
                numpy=subprocess.run([PY, "-c", "import numpy; print(numpy.__version__)"], capture_output=True, text=True).stdout.strip(),
                git=subprocess.run(f"git -C {ROOT} rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip(),
                dirty=subprocess.run(f"git -C {ROOT} status --short", shell=True, capture_output=True, text=True).stdout.strip())
    with open(os.path.join(LOG_DIR, "e15_provenance.json"), "a") as f:
        f.write(json.dumps(prov) + "\n")
    jobid = time.strftime("%Y%m%d%H%M")
    fails = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, lam, seed, args, jobid, i) for i, lam, seed, args in todo]
        n = 0
        for fu in as_completed(futs):
            idx, lam, seed, rc, wall = fu.result()
            n += 1
            if rc != 0:
                fails += 1
                say(f"task {idx} (lambda {lam}, seed {seed}) FAILED rc={rc}; see logs/lam_e15.o{jobid}.{idx}")
            else:
                say(f"task {idx} (lambda {lam}, seed {seed}) done in {wall / 60:.1f} min [{n}/{len(todo)}]")
    say(f"finished, {fails} failure(s); scoring")
    r = subprocess.run([PY, os.path.join(ROOT, "prereg_e15.py"), RESULTS], capture_output=True, text=True)
    print(r.stdout + r.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
