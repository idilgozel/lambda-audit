"""Local runner for E18 (E7 under the continuing objective). Same discipline as
run_e16.py: pre-flight freeze check, --check-clean unless --resume, configs
generated from the prereg's constants so runner and scorer cannot disagree,
duplicate cells refused at the source, one torch thread per run, per-run logs
and a provenance record. Refuses to start unless the harness has the
--gleam-episode-len option (analysis/apply_episode_len_patch.py).

    python3 analysis/run_e18.py [--workers N] [--resume] [--dry-run] [--status]
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
RESULTS = os.path.join(RES_DIR, "results_e18.jsonl")
PY = (os.path.join(ROOT, ".venv", "bin", "python")
      if os.path.exists(os.path.join(ROOT, ".venv", "bin", "python")) else sys.executable)
sys.path.insert(0, ROOT)
import prereg_e18 as P  # noqa: E402

ORDER = [i for i in ["GLEAM_g1_h0.05", "GLEAM_g1_h0.02", "GLEAM_g2_h0.05", "GLEAM_g4_h0.05"] if i in P.INSTANCES]
_lock = threading.Lock()


def say(*a):
    with _lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def configs():
    out = []
    for inst in ORDER:
        for lam in P.LAMBDAS:
            for seed in P.SEEDS:
                args = (f"--env {inst} --arch {P.ARCH} --gae-lambda {lam} --gamma {P.BETA[inst]} "
                        f"--fix-sigma {P.FIX_SIGMA} --total-timesteps {P.TOTAL_STEPS} "
                        f"--gleam-episode-len {P.EPISODE_LEN} --log-critic-fit {P.LOG_CRITIC_FIT} --results-name results_e18.jsonl")
                out.append(((inst, lam), seed, args))
    return out


def done_cells():
    cells = set()
    if os.path.isfile(RESULTS):
        for inst in P.INSTANCES:
            for r in P.rows_of(P.load(RESULTS), inst):
                cells.add(((inst, round(float(r["lam"]), 6)), int(r["seed"])))
    return cells


def run_one(key, seed, args, jobid, idx):
    log = os.path.join(LOG_DIR, f"lam_e18.o{jobid}.{idx}")
    cmd = [PY, "ppo_lambda.py"] + args.split() + ["--seed", str(seed), "--torch-threads", "1", "--out-dir", RES_DIR]
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
    if "gleam_episode_len" not in src or "log_critic_fit" not in src:
        raise SystemExit("harness lacks --gleam-episode-len / --log-critic-fit; run analysis/apply_episode_len_patch.py and "
                         "analysis/apply_critic_fit_patch.py first (after E16 has exited).")
    if a.status:
        print(f"E18: {len(done_cells())}/{len(configs())} cells present")
        r = subprocess.run([PY, os.path.join(ROOT, "prereg_e18.py"), RESULTS], capture_output=True, text=True)
        print(r.stdout + r.stderr); return 0
    if not a.resume and os.path.isfile(RESULTS) and done_cells():
        raise SystemExit("E18 rows already exist. Score them, or pass --resume to complete missing cells only.")
    os.makedirs(RES_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    have = done_cells()
    todo = [(i + 1, key, seed, args) for i, (key, seed, args) in enumerate(configs())
            if ((key[0], round(key[1], 6)), seed) not in have]
    say(f"E18: {len(todo)} of {len(configs())} cells to run on {a.workers} workers (host {platform.node()})")
    if a.dry_run:
        for i, key, seed, args in todo[:5]:
            print("  ", i, key, seed, args)
        return 0
    prov = dict(written=dt.datetime.now(dt.timezone.utc).isoformat(), host=platform.node(),
                cpu=subprocess.run("sysctl -n machdep.cpu.brand_string", shell=True, capture_output=True, text=True).stdout.strip(),
                torch=subprocess.run([PY, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip(),
                numpy=subprocess.run([PY, "-c", "import numpy; print(numpy.__version__)"], capture_output=True, text=True).stdout.strip(),
                harness_md5=subprocess.run(f"md5 -q {HARNESS}/ppo_lambda.py 2>/dev/null || md5sum {HARNESS}/ppo_lambda.py", shell=True, capture_output=True, text=True).stdout.strip(),
                git=subprocess.run(f"git -C {ROOT} rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip(),
                dirty=subprocess.run(f"git -C {ROOT} status --short", shell=True, capture_output=True, text=True).stdout.strip())
    with open(os.path.join(LOG_DIR, "e18_provenance.json"), "a") as f:
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
                say(f"task {idx} ({key}, seed {seed}) FAILED rc={rc}; see logs/lam_e18.o{jobid}.{idx}")
            else:
                say(f"task {idx} ({key}, seed {seed}) done in {wall / 60:.1f} min [{n}/{len(todo)}]")
    say(f"finished, {fails} failure(s); scoring")
    r = subprocess.run([PY, os.path.join(ROOT, "prereg_e18.py"), RESULTS], capture_output=True, text=True)
    print(r.stdout + r.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
