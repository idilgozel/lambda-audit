"""Local runner for E13. Replaces deploy.sh for an experiment that does not
touch Myriad (see prereg_e13.py, WHERE THIS RUNS).

Ports deploy.sh's discipline rather than its transport:
  - refuses to start unless prereg_e13.py's freeze assertion passes AND
    --check-clean confirms no E13 rows exist (a fresh submission), or
    --resume is given (completing missing cells after a crash: the runner
    skips every (nl, lambda, seed) already present in the results file, so a
    resume can only ADD the frozen design's missing cells, never duplicate);
  - records host and torch provenance to results/logs/e13_host.txt;
  - writes per-run logs to results/logs/e13.<nnn>.log;
  - runs the 72 cells of the frozen design and nothing else - the config
    list is generated from prereg_e13.py's constants, so runner and scorer
    cannot disagree about what the design is.

    python3 analysis/run_e13.py [--workers N] [--resume] [--dry-run]

Wall-clock is host-dependent: the pre-freeze smoke measured ~780 us/step in
the cloud container (2 CPUs; 72 runs would take ~9 h) and Myriad's rate was
~150 us/step. On an Apple-Silicon laptop expect a few minutes per run:
72 runs at --workers 6 is roughly 1-2 h. The runner prints per-run times, so
the first few completions calibrate the estimate for wherever it is running.
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import prereg_e13 as P  # noqa: E402  (freeze-asserted on import via main())

RES_DIR = os.path.join(ROOT, "results", "results")
LOG_DIR = os.path.join(ROOT, "results", "logs")
RESULTS = os.path.join(RES_DIR, "results_e13.jsonl")


def configs():
    out = []
    for nl in P.NL_ARMS:
        for lam, k0 in zip(P.LAMBDAS, P.GAIN_INIT):
            for seed in P.SEEDS:
                out.append(dict(nl=nl, lam=lam, seed=seed, gain=k0))
    return out


def done_cells():
    if not os.path.isfile(RESULTS):
        return set()
    cells = set()
    for line in open(RESULTS):
        if not line.strip():
            continue
        r = json.loads(line)
        nl = P._parse_nl(r.get("env", ""))
        if nl is not None:
            cells.add((nl, round(float(r["lam"]), 6), int(r["seed"])))
    return cells


def one(idx, c):
    env = P.env_id(c["nl"])
    log = os.path.join(LOG_DIR, f"e13.{idx:03d}.log")
    cmd = [sys.executable, os.path.join(ROOT, "harness", "ppo_lambda.py"),
           "--env", env, "--arch", "linear",
           "--gae-lambda", str(c["lam"]), "--gamma", repr(P.GAMMA_RL),
           "--fix-sigma", str(P.FIX_SIGMA),
           "--linear-gain-init", str(c["gain"]),
           "--total-timesteps", str(P.STEPS),
           "--seed", str(c["seed"]), "--torch-threads", "1",
           "--out-dir", RES_DIR, "--results-name", "results_e13.jsonl"]
    t0 = time.time()
    with open(log, "w") as f:
        f.write(f"cmd: {' '.join(cmd)}\nstarted: {time.strftime('%FT%TZ', time.gmtime())}\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return idx, c, p.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="complete missing cells of the frozen design")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # freeze pre-flight, exactly as deploy.sh runs each prereg
    if subprocess.run([sys.executable,
                       os.path.join(ROOT, "prereg_e13.py")]).returncode != 0:
        raise SystemExit("prereg_e13.py is not frozen. Nothing runs.")
    if not a.resume:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "prereg_e13.py"),
                            "--check-clean", RESULTS])
        if r.returncode != 0:
            raise SystemExit(
                "results_e13.jsonl already has E13 rows. Use --resume to "
                "complete missing cells of the frozen design; nothing else "
                "is a legitimate second submission.")

    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    todo = configs()
    if a.resume:
        have = done_cells()
        todo = [c for c in todo
                if (c["nl"], round(c["lam"], 6), c["seed"]) not in have]
        print(f"resume: {len(have)} cells present, {len(todo)} to run")
    if a.dry_run:
        for c in todo:
            print(f"  nl={c['nl']:<5} lam={c['lam']:<6} seed={c['seed']} "
                  f"init={c['gain']}")
        print(f"{len(todo)} runs")
        return 0
    if not todo:
        print("nothing to run; design complete")
        return 0

    try:
        import torch
        tv = torch.__version__
    except Exception:
        raise SystemExit("torch unavailable; E13 cannot run here")
    with open(os.path.join(LOG_DIR, "e13_host.txt"), "a") as f:
        f.write(f"{time.strftime('%FT%TZ', time.gmtime())} "
                f"host={platform.node()} python={platform.python_version()} "
                f"torch={tv} workers={a.workers} runs={len(todo)}\n")

    print(f"{len(todo)} runs, {a.workers} workers "
          f"(host {platform.node()}, torch {tv})")
    t0, fails = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, i, c): (i, c) for i, c in enumerate(todo)}
        ndone = 0
        for fut in as_completed(futs):
            idx, c, rc, dt = fut.result()
            ndone += 1
            tag = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"  [{ndone:3d}/{len(todo)}] nl={c['nl']:<5} "
                  f"lam={c['lam']:<6} s{c['seed']} {dt/60:5.1f}m {tag}",
                  flush=True)
            if rc != 0:
                fails.append((c, idx))
    print(f"\nwall-clock {(time.time() - t0)/3600:.2f} h")
    if fails:
        print(f"{len(fails)} runs FAILED (see logs). Re-run with --resume "
              f"after diagnosing; the scorer refuses incomplete arms.")
        return 1
    print("done. Score with:\n  python3 prereg_e13.py "
          "results/results/results_e13.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
