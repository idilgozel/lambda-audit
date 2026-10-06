"""Run E14 locally, with the same discipline myriad/deploy.sh enforces.

Written 2026-10-05, after cluster access ended. E14 was frozen on 2026-09-11
for Myriad (prereg_e14.py; configs generated from it by gen_e14_configs.py);
nothing in the preregistration depends on where the runs execute, so this
file replaces deploy.sh/collect.sh with a process pool on one machine and
keeps every rule they had:

  * pre-flight: prereg_e14.py must be frozen and the config files must match
    it (gen_e14_configs.py --check); otherwise nothing runs.
  * probe first: the 4 lambda=1.0 seed-1 cells run and are SCORED before any
    grid cell exists; a variant's grid runs only if its three gates PASS in
    prereg_e14.py's own VERDICTS line.
  * no duplicate cells, ever: before launching a cell this runner reads the
    results file and refuses any (env, arch, lambda, seed) already present.
    The task -> (config line, seed) mapping is pilot_array.qsub.sh's, and the
    grid skips tasks 15 and 16 (the probe's cells) exactly as deploy.sh did.
  * provenance: one log per task with the same header lines the SGE script
    wrote (host, task, args, versions, start/finish), plus
    results/logs/e14_provenance.json with CPU, versions and git state. All
    E14 rows come from ONE machine and ONE CPU class, so the node-class caveat
    the paper carries for the cluster experiments does not apply to E14.

Usage (from the repo root; .venv is used if present):
    python3 analysis/run_e14.py --probe                 # 4 runs, then scores
    python3 analysis/run_e14.py --grid easy|hard|all    # gated per variant
    python3 analysis/run_e14.py --probe-then-grid       # unattended campaign
    python3 analysis/run_e14.py --status
Workers default to min(14, ncpu - 4); each run uses one torch thread.
"""
import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "harness")
RESULTS_DIR = os.path.join(ROOT, "results", "results")
LOG_DIR = os.path.join(ROOT, "results", "logs")
RESULTS = os.path.join(RESULTS_DIR, "results_e14.jsonl")
PY = (os.path.join(ROOT, ".venv", "bin", "python")
      if os.path.exists(os.path.join(ROOT, ".venv", "bin", "python")) else sys.executable)

sys.path.insert(0, ROOT)
import prereg_e14 as m  # noqa: E402

CFG = {
    "probe": os.path.join(ROOT, "myriad", "configs_e14_probe.txt"),
    "easy": os.path.join(ROOT, "myriad", "configs_e14_easy.txt"),
    "hard": os.path.join(ROOT, "myriad", "configs_e14_hard.txt"),
}
JOBNAME = {"probe": "lam_e14p", "easy": "lam_e14e", "hard": "lam_e14h"}
_print_lock = threading.Lock()


def say(*a):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------
# Pre-flight and bookkeeping
# ---------------------------------------------------------------------------


def preflight():
    r = subprocess.run([PY, os.path.join(ROOT, "prereg_e14.py")], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("FATAL: prereg_e14.py is not frozen. Nothing runs.\n" + r.stdout + r.stderr)
    r = subprocess.run([PY, os.path.join(ROOT, "analysis", "gen_e14_configs.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("FATAL: configs_e14_*.txt do not match prereg_e14.py. Nothing runs.\n" + r.stdout)
    say("pre-flight ok: prereg frozen, configs match")


def parse_cfg(path):
    return [l.strip() for l in open(path) if l.strip()]


def cfg_key(line, seed):
    """(env, arch, lambda, seed) of a config line, the duplicate-guard key."""
    toks = line.split()
    env = toks[toks.index("--env") + 1]
    arch = toks[toks.index("--arch") + 1]
    lam = round(float(toks[toks.index("--gae-lambda") + 1]), 6)
    return (env, arch, lam, int(seed))


def existing_keys():
    keys = set()
    if not os.path.exists(RESULTS):
        return keys
    for l in open(RESULTS):
        if not l.strip():
            continue
        r = json.loads(l)
        try:
            keys.add((r["env"], r["arch"], round(float(r["lam"]), 6), int(r["seed"])))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def tasks_for(mode):
    """(task_id, config line, seed) list, with pilot_array.qsub.sh's mapping."""
    lines = parse_cfg(CFG[mode])
    n = len(lines)
    if mode == "probe":
        ids = range(1, n + 1)                     # 1-4, all seed 1
    else:
        assert n == 16
        ids = list(range(1, 15)) + list(range(17, 81))   # skip 15, 16 = probe cells
    out = []
    for t in ids:
        idx = (t - 1) % n
        seed = (t - 1) // n + 1
        out.append((t, lines[idx], seed))
    return out


def provenance():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    info = {
        "written": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "cpu": sh("sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | grep 'Model name'"),
        "ncpu": os.cpu_count(),
        "python": sh(f"{PY} -c 'import sys; print(sys.version.split()[0])'"),
        "torch": sh(f"{PY} -c 'import torch; print(torch.__version__)'"),
        "numpy": sh(f"{PY} -c 'import numpy; print(numpy.__version__)'"),
        "git_commit": sh(f"git -C {ROOT} rev-parse HEAD"),
        "git_dirty": sh(f"git -C {ROOT} status --short"),
        "prereg_e14_md5": sh(f"md5 -q {ROOT}/prereg_e14.py 2>/dev/null || md5sum {ROOT}/prereg_e14.py | cut -d' ' -f1"),
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "e14_provenance.json")
    hist = []
    if os.path.exists(path):
        try:
            hist = json.load(open(path))
            if isinstance(hist, dict):
                hist = [hist]
        except Exception:
            hist = []
    hist.append(info)
    json.dump(hist, open(path, "w"), indent=1)
    say(f"provenance: {info['cpu']} x{info['ncpu']}, torch {info['torch']}, numpy {info['numpy']}, "
        f"commit {info['git_commit'][:8]}{' DIRTY' if info['git_dirty'] else ''}")
    return info


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_task(mode, task, line, seed, jobid):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, f"{JOBNAME[mode]}.o{jobid}.{task}")
    cmd = [PY, "ppo_lambda.py"] + line.split() + ["--seed", str(seed), "--torch-threads", "1",
                                                  "--out-dir", RESULTS_DIR]
    t0 = time.time()
    with open(log, "w") as f:
        f.write(f"host      : {platform.node()} (local, {platform.machine()})\n")
        f.write(f"task      : {task} -> config line {((task - 1) % len(parse_cfg(CFG[mode]))) + 1}, seed {seed}\n")
        f.write(f"args      : {line}\n")
        f.write(f"starting  : {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=HARNESS, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"finished  : {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  "
                f"rc={p.returncode}  wall={time.time() - t0:.0f}s\n")
    return task, p.returncode, time.time() - t0


def launch(mode, workers):
    tasks = tasks_for(mode)
    have = existing_keys()
    todo, skipped = [], []
    seen = set()
    for t, line, seed in tasks:
        k = cfg_key(line, seed)
        if k in have or k in seen:
            skipped.append((t, k))
        else:
            seen.add(k)
            todo.append((t, line, seed))
    if skipped:
        say(f"{mode}: REFUSING {len(skipped)} task(s) whose cell already exists in results_e14.jsonl "
            f"(e.g. task {skipped[0][0]} {skipped[0][1]}). Running only the missing cells.")
    if not todo:
        say(f"{mode}: nothing to run.")
        return 0
    # gru first: it is ~7x slower than mlp, so it decides the tail.
    todo.sort(key=lambda x: (0 if "--arch gru128" in x[1] else 1, x[0]))
    jobid = time.strftime("%Y%m%d%H%M")
    say(f"{mode}: launching {len(todo)} task(s) on {workers} worker(s), job id {jobid}")
    fails = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_task, mode, t, line, seed, jobid): (t, line, seed) for t, line, seed in todo}
        done = 0
        for fu in as_completed(futs):
            task, rc, wall = fu.result()
            done += 1
            t, line, seed = futs[fu]
            tag = f"{cfg_key(line, seed)}"
            if rc != 0:
                fails += 1
                say(f"{mode}: task {task} FAILED rc={rc} ({tag}); see logs/{JOBNAME[mode]}.o{jobid}.{task}")
            else:
                say(f"{mode}: task {task} done in {wall / 60:.1f} min ({tag})  [{done}/{len(todo)}]")
    say(f"{mode}: finished, {fails} failure(s)")
    return fails


def score():
    r = subprocess.run([PY, os.path.join(ROOT, "prereg_e14.py"), RESULTS], capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out)
    mm = re.findall(r"^VERDICTS:\s*(\{.*\})\s*$", out, re.M)
    verdicts = {}
    if mm:
        try:
            import ast
            verdicts = ast.literal_eval(mm[-1])
        except Exception:
            verdicts = {}
    return verdicts


def gates_pass(verdicts, short):
    return all(verdicts.get(f"G14{g}-{short}") == "PASS" for g in "abc")


def status():
    have = existing_keys()
    for mode in ("easy", "hard"):
        env = [e for e in m.ENVS if m.SHORT[e].lower() == mode][0]
        n = sum(1 for k in have if k[0] == env)
        print(f"{m.SHORT[env]}: {n}/80 cells present")
        missing = [(a, l, s) for a in m.ARCHS for l in m.LAMBDAS for s in m.SEEDS
                   if (env, a, round(l, 6), s) not in have]
        if missing and n:
            print(f"   missing {len(missing)}: " + ", ".join(f"{a}/{l}/s{s}" for a, l, s in missing[:12])
                  + (" ..." if len(missing) > 12 else ""))
    probe = [k for k in have if k[2] == 1.0 and k[3] == 1]
    print(f"probe cells present: {len(probe)}/4")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--grid", choices=["easy", "hard", "all"])
    ap.add_argument("--probe-then-grid", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(14, (os.cpu_count() or 4) - 4)))
    a = ap.parse_args()
    if a.status:
        status()
        return 0
    preflight()
    provenance()
    if a.probe or a.probe_then_grid:
        probe_keys = {cfg_key(line, seed) for _, line, seed in tasks_for("probe")}
        if probe_keys & existing_keys():
            say("probe cells already present; not re-running them (duplicate guard).")
        else:
            launch("probe", min(a.workers, 4))
        say("scoring the probe")
        v = score()
        if a.probe_then_grid:
            cleared = [s for s in ("easy", "hard") if gates_pass(v, s.capitalize())]
            blocked = [s for s in ("easy", "hard") if s not in cleared]
            say(f"gates: cleared {cleared}, not cleared {blocked}")
            for s in cleared:
                launch(s, a.workers)
            if cleared:
                say("scoring the grid")
                score()
        return 0
    if a.grid:
        if not os.path.exists(RESULTS):
            raise SystemExit("no results_e14.jsonl: run --probe first and read the gates.")
        v = score()
        want = ["easy", "hard"] if a.grid == "all" else [a.grid]
        for s in want:
            if not gates_pass(v, s.capitalize()):
                say(f"{s}: gates not PASS in the frozen scorer's output; grid NOT run "
                    f"({ {k: x for k, x in v.items() if k.endswith(s.capitalize())} }).")
                continue
            launch(s, a.workers)
        say("scoring")
        score()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
