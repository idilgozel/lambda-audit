#!/usr/bin/env bash
# Pull results off Myriad and score them against the frozen preregistration.
# Run on the Mac.  Usage: bash myriad/collect.sh <ucl_username>
set -uo pipefail

USER_NAME="${1:-}"
[ -n "$USER_NAME" ] || { echo "usage: bash myriad/collect.sh <ucl_username>"; exit 2; }
HOST="myriad.rc.ucl.ac.uk"
REMOTE="Scratch/lambda-audit"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEST="$ROOT/results"
mkdir -p "$DEST"

mkdir -p "$HOME/.ssh"
CTL="$HOME/.ssh/cm-lambda-collect-$$"
trap 'ssh -o ControlPath="$CTL" -O exit "$USER_NAME@$HOST" >/dev/null 2>&1 || true' EXIT
ssh -o ControlMaster=yes -o ControlPath="$CTL" -o ControlPersist=20m \
    -o StrictHostKeyChecking=accept-new -fN "$USER_NAME@$HOST" \
  || { echo "FATAL: cannot connect"; exit 1; }
SSH=(ssh -o ControlPath="$CTL" "$USER_NAME@$HOST")

echo "=== queue state"
# Every job name deploy.sh can submit must appear here. A name missing from
# this filter makes a RUNNING job report as "no jobs queued", which is worse
# than no queue section at all: it reads as "the run is done, go collect".
# lam_e12 also matches lam_e12p by prefix; both are listed for legibility.
"${SSH[@]}" "qstat 2>/dev/null | grep -E 'lam_cap|lam_pilot|lam_e6|lam_e7|lam_e9|lam_e10|lam_e11|lam_e12p|lam_e12|lam_e14|job-ID|---' || echo '  no lambda-audit jobs queued or running'"

echo
echo "=== pulling results, curves and logs"
"${SSH[@]}" "cd ~/$REMOTE && tar czf - results logs 2>/dev/null" \
  | tar xzf - -C "$DEST" --strip-components=0
# Capture both statuses on the very next line. PIPESTATUS is rewritten by every
# subsequent command, including the test that reads it, so anything later sees
# a one-element array and blows up under set -u.
PULL_STATUS=("${PIPESTATUS[@]}")
# A silent partial transfer that then gets scored is worse than a loud failure.
if [ "${PULL_STATUS[0]}" != "0" ] || [ "${PULL_STATUS[1]}" != "0" ]; then
  echo "  WARNING: the pull did not complete cleanly"
  echo "  (remote tar=${PULL_STATUS[0]}, local tar=${PULL_STATUS[1]}). Everything"
  echo "  below describes whatever DID arrive and may be incomplete."
else
  echo "  into $DEST"
fi

PILOT_JSONL="$DEST/results/results_lambda_pilot.jsonl"
PHASE2_JSONL="$DEST/results/results_phase2.jsonl"
E10_JSONL="$DEST/results/results_e10.jsonl"
E11_JSONL="$DEST/results/results_e11.jsonl"
E12_JSONL="$DEST/results/results_e12.jsonl"
E14_JSONL="$DEST/results/results_e14.jsonl"

echo
echo "=== what arrived"
for f in "$PILOT_JSONL" "$PHASE2_JSONL" "$E10_JSONL" "$E11_JSONL" "$E12_JSONL" "$E14_JSONL"; do
  if [ -f "$f" ]; then
    printf '  %-28s %s rows\n' "$(basename "$f")" "$(wc -l < "$f" | tr -d ' ')"
  else
    printf '  %-28s ABSENT\n' "$(basename "$f")"
  fi
done
echo "  curves total: $(find "$DEST" -name 'curves_*.csv' 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "=== coverage by experiment (phase2)"
# stderr is NOT discarded here. A corrupt results file and an absent one are
# different situations and must not print the same reassuring message.
python3 - "$PHASE2_JSONL" <<'PY'
import json, sys, collections
try:
    fh = open(sys.argv[1])
except FileNotFoundError:
    print("  no phase2 results file yet")
    sys.exit(0)
rows = []
for n, line in enumerate(fh, 1):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError as exc:
        print(f"  CORRUPT: {sys.argv[1]} line {n} is not valid JSON ({exc})")
        print("  Do not score this file. Re-pull, or find the truncated write.")
        sys.exit(1)
if not rows:
    print("  phase2 results file is present but empty")
    sys.exit(0)

# Which experiment each env belongs to, so an absent array is visible as
# absent rather than as a small number nobody looks at.
E6 = {"GLEAM_g1_h0.05", "GLEAM_g1_h0.05_dec"}
E7 = {"GLEAM_g1_h0.02", "GLEAM_g2_h0.05", "GLEAM_g4_h0.05"}
E9 = {"PositionOnlyCartPoleEasy", "PositionOnlyCartPoleHard"}
EXPECT = {"E6": (E6, 80), "E7": (E7, 120), "E9": (E9, 160)}

by_env = collections.Counter(r.get("env", "") for r in rows)
for name, (envs, n_expected) in EXPECT.items():
    got = sum(by_env.get(e, 0) for e in envs)
    mark = "OK" if got >= n_expected else ("ABSENT" if got == 0 else "PARTIAL")
    print(f'  {name}: {got:4d} / {n_expected} rows  [{mark}]')
    for e in sorted(envs):
        print(f'      {e:28} {by_env.get(e, 0):4d}')

# Same key and the same normalisation as duplicate_cells in prereg_phase2.py,
# so this early warning and the scorer's refusal can never disagree.
dups = collections.Counter()
for r in rows:
    try:
        dups[(str(r.get("env", "")), str(r.get("arch", "")),
              round(float(r.get("lam")), 6), int(r.get("seed")))] += 1
    except (TypeError, ValueError):
        continue
bad = {k: n for k, n in dups.items() if n > 1}
if bad:
    print(f'  WARNING: {len(bad)} duplicated (env, arch, lambda, seed) cells.')
    print('  prereg_phase2.py will REFUSE to score. The offending cells:')
    for (env, arch, lam, seed), n in sorted(bad.items()):
        print(f'      {env:26} {arch:8} lam={lam:<6} seed={seed}  x{n}')
PY

echo
echo "=== node-class provenance"
# Cross-node floating point is not reproducible. Measured on 2026-08-14: of 40
# PositionOnlyCartPoleEasy cells run twice, the 33 that stayed on the d class
# (d00a and d00b interchangeably) reproduced bitwise, and the 7 that ran on
# t00a in one pass did not. So the letter class is the reproducibility
# boundary, the number after it is not, and which class a run landed on is
# part of its provenance.
python3 - "$DEST/logs" <<'PY'
import os, re, sys, collections
d = sys.argv[1]
if not os.path.isdir(d):
    print("  no logs directory yet")
    sys.exit(0)
host_re = re.compile(r'^host\s*:\s*node-([a-z])(\d{2}[a-z])-', re.M)
fam = collections.defaultdict(collections.Counter)
for fn in os.listdir(d):
    m = re.match(r'(lam_[a-z0-9]+)\.o', fn)
    if not m:
        continue
    try:
        txt = open(os.path.join(d, fn), errors="replace").read(4096)
    except OSError:
        continue
    h = host_re.search(txt)
    if h:
        fam[m.group(1)][h.group(1) + h.group(2)] += 1
if not fam:
    print("  logs directory holds no job output with a host line yet")
    sys.exit(0)
for name in sorted(fam):
    counts = fam[name]
    total = sum(counts.values())
    spread = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    letters = {k[0] for k in counts}
    flag = ("  SPANS %s CLASSES, runs are not bitwise comparable across them"
            % len(letters)) if len(letters) > 1 else ""
    print(f'  {name:12} {total:4d} logs   {spread}{flag}')
PY

echo
echo "=== preregistered gates: pilot"
if [ -f "$PILOT_JSONL" ] && [ "$(wc -l < "$PILOT_JSONL" | tr -d ' ')" -ge 240 ]; then
  python3 "$ROOT/prereg_lambda_pilot.py" "$PILOT_JSONL"
else
  echo "  not enough runs yet (needs the 240-task array)"
fi

echo
echo "=== E10 coverage"
if [ -f "$E10_JSONL" ]; then
  python3 - "$E10_JSONL" <<'PY2'
import json, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("  results_e10.jsonl is present but empty"); sys.exit(0)
by=collections.Counter((r.get("env",""), r.get("arch","")) for r in rows)
print(f"  {len(rows)} / 192 rows")
for k in sorted(by): print(f"      {k[0]:20} {k[1]:10} {by[k]:4d}")
d=collections.Counter()
for r in rows:
    try: d[(str(r.get("env")),str(r.get("arch")),round(float(r.get("lam")),6),int(r.get("seed")))]+=1
    except (TypeError,ValueError): pass
bad={k:n for k,n in d.items() if n>1}
if bad: print(f"  WARNING: {len(bad)} duplicated (env, arch, lambda, seed) cells")
PY2
else
  echo "  no results_e10.jsonl yet"
fi

echo
echo "=== preregistered gates: E10"
if [ -f "$E10_JSONL" ]; then
  python3 "$ROOT/prereg_e10.py" "$E10_JSONL" 2>&1 || true
else
  echo "  no E10 results file yet"
fi

echo
echo "=== preregistered gates: phase 2 (E6, E7, E9)"
if [ -f "$PHASE2_JSONL" ]; then
  python3 "$ROOT/prereg_phase2.py" "$PHASE2_JSONL"
else
  echo "  no phase2 results file yet"
fi

echo
echo "=== preregistered gates: E11 (BattleshipEasy)"
if [ -f "$E11_JSONL" ]; then
  python3 "$ROOT/prereg_e11.py" "$E11_JSONL" 2>&1 || true
else
  echo "  no results_e11.jsonl yet"
fi

echo
echo "=== E12 coverage (MineSweeperMedium)"
# E12 is submitted in TWO stages against ONE results file: a 2-run probe
# (mlp + gru128 at lambda 1.0, seed 1) whose gates are read BEFORE the 80-run
# grid. So a row count is ambiguous on its own and the stage is named here.
if [ -f "$E12_JSONL" ]; then
  python3 - "$E12_JSONL" <<'PY3'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("  results_e12.jsonl is present but empty"); sys.exit(0)
by = collections.Counter((r.get("arch", ""), r.get("lam")) for r in rows)
n = len(rows)
stage = ("probe (of 2)" if n <= 2 else
         "grid (of 80)" if n >= 3 else "unknown")
print(f"  {n} rows -- stage: {stage}")
for k in sorted(by, key=lambda t: (t[0], t[1] if t[1] is not None else -1)):
    print(f"      {k[0]:10} lam {str(k[1]):<6} {by[k]:4d}")
probe = [r for r in rows if r.get("lam") == 1.0 and r.get("seed") == 1]
have = {r.get("arch") for r in probe}
missing = {"mlp", "gru128"} - have
if missing:
    print(f"  probe INCOMPLETE: still waiting on {sorted(missing)} at lam 1.0 seed 1")
    print("  Gates cannot be read yet. Do NOT submit the grid.")
else:
    print("  probe complete: both arms present at lam 1.0 seed 1. Gates readable below.")
d = collections.Counter()
for r in rows:
    try:
        d[(str(r.get("arch")), round(float(r.get("lam")), 6), int(r.get("seed")))] += 1
    except (TypeError, ValueError):
        pass
bad = {k: c for k, c in d.items() if c > 1}
if bad:
    print(f"  WARNING: {len(bad)} duplicated (arch, lambda, seed) cells")
PY3
else
  echo "  no results_e12.jsonl yet"
fi

echo
echo "=== preregistered gates: E12 (read these BEFORE submitting the grid)"
if [ -f "$E12_JSONL" ]; then
  python3 "$ROOT/prereg_e12.py" "$E12_JSONL" 2>&1 || true
else
  echo "  no results_e12.jsonl yet"
fi

echo
echo "=== E14 coverage (PositionOnlyPendulum Easy / Hard)"
# E14 is submitted in stages against ONE results file: a 4-run probe (mlp +
# gru128 at lambda 1.0, seed 1, both variants) whose per-variant gates are
# read BEFORE each variant's grid (two arrays each, skipping the probe cells).
if [ -f "$E14_JSONL" ]; then
  python3 - "$E14_JSONL" <<'PY4'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("  results_e14.jsonl is present but empty"); sys.exit(0)
envs = ("PositionOnlyPendulumEasy", "PositionOnlyPendulumHard")
print(f"  {len(rows)} rows (probe 4, each variant's grid +78, full 160)")
for env in envs:
    by = collections.Counter((r.get("arch", ""), r.get("lam")) for r in rows if r.get("env") == env)
    n = sum(by.values())
    print(f"  {env}: {n} / 80 rows")
    for k in sorted(by, key=lambda t: (t[0], t[1] if t[1] is not None else -1)):
        print(f"      {k[0]:10} lam {str(k[1]):<6} {by[k]:4d}")
    probe = {r.get("arch") for r in rows
             if r.get("env") == env and r.get("lam") == 1.0 and r.get("seed") == 1}
    missing = {"mlp", "gru128"} - probe
    if missing:
        print(f"      probe INCOMPLETE for this variant: waiting on {sorted(missing)}; do NOT submit its grid")
    else:
        print(f"      probe complete for this variant; gates readable below")
d = collections.Counter()
for r in rows:
    try:
        d[(str(r.get("env")), str(r.get("arch")), round(float(r.get("lam")), 6), int(r.get("seed")))] += 1
    except (TypeError, ValueError):
        pass
bad = {k: c for k, c in d.items() if c > 1}
if bad:
    print(f"  WARNING: {len(bad)} duplicated (env, arch, lambda, seed) cells; prereg_e14.py will refuse")
sig = {r.get("fix_sigma") for r in rows}
print(f"  fix_sigma values present: {sorted(sig, key=str)}")
PY4
else
  echo "  no results_e14.jsonl yet"
fi

echo
echo "=== preregistered gates: E14 (read these BEFORE submitting a variant's grid)"
if [ -f "$E14_JSONL" ]; then
  python3 "$ROOT/prereg_e14.py" "$E14_JSONL" 2>&1 || true
else
  echo "  no results_e14.jsonl yet"
fi
