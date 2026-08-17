#!/usr/bin/env bash
# Verify byte for byte what actually landed on Myriad, and report whether any
# jobs are already queued or running.
#
# Why this exists: the first ship ran while this folder was on the iCloud
# Desktop, where a dataless placeholder can abort the transfer mid-way
# ("cp: read local ...: Operation canceled"). A partial ship is the exact
# failure that killed jobs 17949 and 17953. Size checks are not enough, so
# this compares checksums computed from the local files at run time.
#
# Usage: bash verify_myriad.sh <ucl_username>
set -euo pipefail
USER="${1:?usage: bash verify_myriad.sh <ucl_username>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
REMOTE="~/Scratch/lambda-audit"

FILES=(
  "$ROOT/harness/ppo_lambda.py"
  "$HERE/configs_pilot.txt"
  "$HERE/configs_capability.txt"
  "$HERE/pilot_array.qsub.sh"
  "$HERE/setup_myriad.sh"
)

md5of() {
  if command -v md5 >/dev/null 2>&1; then md5 -q "$1"        # macOS
  else md5sum "$1" | cut -d' ' -f1; fi                        # Linux
}

echo "=== LOCAL: computing checksums from real reads"
manifest="$(mktemp)"
for f in "${FILES[@]}"; do
  # cat through a pipe so a dataless placeholder cannot pass silently
  if ! cat -- "$f" >/dev/null 2>&1; then
    echo "  LOCAL FILE UNREADABLE: $f"
    echo "  Fix the local copy before trusting any comparison."
    exit 1
  fi
  printf '%s  %s\n' "$(md5of "$f")" "$(basename "$f")" >> "$manifest"
done
cat "$manifest" | sed 's/^/  /'

echo "=== REMOTE: comparing against ${REMOTE}"
scp -q "$manifest" "${USER}@myriad.rc.ucl.ac.uk:${REMOTE}/.expected_md5" 2>/dev/null || {
  echo "  Could not reach ${REMOTE}. Either the directory does not exist"
  echo "  (nothing was ever shipped) or SSH failed. Re-ship and retry."
  exit 1
}

ssh "${USER}@myriad.rc.ucl.ac.uk" "
  cd ${REMOTE} 2>/dev/null || { echo '  REMOTE DIR MISSING'; exit 1; }
  echo '--- checksum check (FAILED lines below are the problem):'
  md5sum -c .expected_md5 2>&1 | sed 's/^/    /' || true
  echo '--- files present:'
  ls -la *.py *.txt *.sh 2>/dev/null | awk '{printf \"    %8s  %s\n\", \$5, \$9}' || echo '    none'
  echo '--- venv built?'
  test -x venv/bin/python && echo '    yes' || echo '    NO, setup_myriad.sh has not run'
  echo '--- results so far:'
  wc -l results/results_lambda_pilot.jsonl 2>/dev/null | sed 's/^/    /' || echo '    no results file yet'
  ls results/curves_*.csv 2>/dev/null | wc -l | sed 's/^/    curves: /'
  echo '--- jobs currently queued or running:'
  qstat 2>/dev/null | sed 's/^/    /' || echo '    qstat unavailable'
  echo '--- past job logs:'
  ls logs/ 2>/dev/null | head -5 | sed 's/^/    /' || echo '    no logs dir'
  rm -f .expected_md5
"

cat <<'EOM'

=== HOW TO READ THIS
All five lines say OK, no jobs queued, no logs
    -> the ship landed clean. Proceed to setup and the capability job.
Any line says FAILED, or files are missing
    -> the ship was partial. Re-run ship_to_myriad.sh from ~/dev/lambda-audit,
       then run this again. Do not submit anything first.
Jobs already queued or running
    -> qdel them before re-shipping. A running task reading a half-written
       harness produces results that look real and are not.
Logs exist from an earlier submit
    -> read them. Anything already run against a partial ship is void.
EOM
