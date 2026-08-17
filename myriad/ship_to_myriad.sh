#!/usr/bin/env bash
# Ship the pilot package to Myriad, with the pre-flight and post-flight checks
# that exist because jobs 17949 and 17953 died on all 40 tasks when an scp
# silently failed to land the config file.
#
# Usage: bash ship_to_myriad.sh <ucl_username>
set -euo pipefail
USER="${1:?usage: bash ship_to_myriad.sh <ucl_username>}"
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

echo "=== PRE-FLIGHT: every file must exist and its bytes must be readable here"
echo "    (this catches iCloud placeholders with no local data)"
# This project lived under a synced Desktop (now ~/dev), and any of these can
# still be a macOS
# "dataless" placeholder: the directory entry carries the real size but the
# data is still in iCloud. `[ -r "$f" ]` is only a permission check and
# `wc -c < "$f"` is answered from stat metadata, so BOTH pass on a placeholder
# and the failure only surfaces when scp asks for the bytes:
#   cp: read local "harness/ppo_lambda.py": Operation canceled
# Pull the data down first, then count bytes through a pipe so the read is real.
if command -v brctl >/dev/null 2>&1; then
  brctl download "$ROOT" >/dev/null 2>&1 || true
fi
fail=0
for f in "${FILES[@]}"; do
  if [ ! -r "$f" ]; then
    echo "  UNREADABLE $f"
    fail=1
    continue
  fi
  if ! bytes=$(cat -- "$f" | wc -c | tr -d ' '); then
    echo "  DATALESS OR UNREADABLE (iCloud placeholder not materialized) $f"
    fail=1
    continue
  fi
  if [ "$bytes" -lt 100 ]; then
    echo "  SUSPICIOUSLY SMALL ($bytes bytes) $f"
    fail=1
  else
    echo "  ok  $(printf '%7d' "$bytes")  $f"
  fi
done
[ "$fail" -eq 0 ] || { echo "PRE-FLIGHT FAILED. Nothing shipped."; exit 1; }

echo "=== PRE-FLIGHT: preregistration must be frozen"
if python3 "$ROOT/prereg_lambda_pilot.py" >/dev/null 2>&1; then
  echo "  ok  prereg has no sentinels left"
else
  echo "  PREREG NOT FROZEN. Fill the sentinel TODOs and commit before launch."
  python3 "$ROOT/prereg_lambda_pilot.py" 2>&1 | tail -3
  exit 1
fi

echo "=== SHIPPING"
ssh "${USER}@myriad.rc.ucl.ac.uk" "mkdir -p ${REMOTE}/logs ${REMOTE}/results"
scp "${FILES[@]}" "${USER}@myriad.rc.ucl.ac.uk:${REMOTE}/"

echo "=== POST-FLIGHT: confirm the files actually landed"
ssh "${USER}@myriad.rc.ucl.ac.uk" "
  cd ${REMOTE}
  echo '--- line counts'
  wc -l ppo_lambda.py configs_pilot.txt pilot_array.qsub.sh
  echo '--- config lines (expect 48 pilot, 2 capability)'
  test \$(wc -l < configs_pilot.txt) -eq 48 && echo '  ok 48 configs' || echo '  WRONG CONFIG COUNT'
  test \$(wc -l < configs_capability.txt) -eq 2 && echo '  ok 2 capability configs' || echo '  WRONG CAPABILITY COUNT'
  echo '--- harness flags present'
  grep -q 'gae-lambda' ppo_lambda.py && echo '  ok --gae-lambda' || echo '  MISSING --gae-lambda'
  grep -q 'stack64' ppo_lambda.py && echo '  ok stack64 arch' || echo '  MISSING stack64'
  grep -q 'gap_lambda' ppo_lambda.py && echo '  ok gap diagnostics' || echo '  MISSING gap diagnostics'
"

cat <<'EOM'

=== NEXT
1. On Myriad, run setup once:   bash setup_myriad.sh
   The smoke test must print deployed_return above 0.9. Do not submit otherwise.
2. CAPABILITY CHECK FIRST (gate G1c pre-array protocol, ~3 h):
      qsub -N lam_cap -t 1-2 -v CFG=configs_capability.txt,SEED_OFFSET=100 pilot_array.qsub.sh
   Two minutes later:  tail ~/Scratch/lambda-audit/logs/lam_cap.o<JOBID>.1
   Verdict: at least one of the two runs must reach deployed_return >= 0.5.
   If NEITHER does, do NOT submit the array. The transition env swaps to
   RepeatPreviousK16 per the frozen contingency in prereg_lambda_pilot.py.
3. Main array:
      qsub -N lam_pilot -t 1-240 -v CFG=configs_pilot.txt pilot_array.qsub.sh
4. TWO MINUTES LATER, tail the first output file:
      tail ~/Scratch/lambda-audit/logs/lam_pilot.o<JOBID>.1
   This step is not optional. It is the whole lesson of the 40 task silent failure.
EOM
