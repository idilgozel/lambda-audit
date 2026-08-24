#!/usr/bin/env bash
# Regenerate the provenance chronology in PROVENANCE.md section 2 from the
# committed cluster logs. The logs carry UTC start times and host lines written
# by Myriad's scheduler, and SGE job IDs increase monotonically across the whole
# cluster, so neither can be chosen or reordered by a submitter. This is the
# evidence that each preregistration freeze preceded the data it scores; the git
# history is not (see PROVENANCE.md section 1).
#
# Usage: bash analysis/provenance_chronology.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LOGS="$ROOT/results/logs"

[ -d "$LOGS" ] || { echo "no results/logs directory at $LOGS"; exit 2; }

# Job name -> the preregistration whose freeze governs it. Kept explicit rather
# than inferred, so a new experiment cannot silently appear ungoverned.
governing() {
  case "$1" in
    lam_cap|lam_pilot) echo "prereg_lambda_pilot.py" ;;
    lam_e6|lam_e9)     echo "prereg_phase2.py" ;;
    lam_e7)            echo "prereg_phase2.py + prereg_e7_bias.py" ;;
    lam_e10)           echo "prereg_e10.py" ;;
    lam_e11)           echo "prereg_e11.py" ;;
    lam_e12p|lam_e12)  echo "prereg_e12.py" ;;
    *)                 echo "UNGOVERNED -- add it to governing()" ;;
  esac
}

freeze_date() {
  local f="$ROOT/$1"
  [ -f "$f" ] || { echo "?"; return; }
  grep -m1 -o 'FROZEN [0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}' "$f" | awk '{print $2}'
}

printf '%-12s %-9s %6s  %-21s  %-21s  %-34s %s\n' \
  "job name" "SGE id" "tasks" "first start (UTC)" "last finish (UTC)" "governing freeze" "classes"

# Sort by first start time so the table reads chronologically and any job ID
# that is out of order with its start time is visible rather than hidden.
for j in $(ls "$LOGS"/*.o* 2>/dev/null | xargs -n1 basename | sed 's/\.o.*//' | sort -u); do
  first=$(grep -h '^starting' "$LOGS/$j".o* 2>/dev/null | sort | head -1 | sed 's/starting *: //')
  last=$(grep -h '^finished' "$LOGS/$j".o* 2>/dev/null | sort | tail -1 | sed 's/finished *: //')
  jid=$(ls "$LOGS/$j".o* 2>/dev/null | head -1 | sed 's/.*\.o\([0-9]*\)\..*/\1/')
  n=$(ls "$LOGS/$j".o* 2>/dev/null | wc -l | tr -d ' ')
  cls=$(grep -h '^host' "$LOGS/$j".o* 2>/dev/null \
        | sed 's/.*node-\([a-z0-9]*\)-.*/\1/' | sort -u | paste -sd, -)
  gov=$(governing "$j")
  # Report the freeze date of the first named prereg; multi-prereg jobs are
  # named in full in the governing column.
  fd=$(freeze_date "$(echo "$gov" | awk '{print $1}')")
  printf '%-12s %-9s %6s  %-21s  %-21s  %-34s %s\n' \
    "$j" "$jid" "$n" "${first:-?}" "${last:-?}" "$gov (frozen $fd)" "${cls:-?}"
done | sort -k4

echo
echo "Job IDs must increase with first start time. Any inversion above is a"
echo "finding, not a formatting artifact."
