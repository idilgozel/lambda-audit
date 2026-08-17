#!/usr/bin/env bash
# Pull pilot results, curves and SGE logs off Myriad.
# Usage: bash pull_pilot.sh <ucl_username>
set -euo pipefail
USER="${1:?usage: bash pull_pilot.sh <ucl_username>}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$DEST"

rsync -avz --prune-empty-dirs \
  --include='*/' \
  --include='curves_*.csv' \
  --include='results_lambda_pilot.jsonl' \
  --include='lam_pilot.o*' --include='lam_pilot.e*' \
  --include='lam_cap.o*' --include='lam_cap.e*' \
  --exclude='*' \
  "${USER}@myriad.rc.ucl.ac.uk:~/Scratch/lambda-audit/" "$DEST/"

echo "--- pulled into $DEST"
echo "--- runs recorded:"
find "$DEST" -name 'results_lambda_pilot.jsonl' -exec wc -l {} \;
echo "--- curves files (expect 242: 240 pilot + 2 capability):"
find "$DEST" -name 'curves_*.csv' | wc -l
echo "--- non-empty error files (should be none):"
find "$DEST" -name 'lam_pilot.e*' -size +0 2>/dev/null | head -20 || echo "  none"
