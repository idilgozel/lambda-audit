#!/usr/bin/env bash
# ONE COMMAND deploy for the lambda audit. Run this on the Mac.
#
#   bash myriad/deploy.sh ucapgoz              ship, bootstrap, verify
#   bash myriad/deploy.sh ucapgoz --capability ...and submit the capability job
#   bash myriad/deploy.sh ucapgoz --pilot      ...and submit the full 240-task array
#
# You are asked for your password ONCE. All later connections reuse it via an
# SSH control socket.
#
# Why tar over ssh instead of scp: one stream, one authentication, no per-file
# path resolution, and it cannot half-succeed the way a multi-file scp can.
# Every local file is read through a pipe first, which catches iCloud
# placeholders that report a size but hold no data.

set -uo pipefail

USER_NAME="${1:-}"
MODE="${2:-}"
if [ -z "$USER_NAME" ]; then
  echo "usage: bash myriad/deploy.sh <ucl_username> [--capability|--pilot]"
  exit 2
fi
HOST="myriad.rc.ucl.ac.uk"
REMOTE="Scratch/lambda-audit"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
say "1/6  local files"
# --------------------------------------------------------------------------
FILES=(
  "harness/ppo_lambda.py"
  "harness/envs_native.py"
  "harness/gleam_native.py"
  "harness/gleam_instances.npz"
  "myriad/configs_pilot.txt"
  "myriad/configs_capability.txt"
  "myriad/configs_e6.txt"
  "myriad/configs_e7.txt"
  "myriad/configs_e9.txt"
  "myriad/configs_e10.txt"
  "myriad/configs_e11.txt"
  "myriad/configs_e12_probe.txt"
  "myriad/configs_e12.txt"
  "myriad/pilot_array.qsub.sh"
  "myriad/bootstrap_myriad.sh"
)
if command -v brctl >/dev/null 2>&1; then brctl download "$ROOT" >/dev/null 2>&1 || true; fi

# The preregistration must be frozen before anything ships. Same wiring as the
# pilot's guard in ship_to_myriad.sh: the file is RUN, not read, so a sentinel
# left in place stops the launch instead of being noticed later.
if [ -f "$ROOT/prereg_phase2.py" ]; then
  if python3 "$ROOT/prereg_phase2.py" >/dev/null 2>&1; then
    echo "     ok  prereg_phase2.py is frozen"
  else
    echo
    echo "FATAL: prereg_phase2.py is not frozen. Nothing shipped."
    python3 "$ROOT/prereg_phase2.py" 2>&1 | tail -5
    exit 1
  fi
fi
for f in "${FILES[@]}"; do
  p="$ROOT/$f"
  [ -e "$p" ] || die "missing locally: $f"
  bytes=$(cat -- "$p" 2>/dev/null | wc -c | tr -d ' ') || die "unreadable (dataless placeholder?): $f"
  [ "${bytes:-0}" -gt 50 ] || die "suspiciously small ($bytes bytes): $f"
  printf '     ok  %7d  %s\n' "$bytes" "$f"
done

# --------------------------------------------------------------------------
say "2/6  opening one SSH connection (password asked once)"
# --------------------------------------------------------------------------
mkdir -p "$HOME/.ssh"
CTL="$HOME/.ssh/cm-lambda-audit-$$"
cleanup() { ssh -o ControlPath="$CTL" -O exit "$USER_NAME@$HOST" >/dev/null 2>&1 || true; }
trap cleanup EXIT
ssh -o ControlMaster=yes -o ControlPath="$CTL" -o ControlPersist=60m \
    -o StrictHostKeyChecking=accept-new -fN "$USER_NAME@$HOST" \
  || die "could not connect to $HOST as $USER_NAME"
SSH=(ssh -o ControlPath="$CTL" "$USER_NAME@$HOST")
echo "     connected"

# --------------------------------------------------------------------------
say "3/6  shipping (single tar stream, flat layout)"
# --------------------------------------------------------------------------
"${SSH[@]}" "mkdir -p ~/$REMOTE/logs ~/$REMOTE/results" || die "could not create ~/$REMOTE"

# macOS xattrs, 2026-08-18. Files written to this Mac by tooling carry
# com.apple.provenance. bsdtar encodes every xattr as a pax extended header;
# GNU tar on Myriad then prints "Ignoring unknown extended header keyword" for
# each one, and depending on version and --warning defaults can exit non-zero,
# which this pipeline correctly read as a failed transfer and incorrectly
# reported without saying which end failed. Strip them at the source where the
# local tar supports it, tell the remote tar the keyword is noise, and report
# each side's status separately so the next failure names itself.
TAR_FLAGS=()
_tarhelp="$(tar --help 2>&1 || true)"
case "$_tarhelp" in *--no-xattrs*) TAR_FLAGS+=(--no-xattrs) ;; esac
case "$_tarhelp" in *--no-mac-metadata*) TAR_FLAGS+=(--no-mac-metadata) ;; esac
export COPYFILE_DISABLE=1

# transform strips the leading directories so everything lands flat, which is
# what ppo_lambda.py expects (it imports envs_native from its own directory)
tar ${TAR_FLAGS+"${TAR_FLAGS[@]}"} -czf - -C "$ROOT" "${FILES[@]}" \
  | "${SSH[@]}" "cd ~/$REMOTE && tar xzf - --strip-components=1 \
                 --warning=no-unknown-keyword && chmod +x *.sh"
XFER=("${PIPESTATUS[@]}")
if [ "${XFER[0]}" != "0" ] || [ "${XFER[1]}" != "0" ]; then
  echo
  echo "     diagnosing (local tar=${XFER[0]}, remote extract=${XFER[1]})..."
  "${SSH[@]}" "cd ~/$REMOTE 2>/dev/null && echo '     remote dir OK' || echo '     REMOTE DIR MISSING'; \
               echo -n '     quota: '; lquota 2>/dev/null | tail -3 || df -h ~ | tail -1" || true
  die "transfer failed (local tar=${XFER[0]}, remote extract=${XFER[1]}).
     local non-zero  -> a source file is unreadable, or the local tar rejected
                        a flag. Re-run and read the tar error above.
     remote non-zero -> most often Scratch quota. The audit has written ~1000
                        runs of curves there; check the quota line above and
                        clear old curves on Myriad if it is full."
fi
echo "     sent ${#FILES[@]} files"

# --------------------------------------------------------------------------
say "4/6  verifying checksums on Myriad"
# --------------------------------------------------------------------------
manifest="$(mktemp)"
for f in "${FILES[@]}"; do
  if command -v md5 >/dev/null 2>&1; then h=$(md5 -q "$ROOT/$f"); else h=$(md5sum "$ROOT/$f" | cut -d' ' -f1); fi
  printf '%s  %s\n' "$h" "$(basename "$f")" >> "$manifest"
done
"${SSH[@]}" "cat > ~/$REMOTE/.expected_md5" < "$manifest"
if ! "${SSH[@]}" "cd ~/$REMOTE && md5sum -c .expected_md5" ; then
  die "checksum mismatch after transfer. Re-run this script."
fi
rm -f "$manifest"

# --------------------------------------------------------------------------
say "5/6  bootstrapping the environment on Myriad (a few minutes)"
# --------------------------------------------------------------------------
if ! "${SSH[@]}" "cd ~/$REMOTE && bash -l bootstrap_myriad.sh"; then
  die "bootstrap failed. The output above says which step. Nothing was submitted."
fi

# --------------------------------------------------------------------------
say "6/6  submission"
# --------------------------------------------------------------------------
# Run one remote command that must answer with a single non-negative integer.
# Prints the integer and returns 0, or returns 1 having printed nothing. A
# guard built on this can therefore distinguish "the answer is zero" from "I
# could not get an answer", which is the distinction that matters: a probe that
# silently reads a failed connection as zero is not a guard at all.
probe_count() {
  local out lines
  out=$("${SSH[@]}" "$1" 2>/dev/null) || return 1
  # Exactly one non-empty line, or we do not understand the answer. Without
  # this, a two-line reply like "0\n0" collapses to "00" and reads as zero.
  lines=$(printf '%s\n' "$out" | grep -c '[^[:space:]]') || lines=0
  [ "$lines" -eq 1 ] || return 1
  out=$(printf '%s' "$out" | tr -d '[:space:]')
  case "$out" in
    '' | *[!0-9]*) return 1 ;;
  esac
  printf '%s' "$out"
}

submit() {
  local name="$1" cfg="$2" range="$3" extra="$4"
  shift 4
  local envs=("$@")          # env ids this job writes, may be empty
  local jid

  # Double-submission guard, added 2026-08-14, hardened the same day after an
  # audit found the first version fail-open. Every array appends to one results
  # file, so a second copy of a job doubles every cell it writes and the
  # prereg's duplicate guard then refuses to score anything at all. The hazard
  # is live precisely when a submission is uncertain, which is the moment you
  # are most likely to resubmit, so a probe that cannot reach Myriad must stop
  # the launch rather than wave it through.
  local queued prior rows e
  queued=$(probe_count "qstat 2>/dev/null | awk -v n=$name '\$3==n' | wc -l") \
    || die "could not read the Myriad queue, so I cannot tell whether $name is
     already running. Nothing submitted. Re-run once the connection is healthy;
     submitting blind is how a results file gets duplicated cells."
  if [ "$queued" -gt 0 ]; then
    die "$name already has $queued task(s) in the queue. Nothing submitted.
     Let it finish, or 'qdel' it first if you mean to replace it."
  fi

  # find, not ls, because a glob that matches nothing behaves differently under
  # nullglob and failglob and this runs in the user's remote shell. Any task's
  # output counts, not just task 1: an array can die after task 40 has already
  # appended rows, and that is exactly the case worth catching.
  prior=$(probe_count "find ~/$REMOTE/logs -maxdepth 1 -name '$name.o*' 2>/dev/null | wc -l") \
    || die "could not list ~/$REMOTE/logs on Myriad. Nothing submitted."
  if [ "$prior" -gt 0 ]; then
    die "$name has already run: ~/$REMOTE/logs holds $prior output file(s).
     Re-running would duplicate cells in results_phase2.jsonl.
     Collect and score what is there first."
  fi

  # The check that actually protects the file, rather than a proxy for it. If
  # any row for an env this job writes is already present, stop. This survives
  # the two ways the log check can be defeated: a job resubmitted under a
  # different name, and logs moved aside while the rows they produced remain.
  for e in ${envs+"${envs[@]}"}; do
    rows=$(probe_count "grep -c '\"env\": \"$e\"' ~/$REMOTE/results/results_phase2.jsonl 2>/dev/null || echo 0") \
      || die "could not inspect results_phase2.jsonl on Myriad. Nothing submitted."
    if [ "$rows" -gt 0 ]; then
      die "results_phase2.jsonl on Myriad already holds $rows row(s) for $e,
     which is one of the environments $name writes. Submitting would duplicate
     every one of them. Collect and score what is there first."
    fi
  done
  echo "     safe to submit: $name is not queued, has no logs, and none of its"
  echo "     environments have rows in results_phase2.jsonl yet"

  jid=$("${SSH[@]}" "cd ~/$REMOTE && qsub -terse -N $name -t $range -v CFG=$cfg$extra pilot_array.qsub.sh" | head -1 | cut -d. -f1)
  [ -n "$jid" ] || die "qsub produced no job id"
  echo "     submitted $name as job $jid"
  echo "     waiting 120s then reading the first task's output"
  sleep 120
  "${SSH[@]}" "cd ~/$REMOTE && qstat | head -20; echo '--- first task output:'; tail -n 25 logs/$name.o$jid.1 2>/dev/null || echo '(no output file yet, task still queued)'"
  echo
  echo "     job id: $jid"
}

case "$MODE" in
  --e6)
    # The minimal pair, and the kill test. 2 arms x 8 lambda x 5 seeds.
    "${SSH[@]}" "cd ~/$REMOTE && python3 -c \"
import gleam_native as g
print('  gleam_native loaded on the cluster, numpy only')
for n in g.instance_names(): print('   ', n, g.baselines(n))
\"" || die "gleam_native failed to load on Myriad. Nothing submitted."
    submit lam_e6 configs_e6.txt 1-80 "" GLEAM_g1_h0.05 GLEAM_g1_h0.05_dec
    cat <<'EOM'

     E6 decides the kill criterion in phase2-design.md section 4.
     When it finishes:
         bash myriad/collect.sh <username>
         python3 prereg_phase2.py results/results/results_phase2.jsonl
     E7 runs only if E6 passes.
EOM
    ;;
  --e7)
    # 3 instances x 8 lambda x 5 seeds. GLEAM_g1_h0.05 is NOT here: E6 already
    # ran it coupled at these seeds and its rows are the E7 data for that
    # instance. Re-running it would duplicate every cell.
    submit lam_e7 configs_e7.txt 1-120 "" GLEAM_g1_h0.02 GLEAM_g2_h0.05 GLEAM_g4_h0.05
    cat <<'EOM'

     E7 is the quantitative law. It is CONDITIONAL on E6 passing.
     Scoring:
         python3 prereg_phase2.py results/results/results_phase2.jsonl
EOM
    ;;
  --e10)
    # E10: linear + linearb at k_AC. Writes results_e10.jsonl, so no env list:
    # the phase2-file guard does not apply and must not false-positive on E6.
    if ! python3 "$ROOT/prereg_e10.py" >/dev/null 2>&1; then
      die "prereg_e10.py is not frozen. Nothing submitted."
    fi
    if ! python3 "$ROOT/prereg_e10.py" --check-clean "$ROOT/results/results/results_e10.jsonl"; then
      die "E10 data already exists locally. Collect and score before resubmitting."
    fi
    submit lam_e10 configs_e10.txt 1-192 ""
    cat <<'EOM'

     E10 is the in-domain test of the cost law (linear arm) and the
     perturbation-decay test of the mechanism (linearb arm). Scoring:
         python3 prereg_e10.py results/results/results_e10.jsonl
EOM
    ;;
  --e12probe)
    # THE PROBE COMES FIRST. E11 ran its capability gates after an 80-run grid
    # and spent 119 core-hours learning what 2 runs would have shown. Read this
    # output, check it against prereg_e12.py's G12a/G12b/G12c, and only then
    # submit --e12.
    if ! python3 "$ROOT/prereg_e12.py" >/dev/null 2>&1; then
      die "prereg_e12.py is not frozen. Nothing submitted."
    fi
    submit lam_e12p configs_e12_probe.txt 1-2 ""
    cat <<'EOM'

     Capability probe only: mlp and gru128 at lambda 1.0, seed 1.
     Gates to clear before the grid (prereg_e12.py):
         G12a mlp >= -0.30    G12b gru >= -0.10    G12c gap >= 0.05
     Collect, then:  python3 prereg_e12.py results/results/results_e12.jsonl
EOM
    ;;
  --e12)
    if ! python3 "$ROOT/prereg_e12.py" >/dev/null 2>&1; then
      die "prereg_e12.py is not frozen. Nothing submitted."
    fi
    submit lam_e12 configs_e12.txt 1-80 ""
    cat <<'EOM'

     E12 grid. Scoring:
         python3 prereg_e12.py results/results/results_e12.jsonl
EOM
    ;;
  --e11)
    # E11: BattleshipEasy, the closed-loop POPGym arm. Writes results_e11.jsonl.
    if ! python3 "$ROOT/prereg_e11.py" >/dev/null 2>&1; then
      die "prereg_e11.py is not frozen. Nothing submitted."
    fi
    if ! python3 "$ROOT/prereg_e11.py" --check-clean "$ROOT/results/results/results_e11.jsonl"; then
      die "E11 data already exists locally. Collect and score before resubmitting."
    fi
    submit lam_e11 configs_e11.txt 1-80 ""
    cat <<'EOM'

     E11 tests the scope law's positive side on a third-party benchmark.
     Scoring:
         python3 prereg_e11.py results/results/results_e11.jsonl
EOM
    ;;
  --e9)
    submit lam_e9 configs_e9.txt 1-160 "" PositionOnlyCartPoleEasy PositionOnlyCartPoleHard
    cat <<'EOM'

     E9 is corroboration on PositionOnlyCartPole. Scoring for E9 is added by
     dated amendment to prereg_phase2.py before these runs are read.
EOM
    ;;
  --capability)
    submit lam_cap configs_capability.txt 1-2 ",SEED_OFFSET=100"
    cat <<'EOM'

     The capability job decides gate G1c. It takes roughly 2.5 hours.
     When it finishes:
         bash myriad/collect.sh <username>
     G1c passes if either run shows deployed_return of 0.5 or more.
EOM
    ;;
  --pilot)
    submit lam_pilot configs_pilot.txt 1-240 ""
    cat <<'EOM'

     The full array is running. Collect results with:
         bash myriad/collect.sh <username>
EOM
    ;;
  *)
    cat <<EOM
     Nothing submitted (no flag given). The cluster is ready.

     Phase 1, the pilot (complete, kept for reproduction):
         bash myriad/deploy.sh $USER_NAME --capability
         bash myriad/deploy.sh $USER_NAME --pilot

     Phase 2, in order. E6 first, it is the kill test:
         bash myriad/deploy.sh $USER_NAME --e6      80 runs
         bash myriad/deploy.sh $USER_NAME --e7     120 runs, only if E6 passes
         bash myriad/deploy.sh $USER_NAME --e9     160 runs
         bash myriad/deploy.sh $USER_NAME --e10    192 runs, after E7 is scored
         bash myriad/deploy.sh $USER_NAME --e11     80 runs, Battleship closed-loop
         bash myriad/deploy.sh $USER_NAME --e12probe  2 runs, capability probe FIRST
         bash myriad/deploy.sh $USER_NAME --e12     80 runs, only after the probe clears
EOM
    ;;
esac

echo
echo "=== done"
