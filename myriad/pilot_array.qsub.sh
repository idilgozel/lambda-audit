#!/bin/bash -l
# SGE array job for the lambda-audit pilot on UCL Myriad.
#
# The environment is NOT hardcoded here. bootstrap_myriad.sh discovers a
# working recipe, proves it by training the harness, and writes env.sh. This
# script sources that. If the modules ever change, re-run the bootstrap and
# this file needs no edit.
#
# Capability check first (2 tasks, gate G1c):
#   qsub -N lam_cap -t 1-2 -v CFG=configs_capability.txt,SEED_OFFSET=100 pilot_array.qsub.sh
# Then the main array (48 config lines x 5 seeds = 240 tasks):
#   qsub -N lam_pilot -t 1-240 -v CFG=configs_pilot.txt pilot_array.qsub.sh
#
# One core per task on purpose. The networks are small, extra threads do not
# help, and single core tasks clear the queue far faster at this width.
# h_rt is 8h against a slowest measured task of 2.3h, which backfills better
# than a 24h request.
#$ -l h_rt=8:0:0
#$ -l mem=4G
#$ -pe smp 1
#$ -cwd
#$ -j y
#$ -o logs/

set -e
mkdir -p logs results

if [ ! -r env.sh ]; then
  echo "FATAL: env.sh missing. Run 'bash bootstrap_myriad.sh' on a login node first."
  exit 1
fi
. ./env.sh

NCFG=$(wc -l < "$CFG")
IDX=$(( (SGE_TASK_ID - 1) % NCFG + 1 ))
SEED=$(( (SGE_TASK_ID - 1) / NCFG + 1 + ${SEED_OFFSET:-0} ))
ARGS=$(sed -n "${IDX}p" "$CFG")

echo "host      : $(hostname)"
echo "task      : $SGE_TASK_ID -> config line $IDX, seed $SEED"
echo "args      : $ARGS"
python3 -c "import torch, numpy; print('torch     :', torch.__version__, '| numpy:', numpy.__version__)"
echo "starting  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python3 ppo_lambda.py $ARGS --seed $SEED --torch-threads 1 --out-dir results

echo "finished  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
