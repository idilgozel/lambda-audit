#!/bin/zsh
# Set 5: replication of the no-reset result with a second seed and at a start where the exact
# aliased gradient is large and UP (lambda 0.9, k0 10: ghat = -1.97e-3).
OUT=~/dev/lambda-audit/results/pilots_e17
while [ ! -f $OUT/DONE_MLP ]; do sleep 20; done
cd $OUT/sgd/harness
run() { # lam k0 eta seed
  extra=""; [ "$3" != "0" ] && extra="--linear-sgd-lr $3"
  nice -n 10 ~/dev/lambda-audit/.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.02 --arch linear --gae-lambda $1 \
    --gamma 0.999000499833375 --fix-sigma 0.3 --linear-gain-init $2 $extra --gleam-episode-len 10000000 \
    --total-timesteps 15000000 --seed $4 --torch-threads 1 --out-dir $OUT --results-name pilots_e17_noreset.jsonl \
    --run-name GLEAM_g1_h0.02_linear_opt$3_noreset_lam$1_k$2_s$4 > $OUT/log_noreset2_opt$3_lam$1_k$2_s$4.txt 2>&1
}
run 0.9 10 0 101 & run 0.9 10 0 102 & run 0.9 30 0.01 102 & wait
echo done > $OUT/DONE_NORESET2
