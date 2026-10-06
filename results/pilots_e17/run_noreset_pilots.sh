#!/bin/zsh
# E17 pre-freeze pilots, set 4 (seed 101): the same cells with and without episode resets,
# SGD on the gain (eta 0.01). If the reset transient is the downward push, the no-reset
# arm must drift UP at (0.9, 30) toward the identity's equilibrium 57.8 and DOWN at (0.9, 80).
OUT=~/dev/lambda-audit/results/pilots_e17
while [ ! -f $OUT/DONE_MOB ]; do sleep 20; done
cd $OUT/sgd/harness
run() { # lam k0 eta episode_len tag
  nice -n 10 ~/dev/lambda-audit/.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.02 --arch linear --gae-lambda $1 \
    --gamma 0.999000499833375 --fix-sigma 0.3 --linear-gain-init $2 --linear-sgd-lr $3 --gleam-episode-len $4 \
    --total-timesteps 15000000 --seed 101 --torch-threads 1 --out-dir $OUT --results-name pilots_e17_noreset.jsonl \
    --run-name GLEAM_g1_h0.02_linearSGD$3_$5_lam$1_k$2_s101 > $OUT/log_$5_sgd$3_lam$1_k$2.txt 2>&1
}
run 0.9 30 0.01 10000000 noreset & run 0.9 80 0.01 10000000 noreset & run 0.9 30 0.01 0 reset & wait
echo done > $OUT/DONE_NORESET
