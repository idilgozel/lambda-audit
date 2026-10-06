#!/bin/zsh
# E17 pre-freeze pilots, SGD-on-the-gain vehicle (seed 101). Patched harness COPY in results/pilots_e17/sgd/harness.
OUT=~/dev/lambda-audit/results/pilots_e17
while [ ! -f $OUT/DONE ]; do sleep 20; done
cd $OUT/sgd/harness
run() { # lam k0 eta
  nice -n 10 ~/dev/lambda-audit/.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.02 --arch linear --gae-lambda $1 \
    --gamma 0.999000499833375 --fix-sigma 0.3 --linear-gain-init $2 --linear-sgd-lr $3 --total-timesteps 15000000 \
    --seed 101 --torch-threads 1 --out-dir $OUT --results-name pilots_e17_sgd.jsonl \
    --run-name GLEAM_g1_h0.02_linearSGD$3_lam$1_k$2_s101 > $OUT/log_sgd$3_lam$1_k$2.txt 2>&1
}
run 0.9930 10.5 0.01 & run 0.9966 10.5 0.01 & wait
run 0.9930 10.5 0.003 & run 0.9966 10.5 0.003 & wait
echo done > $OUT/DONE_SGD
