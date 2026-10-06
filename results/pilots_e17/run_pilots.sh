#!/bin/zsh
# E17 pre-freeze pilots (seed 101, never a registered seed). Drift direction of the
# one-parameter linear arm from k0 at three lambdas on GLEAM_g1_h0.02.
cd ~/dev/lambda-audit/harness
OUT=~/dev/lambda-audit/results/pilots_e17
run() { # lam k0
  nice -n 10 ../.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.02 --arch linear --gae-lambda $1 \
    --gamma 0.999000499833375 --fix-sigma 0.3 --linear-gain-init $2 --total-timesteps 15000000 \
    --seed 101 --torch-threads 1 --out-dir $OUT --results-name pilots_e17.jsonl \
    --run-name GLEAM_g1_h0.02_linear_lam$1_k$2_s101 > $OUT/log_lam$1_k$2.txt 2>&1
}
run 0.9959 10.0 & run 0.9959 4.0 & wait
run 0.9930 10.0 & run 0.9975 10.0 & wait
run 0.9966 10.0 & run 0.9975 4.0 & wait
echo done > $OUT/DONE
