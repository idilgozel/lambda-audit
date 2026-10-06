#!/bin/zsh
# Set 8: critic swap. Adam linear arm, continuing training, lambda 0.9, the exact LSTD(lambda)
# aliased quadratic critic in the GAE. Prediction of identity (I): k -> k_AC(0.9) = 57.81 from 10, 30 and 80.
OUT=~/dev/lambda-audit/results/pilots_e17
while [ ! -f $OUT/DONE_KAC ]; do sleep 20; done
cd $OUT/sgd/harness
for k0 in 10 30 80; do
  nice -n 10 ~/dev/lambda-audit/.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.02 --arch linear --gae-lambda 0.9 \
    --gamma 0.999000499833375 --fix-sigma 0.3 --linear-gain-init $k0 --gleam-episode-len 10000000 --oracle-critic lstd --log-critic-fit 1 \
    --total-timesteps 15000000 --seed 101 --torch-threads 1 --out-dir $OUT --results-name pilots_e17_oracle.jsonl \
    --run-name GLEAM_g1_h0.02_linear_oracle_noreset_lam0.9_k${k0}_s101 > $OUT/log_oracle_k$k0.txt 2>&1 &
done
wait
echo done > $OUT/DONE_ORACLE
