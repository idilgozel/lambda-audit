#!/bin/zsh
# Set 7: E10's exact initialisation (lambda 0.9, k0 = k_AC = 57.809930) under continuing training, Adam arm.
OUT=~/dev/lambda-audit/results/pilots_e17
while [ ! -f $OUT/DONE_CFIT ]; do sleep 20; done
cd $OUT/sgd/harness
for seed in 101 102; do
  nice -n 10 ~/dev/lambda-audit/.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.02 --arch linear --gae-lambda 0.9 \
    --gamma 0.999000499833375 --fix-sigma 0.3 --linear-gain-init 57.80993 --gleam-episode-len 10000000 --log-critic-fit 1 \
    --total-timesteps 15000000 --seed $seed --torch-threads 1 --out-dir $OUT --results-name pilots_e17_criticfit.jsonl \
    --run-name GLEAM_g1_h0.02_linear_noreset_kAC_lam0.9_k57.80993_s$seed > $OUT/log_kac_noreset_s$seed.txt 2>&1 &
done
wait
echo done > $OUT/DONE_KAC
