#!/bin/zsh
# Set 6: what curvature did the critic learn? linear arm (0.9, 30) no-reset Adam, seed 103;
# mlp g1_h0.05 lambda 0.9 no-reset, seed 102. Both with --log-critic-fit 1.
OUT=~/dev/lambda-audit/results/pilots_e17
while [ ! -f $OUT/DONE_NORESET2 ]; do sleep 20; done
cd $OUT/sgd/harness
nice -n 10 ~/dev/lambda-audit/.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.02 --arch linear --gae-lambda 0.9 \
  --gamma 0.999000499833375 --fix-sigma 0.3 --linear-gain-init 30 --gleam-episode-len 10000000 --log-critic-fit 1 \
  --total-timesteps 15000000 --seed 103 --torch-threads 1 --out-dir $OUT --results-name pilots_e17_criticfit.jsonl \
  --run-name GLEAM_g1_h0.02_linear_noreset_cfit_lam0.9_k30_s103 > $OUT/log_cfit_linear.txt 2>&1 &
nice -n 10 ~/dev/lambda-audit/.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.05 --arch mlp --gae-lambda 0.9 \
  --gamma 0.9975031223974601 --fix-sigma 0.3 --gleam-episode-len 10000000 --log-critic-fit 1 \
  --total-timesteps 15000000 --seed 102 --torch-threads 1 --out-dir $OUT --results-name pilots_e17_criticfit.jsonl \
  --run-name GLEAM_g1_h0.05_mlp_noreset_cfit_lam0.9_s102 > $OUT/log_cfit_mlp.txt 2>&1 &
wait
echo done > $OUT/DONE_CFIT
