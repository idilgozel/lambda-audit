#!/bin/zsh
# E18 pre-freeze pilots (seed 101): the mlp from scratch under continuing training on
# GLEAM_g1_h0.05 at lambda 0.9 (E7: gain 4-6, E 2.6; k_AC 20.5, E_AC 1.33) and 1.0.
OUT=~/dev/lambda-audit/results/pilots_e17
while [ ! -f $OUT/DONE_NORESET ]; do sleep 20; done
cd $OUT/sgd/harness
run() { # lam
  nice -n 10 ~/dev/lambda-audit/.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.05 --arch mlp --gae-lambda $1 \
    --gamma 0.9975031223974601 --fix-sigma 0.3 --gleam-episode-len 10000000 --total-timesteps 15000000 \
    --seed 101 --torch-threads 1 --out-dir $OUT --results-name pilots_e18_mlp.jsonl \
    --run-name GLEAM_g1_h0.05_mlp_noreset_lam$1_s101 > $OUT/log_mlp_noreset_lam$1.txt 2>&1
}
run 0.9 & run 1.0 & wait
echo done > $OUT/DONE_MLP
