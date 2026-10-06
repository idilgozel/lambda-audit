#!/bin/zsh
# E17 pre-freeze pilots, set 3 (seed 101): does the Adam linear arm move toward the exact
# equilibrium k_AC(lambda) when started away from it, and does that depend on the
# bootstrap horizon 1/(1 - beta lambda) relative to the 128-step rollout?
OUT=~/dev/lambda-audit/results/pilots_e17
while [ ! -f $OUT/DONE_SGD ]; do sleep 20; done
cd ~/dev/lambda-audit/harness
run() { # lam k0
  nice -n 10 ../.venv/bin/python ppo_lambda.py --env GLEAM_g1_h0.02 --arch linear --gae-lambda $1 \
    --gamma 0.999000499833375 --fix-sigma 0.3 --linear-gain-init $2 --total-timesteps 15000000 \
    --seed 101 --torch-threads 1 --out-dir $OUT --results-name pilots_e17_mobility.jsonl \
    --run-name GLEAM_g1_h0.02_linear_lam$1_k$2_s101 > $OUT/log_mob_lam$1_k$2.txt 2>&1
}
run 0.9 30 & run 0.9 80 & run 0.99 20 & wait
run 0.99 55 & run 0.995 12 & run 0.995 40 & wait
run 0.98 25 & run 0.98 70 & wait
echo done > $OUT/DONE_MOB
