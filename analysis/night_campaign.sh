#!/bin/zsh
# Night campaign 2026-10-05/06: after the evening campaign (E14 -> E15 -> E16) has exited,
# apply the five inert-by-default harness patches (one harness version per experiment)
# and run E18 (E7 under the continuing objective) then E19 (critic swap). Logs in results/logs/.
cd ~/dev/lambda-audit
while pgrep -f "analysis/evening_campaign.sh" > /dev/null; do sleep 60; done
while pgrep -f "ppo_lambda.py" > /dev/null; do sleep 60; done
echo "$(date) evening campaign finished; applying E17/E18/E19 harness patches" >> results/logs/night_campaign.log
.venv/bin/python analysis/apply_e17_patch.py >> results/logs/night_campaign.log 2>&1
.venv/bin/python analysis/apply_linear_sgd_patch.py >> results/logs/night_campaign.log 2>&1
.venv/bin/python analysis/apply_episode_len_patch.py >> results/logs/night_campaign.log 2>&1
.venv/bin/python analysis/apply_critic_fit_patch.py >> results/logs/night_campaign.log 2>&1
.venv/bin/python analysis/apply_oracle_critic_patch.py >> results/logs/night_campaign.log 2>&1
md5 -q harness/ppo_lambda.py harness/gleam_native.py >> results/logs/night_campaign.log
echo "$(date) starting E18" >> results/logs/night_campaign.log
.venv/bin/python analysis/run_e18.py --workers 8 > results/logs/run_e18_campaign.log 2>&1
echo "$(date) E18 finished; starting E19" >> results/logs/night_campaign.log
.venv/bin/python analysis/run_e19.py --workers 8 > results/logs/run_e19_campaign.log 2>&1
echo "$(date) E19 finished" >> results/logs/night_campaign.log
