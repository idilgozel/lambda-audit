#!/bin/zsh
# 2026-10-05 unattended sequence: wait for E14 to finish, apply the E15 harness
# patch, run E15 (learned sigma, 40 mlp runs), then E16 (memory arms, 120 runs).
cd ~/dev/lambda-audit
LOG=results/logs/evening_campaign.log
echo "$(date) waiting for run_e14.py to exit" >> $LOG
while pgrep -f "analysis/run_e14.py" > /dev/null; do sleep 60; done
echo "$(date) E14 finished; applying final_sigma patch" >> $LOG
.venv/bin/python analysis/apply_final_sigma_patch.py >> $LOG 2>&1 || { echo "patch failed" >> $LOG; exit 1; }
echo "$(date) starting E15" >> $LOG
.venv/bin/python analysis/run_e15.py --workers 8 > results/logs/run_e15_campaign.log 2>&1
echo "$(date) E15 finished; starting E16" >> $LOG
.venv/bin/python analysis/run_e16.py --workers 8 > results/logs/run_e16_campaign.log 2>&1
echo "$(date) E16 finished" >> $LOG
