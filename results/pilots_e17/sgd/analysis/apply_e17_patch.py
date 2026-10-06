"""Apply the E17 harness change: record the gain a linear arm was started at.

Adds `linear_gain_init` to the results row of `--arch linear` / `linearb` runs.
E17 runs two initial gains in the same (env, lambda, seed) cell, so without
this field the row could not be assigned to its cell except by parsing the run
name. mlp, stack, gru and discrete rows keep their exact schema; E10's released
linear rows were written before the field existed and are not touched (their
initial gain is k_AC(lambda) by construction, see myriad/configs_e10.txt).
Idempotent: refuses to apply twice. Run from the repo root after E16's runs have
all exited, so that E15/E16 were produced by one harness version.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "harness", "ppo_lambda.py")
s = open(p).read()
old = '''    if agent.linear_policy and agent.actor.bias is not None:
        row["final_bias"] = float(agent.actor.bias.item()) * agent.gain_scale
'''
new = old + '''    if agent.linear_policy:
        # E17 (2026-10-05): the starting gain is part of the cell for a
        # drift-direction run, and the gain optimiser is part of the arm.
        # Linear arms only; no other row changes.
        row["linear_gain_init"] = float(args.linear_gain_init)
        row["linear_sgd_lr"] = float(getattr(args, "linear_sgd_lr", 0.0))
'''
if "linear_gain_init\"] = " in s:
    print("already applied"); sys.exit(0)
assert s.count(old) == 1, "anchor not found exactly once"
open(p, "w").write(s.replace(old, new))
print("applied: harness/ppo_lambda.py now records linear_gain_init for linear arms")
