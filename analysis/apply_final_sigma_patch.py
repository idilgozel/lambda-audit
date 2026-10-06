"""Apply the E15 harness change: record the learned exploration scale.

Adds `final_sigma` to the results row for continuous runs whose sigma is
LEARNED (no --fix-sigma). Pinned-sigma and discrete rows keep their exact
schema, so every existing experiment's data remains reproducible row for row.
Idempotent: refuses to apply twice. Run from the repo root after E14's runs
have all exited, so that E14 was produced by one harness version.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "harness", "ppo_lambda.py")
s = open(p).read()
old = '        row["fix_sigma"] = args.fix_sigma\n'
new = old + '''        # E15 (2026-10-05): a LEARNED exploration scale is part of the result,
        # since the theory takes s_e as an input. Recorded only when sigma is
        # learned, so every pinned-sigma row (E6-E14) keeps its exact schema.
        if not agent.fixed_sigma:
            row["final_sigma"] = float(agent.actor_logstd.detach().exp().mean().item())
'''
if "final_sigma" in s:
    print("already applied"); sys.exit(0)
assert s.count(old) == 1, "anchor not found exactly once"
open(p, "w").write(s.replace(old, new))
print("applied: harness/ppo_lambda.py now records final_sigma for learned-sigma runs")
