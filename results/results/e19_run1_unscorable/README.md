# E19 run 1 (2026-10-06 17:00–17:52 BST) — UNSCORED, archived

48/48 runs completed with 0 failures, but `prereg_e19.py` scored none of them (`VERDICTS: {}`,
every cell "NOT SCORABLE YET: 0 seeds"). Cause: `analysis/run_e19.py` passed `--linear-gain-init`
through a `:g` format, which rounds to six significant figures (28.905 for the registered
0.5 x k_AC = 28.904965, etc.), and the frozen scorer matches a row to its cell with abs_tol 1e-6.
The runner was fixed (`:.6f`, with a round-trip assertion) and E19 was re-run on the registered
cells; see results/results/results_e19.jsonl and results/logs/run_e19_campaign_b.log.

The prereg was not edited. No outcome quantity of run 1 was printed or inspected before the
decision to re-run. These rows may be reported as an unregistered replicate (start gains within
5e-5 of the registered ones), never as E19 evidence.
