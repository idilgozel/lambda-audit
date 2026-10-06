# Vendored GLEAM sources

Copied verbatim from `~/Claude/Projects/GLEAM/` on 2026-08-12. Not modified.
The originals remain canonical. If they change, re-copy, do not edit here.

| File | Why it is here |
|---|---|
| `gleam_bench.py` | GLEBench, the reference environment and the exact baselines. |
| `lstd_lambda.py` | Exact LSTD(lambda) / GAE(lambda), gives k_eq(lambda). |
| `theorem2_learning.py` | Closed loop objects and constants that lstd_lambda imports. |
| `gleam_gym.py` | Episode conventions, episode_len 1024 and reward = -cost. |

These need scipy and run LOCALLY only. Nothing here ships to Myriad. They are
used to precompute `harness/gleam_instances.npz` and the frozen reference
values in `prereg_phase2.py`, and to drive the bitwise equivalence test.

`theorem2_learning.py` holds `H`, `BETA` and `CC` as module globals, and
`lstd_lambda.py` reads them. To evaluate an instance with a different step,
set `T.H` and `T.BETA` before calling. The `disc_model` cache is keyed on
`(gamma, H)`, so that is safe.
