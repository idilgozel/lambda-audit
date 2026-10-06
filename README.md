# lambda-audit

A preregistered audit of a hyperparameter confound in memory-versus-memoryless
reinforcement learning evaluation.

When a recurrent agent outperforms a memoryless one on a partially observed
task, the result is read as evidence about memory. Both agents are typically
trained with GAE(λ), whose bootstrap horizon `(1 − γλ)⁻¹` interacts with the
correlation time of the hidden state — the very quantity such benchmarks
manipulate. This repository contains the theory, experiments, and
preregistrations establishing that the memoryless baseline's learning target
moves with λ, that a second computable threshold in λ governs whether training
converges at all, and that both effects vanish when actions do not influence
the hidden state.

**Status: research in progress.** A paper is in preparation. Nothing here is
published or under review. Results are reported as scored, including failures.

## Method

Registered predictions and scoring functions are recorded in executable
preregistration files. Deployment checks reject unset constants, and
`--check-clean` tests the supplied path for existing data when invoked.
These checks do not retrospectively authenticate freeze dates; the saved
run chronology, pre-run commits, and their limits are documented in
`PROVENANCE.md`. Amendments are dated; post-data amendments are
disclosed as such in the file that carries them. Scoring refuses to run on data
with duplicated cells.

Scored to date: 18 predictions passed, 21 failed, of 39 scored; one prediction
(E18's P22.2) is NOT SCORED by design and sits outside the tally. Capability
gates are counted separately: 22 passed, 2 failed (E11). Run
`python3 analysis/recount_ledger.py` for the current generated prediction and
gate tallies. The failures are part of the record and are analysed rather than
removed — one of them (E7) produced the project's most informative result, and
the E10 failures are diagnosed in `e10-above-knee-drift.md`. The October 2026
local campaign (E14–E19; chronology in `PROVENANCE.md` §6) added the second
external test (E14: 2 of 3), the learned-σ control (E15: 2 of 3), the in-domain
memory arms (E16: 0 of 3), the continuing-objective re-run of the cost law
(E18: 0 of 2 scored) and the critic swap (E19: 3 of 3 — the linear arm is
attracted to the predicted equilibrium from both sides when PPO's GAE uses the
exact LSTD(λ) critic, and drifts to under-control with its own trained critic);
the reading is in `e17-pilot-findings.md`.

## Preregistrations

| File | Scope | Declared freeze date |
|---|---|---|
| `prereg_lambda_pilot.py` | POPGym pilot gates G1–G3 | 2026-08-08 |
| `prereg_phase2.py` | E6 minimal pair, E7 quantitative law, E9 corroboration | 2026-08-12 |
| `prereg_e7_bias.py` | Mechanism threshold predictions | 2026-08-14 |
| `prereg_e10.py` | In-domain cost law and perturbation-decay test | 2026-08-14 |
| `prereg_e8.py` | Intermittency diagnostic tripwires | 2026-08-18 |
| `prereg_e11.py` | External validity, BattleshipEasy (gates failed; see `e11-findings.md`) | 2026-08-18 |
| `prereg_e12.py` | External validity, MineSweeperMedium, gated probe first | 2026-08-21 |
| `prereg_e13.py` | Cubic-damping perturbation of the cost-law threshold | 2026-08-21 |
| `prereg_e14.py` | Second in-scope external test, PositionOnlyPendulum, per-variant gated probe first | 2026-09-11 |
| `prereg_e15.py` | Learned exploration σ on GLEAM_g1_h0.05: cost direction, over-control, cost law at the measured σ | 2026-10-05 |
| `prereg_e16.py` | In-domain memory arms (stack8, stack64, gru128) on GLEAM_g1_h0.05, per-arch gated | 2026-10-05 |
| `prereg_e18.py` | E7 cost law under the continuing objective (no episode resets), mlp from scratch; two dated pre-data amendments | 2026-10-05 |
| `prereg_e19.py` | Critic swap: exact LSTD(λ) critic against PPO's trained critic on the linear arm, continuing training | 2026-10-05 |

Run any of them with no arguments to print the frozen predictions; pass a
results file to score. `prereg_e10.py --check-clean <file>` checks for data
in that file at invocation time. E13's header defines freezing at commit;
its first preserved commit is 25 August despite the 21 August constant above.
E14's first preserved commit is 13 September despite its 11 September
declaration. E15, E16, E18 and E19 were frozen and run on 5–6 October without
a pre-run commit; what supports their freeze declarations is listed in
`PROVENANCE.md` §6. The constants are preserved; see `PROVENANCE.md`.

## Layout

| Path | Contents |
|---|---|
| `harness/` | Single-file PPO with GAE(λ); MLP, GRU, frame-stack, and two restricted policy heads used for identification. Native environment reimplementations, verified bitwise against upstream. |
| `vendor/gleam/` | Exact analysis: closed-loop discretisation, LSTD(λ) fixed points, the aliased equilibrium map, and the off-manifold stability index. |
| `myriad/` | SGE array configs and deployment/collection scripts for the UCL Myriad cluster. |
| `analysis/` | Scoring, intermittency diagnostics, and figures. |
| `results/` | Scored results (JSONL) and per-job logs. Raw training curves are excluded by size; see `.gitignore`. |
| `paper/` | Manuscript drafts. |
| `*.md` | Dated findings and design records, including pre-data notes written before the corresponding runs. |

## Reproducing

The analysis code requires `numpy` and `scipy`; the harness additionally
requires `torch`. Cluster runs need no compilation on the compute nodes —
environments are reimplemented natively for that reason.

```
python3 prereg_phase2.py results/results/results_phase2.jsonl   # score E6/E7/E9
python3 vendor/gleam/bias_stability.py --predict                # regenerate thresholds
```

Every constant quoted in the manuscript maps to a generating file; the
provenance table is in `paper/03-method.md`.

### Reference fields in raw result rows

Rows from continuous-environment runs carry `ref_*` fields copied verbatim
from the per-instance baselines table frozen in `harness/gleam_instances.npz`.
Two of them, `ref_k_AC` and `ref_J_AC`, are the aliased actor–critic
equilibrium **at λ = 0** (`gleam_bench.GLEBench.ac_equilibrium`, the TD(0)
result of Props 1–3) and are therefore constant across a file's λ grid — e.g.
24.793797 in every `GLEAM_g1_h0.05` row. They are not the reference for the
row's own λ and nothing is scored against them; the λ-dependent references
live in the preregistration files (`prereg_e10.py`'s frozen `k_AC`/`J_AC`
tables). The fields are kept, rather than dropped or renamed, so that every
released row is byte-identical to the row its run wrote. Recorded when found
in `provenance-node-class.md`.

## Author

Idil Gözel, UCL.
