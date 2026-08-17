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

Every substantive claim is frozen as an *executable* preregistration: constants
and scoring functions live in Python files whose freeze is asserted
mechanically, and the deployment script runs each as a pre-flight and refuses
to submit if a check fails. Amendments are dated; post-data amendments are
disclosed as such in the file that carries them. Scoring refuses to run on data
with duplicated cells.

Scored to date: 11 predictions passed, 6 failed, 1 partial. The failures are
part of the record and are analysed rather than removed — one of them (E7)
produced the project's most informative result.

## Preregistrations

| File | Scope | Frozen |
|---|---|---|
| `prereg_lambda_pilot.py` | POPGym pilot gates G1–G3 | 2026-08-08 |
| `prereg_phase2.py` | E6 minimal pair, E7 quantitative law, E9 corroboration | 2026-08-12 |
| `prereg_e7_bias.py` | Mechanism threshold predictions | 2026-08-14 |
| `prereg_e10.py` | In-domain cost law and perturbation-decay test | 2026-08-14 |

Run any of them with no arguments to print the frozen predictions; pass a
results file to score. `prereg_e10.py --check-clean <file>` verifies no data
existed at freeze time.

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

## Author

Idil Gözel, UCL.
