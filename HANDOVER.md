# Lambda-audit: handover for a new assistant

Written 2026-10-05 for whoever (human or model) continues this project. Everything below is checkable against the repository; nothing here should be trusted over the files it points to. Section 6 (evening addendum) supersedes earlier sections where they disagree.

## 1. What the project is

**Repository:** `~/dev/lambda-audit` (GitHub `idilgozel/lambda-audit`). Sole author: Idil Gözel (UCL).

**One-paragraph aim.** In partially observed RL, "memory helps" comparisons train a memoryless baseline with GAE(λ), whose critic sees only the observation. The paper shows, in closed form on an exactly solvable linear-Gaussian family, that the fixed point this training converges to depends strongly on λ (gain 24.79 at λ=0 → 1.51 at λ=1, true optimum 1.649), derives the threshold λ½ at which the excess cost collapses, and states a scope proposition (the effect needs a hidden coordinate the *action can steer*) that is confirmed on both sides in-domain and on external benchmarks. The whole thing was run as an **executable preregistration**: predictions frozen in code before data existed, every verdict reported. The audit discipline is part of the contribution, not decoration. **Read section 6 before reading the in-domain quantitative claims: the 2026-10-05 pilots changed what E10/E13 can be said to show.**

**Current target venue:** ICML 2027 (abstract 16 Jan 2027, paper 22 Jan 2027, expected 8-page main text + unlimited appendix). ICLR 2027 was the original target and was not submitted. Fallbacks: RLC 2027 (~March), TMLR (rolling).

## 2. Where everything is

### The paper (LaTeX, in the repo)
- `paper/tex/main.tex` — ICLR 2027 style currently; **switch to the ICML 2027 style when it is released**.
- `paper/tex/sections/01-introduction.tex` … `06-related.tex` — §1–§4 drafted and compressed; §5 Discussion and §6 Related work drafted 2026-09-11. Abstract is a sized placeholder. AI-use statement is a placeholder.
- `paper/tex/sections/A1-testbed-details.tex`, `A2-results-details.tex` — appendices. The Derivations appendix (`app:theory`) is still a placeholder; `knee-formula-findings.md` now holds the closed forms it should contain.
- `paper/tex/refs.bib` — 17 entries verified 2026-09-11 except the Andrychowicz et al. λ-sensitivity sentence.
- `paper/tex/measure.sh` — compiles main text only and reports pages per section.
- `paper/00-submission-plan.md`, `paper/00-drafting-plan.md`, `paper/05-results.md`, `paper/03-method.md` — plans and markdown sources (03-method §3.7 holds the number-provenance table).
- `04-results.tex` has a **pending slot** for E14 and a ledger row `P17.1–.3 Pendulum (E14) pending`. Fill from the VERDICTS line in `results/logs/run_e14_campaign_b.log` (or `python3 prereg_e14.py results/results/results_e14.jsonl`) verbatim.

### The theory
- `vendor/gleam/` — vendored generators: `theorem2_learning.py`, `lstd_lambda.py`, `bias_stability.py`, `kernel_general.py`, `gleam_bench.py`. Every number in §3 is annotated with the file that generates it. Numbers are generated, never typed.
- `knee-formula-findings.md` (2026-10-05): identity (I) for the exact GAE(λ) gradient, the direction theorem, the knee formula (II) and its validation, the first-order (bistable) transition for Π = Θc/(h s_e) ≳ 100. Generators `analysis/knee_formula.py` (`--slices`), `analysis/bistability.py`, `analysis/e17_theory.py`.
- `kernel-generality-findings.md` — dimension 2–4 / Prony kernels. `e13-findings.md` — cubic-damping robustness (E13; see section 6 for how to read it now).
- `mechanism-diagnosis-oct.md` (2026-10-05, post hoc): why the S-index predictions missed. The demoted mechanism stays demoted.
- The companion theory note ("GLEAM — Generalized Langevin Engine") lives in `~/dev/GLEAM`; not yet on arXiv.

### Preregistrations and scorers (the spine of the audit)
`prereg_lambda_pilot.py`, `prereg_phase2.py` (E6/E7/E9), `prereg_e7_bias.py`, `prereg_e8.py`, `prereg_e10.py`, `prereg_e11.py`, `prereg_e12.py`, `prereg_e13.py`, `prereg_e14.py`, and (frozen 2026-10-05) `prereg_e15.py`, `prereg_e16.py`, `prereg_e18.py`, `prereg_e19.py`. Each docstring **is** the preregistration: claims, thresholds, how a miss is read, dated amendments. Run with no args to print the frozen predictions; with a results file to score; `--check-clean <file>` proves no data existed at freeze. Synthetic scorer tests: `analysis/test_prereg_e1*.py`. `analysis/recount_ledger.py` regenerates the ledger from the scorers' VERDICTS lines — **never type a ledger count by hand** (it still needs entries for e15/e16/e18/e19).

### Harness and environments
- `harness/ppo_lambda.py` — single-file PPO, heads mlp/gru128/linear/linearb, Gaussian head with pinned σ, single-trajectory evaluator for GLEAM, episodic evaluator for POPGym. Six additive, inert-by-default patches live in `analysis/apply_*_patch.py` (`final_sigma`, `e17` row keys, `linear_sgd`, `episode_len`, `critic_fit`, `oracle_critic`); the night campaign applies them in that order after E16. One harness version per experiment; the md5 is in `results/logs/e1*_provenance.json`.
- `harness/gleam_native.py` + `gleam_instances.npz` — the testbed; `harness/envs_native.py` — numpy ports of the POPGym envs, bitwise-verified by `harness/test_env_equivalence.py`.
- Compute since October: Idil's Mac (Apple M5 Pro, 18 cores), via `analysis/run_e1*.py` (local runners with the same discipline as `myriad/deploy.sh`: pre-flight freeze check, duplicate-cell refusal, per-run logs, provenance JSON). Myriad is no longer available.

### Integrity and provenance
`PROVENANCE.md` (cluster-log chronology; **needs rows for E14–E19, node class = local M5 Pro**), `MEMO-arch-linear.md`, `e10-above-knee-drift.md`, `e11-findings.md`, `e17-pilot-findings.md`, `README.md`.

### Project notes (Claude project "GLEAM")
`claude/lambda-audit-e17-pilots-and-figures.md` (the evening campaign record), `claude/lambda-audit-oct-autonomous-campaign.md`, `claude/lambda-audit-e14-pendulum-design.md`, `claude/lambda-audit-e13-and-phase-a-closeout.md`, `claude/lambda-audit-e10-drift-diagnosis.md`, `claude/lambda-audit-e7-diagnosis-e10.md`, `claude/iclr-status-and-todos.md`, `claude/lambda-audit-mechanism-bias-direction.md`, `claude/lambda-audit-provenance-and-e9-verdicts.md`.

## 3. Rules that must survive the handover
1. **Nothing is rescored, reinterpreted or excused.** A registered prediction that fails is reported as failing. Post-hoc analyses are labelled post hoc and carry no confirmatory weight.
2. **Amendments are dated** and written into the prereg file; post-data amendments are labelled POST-DATA and must be verdict-neutral.
3. **Capability gates are not findings.** A failed gate means NOT SCORED.
4. **Numbers are generated, not typed.**
5. **Probe before grid.** Never re-run a probe cell into the same results file.
6. **Node class is provenance.**
7. **The paper spends its pages on what was accomplished**; failures are stated, not dwelt on.
8. Working rules with the author: end each message with a one-line "what is complete / next step"; diagnose and fix rather than stop; give commands to run.

## 4. What remains for submission
- Paste the E14–E19 verdict lines (verbatim, from the logs named in §5) into the paper's pending slots and ledger; `recount_ledger.py` gives 18 pass / 21 fail of 39 with P22.2 NOT SCORED (README headline updated 2026-10-06; the paper's §1 counts still need it).
- Figures: `analysis/figures_costlaw.py`, `figures_external.py`, `figures_e10_drift.py`, `figures_bistability.py`, `figures_knee_formula.py`, `figures_critic_swap.py` (+ existing `figures_e6.py`) → `analysis/figures/`. All regenerate from committed data.
- Abstract, AI-use statement, Derivations appendix (from `knee-formula-findings.md`).
- Number verification; cold read; ICML style + page squeeze; anonymised release; arXiv the GLEAM companion note.

## 5. Status of the experiments (as of 2026-10-06, 19:00 BST)
Every queued experiment is complete and scored; nothing is running. Verdict lines, verbatim, with the log that carries the scoring block:
- E14 (`results/logs/run_e14_campaign_b.log`): gates 6/6 PASS; `VERDICTS: {'P17.1': 'PASS', 'P17.2': 'PASS', 'P17.3': 'FAIL'}`.
- E15 (`run_e15_campaign.log`): `VERDICTS: {'G15': 'PASS', 'P18.1': 'PASS', 'P18.2': 'PASS', 'P18.3': 'FAIL'}` — P18.3 missed by 0.0011 (|diff| 0.01109 vs 0.01) against the theory knee at the measured σ; the learned σ is 0.04–0.14, far below the pinned 0.3.
- E16 (`run_e16_campaign.log`): `VERDICTS: {'G16-stack8': 'PASS', 'G16-stack64': 'PASS', 'G16-gru128': 'PASS', 'P19.1': 'FAIL', 'P19.2': 'FAIL', 'P19.3': 'FAIL'}` — memory arms do not escape the below-knee penalty (stack64 at λ = 0.9: E = 36.6); all three match the memoryless optimum at λ = 1.
- E18 (`run_e18_campaign.log`): `VERDICTS: {'G18-GLEAM_g1_h0.05': 'PASS', 'P22.1': 'FAIL', 'P22.2': 'NOT SCORED', 'P22.3': 'FAIL'}` — removing the episode resets moves the mlp knee from 0.868 to 0.915 against the frozen 0.984; below-knee plateau J/J_AC = 3.08. The reset transient is not the cause of the E7 gap.
- E19 (`run_e19_campaign_b.log`): `VERDICTS: {'G19-GLEAM_g1_h0.02-0.9': 'PASS', 'G19-GLEAM_g1_h0.02-0.95': 'PASS', 'G19-GLEAM_g1_h0.05-0.9': 'PASS', 'G19-GLEAM_g1_h0.05-0.95': 'PASS', 'P23.1': 'PASS', 'P23.2': 'PASS', 'P23.3': 'PASS'}` — with the exact LSTD(λ) critic the arm moves toward k_AC from both sides without overshoot on both instances and both λ (e.g. g1_h0.05, λ = 0.9: 10.27 → 18.05 and 30.82 → 21.95 around k_AC = 20.55); with PPO's own critic it drifts down from both sides (10.27 → 1.14). E19's first run (17:00–17:52) was never scored: the runner had rounded the start gains to six significant figures and the frozen scorer matched no cell; it is archived in `results/results/e19_run1_unscorable/` and the runner was fixed before the re-run. The prereg was not edited. Details: `PROVENANCE.md` §6.
Ledger (`recount_ledger.py`): 18 pass / 21 fail of 39 scored; P22.2 NOT SCORED outside the tally; gates 22 PASS / 2 FAIL (G11a, G11b). Figures regenerated in `analysis/figures/` (`fig_critic_swap` from E19, `fig_critic_swap_pilots` from the seed-101 pilots). Provenance caveat that the paper must carry: E15, E16, E18 and E19 have no pre-run commit (`PROVENANCE.md` §6). Harness versions: E14 `77c89e53`, E15/E16 `1199ad2f`, E18/E19 `45de828e`; the patched harness is what the repo now contains, every added option off by default.

## 6. Addendum, 2026-10-05 evening: what changed in the reading of the in-domain evidence
Full record: `e17-pilot-findings.md`. In short:
- PPO's one-parameter linear arm (E10's vehicle), trained with its own MLP critic, drifts **down** from every start at every λ, including from k_AC itself under continuing training (57.8 → 48.2, two seeds), where the exact aliased gradient says up. Its critic has learned none of the quadratic value curvature θ₂ through which identity (I)'s displacement acts (fitted ≈ +0.1 against −199…−405).
- **Critic swap**: with the exact LSTD(λ) value substituted into PPO's GAE, the same arm moves toward k_AC from both sides (λ = 0.9: 10 → 24.0, 30 → 41.6, 80 → 66.9). The fixed point is realised by the learner when the critic is the one the theory assumes; PPO's default critic does not learn it at these signal levels (a 1–2 % value modulation). E19 registers this prospectively.
- Consequence: E10's P11.1/P11.3 and E13's passes are the initialisation schedule pushed through the (verified) deployed cost map, not evidence that the learner's equilibrium is at k_AC. Verdicts stand as scored; the paper must say what they do and do not show. E7's failures stand; the training-objective mismatch (1024-step episodes with resets vs the continuing theory; `analysis/episodic_objective.py`) is real but does not explain them (E18 registers the continuing-training control). E6, E9/E14, the externals and the theory are untouched.
- `mechanism-diagnosis-oct.md` was corrected: P7.1 FAILED (the earlier text said it passed).
