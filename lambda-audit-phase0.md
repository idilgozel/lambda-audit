# λ-audit Phase 0 findings

Date: 2026-08-07. Harness added 2026-08-08. Anchors extracted and gates frozen 2026-08-08. Phase 0 is complete and the package is launch-ready.

## 1. The λ table, verified

Every entry checked in source code or the paper itself. Pin the repo links to the listed commits when this table goes public.

| Who | λ | γ | Evidence |
|---|---|---|---|
| POPGym paper (arXiv:2303.01859) | **1.0** | 0.99 | Table 2. Also lr 5e-5, batch 65,536, 15M steps, BPTT ∞, 3 trials. |
| POPGym baselines code | **1.0** (inherited) | 0.99 | `popgym/baselines/ppo.py` sets no lambda key at all. RLlib default applies. Repo commit 410d5aa (2026-06-11). |
| RLlib PPOConfig | **1.0** default | | `self.lambda_ = 1.0` in `rllib/algorithms/ppo/ppo.py`. Pin: ray-project/ray @ 328e6b7646fa. |
| CleanRL ppo_atari_lstm.py | **0.95** | 0.99 | `gae_lambda: float = 0.95`, num_steps 128. Pin: vwxyzjn/cleanrl @ fe8d8a03c41a. |
| SB3-contrib RecurrentPPO | **0.95** default | 0.99 | `gae_lambda: float = 0.95`, n_steps 128 in `ppo_recurrent.py`. Pin: @ 075bd5be8d43. |
| Memory Gym (arXiv:2309.17207, JMLR 2024) | **0.95** | 0.995 | Table 4, Appendix B. GRU vs TrXL compared at this fixed λ. |
| Ni et al. 2022 (recurrent off-policy) | one-step targets | | Extreme case of the short bootstrap horizon. Note only, no rerun. |

Reading: the reference benchmark for memory architectures ran at λ = 1.0. The most-copied practice code (CleanRL, SB3-contrib) and the other major memory benchmark (Memory Gym) compare architectures at 0.95. The two-sided thesis stands.

## 2. Published anchors (POPGym Table 3, MMER μ ± σ, extracted 2026-08-08)

A first extraction attempt returned fabricated numbers (MLP at 0.93 on RepeatPreviousMedium, physically impossible). Caught by sanity check against the random-policy level. The values below came from a verbatim re-extraction and pass that check. Bonus discovery: **Fr.Stack (frame stacking) is one of their 13 baselines**, so our stack arms have published anchors too.

**RepeatPreviousEasy** (crisp separation, the G1 anchor env):
GRU 1.000 ± 0.000, LSTM 1.000, Elman 1.000, LMU 1.000, TCN 1.000, Fr.Stack 1.000, IndRNN 0.957, FWP 0.200, FART 0.060, DNC −0.223, MLP **−0.320 ± 0.007**, PosMLP −0.336, S4D −0.473.

**StatelessCartPoleEasy** (= PositionOnlyCartPoleEasy):
everything memory-capable at or near 1.000, MLP **0.722 ± 0.001**, S4D 0.514.

**RepeatPreviousMedium** (the important surprise):
MLP −0.486, Fr.Stack **−0.484**, GRU **−0.315**, LSTM −0.284, LMU **0.789 ± 0.288**, everything else between −0.49 and −0.30. At their hyperparameters (batch 65,536 at 15M steps = **229 gradient updates**) Medium is essentially unsolved by everything except LMU, which succeeded on some seeds only.

**Consequence.** Medium's published row cannot anchor G1. It is a budget artifact, which our own probe confirms directly (next section). G1 anchors on Easy and cartpole instead, and Medium's solvability in our regime gets its own gate.

## 3. Capability probes (sandbox, seed 101, disclosed, pilot-external)

Our regime is ~7,300 updates in 15M steps (batch 2,048, lr 3e-4).

| Probe | Steps | Result |
|---|---|---|
| stack64, Medium, λ=1.0 | 4M | **0.997**. Solves it. Published Fr.Stack: −0.484. |
| gru128, Medium, λ=1.0 | 800k | −0.490, still at chance. Expected: k=32 recall is slow for a GRU. |

The stack64 probe settles two things. The transition env is solvable above the knee in our harness, and the published Medium failures were undertraining, not impossibility. The GRU question is settled by the 15M capability check on Myriad.

## 4. Pilot design (final, frozen)

**Grid: 3 envs × 4 architectures × 4 λ × 5 seeds = 240 runs**, 15M steps, γ = 0.99. RepeatPreviousEasy was added to the original 2-env pilot on 2026-08-08, before any pilot run. It is the published G1 anchor and a third control arm: its knee (λ ≈ 0.76) sits below the whole grid, so the theory predicts no ranking movement there. If Easy scrambles at 0.9, the knee placement is refuted, not just decorated. Runtime rises to roughly 213 core-hours. Still one single-core Myriad array.

| Env | Hidden timescale | Knee λ | Role |
|---|---|---|---|
| RepeatPreviousEasy | k = 4 | ≈ 0.76 | control, above knee, memory matters |
| RepeatPreviousMedium | k = 32 | ≈ 0.978 | transition, knee inside grid |
| PositionOnlyCartPoleEasy | τ ≈ 8 | ≈ 0.88 | control, above knee, borderline at 0.9 |

**Frozen gates** (full definitions and rationale in prereg_lambda_pilot.py, which passes its self-check with zero sentinels):

- **G1a** Easy at λ=1.0: every memory-capable IQM ≥ 0.90 and MLP IQM ≤ 0.0.
- **G1b** cartpole at λ=1.0: memory-vs-MLP gap ≥ 0.10.
- **G1c** Medium capability: MEM(1.0) − MLP(1.0) ≥ 0.5, where MEM = best of {stack64, gru128}. Pre-array protocol: 2-task capability job at seed 101 on Myriad, at least one run ≥ 0.5, else the transition env swaps to RepeatPreviousK16 (knee ≈ 0.947, already in the harness) and the prereg is amended before any below-knee run.
- **G2, flagship**: gap(0.9) < 0.25 × gap(1.0) on the transition env. Gating on the best memory arch is the conservative choice: if either memory architecture keeps its advantage at 0.9, G2 fails.
- **G3, blind diagnostic**: statistic gap_lambda, threshold 0.25, balanced accuracy ≥ 0.80 classifying below-knee cells (transition env at 0.9 and 0.95) against above-knee cells, with the transition env at 0.99 excluded as boundary. gap_lambda chosen over gap_value because it is identically zero at λ=1 by construction, which removes false positives at the top of the grid and doubles as an implementation check. Threshold 0.25 is twice the largest above-knee value observed in validation (0.12) and below every below-knee smoke value (0.28 to 0.39 at only 60k steps).

One dated amendment exists, made pre-launch: G1c/G2 were first written gru128-specific; the probes showed that could fail on trainer speed while the claim is architecture-generic, so the memory side became max over {stack64, gru128}.

**Scoring machinery validated on synthetic worlds**: a theory-true world passes all five gates; a null world (no λ effect) passes G1a/b/c and fails exactly G2 and G3.

## 5. Harness (harness/ppo_lambda.py, validated 2026-08-08)

Single-file PPO, written from scratch. Positive controls: RepeatPreviousEasy + stack8 at λ=1.0 reaches 0.990 (random −0.54); cartpole + stack8 reaches 0.959 (random 0.105). Design choices written down because a silent harness bug cost a round once: explicit vector loop (no gymnasium autoreset ambiguity), truncation bootstraps from the real final observation, GRU minibatches split by environment so BPTT sequences stay intact, frame stacks zero-filled at episode start.

**Defect caught pre-launch:** the blind diagnostic originally treated cartpole's 200-step truncation as terminal and reported +0.96, a pure time-limit artifact. Fixed by seeding the MC accumulator with V(final) on truncation: cartpole −0.18, RepeatPrevious unchanged. gap_lambda = 0.0 exactly at λ=1.0 in every run since, as the identity requires.

Runtime, measured: mlp ≈ 22 min, stack8 ≈ 24 min, stack64 ≈ 29 min, gru128 ≈ 2.3 h per 15M-step run, single thread.

## 6. Launch protocol (nothing left to freeze)

1. `bash myriad/ship_to_myriad.sh <ucl_username>`. Pre-flight checks file readability and that the prereg passes its frozen self-check. Post-flight verifies 48 + 2 config lines and the harness flags landed.
2. On Myriad: `bash setup_myriad.sh`. The smoke test must print deployed_return above 0.9.
3. Capability job first: `qsub -N lam_cap -t 1-2 -v CFG=configs_capability.txt,SEED_OFFSET=100 pilot_array.qsub.sh`. Tail the first .o file two minutes later. Verdict per G1c.
4. Main array: `qsub -N lam_pilot -t 1-240 -v CFG=configs_pilot.txt pilot_array.qsub.sh`. Tail the first .o file two minutes later.
5. `bash myriad/pull_pilot.sh <ucl_username>`, then `python prereg_lambda_pilot.py results/results_lambda_pilot.jsonl`.

## 7. Files

- `prereg_lambda_pilot.py`: fully frozen, self-checking, scoring implemented and tested.
- `harness/ppo_lambda.py`: the harness, including the RepeatPreviousK16 contingency env.
- `myriad/configs_pilot.txt` (48 lines), `myriad/configs_capability.txt` (2 lines), `myriad/pilot_array.qsub.sh`, `myriad/setup_myriad.sh`, `myriad/ship_to_myriad.sh`, `myriad/pull_pilot.sh`.
