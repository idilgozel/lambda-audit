# Where PPO's learner sits relative to the aliased fixed point — the E17 pilots (2026-10-05)

Status: **pre-freeze pilots, seed 101–103, one to two runs per cell; exploratory, no
confirmatory weight.** Nothing registered is rescored. Every number below is in
`results/pilots_e17/*.jsonl` (rows) and `results/pilots_e17/curves_*.csv`; the
harness used is the shipped `harness/ppo_lambda.py` (md5 77c89e53…) or the copy in
`results/pilots_e17/sgd/harness/` carrying the five inert-by-default patches of
`analysis/apply_{e17,linear_sgd,episode_len,critic_fit,oracle_critic}_patch.py`, whose
default path is bitwise identical to the shipped harness (rows and curves, timing
column excluded; tested on a short run of each arch). The theory objects are those of
`vendor/gleam/kernel_general.py` (identity I, `knee-formula-findings.md`).

## 1. What was being tried, and why it stopped

E17 was to be a drift-direction test of the predicted bistable window on
GLEAM_g1_h0.02 (knee-formula-findings.md §4): start E10's one-parameter linear arm
between the equilibria and read the sign of its drift. Its feasibility pilots showed
the arm's drift does not follow the exact aliased gradient ĝ at all, so the test was
never registered; the drafted prereg/runner/tests stay out of the repository and only the
exact-gradient table generator (`analysis/e17_theory.py`) is kept. What the pilots found
instead is more important for the paper.

## 2. The pilots (GLEAM_g1_h0.02 unless stated; 15M steps; drift = deployed gain − start gain)

Episodic training as shipped (1024-step episodes, reset from the uncontrolled state), Adam, E10's arm:

| λ | start k₀ | exact ĝ(k₀; λ) | theory says | drift |
|---|---|---|---|---|
| 0.9930 | 10 | −7.8e-5 | up (→ 31.8) | −0.31 |
| 0.9959 | 10 | −1.25e-5 | up (→ 21.0) | −0.27 |
| 0.9966 | 10 | +3.5e-6 | ≈ 0 | −0.26 |
| 0.9975 | 10 | +2.4e-5 | down (→ 1.9) | −0.24 |
| 0.9959 | 4 | +2.75e-5 | down (→ 2.5) | −1.13 |
| 0.9975 | 4 | +1.0e-4 | down | −1.02 |
| 0.9 | 30 | −4.5e-4 | up (→ 57.8) | −0.20 |
| 0.9 | 80 | +4.5e-4 | down (→ 57.8) | −2.73 |
| 0.98 | 25 / 70 | −1.3e-4 / +1.9e-4 | up / down (→ 46.2) | −0.10 / −0.52 |
| 0.99 | 20 / 55 | −6.8e-5 / +6.9e-5 | up / down (→ 37.4) | −0.07 / −0.10 |
| 0.995 | 12 / 40 | −3.0e-5 / +3.1e-5 | up / down (→ 25.5) | −0.19 / +0.00 |

Same with plain SGD on the gain (Adam elsewhere), k₀ = 10.5: η = 0.01 → −0.55 (λ 0.9930), −0.46
(0.9966); η = 0.003 → −0.16, −0.13. The drift is linear in η, so it is the mean of PPO's
gradient, not optimiser noise or an Adam artefact.

Continuing training (`--gleam-episode-len 10000000`: no reset ever), λ = 0.9:

| arm | start k₀ | drift | end |
|---|---|---|---|
| SGD η 0.01 | 30 | −1.22 | 28.8 |
| SGD η 0.01 | 80 | −3.05 | 77.0 |
| Adam | 10 (seeds 101, 102) | −9.87, −9.88 | **0.13, 0.12** (no control; E = 2.95) |
| Adam | 30 (seed 103) | −6.86 | 23.1 |
| Adam | 57.81 = k_AC(0.9), E10's initialisation (seeds 101, 102) | −9.55, −9.67 | 48.3, 48.1 |

The Adam arm started at k = 10 descends through the true optimum (deployed E falls from
1.13 to 0.90 by 7.7M steps) and keeps going to zero gain (E 2.6 by 12M): it follows a
myopic objective, not the aliased one. The arm started AT the theory's equilibrium
leaves it by ten units, seeds agreeing to 0.1.

mlp from scratch, GLEAM_g1_h0.05, continuing training: λ 0.9 → gain 4.61 / 5.86 (two
seeds), E 2.16 / 8.94; λ 1.0 → gain 1.646 (= k_ml), E 1.02. E7's episodic numbers at the
same cells: gain 4.1–5.9, E 2.6 (λ 0.9); E 0.996 (λ 1.0). Removing the resets changes
nothing for the mlp.

## 3. The critic is the difference

Identity (I) attributes the whole upward (over-control) displacement to the critic's
curvature θ₂: with a constant baseline (θ₂ = 0) the exact gradient is positive at every
gain above k₁ — down everywhere, growing with k — which is the pattern of every default
cell above. Fitting the trained network critic along the deployed trajectory
(`--log-critic-fit 1`, V̂(v) ≈ a + b v + c v²) gives:

| run | gain | LSTD θ₂ (theory, cost units) | fitted c (reward units; theory says −θ₂) |
|---|---|---|---|
| linear, λ 0.9, continuing, from 30 | 23.1 | 199 | **+0.07** |
| linear, λ 0.9, continuing, from 57.8 (two seeds) | 48.2 | 405 | **+0.12, +0.13** |
| mlp, g1_h0.05, λ 0.9, continuing | 5.9 | 22 | −7.2 (one third, right sign; plus a linear term −19.6 v) |

The modulation the LSTD critic carries over the states the policy visits is 0.1 value
units on a level of 6–10 (1–2 %); the linear arm's critic has learned none of it, the
mlp's about a third.

**Critic swap.** With the exact LSTD(λ) value −θ₂(k, λ) v² substituted for the critic's
values in the GAE (`--oracle-critic lstd`; actor, sampling, surrogate, optimiser
unchanged), continuing training, Adam, λ = 0.9, seed 101:

| start k₀ | default critic → | oracle critic → | k_AC(0.9) |
|---|---|---|---|
| 10 | 0.13 | **23.98** (+14.0) | 57.81 |
| 30 | 23.1 | **41.63** (+11.6) | 57.81 |
| 80 | 77.3 (episodic, Adam) / 77.0 (continuing, SGD) | **66.86** (−13.1) | 57.81 |

With the theory's critic the actor is attracted to the aliased equilibrium from both
sides (the runs are mobility-limited at 15M steps with lr annealing and stop short of it);
with its own trained critic it is not.

## 4. The reset transient (a real effect, but not this one)

Training episodes truncate every 1024 steps and reset from the uncontrolled stationary
state; the settling transient costs ∝ gain. Computed exactly from the shipped matrices
(`analysis/episodic_objective.py`, regenerating the table from `vendor/gleam` and
`harness/gleam_instances.npz`): the per-step episodic cost
is 1.7× / 2.5× / 7.5× the stationary cost at k = 10 / 20 / 57.8 on g1_h0.02, and the
episodic objective's gain-gradient is +6e-4 … +1e-2 per step for k = 10…80, against
ĝ ≤ 5e-4; its equilibria sit at 1/2–1/20 of k_AC and its knee 0.03–0.05 lower in λ
(g1_h0.05: 15.3 / 12.4 / 8.5 / 2.5 at λ = 0.8 / 0.9 / 0.95 / 0.98 against k_AC 22.4 /
20.5 / 17.7 / 11.4). The deployed evaluation has no resets by design; training does. It
is a genuine mismatch between the trained and the analysed objective, worth stating in the
paper, but §2 shows it is not what moves the arm: without resets the arm moves further in
the same direction (the episodic advantage scale, inflated by the transients, had been
shrinking PPO's normalised steps).

## 5. What this does to the evidence

- **E10 (P11.1 PASS cells, P11.3 PASS) and E13 (P16.1, P16.2 PASS)** used the linear arm
  started at k_AC. Under the shipped episodic training that arm moves 0.1–1.6 units in
  15M steps from wherever it is put, for any λ, and under continuing training it leaves
  k_AC by ten units. Those passes are statements about the initialisation schedule
  pushed through the deployed cost map, which is itself verified (E10 fact 2:
  J_measured/J_analytic(k_measured) ∈ [0.99, 1.01]). They are not evidence that the
  learner's equilibrium is at k_AC. Verdicts stand as scored; the reading changes.
- **E7 (P7.1–P7.3 FAIL)** stands, and the objective mismatch does not explain it: the mlp
  reaches the same gains and costs without resets.
- **E6 (P6.2–P6.4 PASS)**, the externals **E9/E14** and the theory are untouched.
- **The modelling claim** "PPO realises the aliased fixed point in-domain" is false for the
  harness's default critic and true when the critic carries the LSTD(λ) curvature. The
  displacement is a property of actor-critic learning *with a critic that has learned the
  aliased value*; PPO's MLP critic, trained on these returns at these signal levels
  (1–2 % value modulation), does not learn it, and the one-parameter actor then follows a
  myopic gradient to under-control.

## 6. Registered follow-ups

- **E18** (`prereg_e18.py`, frozen 2026-10-05, amended pre-data to one instance): E7's
  predictions for the mlp under continuing training, 40 runs, with the critic fit recorded
  in every row. Queued after E16 by `analysis/night_campaign.sh`.
- **E19** (critic swap, `prereg_e19.py`, to be frozen before any run): the linear arm
  with the LSTD oracle critic against the default critic, from below and above k_AC,
  several λ, seeds — the prospective form of §3.

## 7. Files

`results/pilots_e17/` (all rows, curves, logs, orchestrator scripts), `analysis/apply_*_patch.py`
(five patches, inert by default), `analysis/e17_theory.py` (exact gradient table at the
drafted E17 cells), `analysis/episodic_objective.py` (the §4 tables).
