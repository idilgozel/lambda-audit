<!-- STATUS NOTE, added 2026-08-21 when this file was restored to the repository.

     This memo was written on 2026-08-14 and the patch it specifies was applied,
     but the file itself was never committed: it existed only in the working
     directory of the session that produced it, while three places in the
     repository cite it (paper/03-method.md section 3.7, and two comments in
     harness/ppo_lambda.py). A pre-submission audit on 2026-08-21 found the
     dangling reference. It is restored here UNEDITED apart from this note.

     Its section 6 records a one-time before/after bitwise comparison. That
     comparison is superseded, not deleted, by harness/test_arch_regression.py,
     which re-derives the same claim from published curves and can be re-run by
     anyone on the originating node class. Read section 6 as the historical
     record of when the check was first made; read the script as the check.

     Section 7's redesign was carried out and became E10 (prereg_e10.py).
     Sections 8-9 describe scoring against Tier 1b criteria and predate the
     E10 preregistration, which supersedes them. -->

# MEMO for Claude Code: `--arch linear`, and the redesign the pilot forced

Date: 2026-08-14. Written against `harness/ppo_lambda.py`, 30,573 bytes.

**The patch in this memo has already been applied and tested.** The regression
in §6 passes, both guards fire, and the pilot in §7 has been run. What is left
for you is the cluster job and the scoring. Read §7 first — the experiment is
not the one `a5-tier1b-findings.md` proposed, and the pilot is why.

## 0. Why this exists

`a5-tier1b-findings.md` §5: the discriminator result crosses two
implementations, a numpy reimplementation against E6's MLP. Different critic
parameterisation, different learning-rate schedule, different optimiser. The
headline — intermittency 0.000 for the linear policy against 0.492 for the MLP —
is confounded with all of it. This removes that: same harness, same critic, same
optimiser, same evaluator, same seeds, only the policy head differs.

## 1. HARD CONSTRAINT: existing architectures must not move by one bit

E6's 80 rows and E9's 160 rows are registered data that cannot be re-run, and
`prereg_phase2.py` scores against them. The risk is RNG: `layer_init` calls
`nn.init.orthogonal_`, which draws from the torch generator, so creating,
reordering or resizing any parameter on an existing arch's path shifts every
later draw. `Agent.__init__` already documents this — the continuous head is
built from a constant for exactly this reason.

**Add branches, never reorder.** Every new parameter is created inside an
`arch == "linear"` branch, and the linear branch itself uses `zeros_` and
`fill_`, which draw nothing. §6 proves it held.

## 2. HARD CONSTRAINT: never write into `results_phase2.jsonl`

`prereg_phase2.py`'s `cell()` selects rows by `(env, arm, lambda, seed)` and
**does not filter on `arch`** — verify it yourself at line 422. A `linear` row on
`GLEAM_g1_h0.05` at λ=0.9 seed 1 would be pulled into E6's P6.1, P6.2 and P6.4
cells alongside the `mlp` row and corrupt every IQM reading them.

Amendment 1's duplicate guard will NOT catch this. It keys on
`(env, arch, lambda, seed)`, so `mlp` and `linear` look like two distinct cells
to the guard and like one contaminated cell to `cell()`. This is the exact
hazard Amendment 1 exists for, arriving through a door it does not watch.

So every linear run passes `--results-name results_tier1b.jsonl`, and §5 adds a
guard that refuses to start otherwise. Do not "fix" `cell()`: that file is frozen
and nothing but `mlp` has ever been written to the phase 2 file.

## 3. The design

Policy: a single scalar gain, `u = -k v`. No bias, no hidden layer, no learned
sigma, so it cannot leave the gain manifold. Critic: unchanged — same two-layer
Tanh trunk, same `Linear(hidden, 1)` head, same value loss.

The actor is `nn.Linear(obs_dim, act_n, bias=False)` applied to the **raw
observation**, not the trunk. `bias=False` enforces "no bias" structurally, and
the `final_gain` diagnostic in `evaluate_continuous` — a regression of `u` on the
first observation component, reported as `-slope` — is then exactly `k` with no
change to the evaluator.

**Residual difference from E6, to be stated in the write-up, not hidden.** In E6
the trunk is shared, so policy gradients shape the critic's features. With a
linear actor the trunk is trained by the value loss alone. Unavoidable — a policy
reading the trunk is not confined to the gain manifold — but real, and a referee
will ask.

## 4. The patch, applied

Five edits, all additive.

1. `parse_args`: `choices=[..., "linear"]`, plus `--linear-gain-scale`
   (default 1.0) and `--linear-gain-init` (default 0.0).
2. `ARCH_STACK`: `"linear": 1`.
3. `Agent.__init__`: signature gains `gain_scale=1.0, gain_init=0.0`; a branch
   **after** `self.actor` and `self.critic` are built sets
   `self.linear_policy = (arch == "linear")` and, when true, validates
   `continuous` and `obs_dim == act_n == 1`, replaces `self.actor` with the
   bias-free layer, zeroes it, records `gain_scale`, and applies `gain_init` via
   `weight.fill_(-gain_init / gain_scale)` under `no_grad`.
4. `Agent.act`: `mean = self.gain_scale * self.actor(x)` when
   `self.linear_policy`, else `self.actor(z)` as before.
5. `main()`: the §5 guard.

## 5. The results-file guard

```python
    if args.arch == "linear" and "phase2" in args.results_name:
        raise SystemExit(
            "REFUSING TO RUN. --arch linear must not write into "
            f"{args.results_name}. Pass --results-name results_tier1b.jsonl.")
```

Verified: it fires, and `--arch linear` on a discrete env exits with
"has no meaning on a discrete environment."

## 6. The regression, and one correction to how you compare

Run before and after the patch and diff. **Two columns are pure timing and will
always differ: `wallclock_s` in the JSONL row and `sps` in the curve CSV.** Strip
both. An earlier version of this memo named only `wallclock_s` and the curve
comparison then failed for no reason.

Result, on `GLEAM_g1_h0.05`, λ=0.9, seed 1, γ=0.9975031223974601, σ pinned at
0.3, `mlp` at 100k steps and `gru128` at 40k:

```
  GLEAM_g1_h0.05_gru128_lam0.9_s1              ROW IDENTICAL
  GLEAM_g1_h0.05_mlp_lam0.9_s1                 ROW IDENTICAL
  curves_..._gru128_lam0.9_s1.csv    CURVE IDENTICAL apart from sps
  curves_..._mlp_lam0.9_s1.csv       CURVE IDENTICAL apart from sps
```

Re-run this after any further edit. If anything other than those two columns
moves, stop: the published E6/E9 rows are no longer reproducible from the file,
which is a bigger problem than this experiment.

Also confirm the three existing tests still pass:
`test_env_equivalence.py`, `test_gleam_equivalence.py`, `test_eval_calibration.py`.

## 7. THE REDESIGN. Read this before submitting anything.

The experiment `a5-tier1b-findings.md` proposed — start at zero, train, see
whether intermittency appears — **does not work in this harness**, and the pilot
says so rather than a hunch.

Starting from `k = 0` at λ=0.9:

| steps | gain scale | final_gain |
|---|---|---|
| 200k | 1 | 0.026 |
| 1M | 1 | 0.149 |
| 1M | 20 | 0.218 |

The target is `k_AC(0.9) = 20.547`. A one-parameter policy under Adam moves its
gain far more slowly than the MLP, whose effective gain is a sum over 128
weights, and raising the gain scale twentyfold buys a factor of 1.5, not 20. At
this rate 15M steps lands somewhere around 1 to 3. The run would report "no
intermittency at λ=0.9" and it would mean **"never got to where the MLP is
unstable"**, not "is stable there". That is a null for the wrong reason, and it
is the exact failure mode this project keeps catching.

**So invert the experiment. Start AT the analytic equilibrium and ask whether it
stays.** That is the question the discriminator is actually asking, it is the
question the Tier 2 theory answers — local stability of a fixed point — and it
costs less compute rather than more.

Pilot, 1M steps, one seed, `--linear-gain-init` at the analytic `k_AC(λ)`:

| λ | k init | final_gain | J_rate | J analytic at k_AC | ratio |
|---|---|---|---|---|---|
| 0.90 | 20.546902 | 20.468 | 0.2796 | 0.292074 | 0.957 |
| 0.99 | 2.197126 | 2.122 | 0.2110 | 0.222898 | 0.947 |

The gain-only policy sits at the aliased equilibrium at λ=0.9, drifting 0.4
percent over 1M steps, and its cost matches the analytic cost of its own gain to
within 5 percent. E6's MLP at the same λ has intermittency 0.492 and cost ratios
from 1.19 to 3.83.

That is H_B, cleanly, in the real harness: the aliased equilibrium is perfectly
stable for a policy confined to the gain manifold, and the instability the MLP
shows lives in the directions the linear policy does not have. It is also the
direct empirical counterpart of the Tier 2 result, which computes the gain axis
as stable and the bias direction as unstable below λ\* = 0.9759.

### 7a. On `--linear-gain-scale`

Keep the default of 1.0 for the runs in §7b. The flag exists because the pilot
needed it to establish that scaling is not the fix, and that negative result
belongs in the record. Do not use a non-default value without reporting it.

### 7b. The run

`GLEAM_g1_h0.05`, coupled, σ pinned at 0.3, 15M steps to match E6, **three
seeds** — Tier 1b §5 flags one seed as insufficient for a paper table.

Two arms, because both questions are worth answering and the second is nearly
free:

- **Arm S (stability), the headline.** `--linear-gain-init` at the analytic
  `k_AC(λ)` for that λ. Does it stay?
- **Arm R (reachability), reported as a limitation.** `--linear-gain-init 0`.
  Confirms at 15M what the pilot showed at 1M, so the write-up can state the
  reachability limit as measured rather than extrapolated.

λ values: **0.9 and 0.99** are the frozen discriminator pair from Tier 1b §1 and
are mandatory. Add **0.95 and 0.98** if budget allows — they bracket the
predicted onset λ\* = 0.9759 from `prereg_e7_bias.py`, giving a second
independent read on the same threshold.

The analytic `k_AC(λ)` values for this instance, from `prereg_phase2.py`'s
`K_AC_LAMBDA["GLEAM_g1_h0.05"]`, in the order of `LAMBDAS`:

    0.8    22.426983
    0.9    20.546902
    0.95   17.670634
    0.98   11.394604
    0.99    2.197126
    0.995   1.756354
    0.999   1.548754
    1.0     1.506599

Common flags:

```
--env GLEAM_g1_h0.05 --arch linear --gamma 0.9975031223974601 --fix-sigma 0.3
--total-timesteps 15000000 --num-envs 16 --num-steps 128 --lr 0.0003
--eval-steps 200000 --eval-burnin 1000
--results-name results_tier1b.jsonl
```

Build `myriad/configs_tier1b.txt`: 4 λ × 2 arms = 8 lines, submit `1-24` for 3
seeds, job name `lam_t1b`. `pilot_array.qsub.sh` maps
`SEED = (TASK-1)/NCFG + 1`, so 24 tasks over 8 lines gives seeds 1 to 3 —
matching E6's seeds 1 to 3, which is what makes the comparison seed by seed.

Pass no env list to `submit` for this job: `deploy.sh`'s env guard reads
`results_phase2.jsonl`, these rows go elsewhere, and an env list would
false-positive on E6's existing rows.

**Wait for E7 to finish before submitting.** E7 is the prospective test of a
frozen prediction and has queue priority.

## 8. Scoring, against Tier 1b's already-frozen criteria

Quoted from `a5-tier1b-findings.md` §1 so nothing drifts:

    intermittency := fraction of late-half points > 2 * J_ml   (J_ml = 0.2196)
    H_B supported : mean intermittency at lam 0.9 < 0.05
    H_A supported : mean intermittency at lam 0.9 > 0.25
    ambiguous     : between 0.05 and 0.25
    Control required either way: lam 0.99 intermittency < 0.05
    Secondary: J_measured / J_analytic(fitted gain). H_B predicts ~1.

Score **arm S** against these; that is the arm the criteria were written for.
Compute intermittency from the curve CSVs over the late half, report per seed
and pooled. Report arm R separately as the reachability limitation.

One thing to state plainly in the write-up: the criteria were frozen for a
run that started at zero, and arm S starts at `k_AC`. The thresholds are
unchanged and the prediction they encode — no intermittency for a gain-only
policy at λ=0.9 — is unchanged. What changed is that the policy is now placed
where the question can be asked. Say so; do not quietly re-point a frozen
criterion at a different run without a note.

## 9. What to write when it lands

Update `a5-tier1b-findings.md` §5 and §6 in place: strike the
cross-implementation limitation, add the harness table, add the shared-trunk
caveat from §3 and the reachability limit from §7. Keep both bug reports — they
are still true and still worth reading — and keep the numpy result alongside the
harness result, noting whether they agree.

## 10. Checklist

- [x] Five edits applied
- [x] Regression: `mlp` and `gru128` bitwise identical apart from `wallclock_s` and `sps`
- [x] `--arch linear` on a discrete env exits with the right message
- [x] `--arch linear --results-name results_phase2.jsonl` refuses
- [x] Pilot at `k=0` shows the reachability limit
- [x] Pilot at `k=k_AC` shows the equilibrium is held at λ=0.9 and λ=0.99
- [ ] Three existing test files still pass (run them; not run here)
- [ ] `configs_tier1b.txt` built, 8 lines
- [ ] E7 finished before `lam_t1b` is submitted
- [ ] Score arm S against §8, report per seed and pooled
- [ ] Update `a5-tier1b-findings.md` per §9
