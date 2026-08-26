# 3. An exactly solvable testbed for the bootstrap horizon

<!-- Draft v1, 2026-08-14. Every number in this section is generated, not
     typed: the provenance table in section 3.7 maps each constant to the file
     and function that produces it. Verify against sources before submission;
     nothing here should be trusted because it is written down.

     TWO ITEMS NEEDING IDIL'S EYES SPECIFICALLY:
     1. Section 3.2's description of the decoupled arm ("u enters the cost and
        not the dynamics") is written from the design record, not from reading
        gleam_native.py. Verify the sentence against the implementation.
     2. Section 3.3 calls the theory critic LSTD(lambda) on (1, v, v^2); the
        HARNESS critic is a neural value head trained by the PPO value loss.
        3.5 keeps them distinct, but check no reviewer can read 3.3 as a claim
        about the trained critic. -->


The audit needs a setting in which the quantities under dispute — the
equilibrium the learner is pulled toward, the cost of sitting there, and the
threshold at which learning stops arriving anywhere at all — are computable
exactly, so that a failed prediction indicts the theory rather than the
baseline. We construct the smallest such setting that still contains the
phenomenon: a partially observed linear-quadratic-Gaussian control problem in
which the policy-relevant state is hidden, the observation aliases it, and
every object the experiments test admits closed form.

## 3.1 Model and instances

The environment state is $s = (v, z) \in \mathbb{R}^2$, evolving in continuous
time as a damped oscillator driven by noise,

$$\dot s = A s + B u + \xi, \qquad
A = \begin{pmatrix} 0 & 1 \\ -c & -\gamma \end{pmatrix}, \quad
B = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad
\mathbb{E}[\xi \xi^\top] = \mathrm{diag}(0,\, 2\Theta\gamma c),$$

discretised exactly by zero-order hold at step $h$: $F = e^{Ah}$,
$G = \int_0^h e^{A\tau} B\, d\tau$, and process covariance
$\Sigma_d = \int_0^h e^{A\tau} W e^{A^\top\tau} d\tau$, all evaluated
numerically to machine precision rather than by Euler approximation. The agent
observes **only** $v$; the velocity-like coordinate $z$ is hidden. The
per-step cost is $h\,(q v^2 + r u^2)$ with $q = r = 1$, discounted at
$\beta = e^{-\rho h}$, $\rho = 0.05$, and we fix $\Theta = 1$, $c = 0.3$
throughout. Exploration is a pinned Gaussian on the action,
$u = \mu_\theta(v) + \eta$, $\eta \sim \mathcal{N}(0, \sigma_e^2)$ with
$\sigma_e = 0.3$ held fixed rather than learned: earlier rounds of this
project found that entropy collapse otherwise acts as an uncontrolled brake
that confounds the $\lambda$ reading, so exploration must be holdable
constant across every comparison.

Two timescales organise everything. The hidden coordinate decorrelates over
roughly $1/\gamma$ time units, that is $\tau_z \approx 1/(\gamma h)$
environment steps. The GAE($\lambda$) estimator bootstraps over an effective
horizon of $H_{\text{boot}} = (1 - \beta\lambda)^{-1}$ steps. The regime of
interest is their ratio, controlled experimentally through the single product
$\gamma h$. We instantiate four settings,

| instance | $\gamma$ | $h$ | $\gamma h$ | $\beta$ |
|---|---|---|---|---|
| g1\_h0.02 | 1 | 0.02 | 0.02 | 0.999000 |
| g1\_h0.05 | 1 | 0.05 | 0.05 | 0.997503 |
| g2\_h0.05 | 2 | 0.05 | 0.10 | 0.997503 |
| g4\_h0.05 | 4 | 0.05 | 0.20 | 0.997503 |

so that every predicted threshold moves monotonically across the family and
each prediction is tested at four locations, not one.

## 3.2 The two arms: making the effect's scope part of the design

The central confound this testbed exists to remove is that "memory helps" and
"the critic's bias distorts learning" are usually observationally identical.
They separate cleanly here because the critic's bias can only reach the
policy through one door. The GAE advantage is built from TD residuals
$\delta_t = r_t + \beta \hat V(v_{t+1}) - \hat V(v_t)$; a biased $\hat V$
perturbs the estimated advantage of action $a$ at state $v$ only through the
term $\mathbb{E}[\hat V(v_{t+1}) \mid v_t, a]$. If transitions are
action-independent, that term is the same for every action, the bias cancels
from all action preferences at every $\lambda$, and no critic defect of any
kind can move the policy. We therefore run every GLEAM experiment on two
arms:

- **coupled** — the dynamics above, where $u$ steers the state; and
- **decoupled** — identical in every respect except that $u$ enters the cost
  and not the dynamics, so the optimal gain is $0$ and the cancellation
  applies exactly.

Any effect that appears on the coupled arm and vanishes on the decoupled arm
is thereby attributed to the learning channel — the critic's bias acting
through action-dependent transitions — rather than to representation,
optimisation, or capacity, all of which are common to both arms. This
dissociation is registered as predictions P6.3 and P6.4 (section 5).

## 3.3 Exact baselines and the aliased equilibrium map

All reference quantities are generated by executing the model, and the paper's
release includes the generators. For a linear policy $u = -kv$, the closed
loop $F_k = F - G(k\;\ 0)$ has stationary covariance $\Sigma(k)$ solving a
discrete Lyapunov equation, giving the deployed cost rate
$J(k) = (q + rk^2)\,\Sigma_{11}(k)$ in closed form. From it: the best
memoryless policy $k_{\text{ml}} = \arg\min_k J(k)$ with cost
$J_{\text{ml}}$; the full-state optimum $J^\ast$ from the discrete Riccati
equation; and the large-gain plateau $J(k) \to r\,\Theta K(0)$ that bounds
what any high-gain memoryless controller pays.

The object the audit turns on is the **aliased equilibrium map**. A critic
restricted to functions of the observation — here, least-squares
TD($\lambda$) on features $(1, v, v^2)$, the LSTD($\lambda$) fixed point —
is biased whenever $v$ aliases states with different futures. The resulting
expected GAE($\lambda$) policy gradient is computable exactly: all lag
moments of the Gaussian closed loop are Isserlis (Wick) sums, and the
infinite bootstrap series collapse to matrix resolvents such as
$\sum_j (\beta\lambda)^j x_j = e_1^\top(I - \beta\lambda F_k)^{-1}\Sigma e_1$,
with no truncation anywhere. The largest root $k_{\text{AC}}(\lambda)$ of
this exact gradient is the fixed point of the averaged update — we say fixed
point, not limit, because part of what the experiments measure is precisely
whether training arrives there. On g1\_h0.05, $k_{\text{AC}}$ runs from
$24.79$ at $\lambda = 0$ to $1.51$ at $\lambda = 1$ against
$k_{\text{ml}} = 1.649$: an order-of-magnitude displacement of the learning
target, controlled entirely by $\lambda$. The machinery is validated at its
edges: at $\lambda = 0$ it reproduces the project's earlier TD(0) results to
the final digit, and its resolvent sums match brute-force truncated series to
six figures.

## 3.4 Two thresholds, not one

Two distinct critical values of $\lambda$ fall out of the same machinery, and
keeping them separate is essential to reading every experiment.

**The cost-law threshold** $\lambda_{1/2}$ is where the excess cost
$J(k_{\text{AC}}(\lambda))/J_{\text{ml}}$ has collapsed halfway from its
below-horizon plateau to its above-horizon floor; it is located by bisection
on the exact map and frozen per instance **before** the corresponding runs
(0.996552, 0.985382, 0.957977, 0.873138 for $\gamma h$ = 0.02 to 0.20).
Because the experiments can only sample $\lambda$ on an 8-point grid, we also
freeze the **grid-matched** targets — the same estimator applied to the exact
values on the same grid — so that measurement error and interpolation bias
are never conflated (section 6).

**The stability threshold** $\lambda^\ast$ concerns whether the equilibrium
is reachable at all. Extending the Wick machinery to policies with a nonzero
action mean, $u = -kv + b$, the averaged dynamics of $b$ near the parity
fixed point $b = 0$ have an exactly computable stability index
$S(\lambda, k)$; its derivation and validation are in section 4 [THEORY
SECTION CROSS-REF]. The destabilising term carries the factor
$(1-\lambda)/\lambda$ — the bootstrap weight itself — so the critic's bias is
admitted into the off-manifold policy directions in proportion to how much
the estimator bootstraps. The sign change of $S(\lambda,
k_{\text{AC}}(\lambda))$ defines $\lambda^\ast$ per instance
(0.9956, 0.9759, 0.9010, and none for $\gamma h$ = 0.02, 0.05, 0.10, 0.20),
frozen before the relevant data existed; below it the theory predicts
training does not converge and its quality is stopping-time dependent.

## 3.5 Training harness, policy classes, and measurement

All experiments run in one single-file PPO harness [PPO; GAE] differing only
in the policy head, with GAE($\lambda$) advantages, clipped surrogate
(clip 0.2, 4 epochs, 4 minibatches), Adam at $3 \times 10^{-4}$ annealed
linearly to zero, gradient-norm cap 0.5, 16 parallel environments, 128-step
rollouts, and 15M environment steps per run; the harness discount equals the
model's $\beta$ per instance. Policy classes: **mlp** (two Tanh layers of
128), **gru128** (GRU with 128 units) as the memory arm on external
benchmarks, and two restricted heads that exist for identification rather
than performance: **linear** ($u = -kv$, a single scalar read from the raw
observation, no bias, no trunk) and **linearb** ($u = -kv + b$). The linear
head cannot leave the gain manifold, so any instability it fails to show is
thereby located off that manifold; the linearb head adds exactly one
escape direction, the one whose stability the theory computes. Both restricted
heads support initialisation at a chosen gain, and linearb supports kicking
$b$ to a chosen $b_0$ at $t=0$, turning stability claims into
perturbation-decay measurements rather than waits for stochastic escape. All
new parameters are constant-initialised, and we verified the harness produces
bitwise-identical mlp and gru128 trajectories, rows and curves, before and
after these additions: the registered E6/E7/E9 data remains reproducible from
the released file.

**Deployed evaluation.** Reported cost is measured with the policy mean, no
exploration noise, on a single 200{,}000-step trajectory with a 1{,}000-step
burn-in discarded, giving the stationary cost rate $J_{\text{rate}}$. The
obvious alternative — averaging many short episodes — is wrong here in a way
that would have manufactured a registered result: episodes reset from the
uncontrolled stationary distribution, every episode pays the settling
transient, and the inflation is multiplicative and grows with the gain
(measured factors 1.088, 1.270, 2.415 at $k$ = 1.65, 5.99, 20.55), i.e. it is
largest exactly where the predicted effect is largest and $\lambda$ moves the
gain. A calibration test pins policies at known gains and requires the
measured-to-analytic ratio to be flat in $k$; it passes with spread
$1.6\times10^{-4}$ across gains, which is what licenses comparing
$J_{\text{rate}}$ to the frozen baselines at all. Two diagnostics are read
from every continuous run: the **fitted gain** (the OLS slope of $u$ on $v$
over the evaluation trajectory) and, for linearb, the terminal bias $b$.
Training-time behaviour is summarised by the **intermittency fraction**: the
fraction of late-half evaluation points whose cost exceeds
$2 J_{\text{ml}}$, which detects the recurrent-excursion phenomenology
(section 5) without reference to any equilibrium prediction.

**External benchmarks.** The corroboration arm (E9) uses POPGym
[POPGym] environments reimplemented natively — RepeatPrevious and
position-only CartPole — with the upstream seeding path reproduced exactly
(PCG64 via the same SeedSequence construction) and verified bitwise against
the reference implementation over 13{,}542 steps before any run. This removes
the cluster dependency chain entirely; jobs run on plain numpy/torch with no
per-node compilation.

## 3.6 Audit protocol

The claims are adjudicated by preregistration, and the preregistrations are
**executable**: frozen constants and scoring functions live in Python files
whose freeze is asserted mechanically — the deployment script runs every
prereg as a pre-flight and refuses to submit if any sentinel remains — so a
prediction cannot drift between freeze and verdict. Amendments are dated, and
are made before the relevant data exists except where explicitly disclosed
otherwise in the amendment text itself. Scorers for the experiments whose
results files accumulate across submissions (E6/E7/E9, E12, E13) refuse to
run on data with duplicated cells; the E12 guard was added post-data after
its grid re-ran the probe's two reference cells, an episode §6 reports along
with the verdict-neutral separation. Two departures from the discipline are
disclosed rather than smoothed over, because a pre-submission audit found
them and an audit paper does not get to hide its own findings: the scoring
code for P10.1–P10.3 was implemented after the data existed (the predictions
and constants were frozen on 2026-08-14; the file's advertised `--score`
mode was a stub until 2026-08-21, and the implemented scorer prints a
sensitivity table over the one aggregation choice its freeze had left
unspecified), and the diagnostic prereg (E8) lacked a mechanical freeze
assertion until the same date. Verdicts are published as registered: this
paper reports [N] registered failures alongside its passes, and two of its
analyses (sections 6.x, 6.y) are labelled post-hoc and carry no confirmatory
weight.
Because cluster nodes of different CPU generations produce bit-different
trajectories from identical seeds, the node class of every job is recorded
and reported; within-class runs reproduce bitwise. Across classes we measured
seven perturbed reruns: the largest difference is about 1.9 across-seed
standard deviations, the typical one far below one, and substituting the
affected values changes no registered verdict [APPENDIX CROSS-REF]. One
comparison inherits a hardware caveat from this: E9 ran entirely within the
d class while E6 and E7 both include t-class tasks, so P7.2 — an ordering
test across instances — compares cells that were not all produced on
bitwise-comparable hardware. P7.2 failed as registered, so no claim rests on
it; the heterogeneity is recorded because it is part of what that comparison
was. Seeds are 1–5 for GLEAM experiments (1–3 for E10 and E13), matched
across arms and architectures so that comparisons are paired at the seed
level.

## 3.7 Number provenance (drops to appendix or repo README at submission)

| constant in text | generated by |
|---|---|
| $F, G, \Sigma_d$, $J(k)$, $k_{\text{ml}}$, $J_{\text{ml}}$, $J^\ast$ | `vendor/gleam/theorem2_learning.py` (`disc_model`, `J_deploy`, `best_ml`, `J_opt`) |
| $k_{\text{AC}}(\lambda)$, LSTD($\lambda$) fixed point, resolvent identities | `vendor/gleam/lstd_lambda.py` (`k_eq_lambda`, `lstd_lambda_theta2`) |
| $\lambda_{1/2}$ per instance, grid-matched targets, $J(k_{\text{AC}})$ tables | `prereg_phase2.py` (REFERENCE block), `prereg_e10.py` (`C`) |
| $S(\lambda,k)$, $\lambda^\ast$ per instance | `vendor/gleam/bias_stability.py` (`bias_stability`, `predict_all`), frozen in `prereg_e7_bias.py` |
| eval inflation 1.088/1.270/2.415, calibration spread $1.6\times10^{-4}$ | `harness/test_eval_calibration.py`; prereg_phase2.py header note (dated 2026-08-12) |
| bitwise env verification, 13,542 steps | `harness/test_env_equivalence.py` |
| bitwise harness regression after linear/linearb | MEMO-arch-linear.md §6 (rows + curves, timing columns excluded) |
| $k_{\text{AC}}(0)=24.79$ on g1_h0.05 (the TD(0) value; constant in every raw row as `ref_k_AC`) | `vendor/gleam/gleam_bench.py` (`ac_equilibrium`, Props 1–3); reproduced by `lstd_lambda.py` at $\lambda=0$ |
| $k_{\text{AC}}(0.9)=20.55$, $k_{\text{ml}}=1.649$ on g1_h0.05 | `prereg_phase2.py` `K_AC_LAMBDA`, `BASELINES` |
| intermittency definition ($2J_{\text{ml}}$, late half, 0.05 flag) | `a5-tier1b-findings.md` §1 (frozen 2026-08-13); constants carried in `prereg_e7_bias.py` (`INTERMITTENCY_*`, `LATE_FRACTION`) |

<!-- TODO for section 5/6 integration:
  - [FIGURE: J(k_AC(lambda)) curves, four instances, lambda_half marked]
  - [FIGURE: the two-arm dissociation, E6]
  - [THEORY SECTION CROSS-REF] -> section 4 label
  - [APPENDIX CROSS-REF] -> node-provenance appendix label
  - [N] registered failures: count at final scoring (currently P6.1, P7.1-P7.3,
    plus pilot G2/G3 as design-stage) -->
