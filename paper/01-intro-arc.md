# 1. Introduction: auditing a hidden coupling in memory-vs-memoryless RL

<!-- Draft v1, 2026-08-14. The arc is written as it happened, with dates, and
     every verdict below is quoted from a frozen scoring file, not from memory.
     E10 verdicts are PENDING at draft time: both continuation variants are
     provided at the marked fork, so this section does not need rewriting
     when E10 lands — only a choice.

     Verification aids: section 2.3 is a complete ledger of every registered
     prediction with its freeze date, scoring file, and verdict. If a claim in
     sections 1-2 is not traceable to that ledger or to the provenance table
     in 03-method.md section 3.7, it does not belong in the paper. -->

When a recurrent agent beats a memoryless one on a partially observed task,
the comparison is read as evidence about *memory*: the task needed it, the
architecture supplied it. A large empirical literature, and most practical
architecture decisions, rest on comparisons of exactly this shape. This paper
audits a hyperparameter coupling hidden inside them.

Both agents in such a comparison are typically trained with actor-critic
methods whose advantage estimator, GAE($\lambda$) [GAE], bootstraps the
critic over an effective horizon of $(1-\gamma\lambda)^{-1}$ steps. Under
partial observability the critic is restricted to functions of the
observation, so it is biased wherever the observation aliases states with
different futures — and the bootstrap horizon determines how much of that
bias the policy gradient inherits. The horizon is set by $\lambda$, a
parameter tuned by community default (0.95 in most codebases; 1.0 in the
POPGym reference results [POPGym]) and almost never varied in memory studies.
The quantity it silently interacts with — the correlation time of the hidden
state — is the very thing a memory benchmark manipulates. The memoryless
baseline in a memory comparison is therefore not a fixed object: its training
target moves with $\lambda$, and, as we show, below a computable threshold it
stops being reachable at all.

**Claims, in decreasing order of strength.** *Formal:* for an exactly
solvable family of aliased control problems, the fixed point of the averaged
actor-critic update under a biased (observation-restricted) TD($\lambda$)
critic is computable in closed form, and it is displaced from the best
memoryless policy by an amount controlled by $\lambda$ — on our reference
instance, from $k_{\text{AC}} = 24.8$ at $\lambda = 0$ to $1.51$ at
$\lambda = 1$ against a true optimum of $1.65$, an order of magnitude of
displacement from a hyperparameter that is rarely reported to two decimals.
*Mechanistic:* the same machinery, extended off the policy manifold, yields a
stability index whose destabilising term carries the factor
$(1-\lambda)/\lambda$ — the bootstrap weight itself — predicting a second,
distinct threshold below which training does not converge and its quality is
stopping-time dependent. *Scope:* both effects require actions to influence
the hidden state; when transitions are action-independent the critic's bias
cancels identically from every action preference, so the confound has a
computable boundary, not a vibe. *Practical:* the community default
$\lambda = 0.95$ sits below the computed threshold on the slow-hidden-state
members of our family, which is precisely the regime memory benchmarks are
built to probe.

**Method: an audit, not a benchmark run.** Every substantive claim in this
paper was frozen as an executable preregistration — constants and scoring
functions in Python files whose freeze is machine-checked at deployment —
before the data that tests it existed, with dated amendments and with
failures reported as failures. We committed to this discipline because the
subject of the paper is a *measurement pathology*, and an audit of a
measurement pathology that permitted itself flexible analysis would be its
own counterexample. The discipline was not free: of the eighteen predictions
we registered across five experiments, six failed as registered, and the
paper reports every one. Three of those failures are the most informative
results we have.

**The arc.** We tell it in the order it happened, because the order is the
evidence that the process worked.

*The pilot's thesis failed first* (G2, G3). Our initial, one-sided
expectation — that lowering $\lambda$ would shrink measured memory-vs-
memoryless gaps on POPGym — was wrong on the standard benchmark, whose
reference results use $\lambda = 1.0$ and whose gaps proved robust. The
thesis became two-sided: the confound lives below a threshold that standard
POPGym settings sit above. The pilot became the control arm.

*The minimal pair localised the channel* (E6). On the solvable family, with
transitions decoupled from actions, learned gains are flat in $\lambda$ to
$|k| \le 0.008$ (P6.3); coupled, the arm difference at $\lambda = 0.9$ is
$3.5\times$ its value at $\lambda = 1$ (P6.4), and cost at 0.9 exceeds cost
at 1.0 by $2.6\times$ (P6.2). But the point prediction P6.1 — that the
learned gain lands *at* the computed fixed point below the horizon — failed
as registered: training never arrives there. That failure split one
threshold into two and forced the mechanism question.

*The first mechanism died by its own abort criterion* (A-5). Noise-driven
escape of the gain over its quasi-potential barrier — the natural first
story — was falsified by exact computation: escape resistance moves the
wrong way in $\lambda$ under every noise model we could defend. A
discriminating experiment (a policy that *cannot* leave the gain manifold
shows none of the instability that a 2-layer MLP shows at matched
everything) located the true mechanism off the manifold, and the off-manifold
stability index $S(\lambda, k)$ was then derived in closed form and validated
against Monte Carlo to 1–3% at ten points.

*The external benchmark corroborated* (E9, all three predictions passing as
registered): on position-only CartPole, the memoryless architecture steps
down in performance below the horizon (Easy: $-0.021$, 95% CI
$[-0.029, -0.014]$), the step is absent above it, and the memory
architecture's across-seed variance concentrates at low $\lambda$ exactly as
recurrent-excursion phenomenology predicts.

*The quantitative law failed as registered, and the failure was the most
valuable data point of the project* (E7). All three predictions about the
location and shape of the cost collapse failed. The diagnosis, on the landed
data: the registered test conflated the cost law $J(k)$ with the assumption
that 15M steps of PPO *arrives* at the computed fixed point. Below the
horizon it never does — every seed of every below-knee cell stalls at 6–50%
of the target — and the registered estimator anchored itself on cells
contaminated by the off-manifold instability. Decomposed, the law itself
survives untouched: across the 23 cells free of excursions, measured cost
matches the analytic cost of the gain each run actually reached with median
ratio 0.994 (22/23 within 10%); and the mechanism, asked the dynamically
correct question — does the climb path toward the fixed point pass through
the computed instability region — predicts which 32-of-32 cells go unstable
with 30 concordant, both misses at the two smallest predicted growth rates.
The frozen bracket predictions for the instability onset scored two of three
prospectively, with the ordering and two-threshold structure both holding.

<!-- ============ E10 FORK: keep exactly one branch ============ -->

*[IF E10 PASSES]* *The law and the mechanism were then confirmed
prospectively, in-domain* (E10). With the policy placed at the computed fixed
point by construction — removing the reachability failure by design rather
than by argument — the equilibrium holds, the measured cost tracks
$J(k_{\text{AC}}(\lambda))$ across all 32 (instance, $\lambda$) cells, the
half-collapse point lands within 0.01 of its grid-matched analytic target on
all four instances, and a kicked one-parameter perturbation decays exactly
where the stability index says it decays and persists where it says it grows
([N]/29 cells). The audit ends where an audit should: with the registered
instrument's failure diagnosed to a named assumption, and the underlying law
passing a sharper preregistered test than the one it originally failed.

*[IF E10 FAILS — P11.2 or P11.3]* *The rescue itself then failed
prospectively* (E10), and the honest conclusion is that the equilibrium cost
law is wrong somewhere reachability cannot excuse. We report the audit as an
instrument that worked: it caught its own favourite theory twice.

<!-- ============ end fork ============ -->

**Contributions.**

1. **The confound, made exact.** An analytically solvable family (GLEBench,
   released) in which the $\lambda$-dependence of the memoryless baseline is
   closed-form, including the fixed-point map $k_{\text{AC}}(\lambda)$, the
   cost law, and both critical thresholds — with a two-arm design that gives
   the confound a computable scope boundary (action-dependent transitions)
   and a built-in negative control.
2. **A mechanism with a location.** The off-manifold stability index
   $S(\lambda, k)$, derived in closed form, carrying the bootstrap weight
   $(1-\lambda)/\lambda$ as its destabilising factor; validated against Monte
   Carlo; its threshold predictions tested prospectively (2/3 bracket hits;
   30/32 path-test concordance; perturbation-decay confirmed in the training
   harness [E10 PENDING]).
3. **The audit itself, as an artifact.** Executable preregistrations with
   machine-checked freezes, dated amendments, six registered failures
   reported alongside nine passes, and a released diagnostic (the
   intermittency statistic and the its-own-linear-fit ratio) that
   practitioners can run on any continuous-control log without our theory.
4. **A practical warning with numbers.** On slow-hidden-state tasks, the
   default $\lambda = 0.95$ can sit below the instability threshold, where
   "memoryless baseline" performance is stopping-time dependent — and above
   the same threshold, an apparently converged baseline can sit at a fixed
   point displaced by an order of magnitude with modestly elevated cost, a
   regime our diagnostic flags and pure return curves do not.

# 2. The audit design

## 2.1 Claim hierarchy and falsifiers

The audit separates three claims so each can fail alone. The *displacement
claim* (the fixed point moves with $\lambda$) is tested by E6's coupled arm
against exact references, with the kill criterion that no $\lambda$-
dependence at 5 seeds ends the project (registered; not triggered). The
*threshold claim* (two computable critical $\lambda$s, one for cost, one for
stability) is tested by E7 across four instances and by the frozen bracket
predictions of the mechanism file. The *scope claim* (no action-dependent
transitions, no effect) is tested by the decoupled arm (P6.3, P6.4) and by
E9's memory architecture, which the theory says should be flat. Each
registered prediction names its own falsifier in its freeze file; the
scoring never moved after data existed, and the two post-hoc analyses in
this paper (the E7 decomposition and the path test, §6) are labelled as
post-hoc and assigned no confirmatory weight — the prospective version of
both is E10.

## 2.2 Design rules

Four rules, adopted after this project's own early mistakes and enforced
mechanically thereafter. (i) Preregistrations are executable and their freeze
is a deployment precondition. (ii) Environments are reimplemented natively
and verified bitwise against upstream before any run (13,542 steps,
RNG-path-exact), so cluster toolchains cannot silently change the object
under study. (iii) The evaluation estimator is calibrated against exact
references before it is allowed to score anything; the episodic-reset
estimator it replaced inflated cost by up to $2.4\times$, growing with gain
— i.e. confounded with the treatment — and the correction is dated before
any Phase-2 data existed. (iv) Provenance is recorded rather than assumed:
every number in the paper maps to a generating file (§3.7), and every
cluster job's node class is logged because cross-generation CPUs are not
bitwise comparable (App. [X]).

## 2.3 The ledger

Every registered prediction, its freeze date, and its verdict. Scoring files
are released; verdict strings are quoted from their output.

| id | claim (compressed) | frozen | verdict |
|---|---|---|---|
| G1a–c | pilot capability & gap preconditions (POPGym) | 2026-08-08 | PASS ×3 |
| G2 | gaps shrink at $\lambda=0.9$ (POPGym) | 2026-08-08 | **FAIL** |
| G3 | $\lambda$-sweep classifies memory-need, bal. acc. $\ge$ 0.75 | 2026-08-08 | **FAIL** |
| P6.1 | below-horizon gain lands at $k_{\text{AC}}(\lambda)$, cost within 15% | 2026-08-12 | **FAIL** |
| P6.2 | cost at $\lambda=0.9$ exceeds $\lambda=1$ by $\ge 1.25\times$ | 2026-08-12 | PASS (2.63×) |
| P6.3 | decoupled arm flat in $\lambda$ | 2026-08-12 | PASS |
| P6.4 | arm difference ratio $\ge 3$ | 2026-08-12 | PASS (3.52) |
| P7.1 | $\lambda_{1/2}$ within 0.01, four instances | 2026-08-12 | **FAIL** (1/4) |
| P7.2 | $\lambda_{1/2}$ ordering in $\gamma h$ | 2026-08-12 | **FAIL** |
| P7.3 | below-knee plateau at $r\Theta K(0)$ within 10% | 2026-08-12 | **FAIL** (1/4) |
| P9.1 | memoryless step-down below horizon (2 variants) | 2026-08-13 | PASS |
| P9.2 | no step above horizon | 2026-08-13 | PASS |
| P9.3 | memory-arch variance concentrates at low $\lambda$ | 2026-08-13 | PASS |
| P10.1 | instability-onset brackets, 3 prospective instances | 2026-08-14 | 2/3 (miss: $\gamma h{=}0.02$) |
| P10.2 | onset ordering in $\gamma h$ | 2026-08-14 | PASS |
| P10.3 | onset strictly below $\lambda_{1/2}$, gap widens | 2026-08-14 | PASS |
| P11.1–.2 | equilibrium holds & cost law at $k_{\text{AC}}$, 32 cells | 2026-08-14 | [E10 PENDING] |
| P11.3 | $\lambda_{1/2}$ within 0.01 of grid-matched targets | 2026-08-14 | [E10 PENDING] |
| P11.4 | kicked-bias decay/persistence pattern, 29 cells | 2026-08-14 | [E10 PENDING] |

Also registered and honoured: the A-5 abort criterion (the Kramers/escape
mechanism was falsified by exact computation and Tier 2 as designed did not
run; its replacement was frozen before its test data existed), and the E6
kill criterion (not triggered).

<!-- Notes for revision pass:
  - "eighteen predictions ... six failed": recount at final ledger state and
    make the intro numbers match the table mechanically (script it).
  - [N]/29 in the E10-pass branch: fill from prereg_e10.py score output.
  - Tone check on "not a vibe" (sec 1, scope claim) - keep if the venue can
    take one colloquialism, else "not a heuristic".
  - POPGym lambda=1.0 claim: cite Table 2 of the POPGym paper explicitly.
  - Consider moving the ledger to sec 2 opening for maximum early credibility.
  - The Agarwal et al. positioning sentence (audit-of-practice lineage) should
    go in Related Work, not here; intro already long. -->
