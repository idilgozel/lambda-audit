# 5. Results

<!-- Draft v1, 2026-08-21, written against the final ledger. Every number is
     quoted from a frozen scoring file's output; regenerate the tally with
     `python3 analysis/recount_ledger.py`. Target: 3 pages of the 9, less
     whatever the figures take.

     NOTE ON ORGANISATION, for Idil: this section is organised by CLAIM, not
     chronologically, which is what the theory-first framing needs. Section 1
     currently tells the story in the order it happened. Those two can coexist
     (narrative intro, evidence-ordered results), but §1's arc paragraphs will
     need compressing so the reader is not told the same story twice. Flagged
     rather than decided. -->

We report every registered prediction. Of 25 scored, 11 passed and 14 failed;
capability gates are tallied separately (7 passed, 2 failed) because a failed
gate means the predictions behind it are *not scored* rather than refuted.
The ledger is Table 3. The failures are not incidental to the results — three
of them are the mechanism by which the paper's scope claim is tested, and the
one that cost the most produced the paper's strongest prediction.

## 5.1 The confound, made exact

On the reference instance the aliased equilibrium map runs from
$k_{\text{AC}} = 24.79$ at $\lambda = 0$ to $1.51$ at $\lambda = 1$, against a
best memoryless gain of $k_{\text{ml}} = 1.649$: the learning target of the
memoryless baseline moves by an order of magnitude under a hyperparameter that
is rarely reported to two decimals. The map is a closed-form root of the exact
expected GAE($\lambda$) gradient, not a fit.

The learned agent's cost follows. On the coupled arm, cost at $\lambda = 0.9$
exceeds cost at $\lambda = 1.0$ by $2.63\times$ (P6.2, registered threshold
$1.25\times$), and at $\lambda = 1.0$ the learned gain lands at $1.663$
against $k_{\text{ml}} = 1.649$ — the memoryless baseline is only the *best*
memoryless policy at the top of the $\lambda$ range.

One registered prediction about *location* failed here and reshaped the rest of
the paper. P6.1 required the learned gain to land at $k_{\text{AC}}(\lambda)$
below the horizon; at $\lambda \in \{0.8, 0.95\}$ it landed nearer
$k_{\text{ml}}$, and cost missed its 15% band by 96–425%. Training does not
*arrive* at the computed fixed point below the horizon. That failure separated
two questions the design had conflated — where the equilibrium is, and whether
training reaches it — and forced the second threshold (§3.4) into the theory.

## 5.2 The cost law, tested prospectively

E7 tested the quantitative law by locating $\lambda_{1/2}$ from trained agents
across four instances. All three predictions failed: $\lambda_{1/2}$ within
0.01 on 1 of 4 instances (P7.1), the ordering in $\gamma h$ broken (P7.2), the
below-knee plateau within 10% on 1 of 4 (P7.3).

The diagnosis, performed on the landed data and labelled post-hoc: the
registered estimator conflated the cost law $J(k)$ with the assumption that
training arrives at $k_{\text{AC}}$. Every below-knee seed stalls at 6–50% of
its target, and the estimator anchored on cells contaminated by off-manifold
excursions. Decomposed, the law itself was untouched — across the 23
excursion-free cells, measured cost matched the analytic cost *of the gain each
run actually reached* with median ratio 0.994, 22 of 23 within 10%.

E10 tested this prospectively rather than arguing it. Placing a
one-parameter policy **at** $k_{\text{AC}}(\lambda)$ by construction removes
the reachability failure by design, and the registered prediction is the same
quantity E7 got wrong. P11.3: measured $\lambda_{1/2}$ falls within 0.01 of
its grid-matched analytic target on **all four** instances, with absolute
differences of 0.00021, 0.00014, 0.00101 and 0.00497 — a frozen point
prediction hit at four locations, at one to two orders of magnitude inside its
own tolerance. This is the paper's central empirical result.

## 5.3 The scope law, confirmed in both directions

The scope claim is that the confound requires actions to influence the hidden
state: when transitions are action-independent, the critic's bias cancels
identically from every action preference at every $\lambda$ (§3.2). Because it
predicts *absence*, it is testable in a way most mechanism claims are not.

**In-domain.** On the decoupled arm the largest learned gain across the whole
$\lambda$ grid is 0.0079 with spread 0.0091 (P6.3), against a coupled-arm
difference of 5.853 at $\lambda = 0.9$ — an arm ratio of 3.52 (P6.4). Same
representation, same optimiser, same evaluator; only the coupling differs.

**External, scope condition met.** On position-only CartPole, where the agent
applies force and the hidden velocity responds, all three registered
predictions passed: the memoryless architecture steps down below the horizon
(Easy $-0.0213$, 95% CI $[-0.0288, -0.0140]$; Hard $-0.0060$, CI
$[-0.0095, -0.0018]$), no step is resolvable above it (six comparisons, every
CI containing zero), and the memory architecture's across-seed variance
concentrates at low $\lambda$ by a pooled factor of 42.6 (P9.1–P9.3).

**External, scope condition not met.** Two POPGym game environments were
tested and neither shows the effect. On BattleshipEasy the capability gates
failed outright (memoryless and memory arms both at the floor, $-0.441$ and
$-0.446$), so P12.1–P12.3 are *not scored*. On MineSweeperMedium the gates
passed convincingly — our memoryless arm reproduces POPGym's published MLP
score to 0.012, and the memory advantage at $\lambda = 1.0$ is $+0.121$ — and
all three registered predictions then failed: the measured advantage at the
community default $\lambda = 0.95$ is $+0.091$, *smaller* than at
$\lambda = 1.0$, the opposite of the registered direction (P15.1), and no
threshold structure appears at the computed knee (P15.3).

We registered these as positive predictions and they failed as registered.
The reading, which the scope law itself supplies: in both environments the
latent state is fixed at reset and actions reveal information about it rather
than steering it, so the mechanism has no equilibrium to displace. That
distinction — actions steering the latent versus actions revealing it — was
not in our claim set when E12 was designed, and the τ used to place the knee
was an episode length rather than a relaxation time of a mean-reverting
coordinate. Both are our errors, and they are why P15 is reported as three
failures rather than reinterpreted. What survives is a boundary with evidence
on both sides of it: the effect appears where the theory says it should
(P9.1–P9.3) and is absent where the theory says it should be (G2, P15.1–P15.3),
on external benchmarks, under the same harness.

Independently, the released diagnostic (§5.6) finds no trace of the mechanism's
signature in E12's training curves: memoryless-arm excursions are at noise
level and flat across $\lambda$ (1, 0, 1 flagged runs of 5 below the knee
against 2, 1, 1, 0, 1 above), rather than concentrated below it. The
instrument reports the effect absent, not swamped.

## 5.4 Generality beyond the minimal instance

The experiments run on the smallest member of the model family, and three
independent generalisations bound how much that matters.

**Kernel and dimension.** The derivations never used the state dimension.
Recomputing the exact objects for Prony kernels
$K(\tau) = \sum_i c_i e^{-\gamma_i \tau}$ at state dimensions 2, 3 and 4 —
including non-Markovian observed dynamics — reproduces the same structure at
every kernel: an inflated equilibrium below a knee, a 14–16$\times$ collapse
across the $\lambda$ grid, and both thresholds. Two properties are *not*
kernel-universal, and we report them: $\lambda_{1/2}$ does not order by the
kernel correlation time across kernel shapes (it does within the
single-exponential family, which is P10.2), and the two-threshold ordering
$\lambda^\ast < \lambda_{1/2}$ inverts for a kernel with a slow minority mode.

**Nonlinearity.** E13 added a cubic damping the theory cannot absorb, sized as
10% and 50% of the dominant balance at the above-knee equilibrium, and asked
whether the *linearized* theory still locates the threshold. It does: measured
$\lambda_{1/2}$ is 0.985529 and 0.985542 against a linearized 0.983837 —
within 0.0017, at a registered tolerance of 0.005 — while the perturbation
moves the floor cost by a factor of 2.5 and *sharpens* the collapse
(profile ratio 1.34 $\to$ 1.82 $\to$ 3.38). The instrument control at
zero perturbation reproduces the linearized value to $1.8\times10^{-5}$
(G13, P16.1, P16.2).

**Drive distribution.** Reported in the companion derivation: at matched
autocovariance, the equilibrium *location* moves by $3.4\times$ across
Gaussian, bounded and kurtotic drives while the *cost at equilibrium* stays
within 1–7% of $r\Theta K(0)$. Cost levels and thresholds are
distribution-robust; equilibrium locations are not.

That last split is worth stating plainly, because it predicts this paper's own
ledger. Every location prediction that failed — P6.1, P11.1, P10.1 — is in the
fragile class; every cost or threshold prediction that passed — P6.2, P11.3,
P16.1, P16.2, and the analytic cost map holding to $\leq 1.05\%$ at every
measured gain in all 32 E10 cells — is in the robust class.

## 5.5 The mechanism: what is confirmed, and what is not

The off-manifold stability index $S(\lambda, k)$ carries the bootstrap weight
$(1-\lambda)/\lambda$ as its destabilising factor and predicts a second
threshold. It is validated against Monte Carlo to 1–3% at ten points. Its
experimental record is mixed, and the preregistration fixed in advance how to
read that.

What passed: the predicted onset falls with $\gamma h$ in the measured
ordering (P10.2); and in the perturbation-decay test, all 8 cells predicted
unstable read unstable (P11.4's unstable side, 8/8).

What failed: the onset bracket contains $\lambda^\ast$ on 2 of 3 prospective
instances, and the miss sits *below* the predicted bracket, which the frozen
file classifies as a hard failure rather than an incomplete mechanism
(P10.1). The predicted widening of the gap between the two thresholds does not
appear — measured gaps are 0.0066, 0.0054 and 0.0080 where the theory predicts
0.0010, 0.0095 and 0.0570 (P10.3). And 18 of 21 cells predicted stable read
stable, against a registered allowance of 19 (P11.4).

`e7-diagnosis-cost-law.md` §5, written before E10 ran, states that P11.4
failing beyond its allowance demotes the bias-direction mechanism "to a
suggestive calculation". We apply that rule. The mechanism is reported as
suggestive, not confirmed.

Two observations bound what the demotion means, and neither is offered as a
rescue. The three P11.4 misses are the three scored cells with the smallest
positive $S$ ($+2.3$, $+6.0$ and $+4.4 \times 10^{-4}$), each adjacent to its
instance's knee; and resolvability is asymmetric at this budget — growth
resolved down to $|S| = 1.3\times10^{-4}$ while decay resolved only at
$S \geq 1.1\times10^{-2}$, with perfect separation between the groups. The
registered floor was calibrated on a growth cell and applied symmetrically,
which is a design error we would not repeat. The minimal reading of the three
misses is "unresolved at budget"; a genuine sign error at $|S| < 10^{-3}$
cannot be excluded on these data.

Separately, P11.1 and P11.2 failed: 19 of 32 cells drifted outside the 15%
gain tolerance. The decomposition is verified and reported in §6 — all 96 runs
drifted downward (systematic, not diffusive), the analytic cost map holds to
$\leq 1.05\%$ at every measured gain including all failing cells, and failure
aligns with the branch of the equilibrium map rather than with $\lambda$.
P11.3 is estimated from the shape of that intact cost profile, which is why it
survives what P11.1 and P11.2 did not.

## 5.6 The diagnostic

The intermittency statistic is released as a tool practitioners can run on any
continuous-control log without adopting our theory. Its three registered
tripwires fired zero false positives (0 of 40 decoupled-arm runs, 0 of 40 on
the no-onset instance, 0 of 180 above-horizon memoryless runs), and its one
prospective specificity prediction passed on 50 runs with zero flags (P13.1).
Sensitivity against a privileged detector is 17 of 23; the tool is
deliberately blind to chronic elevation, which is documented as a scope
property rather than tuned away.

## 5.7 Ledger

<!-- Generated by analysis/recount_ledger.py; do not hand-edit. -->

| id | claim | verdict |
|---|---|---|
| G2 | POPGym gaps shrink at $\lambda = 0.9$ | **FAIL** |
| G3 | $\lambda$-sweep classifies memory-need | **FAIL** |
| P6.1 | below-horizon gain lands at $k_{\text{AC}}$ | **FAIL** |
| P6.2 | cost ratio $\geq 1.25\times$ | PASS (2.63) |
| P6.3 | decoupled arm flat in $\lambda$ | PASS |
| P6.4 | arm ratio $\geq 3$ | PASS (3.52) |
| P7.1–P7.3 | $\lambda_{1/2}$ from trained agents, 4 instances | **FAIL** ×3 |
| P9.1–P9.3 | memoryless step-down, absence above, variance | PASS ×3 |
| P10.1 | onset bracket contains $\lambda^\ast$ | **FAIL** (2/3) |
| P10.2 | onset ordering in $\gamma h$ | PASS |
| P10.3 | two-threshold gap widens | **FAIL** |
| P11.1–P11.2 | equilibrium holds, cost law at $k_{\text{AC}}$ | **FAIL** ×2 |
| P11.3 | $\lambda_{1/2}$ within 0.01, four instances | PASS |
| P11.4 | kicked-bias decay pattern | **FAIL** (26/29) |
| P13.1 | diagnostic specificity, prospective | PASS |
| P15.1–P15.3 | $\lambda$-dependence on MineSweeperMedium | **FAIL** ×3 |
| P16.1–P16.2 | $\lambda_{1/2}$ under cubic damping | PASS ×2 |

Capability gates, reported separately: G1a–c PASS (pilot), G11a–b **FAIL**
(E11; P12.1–P12.3 consequently not scored), G12a–c PASS (E12), G13 PASS (E13).
