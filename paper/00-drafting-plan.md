# Drafting plan: results, discussion, and the integrity fixes that must land first

Written 2026-08-21, for review before any paper file is edited. Nothing in
`paper/` has been touched.

Before planning the prose I had the repository audited independently against
the two paper drafts. That audit changed the plan, so it comes first.

---

## 0. Headline: the writing is not the binding constraint

Six findings would damage the paper more than any amount of good prose would
help it, and three of them are in the drafts right now. The order of work has
to be **fix, then draft** — not because the fixes are urgent in themselves,
but because two of them change what the results section can claim.

I verified every finding below myself rather than relaying it.

---

## 1. The blocking set

### 1.1 The drafts are stale in a way that inverts a result — BLOCKER

`paper/01-intro-arc.md` still carries `[E10 PENDING]` in the ledger and keeps
an `[IF E10 PASSES]` branch asserting that "the measured cost tracks
J(k_AC(λ)) across all 32 cells" and that the kicked perturbation "decays
exactly where the stability index says it decays."

E10 landed on 18 August. Three of its four predictions **failed**: P11.1 FAIL
(19 of 32 cells), P11.2 FAIL, P11.4 FAIL (18/21 stable, needed 19). Only
P11.3 passed. The pre-written pass branch asserts as confirmed two things the
data disconfirmed.

Worse, `e7-diagnosis-cost-law.md` §5 contains a *pre-committed* reading of
exactly this outcome: "If P11.2 or P11.3 fail: the cost law is wrong somewhere
reachability cannot excuse… If P11.4 fails beyond its allowance: the
bias-direction mechanism is not the driver at k_AC and Tier 2 is demoted to a
suggestive calculation." That rule was written before the data and has now
fired. Honouring it is not optional — it is the whole method.

My reading, which the drafting must reflect: the pre-committed rule was
written before we knew the failure mode, and `e10-above-knee-drift.md` now
shows the failures are branch-aligned harness bias, not map error (cost map
intact to ≤1.05% at every measured gain, 96/96 runs biased downward). The
honest resolution is to **honour the rule's letter** — demote the Tier 2
mechanism claim from "confirmed prospectively" to "confirmed on 26 of 29
scored cells with three misses at the three smallest positive S values, all
adjacent to the knee" — and then present the diagnosis as the reason the
demotion is not fatal. What we must not do is quietly keep the strong claim
because we later found an explanation.

### 1.2 The intro's arithmetic does not add up, and the true count is worse — BLOCKER

`01-intro-arc.md` L62-64 says "of the eighteen predictions we registered
across five experiments, six failed"; L157 says "six registered failures
reported alongside nine passes." 6 + 9 = 15 ≠ 18. `README.md` says a third
thing (13 passed, 9 failed).

The discrepancy is caused by counting capability *gates* as predictions when
they pass (pilot G1a–c) and omitting them when they fail (E11 G11a, G11b).
A hostile reviewer will find this in one minute.

**Proposed fix — separate the two tables and recount mechanically:**

*Registered predictions (20 scored):*

| verdict | count | ids |
|---|---|---|
| PASS | 10 | P6.2, P6.3, P6.4, P9.1, P9.2, P9.3, P10.2, P10.3, P11.3, P13.1 |
| FAIL | 9 | G2, G3, P6.1, P7.1, P7.2, P7.3, P11.1, P11.2, P11.4 |
| PARTIAL | 1 | P10.1 (2 of 3 prospective) |

*Capability gates, reported separately and symmetrically:* pilot G1a–c PASS;
E11 G11a, G11b FAIL (and P12.1–P12.3 consequently NOT SCORED); E12 G12a–c
PASS. Plus P15.1–P15.3 pending, taking the total to 23.

Ten passes against nine failures reads worse than "13 and 9" and is the
number I would put in the paper. The intro already frames it — "the discipline
was not free" — and a near-even split with the failures analysed is far more
credible than a favourable tally a grep can dismantle. **I want your explicit
agreement on this before I write it**, because it is the single most
consequential presentational choice in the paper.

I will also write `analysis/recount_ledger.py` so the intro's numbers are
generated from the scoring output rather than typed — the draft's own revision
note asks for exactly this ("script it") and the script does not exist.

### 1.3 The methodology claims are broader than the code — BLOCKER

Three statements in §3.6 / §2.2 / README are falsified by a five-minute grep.

| claim | reality |
|---|---|
| "the deployment script runs each prereg as a pre-flight" | `deploy.sh` runs phase2, e10, e11, e12. **`prereg_e7_bias.py` and `prereg_e8.py` are never executed by any script.** |
| "Scoring refuses to run on data with duplicated cells" | True in `prereg_phase2.py` only. e10, e11, e12, pilot, e7_bias, e8 have no duplicate guard — `prereg_e10.py`'s `cell()` takes a plain median over whatever matches. |
| "verdict strings are quoted from their output" (for P10.x) | `prereg_e7_bias.py` **has no `score()` function at all.** Its docstring advertises `--score F`; running it prints the prediction table and exits 0. P10.1/P10.2/P10.3 were never scored by the frozen file. |

Two further siblings: `prereg_e8.py` has **no freeze assertion anywhere**
(`grep -c _assert_frozen` → 0) while every other prereg has 3–7 — and it is
the one prereg whose tripwire population was redefined post-data under
Amendment 1, and it underwrites Contribution 3. `prereg_e10.py`'s
`FREEZE_MD5` is never compared to anything (the hash does currently match, so
this is presentational, but it is a "machine-checked freeze" that isn't).

**Two routes.** Either implement the missing pieces — `score()` in
`prereg_e7_bias.py`, `_assert_frozen()` in `prereg_e8.py`, a shared duplicate
guard, both files wired into `deploy.sh`/`collect.sh` — or weaken the prose to
what is true. I recommend implementing, for `prereg_e8.py` especially: it
takes hours, it is the claim reviewers will actually test, and writing
"scoring is executable except in two files" invites the question of why.

Note that implementing `score()` for P10.x is a **post-data scoring
implementation** and must be disclosed as such. The predictions themselves
(brackets, ordering) are frozen and dated; only the code that reads them is
late. That is disclosable, but it is weaker than everything else in the audit
and the paper should say so plainly rather than let it pass as equivalent.

### 1.4 Git history cannot corroborate the freeze order — historical assessment

**Correction, 2026-10-05:** the assessment below is superseded by
`PROVENANCE.md`. The job script prints the saved UTC and host fields; they
are not independent scheduler attestations of preregistration times. The
logs corroborate runs and result rows but do not repair missing early
freeze receipts. The original assessment is retained below as project history.

Everything from 7–18 August arrived in **one commit**, and that commit's
author date (2026-08-17 16:49) precedes its committer date (2026-08-20 22:23)
because it was amended. It therefore contains files asserting freeze dates
*after* its own author date — `prereg_e8.py` and `prereg_e11.py` both say
"FROZEN 2026-08-18" — plus the complete 192-row E10 dataset.

This is an ordinary consequence of `git commit --amend` on a squashed initial
import, not misconduct. But for a paper whose third contribution is "the audit
itself, as an artifact," a repository offered as proof of freeze order that
contains a visible chronological impossibility is the finding most likely to
sink it.

**The fix is that git was never the right evidence, and better evidence is
already committed.** The cluster logs carry UTC start times written by
Myriad's scheduler, with monotonically increasing SGE job IDs:

| experiment | job ID | first task start (UTC) | governing freeze |
|---|---|---|---|
| E6 | 127988 | 2026-08-12T13:22:19Z | prereg_phase2, 2026-08-12 |
| E9 | 133126 | 2026-08-13T11:08:28Z | prereg_phase2, 2026-08-12 |
| E7 | 137245 | 2026-08-14T10:19:49Z | prereg_phase2 2026-08-12, e7_bias 2026-08-14 |
| E10 | 156928 | 2026-08-17T10:23:01Z | prereg_e10, 2026-08-14 |
| E11 | 183670 | 2026-08-20T20:28:53Z | prereg_e11, 2026-08-18 |
| E12 probe | 186912 | 2026-08-21T13:31:22Z | prereg_e12, 2026-08-21 |

Every freeze precedes its data, the job IDs are monotone, and none of these
timestamps were written by us. This is *stronger* provenance than commit
dates. Proposed: a `PROVENANCE.md` stating that git dates before 21 August are
uninformative and why, presenting this table as the actual chronology, and
committing every future freeze in its own commit before submission. An
appendix carries the short version.

### 1.5 `MEMO-arch-linear.md` is missing from the repo — BLOCKER, already recoverable

`03-method.md` §3.5 claims "we verified the harness produces bitwise-identical
mlp and gru128 trajectories, rows and curves, before and after these
additions," citing `MEMO-arch-linear.md §6`. That file **is not in the repo and
is not in any commit** — while `harness/ppo_lambda.py` (17 Aug) postdates
`results_phase2.jsonl` (14 Aug), so the claim is load-bearing for whether E6/E7/E9
are still reproducible.

I have the file. It is in my working directory from the session that wrote it,
12,286 bytes, and its §6 records the regression:

```
GLEAM_g1_h0.05_gru128_lam0.9_s1              ROW IDENTICAL
GLEAM_g1_h0.05_mlp_lam0.9_s1                 ROW IDENTICAL
curves_..._gru128_lam0.9_s1.csv    CURVE IDENTICAL apart from sps
curves_..._mlp_lam0.9_s1.csv       CURVE IDENTICAL apart from sps
```

I can restore it to the repo immediately. Recommended additionally: re-run
that regression now, at current HEAD, and store the output as a file — a memo
recording a past check is weaker than a check anyone can re-run.

### 1.6 §3.6 rounds a number toward its own conclusion — BLOCKER

`03-method.md` L218-220: "the one cross-class perturbation we measured is
smaller than one seed standard deviation and moves no verdict."

`provenance-node-class.md` L63-64, the cited source: "the largest node-swap
difference is about **1.9 seed standard deviations** and the typical one is far
below one." It was also **seven** runs, not one.

"Moves no verdict" is supported by the substitution table. But in a paper about
measurement integrity, rounding 1.9σ down to "<1σ" against one's own cited
document is the worst possible place for it. Must be corrected verbatim.

Related, and currently unstated: E6 ran 61 of 80 tasks on the t00a class while
E9 ran none, so **P7.2 — an ordering test across instances — spans two
non-bitwise-comparable hardware classes.** That belongs in limitations.

---

## 2. Second tier — serious, fix during drafting

- **Two wrong provenance rows in §3.7.** k_AC(0)=24.79 is sourced to
  `prereg_phase2.py K_AC_LAMBDA`, which has no λ=0 entry (index 0 is λ=0.8 =
  22.43); the real generator is `lstd_lambda.py` / `gleam_bench.ac_equilibrium`.
  And the intermittency definition is sourced to `prereg_e10.py`, which
  contains only prose — the constants live in `prereg_e7_bias.py` L143-145.
  Fix both, then write `analysis/regen_provenance.py` so the table cannot rot.
- **Curves are not in the repo, and some scored verdicts need them.** 836 CSVs,
  0 committed. **P13.1** (a published PASS) and E8's T1/T2/T3 tripwires are
  computed from curves and are *not* reproducible from a clean clone — which
  contradicts `.gitignore`'s own comment that "every scored statistic is
  reproducible from the results JSONL plus the harness." Fix the comment; publish
  the curves as a Zenodo/Release asset with a DOI, as that comment already proposes.
- **No code computes the P10.x onset brackets or the 30/32 path test.** Those
  numbers exist only in markdown. Same for the 23-clean/9-intermittent split.
- **Numbers that are prints, not assertions:** "13,542 steps" appears nowhere
  in the repo outside the two paper files; "six figures" is a format-string
  precision at a single validation point; the calibration spread 1.6e-4 sits
  against a test tolerance of 0.02, 125× looser. Convert each to an assert with
  a tolerance, or stop quoting them to that precision.
- **Contradictory documents left live.** `phase2-design.md` still puts
  k_AC=24.794 at λ=0.9 (wrong; `prereg_phase2.py` L26-37 records the error but
  the design doc was deliberately not edited), and the same constant is
  hard-coded in `harness/build_gleam_instances.py` → propagates to `ref_k_AC`
  in all 80 released GLEAM rows. `e6-findings.md` §3 is explicitly corrected by
  `e6-curves-analysis.md` and was never edited. Add correction headers to both.
- **No LICENSE, no version pins, no CITATION** — while the intro promises
  "GLEBench, released." `vendor/gleam/README.md`'s file table omits
  `bias_stability.py`, which generates λ* and underwrites Contribution 2.
- **`prereg_phase2.py` prints "E6 rows: 360 / expected 80"** — it counts all
  rows against the E6-only expectation. Scoring is correct; the first line a
  reviewer sees says the data doesn't match the design.

---

## 3. Proposed structure: §5 Results

Written in the order the experiments happened, matching the intro's arc.

**5.1 The pilot's thesis fails; the control arm is born.** G1a–c pass as
gates; G2 fails (gap at λ=0.9 is 1.495 against a required <0.373 — it did not
shrink); G3 fails (balanced accuracy 0.476). One-sided expectation dies. State
plainly that POPGym reference settings sit *above* the threshold, which is why
the effect is absent there — this is the scope law's first appearance, not an
excuse.

**5.2 The minimal pair localises the channel (E6).** P6.3 (decoupled flat,
|k| ≤ 0.0079), P6.4 (arm ratio 3.52), P6.2 (cost ratio 2.626) pass; P6.1 fails
with the gain landing nearer k_ml than k_AC at λ ∈ {0.8, 0.95}. The failure
splits one threshold into two — this is the hinge of the paper and gets the
most space of any single result.

**5.3 The first mechanism dies by its own abort criterion (A-5).** Kramers
escape falsified by exact computation; the discriminating experiment; S(λ,k)
derived with the (1−λ)/λ factor; MC validation to 1–3% at ten points — with the
caveat from §2 that only five of those ten have a released entry point.

**5.4 Corroboration on an external benchmark (E9).** P9.1 (Easy −0.0213, CI
[−0.0288, −0.0140]; Hard −0.0060, CI [−0.0095, −0.0018]), P9.2 (all six
above-horizon CIs contain zero), P9.3 (pooled variance ratio 42.56). All pass.

**5.5 The quantitative law fails, and the failure is the best data point
(E7).** P7.1 fails 1/4, P7.2 fails, P7.3 fails 1/4. Then the decomposition:
the registered test conflated the cost law with the assumption that training
*arrives*; every below-knee seed stalls at 6–50% of target; on the 23
excursion-free cells measured cost matches analytic cost at the gain actually
reached, median ratio 0.994, 22/23 within 10%. Labelled post-hoc, no
confirmatory weight, with E10 named as its prospective version.

**5.6 The prospective in-domain test (E10).** P11.3 **passes** — λ_half within
0.01 of grid-matched analytic targets on all four instances (0.00021, 0.00014,
0.00101, 0.00497). This is the paper's strongest result and should be
presented as such: a quantitative point prediction, frozen, prospective,
four locations. P11.1, P11.2, P11.4 fail, with `e10-above-knee-drift.md`'s
verified decomposition — 96/96 runs biased downward, cost map intact to
≤1.05%, failure branch-aligned rather than λ-aligned, three candidate causes
excluded by test. P11.4 reported as 26/29 with the three misses at the three
smallest positive S, each adjacent to its instance's knee, and the
resolvability asymmetry (growth resolves to |S|=1.3e-4, decay only at
S ≥ 1.1e-2) reported as a design lesson about the floor.

**5.7 A diagnostic, released (E8).** Tripwires T1 0/40, T2 0/40, T3 0/180;
sensitivity 17/23; P13.1 armed prospectively and passing on 50 runs with zero
flags. Amendment 1 disclosed as post-data in the same breath.

**5.8 External validity (E11, then E12).** E11 in full: gates failed, P12
not scored, 119 core-hours spent, and the design error named — a gate above
the state of the art on an environment whose own paper says memory doesn't
help there. Then E12, with the corrected protocol (gates read on 2 runs before
the 80-run grid) and its outcome.

### The E12 fork — pre-written, exactly as the E10 fork was

- **All of P15 passes** → the headline is that POPGym's published rankings are
  λ=1.0 rankings and the memoryless arm specifically degrades below the
  computed knee. This is the paper's practical contribution and moves to §1.
- **P15.1 + P15.2, P15.3 fails** → effect and mechanism confirmed, predicted
  location wrong; report the observed bracket against 0.97643.
- **P15.1 only** → per the frozen reading, the ordinary variance effect, *not*
  this paper's mechanism, and the paper must not cite it as mechanism evidence.
  §1's practical claim narrows to "the measured value of memory depends on λ"
  with the mechanism resting on in-domain evidence alone.
- **P15.1 fails** → the practical claim does not generalise to this benchmark.
  Reported. The paper becomes an in-domain theory paper with an honest external
  negative, and §1's fourth contribution is rewritten accordingly.

I will write all four branches now, as with E10, so the section needs a choice
rather than a rewrite when the grid lands.

Regardless of branch: our gru scores −0.0546 against POPGym's −0.006, so our
gap (+0.0918) is ~60% of theirs. This must be stated with its answer — our
*mlp* reproduces theirs to 0.012, and the scripted safe-expansion anchor
(−0.1523, computed before any run) predicted it — rather than left for a
reviewer to find.

---

## 4. Proposed structure: §6 Discussion and limitations

1. **What the audit licenses and what it does not.** Everything prospective and
   passing is in-domain on constructed instances; the theory is
   linear-quadratic and MineSweeper is neither, so the bridge to POPGym is the
   τ→knee analogy, not a derivation. Say this before a reviewer does.
2. **Undertraining as the live alternative** for low-λ cells, and the existing
   defence: G2 and the decoupled arm show no effect where the scope law says
   none, which pure undertraining would not respect.
3. **The three E10 failures**, from §1.1 above, honouring the pre-committed
   demotion rule.
4. **Measurement caveats:** node-class 1.9σ stated correctly; P7.2's
   cross-class span; the 5-seed budget; 15M steps.
5. **What we would do differently:** capability gates before grids (E11's
   lesson, costed at 119 core-hours); equivalence tests rather than
   CI-contains-zero (E12's pre-data lesson); tolerances in absolute rather than
   relative units where the target is small (E10's lesson).
6. **The audit as artifact**, with the provenance chronology from §1.4 and an
   honest account of what the git history does and does not show.

---

## 5. Figures

Existing: `fig_e6_a/b/c` (pdf+png) from `analysis/figures_e6.py`. Needed:

1. **J(k_AC(λ))/J_ml against λ, four instances, λ_half marked** — the cost law
   and P11.3. The paper's money figure. Already flagged as a TODO in §3.7.
2. **The two-arm dissociation (E6)** — coupled vs decoupled, gain against λ.
   Partly exists in fig_e6.
3. **S(λ,k) sign map with k_AC(λ) overlaid**, four instances, showing where the
   two thresholds separate and where the P11.4 misses sit relative to the knee.
4. **E10 drift**: measured gain vs k_AC(λ), 32 cells, coloured by branch — makes
   the branch-alignment finding visible in one panel.
5. **E12**: gap(λ) for mlp and gru with the knee marked at 0.97643.

Every figure script goes in `analysis/` and regenerates from committed JSONL,
except (1) and (3) which are pure theory and regenerate from `vendor/gleam/`.

---

## 6. Order of work, and what I need from you

**Phase A — integrity (before I touch `paper/`).** Restore
`MEMO-arch-linear.md` and re-run its regression at HEAD; write `PROVENANCE.md`
with the cluster-log chronology; correct the 1.9σ sentence and the two §3.7
rows; write `recount_ledger.py`. Then implement `score()` in
`prereg_e7_bias.py`, `_assert_frozen()` in `prereg_e8.py`, and a shared
duplicate guard — each disclosed as post-data where it is.

**Phase B — draft §5 and §6** against the corrected ledger, with all four E12
branches pre-written.

**Phase C — when E12 lands:** choose the branch, update §1's fork, regenerate
the ledger, build figures.

**Three decisions I need from you:**

1. **The recount** (§1.2). Do we publish "10 pass / 9 fail / 1 partial of 20
   registered predictions, gates reported separately"? I recommend yes. It is
   the honest number and it is defensible; the current headline is not.
2. **Implement or weaken** (§1.3). Build the missing scoring and freeze checks,
   or narrow the prose to what the code does? I recommend building, and
   disclosing the late implementation.
3. **The Tier 2 demotion** (§1.1). Do we honour `e7-diagnosis-cost-law.md` §5's
   pre-committed rule and demote the mechanism claim to "suggestive," then
   argue back from the drift diagnosis? I recommend yes — the rule is the
   method, and a pre-committed rule obeyed only when convenient is worth
   nothing.

Nothing in `paper/` will be edited until you have answered these.
