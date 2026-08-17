# λ-audit plan — the flagship audit artifact

Date: 2026-08-07. Follow-up to high-profile-targets.md §1. Template: Agarwal et al. 2021 (audit + tool, venue second).

Decisions taken (2026-08-07, Idil):
1. Pilot subject: POPGym subset.
2. Compute: pilot on Myriad CPU. GPU decision deferred until the pilot verdict.
3. Timing: pilot starts now, in parallel with the preprint announcement and MAPPO #336.
4. Context: the main preprint is submitted and awaiting announcement.

---

## 0. Fact base, corrected today

The targets file said memory comparisons run "almost always at 0.95". That needed checking per subject. Verified 2026-08-07:

- **POPGym (arXiv:2303.01859, Table 2): GAE λ = 1.0.** Also γ = 0.99, lr 5e-5, batch 65,536, 15M steps, max episode 1,024, BPTT length ∞, 3 trials per model. All 13 architectures shared these settings.
- **CleanRL ppo_atari_lstm.py: gae_lambda = 0.95**, γ = 0.99, num_steps 128 (confirmed in source).
- RLlib PPO default λ is 1.0. SB3 RecurrentPPO default is 0.95. Both need a permalink check in Phase 0.
- ICLR Blogposts Track: the 2026 deadline was Dec 7, 2025. Expect the 2027 call late 2026, deadline early Dec. The track has a COI rule: a post must not highlight or advertise the authors' own past publications.

**Consequence.** POPGym is not an instance of the 0.95 confound. It is the control arm. The reference benchmark for memory architectures ran above the knee. Much of common practice runs below it. That is a cleaner and more defensible thesis than "the benchmark is wrong."

## 1. The claim under audit

Restated two-sided:

> Architecture rankings for memory in RL are not transportable across the bootstrap horizon. POPGym ranked architectures at λ = 1.0, where the aliased fixed point does not gate learning. Common practice compares architectures at λ = 0.95 or below. Below the critical horizon the comparison measures the λ artifact, not memory. We locate the transition, and in the solvable model we have it in closed form.

What we do NOT claim: that POPGym's published rankings are wrong at their own settings. That the closed form transports verbatim to arbitrary tasks. The public claim is that the diagnostic locates the transition empirically and the theory explains it where the model applies.

Falsifier: if rankings are λ-invariant across the grid on envs where memory matters, the flagship thesis dies. Pre-commitment: the negative audit gets published anyway. An audit that can only find one answer is not an audit.

## 2. The precipice shape, our version

1. **The audit.** Rerun a POPGym subset across λ. Ranking statistics via rliable itself (IQM, stratified bootstrap CIs). Using their tool in an audit shaped after their paper is both correct practice and the right signal.
2. **The tool.** A small package: λ-sweep protocol, bootstrapped-vs-MC gap estimator, empirical knee locator. Tells a user whether their task sits below the knee before they compare architectures.
3. **Distribution.** Short arXiv note, ICLR blogpost, data-backed issues on the audited repos.

## 3. Phase 0 — desk work, no compute (this week)

- [ ] **Verify every λ claim from primary sources with permalinks.** popgym repo config, cleanrl source, sb3-contrib source, rllib defaults. The audit's central table is "who compares memory at what λ" and it must be bulletproof. Extend the survey: Memory Gym, DreamerV3-style baselines, the recurrent off-policy line (Ni et al. 2022; one-step targets are the extreme case, note only).
- [ ] **Choose the env subset.** Criteria: (a) envs where their results show memory clearly mattering (RepeatPrevious, Autoencode, Concentration candidates); (b) one env where it did not, as a second control; (c) hidden-state correlation time must be estimable, since the knee prediction needs it per env. Target 2 envs for the pilot, 4-6 for the full audit.
- [ ] **Fix the architecture set.** MLP, frame stack (one small and one large window), GRU (their top RNN). Nothing else in the pilot.
- [ ] **Harness decision (taken): clean single-file PPO, CleanRL style,** with stacking and recurrent variants. Their baseline stack is legacy RLlib with unmaintained dependencies. We document every delta from Table 2 and match batch, lr, γ, and step budget where feasible. The claim is about the comparison protocol, not their code. Claude builds the harness in the sandbox. Smoke test locally, then Myriad.
- [ ] **Runtime estimate.** One smoke run per architecture on CPU to size the array. 15M steps on vector-obs envs with small nets should sit in minutes-to-hours per run. Confirm before promising a grid.
- [ ] **Prereg file, executable.** Frozen predictions and gates in a script, never a comment (E4 lesson). Predictions from theory: (i) at λ = 1.0 the published qualitative ranking reproduces; (ii) below the knee the spread compresses toward the memoryless level or scrambles; (iii) the transition concentrates near λ_crit ≈ 1 − O(γ_hidden·h), estimated per env from the correlation time.

## 4. Phase 1 — pilot (target 2 weeks, Myriad CPU)

Grid: 2 envs × 4 architectures × λ ∈ {0.9, 0.95, 0.99, 1.0} × 5 seeds = 160 runs, 15M steps each. Curves saved for every run, no exceptions. Tail the first .o file two minutes after submit.

Gates, frozen before launch:
- **G1 sanity.** λ = 1.0 reproduces the published qualitative ranking on the chosen envs. Fails → harness problem. Fix before reading anything else.
- **G2 effect.** At 0.95 or 0.9 the ranking moves beyond seed noise by rliable CIs. This is the flagship gate.
- **G3 mechanism.** The bootstrapped-vs-MC gap estimator flags the below-knee runs, blind to architecture.

Verdicts: G1+G2+G3 pass → expand to Phase 2. G2 fails on POPGym → the flagship reframes. Fallback subjects where the artifact is proven in our own record: velocity-only MuJoCo (Walker, infrastructure exists) and the CleanRL Atari pair. G3 fails with G2 passing → the effect is real but our detector is wrong. That is itself a finding. Investigate before shipping any tool.

## 5. Phase 2 — full audit (Sept-Oct, compute decision here)

- Expand the POPGym subset to 4-6 envs, add the second control env.
- Add the velocity-only MuJoCo leg. ppo_torch.py already has the flags and the Myriad scripts exist.
- **Decide the Atari leg.** Only now does the GPU question matter. Options: Myriad GPU queue, or the #276 LSTM speedup PR first (4x, touches the exact script, buys standing with the maintainer), or drop Atari to an existence-proof single pair. #477's nine flagged envs partition test rides on this decision too.
- Re-check compute mode: if CPU arrays stay sufficient, skip GPU setup entirely.

## 6. Phase 3 — artifacts (Oct-Nov)

- [ ] **Tool release.** Small pip package, standard vocabulary name, no coinages. Contents: sweep protocol, gap estimator, knee locator, rliable wrappers, README stating the horizon rule. Repo public from day one.
- [ ] **arXiv note.** 6-10 pages. Cites the main preprint for the mechanism, stands on the audit data alone. Endorsement should be automatic this time: once the main preprint is announced, Idil has cs.LG authorship plus the UCL email, which satisfies the post-Jan-2026 automatic rule. Verify when submitting.
- [ ] **Blogpost (ICLR 2027).** Deadline expected early Dec 2026. The COI rule constrains framing: the post audits community practice, the data is the object, our preprint appears once in related work. Read the 2027 call the day it opens. If the rule bites harder than expected, fallbacks: a benchmarks/repro workshop, or the note plus issues alone.
- [ ] **Data-backed issues.** CleanRL #477 (the below-knee partition of his nine envs, if the Atari leg ran). POPGym repo (report the transportability result, framed as extension not correction). tianshou #470 and rl_games #333 are already queued with the preprint link per the issues map.

## 7. Sequencing against everything else

- **Now:** preprint announcement lands. Outreach phase E starts per arxiv-plan.md. That plan is separate and unchanged.
- **Parallel, part-time:** MAPPO #336 (Idil). Phase 0 desk work and the pilot harness (Claude). Myriad submissions are Idil's and stay spaced out.
- **Deferred:** #276 PR until the Phase 2 Atari decision. Its standing value is real but it is only load-bearing if the Atari leg happens.
- **Freeze for the blogpost draft: end of October.** Draft in November. Deadline early December. The pilot must reach a verdict by early September for this to be comfortable.

## 8. Risks and honesty rules

- The closed form is proven in the solvable model. In public: the diagnostic locates the knee empirically, the theory predicts it where the model applies. Never claim the formula transports verbatim to Atari.
- POPGym reruns happen in our harness, not their RLlib stack. Say so prominently. Deltas documented in a table.
- A single-author audit shaped after a five-author award paper must be tight. Smallest complete unit first: one benchmark, two-sided, plus the tool. Breadth goes in v2.
- Public text: standard vocabulary only. Bootstrap horizon, state aliasing, biased expected update, critical λ. "Knee" stays internal; write "transition" or "critical horizon" in anything public.
- Preregistration discipline continues. Frozen gates in executable files, committed before launch.
- git push after every session. Curves saved always.
