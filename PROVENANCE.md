# Provenance: what establishes that each freeze preceded its data

Written 2026-08-21, after a pre-submission audit found that this repository's
git history cannot corroborate the preregistration chronology it is offered as
evidence for. This document states the problem plainly and presents the
evidence that does establish the chronology.

## 1. What is wrong with the git history

Everything produced between 7 and 18 August 2026 — every preregistration,
every findings document, `results_phase2.jsonl`, all 192 rows of
`results_e10.jsonl`, and both paper drafts — arrived in a single commit,
`be167fc`. That commit was amended, so its recorded dates disagree:

```
$ git log --format='%h A:%ad C:%cd' --date=iso
dff4175 A:2026-08-21 20:40:54 +0300 C:2026-08-21 20:40:54 +0300
a9ae7a5 A:2026-08-21 16:10:58 +0300 C:2026-08-21 16:10:58 +0300
be167fc A:2026-08-17 16:49:44 +0100 C:2026-08-20 22:23:03 +0300

$ git reflog --date=iso | sed -n '3,5p'
be167fc HEAD@{2026-08-20 23:03:33 +0300}: Branch: renamed refs/heads/main to refs/heads/main
be167fc HEAD@{2026-08-20 22:23:03 +0300}: commit (amend): Lambda audit: ...
59efef8 HEAD@{2026-08-17 16:56:20 +0100}: commit (initial): Lambda audit: ...
```

The consequence is a visible chronological impossibility: `be167fc` carries an
author date of 17 August while containing `prereg_e8.py` and `prereg_e11.py`,
both of which assert `FROZEN 2026-08-18`.

There is no dispute about what happened. The repository was created on 17
August by importing an existing working tree that had been developed locally
since 7 August; the commit was amended on 20 August to add files and fix the
commit message, and `git commit --amend` preserves the original author date
while updating the committer date. Nothing was backdated deliberately and no
content was altered to fit a narrative.

But intent is not the point. **A repository offered as proof of freeze order
should not require the reader to take the author's word for why its dates look
impossible.** For a project whose methodological claim is that predictions were
frozen before the data existed, that is a real defect, and it is recorded here
rather than left for a reviewer to discover.

Git dates in this repository establish nothing about anything before
2026-08-21. Every commit from 2026-08-21 onward is made at the time it claims,
and each preregistration freeze from that date forward is committed on its own
before the corresponding job is submitted.

## 2. What does establish the chronology

The cluster is the witness. Every job on UCL Myriad writes a log carrying a
UTC start time and a host line, stamped by the scheduler rather than by us,
and SGE job IDs increase monotonically across the cluster's whole user base —
they cannot be chosen, reordered, or backdated by a submitter. Those logs are
in `results/logs/` and are deliberately **not** gitignored, precisely because
they carry this record.

| experiment | SGE job | tasks | first task start (UTC) | last finish (UTC) | governing freeze | margin |
|---|---|---|---|---|---|---|
| capability probe | 113384 | 2 | 2026-08-08T14:47:18Z | 2026-08-08T18:23:02Z | `prereg_lambda_pilot.py` 2026-08-08 | same day |
| pilot grid | 119762 | 240 | 2026-08-10T15:29:58Z | 2026-08-11T07:47:19Z | `prereg_lambda_pilot.py` 2026-08-08 | 2 days |
| E6 | 127988 | 80 | 2026-08-12T13:22:19Z | 2026-08-13T06:08:06Z | `prereg_phase2.py` 2026-08-12 | same day |
| E9 | 133126 | 160 | 2026-08-13T11:08:01Z | 2026-08-13T16:50:00Z | `prereg_phase2.py` 2026-08-12 | 1 day |
| E7 | 137245 | 120 | 2026-08-14T10:19:38Z | 2026-08-14T12:59:05Z | `prereg_phase2.py` 2026-08-12; `prereg_e7_bias.py` 2026-08-14 | 2 days / same day |
| E10 | 156928 | 192 | 2026-08-17T10:23:01Z | 2026-08-18T12:39:10Z | `prereg_e10.py` 2026-08-14 | 3 days |
| E11 | 183670 | 80 | 2026-08-20T20:28:53Z | 2026-08-21T09:08:47Z | `prereg_e11.py` 2026-08-18 | 2 days |
| E12 probe | 186912 | 2 | 2026-08-21T13:31:21Z | 2026-08-21T16:40:06Z | `prereg_e12.py` 2026-08-21 | same day |

Job IDs are strictly increasing in start time (113384 < 119762 < 127988 <
133126 < 137245 < 156928 < 183670 < 186912), which independently fixes the
order of the experiments.

Regenerate the table:

    bash analysis/provenance_chronology.sh

### Where this evidence is weaker

Three freezes are same-day with their first run: the pilot capability probe
(2026-08-08), E6 (2026-08-12), and the E12 probe (2026-08-21). For those, the
cluster log establishes the day but not the hour, because the freeze itself
carries only a date. Same-day is still freeze-before-data on the documentary
record, but it is not the three-day margin E10 has, and it should not be
presented as though it were.

The second mechanism, `--check-clean`, is stronger in kind and weaker in
evidence: `prereg_e10.py --check-clean`, `prereg_e11.py --check-clean` and
`prereg_e12.py --check-clean` each refuse to certify a freeze if rows for that
experiment already exist, and `myriad/deploy.sh` runs the preregistration as a
submission pre-flight and refuses to submit if it fails. That is a genuine
guard against freezing after the fact, but it **leaves no artifact** — a
passing check prints to a terminal and is gone. From 2026-08-21 the output of
each `--check-clean` is committed alongside the freeze.

## 3. Node class, and why it is recorded

Myriad nodes span CPU generations. Runs on the same letter class reproduce
bitwise from identical seeds; runs across classes do not. Measured 2026-08-14:
of 40 `PositionOnlyCartPoleEasy` cells run twice, the 33 that stayed within the
d class (d00a and d00b interchangeably) reproduced bitwise and the 7 that
touched t00a did not. The letter is the boundary; the digits are not.

Class spread per experiment, from the same logs:

| experiment | node classes used |
|---|---|
| pilot | d00a, d00b, t00a |
| E6 | d00a, d00b, t00a |
| E9 | d00a, d00b |
| E7 | d00a, d00b, t00a |
| E10 | d00a, d00b, t00a |
| E11 | d00a, t00a |
| E12 probe | d00a |

The largest observed node-swap difference is **about 1.9 across-seed standard
deviations**, with the typical difference far below one (`provenance-node-class.md`
§3). Substituting the affected values changes no registered verdict, which is
the claim the paper makes; the magnitude is stated as measured rather than
rounded down.

**A limitation this creates, stated here because it is not otherwise obvious:**
E9 ran entirely within the d class while E6 and E7 both include t00a tasks. P7.2
is an *ordering* test across instances, so it compares cells that were not all
produced on bitwise-comparable hardware. P7.2 failed as registered, so this does
not rescue a claim — but the comparison's hardware heterogeneity is part of the
record either way.

## 4. What a reviewer cannot reproduce from this repository

Stated so it is not discovered instead.

- **Training curves.** 836 `curves_*.csv` files, 245 MB, excluded by
  `.gitignore`. The scored statistics that depend on them are **P13.1** and the
  E8 tripwires T1/T2/T3 (`prereg_e8.py` reads curves directly), and the
  intermittency inputs to P10.1–P10.3. Everything else — G1a–c, G2, G3,
  P6.1–P6.4, P7.1–P7.3, P9.1–P9.3, P11.1–P11.4, G11a/b, G12a–c — is computed
  from the committed JSONL and reproduces from a clean clone. The curves will
  be published as a versioned archive with a DOI and cited in the paper.
- **Exact bitwise reproduction of any published run**, unless the reviewer has
  access to a Myriad node of the originating class running torch 2.1.0+cpu.
  `harness/test_arch_regression.py` performs the comparison and reports the
  host and node class so that a mismatch cannot be silently misread as a
  harness change.
- **Environment pinning.** The cluster environment is selected and proven by
  `myriad/bootstrap_myriad.sh`, which probes candidate modules, runs a smoke
  test with a pass threshold, and writes `env.sh`. Selected for all runs
  recorded above: `pytorch/2.1.0/cpu`, torch 2.1.0+cpu, numpy 1.26.4,
  python 3.9.6.
