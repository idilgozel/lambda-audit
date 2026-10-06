# Provenance: what the preserved evidence establishes

Originally written 2026-08-21. Corrected after a local evidence audit on
2026-10-05; snapshot completed at 2026-10-05T12:41:10.205414+00:00.
Myriad is no longer accessible. This account is based on saved files and
local Git history, not a fresh query of scheduler accounting.

The audit matched every saved Myriad result row to its exact JSON line in
a preserved job log. It also found that the earlier account overstated
what those logs establish about preregistration times. The corrections below
change provenance claims, not frozen predictions, thresholds, or verdicts.

## 1. Limits of the initial Git history

Files declaring freezes between 8 and 18 August first appear together in
`be167fc`, alongside results from experiments they govern. The commit records
author time `2026-08-17T16:49:44+01:00` and committer time
`2026-08-20T22:23:03+03:00`. The earlier project record explains this as an
initial import followed by an amended commit. Its author date cannot be used
as a timestamp for all of its contents: `prereg_e8.py` and `prereg_e11.py`
declare 18 August freezes, later than that author date.

For those early experiments, the preserved commit does not independently
establish the preregistration file's contents before its governed data existed.
Declared dates and retrospective chronology remain evidence of what was
recorded, with that limitation. Later separate commits provide more specific
ordering evidence, described in section 3; local Git timestamps are not an
external timestamping service.

## 2. Saved Myriad runs and result integrity

The audit found 958 logs carrying Myriad host names, containing
956 result rows. All those rows match saved JSONL
rows exactly, including timing fields; no result row is missing its log and
no logged result is missing from the examined JSONL files. Each result file
has unique `(env, arch, lambda, seed)` cells. Recorded arguments and seeds
agree with the result fields checked by the audit. All Myriad logs record
torch `2.1.0+cpu` and numpy `1.26.4`.

| job | SGE ID in filename | logs / result rows | first start UTC | last recorded finish UTC | node classes | declared freeze |
|---|---|---|---|---|---|---|
| lam_cap | 113384 | 2 / 2 | 2026-08-08T14:47:18Z | 2026-08-08T18:23:02Z | d00b | `prereg_lambda_pilot.py`: 2026-08-08 |
| lam_pilot | 119762 | 240 / 240 | 2026-08-10T15:29:58Z | 2026-08-11T07:47:19Z | d00a, d00b, t00a | `prereg_lambda_pilot.py`: 2026-08-08 |
| lam_e6 | 127988 | 80 / 80 | 2026-08-12T13:22:19Z | 2026-08-13T06:08:06Z | d00a, d00b, t00a | `prereg_phase2.py`: 2026-08-12 |
| lam_e9 | 133126 | 160 / 160 | 2026-08-13T11:08:01Z | 2026-08-13T16:50:00Z | d00a, d00b | `prereg_phase2.py`: 2026-08-12 |
| lam_e7 | 137245 | 120 / 120 | 2026-08-14T10:19:38Z | 2026-08-14T12:59:05Z | d00a, d00b, t00a | `prereg_phase2.py`: 2026-08-12; `prereg_e7_bias.py`: 2026-08-14 |
| lam_e10 | 156928 | 192 / 192 | 2026-08-17T10:23:01Z | 2026-08-18T12:39:10Z | d00a, d00b, t00a | `prereg_e10.py`: 2026-08-14 |
| lam_e11 | 183670 | 80 / 80 | 2026-08-20T20:28:53Z | 2026-08-21T09:08:47Z | d00a, t00a | `prereg_e11.py`: 2026-08-18 |
| lam_e12p | 186912 | 2 / 2 | 2026-08-21T13:31:21Z | 2026-08-21T16:40:06Z | d00a | `prereg_e12.py`: 2026-08-21 |
| lam_e12 | 187384 | 80 / 78 | 2026-08-21T17:58:48Z | 2026-08-21T22:09:50Z | d00a, d00b, t00a | `prereg_e12.py`: 2026-08-21 |
| lam_e12 | 212503 | 2 / 2 | 2026-08-25T21:38:46Z | 2026-08-26T01:45:47Z | d00a | `prereg_e12.py`: 2026-08-21 |

The E12 grid's original job `187384` has no result or finish marker for tasks
61 and 62. Both cells have completed logs under job `212503`, and those
replacement rows match the final grid file. The original incomplete logs
remain present. Their headers alone do not establish why execution stopped;
the scorer's existing note describes a node failure.

The E12 probe and full grid intentionally have separate result files, following
the disclosed post-data repair: the grid repeated the probe's two reference
cells. Each file is internally duplicate-free; these are distinct runs, not
extra seeds in a single scored cell.

Regenerate the chronology and exact result-to-log checks:

    bash analysis/provenance_chronology.sh
    python3 analysis/audit_saved_evidence.py

The script now groups by source, job name **and job ID**, exposes missing
result/finish markers, and reports local runs separately. The earlier script
merged both E12 grid jobs into one 82-log row labelled `187384`, hid the later
job ID, omitted E13, and mislabelled local E14's timestamp-based run label as
an SGE ID. A last-recorded-finish value is not a claim that every task finished.

### What a saved job log can and cannot prove

`myriad/pilot_array.qsub.sh` prints `hostname` and `date -u` itself. These are
job-script output preserved in SGE-style log files, **not independently signed
scheduler timestamps**. Matching stdout to JSONL establishes consistency
between saved artifacts, not independent authentication of either artifact.
No scheduler accounting export or external timestamped freeze receipt was
located in the examined repository/history and related local remnants.

The observed cluster job IDs increase with first recorded starts in this
dataset, but IDs do not date preregistration contents. A date-only freeze
declaration on the same day as a run does not establish within-day ordering.
The previous wording that scheduler stamps prove every freeze preceded data
is withdrawn. This evidence limitation does not by itself show that a freeze
occurred late or that results were altered.

## 3. Freeze declarations versus preserved pre-run commits

| experiment | declared date | first preserved prereg commit, UTC | first preserved run, UTC | what this supports |
|---|---|---|---|---|
| E12 | 2026-08-21 | `a9ae7a5`, 2026-08-21 13:10:58 | Myriad probe, 2026-08-21 13:31:21 | commit precedes saved probe start by 20 min 23 s |
| E13 | `FROZEN_ON = 2026-08-21`; header defines freeze at commit | `12ccb86`, 2026-08-25 19:52:20 | local run, 2026-08-25 19:55:31 | commit precedes saved run start by 3 min 11 s |
| E14 | 2026-09-11 | `9d24003`, 2026-09-13 16:38:27 | local probe, 2026-10-05 12:10:39 | committed prereg precedes the preserved local probe |

The respective experiment's result file is absent from its first prereg commit.
That is a statement about the Git tree; it does not prove absence from every
working directory or machine. E13 and E14 prereg files are byte-identical to
their first preserved versions at this audit. E12 has disclosed post-data
amendments; those are not erased or reclassified here.

E13's draft date and `FROZEN_ON` constant are retained unchanged. Under its
own header's commit-based rule, the preserved freeze event is 25 August, not
21 August. The earlier findings document's claim to have been written after
scoring on 21 August is inconsistent with the saved run chronology; a dated
correction is included there. E14's 11 September declaration and 13 September
commit must likewise be distinguished.

`--check-clean` examines a supplied results path when called. It does not
reconstruct whether data existed at an earlier claimed freeze time. No saved
positive `--check-clean` output was located in the examined tracked files or
history, so the earlier claim that such output was committed with every
freeze from 21 August is withdrawn. A current successful check would not
repair the missing historical receipt.

## 4. Local experiments are separate evidence

E13 was explicitly designed to run locally. `results/logs/e13_host.txt`
records `Idils-MacBook-Pro-3.local`, Python `3.13.5`, torch `2.9.1`, and the
25 August start. All 72 saved result rows match the 72 per-run logs. That
runner does not write finish markers, so exact last-finish time is not
available from those logs; it must not be invented from file modification
times or the first-run start. The G13 calibration gate is reported separately
from the two P16 predictions.

At the audited snapshot, E14 had four local probe logs and two completed
MLP result rows; the two GRU cells had no result/finish markers yet. Its logs
and `e14_provenance.json` identify a Mac with Apple M5 Pro, Python `3.12.14`,
torch `2.14.1`, and numpy `2.5.3`. Run label `202610051310` is generated from
local time by `analysis/run_e14.py`; it is not an SGE job number. The recorded
prereg MD5 matches the preserved E14 prereg file. No Myriad-host E14 log was
found in the searched local evidence. This does not establish that no
uncollected Myriad E14 run ever existed. The active local campaign was not
launched, restarted, stopped, or scored as complete by this audit.

## 5. Node classes and remaining reproducibility limits

The node classes in section 2 are read from the saved host lines. The separate
`provenance-node-class.md` analysis reports that same-class reruns reproduce
bitwise, while seven measured cross-class reruns differ, the largest by about
1.9 across-seed standard deviations without changing a registered verdict.
This audit verifies the saved host/version fields and row correspondence; it
does not rerun that numerical experiment. E9 uses the d class, while E6/E7
span d and t classes; P7.2 therefore compared heterogeneous hardware.

Training curves are ignored by Git and must be included in any release that
claims direct reproduction of curve-based scoring (P10.1--P10.3, P13.1, and
E8 diagnostics). At inventory time 2026-10-05T12:44:01.023436+00:00, the local directory
contained 980 curve files (420,957,822 bytes); this is an
inventory of a changing working directory, not a fixed published release.
Every run identifier in the result snapshot had a same-named curve file,
but a filename is not sufficient to establish which experiment produced it.

There are 66 run identifiers shared across result files: 40 between pilot and
E9, 24 between E10 and E13, and two between the E12 probe and grid. The harness
uses those identifiers in curve filenames. In particular, all 24 E10/E13
shared curve files end at 14,999,552 steps, consistent with the 15M-step E10
budget and incompatible with the 1M-step E13 runs as their complete histories.
The preserved result rows remain separately traceable to their logs, but
these curve files must not be represented as distinct original curves for
both experiments. Original per-experiment curves cannot be reconstructed
from final result rows alone. E13's registered scorer uses its result JSONL,
so this curve-provenance limitation does not alter its verdicts.

Exact training reruns require the originating hardware/software conditions;
Myriad access is no longer available. The saved artifacts retain substantial
checkable evidence, but they cannot replace missing historical authentication,
uncollected jobs, or distinct curves lost through shared filenames.

## 6. Local campaign of 5–6 October 2026 (E14 grids, E15, E16, E18, E19)

All five experiments ran on the same node class as the E14 probe of §4:
`Idils-MacBook-Pro.local`, Apple M5 Pro (18 cores), macOS 26.6, Python
`3.12.14`, torch `2.14.1`, numpy `2.5.3`, one torch thread per run, 8 runs in
parallel (14 for the first part of the E14 grid, see below). Each runner
appends a provenance record (`results/logs/e1N_provenance.json`) at launch with
the git commit, the `git status --short` output, and the harness MD5; every
run has a per-run log with start and finish markers. Times below are local
(BST) as written by the campaign logs.

| experiment | prereg (`FROZEN_ON`, MD5 at scoring) | harness `ppo_lambda.py` MD5 | launched | finished | rows / failures | verdict line |
|---|---|---|---|---|---|---|
| E14 grid | 2026-09-11, `9e0347e7` | `77c89e53` (unpatched) | probe 13:10:39, grid from 13:42:26 in the same process at 14 workers (`run_e14_campaign.log`); killed after the 14:41–15:26 sleep with 18 rows complete; relaunched 15:35:46 at 8 workers, completed cells kept | Easy 18:02:46, Hard 22:10:36 | 162 / 0 | `results/logs/run_e14_campaign_b.log` |
| E15 | 2026-10-05, `db2cb37a` | `1199ad2f` (final_sigma patch applied 22:11:03) | 22:11:03 (05 Oct) | 11:09:32 (06 Oct) | 40 / 0 | `results/logs/run_e15_campaign.log` |
| E16 | 2026-10-05, `a8d2eaa7` | `1199ad2f` | 11:09:33 | 16:17:34 | 120 / 0 | `results/logs/run_e16_campaign.log` |
| E18 | 2026-10-05, `f081cd35` | `45de828e` (five patches applied 16:18:17; `gleam_native.py` `14997426`) | 16:18:17 | 17:00:22 | 40 / 0 | `results/logs/run_e18_campaign.log` |
| E19 | 2026-10-05, `8debb6e4` | `45de828e` (runner `analysis/run_e19.py` `a6824da5`) | 18:06:07 (re-run; run 1 17:00:22–17:52:07 archived unscored, see below) | 18:57:41 | 48 / 0 | `results/logs/run_e19_campaign_b.log` |

**No pre-run commit exists for E15, E16, E18 or E19.** The repository HEAD
stayed at `9d24003` (13 September) throughout the campaign; every provenance
record lists the prereg files as untracked. What supports the freeze
declarations is therefore weaker than for E12–E14 and is listed rather than
asserted: the `FROZEN_ON` constants and the scorers' refusal to run when they
are unset; the synthetic-scenario tests (`analysis/test_prereg_e1N.py`) that
were written and run before each freeze; the file modification times
(E15 13:59:30, E16 14:09:39, E18 20:22:55, E19 21:14:16 on 5 October, all
before the respective first run, but modification times are not evidence of
content); and the project notes in the Claude project "GLEAM", which carry
server-side timestamps and describe each experiment's gates and predictions
(E15/E16 at 15:39 BST on 5 October) or record the prereg MD5s (E18/E19, 5
October evening) before the corresponding data existed. A reader who requires a
committed pre-run prereg must treat E15–E19 as not having one.

**One harness version per experiment.** The harness was modified twice during
the campaign, both times by an idempotent patch script between experiments,
never while an experiment was running: `analysis/apply_final_sigma_patch.py`
(records `final_sigma` for learned-σ runs; applied by `evening_campaign.sh`
after E14 exited) and the five E17/E18/E19 patches
(`apply_e17_patch.py`, `apply_linear_sgd_patch.py`, `apply_episode_len_patch.py`,
`apply_critic_fit_patch.py`, `apply_oracle_critic_patch.py`; applied by
`night_campaign.sh` after E16 exited, each logged with its MD5 in
`results/logs/night_campaign.log`). Each patch adds a command-line option that
is off by default; E18 and E19 use `--gleam-episode-len`, `--log-critic-fit`
and (E19) `--oracle-critic`. No experiment's result file mixes harness versions.

**Amendments.** E18 carries two dated pre-data amendments in its file header
(Amendment 1: diagnostic critic-fit row fields, not scored; Amendment 2: one
instance, so P22.2 is NOT SCORED by construction). E15, E16 and E19 have none.
Nothing was amended after data.

**E19 was run twice; the first run was never scored.** Run 1 (17:00:22–17:52:07,
48 rows, 0 failures) produced `VERDICTS: {}` because `analysis/run_e19.py`
formatted the registered start gains with `:g` (six significant figures:
28.905 for 0.5 × k_AC = 28.904965, and so on) while the frozen scorer matches a
row to its cell with an absolute tolerance of 1e-6; every cell therefore had
zero seeds and the scorer printed no outcome quantity. The runner was corrected
(`:.6f` with a round-trip assertion; `analysis/run_e19.py` MD5 `a6824da5`), the
prereg was not edited, and run 1 was moved whole to
`results/results/e19_run1_unscorable/` with a README. Run 1 may be reported as
an unregistered replicate (start gains within 5e-5 of the registered ones); it
is not E19 evidence.

**Wall-clock stalls.** The Mac slept twice while runs were in flight: on 5
October 14:41–15:26 (battery; the 14-worker E14 grid was killed and relaunched
at 8 workers at 15:35, completed cells kept by the duplicate guard) and from
about 22:30 on 5 October to 10:54 on 6 October (lid closed on battery after the
runs had been paused with SIGSTOP). Stalled processes resumed where they were;
the `done in … min` figures in `run_e15_campaign.log` for tasks 17–24 include
the sleep. No result row was affected: a run that is stopped or asleep does not
advance and writes nothing.
