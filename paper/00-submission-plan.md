# ICLR 2027 submission plan

Written 2026-09-10, 15 days out. Supersedes the scheduling in
`00-drafting-plan.md`; that file's §5/§6 structure and integrity findings still
stand and are referenced below.

Sole author. LaTeX built in this repository. 6–8 hours/day available.

## The clock

| gate | date | what must exist |
|---|---|---|
| **Abstract** | **Fri 18 Sept, 11:59pm AOE** | title, abstract, author registered. No author additions after this. |
| **Paper** | **Fri 25 Sept, 11:59pm AOE** | 9 pages + unlimited references, appendices, AI statement |

Both confirmed on iclr.cc, 2026-09-10.

## Policy compliance — check these off, they carry desk rejection

- [ ] **Reciprocal reviewing.** Every submission needs an author registered to
      review ≥3 papers; eligibility requires an accepted publication at
      ICLR/NeurIPS/ICML/JMLR/TMLR/ACL/CVPR/ICCV/AAAI/IJCAI/KDD/COLT-class
      venues as of 18 Sept. arXiv preprints do not qualify. **The carve-out:**
      each author may appear on *one* paper where no author is eligible. This
      submission uses that carve-out — so it must be the only ICLR 2027
      submission with Idil as an author. Registration is prompted after the
      abstract deadline; failing to register is desk rejection.
- [ ] **AI-use statement.** Mandatory, in-paper, does not count toward the page
      limit, and also declared on the submission form. Required-disclosure
      categories that apply here: developing theoretical frameworks,
      formulating mathematical claims, designing methodology and experiments,
      implementing methods, interpreting results. Write it thoroughly — this
      paper's thesis is disclosed methodology, so a vague statement is
      incongruent in a way it would not be elsewhere.
- [ ] **Double-blind.** No author identity in the PDF, the repository link, or
      the supplementary material. The GLEAM theory preprint is **not** on arXiv
      yet (`GLEAM/arxiv-plan.md` Phase D unchecked), so there is no
      self-citation or anonymity entanglement — keep it that way until after
      submission, or post it and cite in third person. Decide once, early.
- [ ] **Anonymised release.** The repository is public under Idil's name. An
      anonymous mirror (or an anonymous4openscience link) is needed for the
      artifact claims to be checkable without breaking blinding.

## The binding constraint is length, not content

Nine ICLR pages with 4–5 figures and the ledger table is roughly 4,500–5,000
words of body text. Current state:

| section | now | budget | action |
|---|---|---|---|
| §1 Introduction | 2,304 | ~800 | rewrite short, theory-first |
| §2 Testbed & method | 2,431 | ~900 | compress ~60%, overflow to appendix |
| §3 Theory | 0 | ~700 | **write** |
| §4 Results | 2,144 | ~1,200 | compress ~45% |
| §5 Discussion & limitations | 0 | ~500 | **write** |
| §6 Related work | 0 | ~300 | **write** |
| Abstract | 0 | ~200 | **write** |
| **total** | **6,879** | **~4,600** | |

Three sections unwritten *and* already 1.5× over budget. Everything cut has a
home: appendices are unlimited and twenty findings documents are already
appendix-shaped.

This is why LaTeX comes first. Page count is the constraint, and it is
invisible until the thing is typeset.

## Structure (final)

1. Introduction
2. An exactly solvable testbed
3. Theory: the aliased equilibrium map and two thresholds
4. Results
5. Discussion and limitations
6. Related work
   — AI-use statement (unnumbered, off-budget)
   — References (unlimited)
   — Appendices (unlimited)

Framing is **theory-first**, as decided 2026-08-21: this began as an
audit-of-practice paper with theory behind it; what survived is a theory paper
with an unusually rigorous validation protocol. The preregistration is *how* it
was tested, not *what* it is. The single largest review risk is a reader
deciding in the first two pages that this is control theory in RL clothing —
everything in §1 works against that.

## Day by day

### Block A — make it real (10–12 Sept)

**Day 1, Wed 10th.** LaTeX skeleton: `main.tex`, `sections/`, `refs.bib`, ICLR
2027 style files. Port §2 and §4 in verbatim, uncompressed. Compile. **The
output of today is a number**: how many pages the existing prose actually
occupies. Every subsequent cut is measured against it.

**Day 2, Thu 11th.** §3 Theory (~700 words + display equations). The missing
centrepiece: model, aliased LSTD(λ) fixed point, the equilibrium map
k_AC(λ), the cost law and λ_half, the off-manifold index S(λ,k) with the
(1−λ)/λ factor, and the scope proposition. Sources: `vendor/gleam/`,
`a5-tier2-bias-direction.md`, `prereg_e7_bias.py` docstring.

**Day 3, Fri 12th.** §1 rewritten from scratch — short, theory-first, against
the final ledger. The current draft is stale (says `[E10 PENDING]`, claims
"six failed" and "nine passes" against 18 predictions) and its chronological
arc now duplicates §4. Chronology moves to one paragraph or to §5.

### Block B — complete the draft (13–17 Sept)

**Day 4, Sat 13th.** Compression pass on §2 and §4 to budget. Overflow moves to
appendices with cross-references, not to the bin.

**Day 5, Sun 14th.** §5 Discussion and limitations. Content is settled
(`00-drafting-plan.md` §4): scope honesty, undertraining alternative and its
G2 defence, the three E10 failures under the pre-committed demotion, node-class
1.9σ and P7.2's cross-class span, the 5-seed budget, what we would do
differently.

**Day 6, Mon 15th.** §6 Related work + the AI-use statement.

**Day 7, Tue 16th.** Figures. Five: (i) J(k_AC(λ))/J_ml against λ, four
instances, λ_half marked — the money figure; (ii) two-arm dissociation;
(iii) S(λ,k) sign map with k_AC(λ) overlaid; (iv) E10 drift, measured gain vs
k_AC coloured by branch; (v) E13 λ_half invariance under cubic damping.
Functional quality today, beautiful later.

**Day 8, Wed 17th.** Abstract. Full assembly. First complete PDF at true page
count.

### Milestone — Thu 18 Sept: submit the abstract

Buffer day by design. If Block B slipped, this is the slack.

### Block C — polish (19–24 Sept)

**Days 9–10, Fri–Sat.** Appendices assembled from the findings documents.
Anonymised release package: harness, all preregistrations, GLEBench, the
diagnostic, `PROVENANCE.md`.

**Day 11, Sun 21st.** Number-verification pass. Every quoted number re-checked
against its generating file — script it, as `recount_ledger.py` already does
for the tally. `03-method.md` §3.7's provenance table is the index.

**Day 12, Mon 22nd.** Figures to publication quality.

**Day 13, Tue 23rd.** Cold read aloud after 24h away, plus one friendly UCL
reader.

**Day 14, Wed 24th.** Revisions from the read.

**Fri 25 Sept: submit**, with a day in hand.

## Standing risks

1. **Length.** Already 1.5× over with three sections unwritten. Mitigated by
   LaTeX-first and a dedicated compression day, but if Day 1's page count is
   worse than expected, cut §2 harder — the testbed details are the most
   appendix-tolerant material in the paper.
2. **The theory section is load-bearing and does not exist.** Everything else
   is editing; this is generation. It is scheduled second for that reason.
3. **Figures are unstarted** apart from three E6 drafts. Five needed.
4. **The scoreboard reads 11 pass / 14 fail of 25.** Presentation, not
   evidence: most "failures" are the scope law working (see
   `00-drafting-plan.md` §1.2). The ledger table discharges the honesty
   obligation in a third of a page; prose elaborates only what carries meaning.
5. **Three weeks were lost.** The plan has ~2 days of slack. Protect it.

## What does not happen

No new experiments. Data collection closed 2026-08-26 with E12's design
complete and E13 passing. The frozen seeds, grids and tolerances make further
runs illegitimate rather than merely unnecessary.
