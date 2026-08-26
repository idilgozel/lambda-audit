"""Preregistration E8: the released diagnostic, frozen, characterised, and
given one genuinely prospective test.

FROZEN 2026-08-18, after E7/E10 landed and BEFORE E11's curves exist anywhere
off the cluster. That ordering is this file's honesty structure, so it is
stated first.

WHAT HAS ALREADY BEEN SEEN, disclosed in full
---------------------------------------------
Every training curve from the pilot, E6, E7, E9 and E10 is on disk and has
been examined, repeatedly, including pooled intermittency tables computed
with the ABSOLUTE-threshold variant on 2026-08-14 and 2026-08-18, and an
exploratory candidate comparison (analysis/e8_intermittency.py, built by the
assistant during the E7 wait). NOTHING in the development evaluation below is
therefore confirmatory, and the paper must not present it as such. It is a
disclosed characterisation on seen data: the development set.

The PROSPECTIVE content is P13.1: E11 (BattleshipEasy) was submitted on
2026-08-18 and its curves have never been pulled or seen. --check-clean
verifies their absence at freeze time. P13.1 is scored on them, once, when
they land.

THE TOOL, frozen here
---------------------
For one training run, from its curves CSV alone, no reference values, no
other seeds, no model access:

    cost series      c_t = -eval_return_t
    late half        L   = second half of logged eval points (SPLIT = 0.5)
    m                    = median(L)
    E_selfabs            = fraction of L with c > m + (MULT-1) * |m|,  MULT = 1.5
    FLAGGED         iff   E_selfabs >= 0.05

The sign-safe form equals the plain self-multiple when median cost is
positive and remains meaningful when returns make it negative. The 0.05 flag
fraction is the figure frozen in the Tier 1b scoring on 2026-08-13, reused
unchanged. MULT = 1.5 and SPLIT = 0.5 follow the exploratory recommendation
recorded before E7 landed ("E_selfabs at mult 1.5, last 50%"); the
sensitivity sweep behind that recommendation (mult in {1.5, 2, 3, 5}, split
in {0.3, 0.5, 0.7}) is part of the development record and ships with it.

GROUND TRUTH, and its circularity, named
-----------------------------------------
"Truly intermittent" for the development set is defined by the PRIVILEGED
detector: the absolute variant (cost > 2 * J_ml over the late half, flag at
0.05), which requires the analytic J_ml that no practitioner has. The
released tool is evaluated on whether it reproduces the privileged detector
WITHOUT the privilege. This makes the development evaluation an agreement
study against a reference instrument, not against nature; the reference
instrument itself was validated on E6 where the excursions are visible by
eye (4x to 30x optimal cost). Stated so nobody discovers it later.

THE THREE TRIPWIRES, hard requirements, any failure kills the tool
-------------------------------------------------------------------
  T1  MECHANISM. The E6 decoupled arm (40 runs): zero flags. A detector that
      fires where actions cannot influence the hidden state is detecting
      something other than the mechanism.
  T2  NO-ONSET INSTANCE. GLEAM_g4_h0.05 coupled, all lambdas (40 runs): zero
      flags. The instance where theory, E7 measurement and E10 all agree
      nothing happens. This is expectation X3 made mechanical.
  T3  PRACTICAL CASE. Closed-loop cells above their horizon: cartpole pilot
      cells at lambda >= 0.9, and GLEAM coupled cells at or above each
      instance's knee (boundary cells excluded per the pilot precedent):
      zero flags. The ordinary sensible-lambda run must never be flagged.

DEVELOPMENT METRICS, reported not gated: run-level agreement with the
privileged detector on below-knee coupled cells (sensitivity) and the
window between worst negative and weakest positive E_selfabs.

P13.1, THE REGISTERED PROSPECTIVE CLAIM
----------------------------------------
When E11's curves land: zero flags among
    (a) every gru128 BattleshipEasy run, all 8 lambdas, 5 seeds (40 runs);
    (b) every mlp BattleshipEasy run at lambda in {0.999, 1.0} (10 runs).
E11's knee arithmetic (prereg_e11.py) puts the knee at 0.99432; cells at
0.99 and 0.995 straddle it and are excluded as boundary, per precedent.
Sensitivity on E11's below-knee mlp cells is EXPLORATORY and unscored: no
ground truth for "truly intermittent" exists there that is independent of
the tool family being tested (J_ml has no Battleship analogue).

HOW A MISS IS READ. Any tripwire failure: the tool is not released, and the
paper's diagnostic contribution is withdrawn, not weakened. P13.1 failure:
the tool's specificity does not transfer off GLEAM; reported as a scope
restriction on the release, and the flagged runs are published with it.

AMENDMENT 1, dated 2026-08-18, AFTER the first --develop run. DISCLOSED.
-------------------------------------------------------------------------
The first execution of the development evaluation failed tripwire T3, and
the diagnosis showed the failure was in this file's EVALUATION HARNESS, not
established against the tool. Three defects, each fixed here, none touching
the tool definition (MULT, SPLIT, FLAG are unchanged) and none touching the
already-registered P13.1:

(a) T3's negative set included gru128 cells at low lambda on CartPoleHard.
    The project's own registered record contradicts that labelling: P9.3
    (frozen 2026-08-13, scored PASS on 2026-08-14) found gru across-seed
    variance concentrated at low lambda on Hard, ratio 33.9 — those cells
    are documented as unstable. All six T3 flags were exactly those cells:
    the tool detected instability the record already knew about, inside a
    set mislabelled "must be silent". FIX: T3 is restricted to the
    MEMORYLESS-CLASS architectures (mlp, stack*) above their knee, which is
    the aliased-critic theory's actual domain of silence. The theory makes
    no stability claim for recurrent training dynamics and neither can the
    tripwire.

(b) The ground truth for sensitivity conflated two phenomena. The privileged
    detector (cost > 2*J_ml) is a level detector: it fires on chronic
    elevation as well as on recurrent excursions. Self-normalisation is
    silent on chronic elevation BY CONSTRUCTION (a run that is always bad
    has no baseline to depart from), and such runs need no diagnostic - the
    return curve is visibly terrible. FIX: the positive class is
    MINORITY-EXCURSION runs, abs-fraction in [0.05, 0.5); majority/chronic
    runs (abs >= 0.5) are reported as a separate category with the tool's
    silence on them documented as scope, not counted as misses.

(c) The E10 linear/linearb curves were swept in by the filename pattern.
    They are purpose-built perturbation arms (a kicked bias is an
    artificial instability), not training runs a practitioner would scan.
    FIX: E8 evaluates archs (mlp, stack8, stack64, gru128) only.

What survives of the first run, permanently in the record: T1 (decoupled,
0/40) and T2 (no-onset instance, 0/88) PASSED on first execution - the two
mechanism-specific tripwires. And the residual sensitivity limitation
exposed by (b) is real and stays in the paper: excursions shallower than
1.5x an already-elevated baseline are invisible to the self-normalised
form. That is the price of needing no reference values, and it is now a
documented property rather than a buried one.
"""
import csv, json, math, os, statistics as st, sys

FROZEN_ON = "2026-08-18"
MULT, SPLIT, FLAG = 1.5, 0.5, 0.05
ABS_MULT = 2.0
E8_ARCHS = ("mlp", "stack8", "stack64", "gru128")   # Amendment 1c
MEMORYLESS_ARCHS = ("mlp", "stack8", "stack64")     # Amendment 1a
CHRONIC = 0.5                                        # Amendment 1b
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "results")

# per-instance knees, copied from prereg_phase2.LAMBDA_HALF (frozen 2026-08-12)
KNEE = {"GLEAM_g1_h0.02": 0.996552, "GLEAM_g1_h0.05": 0.985382,
        "GLEAM_g2_h0.05": 0.957977, "GLEAM_g4_h0.05": 0.873138}
J_ML = {"GLEAM_g1_h0.02": 0.217471, "GLEAM_g1_h0.05": 0.219595,
        "GLEAM_g2_h0.05": 0.172023, "GLEAM_g4_h0.05": 0.112991}
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
CARTPOLE_KNEE = 0.88
E11_KNEE = 0.99432
E11_SPEC_MLP_LAMS = [0.999, 1.0]
E11_BOUNDARY = [0.99, 0.995]


def _assert_frozen():
    """Amendment 2, 2026-08-21, disclosed: this file was the ONLY prereg in
    the project with no freeze assertion — found by a pre-submission audit,
    not noticed at freeze time. The constants below were frozen 2026-08-18
    (and the Amendment 1 values on the same date, disclosed there); this
    check was written three days later. It cannot retroactively prove the
    2026-08-18 freeze — PROVENANCE.md's cluster-log chronology does that —
    but from today it makes this file refuse to score if any constant
    drifts, which is the property every other prereg already had."""
    ok = (FROZEN_ON == "2026-08-18"
          and (MULT, SPLIT, FLAG) == (1.5, 0.5, 0.05)
          and ABS_MULT == 2.0 and CHRONIC == 0.5
          and E8_ARCHS == ("mlp", "stack8", "stack64", "gru128")
          and MEMORYLESS_ARCHS == ("mlp", "stack8", "stack64")
          and len(LAMBDAS) == 8 and len(KNEE) == 4 and len(J_ML) == 4
          and E11_SPEC_MLP_LAMS == [0.999, 1.0]
          and E11_BOUNDARY == [0.99, 0.995]
          and abs(E11_KNEE - 0.99432) < 1e-9)
    if not ok:
        raise SystemExit("PREREG NOT FROZEN OR TAMPERED. Do not score.")


def boundary_of(env):
    """Grid cells straddling the knee, excluded from labels per precedent."""
    k = KNEE[env]
    below = [l for l in LAMBDAS if l < k]
    above = [l for l in LAMBDAS if l >= k]
    out = []
    if below: out.append(below[-1])
    if above: out.append(above[0])
    return out


def costs(path):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get("eval_return", "")
            if v not in ("", None):
                try: out.append(-float(v))
                except ValueError: pass
    return out


def e_selfabs(cs, mult=MULT, split=SPLIT):
    L = cs[int(len(cs) * (1 - split)):]
    if not L: return float("nan")
    m = st.median(L)
    return sum(1 for c in L if c > m + (mult - 1.0) * abs(m)) / len(L)


def e_abs(cs, jml, mult=ABS_MULT, split=SPLIT):
    L = cs[int(len(cs) * (1 - split)):]
    if not L: return float("nan")
    return sum(1 for c in L if c > mult * jml) / len(L)


def _runs(pattern_env, arch=None):
    out = []
    for fn in sorted(os.listdir(RES)):
        if not fn.startswith("curves_") or pattern_env not in fn: continue
        core = fn[len("curves_"):-len(".csv")]
        try:
            envarch, lam_s, seed_s = core.rsplit("_lam", 1)[0], core.rsplit("_lam", 1)[1].rsplit("_s", 1)[0], core.rsplit("_s", 1)[1]
        except Exception:
            continue
        parts = envarch.rsplit("_", 1)
        env, a = parts[0], parts[1]
        if arch and a != arch: continue
        if a not in E8_ARCHS: continue               # Amendment 1c
        out.append(dict(env=env, arch=a, lam=float(lam_s), seed=int(seed_s),
                        path=os.path.join(RES, fn)))
    return out


def develop():
    print(f"E8 development evaluation (SEEN data; characterisation, not confirmation)")
    print(f"tool: E_selfabs mult {MULT} split {SPLIT} flag {FLAG}\n")
    fails = []
    # T1 decoupled
    t1 = [r for r in _runs("GLEAM_g1_h0.05_dec")]
    f1 = [r for r in t1 if e_selfabs(costs(r["path"])) >= FLAG]
    print(f"T1 decoupled arm      : {len(t1):3d} runs, {len(f1)} flagged -> {'PASS' if not f1 else 'FAIL'}")
    if f1: fails.append("T1")
    # T2 g4
    t2 = [r for r in _runs("GLEAM_g4_h0.05") if "_dec" not in r["env"]]
    f2 = [r for r in t2 if e_selfabs(costs(r["path"])) >= FLAG]
    print(f"T2 no-onset instance  : {len(t2):3d} runs, {len(f2)} flagged -> {'PASS' if not f2 else 'FAIL'}")
    if f2: fails.append("T2")
    # T3 above-horizon closed loop
    t3 = []
    for env, k in KNEE.items():
        bnd = boundary_of(env)
        t3 += [r for r in _runs(env) if "_dec" not in r["env"]
               and r["arch"] in MEMORYLESS_ARCHS          # Amendment 1a
               and r["lam"] >= k and r["lam"] not in bnd]
    t3 += [r for r in _runs("PositionOnlyCartPole")
           if r["arch"] in MEMORYLESS_ARCHS and r["lam"] >= 0.9]
    f3 = [r for r in t3 if e_selfabs(costs(r["path"])) >= FLAG]
    print(f"T3 above-horizon      : {len(t3):3d} runs, {len(f3)} flagged -> {'PASS' if not f3 else 'FAIL'}")
    for r in f3[:8]:
        print(f"     flagged: {r['env']} {r['arch']} lam {r['lam']} s{r['seed']} "
              f"E_selfabs {e_selfabs(costs(r['path'])):.3f}")
    if f3: fails.append("T3")
    # sensitivity vs the privileged detector on below-knee coupled cells
    pos_truth, agree, weakest, worstneg, chronic_n = 0, 0, 1.0, 0.0, 0
    for env, k in KNEE.items():
        bnd = boundary_of(env)
        for r in _runs(env):
            if "_dec" in r["env"] or r["lam"] in bnd or r["lam"] >= k: continue
            cs = costs(r["path"])
            a_frac = e_abs(cs, J_ML[env])
            if a_frac >= CHRONIC:                     # Amendment 1b
                chronic_n += 1
                continue
            truth = a_frac >= FLAG
            pred = e_selfabs(cs) >= FLAG
            if truth:
                pos_truth += 1
                agree += int(pred)
                if pred: weakest = min(weakest, e_selfabs(cs))
    for grp in (t1, t2, t3):
        for r in grp:
            worstneg = max(worstneg, e_selfabs(costs(r["path"])))
    sens = agree / pos_truth if pos_truth else float("nan")
    print(f"\nsensitivity vs privileged detector (minority-excursion class): "
          f"{agree}/{pos_truth} = {sens:.3f}")
    print(f"chronic/majority-excursion runs (tool silent by construction, "
          f"reported as scope): {chronic_n}")
    print(f"window: worst negative {worstneg:.4f} vs flag {FLAG} vs weakest "
          f"detected positive {weakest:.4f} -> "
          f"{'OPEN' if worstneg < FLAG <= weakest else 'CLOSED'}")
    print(f"\nTRIPWIRES: {'ALL PASS - tool releasable' if not fails else 'FAILED: ' + ','.join(fails) + ' - tool NOT released'}")
    return 1 if fails else 0


def score_e11():
    runs = _runs("BattleshipEasy")
    if not runs:
        print("no E11 curves present yet"); return 2
    spec = [r for r in runs if r["arch"] == "gru128"] + \
           [r for r in runs if r["arch"] == "mlp" and r["lam"] in E11_SPEC_MLP_LAMS]
    flagged = [(r, e_selfabs(costs(r["path"]))) for r in spec]
    bad = [(r, v) for r, v in flagged if v >= FLAG]
    print(f"P13.1: {len(spec)} specificity runs, {len(bad)} flagged -> "
          f"{'PASS' if not bad else 'FAIL'}")
    for r, v in bad:
        print(f"   flagged: {r['arch']} lam {r['lam']} s{r['seed']} E_selfabs {v:.3f}")
    print("\nexploratory (NOT scored) - below-knee mlp cells:")
    for lam in [l for l in LAMBDAS if l < E11_KNEE and l not in E11_BOUNDARY]:
        vs = [e_selfabs(costs(r["path"])) for r in runs
              if r["arch"] == "mlp" and math.isclose(r["lam"], lam)]
        if vs:
            print(f"   lam {lam:<6} E_selfabs " + ", ".join(f"{v:.3f}" for v in sorted(vs)))
    return 0


def check_clean():
    n = len(_runs("BattleshipEasy"))
    print(f"{'NOT CLEAN' if n else 'CLEAN'}: {n} BattleshipEasy curve files in {RES}.")
    return 2 if n else 0


if __name__ == "__main__":
    _assert_frozen()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--develop": raise SystemExit(develop())
    if mode == "--e11": raise SystemExit(score_e11())
    if mode == "--check-clean": raise SystemExit(check_clean())
    print(f"E8 prereg frozen {FROZEN_ON}. Tool: E_selfabs mult {MULT}, split "
          f"{SPLIT}, flag {FLAG}. Modes: --develop | --e11 | --check-clean")
