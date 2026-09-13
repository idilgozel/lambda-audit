"""Synthetic-scenario tests for prereg_e14.py, run BEFORE the freeze.

E12's scorer crashed on its own success path at the probe stage, and its
first equivalence criterion was found to be the wrong instrument, both by
running the scorer on synthetic data before freezing. E13 adopted the
practice at birth. This file is E14's version: every path the scorer can
take is driven by a synthetic results file drawn from a known pattern, and
the verdicts are asserted.

    python3 analysis/test_prereg_e14.py

Scenarios
    predicted      mlp steps down below the knee, flat above; gru variance
                   concentrated at low lambda           -> P17.1/2/3 PASS
    null           no lambda dependence anywhere        -> P17.1 FAIL, P17.2 PASS,
                                                           P17.3 FAIL
    trend          mlp rises linearly in lambda         -> P17.1 PASS, P17.2 FAIL
    wide_above     step below, but above cells scatter  -> P17.2 FAIL (TOST
                   with sd 0.06                            cannot resolve)
    hard_gate_fail gru does not learn on Hard           -> Hard gates FAIL,
                                                           Easy scored alone
    all_gate_fail  nothing learns                       -> P17 NOT SCORED
    probe          4 rows, lambda 1.0 seed 1 only       -> gates read,
                                                           NOT SCORABLE YET
    incomplete     one seed missing from one cell       -> NOT SCORABLE YET,
                                                           cell named
    duplicate      a probe cell re-run into the file    -> REFUSED (exit 2)
    unfrozen       sentinel left in place               -> SystemExit
"""
import contextlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import prereg_e14 as m  # noqa: E402

# Fill sentinels for the test ONLY when they are still None, so this file
# works both before and after the freeze without touching the prereg.
if m.FROZEN_ON is None:
    m.FROZEN_ON = "2026-09-11"
if m.FIX_SIGMA is None:
    m.FIX_SIGMA = 0.5

ENVS, ARCHS, LAMS, SEEDS = m.ENVS, m.ARCHS, m.LAMBDAS, m.SEEDS


def row(env, arch, lam, seed, ret):
    return {"run": f"{env}_{arch}_lam{lam}_s{seed}", "env": env, "arch": arch,
            "lam": lam, "seed": seed, "deployed_return": ret, "fix_sigma": 0.5}


def synth(pattern, rng, envs=ENVS, seeds=SEEDS, lams=LAMS, archs=ARCHS):
    rows = []
    for env in envs:
        for arch in archs:
            for lam in lams:
                for seed in seeds:
                    mu, sd = pattern(env, arch, lam)
                    rows.append(row(env, arch, lam, seed, rng.gauss(mu, sd)))
    return rows


def predicted(env, arch, lam, gru_lvl=0.80, mlp_lvl=0.45, step=-0.06):
    if arch == "mlp":
        return (mlp_lvl + (step if lam in m.BELOW else
                           step / 2 if lam in m.BOUNDARY else 0.0), 0.02)
    return gru_lvl, (0.08 if lam in m.P173_LOW else 0.01)


def null(env, arch, lam):
    return (0.45, 0.02) if arch == "mlp" else (0.80, 0.03)


def trend(env, arch, lam):
    """mlp falls smoothly with the bootstrap horizon instead of stepping:
    0.07 per e-fold of horizon, so the 0.98 cell sits 0.076 under 1.0."""
    if arch == "mlp":
        H = 1.0 / (1.0 - m.GAMMA * lam)
        return 0.45 - 0.07 * math.log(100.0 / H), 0.02
    return 0.80, (0.08 if lam in m.P173_LOW else 0.01)


def wide_above(env, arch, lam):
    mu, sd = predicted(env, arch, lam)
    return (mu, 0.06) if arch == "mlp" and lam in m.ABOVE else (mu, sd)


def hard_gate_fail(env, arch, lam):
    if env.endswith("Hard") and arch == "gru128":
        return 0.40, 0.02
    return predicted(env, arch, lam)


def all_gate_fail(env, arch, lam):
    return (0.26, 0.02) if arch == "mlp" else (0.30, 0.02)


def run_score(rows):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.score(path)
    os.unlink(path)
    out = buf.getvalue()
    mm = re.search(r"^VERDICTS:\s*(\{.*\})\s*$", out, re.M)
    verdicts = eval(mm.group(1)) if mm else None
    return rc, out, verdicts


def check(label, cond, out):
    if not cond:
        print(out)
        raise AssertionError(label)
    print(f"  ok  {label}")


def power(replicates=40):
    """Pass rate of P17.2 under TRUE equivalence above the knee, as a
    function of the seed sd, for the registered criterion and for the
    per-cell TOST it replaced. Run with --power. This is what rejected the
    per-cell form before the freeze."""
    keep = m.BOOT_N
    m.BOOT_N = 2000
    print(f"P17.2 pass rate under true equivalence ({replicates} replicates, "
          f"delta {m.P172_DELTA}, 5 seeds)")
    print("  seed sd   registered (pooled TOST + point bound)   per-cell TOST 98.75%")
    for sd in (0.01, 0.015, 0.02, 0.03, 0.04):
        n_reg = n_cell = 0
        rng_d = random.Random(11)
        for _ in range(replicates):
            reg_ok = cell_ok = True
            for env in ENVS:
                ref = [rng_d.gauss(0.45, sd) for _ in SEEDS]
                pool = [[rng_d.gauss(0.45, sd) for _ in SEEDS] for _ in m.P172_POOL]
                rng = random.Random(1)
                lo, hi = m._boot(rng, [(c, 1.0 / len(pool)) for c in pool]
                                 + [(ref, -1.0)], level=m.P172_CI)
                worst = max(abs(m.iqm(c) - m.iqm(ref)) for c in pool)
                reg_ok = reg_ok and lo > -m.P172_DELTA and hi < m.P172_DELTA \
                    and worst <= m.P172_DELTA
                for c in pool:
                    lo, hi = m._boot(random.Random(1), [(c, 1.0), (ref, -1.0)],
                                     level=98.75)
                    cell_ok = cell_ok and lo > -m.P172_DELTA and hi < m.P172_DELTA
            n_reg += reg_ok
            n_cell += cell_ok
        print(f"  {sd:.3f}     {n_reg / replicates:5.2f}"
              f"                                  {n_cell / replicates:5.2f}")
    m.BOOT_N = keep


def main():
    if "--power" in sys.argv:
        power()
        return
    print("prereg_e14 synthetic scenarios")

    rng = random.Random(1)
    rc, out, v = run_score(synth(predicted, rng))
    check("predicted: exit 0", rc == 0, out)
    check("predicted: all six gates PASS",
          all(v[k] == "PASS" for k in v if k.startswith("G14")), out)
    check("predicted: P17.1 PASS", v["P17.1"] == "PASS", out)
    check("predicted: P17.2 PASS", v["P17.2"] == "PASS", out)
    check("predicted: P17.3 PASS", v["P17.3"] == "PASS", out)
    check("predicted: boundary cell reported", "reported, not scored" in out, out)
    check("predicted: VERDICTS is the last line",
          out.strip().splitlines()[-1].startswith("VERDICTS:"), out)

    # the registered pattern must also pass at the seed spreads POPGym reports
    for sd, seed in ((0.02, 21), (0.03, 22)):
        rng = random.Random(seed)
        rc, out, v = run_score(synth(
            lambda e, a, l: (predicted(e, a, l)[0],
                             sd if a == "mlp" else predicted(e, a, l)[1]), rng))
        check(f"predicted at mlp seed sd {sd}: P17.1 PASS", v["P17.1"] == "PASS", out)
        check(f"predicted at mlp seed sd {sd}: P17.2 PASS", v["P17.2"] == "PASS", out)

    rng = random.Random(2)
    rc, out, v = run_score(synth(null, rng))
    check("null: P17.1 FAIL", v["P17.1"] == "FAIL", out)
    check("null: P17.2 PASS (flat above)", v["P17.2"] == "PASS", out)
    check("null: P17.3 FAIL", v["P17.3"] == "FAIL", out)

    rng = random.Random(3)
    rc, out, v = run_score(synth(trend, rng))
    check("trend: P17.1 PASS", v["P17.1"] == "PASS", out)
    check("trend: P17.2 FAIL (not threshold-like)", v["P17.2"] == "FAIL", out)

    rng = random.Random(4)
    rc, out, v = run_score(synth(wide_above, rng))
    check("wide_above: P17.2 FAIL (TOST cannot resolve at sd 0.06)",
          v["P17.2"] == "FAIL", out)

    rng = random.Random(5)
    rc, out, v = run_score(synth(hard_gate_fail, rng))
    check("hard_gate_fail: Hard G14b FAIL", v["G14b-Hard"] == "FAIL", out)
    check("hard_gate_fail: Easy gates PASS",
          all(v[f"G14{g}-Easy"] == "PASS" for g in "abc"), out)
    check("hard_gate_fail: predictions scored on Easy alone",
          "P17.1" in v and "NOT scored (gate failed): ['Hard']" in out, out)
    check("hard_gate_fail: P17.1 PASS on Easy", v["P17.1"] == "PASS", out)

    rng = random.Random(6)
    rc, out, v = run_score(synth(all_gate_fail, rng))
    check("all_gate_fail: exit 0", rc == 0, out)
    check("all_gate_fail: no prediction key in VERDICTS",
          not any(k.startswith("P17") for k in v), out)
    check("all_gate_fail: NOT SCORED printed", "NOT SCORED" in out, out)

    rng = random.Random(7)
    probe = synth(predicted, rng, seeds=(1,), lams=[1.0])
    rc, out, v = run_score(probe)
    check("probe: exit 0 (no crash on the success path)", rc == 0, out)
    check("probe: gates readable", all(v[k] == "PASS" for k in v), out)
    check("probe: NOT SCORABLE YET, probe stage",
          "NOT SCORABLE YET: probe stage" in out, out)
    check("probe: no prediction key", not any(k.startswith("P17") for k in v), out)
    check("probe: next command printed", "--e14easy" in out and "--e14hard" in out, out)

    rng = random.Random(8)
    rows = synth(predicted, rng)
    drop = ("PositionOnlyPendulumHard", "gru128", 0.995, 3)
    rows = [r for r in rows if (r["env"], r["arch"], r["lam"], r["seed"]) != drop]
    rc, out, v = run_score(rows)
    check("incomplete: exit 0", rc == 0, out)
    check("incomplete: not scored", not any(k.startswith("P17") for k in v), out)
    check("incomplete: the cell is named",
          re.search(r"Hard\s+gru128\s+lambda=0\.995\s+4/5 seeds", out) is not None, out)

    rng = random.Random(9)
    rows = synth(predicted, rng)
    rows.append(row("PositionOnlyPendulumEasy", "mlp", 1.0, 1, 0.44))
    rc, out, v = run_score(rows)
    check("duplicate: REFUSED with exit 2", rc == 2 and "REFUSING" in out, out)
    check("duplicate: no VERDICTS line", v is None, out)

    # --check-clean on a missing and on a populated file
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.check_clean("/nonexistent/results_e14.jsonl")
    check("check-clean: missing file is CLEAN", rc == 0 and "CLEAN" in buf.getvalue(),
          buf.getvalue())
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(probe[0]) + "\n")
        p = f.name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.check_clean(p)
    os.unlink(p)
    check("check-clean: populated file is NOT CLEAN", rc == 2, buf.getvalue())

    # unfrozen sentinel stops everything
    keep = m.FIX_SIGMA
    m.FIX_SIGMA = None
    try:
        m._assert_frozen()
        raised = False
    except SystemExit:
        raised = True
    m.FIX_SIGMA = keep
    check("unfrozen: sentinel raises SystemExit", raised, "")

    # grouping consistency guard
    keep = m.BELOW
    m.BELOW = [0.8, 0.9, 0.95]
    try:
        m._assert_frozen()
        raised = False
    except SystemExit:
        raised = True
    m.BELOW = keep
    check("tamper: moving 0.95 below the bracket is refused", raised, "")

    print("\nPASS. Every scorer path behaves as registered.")


if __name__ == "__main__":
    main()
