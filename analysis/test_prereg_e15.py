"""Synthetic-scenario tests for prereg_e15.py, run before the freeze.

    python3 analysis/test_prereg_e15.py

Scenarios: predicted at sigma=0.3 (data drawn around the frozen pinned-sigma
tables) -> G15, P18.1-3 PASS; sigma-collapse at 0.05 (data drawn from the
generator at s_e = 0.0025) -> P18.3 PASS because the theory is evaluated at
the measured sigma; null (flat) -> P18.1/P18.2 FAIL; gate fail -> NOT
SCORED; incomplete -> not scorable; duplicate -> refused; missing
final_sigma -> exit 2.
"""
import contextlib, io, json, math, os, random, re, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))
import prereg_e15 as m  # noqa: E402

# frozen pinned-sigma (s_e = 0.09) references, prereg_phase2.py
K_AC = [22.426983, 20.546902, 17.670634, 11.394604, 2.197126, 1.756354, 1.548754, 1.506599]
J_AC = [0.294022, 0.292074, 0.288756, 0.278065, 0.222898, 0.219773, 0.219782, 0.219987]


def row(lam, seed, J, gain, sigma):
    return {"env": m.ENV, "arch": m.ARCH, "lam": lam, "seed": seed, "J_rate": J,
            "excess_cost": J / m.J_ML, "final_gain": gain, "final_sigma": sigma, "fix_sigma": None}


def synth(pattern, rng):
    rows = []
    for i, lam in enumerate(m.LAMBDAS):
        for s in m.SEEDS:
            J, g, sig = pattern(i, lam)
            rows.append(row(lam, s, rng.gauss(J, 0.01 * J), rng.gauss(g, 0.05 * g), rng.gauss(sig, 0.02 * sig)))
    return rows


def predicted(i, lam):
    return J_AC[i], K_AC[i], 0.30


def null(i, lam):
    return 0.225, 1.6, 0.30


def gate_fail(i, lam):
    return 0.30, 10.0, 0.30


def collapse_pattern():
    """Data generated from the frozen generator at s_e = 0.0025 (sigma 0.05)."""
    import kernel_general as kg
    from gleam_bench import GLEBench
    b = GLEBench(cs=(0.3,), gammas=(1.0,), h=0.05)
    table = {}
    for lam in m.LAMBDAS:
        ka = kg.k_AC(b, lam, 0.0025)
        table[lam] = (b.J_gain(ka), ka)
    return lambda i, lam: (table[lam][0], table[lam][1], 0.05)


def run(rows):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        p = f.name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.score(p)
    os.unlink(p)
    out = buf.getvalue()
    mm = re.search(r"^VERDICTS:\s*(\{.*\})\s*$", out, re.M)
    return rc, out, (eval(mm.group(1)) if mm else None)


def check(label, cond, out):
    if not cond:
        print(out)
        raise AssertionError(label)
    print(f"  ok  {label}")


def main():
    print("prereg_e15 synthetic scenarios")
    rng = random.Random(1)
    rc, out, v = run(synth(predicted, rng))
    check("predicted: G15 PASS", v["G15"] == "PASS", out)
    check("predicted: P18.1 PASS", v["P18.1"] == "PASS", out)
    check("predicted: P18.2 PASS", v["P18.2"] == "PASS", out)
    check("predicted: P18.3 PASS (theory at sigma 0.3 reproduces 0.9838)", v["P18.3"] == "PASS", out)
    check("predicted: VERDICTS last line", out.strip().splitlines()[-1].startswith("VERDICTS:"), out)

    rng = random.Random(2)
    rc, out, v = run(synth(collapse_pattern(), rng))
    check("collapse: P18.3 PASS at measured sigma 0.05", v["P18.3"] == "PASS", out)
    check("collapse: P18.2 PASS (larger over-control)", v["P18.2"] == "PASS", out)
    check("collapse: knee reported away from the pinned value",
          re.search(r"theory-at-measured-sigma 0\.99", out) is not None, out)

    rng = random.Random(3)
    rc, out, v = run(synth(null, rng))
    check("null: P18.1 FAIL", v["P18.1"] == "FAIL", out)
    check("null: P18.2 FAIL", v["P18.2"] == "FAIL", out)

    rng = random.Random(4)
    rc, out, v = run(synth(gate_fail, rng))
    check("gate fail: G15 FAIL and no P18 keys", v["G15"] == "FAIL" and not any(k.startswith("P18") for k in v), out)

    rng = random.Random(5)
    rows = synth(predicted, rng)[:-1]
    rc, out, v = run(rows)
    check("incomplete: not scorable, exit 0", rc == 0 and "NOT SCORABLE YET" in out, out)

    rng = random.Random(6)
    rows = synth(predicted, rng); rows.append(rows[0])
    rc, out, v = run(rows)
    check("duplicate: refused exit 2", rc == 2 and "REFUSING" in out, out)

    rng = random.Random(7)
    rows = synth(predicted, rng)
    for r in rows:
        r["final_sigma"] = None
    rc, out, v = run(rows)
    check("missing final_sigma: exit 2", rc == 2, out)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.check_clean("/nonexistent/results_e15.jsonl")
    check("check-clean: missing file is CLEAN", rc == 0, buf.getvalue())
    print("\nPASS. Every scorer path behaves as registered.")


if __name__ == "__main__":
    main()
