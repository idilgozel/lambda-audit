"""Synthetic-scenario tests for prereg_e18.py, run before the freeze.

    python3 analysis/test_prereg_e18.py
"""
import contextlib, io, json, os, random, re, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import prereg_e18 as m  # noqa: E402


def profile_theory(inst, lam):
    """The fixed point's own cost: E = J_AC(lambda) / J_ml on the grid, whose
    grid-matched knee is by construction the frozen target."""
    return m.J_AC[inst][m.LAMBDAS.index(lam)] / m.J_ML[inst]


def profile_e7(inst, lam):
    """E7-like: thrashing excess far above the plateau below 0.9, collapsing early."""
    return {0.8: 2.8, 0.9: 3.1, 0.95: 1.23, 0.98: 1.1, 0.99: 1.02}.get(lam, 1.0)


def make(tmp, prof, drop=None, dup=False, episode_len=m.EPISODE_LEN, gains=None):
    rng = random.Random(5)
    rows = []
    for inst in m.INSTANCES:
        for lam in m.LAMBDAS:
            for seed in m.SEEDS:
                if drop and (inst, lam, seed) == drop:
                    continue
                E = prof(inst, lam)
                J = rng.gauss(E * m.J_ML[inst], 0.004 * m.J_ML[inst])
                rows.append({"run": f"{inst}_mlp_lam{lam}_s{seed}", "env": inst, "arch": "mlp", "arm": "coupled",
                             "lam": lam, "seed": seed, "J_rate": J, "excess_cost": J / m.J_ML[inst], "fix_sigma": 0.3,
                             "final_gain": (gains or (lambda i, l: 20.0 if E > 1.2 else 1.6))(inst, lam),
                             "gleam_episode_len": episode_len})
    if dup:
        rows.append(rows[0])
    path = os.path.join(tmp, "results_e18.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def run(prof, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        path = make(tmp, prof, **kw)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = m.score(path)
    out = buf.getvalue()
    mm = re.search(r"^VERDICTS:\s*(\{.*\})\s*$", out, re.M)
    return rc, out, (eval(mm.group(1)) if mm else None)


def check(label, cond, out):
    if not cond:
        print(out); raise AssertionError(label)
    print(f"  ok  {label}")


def main():
    print("prereg_e18 synthetic scenarios")
    rc, out, v = run(profile_theory)
    check("theory profile: gates PASS, P22.1 PASS, P22.2 as scorable", all(v[f"G18-{i}"] == "PASS" for i in m.INSTANCES)
          and v["P22.1"] == "PASS" and v["P22.2"] == ("PASS" if len(m.INSTANCES) > 1 else "NOT SCORED"), out)
    check("theory profile: P22.3 PASS", v["P22.3"] == "PASS", out)
    rc, out, v = run(lambda i, l: 1.2 * profile_theory(i, l) if l < 0.97 else profile_theory(i, l))
    check("plateau 20% high: P22.1 still PASS, P22.3 FAIL", v["P22.1"] == "PASS" and v["P22.3"] == "FAIL", out)
    check("VERDICTS last line", out.strip().splitlines()[-1].startswith("VERDICTS:"), out)
    rc, out, v = run(profile_e7)
    check("E7-like profile: gates PASS, P22.1 FAIL", all(v[f"G18-{i}"] == "PASS" for i in m.INSTANCES) and v["P22.1"] == "FAIL", out)
    rc, out, v = run(lambda i, l: 1.5)
    check("flat high profile: every gate FAIL, nothing scored", all(v[f"G18-{i}"] == "FAIL" for i in m.INSTANCES)
          and not any(k.startswith("P22") for k in v), out)
    rc, out, v = run(profile_theory, drop=(m.INSTANCES[0], 0.99, 2))
    check("incomplete instance: not scorable", f"{m.INSTANCES[0]}: NOT SCORABLE YET" in out and f"G18-{m.INSTANCES[0]}" not in v, out)
    rc, out, v = run(profile_theory, dup=True)
    check("duplicate: refused", rc == 2 and "REFUSING" in out, out)
    rc, out, v = run(profile_theory, episode_len=1024)
    check("episodic rows are not E18 rows", v == {}, out)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.check_clean("/nonexistent/results_e18.jsonl")
    check("check-clean: missing file CLEAN", rc == 0, buf.getvalue())
    print("\nPASS. Every scorer path behaves as registered.")


if __name__ == "__main__":
    main()
