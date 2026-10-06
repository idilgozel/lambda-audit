"""Synthetic-scenario tests for prereg_e16.py, run before the freeze.

    python3 analysis/test_prereg_e16.py
"""
import contextlib, csv, io, json, os, random, re, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import prereg_e16 as m  # noqa: E402


def make(tmp, pattern, archs=m.ARCHS, drop=None, dup=False, no_curves=False):
    rows = []
    rng = random.Random(3)
    res_dir = os.path.join(tmp, "results"); os.makedirs(res_dir, exist_ok=True)
    for arch in archs:
        for lam in m.LAMBDAS:
            for seed in m.SEEDS:
                if drop and (arch, lam, seed) == drop:
                    continue
                E, frac = pattern(arch, lam)
                J = rng.gauss(E * m.J_ML, 0.005 * m.J_ML)
                run = f"{m.ENV}_{arch}_lam{lam}_s{seed}"
                rows.append({"run": run, "env": m.ENV, "arch": arch, "lam": lam, "seed": seed,
                             "J_rate": J, "excess_cost": J / m.J_ML, "fix_sigma": 0.3, "final_gain": 1.5})
                if not no_curves:
                    with open(os.path.join(res_dir, f"curves_{run}.csv"), "w", newline="") as f:
                        w = csv.writer(f); w.writerow(["update", "global_step", "train_return", "eval_return"])
                        n = 100
                        for i in range(n):
                            excursion = rng.random() < frac and i >= n // 2
                            cost = (2.5 if excursion else E) * m.J_ML
                            w.writerow([i, i * 1000, "", f"{-cost:.5f}"])
    if dup:
        rows.append(rows[0])
    path = os.path.join(tmp, "results_e16.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path, res_dir


def run(pattern, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        path, res_dir = make(tmp, pattern, **kw)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = m.score(path, res_dir)
    out = buf.getvalue()
    mm = re.search(r"^VERDICTS:\s*(\{.*\})\s*$", out, re.M)
    return rc, out, (eval(mm.group(1)) if mm else None)


def check(label, cond, out):
    if not cond:
        print(out); raise AssertionError(label)
    print(f"  ok  {label}")


predicted = lambda a, l: (0.93, 0.0)
mlp_like = lambda a, l: ({0.8: 7.0, 0.9: 2.6, 0.95: 4.4, 0.98: 1.3}.get(l, 0.996), 0.6 if l < 0.98 else 0.0)
displaced = lambda a, l: ((1.3, 0.5) if l <= 0.95 else (0.93, 0.0))
stack8_partial = lambda a, l: ((1.07, 0.0) if (a == "stack8" and l <= 0.95) else (0.93, 0.0))   # ratio 1.15 on stack8


def main():
    print("prereg_e16 synthetic scenarios")
    rc, out, v = run(predicted)
    check("predicted: all gates PASS", all(v[f"G16-{a}"] == "PASS" for a in m.ARCHS), out)
    check("predicted: P19.1-3 PASS", all(v[k] == "PASS" for k in ("P19.1", "P19.2", "P19.3")), out)
    check("predicted: VERDICTS last", out.strip().splitlines()[-1].startswith("VERDICTS:"), out)
    rc, out, v = run(mlp_like)
    check("mlp-like: every gate FAIL, nothing scored", all(v[f"G16-{a}"] == "FAIL" for a in m.ARCHS)
          and not any(k.startswith("P19") for k in v), out)
    rc, out, v = run(displaced)
    check("displaced: P19.1 FAIL", v["P19.1"] == "FAIL", out)
    check("displaced: P19.2 FAIL (intermittent below the knee)", v["P19.2"] == "FAIL", out)
    check("displaced: P19.3 PASS (1.3 is still under half the mlp)", v["P19.3"] == "PASS", out)
    rc, out, v = run(stack8_partial)
    check("stack8 partial: P19.1 FAIL overall, named", v["P19.1"] == "FAIL" and "stack8: E(0.9)/E(1.0)" in out, out)
    rc, out, v = run(predicted, drop=("gru128", 0.99, 3))
    check("incomplete gru: gru not scorable, others scored", "gru128: NOT SCORABLE YET" in out and v["P19.1"] == "PASS"
          and "G16-gru128" not in v, out)
    rc, out, v = run(predicted, dup=True)
    check("duplicate: refused", rc == 2 and "REFUSING" in out, out)
    rc, out, v = run(predicted, no_curves=True)
    check("no curves: P19.2 NOT SCORED, others scored", v["P19.2"] == "NOT SCORED" and v["P19.1"] == "PASS", out)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.check_clean("/nonexistent/results_e16.jsonl")
    check("check-clean: missing file CLEAN", rc == 0, buf.getvalue())
    print("\nPASS. Every scorer path behaves as registered.")


if __name__ == "__main__":
    main()
