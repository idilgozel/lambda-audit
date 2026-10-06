"""Synthetic-scenario tests for prereg_e19.py, run before the freeze.

    python3 analysis/test_prereg_e19.py
"""
import contextlib, io, json, os, random, re, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import prereg_e19 as m  # noqa: E402


def make(tmp, end_fn, drop=None, dup=False, sgd=0.0, eplen=m.EPISODE_LEN):
    rng = random.Random(9)
    rows = []
    for inst in m.INSTANCES:
        for lam in m.LAMBDAS:
            for side in m.START_FACTORS:
                for critic in m.CRITICS:
                    for seed in m.SEEDS:
                        if drop and (inst, lam, side, critic, seed) == drop:
                            continue
                        k0 = m.k0_of(inst, lam, side)
                        end = end_fn(inst, lam, side, critic, k0) + rng.gauss(0, 0.2)
                        row = {"run": f"{inst}_linear_{critic}_lam{lam}_k{k0}_s{seed}", "env": inst, "arch": "linear",
                               "lam": lam, "seed": seed, "final_gain": end, "J_rate": 0.3, "excess_cost": 1.35,
                               "fix_sigma": 0.3, "linear_gain_init": k0, "linear_sgd_lr": sgd, "gleam_episode_len": eplen,
                               "critic_fit": [-8.0, 0.0, (-150.0 if critic == "oracle" else 0.1)]}
                        if critic == "oracle":
                            row["oracle_critic"] = "lstd"
                        rows.append(row)
    if dup:
        rows.append(rows[0])
    path = os.path.join(tmp, "results_e19.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def run(end_fn, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        path = make(tmp, end_fn, **kw)
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


def pilot_like(inst, lam, side, critic, k0):
    kac = m.K_AC[inst][lam]
    if critic == "oracle":
        return k0 + 0.45 * (kac - k0)            # moves 45% of the way toward k_AC
    return k0 - 0.2 * k0                          # default: down by 20% from either start


def exact_critic_insufficient(inst, lam, side, critic, k0):
    return k0 - 0.2 * k0                          # oracle behaves like default


def default_moves_up(inst, lam, side, critic, k0):
    kac = m.K_AC[inst][lam]
    return k0 + 0.45 * (kac - k0) if side == "below" else k0 - 0.2 * k0   # both critics climb from below


def overshoot(inst, lam, side, critic, k0):
    kac = m.K_AC[inst][lam]
    if critic == "oracle":
        return kac + (0.3 * kac if side == "below" else -0.3 * kac)       # crosses to the far side
    return k0 - 0.2 * k0


def main():
    print("prereg_e19 synthetic scenarios")
    rc, out, v = run(pilot_like)
    check("pilot-like: gates PASS, P23.1-3 PASS", all(x == "PASS" for k, x in v.items() if k.startswith("G19"))
          and v["P23.1"] == "PASS" and v["P23.2"] == "PASS" and v["P23.3"] == "PASS", out)
    check("VERDICTS last line", out.strip().splitlines()[-1].startswith("VERDICTS:"), out)
    rc, out, v = run(exact_critic_insufficient)
    check("oracle no better than default: P23.1 FAIL, P23.3 FAIL, P23.2 PASS", v["P23.1"] == "FAIL" and v["P23.3"] == "FAIL" and v["P23.2"] == "PASS", out)
    rc, out, v = run(default_moves_up)
    check("default arm climbs from below: P23.2 FAIL, P23.1 PASS", v["P23.2"] == "FAIL" and v["P23.1"] == "PASS", out)
    rc, out, v = run(overshoot)
    check("oracle overshoots: P23.1 PASS, P23.3 FAIL", v["P23.1"] == "PASS" and v["P23.3"] == "FAIL", out)
    rc, out, v = run(lambda i, l, s, c, k0: k0 + 0.1)
    check("immobile arm: every gate FAIL, nothing scored", all(x == "FAIL" for x in v.values()) and "P23.1" not in v, out)
    rc, out, v = run(pilot_like, drop=("GLEAM_g1_h0.05", 0.95, "above", "oracle", 2))
    check("incomplete cell: that (instance, lambda) not scorable, the rest scored", "GLEAM_g1_h0.05 lambda 0.95: NOT SCORABLE YET" in out
          and "G19-GLEAM_g1_h0.05-0.95" not in v and v["P23.1"] == "PASS", out)
    rc, out, v = run(pilot_like, dup=True)
    check("duplicate: refused", rc == 2 and "REFUSING" in out, out)
    rc, out, v = run(pilot_like, sgd=0.01)
    check("SGD-arm rows are not E19 rows", v == {}, out)
    rc, out, v = run(pilot_like, eplen=1024)
    check("episodic rows are not E19 rows", v == {}, out)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.check_clean("/nonexistent/results_e19.jsonl")
    check("check-clean: missing file CLEAN", rc == 0, buf.getvalue())
    print("\nPASS. Every scorer path behaves as registered.")


if __name__ == "__main__":
    main()
