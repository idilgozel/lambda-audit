"""Preregistration E16: does MEMORY remove the displacement in-domain?

FROZEN 2026-10-05, BEFORE ANY E16 RUN EXISTS. Verified mechanically:
    python3 prereg_e16.py --check-clean results/results/results_e16.jsonl

WHY THIS EXPERIMENT
-------------------
The scope proposition and the decoupled arm (E6, P6.3) show the confound
vanishes when the action cannot STEER the hidden state. The other hypothesis
of the mechanism -- that the critic is ALIASED, i.e. cannot see the hidden
coordinate -- has only been tested externally (CartPole E9, Pendulum E14),
where the memory architecture's behaviour is a prediction about variance, not
about the equilibrium. In-domain, every memory experiment so far used the
memoryless mlp. E16 gives the same instance a memory: a frame stack of 8
observations (0.4 hidden correlation times), a frame stack of 64 (3.2), and
the gru128 used externally. If the displacement is an aliasing effect, a
critic that can reconstruct the hidden coordinate from history has nothing to
alias, the amplification Lambda (analysis/knee_formula.py, identity I) is
absent, and the lambda dependence of the deployed cost should disappear along
with the below-knee intermittency. If it does not disappear, lambda acts on
PPO through a channel the theory does not describe, and the paper must say so.

DESIGN (frozen)
---------------
GLEAM_g1_h0.05, coupled arm, sigma pinned at 0.3 (as E6, so the mlp cells of
E6 are the comparison), archs stack8, stack64, gru128; 8 lambdas x 5 seeds x 3
archs = 120 runs, 15M steps, gamma = beta = 0.9975031223974601, results file
results_e16.jsonl. Deployed cost J_rate: policy mean, one 200,000-step
trajectory after a 1,000-step burn-in, exactly as E6. Each arch is scored
separately; a prediction PASSES only if it passes on every arch that clears
its gate. Runs locally (analysis/run_e16.py), one machine.

E6 mlp reference (measured, 5 seeds, IQM of J_rate/J_ml; context, not scored):
    lambda 0.8: 7.03   0.9: 2.62   0.95: 4.40   0.98: 1.30   0.99: 1.02   1.0: 0.996
so the mlp's below-knee excess is 2.6-7x and its ratio E(0.9)/E(1.0) is 2.62.

CAPABILITY GATE, per arch, scored first.
  G16  IQM deployed E at lambda = 1.0 <= 0.98. The full-information optimum is
       J_opt/J_ml = 0.906 and the mlp reaches 0.996: an arch that does not beat
       the memoryless optimum by 2% at lambda = 1 is not using its memory, and
       its predictions are NOT SCORED.

THE REGISTERED PREDICTIONS (IQM over seeds within cells; scoring requires all
40 cells of an arch present once)
  P19.1  NO DISPLACEMENT. IQM E(0.9) / IQM E(1.0) <= 1.10 on every gated-in arch.
  P19.2  NO INTERMITTENCY. On every gated-in arch and every lambda, the cell
         median of the late-half fraction of evaluation points costing more
         than 2 x J_ml is <= 0.05 (the statistic frozen for the mlp in
         prereg_e7_bias.py, where it read 0.36-0.89 below the knee).
  P19.3  MEMORY REMOVES THE PENALTY. On every gated-in arch, IQM E at each of
         lambda in {0.8, 0.9, 0.95} is at most half the mlp's E6 value at the
         same lambda (7.03, 2.62, 4.40).

HOW A MISS IS READ. P19.1 failing on stack8 alone: the 8-step window does not
span the hidden correlation time (20 steps) and residual aliasing remains --
reported as a statement about memory span. P19.1 failing on gru128 or stack64:
lambda moves PPO through a channel other than critic aliasing; the paper's
mechanism claim is then incomplete in a way the theory cannot absorb. P19.2
failing with P19.1 passing: the below-knee intermittency is not an aliasing
phenomenon. P19.3 is the weakest claim and failing it alongside P19.1 is the
same finding.

AMENDMENTS. None at freeze.
"""
import csv
import json
import math
import os
import sys

FROZEN_ON = "2026-10-05"
ENV = "GLEAM_g1_h0.05"
ARCHS = ("stack8", "stack64", "gru128")
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
SEEDS = (1, 2, 3, 4, 5)
GAMMA = 0.9975031223974601
FIX_SIGMA = 0.3
TOTAL_STEPS = 15_000_000
J_ML = 0.219595
J_OPT_OVER_J_ML = 0.198914 / 0.219595
G16_MAX_E1 = 0.98
P191_LAM, REF_LAM, P191_MAX_RATIO = 0.9, 1.0, 1.10
INTERMITTENCY_COST_MULTIPLE, INTERMITTENCY_FRACTION, LATE_FRACTION = 2.0, 0.05, 0.5
P193_LAMS = [0.8, 0.9, 0.95]
MLP_E6_E = {0.8: 7.029, 0.9: 2.615, 0.95: 4.397}      # measured E6, context for P19.3
P193_FACTOR = 0.5

ROOT = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(ROOT, "results", "results")


def _assert_frozen():
    for v in (FROZEN_ON, G16_MAX_E1, P191_MAX_RATIO, P193_FACTOR, J_ML):
        if v is None:
            raise SystemExit("PREREG NOT FROZEN. Do not launch, do not score.")
    if FROZEN_ON != "2026-10-05" or ENV != "GLEAM_g1_h0.05" or FIX_SIGMA != 0.3:
        raise SystemExit("PREREG TAMPERED. Do not score.")
    if set(P193_LAMS) - set(LAMBDAS) or P191_LAM not in LAMBDAS or set(MLP_E6_E) != set(P193_LAMS):
        raise SystemExit("PREREG INCONSISTENT.")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def rows_of(rows, arch):
    return [r for r in rows if r.get("env") == ENV and r.get("arch") == arch
            and r.get("seed") in SEEDS and r.get("fix_sigma") == FIX_SIGMA]


def cell(rows, arch, lam, field="J_rate"):
    out = []
    for r in rows_of(rows, arch):
        if math.isclose(float(r.get("lam", -1)), lam):
            v = r.get(field)
            if v is not None and math.isfinite(float(v)):
                out.append(float(v))
    return out


def runs_in_cell(rows, arch, lam):
    return [r["run"] for r in rows_of(rows, arch) if math.isclose(float(r.get("lam", -1)), lam)]


def iqm(xs):
    xs = sorted(xs); n = len(xs); lo, hi = n // 4, n - n // 4
    m = xs[lo:hi] if hi > lo else xs
    return sum(m) / len(m)


def late_fraction(run, res_dir=RES_DIR):
    """Fraction of the late-half evaluation points whose cost exceeds 2 J_ml.
    eval_return is -J_rate for continuous tasks. None if the curve is absent."""
    p = os.path.join(res_dir, f"curves_{run}.csv")
    if not os.path.exists(p):
        return None
    ev = [float(r["eval_return"]) for r in csv.DictReader(open(p)) if r["eval_return"] not in ("", None)]
    late = ev[int(len(ev) * (1 - LATE_FRACTION)):]
    if not late:
        return None
    return sum(1 for e in late if -e > INTERMITTENCY_COST_MULTIPLE * J_ML) / len(late)


def duplicate_cells(rows):
    counts = {}
    for arch in ARCHS:
        for r in rows_of(rows, arch):
            k = (arch, round(float(r.get("lam", -1)), 6), int(r.get("seed")))
            counts[k] = counts.get(k, 0) + 1
    return {k: n for k, n in counts.items() if n > 1}


def check_clean(path):
    _assert_frozen()
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"CLEAN: {path} does not exist. Freeze is legitimate.")
        return 0
    n = sum(len(rows_of(rows, a)) for a in ARCHS)
    print(f"{'NOT CLEAN' if n else 'CLEAN'}: {n} E16 rows in {path}.")
    return 2 if n else 0


def _v(b):
    return "NOT SCORED" if b is None else ("PASS" if b else "FAIL")


def score(path, res_dir=RES_DIR):
    _assert_frozen()
    rows = load(path)
    dups = duplicate_cells(rows)
    if dups:
        print(f"REFUSING TO SCORE: {len(dups)} duplicated (arch, lambda, seed) cells, e.g. {sorted(dups)[0]}.")
        return 2
    v = {}
    complete, gated = [], []
    for arch in ARCHS:
        missing = [(l, len(cell(rows, arch, l))) for l in LAMBDAS if len(cell(rows, arch, l)) != len(SEEDS)]
        if missing:
            if any(n for _, n in missing) or len(missing) < len(LAMBDAS):
                print(f"{arch}: NOT SCORABLE YET, incomplete cells (lambda, seeds present): "
                      + ", ".join(f"({l}, {n})" for l, n in missing))
            else:
                print(f"{arch}: no rows yet.")
            continue
        complete.append(arch)
        e1 = iqm(cell(rows, arch, REF_LAM)) / J_ML
        g = e1 <= G16_MAX_E1
        v[f"G16-{arch}"] = g
        print(f"{arch}: G16 E(1.0) IQM {e1:.4f} (<= {G16_MAX_E1}; J_opt/J_ml {J_OPT_OVER_J_ML:.3f}, mlp 0.996) -> {_v(g)}")
        if g:
            gated.append(arch)
    if not complete:
        print("VERDICTS:", {})
        return 0
    if not gated:
        print("\nP19.1, P19.2, P19.3 NOT SCORED: no arch cleared its gate.")
        print("VERDICTS:", {k: _v(b) for k, b in v.items()})
        return 0
    print(f"\nGated in: {gated}" + (f"; not scored: {[a for a in complete if a not in gated]}" if len(gated) < len(complete) else ""))

    ok1 = ok2 = ok3 = True
    p192_unscorable = False
    for arch in gated:
        E = {l: iqm(cell(rows, arch, l)) / J_ML for l in LAMBDAS}
        ratio = E[P191_LAM] / E[REF_LAM]
        good = ratio <= P191_MAX_RATIO
        ok1 = ok1 and good
        print(f"  P19.1 {arch}: E(0.9)/E(1.0) = {E[P191_LAM]:.4f}/{E[REF_LAM]:.4f} = {ratio:.3f} (<= {P191_MAX_RATIO}) -> {_v(good)}"
              f"    [mlp E6: 2.62]")
        fr_line, worst = [], 0.0
        for l in LAMBDAS:
            fr = [late_fraction(run, res_dir) for run in runs_in_cell(rows, arch, l)]
            fr = [f for f in fr if f is not None]
            if len(fr) != len(SEEDS):
                p192_unscorable = True
                fr_line.append(f"{l}: curves missing")
                continue
            med = sorted(fr)[len(fr) // 2]
            worst = max(worst, med)
            fr_line.append(f"{l}: {med:.2f}")
        good2 = worst <= INTERMITTENCY_FRACTION
        ok2 = ok2 and good2
        print(f"  P19.2 {arch}: cell-median late fraction > 2 J_ml by lambda: " + ", ".join(fr_line)
              + f"  (max {worst:.2f} <= {INTERMITTENCY_FRACTION}) -> {_v(good2)}")
        det = []
        good3 = True
        for l in P193_LAMS:
            ok_l = E[l] <= P193_FACTOR * MLP_E6_E[l]
            good3 = good3 and ok_l
            det.append(f"{l}: {E[l]:.3f} vs mlp {MLP_E6_E[l]:.2f} ({'ok' if ok_l else 'NOT <= half'})")
        ok3 = ok3 and good3
        print(f"  P19.3 {arch}: " + "; ".join(det) + f" -> {_v(good3)}")
        print(f"        E by lambda: " + ", ".join(f"{l}: {E[l]:.3f}" for l in LAMBDAS))
    v["P19.1"] = ok1
    v["P19.2"] = None if p192_unscorable else ok2
    v["P19.3"] = ok3
    if p192_unscorable:
        print("  P19.2 NOT SCORED: training curves absent for some cells (they are derived data; "
              "re-run or restore results/results/curves_*.csv).")
    print(f"P19.1: {_v(v['P19.1'])}   P19.2: {_v(v['P19.2'])}   P19.3: {_v(v['P19.3'])}")
    print("\nVERDICTS:", {k: _v(b) for k, b in v.items()})
    return 0


def main(argv):
    _assert_frozen()
    if len(argv) > 2 and argv[1] == "--check-clean":
        return check_clean(argv[2])
    if len(argv) > 1:
        return score(argv[1])
    print(f"E16 prereg frozen {FROZEN_ON}. {ENV} coupled, sigma {FIX_SIGMA}, archs {ARCHS}, "
          f"{len(LAMBDAS)} lambdas x {len(SEEDS)} seeds. Gate G16 per arch (E(1.0) <= {G16_MAX_E1}), "
          f"then P19.1 no displacement (ratio <= {P191_MAX_RATIO}), P19.2 no intermittency, P19.3 penalty halved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
