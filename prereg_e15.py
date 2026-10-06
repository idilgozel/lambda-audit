"""Preregistration E15: does the lambda displacement survive LEARNED exploration?

FROZEN 2026-10-05, BEFORE ANY E15 RUN EXISTS. Verified mechanically:
    python3 prereg_e15.py --check-clean results/results/results_e15.jsonl
analysis/run_e15.py refuses to start unless this file is frozen and that
check passes.

WHY THIS EXPERIMENT
-------------------
Every in-domain experiment of the audit pins the exploration standard
deviation (sigma_e = 0.3, s_e = 0.09) so that entropy collapse cannot act as
an uncontrolled brake on the lambda reading. A reviewer will say that
practice learns sigma, and that a learned sigma might remove the effect. The
theory says the opposite: with s_e as an input, the aliased equilibrium
k_AC(lambda; s_e) moves FURTHER from the memoryless optimum as s_e shrinks
(exact slice, GLEAM_g1_h0.05, analysis/knee_formula.py --slices: the TD(0)
plateau gain is 6.9 at s_e = 1.0, 24.8 at 0.09, 36.3 at 0.01, and the knee
lambda_half is 0.903, 0.985, 0.993). Entropy collapse makes the confound
worse, not better. E15 tests this on the paper's reference instance with the
standard learned Gaussian head (log-std parameter initialised at 0, i.e.
sigma = 1, entropy coefficient 0.01, everything else as E6).

The direction of the displacement is now a theorem (analysis/knee_formula.py
section 2): for every lambda < 1 the aliased critic multiplies the perceived
benefit of control by Lambda_k(lambda) >= 1, so the equilibrium gain is
displaced upward whenever the observation's squared-autocorrelation time
exceeds the mean lag of the control response -- a condition that holds for
every s_e, because s_e enters both sides the same way. So P18.1 and P18.2
below are predictions of a sign that the theory makes unconditionally; P18.3
is the quantitative cost law with the MEASURED exploration fed to the frozen
generator, which is the only honest way to make a location prediction when
the agent chooses its own s_e.

DESIGN (frozen)
---------------
Instance GLEAM_g1_h0.05, coupled arm, arch mlp, 8 lambdas x 5 seeds = 40
runs, 15M steps, gamma = beta = 0.9975031223974601, NO --fix-sigma (learned
log-std), results file results_e15.jsonl. The harness records the learned
sigma at the end of training as final_sigma (field added 2026-10-05 for
learned-sigma continuous runs only; pinned and discrete rows are untouched).
Deployed cost J_rate is measured with the policy MEAN on one 200,000-step
trajectory after a 1,000-step burn-in, exactly as E6/E7; final_gain is the
OLS slope of u on v over the same window.

WHERE THIS RUNS. Locally (analysis/run_e15.py), one machine, one CPU class.

CAPABILITY GATE, scored first.
  G15  IQM deployed cost at lambda = 1.0 within 15% of J_ml = 0.219595.
       At lambda = 1 the gradient is unbiased for any s_e (Lambda = 1), so a
       learner that cannot land near the memoryless optimum here has an
       optimisation problem, not an aliasing one, and P18 is NOT SCORED.

THE REGISTERED PREDICTIONS (seed-level bootstrap, 10,000 draws, seed
20261005, 95% CI, IQM within cells; scoring requires all 40 cells present once)
  P18.1  DIRECTION OF COST. IQM J_rate(0.9) - IQM J_rate(1.0) > 0, CI excluding 0.
  P18.2  OVER-CONTROL. Mean over lambda in {0.8, 0.9} of IQM final_gain minus
         IQM final_gain(1.0) > 0, CI excluding 0.
  P18.3  COST LAW AT THE MEASURED EXPLORATION. For each lambda, s_e(lambda) :=
         IQM over seeds of final_sigma^2. The frozen generator
         (vendor/gleam/kernel_general.k_AC, validated against prereg_e10's
         tables) gives k_AC(lambda; s_e(lambda)) and E_theory(lambda) =
         J(k_AC)/J_ml on the 8-point grid; the grid-matched midpoint
         estimator of prereg_e10 applied to E_theory gives lambda_half_theory,
         and applied to the MEASURED IQM J_rate/J_ml gives lambda_half_meas.
         Prediction: |lambda_half_meas - lambda_half_theory| <= 0.01, P11.3's
         tolerance. Reported alongside: the pinned-sigma value 0.983837, so
         the reader sees how far the learned exploration moved the knee.

HOW A MISS IS READ. P18.1 or P18.2 failing: the displacement does not
survive learned exploration on this instance; the paper's in-domain claims
are then claims about pinned exploration and must say so. P18.3 failing with
P18.1-2 passing: the effect is there but the theory does not locate it once
sigma is learned (most likely because sigma drifts during training and the
end-of-training sigma is not the one that shaped the equilibrium); reported.
Entropy collapse (final sigma << 0.3) or inflation (>> 0.3) is REPORTED as a
diagnostic, not scored.

AMENDMENTS. None at freeze.
"""
import json
import math
import os
import random
import sys

FROZEN_ON = "2026-10-05"
ENV = "GLEAM_g1_h0.05"
ARCH = "mlp"
LAMBDAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
SEEDS = (1, 2, 3, 4, 5)
GAMMA = 0.9975031223974601
TOTAL_STEPS = 15_000_000
J_ML = 0.219595                   # prereg_phase2.BASELINES[ENV]["J_ml"]
K_ML = 1.649001
PINNED_LAMBDA_HALF_GRID = 0.983837   # prereg_e10.py GRID_LAMBDA_HALF[ENV], s_e = 0.09
G15_TOL = 0.15
P181_LAM, REF_LAM = 0.9, 1.0
P182_LAMS = [0.8, 0.9]
P183_TOL = 0.01
BOOT_N, BOOT_SEED, CI = 10_000, 20261005, 95.0
S_E_FLOOR = 1e-6                  # a learned sigma below 1e-3 is treated as 1e-3 for the generator

ROOT = os.path.dirname(os.path.abspath(__file__))


def _assert_frozen():
    for v in (FROZEN_ON, G15_TOL, P183_TOL, BOOT_SEED, J_ML, PINNED_LAMBDA_HALF_GRID):
        if v is None:
            raise SystemExit("PREREG NOT FROZEN. Do not launch, do not score.")
    if FROZEN_ON != "2026-10-05" or ENV != "GLEAM_g1_h0.05" or REF_LAM != 1.0:
        raise SystemExit("PREREG TAMPERED. Do not score.")
    if sorted(LAMBDAS) != LAMBDAS or P181_LAM not in LAMBDAS or set(P182_LAMS) - set(LAMBDAS):
        raise SystemExit("PREREG INCONSISTENT: lambda grid.")


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def rows_of(rows):
    return [r for r in rows if r.get("env") == ENV and r.get("arch") == ARCH
            and r.get("seed") in SEEDS and r.get("fix_sigma") is None]


def cell(rows, lam, field):
    out = []
    for r in rows_of(rows):
        if math.isclose(float(r.get("lam", -1)), lam):
            v = r.get(field)
            if v is not None and math.isfinite(float(v)):
                out.append(float(v))
    return out


def iqm(xs):
    xs = sorted(xs); n = len(xs); lo, hi = n // 4, n - n // 4
    m = xs[lo:hi] if hi > lo else xs
    return sum(m) / len(m)


def _ci(draws, level=CI):
    draws.sort()
    return (draws[int(len(draws) * (100 - level) / 200)],
            draws[int(len(draws) * (1 - (100 - level) / 200)) - 1])


def _boot(rng, spec, level=CI):
    out = []
    for _ in range(BOOT_N):
        tot = 0.0
        for c, w in spec:
            tot += w * iqm([c[rng.randrange(len(c))] for _ in c])
        out.append(tot)
    return _ci(out, level)


def grid_lambda_half(pts):
    """Verbatim port of prereg_e10.lambda_half: midpoint of the max and min
    of E over the grid, linear interpolation at the first crossing."""
    hi = max(v for _, v in pts); lo = min(v for _, v in pts)
    if hi - lo < 1e-9:
        return None
    mid = 0.5 * (hi + lo)
    for (l0, v0), (l1, v1) in zip(pts, pts[1:]):
        if (v0 - mid) * (v1 - mid) <= 0 and abs(v1 - v0) > 1e-12:
            return l0 + (mid - v0) * (l1 - l0) / (v1 - v0)
    return None


def theory_E(s_e_by_lam):
    """E_theory(lambda) = J(k_AC(lambda; s_e(lambda))) / J_ml from the frozen generator."""
    sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))
    import kernel_general as kg
    from gleam_bench import GLEBench
    if not kg.validate():
        raise SystemExit("kernel_general does not reproduce the frozen E10 tables; refusing to score P18.3.")
    b = GLEBench(cs=(0.3,), gammas=(1.0,), h=0.05)
    _, J_ml = b.best_memoryless()
    out = []
    for lam in LAMBDAS:
        s_e = max(float(s_e_by_lam[lam]), S_E_FLOOR)
        ka = kg.k_AC(b, lam, s_e)
        out.append((lam, b.J_gain(ka) / J_ml, ka))
    return out


def duplicate_cells(rows):
    counts = {}
    for r in rows_of(rows):
        k = (round(float(r.get("lam", -1)), 6), int(r.get("seed")))
        counts[k] = counts.get(k, 0) + 1
    return {k: n for k, n in counts.items() if n > 1}


def check_clean(path):
    _assert_frozen()
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"CLEAN: {path} does not exist. Freeze is legitimate.")
        return 0
    n = len(rows_of(rows))
    print(f"{'NOT CLEAN' if n else 'CLEAN'}: {n} E15 rows in {path}.")
    return 2 if n else 0


def _v(b):
    return "NOT SCORED" if b is None else ("PASS" if b else "FAIL")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(path):
    _assert_frozen()
    rows = load(path)
    dups = duplicate_cells(rows)
    if dups:
        print(f"REFUSING TO SCORE: {len(dups)} duplicated (lambda, seed) cells, e.g. {sorted(dups)[0]}.")
        return 2
    missing = [(l, len(cell(rows, l, "J_rate"))) for l in LAMBDAS if len(cell(rows, l, "J_rate")) != len(SEEDS)]
    sig_missing = [l for l in LAMBDAS if len(cell(rows, l, "final_sigma")) != len(SEEDS)]
    if missing:
        print("E15 NOT SCORABLE YET: incomplete cells (lambda, seeds present): " +
              ", ".join(f"({l}, {n})" for l, n in missing))
        print("VERDICTS:", {})
        return 0
    if sig_missing:
        print(f"E15 NOT SCORABLE: final_sigma missing at lambda {sig_missing}; the harness must record the learned sigma.")
        print("VERDICTS:", {})
        return 2

    v = {}
    j1 = cell(rows, REF_LAM, "J_rate")
    ij1 = iqm(j1)
    g15 = abs(ij1 - J_ML) / J_ML <= G15_TOL
    print(f"G15  J(1.0) IQM {ij1:.4f} vs J_ml {J_ML:.4f} ({100 * abs(ij1 - J_ML) / J_ML:.1f}%, "
          f"<= {int(100 * G15_TOL)}%) -> {_v(g15)}")
    v["G15"] = g15
    sig = {l: iqm(cell(rows, l, "final_sigma")) for l in LAMBDAS}
    print("learned sigma (IQM over seeds) by lambda: " +
          ", ".join(f"{l}: {sig[l]:.3f}" for l in LAMBDAS) + "   [pinned runs used 0.300; diagnostic]")
    if not g15:
        print("\nP18.1, P18.2, P18.3 NOT SCORED (capability gate failed).")
        print("VERDICTS:", {k: _v(b) for k, b in v.items()})
        return 0

    rng = random.Random(BOOT_SEED)
    j09 = cell(rows, P181_LAM, "J_rate")
    lo, hi = _boot(rng, [(j09, 1.0), (j1, -1.0)])
    d1 = iqm(j09) - ij1
    ok1 = d1 > 0 and lo > 0
    print(f"\nP18.1 J(0.9) - J(1.0) = {d1:+.4f} (excess ratio {iqm(j09) / ij1:.3f}), CI [{lo:+.4f}, {hi:+.4f}] -> {_v(ok1)}")
    v["P18.1"] = ok1

    rng = random.Random(BOOT_SEED + 1)
    low = [cell(rows, l, "final_gain") for l in P182_LAMS]
    g1 = cell(rows, REF_LAM, "final_gain")
    lo, hi = _boot(rng, [(c, 1.0 / len(low)) for c in low] + [(g1, -1.0)])
    d2 = sum(iqm(c) for c in low) / len(low) - iqm(g1)
    ok2 = d2 > 0 and lo > 0
    print(f"P18.2 gain({P182_LAMS}) - gain(1.0) = {d2:+.3f} (gains "
          f"{[round(iqm(c), 3) for c in low]} vs {iqm(g1):.3f}), CI [{lo:+.3f}, {hi:+.3f}] -> {_v(ok2)}")
    v["P18.2"] = ok2

    s_e_by_lam = {l: sig[l] ** 2 for l in LAMBDAS}
    th = theory_E(s_e_by_lam)
    meas = [(l, iqm(cell(rows, l, "J_rate")) / J_ML) for l in LAMBDAS]
    lh_th = grid_lambda_half([(l, e) for l, e, _ in th])
    lh_me = grid_lambda_half(meas)
    print("P18.3 per lambda: measured E, theory E at measured s_e, k_AC(lambda; s_e), measured gain IQM")
    for (l, e_th, ka), (_, e_me) in zip(th, meas):
        print(f"      {l:<6} E_meas {e_me:.4f}  E_theory {e_th:.4f}  k_AC {ka:7.3f}  gain {iqm(cell(rows, l, 'final_gain')):7.3f}  sigma {sig[l]:.3f}")
    if lh_th is None or lh_me is None:
        ok3 = False
        print(f"P18.3 lambda_half undefined (theory {lh_th}, measured {lh_me}) -> FAIL")
    else:
        ok3 = abs(lh_me - lh_th) <= P183_TOL
        print(f"P18.3 lambda_half measured {lh_me:.5f} vs theory-at-measured-sigma {lh_th:.5f} "
              f"(|diff| {abs(lh_me - lh_th):.5f}, <= {P183_TOL}) -> {_v(ok3)}   "
              f"[pinned-sigma reference {PINNED_LAMBDA_HALF_GRID}]")
    v["P18.3"] = ok3
    print("\nVERDICTS:", {k: _v(b) for k, b in v.items()})
    return 0


def main(argv):
    _assert_frozen()
    if len(argv) > 2 and argv[1] == "--check-clean":
        return check_clean(argv[2])
    if len(argv) > 1:
        return score(argv[1])
    print(f"E15 prereg frozen {FROZEN_ON}. {ENV} coupled, {ARCH}, learned sigma, "
          f"{len(LAMBDAS)} lambdas x {len(SEEDS)} seeds. Gate G15, then P18.1 direction, "
          f"P18.2 over-control, P18.3 cost law at measured exploration (tol {P183_TOL}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
