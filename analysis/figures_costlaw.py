#!/usr/bin/env python3
"""The cost law, four instances: the paper's central figure.

  fig_costlaw   E(lambda) = J(k_AC(lambda)) / J_ml, the exact deployed excess
                cost of the aliased actor-critic fixed point (vendor/gleam,
                re-validated against the frozen prereg_phase2 tables), against
                lambda, for the four GLEAM instances. Overlaid: the deployed
                excess cost of the one-parameter linear arm started at k_AC
                (E10, median of 3 seeds) and of the memoryless mlp trained
                from scratch (E7, IQM of 5 seeds, seeds shown faintly). Ticks
                on each curve mark the predicted knee lambda_half (continuum);
                the triangles on the axis mark the grid-matched knee measured
                from the E10 cells (P11.3). For GLEAM_g1_h0.02 the fixed point
                is not unique inside [0.99529, 0.99655] (knee-formula-findings.md
                section 4); the thin branch is the low-gain equilibrium and the
                vertical dotted line the jump of the largest-root branch.

Every number is generated: the curves from vendor/gleam, the points from the
committed results files, the knees from the frozen estimators.

Run:  python3 analysis/figures_costlaw.py          -> analysis/figures/fig_costlaw.{pdf,png}
      python3 analysis/figures_costlaw.py --wide   -> two-column width
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))
import prereg_phase2 as P2  # noqa: E402
import prereg_e10 as P10  # noqa: E402
import kernel_general as kg  # noqa: E402
from gleam_bench import GLEBench  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

PHASE2 = os.path.join(ROOT, "results", "results", "results_phase2.jsonl")
E10 = os.path.join(ROOT, "results", "results", "results_e10.jsonl")
FIGS = os.path.join(ROOT, "analysis", "figures")
INSTANCES = list(P2.E7_INSTANCES)
X_ONE = 4e-4            # where lambda = 1 is drawn on the log(1 - lambda) axis
TICKS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
STYLE = {"GLEAM_g1_h0.02": "-", "GLEAM_g1_h0.05": "--", "GLEAM_g2_h0.05": "-.", "GLEAM_g4_h0.05": ":"}
TAU = {"GLEAM_g1_h0.02": 50, "GLEAM_g1_h0.05": 20, "GLEAM_g2_h0.05": 10, "GLEAM_g4_h0.05": 5}


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def bench_of(name):
    g = float(name.split("_g")[1].split("_")[0]); h = float(name.split("_h")[1])
    return GLEBench(cs=(0.3,), gammas=(g,), h=h)


def validate(b, name):
    """The curve must reproduce the frozen tables at the eight grid lambdas."""
    j_ml = P2.BASELINES[name]["J_ml"]
    for lam, k_ref, j_ref in zip(P2.LAMBDAS, P2.K_AC_LAMBDA[name], P2.J_AC_LAMBDA[name]):
        k = kg.k_AC(b, lam)
        if abs(k - k_ref) > 2e-3 * max(1.0, k_ref) or abs(b.J_gain(k) - j_ref) > 1e-4:
            raise SystemExit(f"vendor/gleam disagrees with the frozen tables at {name}, lambda {lam}: "
                             f"k {k:.6f} vs {k_ref}, J {b.J_gain(k):.6f} vs {j_ref}")
    return j_ml


def roots(b, lam, n=300):
    ks = np.geomspace(0.3, 1.9 / b.h, n)
    vs = np.array([kg.ghat(b, k, lam) for k in ks])
    out = []
    for i in range(len(ks) - 1):
        if np.isfinite(vs[i]) and np.isfinite(vs[i + 1]) and np.sign(vs[i]) != np.sign(vs[i + 1]):
            r = brentq(lambda k: kg.ghat(b, k, lam), ks[i], ks[i + 1], xtol=1e-8)
            out.append((r, vs[i + 1] > vs[i]))
    return out


def x_of(lam):
    return X_ONE if lam >= 1.0 else 1.0 - lam


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def iqm(xs):
    xs = sorted(xs); n = len(xs); lo, hi = n // 4, n - n // 4
    m = xs[lo:hi] if hi > lo else xs
    return sum(m) / len(m)


def curve(b, j_ml):
    one_minus = np.geomspace(0.25, X_ONE, 220)
    lams = 1.0 - one_minus
    E = np.array([b.J_gain(kg.k_AC(b, l)) / j_ml for l in lams])
    return one_minus, E


def low_branch(b, j_ml):
    """E on the lower stable branch where the fixed point is not unique."""
    xs, Es, jump = [], [], None
    for om in np.geomspace(0.01, 0.002, 160):
        rs = roots(b, 1.0 - om)
        st = [r for r, s in rs if s]
        if len(st) >= 2:
            xs.append(om); Es.append(b.J_gain(st[0]) / j_ml)
            jump = om if jump is None else min(jump, om)
    return np.array(xs), np.array(Es), jump


def main(wide=False):
    plt = setup()
    os.makedirs(FIGS, exist_ok=True)
    rows2 = load(PHASE2) if os.path.exists(PHASE2) else []
    rows10 = load(E10) if os.path.exists(E10) else []
    fig, (ax, bx) = plt.subplots(2, 1, sharex=True, figsize=((6.75, 4.0) if wide else (3.4, 4.4)),
                                 gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.08))
    knees = []
    for name in INSTANCES:
        b = bench_of(name)
        j_ml = validate(b, name)
        om, E = curve(b, j_ml)
        ls = STYLE[name]
        ax.plot(om, E, ls, color="k", lw=1.0)
        bx.plot(om, E, ls, color="0.6", lw=0.7)
        # predicted knee (continuum) on the curve
        lh = P2.LAMBDA_HALF[name]
        ax.plot([1 - lh], [b.J_gain(kg.k_AC(b, lh)) / j_ml], marker="D", ms=3.6, mfc="w", mec="k", mew=0.8, ls="none", zorder=5)
        # predicted knee, grid-matched (the quantity P11.3 compares against)
        lh_pred_grid = P10.C[name]["grid_lambda_half"]
        ax.plot([x_of(lh_pred_grid)], [0.955], marker="^", ms=7.0, mfc="w", mec="k", mew=0.7, ls="none", clip_on=False, zorder=6)
        if name == "GLEAM_g1_h0.02":
            xs, Es, jump = low_branch(b, j_ml)
            if len(xs):
                ax.plot(xs, Es, "-", color="0.55", lw=0.6)
                ax.axvline(jump, color="0.55", lw=0.5, ls=":")
        # E10: linear arm started at k_AC, median over seeds
        pts = []
        for lam in P2.LAMBDAS:
            es = [float(r["excess_cost"]) for r in rows10 if r.get("env") == name and r.get("arch") == "linear"
                  and abs(float(r["lam"]) - lam) < 1e-9 and r.get("excess_cost") is not None]
            if es:
                m = float(np.median(es)); pts.append((lam, m))
                ax.plot([x_of(lam)], [m], marker="s", ms=3.0, mfc="w", mec="k", mew=0.7, ls="none", zorder=4)
        if len(pts) >= 3:
            lh_grid = P10.lambda_half(pts)
            if lh_grid is not None:
                knees.append((name, lh_grid, lh_pred_grid, lh))
                ax.plot([x_of(lh_grid)], [0.955], marker="^", ms=3.8, color="k", ls="none", clip_on=False, zorder=7)
        # E7: mlp from scratch, IQM over seeds, seeds faint
        for lam in P2.LAMBDAS:
            es = [float(r["excess_cost"]) for r in rows2 if r.get("env") == name and r.get("arch") == "mlp"
                  and r.get("arm") == "coupled" and abs(float(r["lam"]) - lam) < 1e-9 and r.get("excess_cost") is not None]
            if es:
                bx.plot([x_of(lam)] * len(es), es, marker=".", ms=2.5, color="0.6", ls="none", alpha=0.8)
                bx.plot([x_of(lam)], [iqm(es)], marker="o", ms=3.0, color="k", ls="none")
    for a in (ax, bx):
        a.set_xscale("log")
        a.set_yscale("log")
        a.axhline(1.0, color="0.75", lw=0.5)
        a.set_yticks([], minor=True)
    ax.invert_xaxis()
    ax.set_xlim(0.25, X_ONE * 0.8)
    bx.set_xticks([1 - t for t in TICKS] + [X_ONE])
    bx.set_xticklabels([f"{t:g}" for t in TICKS] + ["1"], rotation=35, ha="right", rotation_mode="anchor")
    bx.set_xticks([], minor=True)
    bx.set_xlabel(r"GAE parameter $\lambda$  (axis: $\log(1-\lambda)$)")
    ax.set_ylim(0.95, 2.6); ax.set_yticks([1, 1.2, 1.5, 2, 2.5]); ax.set_yticklabels(["1", "1.2", "1.5", "2", "2.5"])
    bx.set_ylim(0.9, 9.0); bx.set_yticks([1, 1.5, 2, 3, 5, 8]); bx.set_yticklabels(["1", "1.5", "2", "3", "5", "8"])
    ax.set_ylabel(r"$J/J_{\mathrm{ml}}$  (fixed point, linear arm)")
    bx.set_ylabel(r"$J/J_{\mathrm{ml}}$  (mlp from scratch)")
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], color="k", ls=STYLE[n], lw=1.0) for n in INSTANCES]
    l1 = [f"$\\gamma$={n.split('_g')[1].split('_')[0]}, $h$={n.split('_h')[1]}  ($\\tau$={TAU[n]})" for n in INSTANCES]
    h2 = [Line2D([], [], marker="s", ms=3.0, mfc="w", mec="k", ls="none"),
          Line2D([], [], marker="D", ms=3.6, mfc="w", mec="k", ls="none"),
          Line2D([], [], marker="^", ms=7.0, mfc="w", mec="k", ls="none"),
          Line2D([], [], marker="^", ms=3.8, color="k", ls="none")]
    l2 = ["linear arm started at $k_{AC}(\\lambda)$ (E10)", r"predicted $\lambda_{1/2}$ (continuum)",
          r"predicted $\lambda_{1/2}$ (grid-matched)", r"measured $\lambda_{1/2}$ (E10, grid)"]
    leg1 = ax.legend(h1, l1, loc="upper right", fontsize=6.3, handlelength=2.2, borderaxespad=0.2, ncol=2, columnspacing=0.8)
    ax.add_artist(leg1)
    fig.legend(h2, l2, loc="upper center", ncol=2, fontsize=6.0, handlelength=1.4, columnspacing=1.0, borderaxespad=0.1,
               bbox_to_anchor=(0.52, 1.0))
    h3 = [Line2D([], [], marker="o", ms=3.0, color="k", ls="none"), Line2D([], [], marker=".", ms=2.5, color="0.6", ls="none"),
          Line2D([], [], color="0.6", lw=0.7)]
    l3 = ["mlp from scratch, IQM of 5 seeds (E7)", "seeds", "fixed point (panel above)"]
    bx.legend(h3, l3, loc="upper right", fontsize=6.3, handlelength=1.6, borderaxespad=0.2)
    ax.text(0.01, 0.97, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    bx.text(0.01, 0.97, "b", transform=bx.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.935))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_costlaw{'_wide' if wide else ''}.{ext}"), dpi=220)
    print("knee lambda_half per instance: measured E10 grid | predicted grid-matched | predicted continuum")
    for n, k, kp, kc in knees:
        print(f"  {n}: {k:.6f} | {kp:.6f} | {kc:.6f}   (|measured - grid-matched| = {abs(k - kp):.4f})")
    print("wrote", os.path.join(FIGS, f"fig_costlaw{'_wide' if wide else ''}.pdf/.png"))


if __name__ == "__main__":
    main(wide="--wide" in sys.argv)
