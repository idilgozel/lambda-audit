#!/usr/bin/env python3
"""E10: the one-parameter linear arm started at the exact equilibrium k_AC(lambda).

  fig_e10_drift   measured deployed gain (median of 3 seeds, seed range as a bar)
                  against the equilibrium it was started at, 32 cells (4 instances
                  x 8 lambdas), log-log, identity line. Filled markers: cells on
                  the inflated (over-control) branch above the knee; hollow
                  markers: cells on the collapsed branch (k_AC near k_ml). Every
                  cell sits below the line -- the uniform downward displacement of
                  e10-above-knee-drift.md -- and the 15% band (dotted) shows why
                  the registered relative tolerance localised the P11.1 misses to
                  the collapsed branch. Inset text: drift in gain units per cell.

Run:  python3 analysis/figures_e10_drift.py   -> analysis/figures/fig_e10_drift.{pdf,png}
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import prereg_phase2 as P2  # noqa: E402
import prereg_e10 as P10  # noqa: E402

E10 = os.path.join(ROOT, "results", "results", "results_e10.jsonl")
FIGS = os.path.join(ROOT, "analysis", "figures")
MARK = {"GLEAM_g1_h0.02": "o", "GLEAM_g1_h0.05": "s", "GLEAM_g2_h0.05": "D", "GLEAM_g4_h0.05": "^"}
COLLAPSED_MAX = 5.0        # k_AC below this is the collapsed branch (k_ml = 1.2-1.7)


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


def main():
    plt = setup()
    os.makedirs(FIGS, exist_ok=True)
    rows = [json.loads(l) for l in open(E10) if l.strip()]
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    lo_all, hi_all = 1e9, 0
    n_below, n_cells = 0, 0
    for name in P2.E7_INSTANCES:
        for lam, k_ac in zip(P2.LAMBDAS, P10.C[name]["k_AC"]):
            ks = [float(r["final_gain"]) for r in rows if r.get("env") == name and r.get("arch") == "linear"
                  and abs(float(r["lam"]) - lam) < 1e-9 and r.get("final_gain") is not None]
            if not ks:
                continue
            n_cells += 1
            med = float(np.median(ks))
            n_below += med < k_ac
            collapsed = k_ac < COLLAPSED_MAX
            ax.plot([k_ac, k_ac], [min(ks), max(ks)], color="0.5", lw=0.6)
            ax.plot([k_ac], [med], marker=MARK[name], ms=3.6, mfc=("w" if collapsed else "k"), mec="k", mew=0.7, ls="none")
            lo_all, hi_all = min(lo_all, k_ac, min(ks)), max(hi_all, k_ac, max(ks))
    g = np.geomspace(lo_all * 0.7, hi_all * 1.3, 50)
    ax.plot(g, g, color="0.3", lw=0.7)
    ax.plot(g, 0.85 * g, color="0.3", lw=0.6, ls=":")
    ax.plot(g, 1.15 * g, color="0.3", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo_all * 0.7, hi_all * 1.3); ax.set_ylim(lo_all * 0.7, hi_all * 1.3)
    ax.set_xlabel(r"equilibrium gain started at, $k_{AC}(\lambda)$")
    ax.set_ylabel("deployed gain after 15M steps")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=MARK[n], ms=3.6, color="k", ls="none") for n in P2.E7_INSTANCES]
    l = [f"$\\gamma$={n.split('_g')[1].split('_')[0]}, $h$={n.split('_h')[1]}" for n in P2.E7_INSTANCES]
    h += [Line2D([], [], marker="o", ms=3.6, mfc="k", mec="k", ls="none"), Line2D([], [], marker="o", ms=3.6, mfc="w", mec="k", ls="none"),
          Line2D([], [], color="0.3", lw=0.6, ls=":")]
    l += ["inflated branch", "collapsed branch", r"$\pm$15% (P11.1 tolerance)"]
    ax.legend(h, l, loc="upper left", fontsize=6.3, handlelength=1.5, borderaxespad=0.2)
    ax.text(0.98, 0.04, f"{n_below}/{n_cells} cells below the line", transform=ax.transAxes, ha="right", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_e10_drift.{ext}"), dpi=220)
    print(f"{n_below}/{n_cells} cell medians below k_AC;", "wrote", os.path.join(FIGS, "fig_e10_drift.pdf/.png"))


if __name__ == "__main__":
    main()
