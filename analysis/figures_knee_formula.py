#!/usr/bin/env python3
"""The knee formula (knee-formula-findings.md, equation II) against the exact knee.

  fig_knee_formula  (a) predicted knee horizon H_1/2 = 1/[(R_m-1)/C_m + (1-beta)]
                        against the exact cost-midpoint knee horizon 1/(1 - beta
                        lambda_E), log-log, for the four paper instances, the three
                        Prony kernels and the 25 one-parameter slices of
                        analysis/knee_formula.py --slices. (b) the error of the
                        first-order expansion, lambda_asym - lambda_E, against the
                        expansion parameter 1 - lambda_E. Hollow markers are the
                        first-order (bistable) instances, Pi = Theta c/(h s_e) > 100,
                        for which the exact cost-midpoint knee is the jump of the
                        largest-root branch rather than a smooth crossing.

Reads analysis/knee_formula_table.json (written by analysis/knee_formula.py --slices).

Run:  python3 analysis/figures_knee_formula.py   -> analysis/figures/fig_knee_formula.{pdf,png}
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "analysis", "knee_formula_table.json")
FIGS = os.path.join(ROOT, "analysis", "figures")
PI_FIRST_ORDER = 100.0
DEFAULT = dict(c=0.3, h=0.05, s_e=0.09, theta=1.0)


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


def pi_of(name):
    """Theta c / (h s_e) from the row's name; defaults are the paper's instance."""
    p = dict(DEFAULT)
    m = re.match(r"GLEAM_g([\d.]+)_h([\d.]+)", name)
    if m:
        p["h"] = float(m.group(2))
    m = re.match(r"(c|h|s_e|theta)=([\d.]+)$", name)
    if m:
        p[m.group(1)] = float(m.group(2))
    return p["theta"] * p["c"] / (p["h"] * p["s_e"])


def kind_of(name):
    if name.startswith("GLEAM_"):
        return "instance"
    if "mode" in name:
        return "prony"
    return "slice"


def main():
    plt = setup()
    os.makedirs(FIGS, exist_ok=True)
    rows = json.load(open(TABLE))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    style = {"instance": dict(marker="o", ms=4.5), "prony": dict(marker="^", ms=4.5), "slice": dict(marker=".", ms=4.0)}
    seen = set()
    worst_small = 0.0
    for r in rows:
        kind = kind_of(r["name"])
        first = pi_of(r["name"]) > PI_FIRST_ORDER
        H_E = 1.0 / (1.0 - r["beta"] * r["lamE"])
        err = r["lam_asym"] - r["lamE"]
        kw = dict(style[kind], color="k", mfc=("w" if first else "k"), mec="k", mew=0.7, ls="none")
        ax.plot([H_E], [r["H_asym"]], **kw)
        bx.plot([1 - r["lamE"]], [err], **kw)
        seen.add((kind, first))
        if (1 - r["lamE"]) <= 0.05 and not first:
            worst_small = max(worst_small, abs(err))
        if kind == "instance":
            ax.annotate(r["name"].replace("GLEAM_", ""), (H_E, r["H_asym"]), textcoords="offset points", xytext=(4, -7), fontsize=5.5)
    g = np.geomspace(2, 800, 50)
    ax.plot(g, g, color="0.4", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2, 800); ax.set_ylim(2, 800)
    ax.set_xlabel(r"exact knee horizon $1/(1-\beta\lambda_E)$  (steps)")
    ax.set_ylabel(r"formula (II): $1/[(R_m-1)/C_m + (1-\beta)]$")
    bx.axhline(0, color="0.4", lw=0.7)
    bx.axhspan(-0.003, 0.003, color="0.9", lw=0)
    bx.set_xscale("log")
    bx.set_xlabel(r"expansion parameter $1-\lambda_E$")
    bx.set_ylabel(r"$\lambda_{\mathrm{asym}} - \lambda_E$")
    bx.set_ylim(-0.08, 0.08)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", ls="none", **style["instance"]), Line2D([], [], color="k", ls="none", **style["prony"]),
         Line2D([], [], color="k", ls="none", **style["slice"]), Line2D([], [], color="k", mfc="w", ls="none", **style["instance"])]
    l = ["paper instances", "Prony kernels (dim 3-4)", "one-parameter slices", r"first-order ($\Pi>100$): knee is a jump"]
    bx.legend(h, l, loc="lower left", fontsize=6.0, handlelength=1.4, borderaxespad=0.2)
    ax.text(0.02, 0.97, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    bx.text(0.02, 0.97, "b", transform=bx.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.tight_layout(pad=0.3, w_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_knee_formula.{ext}"), dpi=220)
    print(f"{len(rows)} knees; worst |lambda_asym - lambda_E| for continuous instances with 1 - lambda <= 0.05: {worst_small:.4f}")
    print("wrote", os.path.join(FIGS, "fig_knee_formula.pdf/.png"))


if __name__ == "__main__":
    main()
