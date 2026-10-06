#!/usr/bin/env python3
"""The first-order collapse on GLEAM_g1_h0.02 (knee-formula-findings.md, section 4).

  fig_bistability   equilibria of the averaged GAE(lambda) update, i.e. the roots
                    of the exact gradient ghat(k; lambda) of identity (I), against
                    lambda near the knee. Solid: stable branches; dashed: the
                    unstable branch; shaded: the window [0.99529, 0.99655] in which
                    a low-gain and a high-gain equilibrium coexist. The paper's
                    k_AC(lambda) is the largest root and jumps at the window's
                    upper edge. Overlaid, where the committed results contain
                    them: the deployed gain of every mlp seed trained from scratch
                    (E7, circles), the median gain of the linear arm started at
                    k_AC (E10, squares), and the E17 drift-direction cells (arrows
                    from the start gain to the median deployed gain), which test
                    the sign of ghat between the branches.

Run:  python3 analysis/figures_bistability.py   -> analysis/figures/fig_bistability.{pdf,png}
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "gleam"))
import kernel_general as kg  # noqa: E402
from gleam_bench import GLEBench  # noqa: E402

NAME = "GLEAM_g1_h0.02"
PHASE2 = os.path.join(ROOT, "results", "results", "results_phase2.jsonl")
E10 = os.path.join(ROOT, "results", "results", "results_e10.jsonl")
E17 = os.path.join(ROOT, "results", "results", "results_e17.jsonl")
FIGS = os.path.join(ROOT, "analysis", "figures")
LAM_LO, LAM_HI = 0.9880, 0.9992          # plotted range
TICKS = [0.988, 0.99, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]


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


def roots(b, lam, n=320):
    ks = np.geomspace(0.3, 1.9 / b.h, n)
    vs = np.array([kg.ghat(b, k, lam) for k in ks])
    out = []
    for i in range(len(ks) - 1):
        if np.isfinite(vs[i]) and np.isfinite(vs[i + 1]) and np.sign(vs[i]) != np.sign(vs[i + 1]):
            r = brentq(lambda k: kg.ghat(b, k, lam), ks[i], ks[i + 1], xtol=1e-8)
            out.append((r, vs[i + 1] > vs[i]))
    return out


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def main():
    plt = setup()
    os.makedirs(FIGS, exist_ok=True)
    kg.validate()
    b = GLEBench(cs=(0.3,), gammas=(1.0,), h=0.02)
    lams = 1.0 - np.geomspace(1 - LAM_LO, 1 - LAM_HI, 260)
    stable, unstable = [], []
    window = []
    for lam in lams:
        rs = roots(b, lam)
        st = [r for r, s in rs if s]; un = [r for r, s in rs if not s]
        for r in st:
            stable.append((lam, r))
        for r in un:
            unstable.append((lam, r))
        if len(st) >= 2:
            window.append(lam)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    if window:
        ax.axvspan(1 - min(window), 1 - max(window), color="0.9", lw=0, zorder=0)
    # draw branches as point clouds dense enough to read as curves
    s = np.array(stable); u = np.array(unstable)
    ax.plot(1 - s[:, 0], s[:, 1], ".", ms=1.6, color="k")
    if len(u):
        ax.plot(1 - u[:, 0], u[:, 1], ".", ms=1.6, color="0.55")
    # E7 mlp seeds
    rows2 = [r for r in load(PHASE2) if r.get("env") == NAME and r.get("arch") == "mlp" and r.get("arm") == "coupled"]
    for r in rows2:
        lam = float(r["lam"])
        if LAM_LO <= lam <= LAM_HI and r.get("final_gain") is not None:
            ax.plot([1 - lam], [float(r["final_gain"])], marker="o", ms=3.0, mfc="w", mec="k", mew=0.6, ls="none", zorder=4)
    # E10 linear medians
    rows10 = [r for r in load(E10) if r.get("env") == NAME and r.get("arch") == "linear"]
    for lam in sorted({float(r["lam"]) for r in rows10}):
        if LAM_LO <= lam <= LAM_HI:
            ks = [float(r["final_gain"]) for r in rows10 if abs(float(r["lam"]) - lam) < 1e-9]
            ax.plot([1 - lam], [float(np.median(ks))], marker="s", ms=3.4, color="k", ls="none", zorder=5)
    # E17 drift cells
    rows17 = [r for r in load(E17) if r.get("env") == NAME and r.get("arch") == "linear" and r.get("linear_gain_init") is not None]
    cells = {}
    for r in rows17:
        cells.setdefault((round(float(r["lam"]), 6), round(float(r["linear_gain_init"]), 6)), []).append(float(r["final_gain"]))
    for (lam, k0), ks in sorted(cells.items()):
        if LAM_LO <= lam <= LAM_HI:
            med = float(np.median(ks))
            ax.annotate("", xy=(1 - lam, med), xytext=(1 - lam, k0),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=6))
            ax.plot([1 - lam], [k0], marker="_", ms=5, color="k", ls="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlim(1 - LAM_LO, 1 - LAM_HI)
    ax.set_xticks([1 - t for t in TICKS]); ax.set_xticklabels([f"{t:g}" for t in TICKS], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xticks([], minor=True)
    ax.set_ylim(0.5, 70)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50"]); ax.set_yticks([], minor=True)
    ax.set_xlabel(r"GAE parameter $\lambda$  (axis: $\log(1-\lambda)$)")
    ax.set_ylabel("gain $k$")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", lw=1.2), Line2D([], [], color="0.55", lw=1.2, ls=":"),
         Line2D([], [], marker="o", ms=3.0, mfc="w", mec="k", ls="none"), Line2D([], [], marker="s", ms=3.4, color="k", ls="none")]
    l = ["stable equilibria", "unstable equilibrium", "mlp from scratch, each seed (E7)", "linear arm from $k_{AC}$, median (E10)"]
    if cells:
        h.append(Line2D([], [], marker=r"$\downarrow$", ms=6, color="k", ls="none")); l.append("E17: start gain $\\to$ median deployed gain")
    ax.legend(h, l, loc="lower left", fontsize=6.2, handlelength=1.6, borderaxespad=0.2)
    if window:
        ax.text(1 - 0.5 * (min(window) + max(window)), 60, f"bistable\n[{min(window):.5f}, {max(window):.5f}]",
                ha="center", va="top", fontsize=6)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_bistability.{ext}"), dpi=220)
    print(f"window on the plotted grid: [{min(window):.5f}, {max(window):.5f}]" if window else "no window found")
    print("wrote", os.path.join(FIGS, "fig_bistability.pdf/.png"))


if __name__ == "__main__":
    main()
