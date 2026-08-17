#!/usr/bin/env python3
"""E6 figures. Same conventions as analyze_pilot.py: serif, black, line styles.

  fig_e6_a  the minimal pair. J_rate against lambda for both arms, log J,
            with J_ml and the rThK0 ceiling drawn, and every seed shown.
  fig_e6_b  measured gain against lambda with the frozen k_AC(lambda) curve
            overlaid and k_ml marked. The whole experiment in one panel.
  fig_e6_c  per seed J at 0.95 and 0.99 as strip plots. Bimodality on one
            side of the critical horizon and its absence on the other.

Run:  python3 analysis/figures_e6.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import prereg_phase2 as P  # noqa: E402

DATA = os.path.join(ROOT, "results", "results", "results_phase2.jsonl")
FIGS = os.path.join(ROOT, "analysis", "figures")
INSTANCE = P.E6_INSTANCE
LAMS = P.LAMBDAS
KNEE = P.LAMBDA_HALF[INSTANCE]
J_ML = P.BASELINES[INSTANCE]["J_ml"]
K_ML = P.BASELINES[INSTANCE]["k_ml"]
LAW = P.P73_LAW


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
        "axes.grid": False, "lines.linewidth": 1.2, "figure.dpi": 150,
        "savefig.bbox": "tight", "text.color": "black",
        "axes.labelcolor": "black", "xtick.color": "black",
        "ytick.color": "black", "axes.edgecolor": "black",
    })
    return plt


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=300)
    print(f"    wrote {name}.pdf and {name}.png")


def load():
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    out = {}
    for arm in ("coupled", "decoupled"):
        out[arm] = {}
        for lam in LAMS:
            out[arm][lam] = [r for r in rows
                             if r.get("arm") == arm
                             and abs(float(r["lam"]) - lam) < 1e-9]
    return out


def xpos(lam):
    return LAMS.index(lam)


def lam_axis(ax):
    """Categorical lambda, since 0.995 and 0.999 collide on a linear axis."""
    ax.set_xticks(range(len(LAMS)))
    ax.set_xticklabels([str(l) for l in LAMS], fontsize=7.5)
    ax.set_xlim(-0.4, len(LAMS) - 0.6)
    ax.set_xlabel(r"GAE $\lambda$")


def knee_x():
    for i in range(len(LAMS) - 1):
        if LAMS[i] <= KNEE <= LAMS[i + 1]:
            return i + (KNEE - LAMS[i]) / (LAMS[i + 1] - LAMS[i])
    return None


def fig_a(plt, d):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    styles = {"coupled": dict(ls="solid", marker="o", label="Coupled"),
              "decoupled": dict(ls=(0, (1, 2)), marker="s", label="Decoupled")}
    for arm, st in styles.items():
        med, xs = [], []
        for lam in LAMS:
            J = [r["J_rate"] for r in d[arm][lam]]
            if not J:
                continue
            xs.append(xpos(lam))
            med.append(float(np.median(J)))
            ax.plot([xpos(lam)] * len(J), J, marker=".", linestyle="none",
                    color="black", markersize=3, alpha=0.45)
        ax.plot(xs, med, color="black", linestyle=st["ls"], marker=st["marker"],
                markersize=4, markerfacecolor="white", label=st["label"])
    ax.axhline(J_ML, color="black", lw=0.7, ls=(0, (4, 2)))
    ax.annotate(r"$J_{ml}=%.4f$" % J_ML, xy=(len(LAMS) - 0.65, J_ML),
                fontsize=6.5, va="bottom", ha="right")
    ax.axhline(LAW, color="black", lw=0.7, ls=(0, (1, 3)))
    ax.annotate(r"$r\Theta K(0)=%.3f$" % LAW, xy=(len(LAMS) - 0.65, LAW),
                fontsize=6.5, va="bottom", ha="right")
    kx = knee_x()
    if kx is not None:
        ax.axvline(kx, color="black", lw=0.8, ls=(0, (5, 2)))
        ax.annotate(f"critical horizon {KNEE:.3f}", xy=(kx, ax.get_ylim()[1]),
                    xytext=(-3, -4), textcoords="offset points", rotation=90,
                    ha="right", va="top", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylabel("deployed cost rate J")
    lam_axis(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_title("E6, the minimal pair. One term in the state update.",
                 fontsize=9)
    save(fig, "fig_e6_a")
    plt.close(fig)


def fig_b(plt, d):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    kac = P.K_AC_LAMBDA[INSTANCE]
    ax.plot(range(len(LAMS)), kac, color="black", lw=1.0, ls=(0, (4, 2)),
            marker="^", markersize=3.5, markerfacecolor="white",
            label=r"$k_{AC}(\lambda)$, frozen prediction")
    xs, med = [], []
    for lam in LAMS:
        g = [r["final_gain"] for r in d["coupled"][lam]]
        if not g:
            continue
        xs.append(xpos(lam))
        med.append(float(np.median(g)))
        ax.plot([xpos(lam)] * len(g), g, marker=".", linestyle="none",
                color="black", markersize=4, alpha=0.5)
    ax.plot(xs, med, color="black", lw=1.3, marker="o", markersize=4,
            markerfacecolor="white", label="measured gain, coupled")
    gd = [float(np.median([r["final_gain"] for r in d["decoupled"][lam]]))
          for lam in LAMS]
    ax.plot(range(len(LAMS)), gd, color="black", lw=1.0, ls=(0, (1, 2)),
            marker="s", markersize=3.5, markerfacecolor="white",
            label="measured gain, decoupled")
    ax.axhline(K_ML, color="black", lw=0.7, ls=(0, (1, 3)))
    ax.annotate(r"$k_{ml}=%.3f$" % K_ML, xy=(0.1, K_ML), xytext=(0, 3),
                textcoords="offset points", fontsize=6.5)
    kx = knee_x()
    if kx is not None:
        ax.axvline(kx, color="black", lw=0.8, ls=(0, (5, 2)))
    # symlog, not log. The decoupled gains straddle zero and several are
    # slightly negative, and a log axis silently drops every one of them,
    # which reads as though the arm has no data at those lambdas.
    ax.set_yscale("symlog", linthresh=0.05, linscale=0.5)
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    ax.set_ylabel("linear gain k")
    lam_axis(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("E6 gain against the frozen prediction. Agreement above the "
                 "horizon, scatter below.", fontsize=8.5)
    save(fig, "fig_e6_b")
    plt.close(fig)


def fig_c(plt, d):
    pair = [0.95, 0.99]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.6), sharey=True)
    for ax, lam in zip(axes, pair):
        J = sorted(r["J_rate"] for r in d["coupled"][lam])
        rng = np.random.default_rng(7)
        jitter = rng.uniform(-0.10, 0.10, size=len(J))
        ax.plot(jitter, J, marker="o", linestyle="none", color="black",
                markersize=6, markerfacecolor="white")
        # Label only the extremes. At lambda 0.99 the five seeds agree to
        # three decimals, so labelling each one produces an unreadable smear.
        # min below its marker, max above its own, so the two labels cannot
        # collide even when the seeds agree to three decimals
        ax.annotate(f"{J[0]:.3f}", xy=(jitter[0], J[0]), xytext=(10, -7),
                    textcoords="offset points", fontsize=7, va="center")
        ax.annotate(f"{J[-1]:.3f}", xy=(jitter[-1], J[-1]), xytext=(10, 7),
                    textcoords="offset points", fontsize=7, va="center")
        ax.axhline(J_ML, color="black", lw=0.7, ls=(0, (4, 2)))
        ax.set_xlim(-0.6, 0.8)
        ax.set_xticks([])
        side = "below" if lam < KNEE else "above"
        spread = max(J) / min(J)
        ax.set_title(rf"$\lambda={lam}$, {side} the horizon"
                     "\n" rf"max/min $= {spread:.1f}$", fontsize=8.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("deployed cost rate J, one point per seed")
    fig.suptitle("E6 per seed cost. Escape or blow up on one side only.",
                 fontsize=9, y=1.10)
    save(fig, "fig_e6_c")
    plt.close(fig)


def main():
    d = load()
    plt = setup()
    print("  E6 figures")
    fig_a(plt, d)
    fig_b(plt, d)
    fig_c(plt, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
