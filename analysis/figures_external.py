#!/usr/bin/env python3
"""The in-scope external replications: deployed return against lambda for the
memoryless mlp and the gru128 on POPGym's PositionOnlyCartPole (E9, Easy and
Hard) and PositionOnlyPendulum (E14, Easy and Hard). IQM over seeds, seeds
shown faintly. The shaded band is the registered knee region: for Pendulum the
bracket computed from the swing period in prereg_e14.py (half to full period);
for CartPole the gap between the registered BELOW and ABOVE groups of
prereg_phase2.py. Horizontal marks on the right are the published POPGym MMER
values where the preregistration recorded them (E14).

Every number is read from the committed results files and the frozen prereg
constants. Cells that have not been collected yet are simply absent.

Run:  python3 analysis/figures_external.py   -> analysis/figures/fig_external.{pdf,png}
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import prereg_phase2 as P2  # noqa: E402
import prereg_e14 as P14  # noqa: E402

PHASE2 = os.path.join(ROOT, "results", "results", "results_phase2.jsonl")
E14 = os.path.join(ROOT, "results", "results", "results_e14.jsonl")
FIGS = os.path.join(ROOT, "analysis", "figures")
X_ONE = 4e-4
TICKS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
PANELS = [("PositionOnlyCartPoleEasy", PHASE2, "CartPole Easy (E9)"), ("PositionOnlyCartPoleHard", PHASE2, "CartPole Hard (E9)"),
          ("PositionOnlyPendulumEasy", E14, "Pendulum Easy (E14)"), ("PositionOnlyPendulumHard", E14, "Pendulum Hard (E14)")]
ARCHS = [("mlp", "o", "-", "memoryless mlp"), ("gru128", "s", "--", "gru128")]


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


def x_of(lam):
    return X_ONE if lam >= 1.0 else 1.0 - lam


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def iqm(xs):
    xs = sorted(xs); n = len(xs); lo, hi = n // 4, n - n // 4
    m = xs[lo:hi] if hi > lo else xs
    return sum(m) / len(m)


def main():
    plt = setup()
    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(6.75, 3.9))
    for ax, (env, path, title) in zip(axes.ravel(), PANELS):
        rows = [r for r in load(path) if r.get("env") == env]
        for arch, mk, ls, label in ARCHS:
            xs, ys = [], []
            for lam in P2.LAMBDAS:
                vs = [float(r["deployed_return"]) for r in rows if r.get("arch") == arch
                      and abs(float(r["lam"]) - lam) < 1e-9 and r.get("deployed_return") is not None]
                if not vs:
                    continue
                ax.plot([x_of(lam)] * len(vs), vs, marker=".", ms=2.2, color="0.65", ls="none", alpha=0.8)
                xs.append(x_of(lam)); ys.append(iqm(vs))
                ax.text(x_of(lam), 0.02, str(len(vs)) if len(vs) < len(P2.SEEDS) else "", transform=ax.get_xaxis_transform(),
                        fontsize=5, ha="center", color="0.4")
            if xs:
                ax.plot(xs, ys, marker=mk, ms=3.2, ls=ls, color="k", mfc=("k" if arch == "mlp" else "w"), mew=0.8, label=label)
        # knee region
        if "Pendulum" in env:
            lo, hi = P14.KNEE_BRACKET
            ax.axvspan(1 - lo, 1 - hi, color="0.88", lw=0, zorder=0)
            pub = P14.POPGYM.get(env)
            if pub:
                for k, v in pub.items():
                    ax.plot([1 - 0.999, X_ONE], [v, v], color="0.3", lw=0.8, ls=":")
                    ax.text(1 - 0.9985, v, f"POPGym {k} {v:.3f}", fontsize=5.2, va="bottom", ha="right", color="0.3")
        else:
            ax.axvspan(1 - max(P2.P91_LOW_LAMBDAS), 1 - min(P2.P91_HIGH_LAMBDAS), color="0.88", lw=0, zorder=0)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0, 0].invert_xaxis()
    axes[0, 0].set_xlim(0.25, X_ONE * 0.8)
    for ax in axes[1]:
        ax.set_xticks([1 - t for t in TICKS] + [X_ONE])
        ax.set_xticklabels([f"{t:g}" for t in TICKS] + ["1"], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xticks([], minor=True)
        ax.set_xlabel(r"GAE parameter $\lambda$  (axis: $\log(1-\lambda)$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("deployed return (MMER)")
    axes[0, 0].legend(loc="lower right", fontsize=6.5, handlelength=2.0)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_external.{ext}"), dpi=220)
    print("wrote", os.path.join(FIGS, "fig_external.pdf/.png"))


if __name__ == "__main__":
    main()
