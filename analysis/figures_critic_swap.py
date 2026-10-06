#!/usr/bin/env python3
"""The critic swap (E19; pilots in e17-pilot-findings.md section 3).

  fig_critic_swap   For each instance and lambda, the linear arm's gain after 15M
                    steps of continuing training against its start gain, as an
                    arrow from start to median end over seeds (seed ends as dots).
                    Black: PPO with the exact LSTD(lambda) critic of identity (I)
                    in its GAE; grey: PPO with its own trained MLP critic. The
                    dashed line is the predicted equilibrium k_AC(lambda). With the
                    theory's critic the actor is attracted to k_AC from both sides;
                    with its own critic it drifts toward under-control.

Reads results/results/results_e19.jsonl. With --pilots it reads the seed-101 pilot
rows of results/pilots_e17/ instead (lambda 0.9, GLEAM_g1_h0.02, starts 10/30/80).

Run:  python3 analysis/figures_critic_swap.py [--pilots]   -> analysis/figures/fig_critic_swap{,_pilots}.{pdf,png}
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIGS = os.path.join(ROOT, "analysis", "figures")
E19 = os.path.join(ROOT, "results", "results", "results_e19.jsonl")
PILOTS = [os.path.join(ROOT, "results", "pilots_e17", f) for f in ("pilots_e17_oracle.jsonl", "pilots_e17_noreset.jsonl", "pilots_e17_criticfit.jsonl")]


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


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def k_ac(inst, lam):
    import prereg_phase2 as P2
    return P2.K_AC_LAMBDA[inst][P2.LAMBDAS.index(lam)]


def cells_from_e19(rows):
    import prereg_e19 as P
    out = {}
    for r in P.rows_of(rows):
        key = (r["env"], round(float(r["lam"]), 6), round(float(r["linear_gain_init"]), 6), P.critic_of(r))
        out.setdefault(key, []).append(float(r["final_gain"]))
    return out


def cells_from_pilots(rows):
    out = {}
    for r in rows:
        run = r["run"]
        if r.get("gleam_episode_len") != 10_000_000 or r.get("arch") != "linear" or r.get("linear_sgd_lr"):
            continue
        m = re.search(r"_k([\d.]+)_s", run)
        if not m:
            continue
        critic = "oracle" if r.get("oracle_critic") == "lstd" else "default"
        key = (r["env"], round(float(r["lam"]), 6), round(float(m.group(1)), 6), critic)
        out.setdefault(key, []).append(float(r["final_gain"]))
    return out


def main(pilots=False):
    plt = setup()
    os.makedirs(FIGS, exist_ok=True)
    rows = sum((load(p) for p in PILOTS), []) if pilots else load(E19)
    cells = cells_from_pilots(rows) if pilots else cells_from_e19(rows)
    if not cells:
        raise SystemExit("no rows to draw")
    panels = sorted({(e, l) for e, l, _, _ in cells})
    fig, axes = plt.subplots(1, len(panels), figsize=(min(6.75, max(3.4, 2.3 * len(panels) + 0.6)), 2.9 if len(panels) == 1 else 2.6), squeeze=False)
    for ax, (inst, lam) in zip(axes[0], panels):
        kac = k_ac(inst, lam)
        ax.axhline(kac, color="0.3", lw=0.8, ls="--")
        ax.text(0.02, kac, r" $k_{AC}$", va="bottom", ha="left", fontsize=7, transform=ax.get_yaxis_transform())
        starts = sorted({k for e, l, k, _ in cells if e == inst and l == lam})
        for i, k0 in enumerate(starts):
            for critic, color, dx in (("default", "0.6", -0.12), ("oracle", "k", 0.12)):
                ends = cells.get((inst, lam, k0, critic))
                if not ends:
                    continue
                med = float(np.median(ends))
                x = i + dx
                ax.annotate("", xy=(x, med), xytext=(x, k0),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=0, shrinkB=0, mutation_scale=7))
                ax.plot([x] * len(ends), ends, ".", color=color, ms=3)
                ax.plot([x], [k0], "_", color=color, ms=7, mew=1.2)
        ax.set_xticks(range(len(starts)))
        ax.set_xticklabels([f"{k:.1f}".rstrip("0").rstrip(".") for k in starts])
        ax.set_xlim(-0.6, len(starts) - 0.4)
        ax.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 3.0, 5.0, 7.0)))
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(plain); ax.yaxis.set_minor_formatter(plain)
        ax.tick_params(axis="y", which="minor", labelsize=6.5)
        ax.set_xlabel("start gain $k_0$")
        ax.set_title(f"{inst.replace('GLEAM_', '')}, $\\lambda$ = {lam:g}", fontsize=8, pad=3)
    axes[0][0].set_ylabel("gain after 15M steps")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", lw=1.1, marker=">", ms=4), Line2D([], [], color="0.6", lw=1.1, marker=">", ms=4),
         Line2D([], [], color="0.3", lw=0.8, ls="--")]
    l = ["PPO with the exact LSTD($\\lambda$) critic", "PPO with its trained MLP critic", "predicted equilibrium $k_{AC}(\\lambda)$"]
    ncol = 3 if len(panels) >= 3 else 1
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=6.5, handlelength=1.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3, rect=(0, 0.08 if ncol == 3 else 0.2, 1, 1))
    name = f"fig_critic_swap{'_pilots' if pilots else ''}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=220)
    print("wrote", os.path.join(FIGS, name + ".pdf/.png"))


if __name__ == "__main__":
    main(pilots="--pilots" in sys.argv)
