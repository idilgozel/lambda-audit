#!/usr/bin/env python3
"""Analysis for the lambda-audit pilot, beyond the frozen gates.

The gates live in prereg_lambda_pilot.py and are scored by running that file.
This script never reimplements them. It imports the frozen constants and the
frozen IQM definition so that every number here rests on the same footing as
the gate verdicts.

What it produces:

  1. Per cell IQM of deployed_return over the 5 seeds, with 95 percent
     stratified bootstrap confidence intervals, 10,000 resamples over seeds.
     The bootstrap is written directly against numpy. No new dependencies.
  2. The G2 quantity as a function of lambda. gap(lam) is the best of the
     memory capable architectures minus mlp, per environment, with CIs.
  3. The G3 classification table. Per cell median absolute gap_lambda against
     the frozen threshold of 0.25.
  4. Learning curves for every RepeatPreviousMedium cell, 5 seeds overlaid,
     one small multiples figure per lambda.

It also runs an integrity pass before any of that. Missing cells are mapped
back to array task ids and printed as a ready to paste resubmission command.

Usage:
    python3 analysis/analyze_pilot.py
    python3 analysis/analyze_pilot.py --jsonl path/to/results.jsonl

Exit status is 0 when the integrity pass is clean, 1 when it is not. The
analysis still runs either way, on whatever rows are present.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The frozen preregistration is the single source of truth for the grid, the
# gate constants and the IQM definition. Imported, never copied, never edited.
import prereg_lambda_pilot as prereg  # noqa: E402

ENV_ORDER = ["RepeatPreviousEasy", "RepeatPreviousMedium",
             "PositionOnlyCartPoleEasy"]
ARCH_ORDER = list(prereg.ARCHS)
LAM_ORDER = list(prereg.LAMBDAS)
SEEDS = list(prereg.SEEDS)

N_BOOT = 10_000
BOOT_SEED = 20260810  # fixed so the intervals are reproducible run to run
CI = (2.5, 97.5)

# Architecture styling. Black only. Architectures are separated by line style
# and marker, never by colour, so the figures survive greyscale printing.
ARCH_STYLE = {
    "mlp":     {"ls": (0, (1, 2)),   "marker": "o", "label": "MLP"},
    "stack8":  {"ls": (0, (4, 2)),   "marker": "s", "label": "Frame stack 8"},
    "stack64": {"ls": (0, (6, 1, 1, 1)), "marker": "^", "label": "Frame stack 64"},
    "gru128":  {"ls": "solid",       "marker": "D", "label": "GRU 128"},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def locate_jsonl(explicit=None):
    """collect.sh untars into results/, so the file can sit at either depth."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for cand in (os.path.join(ROOT, "results", "results",
                              "results_lambda_pilot.jsonl"),
                 os.path.join(ROOT, "results", "results_lambda_pilot.jsonl")):
        if os.path.isfile(cand):
            return cand
    return None


def locate_curves(explicit=None):
    if explicit:
        return explicit
    for cand in (os.path.join(ROOT, "results", "results"),
                 os.path.join(ROOT, "results")):
        if glob.glob(os.path.join(cand, "curves_*.csv")):
            return cand
    return os.path.join(ROOT, "results", "results")


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key_of(r):
    return (r["env"], r["arch"], round(float(r["lam"]), 4), int(r["seed"]))


def dedupe(rows):
    """Keep the LAST row per grid cell and seed. Returns (kept, n_dropped)."""
    seen = {}
    for r in rows:
        seen[key_of(r)] = r
    if len(seen) == len(rows):
        return list(seen.values()), 0
    return list(seen.values()), len(rows) - len(seen)


# ---------------------------------------------------------------------------
# Grid arithmetic. Verified against configs_pilot.txt lines 1, 16, 17 and 48.
# ---------------------------------------------------------------------------

def config_line(env, arch, lam):
    return (ENV_ORDER.index(env) * 16
            + ARCH_ORDER.index(arch) * 4
            + LAM_ORDER.index(lam) + 1)


def task_id(env, arch, lam, seed):
    return (seed - 1) * 48 + config_line(env, arch, lam)


def compress_ranges(ids):
    """[1,2,3,7,9,10] becomes '1-3,7,9-10' for the qsub -t argument."""
    ids = sorted(ids)
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if j == i else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


# ---------------------------------------------------------------------------
# Integrity, Task B
# ---------------------------------------------------------------------------

def integrity(rows, curves_dir, n_dropped):
    print("=" * 74)
    print(" INTEGRITY")
    print("=" * 74)
    ok = True

    grid = [r for r in rows if r["seed"] in SEEDS]
    cap = [r for r in rows if r["seed"] not in SEEDS]
    expected = len(ENV_ORDER) * len(ARCH_ORDER) * len(LAM_ORDER) * len(SEEDS)
    print(f"\n  grid rows      : {len(grid)} of {expected}")
    print(f"  capability rows: {len(cap)} (excluded from every gate)")
    if n_dropped:
        print(f"  duplicates     : {n_dropped} dropped, kept the LAST row "
              f"per cell and seed")
    else:
        print("  duplicates     : none")

    have = {key_of(r) for r in grid}
    missing = []
    for env in ENV_ORDER:
        for arch in ARCH_ORDER:
            for lam in LAM_ORDER:
                for seed in SEEDS:
                    if (env, arch, round(lam, 4), seed) not in have:
                        missing.append((env, arch, lam, seed))

    if missing:
        ok = False
        print(f"\n  MISSING {len(missing)} cells:")
        by_cell = {}
        for env, arch, lam, seed in missing:
            by_cell.setdefault((env, arch, lam), []).append(seed)
        for (env, arch, lam), seeds in sorted(by_cell.items()):
            print(f"    {env:26} {arch:8} lam={lam:<5} seeds={sorted(seeds)}")
        ids = [task_id(e, a, l, s) for e, a, l, s in missing]
        print("\n  Resubmit exactly these tasks. Run on Myriad from "
              "~/Scratch/lambda-audit:")
        print(f"    qsub -N lam_pilot -t {compress_ranges(ids)} "
              f"-v CFG=configs_pilot.txt pilot_array.qsub.sh")
    else:
        print("\n  missing cells  : none")

    # Curves. One file per run, matched both directions.
    curve_files = glob.glob(os.path.join(curves_dir, "curves_*.csv"))
    curve_keys = set()
    for p in curve_files:
        base = os.path.basename(p)[len("curves_"):-len(".csv")]
        try:
            env, arch, lampart, seedpart = base.rsplit("_", 3)
            curve_keys.add((env, arch, round(float(lampart[3:]), 4),
                            int(seedpart[1:])))
        except ValueError:
            print(f"    unparsable curve filename: {base}")
            ok = False
    row_keys = {key_of(r) for r in rows}
    no_curve = row_keys - curve_keys
    no_row = curve_keys - row_keys
    print(f"\n  curve files    : {len(curve_files)}")
    if no_curve:
        ok = False
        print(f"    rows with NO curve ({len(no_curve)}):")
        for k in sorted(no_curve)[:20]:
            print(f"      {k}")
    if no_row:
        ok = False
        print(f"    curves with NO row ({len(no_row)}):")
        for k in sorted(no_row)[:20]:
            print(f"      {k}")
    if not no_curve and not no_row:
        print("    every row has a curve and every curve has a row")

    # gap_lambda must be identically zero at lambda = 1.0 by construction.
    bad = [r for r in rows
           if math.isclose(float(r["lam"]), 1.0)
           and r.get("gap_lambda") is not None
           and float(r["gap_lambda"]) != 0.0]
    print("\n  gap_lambda at lambda=1.0:")
    if bad:
        ok = False
        print(f"    NONZERO in {len(bad)} rows. This is a harness bug and it "
              f"blocks analysis.")
        for r in bad[:10]:
            print(f"      {r['env']} {r['arch']} seed {r['seed']}: "
                  f"{r['gap_lambda']!r}")
    else:
        n1 = len([r for r in rows if math.isclose(float(r["lam"]), 1.0)])
        print(f"    exactly 0.0 in all {n1} rows, as required")

    print(f"\n  INTEGRITY: {'CLEAN' if ok else 'PROBLEMS ABOVE'}")
    return ok


# ---------------------------------------------------------------------------
# Bootstrap. Stratified over seeds, resampled within each cell independently.
# ---------------------------------------------------------------------------

def iqm_np(a, axis=-1):
    """Vectorised interquartile mean, identical to prereg.iqm for any n."""
    a = np.sort(np.asarray(a, dtype=float), axis=axis)
    n = a.shape[axis]
    lo, hi = n // 4, n - n // 4
    if hi <= lo:
        return a.mean(axis=axis)
    return np.take(a, range(lo, hi), axis=axis).mean(axis=axis)


def _check_iqm_matches_prereg(values):
    """Guard: the vectorised IQM must equal the frozen scalar one exactly."""
    if len(values) == 0:
        return
    mine = float(iqm_np(np.asarray(values)))
    theirs = float(prereg.iqm(list(values)))
    if not math.isclose(mine, theirs, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(
            f"FATAL: vectorised IQM {mine!r} disagrees with the frozen "
            f"prereg IQM {theirs!r}. Refusing to report numbers.")


def boot_indices(rng, n, b=N_BOOT):
    return rng.integers(0, n, size=(b, n))


def cell_values(rows, env, arch, lam, field="deployed_return"):
    vals = [(r["seed"], float(r[field])) for r in rows
            if r["env"] == env and r["arch"] == arch
            and math.isclose(float(r["lam"]), lam) and r["seed"] in SEEDS]
    vals.sort()
    return np.array([v for _, v in vals], dtype=float)


def iqm_with_ci(rng, values):
    if values.size == 0:
        return None
    _check_iqm_matches_prereg(values)
    point = float(iqm_np(values))
    draws = values[boot_indices(rng, values.size)]
    stat = iqm_np(draws, axis=1)
    lo, hi = np.percentile(stat, CI)
    return point, float(lo), float(hi), int(values.size)


def gap_with_ci(rng, rows, env, lam):
    """gap(lam) = max over memory capable IQM, minus mlp IQM.

    Seeds are resampled independently within each architecture, which is the
    stratified scheme. The max is taken inside every bootstrap replicate, so
    the interval carries the selection through rather than around it.
    """
    mem_archs = [a for a in prereg.G2_MEMORY_ARCHS
                 if a in prereg.MEMORY_CAPABLE[env]] or \
                [a for a in prereg.MEMORY_CAPABLE[env]]
    mlp = cell_values(rows, env, "mlp", lam)
    mems = {a: cell_values(rows, env, a, lam) for a in mem_archs}
    mems = {a: v for a, v in mems.items() if v.size}
    if mlp.size == 0 or not mems:
        return None
    point = max(float(iqm_np(v)) for v in mems.values()) - float(iqm_np(mlp))
    mlp_draw = iqm_np(mlp[boot_indices(rng, mlp.size)], axis=1)
    mem_draw = np.max(
        np.vstack([iqm_np(v[boot_indices(rng, v.size)], axis=1)
                   for v in mems.values()]), axis=0)
    stat = mem_draw - mlp_draw
    lo, hi = np.percentile(stat, CI)
    return point, float(lo), float(hi), sorted(mems)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def per_cell_table(rng, rows, out_dir):
    print("\n" + "=" * 74)
    print(" PER CELL IQM OF deployed_return, 95% stratified bootstrap CI")
    print(f" {N_BOOT:,} resamples over seeds, seed {BOOT_SEED}")
    print("=" * 74)
    recs = []
    for env in ENV_ORDER:
        print(f"\n  {env}")
        print(f"    {'arch':9} {'lam':>5} {'n':>2}  {'IQM':>8}  "
              f"{'95% CI':>18}")
        for arch in ARCH_ORDER:
            for lam in LAM_ORDER:
                res = iqm_with_ci(rng, cell_values(rows, env, arch, lam))
                if res is None:
                    print(f"    {arch:9} {lam:5} {'0':>2}  {'missing':>8}")
                    recs.append(dict(env=env, arch=arch, lam=lam, n=0,
                                     iqm="", ci_lo="", ci_hi=""))
                    continue
                point, lo, hi, n = res
                print(f"    {arch:9} {lam:5} {n:2}  {point:8.4f}  "
                      f"[{lo:7.4f}, {hi:7.4f}]")
                recs.append(dict(env=env, arch=arch, lam=lam, n=n,
                                 iqm=f"{point:.6f}", ci_lo=f"{lo:.6f}",
                                 ci_hi=f"{hi:.6f}"))
    write_csv(os.path.join(out_dir, "per_cell_iqm.csv"), recs,
              ["env", "arch", "lam", "n", "iqm", "ci_lo", "ci_hi"])
    return recs


def gap_table(rng, rows, out_dir):
    print("\n" + "=" * 74)
    print(" G2 QUANTITY: gap(lambda) = best memory IQM minus mlp IQM")
    print("=" * 74)
    recs = []
    for env in ENV_ORDER:
        print(f"\n  {env}")
        print(f"    {'lam':>5}  {'gap':>8}  {'95% CI':>18}   memory archs")
        for lam in LAM_ORDER:
            res = gap_with_ci(rng, rows, env, lam)
            if res is None:
                print(f"    {lam:5}  {'missing':>8}")
                recs.append(dict(env=env, lam=lam, gap="", ci_lo="", ci_hi="",
                                 archs=""))
                continue
            point, lo, hi, archs = res
            print(f"    {lam:5}  {point:8.4f}  [{lo:7.4f}, {hi:7.4f}]   "
                  f"{','.join(archs)}")
            recs.append(dict(env=env, lam=lam, gap=f"{point:.6f}",
                             ci_lo=f"{lo:.6f}", ci_hi=f"{hi:.6f}",
                             archs="|".join(archs)))
    write_csv(os.path.join(out_dir, "gap_vs_lambda.csv"), recs,
              ["env", "lam", "gap", "ci_lo", "ci_hi", "archs"])

    # The frozen G2 rule restated on the transition env, for orientation only.
    # The verdict of record comes from running prereg_lambda_pilot.py.
    t = prereg.TRANSITION_ENV
    g10 = gap_with_ci(rng, rows, t, 1.0)
    g09 = gap_with_ci(rng, rows, t, 0.9)
    if g10 and g09:
        thresh = prereg.G2_SHRINK_FRACTION * g10[0]
        print(f"\n  For orientation only, the frozen gate is scored by "
              f"prereg_lambda_pilot.py:")
        print(f"    gap(1.0) = {g10[0]:.4f}, so G2 needs gap(0.9) < "
              f"{thresh:.4f}. Observed gap(0.9) = {g09[0]:.4f}.")
    return recs


def g3_table(rows, out_dir):
    print("\n" + "=" * 74)
    print(f" G3 DIAGNOSTIC: median |gap_lambda| per cell, threshold "
          f"{prereg.G3_FLAG_THRESHOLD}")
    print("=" * 74)
    excluded = {(e, l) for e, l in prereg.G3_EXCLUDED_CELLS}
    recs = []
    for env in ENV_ORDER:
        print(f"\n  {env}")
        print(f"    {'arch':9} {'lam':>5} {'n':>2}  {'med|gap_lam|':>12}  "
              f"{'flagged':>7}  label")
        for arch in ARCH_ORDER:
            for lam in LAM_ORDER:
                vals = cell_values(rows, env, arch, lam, field="gap_lambda")
                if vals.size == 0:
                    continue
                med = float(np.median(np.abs(vals)))
                flagged = med > prereg.G3_FLAG_THRESHOLD
                if (env, lam) in excluded:
                    label = "excluded, boundary"
                elif env == prereg.TRANSITION_ENV and lam in (0.9, 0.95):
                    label = "below horizon"
                else:
                    label = "above horizon"
                print(f"    {arch:9} {lam:5} {vals.size:2}  {med:12.4f}  "
                      f"{'yes' if flagged else 'no':>7}  {label}")
                recs.append(dict(env=env, arch=arch, lam=lam, n=int(vals.size),
                                 median_abs_gap_lambda=f"{med:.6f}",
                                 flagged=int(flagged), label=label))
    write_csv(os.path.join(out_dir, "g3_cells.csv"), recs,
              ["env", "arch", "lam", "n", "median_abs_gap_lambda", "flagged",
               "label"])
    return recs


def write_csv(path, recs, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.edgecolor": "black",
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })
    return plt


def lam_axis(ax):
    """Evenly spaced lambda ticks.

    The grid is 0.9, 0.95, 0.99, 1.0. On a linear axis the last two collide at
    any readable font size, so lambda is plotted categorically. Positions are
    indices, labels are the lambda values.
    """
    ax.set_xticks(range(len(LAM_ORDER)))
    ax.set_xticklabels([str(l) for l in LAM_ORDER])
    ax.set_xlim(-0.35, len(LAM_ORDER) - 0.65)
    ax.set_xlabel(r"GAE $\lambda$")


def knee_position(knee):
    """Place the critical horizon inside the categorical axis."""
    if knee <= LAM_ORDER[0] or knee >= LAM_ORDER[-1]:
        return None
    for i in range(len(LAM_ORDER) - 1):
        a, b = LAM_ORDER[i], LAM_ORDER[i + 1]
        if a <= knee <= b:
            return i + (knee - a) / (b - a)
    return None


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name + ".pdf"))
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300)
    print(f"    wrote {name}.pdf and {name}.png")


def fig_gap(plt, rng, rows, fig_dir):
    fig, axes = plt.subplots(1, len(ENV_ORDER), figsize=(9.5, 3.0),
                             sharey=True)
    for ax, env in zip(np.atleast_1d(axes), ENV_ORDER):
        xs, ys, los, his = [], [], [], []
        for i, lam in enumerate(LAM_ORDER):
            res = gap_with_ci(rng, rows, env, lam)
            if res is None:
                continue
            xs.append(i)
            ys.append(res[0])
            los.append(res[0] - res[1])
            his.append(res[2] - res[0])
        if xs:
            ax.errorbar(xs, ys, yerr=[los, his], color="black", marker="o",
                        markersize=3.5, capsize=2.5, linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle=(0, (1, 3)))
        kp = knee_position(prereg.ENVS[env][1])
        if kp is not None:
            ax.axvline(kp, color="black", linewidth=0.8,
                       linestyle=(0, (4, 2)))
            ax.annotate(f"critical horizon {prereg.ENVS[env][1]}",
                        xy=(kp, ax.get_ylim()[1]),
                        xytext=(-3, -6), textcoords="offset points",
                        rotation=90, ha="right", va="top", fontsize=6.5)
        ax.set_title(env, fontsize=8)
        lam_axis(ax)
    np.atleast_1d(axes)[0].set_ylabel("memory advantage over MLP")
    fig.suptitle("Memory advantage against the bootstrap horizon, "
                 "IQM with 95% bootstrap CI", fontsize=9)
    save(fig, fig_dir, "fig_gap_vs_lambda")
    plt.close(fig)


def fig_iqm(plt, rng, rows, fig_dir):
    fig, axes = plt.subplots(1, len(ENV_ORDER), figsize=(9.5, 3.0),
                             sharey=True)
    for ax, env in zip(np.atleast_1d(axes), ENV_ORDER):
        for arch in ARCH_ORDER:
            xs, ys, los, his = [], [], [], []
            for i, lam in enumerate(LAM_ORDER):
                res = iqm_with_ci(rng, cell_values(rows, env, arch, lam))
                if res is None:
                    continue
                xs.append(i)
                ys.append(res[0])
                los.append(res[0] - res[1])
                his.append(res[2] - res[0])
            if not xs:
                continue
            st = ARCH_STYLE[arch]
            ax.errorbar(xs, ys, yerr=[los, his], color="black",
                        linestyle=st["ls"], marker=st["marker"],
                        markersize=3.5, markerfacecolor="white",
                        capsize=2.0, linewidth=1.1, label=st["label"])
        kp = knee_position(prereg.ENVS[env][1])
        if kp is not None:
            ax.axvline(kp, color="black", linewidth=0.8,
                       linestyle=(0, (4, 2)))
        ax.set_title(env, fontsize=8)
        lam_axis(ax)
    np.atleast_1d(axes)[0].set_ylabel("IQM deployed return")
    last = np.atleast_1d(axes)[-1]
    if last.get_legend_handles_labels()[0]:
        last.legend(fontsize=6.5, frameon=False, loc="best")
    fig.suptitle("Per architecture performance against the bootstrap horizon",
                 fontsize=9)
    save(fig, fig_dir, "fig_iqm_by_lambda")
    plt.close(fig)


def fig_curves(plt, curves_dir, fig_dir, env=None):
    env = env or prereg.TRANSITION_ENV
    made = 0
    for lam in LAM_ORDER:
        fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(9.5, 2.9),
                                 sharey=True, sharex=True)
        any_data = False
        for ax, arch in zip(np.atleast_1d(axes), ARCH_ORDER):
            for seed in SEEDS:
                p = os.path.join(
                    curves_dir, f"curves_{env}_{arch}_lam{lam}_s{seed}.csv")
                if not os.path.isfile(p):
                    continue
                step, val = [], []
                with open(p) as f:
                    for row in csv.DictReader(f):
                        v = row.get("eval_return", "")
                        if v not in ("", None):
                            step.append(float(row["global_step"]))
                            val.append(float(v))
                if step:
                    # millions of steps, so no 1e7 offset label collides
                    # with the axis title
                    ax.plot([s / 1e6 for s in step], val, color="black",
                            linewidth=0.7, alpha=0.75)
                    any_data = True
            ax.set_title(ARCH_STYLE[arch]["label"], fontsize=8)
            ax.set_xlabel("environment steps, millions")
        if not any_data:
            plt.close(fig)
            continue
        np.atleast_1d(axes)[0].set_ylabel("evaluation return")
        fig.suptitle(f"{env}, GAE lambda = {lam}, five seeds overlaid",
                     fontsize=9, y=1.05)
        save(fig, fig_dir, f"fig_curves_{env}_lam{lam}")
        plt.close(fig)
        made += 1
    if made == 0:
        print("    no curve files matched, skipped the learning curve figures")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--curves-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "analysis"))
    args = ap.parse_args()

    path = locate_jsonl(args.jsonl)
    if path is None:
        raise SystemExit(
            "No results file found. Run 'bash myriad/collect.sh ucapgoz' "
            "first, or pass --jsonl.")
    curves_dir = locate_curves(args.curves_dir)
    print(f"results : {path}")
    print(f"curves  : {curves_dir}")

    rows, n_dropped = dedupe(load_rows(path))
    clean = integrity(rows, curves_dir, n_dropped)

    rng = np.random.default_rng(BOOT_SEED)
    tables = os.path.join(args.out_dir, "tables")
    figs = os.path.join(args.out_dir, "figures")

    per_cell_table(rng, rows, tables)
    gap_table(rng, rows, tables)
    g3_table(rows, tables)

    print("\n" + "=" * 74)
    print(" FIGURES")
    print("=" * 74)
    plt = setup_mpl()
    fig_iqm(plt, np.random.default_rng(BOOT_SEED), rows, figs)
    fig_gap(plt, np.random.default_rng(BOOT_SEED), rows, figs)
    fig_curves(plt, curves_dir, figs)

    print(f"\ntables in {tables}")
    print(f"figures in {figs}")
    print("\nThe gate verdicts of record come from:")
    print(f"    python3 prereg_lambda_pilot.py {path}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
