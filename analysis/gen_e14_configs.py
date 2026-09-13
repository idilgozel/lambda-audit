"""Generate the E14 config files FROM prereg_e14.py, so the deployed runs
cannot disagree with the frozen constants (sigma, eval episodes, lambda grid,
budget). deploy.sh runs this with --check as part of its pre-flight.

    python3 analysis/gen_e14_configs.py            # (re)write the files
    python3 analysis/gen_e14_configs.py --check    # exit 1 if they differ

Files
    myriad/configs_e14_probe.txt   4 lines: mlp, gru128 x Easy, Hard at
                                   lambda 1.0. Submitted as tasks 1-4, i.e.
                                   seed 1 for every line.
    myriad/configs_e14_easy.txt    16 lines: 7 non-1.0 lambdas x 2 archs,
    myriad/configs_e14_hard.txt    THEN the two lambda 1.0 lines LAST (15,16).

WHY THE 1.0 LINES ARE LAST. pilot_array.qsub.sh maps task t to config line
((t-1) mod 16)+1 and seed ((t-1) div 16)+1, so tasks 1-16 are seed 1. The
probe has already produced seed 1 at lambda 1.0 for both archs, and E12
showed what happens when a grid re-runs its probe cells: the reference cell
carried seed 1 twice and the scorer had to be taught to refuse. With the 1.0
lines at 15 and 16, the grid is submitted as TWO arrays that skip them:
    tasks 1-14   seed 1, lambdas below 1.0          (lam_e14ea / lam_e14ha)
    tasks 17-80  seeds 2-5, every lambda            (lam_e14eb / lam_e14hb)
80 grid tasks + 4 probe = 160 rows, every cell exactly once, no post-hoc
splitting.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import prereg_e14 as m  # noqa: E402


def line(env, arch, lam):
    return (f"--env {env} --arch {arch} --gae-lambda {lam} --gamma {m.GAMMA} "
            f"--total-timesteps {m.TOTAL_STEPS} --fix-sigma {m.FIX_SIGMA} "
            f"--final-eval-episodes {m.FINAL_EVAL_EPISODES} "
            f"--results-name results_e14.jsonl")


def build():
    files = {}
    files["configs_e14_probe.txt"] = [line(e, a, m.REF_LAM)
                                      for e in m.ENVS for a in m.ARCHS]
    for env in m.ENVS:
        name = f"configs_e14_{m.SHORT[env].lower()}.txt"
        non_ref = [l for l in m.LAMBDAS if l != m.REF_LAM]
        rows = [line(env, a, l) for a in m.ARCHS for l in non_ref]
        rows += [line(env, a, m.REF_LAM) for a in m.ARCHS]
        assert len(rows) == 16 and rows[14:] == [line(env, a, m.REF_LAM) for a in m.ARCHS]
        files[name] = rows
    return files


def main():
    m._assert_frozen()
    files = build()
    check = "--check" in sys.argv
    bad = 0
    for name, rows in files.items():
        path = os.path.join(ROOT, "myriad", name)
        text = "\n".join(rows) + "\n"
        if check:
            have = open(path).read() if os.path.isfile(path) else None
            ok = have == text
            print(f"  {'ok ' if ok else 'BAD'} {name}"
                  + ("" if ok else "  (absent or differs from prereg_e14.py)"))
            bad += not ok
        else:
            with open(path, "w") as f:
                f.write(text)
            print(f"  wrote {name}: {len(rows)} lines")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
