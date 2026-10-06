"""Apply the critic-curvature diagnostic: `--log-critic-fit 1` (default 0 = off).

Why (e17-pilot-findings.md): in identity (I) the whole upward (over-control)
displacement of the equilibrium comes from the critic's curvature theta2 -- with
a constant baseline the exact gradient is positive (down) at every gain above
k_1, close to the true gradient. The pilots show PPO's one-parameter arm drifting
down everywhere, i.e. behaving as if its critic carried little curvature. This
records what curvature the critic actually learned: at the final deployed
evaluation the critic's values along the trajectory are already collected
(`ep_v`, with the observations `obs_v`); a least-squares quadratic fit
V(v) ~ a + b v + c v^2 over them is stored as `critic_fit = [a, b, c]` together
with the deployed observation variance `v_var`. Compare c with -theta2(k, lambda)
of vendor/gleam/kernel_general.py (values are returns of rewards = -cost).

Inert by default: the fit is computed into the eval's ctrl dict, and written to
the row only when the flag is set, so every existing row keeps its schema.
Idempotent: refuses to apply twice.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "harness", "ppo_lambda.py")
s = open(p).read()
if "log_critic_fit" in s:
    print("already applied"); sys.exit(0)

edits = [
    ('''                gap_l = float((np.array(gl) - gs_arr).mean() / sd)
        else:
            gap_v, gap_l = 0.0, 0.0
    return mean_ret, gap_v, gap_l, ctrl
''',
     '''                gap_l = float((np.array(gl) - gs_arr).mean() / sd)
        else:
            gap_v, gap_l = 0.0, 0.0
        # E17 (2026-10-05): the curvature the critic actually learned, fitted
        # over the deployed trajectory. Diagnostic; written only with --log-critic-fit.
        if len(ep_v) >= 100 and len(obs_v) >= len(ep_v):
            vv = np.asarray(obs_v[:len(ep_v)], dtype=np.float64)
            yy = np.asarray(ep_v, dtype=np.float64)
            A = np.stack([np.ones_like(vv), vv, vv * vv], 1)
            coef = np.linalg.lstsq(A, yy, rcond=None)[0]
            ctrl["critic_fit"] = [float(c) for c in coef]
            ctrl["v_var"] = float(vv.var())
    return mean_ret, gap_v, gap_l, ctrl
'''),
    ('''    p.add_argument("--gleam-episode-len", type=int, default=0)
''',
     '''    p.add_argument("--gleam-episode-len", type=int, default=0)
    # E17 (2026-10-05): record the quadratic fit of the critic along the deployed trajectory.
    p.add_argument("--log-critic-fit", type=int, default=0)
'''),
    ('''    if args.gleam_episode_len:
        row["gleam_episode_len"] = int(args.gleam_episode_len)
''',
     '''    if args.gleam_episode_len:
        row["gleam_episode_len"] = int(args.gleam_episode_len)
    if args.log_critic_fit and ctrl.get("critic_fit") is not None:
        row["critic_fit"] = ctrl["critic_fit"]
        row["v_var"] = ctrl["v_var"]
'''),
]
for old, new in edits:
    assert s.count(old) == 1, f"anchor not found exactly once: {old[:60]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("applied: harness/ppo_lambda.py accepts --log-critic-fit (off by default)")
