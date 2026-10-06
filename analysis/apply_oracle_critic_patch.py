"""Apply the critic-swap option: `--oracle-critic lstd` (default "" = off).

Why (e17-pilot-findings.md): the trained MLP critic of the linear arm carries
none of the quadratic curvature theta2 that identity (I) attributes the
over-control displacement to (fitted +0.07 against -199 at k = 23, lambda = 0.9
on GLEAM_g1_h0.02), and the arm drifts the way a critic-free learner would.
This option replaces the critic's VALUES in the GAE -- the rollout values, the
truncation bootstrap and the final bootstrap -- by the exact LSTD(lambda)
aliased quadratic value of vendor/gleam/kernel_general.theta2 at the arm's
current gain: V(v) = -theta2(k, lambda) v^2 (reward units; the constant term
cancels in the advantage). The actor, the sampling, PPO's surrogate, the
optimiser and the value loss (which keeps training the now-unused network
critic) are untouched. If the arm then climbs to k_AC(lambda) from below and
descends to it from above, the fixed point of the theory is realised by the
learner once the critic is the one the theory assumes; the default run's
failure is then the critic's, not the theory's. Linear arms on GLEAM only.

Inert by default. Idempotent: refuses to apply twice. Requires the
--gleam-episode-len patch (anchors) and vendor/gleam/ importable at run time.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "harness", "ppo_lambda.py")
s = open(p).read()
if "oracle_critic" in s:
    print("already applied"); sys.exit(0)

edits = [
    ('''    p.add_argument("--log-critic-fit", type=int, default=0)
''',
     '''    p.add_argument("--log-critic-fit", type=int, default=0)
    # E17 (2026-10-05): replace the critic's values in the GAE by the exact LSTD(lambda)
    # aliased quadratic value at the current gain (linear arms on GLEAM only).
    p.add_argument("--oracle-critic", type=str, default="", choices=["", "lstd"])
'''),
    ('''    if args.gleam_episode_len:
        if gleam_native is None:
            raise SystemExit("--gleam-episode-len needs gleam_native")
        gleam_native.EPISODE_LEN = int(args.gleam_episode_len)
''',
     '''    if args.gleam_episode_len:
        if gleam_native is None:
            raise SystemExit("--gleam-episode-len needs gleam_native")
        gleam_native.EPISODE_LEN = int(args.gleam_episode_len)
    oracle = None
    if args.oracle_critic == "lstd":
        if gleam_native is None or not args.arch.startswith("linear") or args.fix_sigma is None:
            raise SystemExit("--oracle-critic lstd is defined for --arch linear/linearb on GLEAM with --fix-sigma.")
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "gleam"))
        import kernel_general as _kg
        from gleam_bench import GLEBench as _GLEBench
        _ip = gleam_native.instance_params(args.env.replace("_dec", "").split("_nl")[0])
        _bench = _GLEBench(cs=(_ip["c"],), gammas=(_ip["gamma_bath"],), theta=_ip["theta"], h=_ip["h"])
        _s_e = float(args.fix_sigma) ** 2
        _state = {"k": None, "th2": 0.0}

        def oracle(agent_, obs):
            """-theta2(k, lambda) * v^2 at the arm's current gain (recomputed when k moves)."""
            k = -float(agent_.actor.weight.detach().reshape(-1)[0].item()) * agent_.gain_scale
            if _state["k"] is None or abs(k - _state["k"]) > 1e-9:
                th2 = float(_kg.theta2(_bench, k, args.gae_lambda, _s_e)) if k > 0 else float("nan")
                _state["k"], _state["th2"] = k, (th2 if th2 == th2 else 0.0)
            return -_state["th2"] * obs[:, 0] ** 2
'''),
    ('''            with torch.no_grad():
                a, lp, _, v, h = agent.act(next_obs, h, next_done)
            acts[step], logps[step], vals[step] = a, lp, v
''',
     '''            with torch.no_grad():
                a, lp, _, v, h = agent.act(next_obs, h, next_done)
                if oracle is not None:
                    v = oracle(agent, next_obs)
            acts[step], logps[step], vals[step] = a, lp, v
'''),
    ('''                with torch.no_grad():
                    fv = agent.value(torch.tensor(final, dtype=torch.float32),
                                     h, torch.zeros(args.num_envs))
''',
     '''                with torch.no_grad():
                    fv = agent.value(torch.tensor(final, dtype=torch.float32),
                                     h, torch.zeros(args.num_envs))
                    if oracle is not None:
                        fv = oracle(agent, torch.tensor(final, dtype=torch.float32))
'''),
    ('''        with torch.no_grad():
            next_val = agent.value(next_obs, h, next_done)
            adv = torch.zeros_like(rews)
''',
     '''        with torch.no_grad():
            next_val = agent.value(next_obs, h, next_done)
            if oracle is not None:
                next_val = oracle(agent, next_obs)
            adv = torch.zeros_like(rews)
'''),
    ('''    if args.log_critic_fit and ctrl.get("critic_fit") is not None:
''',
     '''    if args.oracle_critic:
        row["oracle_critic"] = args.oracle_critic
    if args.log_critic_fit and ctrl.get("critic_fit") is not None:
'''),
]
for old, new in edits:
    assert s.count(old) == 1, f"anchor not found exactly once: {old[:70]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("applied: harness/ppo_lambda.py accepts --oracle-critic lstd (off by default)")
