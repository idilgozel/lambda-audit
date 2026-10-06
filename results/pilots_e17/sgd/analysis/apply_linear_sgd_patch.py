"""Apply the E17 vehicle change: plain SGD on the gain of a linear arm.

Adds `--linear-sgd-lr ETA` (default 0.0 = off). When set, and only for
--arch linear / linearb, the single gain parameter is updated by vanilla SGD
with step ETA (annealed like the Adam lr) while every other parameter keeps
the Adam optimiser, the global gradient-norm clip and the lr of the published
harness. Rationale (prereg_e17.py): Adam's normalised step is a skew-robust
average of the sampled gradient, so a one-parameter policy under Adam settles
where the TYPICAL sampled gradient vanishes, not where its MEAN does; the
pre-freeze pilots measured that displacement at k0 = 10 as a downward drift
comparable to the effect under test. SGD's expected step is proportional to
the mean sampled gradient, so the drift of the gain is an unbiased estimate
of the direction of the averaged update, which is what the test needs.

With the flag off nothing changes: same optimiser object, same clipping, same
RNG consumption, so every existing row (E6-E16) reproduces as before.
Idempotent: refuses to apply twice. Run from the repo root after E16's runs have
all exited.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "harness", "ppo_lambda.py")
s = open(p).read()
if '"--linear-sgd-lr"' in s:
    print("already applied"); sys.exit(0)

edits = [
    ('    p.add_argument("--linear-bias-init", type=float, default=0.0)\n',
     '    p.add_argument("--linear-bias-init", type=float, default=0.0)\n'
     '    # E17 (2026-10-05): vanilla SGD on the gain of a linear arm, Adam elsewhere.\n'
     '    # 0.0 (default) leaves the published optimiser untouched.\n'
     '    p.add_argument("--linear-sgd-lr", type=float, default=0.0)\n'),
    ('    opt = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)\n',
     '    if args.linear_sgd_lr > 0:\n'
     '        if not agent.linear_policy:\n'
     '            raise SystemExit("--linear-sgd-lr is defined for --arch linear/linearb only.")\n'
     '        _other = [q for n, q in agent.named_parameters() if n != "actor.weight"]\n'
     '        opt = optim.Adam(_other, lr=args.lr, eps=1e-5)\n'
     '        opt_k = optim.SGD([agent.actor.weight], lr=args.linear_sgd_lr)\n'
     '        clip_params = _other\n'
     '    else:\n'
     '        opt = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)\n'
     '        opt_k = None\n'
     '        clip_params = list(agent.parameters())\n'),
    ('            opt.param_groups[0]["lr"] = frac * args.lr\n',
     '            opt.param_groups[0]["lr"] = frac * args.lr\n'
     '            if opt_k is not None:\n'
     '                opt_k.param_groups[0]["lr"] = frac * args.linear_sgd_lr\n'),
    ('                opt.zero_grad()\n'
     '                loss.backward()\n'
     '                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)\n'
     '                opt.step()\n',
     '                opt.zero_grad()\n'
     '                if opt_k is not None:\n'
     '                    opt_k.zero_grad()\n'
     '                loss.backward()\n'
     '                nn.utils.clip_grad_norm_(clip_params, args.max_grad_norm)\n'
     '                opt.step()\n'
     '                if opt_k is not None:\n'
     '                    opt_k.step()\n'),
]
for old, new in edits:
    assert s.count(old) == 1, f"anchor not found exactly once: {old[:60]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("applied: harness/ppo_lambda.py accepts --linear-sgd-lr (off by default)")
