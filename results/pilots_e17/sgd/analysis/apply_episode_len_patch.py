"""Apply the continuing-training option: `--gleam-episode-len N` (default 0 = the
instance's shipped episode length, 1024 steps).

Why (e17-pilot-findings.md): GleamNative truncates every 1024 steps and the
training wrapper then RESETS from the uncontrolled stationary distribution
(Var v = 1). The settling transient after each reset costs the controller an
amount that grows with the gain, so the TRAINING objective (episodic, reset
every 1024 steps) has a gain-gradient the CONTINUING objective (the theory's
and the deployed evaluation's) does not have: +6e-4 to +1e-2 per step at k = 10
to 80 on GLEAM_g1_h0.02, against an aliasing gradient ghat of at most 5e-4.
Setting N far above the per-env step budget makes training continuing (no
reset ever happens), so the trained objective is the one the theory analyses.

Two edits, both inert at the default: gleam_native.py gains a module-level
EPISODE_LEN override consulted by the registry entries; ppo_lambda.py gains the
flag, sets the override before any environment is built, and records
`gleam_episode_len` in the row ONLY when the flag is set, so every existing row
keeps its schema. Idempotent: refuses to apply twice.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pg = os.path.join(ROOT, "harness", "gleam_native.py")
pp = os.path.join(ROOT, "harness", "ppo_lambda.py")
g = open(pg).read(); s = open(pp).read()
if "EPISODE_LEN" in g or "gleam_episode_len" in s:
    print("already applied"); sys.exit(0)

g_edits = [
    ('ACT_LIMIT = 40.0\n',
     'ACT_LIMIT = 40.0\n'
     '# E17 (2026-10-05): training-time episode length override. None keeps the\n'
     '# shipped 1024-step truncation; ppo_lambda.py sets it from --gleam-episode-len.\n'
     'EPISODE_LEN = None\n'),
    ('def _entry(instance, coupled):\n    def make():\n        return GleamNative(instance, coupled=coupled)\n',
     'def _entry(instance, coupled):\n    def make():\n        return GleamNative(instance, coupled=coupled, episode_len=EPISODE_LEN)\n'),
]
s_edits = [
    ('    p.add_argument("--linear-sgd-lr", type=float, default=0.0)\n',
     '    p.add_argument("--linear-sgd-lr", type=float, default=0.0)\n'
     '    # E17 (2026-10-05): continuing training for GLEAM. 0 (default) keeps the\n'
     '    # shipped 1024-step episodes; a value above the per-env step budget means\n'
     '    # no reset ever happens and the trained objective is the continuing one.\n'
     '    p.add_argument("--gleam-episode-len", type=int, default=0)\n'),
    ('    torch.set_num_threads(args.torch_threads)\n',
     '    if args.gleam_episode_len:\n'
     '        if gleam_native is None:\n'
     '            raise SystemExit("--gleam-episode-len needs gleam_native")\n'
     '        gleam_native.EPISODE_LEN = int(args.gleam_episode_len)\n'
     '    torch.set_num_threads(args.torch_threads)\n'),
    ('        row["linear_sgd_lr"] = float(getattr(args, "linear_sgd_lr", 0.0))\n',
     '        row["linear_sgd_lr"] = float(getattr(args, "linear_sgd_lr", 0.0))\n'
     '    if args.gleam_episode_len:\n'
     '        row["gleam_episode_len"] = int(args.gleam_episode_len)\n'),
]
for old, new in g_edits:
    assert g.count(old) == 1, f"gleam_native anchor not found exactly once: {old[:50]!r}"
    g = g.replace(old, new)
for old, new in s_edits:
    assert s.count(old) == 1, f"ppo_lambda anchor not found exactly once: {old[:50]!r}"
    s = s.replace(old, new)
open(pg, "w").write(g); open(pp, "w").write(s)
print("applied: --gleam-episode-len (continuing training) in harness/ppo_lambda.py + gleam_native.py")
