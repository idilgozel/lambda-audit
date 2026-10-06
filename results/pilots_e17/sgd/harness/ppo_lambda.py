"""Single-file PPO for the lambda audit pilot.

Purpose: compare memory architectures across GAE lambda on POPGym environments,
under one harness with everything except lambda and the architecture held fixed.

Design notes, deliberately explicit because a silent harness bug cost us a
round once already (the --frame-skip flag that was never implemented):

1. No gymnasium vector wrapper. Autoreset semantics changed in gymnasium 1.0
   and the failure mode is silent. The vector loop here is written out.
2. Truncation and termination are distinguished. Terminated bootstraps from
   zero. Truncated bootstraps from the value of the real final observation.
3. The GRU minibatches split by environment, never by timestep, so every
   sequence stays intact for backprop through time.
4. Frame stacks are zero filled at episode start, not carried across episodes.
5. Every run writes a curves CSV and one results.jsonl row. No exceptions.

Usage:
    python ppo_lambda.py --env RepeatPreviousMedium --arch gru128 \
        --gae-lambda 0.95 --seed 1 --total-timesteps 15000000 --out-dir ../results
"""

import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

import envs_native

# Environments come from envs_native, which is a bitwise-verified
# reimplementation of the POPGym originals (see test_env_equivalence.py:
# 13,542 steps compared bit for bit). This removes popgym, gymnasium and a
# numpy build from the cluster's dependency list, which is what made the
# runs portable to Myriad at all.
#
# Phase 2 adds gleam_native, verified the same way (test_gleam_equivalence.py:
# 24,000 steps bit for bit against GLEBench across four instances and both
# arms). It needs gleam_instances.npz alongside it. The import is guarded so a
# missing npz cannot break a discrete run, and asking for a GLEAM environment
# without it raises the real error rather than "unknown env".
ENV_REGISTRY = dict(envs_native.REGISTRY)
_GLEAM_IMPORT_ERROR = None
try:
    import gleam_native
    ENV_REGISTRY.update(gleam_native.REGISTRY)
except Exception as exc:                                   # pragma: no cover
    gleam_native = None
    _GLEAM_IMPORT_ERROR = exc


def make_env(env_id):
    if env_id in envs_native.REGISTRY:
        return envs_native.make(env_id)
    if gleam_native is None:
        raise RuntimeError(
            f"env {env_id} needs gleam_native, which failed to import: "
            f"{_GLEAM_IMPORT_ERROR}")
    return gleam_native.make(env_id)


def _valid_env(env_id):
    """argparse validator: accept exactly the ids make_env can construct."""
    import argparse as _ap
    if env_id in ENV_REGISTRY:
        return env_id
    if gleam_native is not None and "_nl" in env_id:
        try:
            gleam_native.make(env_id)    # constructs and discards; no RNG use
            return env_id
        except KeyError as exc:
            raise _ap.ArgumentTypeError(str(exc))
    raise _ap.ArgumentTypeError(
        f"unknown env {env_id!r}; known: {sorted(ENV_REGISTRY)} "
        f"(plus GLEAM_*_nl<value> cubic-damping variants)")


def is_continuous(space):
    return isinstance(space, envs_native.Box)


def obs_encoder(space):
    """Return (encode_fn, dim). Discrete becomes one hot, Box passes through."""
    if isinstance(space, envs_native.Discrete):
        n = int(space.n)

        def enc(o):
            v = np.zeros(n, dtype=np.float32)
            v[int(o)] = 1.0
            return v

        return enc, n
    if isinstance(space, envs_native.Box):
        dim = int(np.prod(space.shape))

        def enc(o):
            return np.asarray(o, dtype=np.float32).reshape(-1)

        return enc, dim
    raise NotImplementedError(f"unsupported observation space {space}")


class SyncVec:
    """Minimal synchronous vector env with explicit reset semantics.

    step() returns the observation actually used for the next action. When an
    episode ends, that is the reset observation of the new episode, and the
    real final observation is handed back separately so truncation can
    bootstrap correctly.
    """

    def __init__(self, env_id, num_envs, stack, seed):
        self.envs = [make_env(env_id) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.stack = stack
        self.enc, self.base_dim = obs_encoder(self.envs[0].observation_space)
        self.obs_dim = self.base_dim * stack
        aspace = self.envs[0].action_space
        self.continuous = is_continuous(aspace)
        if self.continuous:
            self.act_n = int(np.prod(aspace.shape))       # action dimension
            self.act_low = float(aspace.low)
            self.act_high = float(aspace.high)
        else:
            self.act_n = int(aspace.n)
            self.act_low = self.act_high = None
        self.buffers = [deque(maxlen=stack) for _ in range(num_envs)]
        self.ep_ret = np.zeros(num_envs, dtype=np.float64)
        self.ep_len = np.zeros(num_envs, dtype=np.int64)
        self.returns_log = deque(maxlen=200)
        self._seed = seed

    def _fill(self, i, o):
        self.buffers[i].clear()
        for _ in range(self.stack - 1):
            self.buffers[i].append(np.zeros(self.base_dim, dtype=np.float32))
        self.buffers[i].append(self.enc(o))

    def _view(self, i):
        return np.concatenate(list(self.buffers[i]))

    def reset(self):
        out = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=self._seed + i)
            self._fill(i, o)
            out[i] = self._view(i)
        self.ep_ret[:] = 0.0
        self.ep_len[:] = 0
        return out

    def step(self, actions):
        n = self.num_envs
        nxt = np.zeros((n, self.obs_dim), dtype=np.float32)
        final = np.zeros((n, self.obs_dim), dtype=np.float32)
        rew = np.zeros(n, dtype=np.float32)
        term = np.zeros(n, dtype=np.float32)
        trunc = np.zeros(n, dtype=np.float32)
        for i, e in enumerate(self.envs):
            if self.continuous:
                # clip to the declared bounds before stepping. The environment
                # itself never clips, so it stays bitwise comparable to
                # GLEBench; this is the standard continuous PPO guard against
                # an unbounded Gaussian walking the state to infinity.
                a_i = np.clip(np.asarray(actions[i], dtype=np.float64).reshape(-1),
                              self.act_low, self.act_high)
                o, r, te, tr, _ = e.step(a_i)
            else:
                o, r, te, tr, _ = e.step(int(actions[i]))
            rew[i] = r
            term[i] = float(te)
            trunc[i] = float(tr)
            self.ep_ret[i] += r
            self.ep_len[i] += 1
            self.buffers[i].append(self.enc(o))
            final[i] = self._view(i)
            if te or tr:
                self.returns_log.append(self.ep_ret[i])
                self.ep_ret[i] = 0.0
                self.ep_len[i] = 0
                o2, _ = e.reset()
                self._fill(i, o2)
            nxt[i] = self._view(i)
        return nxt, rew, term, trunc, final


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_n, arch, hidden=128, continuous=False,
                 fix_sigma=None, gain_scale=1.0, gain_init=0.0, bias_init=0.0):
        super().__init__()
        self.arch = arch
        self.hidden = hidden
        self.continuous = continuous
        self.recurrent = arch.startswith("gru")
        if self.recurrent:
            self.trunk = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden)), nn.ReLU()
            )
            self.gru = nn.GRU(hidden, hidden)
            for name, p in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(p, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(p, 1.0)
        else:
            self.trunk = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)),
                nn.Tanh(),
            )
        self.actor = layer_init(nn.Linear(hidden, act_n), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)
        # A-5 Tier 1b discriminator. The policy becomes one scalar gain,
        # u = -k v, read off the RAW observation rather than the trunk, so it
        # cannot leave the gain manifold. The critic above is untouched: same
        # trunk, same head, same value loss. Built strictly AFTER the standard
        # heads, and with a zero fill rather than an orthogonal draw, so every
        # other arch consumes exactly the RNG it consumed before this branch
        # existed. E6's 80 rows and E9's 160 rows depend on that.
        # linearb adds ONE degree of freedom to linear: a bias, u = -k v + b,
        # zero-initialised. It is the surgical probe of the Tier 2 mechanism:
        # placed at k_AC, the bias is predicted unstable below lambda* and
        # stable above, and b is the only direction it can leave through.
        self.linear_policy = arch in ("linear", "linearb")
        if self.linear_policy:
            if not continuous:
                raise SystemExit(
                    "--arch linear is a continuous-control discriminator and "
                    "has no meaning on a discrete environment.")
            if obs_dim != 1 or act_n != 1:
                raise SystemExit(
                    f"--arch linear expects a scalar observation and a scalar "
                    f"action, got obs_dim={obs_dim}, act_n={act_n}. It is "
                    f"defined only for the aliased GLEAM instances at stack 1.")
            self.actor = nn.Linear(obs_dim, act_n, bias=(arch == "linearb"))
            nn.init.zeros_(self.actor.weight)
            if arch == "linearb":
                nn.init.zeros_(self.actor.bias)
                # Perturbation-decay mode: kick the bias to b0 and watch the
                # sign of its dynamics, rather than waiting for stochastic
                # escape from exactly 0, which near the threshold can take
                # longer than any budget. mean = scale*(W v + bias), so the
                # action bias is scale*bias and we divide the kick by scale.
                if bias_init:
                    with torch.no_grad():
                        self.actor.bias.fill_(float(bias_init) / float(gain_scale))
            # Adam is scale free, so a ONE parameter policy moves its gain by at
            # most ~lr per optimiser step, while the MLP's effective gain is a
            # sum over 128 weights and moves far faster at the same lr. Without
            # a compensating scale the linear policy is optimiser limited rather
            # than equilibrium limited, and "no intermittency" would then mean
            # "never got there". The scale reparameterises k, it does not change
            # the policy class, which is the thing the discriminator is about.
            self.gain_scale = float(gain_scale)
            # Optional start AT a chosen gain. u = -k v and mean = scale*W*v, so
            # W = -k/scale. Starting at the analytic k_AC(lambda) turns the run
            # from "can a one parameter policy get there in the budget" into
            # "once there, does it stay", which is the stability question the
            # discriminator actually asks and the one the Tier 2 theory answers.
            if gain_init:
                with torch.no_grad():
                    self.actor.weight.fill_(-float(gain_init) / self.gain_scale)
        # Continuous head. Created only when needed, and from a constant, so a
        # discrete run consumes exactly the RNG it consumed before Phase 2 and
        # every published discrete number stays reproducible.
        if self.continuous:
            init = 0.0 if fix_sigma is None else float(np.log(fix_sigma))
            self.actor_logstd = nn.Parameter(
                torch.full((1, act_n), init, dtype=torch.float32),
                requires_grad=fix_sigma is None)
            self.fixed_sigma = fix_sigma is not None

    def features(self, x, h, done):
        """x: (T*N, obs). h: (1,N,H). done: (T*N,). Returns (T*N,H), h."""
        z = self.trunk(x)
        if not self.recurrent:
            return z, h
        n = h.shape[1]
        t = z.shape[0] // n
        z = z.reshape(t, n, self.hidden)
        d = done.reshape(t, n, 1)
        outs = []
        for step in range(t):
            # reset the hidden state wherever the previous step ended an episode
            h = (1.0 - d[step]).unsqueeze(0) * h
            o, h = self.gru(z[step].unsqueeze(0), h)
            outs.append(o)
        return torch.cat(outs).reshape(t * n, self.hidden), h

    def value(self, x, h, done):
        z, _ = self.features(x, h, done)
        return self.critic(z).flatten()

    def act(self, x, h, done, action=None, deterministic=False):
        z, h = self.features(x, h, done)
        if self.continuous:
            mean = (self.gain_scale * self.actor(x) if self.linear_policy
                    else self.actor(z))
            if deterministic:
                # Deployed evaluation uses the policy mean, no exploration
                # noise. That is what makes J_rate comparable to the analytic
                # baselines, which are all computed at s_e = 0.
                zero = torch.zeros(mean.shape[0], device=mean.device)
                return mean, zero, zero, self.critic(z).flatten(), h
            std = self.actor_logstd.expand_as(mean).exp()
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            return (action, dist.log_prob(action).sum(-1),
                    dist.entropy().sum(-1), self.critic(z).flatten(), h)
        logits = self.actor(z)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z).flatten(), h


# ---------------------------------------------------------------------------
# Evaluation and the architecture blind diagnostic
# ---------------------------------------------------------------------------


def _continuous_env_id(env_id):
    """Dispatch without constructing an environment, so the discrete path is
    provably untouched by the Phase 2 estimator.

    E13 (2026-08-21): the cubic-damping variants (GLEAM_*_nl<value>) must
    dispatch to the SAME continuing-trajectory estimator as their base
    instance. Before this line, an _nl id silently fell through to the
    episodic evaluator -- the estimator prereg_phase2.py's header documents
    as inflating cost with the reset transient, confounded with the gain.
    Caught by the pre-freeze smoke run (J_rate came back None); fixed by
    stripping the suffix for dispatch only. Discrete ids never contain
    "_nl", so their path is untouched."""
    if gleam_native is None:
        return False
    base = env_id.rpartition("_nl")[0] if "_nl" in env_id else env_id
    return base in gleam_native.REGISTRY


@torch.no_grad()
def evaluate_continuous(agent, env_id, stack, seed, gamma, device,
                        collect_gap, lam, eval_steps, eval_burnin):
    """Deployed evaluation for continuing control tasks. ONE long trajectory.

    Why this is not the episodic estimator used for the discrete environments:
    GLEBench resets from the UNCONTROLLED stationary covariance, where
    Var(v) = 1.0. Under control the stationary variance is far smaller, 0.059
    at k = 1.649 and 0.00069 at k = 20.55. Every reset therefore pays a
    settling transient, and averaging many short episodes folds that transient
    into the reported cost.

    The inflation is multiplicative, does not average out, and GROWS with the
    gain, which lambda controls. Measured against analytic J over 1024 step
    episodes it is 1.088x at k = 1.649, 1.270x at k = 5.99 and 2.415x at
    k = 20.55. That would confound the measurement with the effect under test,
    largest exactly where the predicted effect is largest.

    So: roll one trajectory, no resets, policy mean, discard a burn-in, and
    report the cost rate over the remainder. A 100 step burn-in already gives
    a ratio of 1.0000 at all three gains over a 200,000 step window; the
    default of 1000 is margin. test_eval_calibration.py pins this down against
    GLEBench.J_gain and is what licenses comparing J_rate to the baselines.
    """
    env = make_env(env_id)
    enc, base = obs_encoder(env.observation_space)
    a_low = float(env.action_space.low)
    a_high = float(env.action_space.high)
    h = getattr(env, "h", None)

    buf = deque(maxlen=stack)
    o, _ = env.reset(seed=seed + 10_000)
    for _ in range(stack - 1):
        buf.append(np.zeros(base, dtype=np.float32))
    buf.append(enc(o))
    hs = torch.zeros(1, 1, agent.hidden, device=device)
    done_flag = torch.zeros(1, device=device)

    obs_v, act_u, costs, ep_v, ep_r = [], [], [], [], []
    x = None
    for t in range(eval_burnin + eval_steps):
        x = torch.tensor(np.concatenate(list(buf)), dtype=torch.float32,
                         device=device).unsqueeze(0)
        a, _, _, v, hs = agent.act(x, hs, done_flag, deterministic=True)
        u = np.clip(a.detach().numpy().reshape(-1), a_low, a_high)
        v_obs = float(np.asarray(o, dtype=np.float64).reshape(-1)[0])
        o, r, te, tr, info = env.step(u)
        if t >= eval_burnin:
            obs_v.append(v_obs)
            act_u.append(float(u[0]))
            costs.append(float(info["cost"]))
            if collect_gap:
                ep_v.append(float(v.item()))
                ep_r.append(float(r))
        buf.append(enc(o))
        # No reset, ever. The task is continuing and GleamNative does not reset
        # itself, so ignoring the truncation flag keeps one unbroken
        # trajectory. Resetting here would reintroduce the exact transient this
        # estimator exists to remove.
        done_flag = torch.zeros(1, device=device)

    ctrl = {"final_gain": None, "J_rate": None, "excess_cost": None}
    v_arr = np.asarray(obs_v, dtype=np.float64)
    u_arr = np.asarray(act_u, dtype=np.float64)
    if v_arr.size and float(v_arr.var()) > 1e-12:
        slope = float(((v_arr - v_arr.mean()) * (u_arr - u_arr.mean())).mean()
                      / v_arr.var())
        ctrl["final_gain"] = -slope
    if h and costs:
        ctrl["J_rate"] = float(np.mean(costs)) / float(h)
        try:
            j_ml = env.baselines()["J_ml"]
            if j_ml:
                ctrl["excess_cost"] = ctrl["J_rate"] / float(j_ml)
        except Exception:
            pass

    # Higher is better, scale free, and directly interpretable for a
    # continuing task. Episode sums would just scale with eval_steps.
    mean_ret = -ctrl["J_rate"] if ctrl["J_rate"] is not None else float("nan")

    gap_v = gap_l = None
    if collect_gap and ep_r:
        v_final = float(agent.value(x, hs, done_flag).item())
        n = len(ep_r)
        g = [0.0] * n
        acc = v_final
        for i in range(n - 1, -1, -1):
            acc = ep_r[i] + gamma * acc
            g[i] = acc
        gs_arr, vs_arr = np.array(g), np.array(ep_v)
        sd = float(gs_arr.std())
        if sd > 1e-8:
            gap_v = float((vs_arr - gs_arr).mean() / sd)
            if lam is not None:
                gl = [0.0] * n
                acc_l = v_final
                for i in range(n - 1, -1, -1):
                    v_next = ep_v[i + 1] if i + 1 < n else v_final
                    acc_l = ep_r[i] + gamma * ((1.0 - lam) * v_next + lam * acc_l)
                    gl[i] = acc_l
                gap_l = float((np.array(gl) - gs_arr).mean() / sd)
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


@torch.no_grad()
def evaluate(agent, env_id, stack, episodes, seed, gamma, device,
             collect_gap=False, lam=None, eval_steps=200_000,
             eval_burnin=1000):
    """Run whole episodes. Returns mean return and, optionally, two diagnostics.

    Both diagnostics are architecture blind by construction, which is what
    makes them usable for gate G3. They see only rewards and the run's own
    value predictions.

      gap_value  = mean(V - G_mc) / std(G_mc)
      gap_lambda = mean(G_lambda - G_mc) / std(G_mc)

    G_mc is the realised discounted return to go. A truncated episode is NOT
    treated as terminal. Its accumulator is seeded with V of the final
    observation, otherwise the statistic just measures the time limit rather
    than aliasing. PositionOnlyCartPole truncates at 200 steps, so this
    correction is load bearing there and inert on RepeatPrevious, which only
    ever terminates.
    """
    # Continuing control tasks use the single long trajectory estimator. The
    # episodic code below is the pilot's, unchanged, and is now reached only
    # by discrete environments.
    if _continuous_env_id(env_id):
        return evaluate_continuous(agent, env_id, stack, seed, gamma, device,
                                   collect_gap, lam, eval_steps, eval_burnin)

    env = make_env(env_id)
    enc, base = obs_encoder(env.observation_space)
    # E14 (2026-09-11): episodic environments with a CONTINUOUS action
    # (PositionOnlyPendulum). Deployed evaluation uses the policy MEAN with
    # the declared clip, the same convention as evaluate_continuous, so the
    # reported return is the learned policy's and not the pinned exploration
    # noise's. The test is on the action space and consumes no RNG; a
    # discrete environment takes exactly the path it took before this line,
    # call for call, so every published discrete number is untouched
    # (harness/test_arch_regression.py).
    episodic_continuous = is_continuous(env.action_space)
    if episodic_continuous:
        a_low = float(env.action_space.low)
        a_high = float(env.action_space.high)
    buf = deque(maxlen=stack)
    rets, vs, gs, gls = [], [], [], []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + 10_000 + ep)
        buf.clear()
        for _ in range(stack - 1):
            buf.append(np.zeros(base, dtype=np.float32))
        buf.append(enc(o))
        h = torch.zeros(1, 1, agent.hidden, device=device)
        done_flag = torch.zeros(1, device=device)
        total, ep_v, ep_r = 0.0, [], []
        truncated_end = False
        while True:
            x = torch.tensor(np.concatenate(list(buf)), dtype=torch.float32,
                             device=device).unsqueeze(0)
            if episodic_continuous:
                a, _, _, v, h = agent.act(x, h, done_flag, deterministic=True)
                u = np.clip(a.detach().numpy().reshape(-1), a_low, a_high)
                o, r, te, tr, _ = env.step(u)
            else:
                a, _, _, v, h = agent.act(x, h, done_flag)
                o, r, te, tr, _ = env.step(int(a.item()))
            total += r
            if collect_gap:
                ep_v.append(float(v.item()))
                ep_r.append(float(r))
            buf.append(enc(o))
            done_flag = torch.zeros(1, device=device)
            if te or tr:
                truncated_end = bool(tr) and not bool(te)
                break
        rets.append(total)
        if collect_gap and ep_v:
            # value of the state the episode was cut off in, zero if it truly ended
            v_final = 0.0
            if truncated_end:
                xf = torch.tensor(np.concatenate(list(buf)), dtype=torch.float32,
                                  device=device).unsqueeze(0)
                v_final = float(agent.value(xf, h, done_flag).item())
            n = len(ep_r)
            g = [0.0] * n
            acc = v_final
            for i in range(n - 1, -1, -1):
                acc = ep_r[i] + gamma * acc
                g[i] = acc
            vs.extend(ep_v)
            gs.extend(g)
            if lam is not None:
                gl = [0.0] * n
                acc_l = v_final
                for i in range(n - 1, -1, -1):
                    v_next = ep_v[i + 1] if i + 1 < n else v_final
                    acc_l = ep_r[i] + gamma * ((1.0 - lam) * v_next + lam * acc_l)
                    gl[i] = acc_l
                gls.extend(gl)
    ctrl = {"final_gain": None, "J_rate": None, "excess_cost": None}
    mean_ret = float(np.mean(rets))
    gap_v = gap_l = None
    if collect_gap and gs:
        gs_arr, vs_arr = np.array(gs), np.array(vs)
        sd = float(gs_arr.std())
        if sd > 1e-8:
            gap_v = float((vs_arr - gs_arr).mean() / sd)
            if gls:
                gap_l = float((np.array(gls) - gs_arr).mean() / sd)
        else:
            gap_v, gap_l = 0.0, 0.0
    return mean_ret, gap_v, gap_l, ctrl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    # E13 (2026-08-21): --env validation moved from a static choices= list to
    # _valid_env, because the cubic-damping variants (GLEAM_*_nl<value>) are a
    # continuum that cannot be enumerated. _valid_env accepts exactly what
    # make_env can construct and nothing else; for every previously legal id
    # the behaviour is identical, and an illegal id still dies at the parser
    # with the known-env list. No RNG path is touched by this change.
    p.add_argument("--env", type=_valid_env, default="RepeatPreviousMedium")
    p.add_argument("--arch", type=str, default="gru128",
                   choices=["mlp", "stack8", "stack64", "gru128", "linear", "linearb"])
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=15_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--anneal-lr", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--final-eval-episodes", type=int, default=100)
    p.add_argument("--out-dir", type=str, default="results")
    # Default keeps every pilot invocation writing exactly where it wrote
    # before. Phase 2 passes results_phase2.jsonl so the two records never mix.
    p.add_argument("--results-name", type=str,
                   default="results_lambda_pilot.jsonl")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--torch-threads", type=int, default=1)
    # Continuous control only. Pins the Gaussian standard deviation and stops
    # it being learned. E4 showed entropy collapse acts as a brake that
    # confounds the lambda reading, so exploration has to be holdable fixed.
    p.add_argument("--fix-sigma", type=float, default=None)
    # --arch linear only. See MEMO-arch-linear.md section 7a.
    p.add_argument("--linear-gain-scale", type=float, default=1.0)
    p.add_argument("--linear-gain-init", type=float, default=0.0)
    p.add_argument("--linear-bias-init", type=float, default=0.0)
    # E17 (2026-10-05): vanilla SGD on the gain of a linear arm, Adam elsewhere.
    # 0.0 (default) leaves the published optimiser untouched.
    p.add_argument("--linear-sgd-lr", type=float, default=0.0)
    # E17 (2026-10-05): continuing training for GLEAM. 0 (default) keeps the
    # shipped 1024-step episodes; a value above the per-env step budget means
    # no reset ever happens and the trained objective is the continuing one.
    p.add_argument("--gleam-episode-len", type=int, default=0)
    # E17 (2026-10-05): record the quadratic fit of the critic along the deployed trajectory.
    p.add_argument("--log-critic-fit", type=int, default=0)
    # E17 (2026-10-05): replace the critic's values in the GAE by the exact LSTD(lambda)
    # aliased quadratic value at the current gain (linear arms on GLEAM only).
    p.add_argument("--oracle-critic", type=str, default="", choices=["", "lstd"])
    # Continuous evaluation only. One long trajectory, burn-in discarded, so
    # the reported cost is a stationary estimate rather than an average over
    # repeated settling transients. Discrete evaluation ignores both.
    p.add_argument("--eval-steps", type=int, default=200_000)
    p.add_argument("--eval-burnin", type=int, default=1000)
    return p.parse_args()


ARCH_STACK = {"mlp": 1, "stack8": 8, "stack64": 64, "gru128": 1, "linear": 1,
              "linearb": 1}


def main():
    args = parse_args()
    # See MEMO-arch-linear.md section 2. prereg_phase2.py's cell() selects on
    # (env, arm, lambda, seed) and IGNORES arch, so a linear row in the phase 2
    # file would be absorbed into E6's cells and corrupt every IQM that reads
    # them. Amendment 1's duplicate guard cannot see it: to that guard, mlp and
    # linear are two legitimately distinct cells.
    if args.arch.startswith("linear") and "phase2" in args.results_name:
        raise SystemExit(
            "REFUSING TO RUN. --arch linear must not write into "
            f"{args.results_name}. Pass --results-name results_tier1b.jsonl.")
    if args.gleam_episode_len:
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
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    stack = ARCH_STACK[args.arch]
    run = args.run_name or f"{args.env}_{args.arch}_lam{args.gae_lambda}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    curves_path = os.path.join(args.out_dir, f"curves_{run}.csv")
    results_path = os.path.join(args.out_dir, args.results_name)

    envs = SyncVec(args.env, args.num_envs, stack, args.seed)
    agent = Agent(envs.obs_dim, envs.act_n, args.arch,
                  continuous=envs.continuous,
                  fix_sigma=args.fix_sigma if envs.continuous else None,
                  gain_scale=args.linear_gain_scale,
                  gain_init=args.linear_gain_init,
                  bias_init=args.linear_bias_init).to(device)
    if args.linear_sgd_lr > 0:
        if not agent.linear_policy:
            raise SystemExit("--linear-sgd-lr is defined for --arch linear/linearb only.")
        _other = [q for n, q in agent.named_parameters() if n != "actor.weight"]
        opt = optim.Adam(_other, lr=args.lr, eps=1e-5)
        opt_k = optim.SGD([agent.actor.weight], lr=args.linear_sgd_lr)
        clip_params = _other
    else:
        opt = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
        opt_k = None
        clip_params = list(agent.parameters())

    batch = args.num_envs * args.num_steps
    minibatch_envs = max(1, args.num_envs // args.num_minibatches)
    updates = args.total_timesteps // batch

    obs = torch.zeros(args.num_steps, args.num_envs, envs.obs_dim)
    if envs.continuous:
        acts = torch.zeros(args.num_steps, args.num_envs, envs.act_n)
    else:
        acts = torch.zeros(args.num_steps, args.num_envs, dtype=torch.long)
    logps = torch.zeros(args.num_steps, args.num_envs)
    rews = torch.zeros(args.num_steps, args.num_envs)
    dones = torch.zeros(args.num_steps, args.num_envs)
    vals = torch.zeros(args.num_steps, args.num_envs)

    next_obs = torch.tensor(envs.reset(), dtype=torch.float32)
    next_done = torch.zeros(args.num_envs)
    h = torch.zeros(1, args.num_envs, agent.hidden)

    cf = open(curves_path, "w", newline="")
    cw = csv.writer(cf)
    cw.writerow(["update", "global_step", "train_return", "eval_return",
                 "value_loss", "policy_loss", "entropy", "approx_kl", "sps"])

    t0 = time.time()
    gstep = 0
    for update in range(1, updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / updates
            opt.param_groups[0]["lr"] = frac * args.lr
            if opt_k is not None:
                opt_k.param_groups[0]["lr"] = frac * args.linear_sgd_lr
        init_h = h.clone()

        for step in range(args.num_steps):
            gstep += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                a, lp, _, v, h = agent.act(next_obs, h, next_done)
                if oracle is not None:
                    v = oracle(agent, next_obs)
            acts[step], logps[step], vals[step] = a, lp, v
            nxt, r, term, trunc, final = envs.step(a.numpy())
            rews[step] = torch.tensor(r)
            d = np.maximum(term, trunc)
            # truncation bootstraps from the real final observation
            if trunc.any():
                with torch.no_grad():
                    fv = agent.value(torch.tensor(final, dtype=torch.float32),
                                     h, torch.zeros(args.num_envs))
                    if oracle is not None:
                        fv = oracle(agent, torch.tensor(final, dtype=torch.float32))
                rews[step] += torch.tensor(trunc) * args.gamma * fv
            next_obs = torch.tensor(nxt, dtype=torch.float32)
            next_done = torch.tensor(d, dtype=torch.float32)

        with torch.no_grad():
            next_val = agent.value(next_obs, h, next_done)
            if oracle is not None:
                next_val = oracle(agent, next_obs)
            adv = torch.zeros_like(rews)
            lastgae = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nonterm, nextvalues = 1.0 - next_done, next_val
                else:
                    nonterm, nextvalues = 1.0 - dones[t + 1], vals[t + 1]
                delta = rews[t] + args.gamma * nextvalues * nonterm - vals[t]
                lastgae = delta + args.gamma * args.gae_lambda * nonterm * lastgae
                adv[t] = lastgae
            rets = adv + vals

        env_idx = np.arange(args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_idx)
            for start in range(0, args.num_envs, minibatch_envs):
                mb = env_idx[start:start + minibatch_envs]
                mb_obs = obs[:, mb].reshape(-1, envs.obs_dim)
                mb_done = dones[:, mb].reshape(-1)
                mb_act = (acts[:, mb].reshape(-1, envs.act_n) if envs.continuous
                          else acts[:, mb].reshape(-1))
                mb_logp = logps[:, mb].reshape(-1)
                mb_adv = adv[:, mb].reshape(-1)
                mb_ret = rets[:, mb].reshape(-1)
                mb_val = vals[:, mb].reshape(-1)
                _, newlp, ent, newv, _ = agent.act(
                    mb_obs, init_h[:, mb].clone(), mb_done, mb_act)
                ratio = (newlp - mb_logp).exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (newlp - mb_logp)).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                nadv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg = torch.max(-nadv * ratio,
                               -nadv * torch.clamp(ratio, 1 - args.clip_coef,
                                                   1 + args.clip_coef)).mean()
                v_un = (newv - mb_ret) ** 2
                v_cl = (mb_val + torch.clamp(newv - mb_val, -args.clip_coef,
                                             args.clip_coef) - mb_ret) ** 2
                vloss = 0.5 * torch.max(v_un, v_cl).mean()
                loss = pg - args.ent_coef * ent.mean() + args.vf_coef * vloss
                opt.zero_grad()
                if opt_k is not None:
                    opt_k.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(clip_params, args.max_grad_norm)
                opt.step()
                if opt_k is not None:
                    opt_k.step()

        h = h.detach()
        train_ret = float(np.mean(envs.returns_log)) if envs.returns_log else float("nan")
        eval_ret = ""
        if update % args.eval_every == 0 or update == updates:
            # Mid-training curve point. On continuous environments a short
            # window keeps this at the same cost the episodic evaluator had
            # (20 episodes x 1024 steps), and the burn-in keeps it unbiased.
            eval_ret, _, _, _ = evaluate(agent, args.env, stack, args.eval_episodes,
                                         args.seed, args.gamma, device,
                                         eval_steps=20_000,
                                         eval_burnin=args.eval_burnin)
        sps = int(gstep / (time.time() - t0))
        cw.writerow([update, gstep, f"{train_ret:.5f}",
                     f"{eval_ret:.5f}" if eval_ret != "" else "",
                     f"{vloss.item():.5f}", f"{pg.item():.5f}",
                     f"{ent.mean().item():.5f}", f"{approx_kl.item():.5f}", sps])
        cf.flush()

    deployed, gap_v, gap_l, ctrl = evaluate(
        agent, args.env, stack, args.final_eval_episodes, args.seed,
        args.gamma, device, collect_gap=True, lam=args.gae_lambda,
        eval_steps=args.eval_steps, eval_burnin=args.eval_burnin)
    final_train = float(np.mean(envs.returns_log)) if envs.returns_log else float("nan")
    cf.close()

    row = {
        "run": run, "env": args.env, "arch": args.arch,
        "lam": args.gae_lambda, "seed": args.seed,
        "final_return": final_train, "deployed_return": deployed,
        "gap_value": gap_v, "gap_lambda": gap_l,
        "total_timesteps": args.total_timesteps,
        "gamma": args.gamma, "lr": args.lr, "num_envs": args.num_envs,
        "num_steps": args.num_steps, "stack": stack,
        "wallclock_s": round(time.time() - t0, 1),
    }
    if agent.linear_policy and agent.actor.bias is not None:
        row["final_bias"] = float(agent.actor.bias.item()) * agent.gain_scale
    if agent.linear_policy:
        # E17 (2026-10-05): the starting gain is part of the cell for a
        # drift-direction run, and the gain optimiser is part of the arm.
        # Linear arms only; no other row changes.
        row["linear_gain_init"] = float(args.linear_gain_init)
        row["linear_sgd_lr"] = float(getattr(args, "linear_sgd_lr", 0.0))
    if args.gleam_episode_len:
        row["gleam_episode_len"] = int(args.gleam_episode_len)
    if args.oracle_critic:
        row["oracle_critic"] = args.oracle_critic
    if args.log_critic_fit and ctrl.get("critic_fit") is not None:
        row["critic_fit"] = ctrl["critic_fit"]
        row["v_var"] = ctrl["v_var"]
    if envs.continuous:
        # Control diagnostics. None on discrete environments, so every pilot
        # row keeps exactly the schema it was written with.
        row["final_gain"] = ctrl["final_gain"]
        row["J_rate"] = ctrl["J_rate"]
        row["excess_cost"] = ctrl["excess_cost"]
        row["fix_sigma"] = args.fix_sigma
        try:
            row.update({f"ref_{k}": v for k, v in
                        gleam_native.baselines(args.env.replace("_dec", "")).items()})
            row["arm"] = "decoupled" if args.env.endswith("_dec") else "coupled"
        except Exception:
            pass
    with open(results_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
