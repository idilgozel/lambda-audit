"""GLEAM environment for the cluster. numpy only, no scipy.

Everything scipy was needed for is precomputed by build_gleam_instances.py and
shipped in gleam_instances.npz: the ZOH matrices F, G, Sd, the two cholesky
factors Lw and L0, and the exact analytic baselines. This module only steps
the recursion, which is four numpy operations.

The step expression is written to be bit for bit identical to GLEBench.step,
including the order of operations and the shape of every RNG draw, because
test_gleam_equivalence.py asserts bitwise equality rather than closeness. Do
not reorder the arithmetic here without re-running that test.

Two arms, per phase2-design.md section 4:

    coupled     s <- s F' + outer(u, G[:,0]) + noise      the real environment
    decoupled   s <- s F'                    + noise      one term dropped

The decoupled arm changes nothing else. The action still costs r u^2 and the
agent still observes only v. It is the preregistered replication of the
pilot's null on an environment where the closed loop can be switched off, so
the RepeatPrevious explanation stops being post hoc.

Conventions follow gleam_gym.py: reward is minus the per step cost, and the
1024 step episode boundary is a truncation, never a termination. The task is
continuing, so a reset draws a fresh stationary state and the RNG stream runs
on unbroken.
"""

import os

import numpy as np

from envs_native import Box

_NPZ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "gleam_instances.npz")
_CACHE = None

# Declared action bound. GLEBench does not clip and neither does this class,
# so the recursion stays bitwise comparable. The harness clips to these bounds
# before stepping, which is standard continuous PPO practice and keeps an
# unbounded Gaussian from walking the state to infinity.
ACT_LIMIT = 40.0


def _load():
    global _CACHE
    if _CACHE is None:
        with np.load(_NPZ_PATH, allow_pickle=False) as z:
            _CACHE = {k: z[k] for k in z.files}
    return _CACHE


def instance_names():
    return [str(n) for n in _load()["__names"]]


def baselines(instance):
    """J_opt, k_ml, J_ml, k_AC, J_AC, rThK0 for one instance."""
    b = _load()[f"{instance}__baselines"]
    return dict(J_opt=float(b[0]), k_ml=float(b[1]), J_ml=float(b[2]),
                k_AC=float(b[3]), J_AC=float(b[4]), rThK0=float(b[5]))


def instance_params(instance):
    s = _load()[f"{instance}__scalars"]
    return dict(h=float(s[0]), q=float(s[1]), r=float(s[2]), theta=float(s[3]),
                rho=float(s[4]), c=float(s[5]), gamma_bath=float(s[6]),
                nx=int(s[7]), episode_len=int(s[8]))


class GleamNative:
    """Single environment, gym-like, matching the envs_native API."""

    def __init__(self, instance, coupled=True, episode_len=None, seed=0,
                 nl=0.0):
        d = _load()
        if f"{instance}__F" not in d:
            raise KeyError(f"unknown GLEAM instance {instance}; "
                           f"known: {instance_names()}")
        self.instance = instance
        self.coupled = bool(coupled)
        self.F = d[f"{instance}__F"]
        self.G = d[f"{instance}__G"]
        self.Lw = d[f"{instance}__Lw"]
        self.L0 = d[f"{instance}__L0"]
        p = instance_params(instance)
        self.h, self.q, self.r = p["h"], p["q"], p["r"]
        self.nx = p["nx"]
        self.episode_len = p["episode_len"] if episode_len is None else int(episode_len)
        self.n_envs = 1                      # one env per instance, as envs_native does
        self.observation_space = Box(-np.inf, np.inf, (1,), np.float64)
        self.action_space = Box(-ACT_LIMIT, ACT_LIMIT, (1,), np.float64)
        self.rng = np.random.default_rng(seed)
        self.s = None
        self.t = 0
        # E13 (2026-08-21): optional cubic damping on the observed coordinate,
        # applied AFTER the linear ZOH update in FOUR substeps:
        #     for _ in range(4): v -= (h/4) * nl * v**3
        # The substepping is part of the frozen map, not an implementation
        # detail: the single-step version overflows at nl = 11.2 when |v|
        # exceeds sqrt(2/(h*nl)) ~ 1.9, which exploration reaches; substepping
        # doubles the threshold. Matches the pre-freeze pilot in
        # prereg_e13.py exactly. nl = 0.0 takes no branch, draws no RNG, and
        # changes no arithmetic, so every existing GLEAM row is untouched.
        self.nl = float(nl)

    # -- gym-like API -------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s = (self.L0 @ self.rng.standard_normal((self.nx, self.n_envs))).T
        self.t = 0
        return self.s[:, 0:1].copy()[0], {}

    def step(self, action):
        u = np.asarray(action, dtype=np.float64).reshape(-1)
        # cost uses the state BEFORE the update, exactly as GLEBench does
        cost = self.h * (self.q * self.s[:, 0] ** 2 + self.r * u ** 2)
        noise = (self.Lw @ self.rng.standard_normal((self.nx, self.n_envs))).T
        if self.coupled:
            self.s = self.s @ self.F.T + np.outer(u, self.G[:, 0]) + noise
        else:
            self.s = self.s @ self.F.T + noise
        if self.nl:
            for _ in range(4):
                self.s[:, 0] -= (self.h / 4) * self.nl * self.s[:, 0] ** 3
        self.t += 1
        trunc = self.t >= self.episode_len
        obs = self.s[:, 0:1].copy()[0]
        return obs, float(-cost[0]), False, bool(trunc), {"cost": float(cost[0])}

    # -- analytic reference -------------------------------------------------

    def baselines(self):
        return baselines(self.instance)

    def __repr__(self):
        arm = "coupled" if self.coupled else "decoupled"
        tail = f", nl={self.nl}" if self.nl else ""
        return f"GleamNative({self.instance}, {arm}{tail})"


def _entry(instance, coupled):
    def make():
        return GleamNative(instance, coupled=coupled)
    return make


def build_registry():
    """Every instance in both arms, keyed by the id used on the command line."""
    reg = {}
    for name in instance_names():
        reg[name] = _entry(name, True)
        reg[f"{name}_dec"] = _entry(name, False)
    return reg


REGISTRY = build_registry()


def make(env_id):
    # E13: an optional trailing _nl<value> selects the cubic-damping variant,
    # e.g. GLEAM_g1_h0.05_nl2.24. Parsed here rather than pre-registered
    # because nl is a continuum; the base id must still be a known instance.
    nl = 0.0
    base = env_id
    if "_nl" in env_id:
        base, _, tail = env_id.rpartition("_nl")
        try:
            nl = float(tail)
        except ValueError:
            raise KeyError(f"bad nl suffix in {env_id!r}: {tail!r}")
        if not (nl > 0):
            raise KeyError(f"nl must be positive in {env_id!r}; for the "
                           f"linear environment use the base id {base!r}")
    if base not in REGISTRY:
        raise KeyError(f"unknown GLEAM env {base}; known: {sorted(REGISTRY)}")
    env = REGISTRY[base]()
    if nl:
        env.nl = float(nl)
    return env


if __name__ == "__main__":
    for n in instance_names():
        p, b = instance_params(n), baselines(n)
        print(f"{n:16} gamma_bath={p['gamma_bath']:.1f} h={p['h']:.2f} "
              f"gamma*h={p['gamma_bath'] * p['h']:.3f}  "
              f"k_ml={b['k_ml']:.3f} k_AC={b['k_AC']:.3f} "
              f"J_ml={b['J_ml']:.4f} rThK0={b['rThK0']:.4f}")
