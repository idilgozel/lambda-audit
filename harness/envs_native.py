"""Dependency-free reimplementations of the two POPGym environments used by the
lambda audit.

WHY THIS FILE EXISTS
--------------------
The audit runs on UCL Myriad, whose glibc is too old for current numpy wheels.
Installing popgym there forces pip to build numpy from source under the Intel
compiler, which fails. Rather than fight the toolchain, the two environments we
need are reimplemented here against numpy alone. The cluster then needs nothing
beyond the torch module it already provides.

CORRECTNESS
-----------
These are not "close enough" reimplementations. test_env_equivalence.py runs
both these and the real popgym environments through identical seeds and
identical action sequences and asserts bitwise-identical observations, rewards,
termination and truncation flags. popgym remains the reference. This file is a
verified drop-in, and the equivalence test is part of the audit's
reproducibility package.

The RNG is the subtle part. Gymnasium seeds with
    np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
and, crucially, does NOT reseed when reset() is called without a seed, so the
stream continues across episodes. Both behaviours are reproduced exactly.
"""

import math

import numpy as np

__all__ = [
    "RepeatPrevious", "RepeatPreviousEasy", "RepeatPreviousMedium",
    "RepeatPreviousHard", "PositionOnlyCartPole", "PositionOnlyCartPoleEasy",
    "PositionOnlyCartPoleMedium", "PositionOnlyCartPoleHard",
    "Battleship", "BattleshipEasy",
    "MineSweeper", "MineSweeperEasy", "MineSweeperMedium",
    "Discrete", "Box", "make",
]

DECK_SIZE = 52
N_SUITS = 4
N_RANKS = 13


def _make_rng(seed):
    """Exactly gymnasium.utils.seeding.np_random(seed)[0]."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


# --- minimal space objects, only what the harness touches ------------------

class Discrete:
    def __init__(self, n):
        self.n = int(n)
        self._rng = np.random.default_rng()

    def sample(self):
        return int(self._rng.integers(self.n))

    def __repr__(self):
        return f"Discrete({self.n})"


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"Box({self.shape}, {np.dtype(self.dtype).name})"


# --- RepeatPrevious --------------------------------------------------------

class RepeatPrevious:
    """Output the suit of the card dealt k steps ago.

    Mirrors popgym.envs.repeat_previous.RepeatPrevious, including the deck
    ordering, the deal-from-the-top convention and the reward scaling.
    """

    def __init__(self, num_decks=1, k=4):
        self.num_decks = int(num_decks)
        self.k = int(k)
        self.num_cards = DECK_SIZE * self.num_decks
        assert self.num_cards > self.k, "k cannot exceed the deck size"
        self.max_episode_length = self.num_cards - 1
        # popgym: np.tile(np.tile(np.arange(4), 13), num_decks)
        self.suits_idx = np.tile(np.tile(np.arange(N_SUITS), N_RANKS), self.num_decks)
        self.action_space = Discrete(N_SUITS)
        self.observation_space = Discrete(N_SUITS)
        self.rng = _make_rng(None)
        self.idx = np.arange(self.num_cards)
        self.deck_len = self.num_cards
        self.hand = []

    def _deal(self, n=1):
        new_len = self.deck_len - n
        if new_len < 0:
            raise RuntimeError("deck empty")
        self.hand += self.idx[new_len:self.deck_len].tolist()
        self.deck_len = new_len

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = _make_rng(seed)
        self.hand = []
        self.deck_len = self.num_cards
        self.idx = np.arange(self.num_cards)
        self.rng.shuffle(self.idx)
        self._deal(1)
        card = self.suits_idx[self.hand[-1]]
        return int(card), {}

    def step(self, action):
        reward_scale = 1.0 / (self.num_cards - self.k)
        reward = 0.0
        terminated = self.deck_len == 1
        if len(self.hand) >= self.k:
            if int(action) == int(self.suits_idx[self.hand[-self.k]]):
                reward = reward_scale
            else:
                reward = -reward_scale
        self._deal(1)
        card = self.suits_idx[self.hand[-1]]
        return int(card), reward, terminated, False, {}


class RepeatPreviousEasy(RepeatPrevious):
    def __init__(self):
        super().__init__(num_decks=1, k=4)


class RepeatPreviousMedium(RepeatPrevious):
    def __init__(self):
        super().__init__(num_decks=2, k=32)


class RepeatPreviousHard(RepeatPrevious):
    def __init__(self):
        super().__init__(num_decks=3, k=64)


# --- PositionOnlyCartPole --------------------------------------------------

class PositionOnlyCartPole:
    """CartPole with both velocities removed from the observation.

    Mirrors popgym.envs.position_only_cartpole.PositionOnlyCartPole, which
    subclasses gymnasium's CartPoleEnv. Physics constants, the euler
    integrator, the termination test and the reward transform are copied
    exactly.
    """

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4

    def __init__(self, max_episode_length=200):
        self.max_episode_length = int(max_episode_length)
        self.action_space = Discrete(2)
        high = np.array([self.x_threshold * 2, self.theta_threshold_radians * 2],
                        dtype=np.float32)
        self.observation_space = Box(-high, high, (2,), np.float32)
        self.rng = _make_rng(None)
        self.state = None
        self.num_steps = 0
        self.steps_beyond_terminated = None

    def _reward_transform(self, reward):
        if math.isclose(reward, 0):
            return -1.0
        return 1.0 / self.max_episode_length

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = _make_rng(seed)
        self.num_steps = 0
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_terminated = None
        full = np.array(self.state, dtype=np.float32)
        return np.array([full[0], full[2]]), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * np.square(theta_dot) * sintheta) \
            / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # euler
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = 0.0

        self.num_steps += 1
        truncated = False
        if self.num_steps >= self.max_episode_length:
            truncated = True
        reward = self._reward_transform(reward)
        full = np.array(self.state, dtype=np.float32)
        return np.array([full[0], full[2]]), reward, terminated, truncated, {}


class PositionOnlyCartPoleEasy(PositionOnlyCartPole):
    def __init__(self):
        super().__init__(max_episode_length=200)


class PositionOnlyCartPoleMedium(PositionOnlyCartPole):
    def __init__(self):
        super().__init__(max_episode_length=400)


class PositionOnlyCartPoleHard(PositionOnlyCartPole):
    def __init__(self):
        super().__init__(max_episode_length=600)


# --- Battleship ------------------------------------------------------------

class Battleship:
    """Battleship with the board obscured: the agent sees only hit/miss.

    Mirrors popgym.envs.battleship.Battleship. The board layout, the
    rejection-sampled ship placement (draw for draw), the guess bookkeeping,
    the reward scaling and the termination rules are copied exactly;
    test_env_equivalence.py asserts bitwise identity against popgym including
    the hidden board after seeded resets.

    Two deliberate interface adaptations, both OURS and both documented:

    1. popgym's action space is MultiDiscrete([n, n]); ours is the flat
       Discrete(n*n) with action = row * n + col, because the harness's
       policy head is a single Categorical. The mapping is a bijection and
       the equivalence test drives both envs through matched pairs.
    2. popgym sets obs_requires_prev_action = True and its baselines append
       the previous action via a wrapper. We fold that wrapper in: the
       observation is a float vector [hit, onehot_n(prev_row),
       onehot_n(prev_col)], zeros for the prev-action block at reset. The
       hit component is popgym's entire observation; the appended block is
       exactly what their PreviousAction wrapper supplies. Without it the
       task is not the POMDP the benchmark intends (the agent could not even
       in principle avoid refiring).
    """

    def __init__(self, board_size=10, ship_sizes=(2, 3, 3, 4)):
        self.board_size = int(board_size)
        self.ship_sizes = list(ship_sizes)
        self.max_episode_length = self.board_size ** 2
        self.needed_hits = sum(self.ship_sizes)
        n = self.board_size
        self.obs_dim = 1 + 2 * n
        self.action_space = Discrete(n * n)
        self.observation_space = Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self.rng = _make_rng(None)
        self.board = None
        self.guesses = None
        self.hits = 0
        self.num_steps = 0

    def _obs(self, hit, row=None, col=None):
        v = np.zeros(self.obs_dim, dtype=np.float32)
        v[0] = float(hit)
        if row is not None:
            v[1 + row] = 1.0
            v[1 + self.board_size + col] = 1.0
        return v

    def _place_ship(self, board, ship_size):
        # popgym's loop, draw for draw: idx_start then direction each
        # attempt, bounds rejection BEFORE overlap rejection, both consuming
        # the same draws as the reference on every path.
        while True:
            idx_start = self.rng.integers(self.board_size, size=2)
            direction = self.rng.integers(2, size=2)
            idx_end = idx_start.copy()
            idx_end[direction[0]] = idx_start[direction[0]] + (-1) ** direction[1] * (
                ship_size - 1
            )
            if np.any(idx_end < 0) or np.any(idx_end >= self.board_size):
                continue
            start = np.minimum(idx_start, idx_end)
            end = np.maximum(idx_start, idx_end)
            if np.sum(board[start[0]:end[0] + 1, start[1]:end[1] + 1]) == 0:
                board[start[0]:end[0] + 1, start[1]:end[1] + 1] = 1
                return board

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = _make_rng(seed)
        self.num_steps = 0
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        for ship_size in self.ship_sizes:
            self.board = self._place_ship(self.board, ship_size)
        self.guesses = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.hits = 0
        return self._obs(0), {}

    def step(self, action):
        a = int(action)
        row, col = divmod(a, self.board_size)
        is_ship = self.board[row, col]
        guessed_before = self.guesses[row, col]
        self.guesses[row, col] = 1
        hit = bool(is_ship and not guessed_before)
        if hit:
            self.hits += 1
        self.num_steps += 1
        truncated = self.num_steps >= self.max_episode_length
        terminated = self.hits == self.needed_hits
        reward = (int(hit) * (1.0 / self.needed_hits)) + (
            int(not hit) * (-1.0 / (self.max_episode_length - self.needed_hits))
        )
        return self._obs(int(hit), row, col), reward, terminated, truncated, {}


class BattleshipEasy(Battleship):
    def __init__(self):
        super().__init__(board_size=8, ship_sizes=(2, 3, 3, 4))


# --- MineSweeper -----------------------------------------------------------

class MineSweeper:
    """MineSweeper with the board obscured: the agent sees only the adjacent
    mine count of the tile it clicked.

    Mirrors popgym.envs.minesweeper.MineSweeper. Mine placement (a single
    replace=False choice), the neighbour-count convolution, the three reward
    branches, the win test and the truncation rule are copied exactly;
    test_minesweeper_equivalence.py asserts bitwise identity against popgym
    including the hidden board after seeded resets.

    Two interface adaptations, ours, both documented:

    1. popgym's action space is MultiDiscrete([r, c]); ours is the flat
       Discrete(r*c) with action = row * c + col, because the harness's policy
       head is a single Categorical. Bijection; the equivalence test drives
       both envs through matched pairs.
    2. popgym sets obs_requires_prev_action = True and their baselines append
       the previous action via a wrapper. We fold that in: the observation is
       [onehot_k(count), onehot_r(prev_row), onehot_c(prev_col)], zeros for the
       prev-action block at reset. Without it the agent cannot know WHICH tile
       produced the count, and the task is not the POMDP the benchmark intends.
    """

    CLEAR, MINE, VIEWED = 0, 1, 2

    def __init__(self, dims=(4, 4), num_mines=2):
        self.dims = tuple(dims)
        self.num_mines = int(num_mines)
        self.max_episode_length = self.dims[0] * self.dims[1] - self.num_mines
        self.success_reward_scale = 1 / self.max_episode_length
        self.fail_reward_scale = -0.5 - self.success_reward_scale
        self.bad_action_reward_scale = -0.5 / (self.max_episode_length - 2)
        self.n_counts = min(self.num_mines + 1, 10)
        self.obs_dim = self.n_counts + self.dims[0] + self.dims[1]
        self.action_space = Discrete(self.dims[0] * self.dims[1])
        self.observation_space = Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self.rng = _make_rng(None)
        self.hidden_grid = None
        self.neighbor_grid = None
        self.timestep = 0

    def _obs(self, count, row=None, col=None):
        v = np.zeros(self.obs_dim, dtype=np.float32)
        v[min(int(count), self.n_counts - 1)] = 1.0
        if row is not None:
            v[self.n_counts + row] = 1.0
            v[self.n_counts + self.dims[0] + col] = 1.0
        return v

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = _make_rng(seed)
        self.hidden_grid = np.full(self.dims, self.CLEAR, dtype=np.int8)
        mines_flat = self.rng.choice(
            np.arange(self.dims[0] * self.dims[1]), size=self.num_mines,
            replace=False)
        self.hidden_grid.ravel()[mines_flat] = self.MINE
        self.neighbor_grid = np.zeros_like(self.hidden_grid)
        src = np.pad(self.hidden_grid, [(1, 1), (1, 1)], constant_values=0)
        for si in (-1, 0, 1):
            for sj in (-1, 0, 1):
                self.neighbor_grid += np.roll(np.roll(src, si, 0), sj, 1)[1:-1, 1:-1]
        self.timestep = 0
        return self._obs(0), {}

    def step(self, action):
        a = int(action)
        row, col = divmod(a, self.dims[1])
        idx = (row, col)
        terminated = truncated = False
        if self.hidden_grid[idx] == self.MINE:
            terminated = True
            reward = self.fail_reward_scale
        elif self.hidden_grid[idx] == self.VIEWED:
            reward = self.bad_action_reward_scale
        else:
            self.hidden_grid[idx] = self.VIEWED
            reward = self.success_reward_scale
        self.timestep += 1
        terminated |= bool(np.all(self.hidden_grid != self.CLEAR).item())
        truncated = (self.timestep == self.max_episode_length) and not terminated
        count = self.neighbor_grid[idx].item()
        return self._obs(count, row, col), reward, terminated, truncated, {}


class MineSweeperEasy(MineSweeper):
    def __init__(self):
        super().__init__(dims=(4, 4), num_mines=2)


class MineSweeperMedium(MineSweeper):
    def __init__(self):
        super().__init__(dims=(6, 6), num_mines=6)


# --- registry --------------------------------------------------------------

REGISTRY = {
    "RepeatPreviousEasy": RepeatPreviousEasy,
    "RepeatPreviousMedium": RepeatPreviousMedium,
    "RepeatPreviousHard": RepeatPreviousHard,
    "RepeatPreviousK16": lambda: RepeatPrevious(num_decks=1, k=16),
    "PositionOnlyCartPoleEasy": PositionOnlyCartPoleEasy,
    "PositionOnlyCartPoleMedium": PositionOnlyCartPoleMedium,
    "PositionOnlyCartPoleHard": PositionOnlyCartPoleHard,
    "BattleshipEasy": BattleshipEasy,
    "MineSweeperEasy": MineSweeperEasy,
    "MineSweeperMedium": MineSweeperMedium,
}


def make(env_id):
    if env_id not in REGISTRY:
        raise KeyError(f"unknown env {env_id}; known: {sorted(REGISTRY)}")
    return REGISTRY[env_id]()
