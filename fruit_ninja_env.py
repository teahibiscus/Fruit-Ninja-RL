"""
fruit_ninja_env.py - a lightweight Fruit Ninja simulator as a Gymnasium environment.

Design (matches the architecture we discussed):
  * Playfield is the unit square: x in [0, 1], y in [0, 1] with y UP (y = 0 is the bottom edge).
  * Fruits and bombs are thrown from the bottom in waves and follow ballistic arcs.
    Objects in a wave share a similar apex height, so a long horizontal swipe near the
    apex can hit several at once (that is what makes combos learnable).
  * Observation = the same state vector a detector + tracker would produce on the real game:
    up to N_SLOTS objects (sorted by urgency) x 8 features, plus swipe-cooldown fraction.
  * Action = MultiDiscrete([N_CX, N_CY, N_ANG, N_LEN]): a swipe segment CENTERED at a grid
    cell, at one of N_ANG angles, with one of N_LEN lengths. Length index 0 means "no swipe".
  * Latency: a swipe executes 1-3 steps (50-150 ms at the default dt) AFTER it is chosen,
    so the policy must lead moving targets using the velocity features.
  * Rewards live in RewardConfig, so you can sweep them without touching the physics.

Only numpy and gymnasium are required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --------------------------------------------------------------------------------------
# Action grid (edit these to change the action space)
# --------------------------------------------------------------------------------------
N_CX, N_CY, N_ANG, N_LEN = 12, 10, 8, 4
CX = (np.arange(N_CX) + 0.5) / N_CX                 # swipe-center x positions
CY = (np.arange(N_CY) + 0.5) / N_CY                 # swipe-center y positions
ANGLES = np.arange(N_ANG) * (math.pi / N_ANG)       # 0 .. 157.5 deg (0 = horizontal)
LENGTHS = np.array([0.0, 0.2, 0.4, 0.7])            # index 0 = no swipe
ANG_HORIZONTAL = 0
ANG_VERTICAL = N_ANG // 2                           # 90 deg when N_ANG = 8

N_FEAT = 8   # per-object features: present, is_bomb, x, y, vx, vy, pred_x, pred_y
V_SCALE = 2.0


@dataclass
class RewardConfig:
    fruit: float = 1.0              # per fruit sliced
    combo_bonus_k2: float = 1.0     # SHAPING bonus for a 2-fruit swipe (anneal to 0 later)
    combo_bonus_scale: float = 1.0  # bonus = scale * k for a swipe that slices k >= 3 fruits
    bomb: float = -10.0             # slicing a bomb
    miss: float = -0.5              # a fruit falls off the bottom unsliced
    swipe_cost: float = -0.02       # every swipe issued (stops swipe-spamming)


@dataclass
class SimConfig:
    dt: float = 0.05                # seconds per env step (20 decisions / second)
    max_seconds: float = 60.0       # episode length (truncation)
    gravity: float = 1.6
    radius: float = 0.06            # fruit and bomb radius
    n_slots: int = 8                # objects visible to the policy
    swipe_cooldown_steps: int = 3   # minimum spacing between swipes (3 steps = 0.15 s)
    latency_steps: tuple = (1, 3)   # inclusive range sampled per swipe; (0, 0) disables
    p_bomb: float = 0.15            # chance an object is a bomb
    wave_gap: tuple = (0.7, 1.5)    # seconds between waves
    wave_size_probs: tuple = (0.15, 0.25, 0.30, 0.20, 0.10)   # wave sizes 1..5
    stagger: float = 0.12           # objects in a wave launch within this many seconds
    lookahead: float = 0.10         # seconds ahead for the predicted-position features
    bomb_terminates: bool = True    # classic mode: a bomb ends the episode
    max_misses: Optional[int] = None  # e.g. 3 for "three strikes"; None = never ends the episode


def segment_from_action(cx_i: int, cy_i: int, ang_i: int, len_i: int):
    """Return the swipe segment endpoints (p0, p1) for a grid action."""
    c = np.array([CX[cx_i], CY[cy_i]])
    d = np.array([math.cos(ANGLES[ang_i]), math.sin(ANGLES[ang_i])])
    half = 0.5 * LENGTHS[len_i]
    return c - d * half, c + d * half


def seg_point_dist(p0: np.ndarray, p1: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance from each point in pts (n, 2) to the segment p0-p1."""
    d = p1 - p0
    denom = float(d @ d)
    if denom < 1e-12:
        t = np.zeros(len(pts))
    else:
        t = np.clip(((pts - p0) @ d) / denom, 0.0, 1.0)
    proj = p0 + t[:, None] * d
    return np.linalg.norm(pts - proj, axis=1)


class FruitNinjaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, sim: Optional[SimConfig] = None, reward: Optional[RewardConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.sim = sim or SimConfig()
        self.rw = reward or RewardConfig()
        self.render_mode = render_mode
        self.action_space = spaces.MultiDiscrete([N_CX, N_CY, N_ANG, N_LEN])
        self.observation_space = spaces.Box(
            -2.0, 2.0, shape=(self.sim.n_slots * N_FEAT + 1,), dtype=np.float32)
        self.max_steps = int(round(self.sim.max_seconds / self.sim.dt))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.is_bomb = np.zeros(0, dtype=bool)
        self.seen = np.zeros(0, dtype=bool)      # has the object ever been on screen?
        self.spawn_queue = []                    # (time_s, x0, vx, vy, is_bomb)
        self.pending = []                        # (due_step, p0, p1): swipes in flight
        self.cooldown = 0
        self.next_wave_t = float(self.np_random.uniform(0.2, 0.6))
        self.last_swipe = None                   # (p0, p1, step) for rendering
        self.stats = dict(score=0.0, fruits=0, combos=0, multi=0, bombs=0, misses=0,
                          swipes=0, steps=0, **{"return": 0.0})
        return self._obs(), {}

    def step(self, action):
        cfg, rw = self.sim, self.rw
        reward, terminated = 0.0, False
        info = {"sliced": 0, "bomb": False, "missed": 0}

        # 1) issue a swipe (ignored during cooldown or when the length index is 0)
        cx_i, cy_i, ang_i, len_i = (int(a) for a in action)
        if len_i > 0 and self.cooldown == 0:
            p0, p1 = segment_from_action(cx_i, cy_i, ang_i, len_i)
            lo, hi = cfg.latency_steps
            delay = int(self.np_random.integers(lo, hi + 1))
            self.pending.append((self.t + delay, p0, p1))
            self.cooldown = cfg.swipe_cooldown_steps
            reward += rw.swipe_cost
            self.stats["swipes"] += 1

        # 2) execute swipes that are due, against the world the agent just observed
        due = [s for s in self.pending if s[0] <= self.t]
        self.pending = [s for s in self.pending if s[0] > self.t]
        for _, p0, p1 in due:
            r, n_fruit, bomb = self._execute_swipe(p0, p1)
            reward += r
            info["sliced"] += n_fruit
            info["bomb"] = info["bomb"] or bomb
            terminated = terminated or (bomb and cfg.bomb_terminates)

        # 3) advance the world
        n_miss = self._advance_world()
        reward += rw.miss * n_miss
        info["missed"] = n_miss
        self.stats["misses"] += n_miss
        if cfg.max_misses is not None and self.stats["misses"] >= cfg.max_misses:
            terminated = True

        # 4) bookkeeping
        self.t += 1
        self.cooldown = max(0, self.cooldown - 1)
        self.stats["steps"] = self.t
        self.stats["return"] += reward
        truncated = self.t >= self.max_steps
        if terminated or truncated:
            info["episode_stats"] = dict(self.stats)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        S = 256
        img = np.full((S, S, 3), 20, dtype=np.uint8)
        yy, xx = np.mgrid[0:S, 0:S]
        X, Y = (xx + 0.5) / S, 1.0 - (yy + 0.5) / S
        for (x, y), bomb in zip(self.pos, self.is_bomb):
            if y < 0:
                continue
            mask = (X - x) ** 2 + (Y - y) ** 2 <= self.sim.radius ** 2
            img[mask] = (220, 50, 50) if bomb else (60, 200, 80)
        if self.last_swipe is not None and self.t - self.last_swipe[2] <= 2:
            p0, p1, _ = self.last_swipe
            pts = p0 + np.linspace(0, 1, 120)[:, None] * (p1 - p0)
            px = np.clip((pts[:, 0] * S).astype(int), 0, S - 1)
            py = np.clip(((1.0 - pts[:, 1]) * S).astype(int), 0, S - 1)
            img[py, px] = (255, 255, 255)
        return img

    # ------------------------------------------------------------------ rewards
    def _game_bonus(self, k: int) -> float:
        """Bonus the 'real' game score would give (used for the score metric)."""
        return self.rw.combo_bonus_scale * k if k >= 3 else 0.0

    def _combo_bonus(self, k: int) -> float:
        """Training bonus = game bonus + optional shaping for 2-fruit swipes."""
        return self._game_bonus(k) + (self.rw.combo_bonus_k2 if k == 2 else 0.0)

    # ------------------------------------------------------------------ internals
    def _execute_swipe(self, p0, p1):
        """Slice everything the segment touches. Returns (reward, n_fruit, bomb_hit)."""
        self.last_swipe = (p0, p1, self.t)
        visible = self.pos[:, 1] >= 0.0
        if len(self.pos) == 0 or not visible.any():
            return 0.0, 0, False
        hit = visible & (seg_point_dist(p0, p1, self.pos) <= self.sim.radius)
        n_bomb = int((hit & self.is_bomb).sum())
        n_fruit = int((hit & ~self.is_bomb).sum())
        r = n_fruit * self.rw.fruit
        if n_bomb > 0:
            r += self.rw.bomb                 # no combo bonus on a swipe that hits a bomb
            self.stats["bombs"] += 1
        else:
            r += self._combo_bonus(n_fruit)
        if n_fruit > 0:
            self.stats["fruits"] += n_fruit
            self.stats["score"] += n_fruit + (self._game_bonus(n_fruit) if n_bomb == 0 else 0.0)
            if n_fruit >= 2:
                self.stats["multi"] += 1
            if n_fruit >= 3 and n_bomb == 0:
                self.stats["combos"] += 1
        keep = ~hit
        self.pos, self.vel = self.pos[keep], self.vel[keep]
        self.is_bomb, self.seen = self.is_bomb[keep], self.seen[keep]
        return r, n_fruit, n_bomb > 0

    def _spawn_wave(self):
        cfg, rng = self.sim, self.np_random
        n = 1 + int(rng.choice(len(cfg.wave_size_probs), p=cfg.wave_size_probs))
        h_wave = rng.uniform(0.55, 0.90)       # shared apex height -> objects form a row
        cx = rng.uniform(0.2, 0.8)
        dx = rng.uniform(-0.3, 0.3)            # shared horizontal drift
        now = self.t * cfg.dt
        for _ in range(n):
            h = float(np.clip(h_wave + rng.uniform(-0.05, 0.05), 0.4, 0.95))
            vy = math.sqrt(2.0 * cfg.gravity * h)
            flight = 2.0 * vy / cfg.gravity
            x0 = float(np.clip(cx + rng.uniform(-0.18, 0.18), 0.05, 0.95))
            x1 = float(np.clip(x0 + dx + rng.uniform(-0.05, 0.05), 0.05, 0.95))
            vx = (x1 - x0) / flight
            launch = now + float(rng.uniform(0.0, cfg.stagger))
            bomb = bool(rng.random() < cfg.p_bomb)
            self.spawn_queue.append((launch, x0, vx, vy, bomb))

    def _advance_world(self) -> int:
        cfg = self.sim
        now = self.t * cfg.dt
        while now >= self.next_wave_t:
            self._spawn_wave()
            self.next_wave_t += float(self.np_random.uniform(*cfg.wave_gap))
        ready = [s for s in self.spawn_queue if s[0] <= now]
        if ready:
            self.spawn_queue = [s for s in self.spawn_queue if s[0] > now]
            new_pos = np.array([[s[1], -0.03] for s in ready])
            new_vel = np.array([[s[2], s[3]] for s in ready])
            self.pos = np.vstack([self.pos, new_pos])
            self.vel = np.vstack([self.vel, new_vel])
            self.is_bomb = np.concatenate([self.is_bomb, [s[4] for s in ready]])
            self.seen = np.concatenate([self.seen, np.zeros(len(ready), dtype=bool)])
        if len(self.pos) == 0:
            return 0
        g, dt = cfg.gravity, cfg.dt
        self.pos[:, 0] += self.vel[:, 0] * dt
        self.pos[:, 1] += self.vel[:, 1] * dt - 0.5 * g * dt * dt
        self.vel[:, 1] -= g * dt
        self.seen |= self.pos[:, 1] >= 0.0
        gone = (self.seen & (self.pos[:, 1] < 0.0)) | (self.pos[:, 0] < -0.15) | (self.pos[:, 0] > 1.15)
        n_miss = int((gone & self.seen & ~self.is_bomb).sum())   # a falling bomb is not a miss
        keep = ~gone
        self.pos, self.vel = self.pos[keep], self.vel[keep]
        self.is_bomb, self.seen = self.is_bomb[keep], self.seen[keep]
        return n_miss

    def _obs(self) -> np.ndarray:
        cfg = self.sim
        N, g, tau = cfg.n_slots, cfg.gravity, cfg.lookahead
        obs = np.zeros(N * N_FEAT + 1, dtype=np.float32)
        vis = np.flatnonzero(self.pos[:, 1] >= 0.0) if len(self.pos) else np.array([], dtype=int)
        if len(vis):
            y, vy = self.pos[vis, 1], self.vel[vis, 1]
            t_fall = (vy + np.sqrt(vy ** 2 + 2.0 * g * np.maximum(y, 0.0))) / g
            order = vis[np.argsort(t_fall)][:N]          # slot 0 = soonest to leave the screen
            for slot, i in enumerate(order):
                x, y = self.pos[i]
                vx, vy = self.vel[i]
                px = x + vx * tau
                py = y + vy * tau - 0.5 * g * tau ** 2
                obs[slot * N_FEAT:(slot + 1) * N_FEAT] = [
                    1.0, float(self.is_bomb[i]), x, y, vx / V_SCALE, vy / V_SCALE, px, py]
        obs[-1] = self.cooldown / max(1, cfg.swipe_cooldown_steps)
        return np.clip(obs, -2.0, 2.0)


# ======================================================================================
# Baseline (stands in for the prior-work heuristic) and an evaluation helper
# ======================================================================================
class GreedySingleFruitBaseline:
    """Most urgent fruit -> short vertical swipe through its predicted position,
    skipping targets whose swipe passes near a bomb. One fruit per swipe, no combo logic."""

    def __init__(self, sim: Optional[SimConfig] = None, bomb_margin: float = 0.10):
        self.sim = sim or SimConfig()
        self.margin = bomb_margin

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        noop = np.array([0, 0, 0, 0])
        if obs[-1] > 0:
            return noop
        slots = obs[:-1].reshape(self.sim.n_slots, N_FEAT)
        bombs = np.array([s[6:8] for s in slots if s[0] > 0.5 and s[1] > 0.5]).reshape(-1, 2)
        for s in slots:                                   # already sorted by urgency
            if s[0] < 0.5 or s[1] > 0.5:
                continue
            px, py = float(s[6]), float(s[7])
            if not (0.02 < py < 0.98):
                continue
            a = np.array([np.abs(CX - px).argmin(), np.abs(CY - py).argmin(), ANG_VERTICAL, 1])
            p0, p1 = segment_from_action(*a)
            if len(bombs) and (seg_point_dist(p0, p1, bombs) < self.sim.radius + self.margin).any():
                continue
            return a
        return noop


def evaluate(env: FruitNinjaEnv, policy, n_episodes: int = 20, seed: int = 0) -> dict:
    """Run `policy(obs) -> action` and return mean episode stats (score, combos, bombs...)."""
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        rows.append(info["episode_stats"])
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


if __name__ == "__main__":
    env = FruitNinjaEnv()
    rand = lambda obs: env.action_space.sample()
    greedy = GreedySingleFruitBaseline(env.sim)
    for name, pol in [("random", rand), ("greedy single-fruit", greedy)]:
        stats = evaluate(env, pol, n_episodes=20)
        print(f"{name:22s}", {k: round(v, 2) for k, v in stats.items()})
