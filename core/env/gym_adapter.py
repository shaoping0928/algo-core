"""Generic Gymnasium adapter matching the ``env_factory`` protocol.

Wraps any ``gymnasium.Env`` with a discrete action space into the task-agnostic
interface expected by ``core/runner/trainer.py`` and ``core/runner/evaluator.py``:
  - ``obs()``, ``step(action)``, ``reset()``, ``pending()``, ``feedback()``
  - ``dead``, ``passed``, ``t``, ``max_t``, ``action_space``
  - Configurable action delay ``lag`` and optional monotonic ``alive_clock`` channel.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional, Tuple

import numpy as np


class GymnasiumLagWrapper:
    """Wrap a discrete-action Gymnasium environment with action delay ``lag``."""

    def __init__(
        self,
        env: Any,
        seed: int = 0,
        lag: int = 2,
        cap: int = 1200,
        include_alive_clock: bool = True,
        goal_reward_threshold: float = 0.0,
    ):
        self.env = env
        self.seed = int(seed)
        self.lag = max(int(lag), 0)
        self.max_t = int(cap)
        self.include_alive_clock = bool(include_alive_clock)
        self.goal_reward_threshold = float(goal_reward_threshold)
        n_actions = int(getattr(getattr(env, "action_space", None), "n", 2))
        self.action_space: Tuple[int, ...] = tuple(range(n_actions))
        self.reset()

    def reset(self):
        out = self.env.reset(seed=int(self.seed) & 0xFFFFFFFF)
        obs = out[0] if isinstance(out, tuple) else out
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._q = deque([0] * (self.lag + 1), maxlen=self.lag + 1)
        self.t = 0
        self.dead = False
        self.passed = 0
        self._alive = 0.0
        self._last_reward = 0.0
        self._terminated = False
        self._truncated = False
        return self.obs()

    def pending(self):
        return list(self._q)[1:]

    def feedback(self) -> dict:
        if self._terminated and not self._truncated and self._last_reward <= self.goal_reward_threshold:
            return dict(dopamine=0.0, pain=1.0)
        return dict(
            dopamine=float(max(self._last_reward, 0.0)),
            pain=float(max(-self._last_reward, 0.0)) if self._last_reward < 0.0 else 0.0,
        )

    def obs(self) -> np.ndarray:
        if not self.include_alive_clock:
            return self._last_obs
        return np.concatenate(
            [self._last_obs, np.asarray([self._alive], dtype=np.float32)],
        ).astype(np.float32)

    def step(self, action):
        self._q.append(int(action))
        eff_action = int(self._q[0])
        res = self.env.step(eff_action)
        if len(res) == 5:
            next_obs, reward, terminated, truncated, info = res
        else:
            next_obs, reward, done, info = res
            terminated = bool(done)
            truncated = False
        self._last_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self._last_reward = float(reward)
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self.t += 1
        self._alive += 1.0
        if isinstance(info, dict) and info.get("passed"):
            self.passed += int(info["passed"])
        elif self.goal_reward_threshold > 0.0 and self._last_reward >= self.goal_reward_threshold:
            self.passed += 1
        self.dead = bool(self._terminated or self._truncated or self.t >= self.max_t)
        dopamine = 0.0 if (self._terminated and not self._truncated and self._last_reward <= self.goal_reward_threshold) else float(max(self._last_reward, 0.0))
        return self.obs(), dopamine, self.dead


def make_gym_factory(
    env_id: str,
    lag: int = 2,
    cap: int = 1200,
    include_alive_clock: bool = True,
    goal_reward_threshold: float = 0.0,
    **gym_kwargs,
) -> Callable:
    """Return an ``env_factory`` wrapping ``gymnasium.make(env_id, **gym_kwargs)``."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            f"gymnasium is required for --env gym:{env_id}; install it with "
            "'pip install gymnasium' or 'uv pip install gymnasium'"
        ) from exc

    def factory(seed, spacing=None, gravity=None, flap=None, speed=None):
        del spacing, gravity, flap, speed
        env = gym.make(env_id, **gym_kwargs)
        return GymnasiumLagWrapper(
            env,
            seed=int(seed),
            lag=lag,
            cap=cap,
            include_alive_clock=include_alive_clock,
            goal_reward_threshold=goal_reward_threshold,
        )

    return factory
