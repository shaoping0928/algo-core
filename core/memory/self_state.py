"""Task-agnostic behavior-into-state (self_state) tracker and env wrapper.

Encodes the agent's own recent action and feedback consequences
``[last_action_norm, last_dopamine, last_pain, ema_success_rate]``
directly into the observation vector so downstream world models, feature rules,
and planners can condition on the agent's own behavioral state.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


class SelfStateTracker:
    """Fixed-width 4D tracker of recent action and feedback consequences."""

    DIM = 4

    def __init__(self, n_actions: int = 2, ema_alpha: float = 0.25):
        self.n_actions = max(int(n_actions), 2)
        self.ema_alpha = float(ema_alpha)
        self.reset()

    def reset(self) -> None:
        self._last_action = 0.0
        self._last_dopamine = 0.0
        self._last_pain = 0.0
        self._ema_success = 0.5

    def update(self, action: int, dopamine: float, pain: float) -> None:
        denom = float(max(self.n_actions - 1, 1))
        self._last_action = float(np.clip(float(action) / denom, 0.0, 1.0))
        self._last_dopamine = float(np.clip(float(dopamine), 0.0, 10.0))
        self._last_pain = float(np.clip(float(pain), 0.0, 10.0))
        outcome = 1.0 if float(dopamine) > float(pain) else 0.0
        a = self.ema_alpha
        self._ema_success = (1.0 - a) * self._ema_success + a * outcome

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self._last_action,
                self._last_dopamine,
                self._last_pain,
                self._ema_success,
            ],
            dtype=np.float32,
        )

    def augment_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([arr, self.vector()]).astype(np.float32)


class SelfStateEnvWrapper:
    """Wrap any ``env_factory`` environment to append 4D ``self_state`` to ``obs()``."""

    def __init__(self, env: Any, ema_alpha: float = 0.25):
        self.env = env
        n_act = len(getattr(env, "action_space", (0, 1)))
        self._tracker = SelfStateTracker(n_actions=n_act, ema_alpha=ema_alpha)

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self):
        self._tracker.reset()
        base = self.env.reset()
        return self._tracker.augment_obs(base)

    def obs(self) -> np.ndarray:
        return self._tracker.augment_obs(self.env.obs())

    def step(self, action):
        res = self.env.step(action)
        fb = self.env.feedback()
        self._tracker.update(
            int(action),
            dopamine=float(fb.get("dopamine", 0.0)),
            pain=float(fb.get("pain", 0.0)),
        )
        if isinstance(res, tuple) and len(res) == 3:
            _obs, dop, dead = res
            return self.obs(), dop, dead
        return res


def wrap_env_factory_with_self_state(
    env_factory: Callable, ema_alpha: float = 0.25,
) -> Callable:
    """Wrap an ``env_factory`` so every spawned env exposes ``self_state`` channels."""
    def factory(seed, spacing=None, gravity=None, flap=None, speed=None):
        env = env_factory(
            seed, spacing=spacing, gravity=gravity, flap=flap, speed=speed,
        )
        return SelfStateEnvWrapper(env, ema_alpha=ema_alpha)

    return factory
