"""Self-contained Delayed MountainCar environment (3 actions, sparse goal).

Exercises two capabilities not tested by Flappy or CartPole:
  - 3 discrete actions (``0=push_left``, ``1=coast``, ``2=push_right``)
  - Sparse goal termination (``position >= 0.5``) requiring momentum
    accumulation rather than myopic step-by-step survival.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np


class DelayedMountainCar:
    """Classic MountainCar-v0 dynamics with configurable action-execution lag."""

    action_space = (0, 1, 2)

    def __init__(
        self,
        seed: int = 0,
        lag: int = 2,
        cap: int = 200,
        force: float = 0.001,
        gravity: float = 0.0025,
    ):
        self.seed = int(seed)
        self.lag = max(int(lag), 0)
        self.max_t = int(cap)
        self.force = float(force)
        self.gravity = float(gravity)
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.reset()

    def reset(self):
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        self.position = float(rng.uniform(-0.6, -0.4))
        self.velocity = 0.0
        self._q = deque([1] * (self.lag + 1), maxlen=self.lag + 1)
        self.t = 0
        self.dead = False
        self.passed = 0
        self._goal_reached = False
        self._alive = 0.0
        return self.obs()

    def pending(self):
        return list(self._q)[1:]

    def feedback(self) -> dict:
        if self._goal_reached:
            return dict(dopamine=10.0, pain=0.0)
        if self.dead:
            return dict(dopamine=0.0, pain=1.0)
        return dict(dopamine=0.0, pain=0.01)

    def obs(self) -> np.ndarray:
        return np.asarray(
            [self.position, self.velocity, self._alive], dtype=np.float32,
        )

    def step(self, action):
        self._q.append(int(action))
        eff_action = int(self._q[0])
        self.velocity += (eff_action - 1) * self.force - math.cos(3.0 * self.position) * self.gravity
        self.velocity = float(np.clip(self.velocity, -self.max_speed, self.max_speed))
        self.position += self.velocity
        self.position = float(np.clip(self.position, self.min_position, self.max_position))
        if self.position == self.min_position and self.velocity < 0.0:
            self.velocity = 0.0
        self.t += 1
        self._alive += 1.0
        if self.position >= self.goal_position:
            self._goal_reached = True
            self.passed += 1
            self.dead = True
            return self.obs(), 10.0, True
        if self.t >= self.max_t:
            self.dead = True
            return self.obs(), 0.0, True
        return self.obs(), 0.0, False


def make_mountaincar_factory(lag: int = 2, cap: int = 200):
    """Return an ``env_factory`` compatible with ``trainer.fit`` / ``evaluator.evaluate``."""
    def factory(seed, spacing=None, gravity=None, flap=None, speed=None):
        del spacing, flap, speed
        kwargs = dict(seed=int(seed), lag=lag, cap=cap)
        if gravity is not None:
            kwargs["gravity"] = float(gravity)
        return DelayedMountainCar(**kwargs)

    return factory
