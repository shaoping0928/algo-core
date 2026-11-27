"""Self-contained Delayed CartPole environment for cross-task generalization testing.

Structurally distinct from Flappy:
  - Nonlinear coupled cart-pole ODE (trigonometric instability, symmetric ±F control).
  - 5D observation: ``[x, x_dot, theta, theta_dot, alive_clock]`` (includes a pure
    monotonic drift channel to exercise ``detect_pure_drift_dims`` on a new task).
  - Configurable action delay ``lag`` (action issued at ``t`` physically acts at
    ``t + lag``), matching the exact ``env_factory`` protocol of ``substrate_nodes.py``.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np


class DelayedCartPole:
    """Classic CartPole-v1 dynamics with configurable action-execution lag."""

    def __init__(
        self,
        seed: int = 0,
        lag: int = 2,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        half_length: float = 0.5,
        force_mag: float = 10.0,
        dt: float = 0.02,
        x_threshold: float = 2.4,
        theta_threshold: float = 12.0 * 2.0 * math.pi / 360.0,
        cap: int = 1200,
    ):
        self.seed = int(seed)
        self.lag = max(int(lag), 0)
        self.gravity = float(gravity)
        self.masscart = float(masscart)
        self.masspole = float(masspole)
        self.total_mass = self.masscart + self.masspole
        self.half_length = float(half_length)
        self.polemass_length = self.masspole * self.half_length
        self.force_mag = float(force_mag)
        self.dt = float(dt)
        self.x_threshold = float(x_threshold)
        self.theta_threshold = float(theta_threshold)
        self.max_t = int(cap)
        self.reset()

    def reset(self):
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        self.state = rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(float)
        self._q = deque([0] * (self.lag + 1), maxlen=self.lag + 1)
        self.t = 0
        self.dead = False
        self.passed = 0
        self._alive = 0.0
        return self.obs()

    def pending(self):
        return list(self._q)[1:]

    def feedback(self) -> dict:
        return dict(
            dopamine=0.0 if self.dead else 1.0,
            pain=1.0 if self.dead else 0.0,
        )

    def obs(self) -> np.ndarray:
        x, x_dot, theta, theta_dot = self.state
        return np.asarray(
            [x, x_dot, theta, theta_dot, self._alive], dtype=np.float32
        )

    def step(self, action):
        self._q.append(int(action))
        eff_action = int(self._q[0])
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if eff_action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * (theta_dot ** 2) * sintheta) / self.total_mass
        thetaacc = (
            (self.gravity * sintheta - costheta * temp)
            / (
                self.half_length
                * (4.0 / 3.0 - self.masspole * (costheta ** 2) / self.total_mass)
            )
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        self.state = np.asarray([x, x_dot, theta, theta_dot], dtype=float)
        self.t += 1

        terminated = bool(
            abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
        )
        self.dead = terminated
        dopamine = 0.0 if terminated else 1.0
        self._alive += dopamine
        return self.obs(), dopamine, terminated


def make_cartpole_factory(lag: int = 2, cap: int = 1200, **physics_kw):
    """Return an ``env_factory(seed, **kwargs)`` compatible with ``substrate_nodes``."""
    def _factory(seed, **_ignored):
        return DelayedCartPole(seed=int(seed), lag=int(lag), cap=int(cap), **physics_kw)
    return _factory
