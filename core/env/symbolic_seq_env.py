"""Self-contained large-vocabulary symbolic sequence prediction environment.

Designed to test sparse inverted-index retrieval and heuristic Top-K action
pruning on a large discrete vocabulary (e.g., ``vocab_size = 64`` or ``128``)
with order-sensitive context and action-execution delay ``lag``.
"""
from __future__ import annotations

from collections import deque
import numpy as np


class DelayedSymbolicSeqEnv:
    """Structured symbolic sequence prediction with configurable ``vocab_size`` and ``lag``."""

    def __init__(
        self,
        seed: int = 0,
        vocab_size: int = 64,
        lag: int = 1,
        cap: int = 120,
    ):
        self.seed = int(seed)
        self.vocab_size = max(int(vocab_size), 4)
        self.action_space = tuple(range(self.vocab_size))
        self.lag = max(int(lag), 0)
        self.max_t = int(cap)
        self.reset()

    def _expected_token(self) -> int:
        # Order-sensitive deterministic grammar rule over recent state + delayed token
        return int((3 * self._tok_prev2 + 5 * self._tok_prev1 + 7) % self.vocab_size)

    def reset(self):
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        self._tok_prev2 = int(rng.randint(self.vocab_size))
        self._tok_prev1 = int(rng.randint(self.vocab_size))
        self._q = deque([0] * (self.lag + 1), maxlen=self.lag + 1)
        self.t = 0
        self.dead = False
        self.passed = 0
        self._last_match = False
        self._alive = 0.0
        return self.obs()

    def pending(self):
        return list(self._q)[1:]

    def feedback(self) -> dict:
        if self._last_match:
            return dict(dopamine=1.0, pain=0.0)
        return dict(dopamine=0.0, pain=1.0 if self.dead else 0.25)

    def obs(self) -> np.ndarray:
        target = float(self._expected_token())
        inv_v = 1.0 / float(self.vocab_size)
        return np.asarray(
            [
                float(self._tok_prev2) * inv_v,
                float(self._tok_prev1) * inv_v,
                target * inv_v,
                self._alive,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self._q.append(int(action))
        eff_action = int(self._q[0]) % self.vocab_size
        target = self._expected_token()
        self._last_match = (eff_action == target)
        if self._last_match:
            self.passed += 1
        self._tok_prev2 = self._tok_prev1
        self._tok_prev1 = target
        self.t += 1
        self._alive += 1.0
        self.dead = bool(self.t >= self.max_t)
        dopamine = 1.0 if self._last_match else 0.0
        return self.obs(), dopamine, self.dead


def make_symbolic_seq_factory(vocab_size: int = 64, lag: int = 1, cap: int = 120):
    """Return an ``env_factory`` compatible with ``trainer.fit`` / ``evaluator.evaluate``."""
    def factory(seed, spacing=None, gravity=None, flap=None, speed=None):
        del spacing, gravity, flap, speed
        return DelayedSymbolicSeqEnv(
            seed=int(seed), vocab_size=vocab_size, lag=lag, cap=cap,
        )

    return factory
