"""Tests for behavior-into-state (self_state) tracker, wrapper, and CLI/runner integration."""
from __future__ import annotations

import unittest
import numpy as np


class SelfStateTests(unittest.TestCase):
    def test_self_state_tracker_updates_and_augments(self):
        from self_state import SelfStateTracker

        tr = SelfStateTracker(n_actions=4, ema_alpha=0.5)
        v0 = tr.vector()
        self.assertEqual(v0.shape, (4,))
        self.assertAlmostEqual(float(v0[3]), 0.5)

        tr.update(action=3, dopamine=1.0, pain=0.0)
        v1 = tr.vector()
        self.assertAlmostEqual(float(v1[0]), 1.0)
        self.assertAlmostEqual(float(v1[1]), 1.0)
        self.assertAlmostEqual(float(v1[2]), 0.0)
        self.assertAlmostEqual(float(v1[3]), 0.75)

        aug = tr.augment_obs(np.array([0.1, -0.2], dtype=np.float32))
        self.assertEqual(aug.shape, (6,))

    def test_self_state_env_wrapper_on_cartpole(self):
        from cartpole_env import DelayedCartPole
        from self_state import SelfStateEnvWrapper

        env = SelfStateEnvWrapper(DelayedCartPole(seed=10, lag=2, cap=50))
        obs0 = env.obs()
        self.assertEqual(obs0.shape, (9,))
        obs1, _dop, _dead = env.step(1)
        self.assertEqual(obs1.shape, (9,))
        self.assertAlmostEqual(float(obs1[5]), 1.0)

    def test_run_seed_with_self_state_enabled(self):
        from evaluator import run_seed

        row = run_seed(
            dict(
                env="cartpole",
                self_state=True,
                seed=100,
                steps=400,
                coverage=0.15,
                cap=60,
                max_nodes=120,
                epochs=1,
                lag=2,
                eval_seeds=[200],
                eval_episodes=1,
                decision="chained",
                chain_adaptive=True,
                chain_chunk_size=2,
                chain_hops=2,
                chain_max_hops=3,
                chain_beam_width=2,
                chain_wide_beam_width=3,
            )
        )
        self.assertEqual(len(row["evals"]), 1)
        self.assertGreater(row["frames"], 0)


if __name__ == "__main__":
    unittest.main()
