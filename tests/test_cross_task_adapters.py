"""Cross-task adapter and multi-action/sparse-reward integration tests."""
from __future__ import annotations

import unittest
import numpy as np


class _MockDiscreteSpace:
    def __init__(self, n: int):
        self.n = int(n)


class _MockGym3ActionEnv:
    """Minimal Gymnasium-like 3-action env for testing GymnasiumLagWrapper."""

    def __init__(self):
        self.action_space = _MockDiscreteSpace(3)
        self.x = 0.0
        self.steps = 0

    def reset(self, seed=None):
        del seed
        self.x = 0.0
        self.steps = 0
        return np.array([self.x, 0.0], dtype=np.float32), {}

    def step(self, action: int):
        self.x += float(action - 1) * 0.25
        self.steps += 1
        terminated = abs(self.x) >= 1.0
        truncated = self.steps >= 30
        reward = -1.0 if terminated else 1.0
        return np.array([self.x, float(action - 1)], dtype=np.float32), reward, terminated, truncated, {}


class CrossTaskAdapterTests(unittest.TestCase):
    def test_gymnasium_lag_wrapper_protocol(self):
        from gym_adapter import GymnasiumLagWrapper

        env = GymnasiumLagWrapper(_MockGym3ActionEnv(), seed=7, lag=2, cap=25)
        self.assertEqual(env.action_space, (0, 1, 2))
        self.assertEqual(env.obs().shape, (3,))
        self.assertEqual(len(env.pending()), 2)
        # Action 2 is queued behind 2 zeros -> first step executes 0 (x=-0.25)
        obs1, dop1, dead1 = env.step(2)
        self.assertEqual(obs1.shape, (3,))
        self.assertEqual(dop1, 1.0)
        self.assertFalse(dead1)
        self.assertAlmostEqual(float(obs1[0]), -0.25, places=5)

    def test_delayed_mountaincar_3actions_and_lag(self):
        from mountaincar_env import DelayedMountainCar

        env = DelayedMountainCar(seed=11, lag=2, cap=50)
        self.assertEqual(env.action_space, (0, 1, 2))
        self.assertEqual(env.obs().shape, (3,))
        self.assertEqual(len(env.pending()), 2)
        for _ in range(50):
            obs, _dop, dead = env.step(2)
            self.assertEqual(obs.shape, (3,))
            if dead:
                break
        self.assertTrue(env.dead)

    def test_full_phase17_run_seed_on_3action_mountaincar(self):
        from evaluator import run_seed

        row = run_seed(
            dict(
                env="mountaincar",
                seed=100,
                steps=400,
                coverage=0.2,
                cap=60,
                max_nodes=120,
                epochs=1,
                lag=2,
                eval_seeds=[200, 201],
                eval_episodes=1,
                decision="chained",
                chain_adaptive=True,
                chain_chunk_size=2,
                chain_hops=3,
                chain_max_hops=4,
                chain_beam_width=3,
                chain_wide_beam_width=4,
                chain_demand_online_every=2,
                chain_demand_rel_pairs=2,
                chain_demand_min_terminal=4,
                world_online_refit_gain=0.30,
                world_online_min_frames=50,
                world_online_every=1,
                regime_value_weight=0.5,
                regime_n_regimes=6,
                regime_online_every=1,
                regime_online_frames=40,
                regime_failure_weight=0.25,
                regime_abstain_lcb=0.45,
            )
        )
        self.assertEqual(len(row["evals"]), 2)
        self.assertGreater(row["frames"], 0)


if __name__ == "__main__":
    unittest.main()
