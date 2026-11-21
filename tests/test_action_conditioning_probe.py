"""Regression tests for the action-conditioning LOEO probe."""
import unittest

import numpy as np

from action_conditioning_probe import loeo_r2, ridge_fit


class ActionConditioningProbeTest(unittest.TestCase):
    def test_ridge_recovers_a_linear_map(self):
        rng = np.random.RandomState(0)
        features = rng.normal(size=(200, 3))
        target = features @ np.asarray([1.0, -2.0, 0.5]) + 0.3
        weights = ridge_fit(features, target, alpha=1e-9)
        predicted = np.concatenate(
            [features, np.ones((len(features), 1))], axis=1
        ) @ weights
        self.assertLess(float(np.abs(predicted - target).max()), 1e-6)

    def test_action_adds_predictive_power_and_beats_shuffle(self):
        rng = np.random.RandomState(1)
        groups = np.repeat(np.arange(12), 10)
        actions = rng.randint(0, 2, (120, 1)).astype(float)
        states = rng.normal(size=(120, 3))
        target = (
            actions * np.asarray([1.0, -1.0])
            + 0.01 * rng.normal(size=(120, 2))
        )
        state_r2 = loeo_r2(states, target, groups)
        action_r2 = loeo_r2(
            np.concatenate([states, actions], axis=1), target, groups
        )
        shuffled_r2 = loeo_r2(
            np.concatenate([states, rng.permutation(actions)], axis=1),
            target, groups,
        )
        self.assertGreater(action_r2 - state_r2, 0.5)
        self.assertGreater(action_r2, shuffled_r2 + 0.25)

    def test_no_action_signal_gives_small_gain(self):
        rng = np.random.RandomState(2)
        groups = np.repeat(np.arange(10), 10)
        actions = rng.randint(0, 2, (100, 1)).astype(float)
        states = rng.normal(size=(100, 2))
        target = rng.normal(size=(100, 2))
        state_r2 = loeo_r2(states, target, groups)
        action_r2 = loeo_r2(
            np.concatenate([states, actions], axis=1), target, groups
        )
        self.assertLess(action_r2 - state_r2, 0.15)


if __name__ == "__main__":
    unittest.main()
