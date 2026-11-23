"""Tests for the time-in-state feature wrapper."""

import unittest

import numpy as np

from time_in_state import (
    TimeInStateEnv,
    augment_observations,
    estimate_period,
    make_env_factory,
)
from substrate_generic import collect
from substrate_nodes import evaluate, fit


class ToyEnv:
    action_space = (0, 1)

    def __init__(self):
        self.t = 0
        self.dead = False

    def obs(self):
        return np.asarray([float(self.t), 1.0])

    def step(self, action):
        self.t += 1
        return self.obs(), 1.0, self.dead

    def feedback(self):
        return dict(dopamine=1.0, pain=0.0)

    def pending(self):
        return [0, 1]

    def reset(self):
        self.t = 0
        self.dead = False


class TimeInStateTest(unittest.TestCase):
    def test_wrapper_appends_action_ages_and_phase(self):
        wrapper = TimeInStateEnv(ToyEnv(), period=10, max_age=4)
        self.assertEqual(wrapper.feature_dim, 5)
        observation = wrapper.obs()
        self.assertEqual(len(observation), 7)
        # Both actions are initially "long ago" at the capped value.
        self.assertEqual(observation[2], 1.0)
        self.assertEqual(observation[3], 1.0)
        self.assertAlmostEqual(observation[4], 0.0)   # sin(0)
        self.assertAlmostEqual(observation[5], 1.0)   # cos(0)
        self.assertAlmostEqual(observation[6], 0.1)   # 1/period

    def test_taking_an_action_resets_its_age_and_increments_the_other(self):
        wrapper = TimeInStateEnv(ToyEnv(), period=10, max_age=4)
        wrapper.step(1)
        observation = wrapper.obs()
        self.assertEqual(observation[2], 1.0)  # action 0 still capped
        self.assertEqual(observation[3], 0.0)  # action 1 just taken
        self.assertAlmostEqual(observation[4], np.sin(2.0 * np.pi / 10.0))
        self.assertAlmostEqual(observation[5], np.cos(2.0 * np.pi / 10.0))
        wrapper.step(0)
        observation = wrapper.obs()
        self.assertEqual(observation[2], 0.0)
        self.assertAlmostEqual(observation[3], 0.25)

    def test_wrapper_delegates_world_attributes(self):
        wrapper = TimeInStateEnv(ToyEnv(), period=10)
        self.assertEqual(wrapper.t, 0)
        self.assertFalse(wrapper.dead)
        self.assertEqual(wrapper.pending(), [0, 1])
        self.assertEqual(wrapper.feedback()["dopamine"], 1.0)

    def test_reset_restores_the_initial_ages(self):
        wrapper = TimeInStateEnv(ToyEnv(), period=10, max_age=4)
        wrapper.step(1)
        wrapper.reset()
        observation = wrapper.obs()
        self.assertEqual(observation[2], 1.0)
        self.assertEqual(observation[3], 1.0)
        self.assertEqual(wrapper.t, 0)

    def test_offline_augmentation_matches_the_wrapper_shape(self):
        observations = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        actions = np.asarray([1, 0, 1])
        episodes = np.asarray([0, 0, 0])
        augmented = augment_observations(
            observations, actions, episodes, period=10, max_age=4,
        )
        self.assertEqual(augmented.shape, (3, 7))
        self.assertEqual(augmented[0, 2], 1.0)
        self.assertEqual(augmented[1, 3], 0.0)
        self.assertEqual(augmented[2, 3], 0.25)

    def test_action_only_and_phase_only_widths(self):
        actions = TimeInStateEnv(
            ToyEnv(), period=10, include_phase=False,
        )
        self.assertEqual(actions.feature_dim, 2)
        self.assertEqual(len(actions.obs()), 4)
        phase = TimeInStateEnv(
            ToyEnv(), period=10, include_actions=False,
        )
        self.assertEqual(phase.feature_dim, 3)
        self.assertEqual(len(phase.obs()), 5)

    def test_period_estimator_finds_a_synthetic_sinusoid(self):
        rng = np.random.RandomState(0)
        period = 20
        steps = 200
        t = np.arange(steps)
        observations = np.sin(2.0 * np.pi * t / period).reshape(-1, 1)
        observations += 0.01 * rng.normal(size=observations.shape)
        episodes = np.zeros(steps, dtype=int)
        self.assertEqual(estimate_period(observations, episodes), period)

    def test_collect_accepts_the_time_env_factory(self):
        factory = make_env_factory(
            period=10, lag=3, include_phase=False,
        )
        rows, _ = collect(
            seed=100, steps=20, lag=3, env_factory=factory,
        )
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows[0]["obs"]), 9)
        self.assertEqual(len(rows[0]["next_obs"]), 9)

    def test_fit_and_evaluate_accept_the_time_env_factory(self):
        factory = make_env_factory(
            period=10, lag=3, include_phase=False,
        )
        world, agent = fit(
            seed=100, steps=30, epochs=1, max_nodes=50,
            birth_threshold=0.45, dagger_rounds=0,
            env_factory=factory,
        )
        self.assertEqual(agent.obs_dim, 9)
        frames, _actions, _rows = evaluate(
            100, world, agent, cap=10, horizon=4, lag=3,
            env_seed=1200000, env_factory=factory,
        )
        self.assertGreater(frames, 0)


if __name__ == "__main__":
    unittest.main()
