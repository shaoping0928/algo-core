"""Tests for the time-in-state feature wrapper."""

import unittest

import numpy as np

from time_in_state import (
    ConditionedTimeWorld,
    ControlFeatureEnv,
    PerTargetAgeWorld,
    TimeInStateEnv,
    augment_observations,
    estimate_period,
    make_control_env_factory,
    make_env_factory,
)
from substrate_generic import LinearObservationModel, collect
from substrate_nodes import evaluate, fit
from time_world_probe import StandardizedLinearModel


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

    def test_control_feature_is_nonconstant_and_task_free(self):
        wrapper = ControlFeatureEnv(ToyEnv(), frequency=0.7)
        self.assertEqual(wrapper.feature_dim, 1)
        self.assertEqual(len(wrapper.obs()), 3)
        self.assertAlmostEqual(wrapper.obs()[2], 0.0)
        wrapper.step(1)
        self.assertAlmostEqual(wrapper.obs()[2], np.sin(0.7))
        self.assertEqual(wrapper.t, 1)

    def test_standardized_world_matches_the_linear_fit(self):
        rng = np.random.RandomState(9)
        observations = rng.normal(size=(300, 3))
        histories = rng.normal(size=(300, 1))
        next_observations = np.zeros((300, 2))
        next_observations[:, 0] = (
            0.5 + 0.4 * observations[:, 0] - 0.2 * histories[:, 0]
        )
        next_observations[:, 1] = -0.3 + 0.6 * observations[:, 2]
        base = LinearObservationModel(3, 1).fit(
            observations, histories, next_observations,
        )
        standardized = StandardizedLinearModel(3, 1).fit(
            observations, histories, next_observations,
        )
        np.testing.assert_allclose(
            standardized.predict(observations, histories),
            base.predict(observations, histories),
            atol=1e-6,
        )

    def test_conditioned_world_advances_ages_and_keeps_raw_predictions(self):
        rng = np.random.RandomState(12)
        raw = rng.normal(size=(200, 2))
        histories = rng.randint(0, 2, size=(200, 1)).astype(float)
        next_raw = 0.5 * raw + 0.1
        base = LinearObservationModel(2, 1).fit(raw, histories, next_raw)
        ages = np.zeros((200, 1))
        observations = np.concatenate([raw, ages], axis=1)
        next_observations = np.concatenate(
            [next_raw, np.zeros((200, 1))], axis=1
        )
        world = ConditionedTimeWorld.from_base(
            base, observations, histories, next_observations,
            max_age=4, age_actions=(1,),
        )
        states = np.asarray([
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.25],
        ])
        actions = np.asarray([[0.0], [1.0]])
        prediction = world.predict(states, actions)
        self.assertEqual(prediction.shape, (2, 3))
        self.assertAlmostEqual(prediction[0, 2], 0.5)
        self.assertEqual(prediction[1, 2], 0.0)
        expected_raw = base.predict(states[:, :2], actions)
        np.testing.assert_allclose(
            prediction[:, :2], expected_raw, atol=1e-3,
        )
        zero = ConditionedTimeWorld.from_base(
            base, observations, histories, next_observations,
            max_age=4, age_actions=(1,), residual_scale=0.0,
        )
        np.testing.assert_allclose(
            zero.predict(states, actions)[:, :2],
            expected_raw,
            atol=1e-12,
        )

    def test_conditioned_world_gate_keeps_a_predictive_feature(self):
        rng = np.random.RandomState(13)
        raw = rng.normal(size=(300, 2))
        histories = rng.randint(0, 2, size=(300, 1)).astype(float)
        base_target = 0.4 * raw + 0.1
        base = LinearObservationModel(2, 1).fit(
            raw, histories, base_target,
        )
        features = rng.normal(size=(300, 1))
        target = base_target + 2.0 * features
        observations = np.concatenate([raw, features], axis=1)
        next_observations = np.concatenate(
            [target, np.zeros((300, 1))], axis=1
        )
        episodes = np.arange(300) // 15
        world = ConditionedTimeWorld.from_base(
            base, observations, histories, next_observations,
            max_age=16, age_actions=(1,), episodes=episodes,
        )
        self.assertGreater(world.validation_gain, 0.0)
        self.assertGreater(world.residual_scale, 0.0)

    def test_conditioned_world_gate_drops_an_irrelevant_feature(self):
        rng = np.random.RandomState(14)
        raw = rng.normal(size=(300, 2))
        histories = rng.randint(0, 2, size=(300, 1)).astype(float)
        target = 0.4 * raw + 0.1
        base = LinearObservationModel(2, 1).fit(raw, histories, target)
        features = rng.normal(size=(300, 1))
        observations = np.concatenate([raw, features], axis=1)
        next_observations = np.concatenate(
            [target, np.zeros((300, 1))], axis=1
        )
        episodes = np.arange(300) // 15
        world = ConditionedTimeWorld.from_base(
            base, observations, histories, next_observations,
            max_age=16, age_actions=(1,), episodes=episodes,
        )
        self.assertLessEqual(world.validation_gain, 1e-6)
        self.assertEqual(world.residual_scale, 0.0)

    def test_per_target_age_world_enables_only_a_predictive_target(self):
        rng = np.random.RandomState(15)
        raw = rng.normal(size=(300, 2))
        histories = rng.randint(0, 2, size=(300, 1)).astype(float)
        base_target = 0.4 * raw + 0.1
        base = LinearObservationModel(2, 1).fit(
            raw, histories, base_target,
        )
        features = rng.uniform(size=(300, 1))
        target = base_target.copy()
        target[:, 0] += 2.0 * features[:, 0]
        observations = np.concatenate([raw, features], axis=1)
        next_observations = np.concatenate(
            [target, np.zeros((300, 1))], axis=1
        )
        episodes = np.arange(300) // 15
        world = PerTargetAgeWorld.from_base(
            base, observations, histories, next_observations,
            episodes, max_age=16, age_actions=(1,), targets=(0, 1),
        )
        self.assertGreater(world.gains[0], 0.0)
        self.assertIn(0, world.corrections)
        self.assertNotIn(1, world.corrections)
        prediction = world.predict(
            np.concatenate([raw, features], axis=1), histories
        )
        self.assertEqual(prediction.shape, (300, 3))
        np.testing.assert_allclose(
            prediction[:, 1], base_target[:, 1], atol=1e-9,
        )
        expected_feature = (
            np.minimum(np.rint(features[:, 0] * 16) + 1, 16) / 16.0
        )
        expected_feature[histories[:, 0] == 1] = 0.0
        np.testing.assert_allclose(
            prediction[:, 2], expected_feature, atol=1e-9,
        )

    def test_per_target_age_world_with_no_targets_is_the_base(self):
        rng = np.random.RandomState(16)
        raw = rng.normal(size=(200, 2))
        histories = rng.randint(0, 2, size=(200, 1)).astype(float)
        target = 0.4 * raw + 0.1
        base = LinearObservationModel(2, 1).fit(raw, histories, target)
        features = rng.uniform(size=(200, 1))
        observations = np.concatenate([raw, features], axis=1)
        next_observations = np.concatenate(
            [target, np.zeros((200, 1))], axis=1
        )
        episodes = np.arange(200) // 10
        world = PerTargetAgeWorld.from_base(
            base, observations, histories, next_observations,
            episodes, max_age=16, age_actions=(1,), targets=(),
        )
        self.assertFalse(world.corrections)
        prediction = world.predict(
            np.concatenate([raw, features], axis=1), histories
        )
        np.testing.assert_allclose(
            prediction[:, :2], base.predict(raw, histories), atol=1e-12,
        )

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
