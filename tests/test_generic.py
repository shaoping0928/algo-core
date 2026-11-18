import unittest

import numpy as np

from mixed_nodes import ContinuousMixedMemory, LayeredMixedMemory
from substrate_generic import LinearObservationModel, history_features
from trace_signal_world import signal_features


class GenericWorldTests(unittest.TestCase):
    def test_history_features_use_newest_action_first(self):
        sequences = np.asarray([[1.0, 0.0, 1.0]])
        histories = history_features(sequences, 3, prefix=[0.0, 1.0, 0.0])
        np.testing.assert_allclose(histories[0], [1.0, 0.0, 1.0, 0.0])

    def test_history_features_fill_early_steps_from_prefix(self):
        sequences = np.asarray([[1.0, 0.0]])
        histories = history_features(sequences, 1, prefix=[0.0, 1.0, 0.0])
        np.testing.assert_allclose(histories[0], [1.0, 0.0, 1.0, 0.0])

    def test_linear_observation_model_predicts_simple_transition(self):
        rng = np.random.RandomState(5)
        observations = rng.normal(size=(200, 2))
        histories = rng.normal(size=(200, 2))
        next_observations = (
            observations @ np.asarray([[1.0, 0.0], [0.0, 0.5]])
            + histories @ np.asarray([[0.0, 0.2], [0.3, 0.0]])
        )
        model = LinearObservationModel(2, 2).fit(
            observations, histories, next_observations
        )
        prediction = model.predict(observations[:5], histories[:5])
        np.testing.assert_allclose(prediction, next_observations[:5], atol=1e-7)
        self.assertLess(model.holdout_error, 1e-7)

    def test_signal_features_produce_parallel_channels(self):
        window = np.asarray([[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0],
                              [4.0, 8.0], [5.0, 10.0]]])
        raw = signal_features(window, "raw")
        diff = signal_features(window, "diff")
        buffered = signal_features(window, "buffer")
        all_signals = signal_features(window, "all")
        np.testing.assert_allclose(raw[0], [5.0, 10.0])
        np.testing.assert_allclose(diff[0, :2], [5.0, 10.0])
        np.testing.assert_allclose(diff[0, 2:4], [1.0, 2.0])
        np.testing.assert_allclose(diff[0, 4:6], [0.0, 0.0])
        self.assertEqual(raw.shape[-1], 2)
        self.assertEqual(diff.shape[-1], 6)
        self.assertEqual(buffered.shape[-1], 12)
        self.assertEqual(all_signals.shape[-1], 16)

    def test_continuous_mixed_memory_has_bounded_nodes_and_risk(self):
        rng = np.random.RandomState(7)
        observations = rng.normal(size=(200, 2))
        pain = ((observations[:, 0] > 1.0)
                & (observations[:, 1] > 1.0)).astype(float)
        memory = ContinuousMixedMemory(capacity=8).fit(observations, pain)
        self.assertGreater(len(memory.gates), 0)
        self.assertLessEqual(len(memory.nodes), 8)
        risk = memory.risk(observations)
        self.assertEqual(risk.shape, (200,))
        self.assertTrue(np.isfinite(risk).all())

    def test_layered_memory_reserves_rare_slots(self):
        rng = np.random.RandomState(8)
        observations = rng.normal(size=(300, 2))
        pain = ((observations[:, 0] > 0.8)
                & (observations[:, 1] < -0.8)).astype(float)
        memory = LayeredMixedMemory(
            common_capacity=4, rare_capacity=3, min_support=4.0
        ).fit(observations, pain)
        self.assertLessEqual(len(memory.nodes), 7)
        self.assertLessEqual(memory.rare_nodes, 3)
        risk = memory.risk(observations)
        self.assertEqual(risk.shape, (300,))
        self.assertTrue(np.isfinite(risk).all())


if __name__ == "__main__":
    unittest.main()
