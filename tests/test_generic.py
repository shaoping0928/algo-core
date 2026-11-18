import unittest

import numpy as np

from substrate_generic import LinearObservationModel, history_features


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


if __name__ == "__main__":
    unittest.main()
