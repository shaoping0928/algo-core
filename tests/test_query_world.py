"""Tests for the query-graph world-model adapter."""

import unittest

import numpy as np

from query_graph import RuleBank
from query_world import QueryWorldModel, fit_query_world
from substrate_generic import LinearObservationModel


class QueryWorldTest(unittest.TestCase):
    def make_data(self, rows=300, seed=5):
        rng = np.random.RandomState(seed)
        observations = rng.normal(size=(rows, 3))
        histories = rng.normal(size=(rows, 1))
        next_observations = np.zeros((rows, 2))
        next_observations[:, 0] = (
            0.4 + 0.8 * observations[:, 0] - 0.3 * histories[:, 0]
        )
        next_observations[:, 1] = -0.2 + 0.5 * observations[:, 2]
        return observations, histories, next_observations

    def test_query_world_matches_the_linear_model_on_heldout_rows(self):
        observations, histories, next_observations = self.make_data()
        train = slice(0, 240)
        test = slice(240, 300)
        baseline = LinearObservationModel(
            observations.shape[1], histories.shape[1]
        ).fit(
            observations[train], histories[train],
            next_observations[train],
        )
        query = fit_query_world(
            observations[train], histories[train],
            next_observations[train], max_terms=4,
        )
        expected = baseline.predict(observations[test], histories[test])
        actual = query.predict(observations[test], histories[test])
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_missing_rules_return_nan_not_a_source_value(self):
        model = QueryWorldModel(
            bank=RuleBank([]), n_state=2, n_action=1,
        )
        prediction = model.predict(
            np.asarray([[1.0, 2.0]]), np.asarray([[0.0]]),
        )
        self.assertTrue(np.all(np.isnan(prediction)))


if __name__ == "__main__":
    unittest.main()
