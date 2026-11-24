"""Tests for similarity layers over integrated stage types."""
import unittest

import numpy as np

from stage_type_clusters import StageSample
from stage_type_layers import StageTypeLayers


def samples(outcomes, profiles, episode_offset=0):
    return [
        StageSample(
            episode=episode_offset + index,
            profile=np.asarray(profile, dtype=float),
            outcome=(outcome,),
        )
        for index, (outcome, profile) in enumerate(zip(outcomes, profiles))
    ]


class StageTypeLayerTests(unittest.TestCase):
    def setUp(self):
        self.train = samples(
            ["A", "A", "A", "B", "B", "B"],
            [
                [1.0, 0.05], [0.98, 0.02], [1.0, 0.10],
                [0.05, 1.0], [0.02, 0.98], [0.10, 1.0],
            ],
        )
        self.validation = samples(
            ["A", "B", "A", "B"],
            [[1.0, 0.02], [0.03, 1.0], [0.97, 0.05], [0.08, 0.99]],
            episode_offset=10,
        )

    def fit(self, **kwargs):
        params = dict(
            min_similarity=0.5, min_coverage=0.5, min_gain=0.01,
            tolerance=0.01, min_layer_support=2,
        )
        params.update(kwargs)
        return StageTypeLayers.fit(self.train, self.validation, **params)

    def test_two_profiles_choose_two_layers(self):
        model = self.fit()
        self.assertTrue(model.accepted)
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model.coverage, 1.0)
        self.assertGreater(model.predictive_gain, 0.1)
        self.assertEqual(
            [layer.support for layer in model.layers], [3, 3],
        )

    def test_three_outcome_groups_choose_three_layers(self):
        train = samples(
            ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            [
                [1, 0, 0], [0.98, 0.02, 0], [1, 0, 0.02],
                [0, 1, 0], [0.02, 0.98, 0], [0, 1, 0.02],
                [0, 0, 1], [0.02, 0, 0.98], [0, 0.02, 1],
            ],
        )
        validation = samples(
            ["A", "B", "C"],
            [[1, 0, 0.01], [0.01, 1, 0], [0, 0.01, 1]],
            episode_offset=20,
        )
        model = StageTypeLayers.fit(
            train, validation, min_similarity=0.5,
            min_coverage=0.5, min_gain=0.01, tolerance=0.01,
            min_layer_support=2,
        )
        self.assertTrue(model.accepted)
        self.assertEqual(len(model.layers), 3)

    def test_far_validation_is_rejected(self):
        shifted = samples(
            ["A", "B"], [[0.7, 0.7], [0.7, 0.7]], episode_offset=30,
        )
        model = StageTypeLayers.fit(
            self.train, shifted, min_similarity=0.8,
            min_coverage=0.5, min_gain=0.01, min_layer_support=2,
        )
        self.assertFalse(model.accepted)

    def test_assignment_and_unknown(self):
        model = self.fit()
        layer_id, similarity, margin = model.assign(
            np.asarray([1.0, 0.0])
        )
        self.assertIsNotNone(layer_id)
        self.assertGreater(similarity, 0.9)
        self.assertGreaterEqual(margin, 0.0)
        strict = self.fit(min_similarity=0.8)
        unknown = strict.assign(np.asarray([0.7071, 0.7071]))
        self.assertIsNone(unknown[0])

    def test_tree_and_frontier_snapshot(self):
        model = self.fit()
        snapshot = model.snapshot()
        self.assertEqual(snapshot["nodes"], 2 * len(self.train) - 1)
        self.assertEqual(snapshot["candidates"], len(self.train))
        self.assertEqual(snapshot["layers"], len(model.layers))
        self.assertGreater(snapshot["level"], 0.0)
        self.assertTrue(snapshot["accepted"])

    def test_fit_is_deterministic(self):
        first = self.fit()
        second = self.fit()
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(
            [layer.members for layer in first.layers],
            [layer.members for layer in second.layers],
        )


if __name__ == "__main__":
    unittest.main()
