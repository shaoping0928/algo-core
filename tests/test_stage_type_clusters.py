"""Tests for similarity-based stage types over integrated states."""
import unittest

import numpy as np

from stage_type_clusters import StageSample, StageTypeClusters


def samples(outcomes, profiles, episode_offset=0):
    return [
        StageSample(
            episode=episode_offset + index,
            profile=np.asarray(profile, dtype=float),
            outcome=(outcome,),
        )
        for index, (outcome, profile) in enumerate(zip(outcomes, profiles))
    ]


class StageTypeClusterTests(unittest.TestCase):
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
        return StageTypeClusters.fit(
            self.train, self.validation, min_similarity=0.8,
            min_type_support=2, **kwargs,
        )

    def test_two_profiles_become_two_types(self):
        model = self.fit()
        self.assertEqual(len(model.types), 2)
        self.assertEqual(
            [stage_type.support for stage_type in model.types.values()],
            [3, 3],
        )
        self.assertEqual(
            [stage_type.purity for stage_type in model.types.values()],
            [1.0, 1.0],
        )

    def test_held_out_gate_accepts_a_reachable_useful_type(self):
        model = self.fit()
        self.assertTrue(model.accepted)
        self.assertEqual(model.coverage, 1.0)
        self.assertGreater(model.mean_margin, 0.5)
        self.assertGreater(model.predictive_gain, 0.0)

    def test_far_validation_is_rejected(self):
        shifted = samples(
            ["A", "B", "A", "B"],
            [[0.7, 0.7]] * 4, episode_offset=20,
        )
        model = StageTypeClusters.fit(
            self.train, shifted, min_similarity=0.8,
            min_type_support=2,
        )
        self.assertEqual(model.coverage, 0.0)
        self.assertFalse(model.accepted)

    def test_similarity_threshold_controls_merging(self):
        loose = StageTypeClusters.fit(
            self.train, min_similarity=0.8, min_type_support=1,
        )
        strict = StageTypeClusters.fit(
            self.train, min_similarity=0.999, min_type_support=1,
        )
        self.assertEqual(len(loose.types), 2)
        self.assertGreater(len(strict.types), len(loose.types))

    def test_unknown_profile_returns_none(self):
        model = self.fit()
        type_id, similarity, _margin = model.assign(np.asarray([0.0, 1.0]))
        self.assertIsNotNone(type_id)
        other = model.assign(np.asarray([0.7071, 0.7071]))
        self.assertIsNone(other[0])
        self.assertLess(other[1], 0.8)

    def test_low_support_type_is_unknown(self):
        train = samples(
            ["A", "B", "B", "B"],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.99], [0.01, 1.0]],
        )
        model = StageTypeClusters.fit(
            train, min_similarity=0.8, min_type_support=2,
        )
        singleton = model.assign(np.asarray([1.0, 0.0]))
        self.assertIsNone(singleton[0])
        supported = model.assign(np.asarray([0.0, 1.0]))
        self.assertIsNotNone(supported[0])

    def test_fit_is_deterministic(self):
        first = self.fit()
        second = self.fit()
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(
            [stage_type.members for stage_type in first.types.values()],
            [stage_type.members for stage_type in second.types.values()],
        )


if __name__ == "__main__":
    unittest.main()
