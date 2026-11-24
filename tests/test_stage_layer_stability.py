"""Tests for cross-fold stability selection of stage-type layers."""
import unittest

import numpy as np

from stage_layer_stability import LayerStabilitySelector, episode_folds
from stage_type_clusters import StageSample


def clustered_samples(episodes=24):
    samples = []
    for index in range(episodes):
        if index % 2 == 0:
            outcome = "A"
            profile = np.asarray([1.0, 0.02 * (index % 3)])
        else:
            outcome = "B"
            profile = np.asarray([0.02 * (index % 3), 1.0])
        samples.append(StageSample(
            episode=index, profile=profile, outcome=(outcome,),
        ))
    return samples


class LayerStabilityTests(unittest.TestCase):
    def test_stable_clusters_give_a_stable_layer_count(self):
        selector = LayerStabilitySelector.fit(
            clustered_samples(), folds=4, seed=0,
            min_similarities=(0.5,), tolerances=(0.01, 0.2),
            min_agreement=0.6, min_coverage=0.5, min_gain=0.01,
            min_acceptance=0.8, min_type_support=2,
        )
        self.assertTrue(selector.accepted)
        self.assertIsNotNone(selector.chosen)
        self.assertEqual(selector.chosen.layers, (2, 2, 2, 2))
        self.assertEqual(selector.chosen.agreement, 1.0)
        self.assertGreater(selector.chosen.mean_gain, 0.1)

    def test_unreachable_gain_floor_is_rejected(self):
        selector = LayerStabilitySelector.fit(
            clustered_samples(), folds=3, seed=0,
            min_similarities=(0.5,), tolerances=(0.05,),
            min_agreement=0.6, min_coverage=0.5, min_gain=10.0,
        )
        self.assertFalse(selector.accepted)
        self.assertIsNone(selector.chosen)

    def test_folds_group_complete_episodes(self):
        samples = clustered_samples(episodes=12)
        seen = set()
        for train, validation in episode_folds(samples, folds=3, seed=1):
            train_episodes = {sample.episode for sample in train}
            validation_episodes = {sample.episode for sample in validation}
            self.assertFalse(train_episodes & validation_episodes)
            seen.update(validation_episodes)
        self.assertEqual(seen, {sample.episode for sample in samples})

    def test_selection_is_deterministic(self):
        first = LayerStabilitySelector.fit(
            clustered_samples(), folds=4, seed=3,
            min_similarities=(0.5, 0.9),
            tolerances=(0.01, 0.1), min_gain=0.01,
        )
        second = LayerStabilitySelector.fit(
            clustered_samples(), folds=4, seed=3,
            min_similarities=(0.5, 0.9),
            tolerances=(0.01, 0.1), min_gain=0.01,
        )
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_snapshot_reports_candidates(self):
        selector = LayerStabilitySelector.fit(
            clustered_samples(), folds=3, seed=0,
            min_similarities=(0.5,), tolerances=(0.05,),
        )
        snapshot = selector.snapshot()
        self.assertEqual(snapshot["candidates"], 1)
        self.assertGreaterEqual(snapshot["folds"], 2)
        self.assertIn("chosen", snapshot)


if __name__ == "__main__":
    unittest.main()
