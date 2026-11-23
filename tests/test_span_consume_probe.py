"""Tests for the E2 consumption-distribution probe helpers."""

import unittest
from types import SimpleNamespace

import numpy as np

from span_consume_probe import (
    reliability_bins, restandardize, table_oof_windows,
)


class RestandardizeTest(unittest.TestCase):
    def test_restandardizes_back_to_training_stats(self):
        eval_entries = dict(
            phi=np.asarray([[1.0], [-1.0]]),
            mean=np.asarray([10.0]), scale=np.asarray([2.0]),
        )
        train_entries = dict(
            mean=np.asarray([0.0]), scale=np.asarray([1.0]),
        )
        # eval raw = 12, 8; train standardized = 12, 8.
        values = restandardize(eval_entries, train_entries)
        self.assertAlmostEqual(float(values[0, 0]), 12.0)
        self.assertAlmostEqual(float(values[1, 0]), 8.0)


class TableOofTest(unittest.TestCase):
    def test_predictions_are_finite_for_every_entry(self):
        rng = np.random.RandomState(0)
        features = np.concatenate([
            rng.normal(0.0, 0.1, (40, 2)),
            rng.normal(5.0, 0.1, (40, 2)),
        ])
        step = np.concatenate([
            np.full(40, 2), np.zeros(40),
        ]).astype(int)
        remaining = np.concatenate([
            np.full(40, 2), np.full(40, 40),
        ]).astype(int)
        episode = np.repeat(np.arange(8), 10)
        args = SimpleNamespace(k_fine=8, k_coarse=2, seed=0)
        output = table_oof_windows(
            features, step, remaining, episode, (8, 12), args, folds=4,
        )
        self.assertEqual(output[12].shape, (80,))
        self.assertTrue(np.isfinite(output[12]).all())
        self.assertGreater(
            float(output[12][:40].mean()), float(output[12][40:].mean())
        )


class ReliabilityTest(unittest.TestCase):
    def test_bins_report_prediction_and_truth(self):
        prediction = np.asarray([0.05, 0.1, 0.9, 0.95])
        step = np.asarray([50, 50, 2, 3])
        remaining = np.asarray([50, 50, 2, 3])
        report = reliability_bins(prediction, step, remaining, 8, bins=2)
        self.assertEqual(len(report), 2)
        self.assertAlmostEqual(report[0]["mean_truth"], 0.0)
        self.assertAlmostEqual(report[1]["mean_truth"], 1.0)


if __name__ == "__main__":
    unittest.main()
