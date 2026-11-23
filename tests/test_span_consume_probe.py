"""Tests for the E2 consumption-distribution probe helpers."""

import unittest
from types import SimpleNamespace

import numpy as np

from span_consume_probe import (
    blend_ranks, fusion_arm, percentile_rank, reliability_bins,
    restandardize, table_oof_windows,
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


class RankFusionTest(unittest.TestCase):
    def test_percentile_rank_is_monotone_and_scale_free(self):
        values = np.asarray([10.0, 20.0, 20.0, 40.0, np.nan])
        ranks = percentile_rank(values)
        self.assertEqual(ranks[0], 0.0)
        self.assertAlmostEqual(ranks[1], 0.5)
        self.assertAlmostEqual(ranks[2], 0.5)
        self.assertEqual(ranks[3], 1.0)
        self.assertTrue(np.isnan(ranks[4]))
        scaled = percentile_rank(values * 3.0 + 7.0)
        np.testing.assert_allclose(ranks[:4], scaled[:4])

    def test_blend_falls_back_to_the_available_side(self):
        left = np.asarray([0.2, np.nan, 0.8])
        right = np.asarray([0.6, 0.4, np.nan])
        fused = blend_ranks(left, right, 0.5)
        self.assertAlmostEqual(fused[0], 0.4)
        self.assertAlmostEqual(fused[1], 0.4)
        self.assertAlmostEqual(fused[2], 0.8)

    def test_fusion_arm_keeps_tau_on_the_training_stream(self):
        train = np.arange(20, dtype=float)
        train_step = np.where(np.arange(20) >= 18, 2, 50).astype(int)
        eval_values = np.asarray([1.0, 2.0, 3.0, 8.0, 9.0])
        step = np.asarray([50, 50, 50, 2, 2])
        remaining = np.asarray([50, 50, 50, 2, 2])
        table_output = dict(
            train_prediction={12: train},
            eval_prediction={12: eval_values},
            step=train_step, remaining=np.full(20, 50),
            eval_step=step, eval_remaining=remaining,
        )
        logistic_output = dict(
            train_prediction=np.column_stack([train] * 16),
            eval_prediction=np.column_stack([eval_values] * 16),
        )
        args = SimpleNamespace(precision_target=0.5)
        report, entries = fusion_arm(table_output, logistic_output, args)
        row = report["12"]
        self.assertIn("tau", row)
        self.assertGreaterEqual(row["precision"], 0.5)
        self.assertEqual(len(entries["12"]["label"]), len(step))
        self.assertEqual(len(entries["12"]["prediction"]), len(step))


if __name__ == "__main__":
    unittest.main()
