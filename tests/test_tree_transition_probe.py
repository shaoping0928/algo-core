"""Tests for the two-level span tree (E1)."""

import unittest

import numpy as np

from tree_transition_probe import (
    calibration_transfer, evaluate_table, fit_tree, kmeans,
    predict_tree,
)


class KmeansTest(unittest.TestCase):
    def test_two_blobs_split(self):
        rng = np.random.RandomState(0)
        features = np.concatenate([
            rng.normal(0.0, 0.05, (40, 2)),
            rng.normal(10.0, 0.05, (40, 2)),
        ])
        centers = kmeans(features, 2, seed=3)
        self.assertEqual(len(centers), 2)
        spread = np.sort(np.linalg.norm(centers - centers.mean(0), axis=1))
        self.assertGreater(spread[-1], 3.0)

    def test_k_larger_than_points(self):
        centers = kmeans(np.zeros((3, 2)), 8, seed=0)
        self.assertEqual(len(centers), 3)


class TreeTest(unittest.TestCase):
    def _synthetic(self):
        rng = np.random.RandomState(1)
        early = rng.normal(0.0, 0.05, (60, 2))
        late = rng.normal(10.0, 0.05, (60, 2))
        features = np.concatenate([early, late])
        step = np.concatenate([
            np.full(60, 2),          # early cluster dies at step 2
            np.zeros(60),            # late cluster survives the horizon
        ]).astype(int)
        remaining = np.concatenate([
            np.full(60, 2), np.full(60, 40),
        ]).astype(int)
        return features, step, remaining

    def test_prediction_separates_the_clusters(self):
        features, step, remaining = self._synthetic()
        model = fit_tree(
            features, step, remaining, horizon=16, k_fine=8, k_coarse=2,
            seed=0,
        )
        values = predict_tree(model, features, 12)
        self.assertGreater(float(values[:60].mean()), 0.8)
        self.assertLess(float(values[60:].mean()), 0.2)

    def test_horizon_is_monotone(self):
        features, step, remaining = self._synthetic()
        model = fit_tree(
            features, step, remaining, horizon=16, k_fine=8, k_coarse=2,
            seed=0,
        )
        low = predict_tree(model, features, 4)[:60].mean()
        high = predict_tree(model, features, 12)[:60].mean()
        self.assertLessEqual(low, high)

    def test_auc_is_one_on_separable_data(self):
        features, step, remaining = self._synthetic()
        model = fit_tree(
            features, step, remaining, horizon=16, k_fine=8, k_coarse=2,
            seed=0,
        )
        prediction = predict_tree(model, features, 12)
        metrics = evaluate_table(prediction, step, remaining, 12)
        self.assertAlmostEqual(metrics["auc"], 1.0)


class CalibrationTest(unittest.TestCase):
    def test_threshold_moves_from_train_to_test(self):
        step_train = np.asarray([1, 2, 40, 40, 40, 40])
        remaining_train = np.asarray([1, 2, 40, 40, 40, 40])
        prediction_train = np.asarray([0.9, 0.8, 0.1, 0.1, 0.1, 0.1])
        step_test = np.asarray([1, 50])
        remaining_test = np.asarray([1, 50])
        prediction_test = np.asarray([0.85, 0.05])
        result = calibration_transfer(
            prediction_train, step_train, remaining_train, 8,
            prediction_test, step_test, remaining_test,
        )
        self.assertIsNotNone(result["tau"])
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
