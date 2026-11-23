"""Tests for the B1-plan endpoint correction probe."""

import unittest

import numpy as np

from endpoint_correction_probe import (
    calibration_error, choose, feature_matrix, fit_logistic,
    predict_logistic, rank_auc, standardize, summarize_arm,
    within_decision_metrics,
)


class EndpointCorrectionProbeTest(unittest.TestCase):
    def test_logistic_orders_a_separable_pair(self):
        features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0.0, 0.0, 1.0, 1.0])
        coefficients = fit_logistic(features, labels)
        prediction = predict_logistic(coefficients, features)
        self.assertLess(prediction[0], 0.5)
        self.assertGreater(prediction[-1], 0.5)

    def test_standardize_uses_train_statistics(self):
        train = np.asarray([[1.0], [3.0]])
        test = np.asarray([[5.0]])
        train_scaled, test_scaled = standardize(train, test)
        self.assertAlmostEqual(float(train_scaled.mean()), 0.0)
        self.assertAlmostEqual(float(test_scaled[0, 0]), 3.0)

    def test_rank_auc_perfect_and_reversed(self):
        self.assertEqual(rank_auc([0.1, 0.4, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(rank_auc([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]), 0.0)

    def test_calibration_error_zero_for_perfect_bins(self):
        self.assertAlmostEqual(
            calibration_error([0.05, 0.95], [0.0, 1.0]), 0.05
        )

    def test_choose_uses_prediction_then_fine_cost(self):
        prediction = np.asarray([0.2, 0.1, 0.1])
        scores = np.asarray([[0.5], [0.4], [0.7]])
        self.assertEqual(choose(prediction, np.ones(3, dtype=bool), scores), 1)
        self.assertEqual(
            choose(prediction, np.asarray([True, False, True]), scores), 2
        )

    def test_feature_matrix_shapes(self):
        decision = dict(
            endpoint=np.zeros((4, 3)),
            distance=np.zeros(4),
            scores=np.zeros((4, 6)),
        )
        self.assertEqual(feature_matrix(decision, "endpoint").shape, (4, 3))
        self.assertEqual(
            feature_matrix(decision, "endpoint_ood").shape, (4, 4)
        )
        self.assertEqual(
            feature_matrix(decision, "endpoint_fine").shape, (4, 10)
        )
        self.assertEqual(feature_matrix(decision, "fine").shape, (4, 6))

    def test_summarize_arm_counts_flips(self):
        rows = [
            dict(
                same_action_oracle=True, same_action_regret=0,
                global_oracle=True, global_regret=0,
                relevant_miss=False, same_action_changed=True,
                same_action_improved=True, same_action_worsened=False,
            ),
            dict(
                same_action_oracle=False, same_action_regret=2,
                global_oracle=False, global_regret=3,
                relevant_miss=True, same_action_changed=True,
                same_action_improved=False, same_action_worsened=True,
            ),
        ]
        summary = summarize_arm(rows)
        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["same_action_oracle"], 1)
        self.assertEqual(summary["improved"], 1)
        self.assertEqual(summary["worsened"], 1)
        self.assertEqual(summary["flip_precision"], 0.5)

    def test_within_decision_metrics_use_local_ranking(self):
        decision = dict(
            indices=np.asarray([0, 1, 2]),
            same_mask=np.asarray([True, True, False]),
            true_value=np.asarray([1, 3, 2]),
            same_best_value=3,
            scores=np.asarray([[0.1], [0.2], [0.3]]),
        )
        predictions = {
            (120, 1): dict(
                prediction=np.asarray([0.9, 0.1, 0.5]),
                decision=decision,
            )
        }
        result = within_decision_metrics(predictions, horizon=2)
        self.assertEqual(result["same_action_top1"], 1)
        self.assertEqual(result["decision_auc_count"], 1)
        self.assertEqual(result["same_action_auc_count"], 1)
        self.assertAlmostEqual(result["decision_auc_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
