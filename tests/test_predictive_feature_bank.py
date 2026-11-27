import unittest

import numpy as np

from predictive_feature_bank import (
    EpisodeBalancedFeatureBank,
    FeatureAtom,
    FeatureExpression,
)


def repeated_pattern(episode_ids, repeats=4):
    values = []
    labels = []
    ids = []
    for episode in episode_ids:
        for state, target in ((0.0, 0.0), (1.0, 1.0)):
            values.extend([[state]] * repeats)
            labels.extend([target] * repeats)
            ids.extend([episode] * repeats)
    return (
        np.asarray(values, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(ids, dtype=object),
    )


class PredictiveFeatureBankTests(unittest.TestCase):
    def test_threshold_feature_is_typed_and_interpretable(self):
        expression = FeatureExpression(
            family="state",
            operator="atom",
            atoms=(FeatureAtom(0, "abs", ">", 0.5),),
        )

        np.testing.assert_array_equal(
            expression.mask(np.asarray([[0.1], [-0.8], [0.5]])),
            [False, True, False],
        )
        self.assertEqual(expression.label(["velocity"]), "|velocity| > 0.5")
        self.assertEqual(expression.complexity, 1)

    def test_repeating_frames_in_one_episode_does_not_pass_episode_gate(self):
        values = np.asarray(
            [[1.0]] * 100 + [[0.0]] * 40, dtype=float,
        )
        labels = np.asarray([1.0] * 100 + [0.0] * 40)
        episode_ids = np.asarray(
            [0] * 100 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10,
            dtype=object,
        )
        bank = EpisodeBalancedFeatureBank(
            family="sparse", quantiles=(0.5,),
            min_train_episode_support=2, max_compose_atoms=0,
        ).fit(values, labels, episode_ids)

        # The high-valued predicate occurs in 100 correlated rows, but only
        # one independent episode, so it is not admitted to the bank.
        self.assertEqual(bank.records, [])

    def test_threshold_quantiles_weight_episodes_not_frame_counts(self):
        values = np.asarray([[0.0]] * 20 + [[10.0]], dtype=float)
        labels = np.asarray([0.0] * 20 + [1.0], dtype=float)
        episode_ids = np.asarray([0] * 20 + [1], dtype=object)
        bank = EpisodeBalancedFeatureBank(
            family="state", quantiles=(0.5,),
            min_train_episode_support=1, max_compose_atoms=0,
        ).fit(values, labels, episode_ids)

        thresholds = [
            atom.threshold
            for record in bank.records
            for atom in record.expression.atoms
        ]

        self.assertTrue(thresholds)
        self.assertTrue(all(0.0 < value < 10.0 for value in thresholds))

    def test_calibration_selects_only_supported_episode_stable_features(self):
        train = repeated_pattern(range(12))
        calibration = repeated_pattern(range(100, 108))
        test = repeated_pattern(range(200, 206))
        bank = EpisodeBalancedFeatureBank(
            family="state", quantiles=(0.5,),
            min_train_episode_support=3,
            min_calibration_episode_support=3,
            min_coverage=0.25, max_compose_atoms=0,
            bootstrap_samples=200, seed=4,
        ).fit(*train)

        calibration_report = bank.calibrate(*calibration)
        outer = bank.evaluate(
            *test, feature_ids=calibration_report["frontier_ids"],
        )

        self.assertGreater(calibration_report["candidates"], 0)
        self.assertGreater(calibration_report["calibration_supported"], 0)
        self.assertTrue(outer)
        self.assertTrue(any(item["outer_test_supported"] for item in outer))
        self.assertEqual(
            {record.status for record in bank.records
             if record.feature_id in bank.selected_ids},
            {"outer_validated"},
        )

    def test_uncalibrated_bank_abstains(self):
        train = repeated_pattern(range(8))
        bank = EpisodeBalancedFeatureBank(
            family="state", quantiles=(0.5,),
            min_train_episode_support=2,
            min_calibration_episode_support=20,
            max_compose_atoms=0,
        ).fit(*train)
        bank.calibrate(*repeated_pattern(range(100, 104)))

        prediction = bank.predict(np.asarray([[1.0]]))[0]

        self.assertEqual(prediction["status"], "unknown")
        self.assertIsNone(prediction["probability"])

    def test_out_of_range_state_is_unknown_even_after_calibration(self):
        train = repeated_pattern(range(12))
        calibration = repeated_pattern(range(100, 108))
        bank = EpisodeBalancedFeatureBank(
            family="state", quantiles=(0.5,),
            min_train_episode_support=3,
            min_calibration_episode_support=3,
            min_coverage=0.25, max_compose_atoms=0,
            bootstrap_samples=200, seed=4,
        ).fit(*train)
        bank.calibrate(*calibration)

        low, high, outside = bank.predict(
            np.asarray([[0.0], [1.0], [1000.0]]),
        )

        self.assertEqual(low["status"], "supported")
        self.assertEqual(high["status"], "supported")
        self.assertEqual(outside["status"], "unknown")
        self.assertIsNone(outside["probability"])

    def test_calibration_does_not_refit_train_thresholds(self):
        train = repeated_pattern(range(8))
        bank = EpisodeBalancedFeatureBank(
            family="state", quantiles=(0.5,),
            min_train_episode_support=2, max_compose_atoms=0,
        ).fit(*train)
        before = tuple(record.feature_id for record in bank.records)
        calibration = repeated_pattern(range(100, 104))
        calibration_x = calibration[0] * 1000.0
        bank.calibrate(calibration_x, calibration[1], calibration[2])

        self.assertEqual(
            tuple(record.feature_id for record in bank.records), before,
        )
        self.assertTrue(all(
            record.expression.family == "state" for record in bank.records
        ))


if __name__ == "__main__":
    unittest.main()
