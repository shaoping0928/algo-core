import unittest

import numpy as np

from accuracy_coverage_probe import (
    apply_option_prior_shrinkage,
    calibration_report,
    fit_heldout_memory,
    select_shrinkage_alpha,
)


def make_rows(episode, values, terminal=True):
    rows = []
    for index in range(len(values) - 1):
        last = index == len(values) - 2
        rows.append(dict(
            episode=episode,
            obs=np.asarray([values[index]], dtype=float),
            next_obs=np.asarray([values[index + 1]], dtype=float),
            history=np.asarray([0.0, 0.0, 0.0, 0.0]),
            signals=np.zeros(2, dtype=float),
            terminal=bool(terminal and last),
            done=bool(terminal and last),
        ))
    return rows


class AccuracyCoverageProbeTests(unittest.TestCase):
    def test_heldout_detector_and_memory_use_train_episodes_only(self):
        rows = []
        for episode in range(4):
            rows.extend(make_rows(episode, [0.0, 0.2, 0.2, 0.2]))
        rows.extend(make_rows(4, [0.0, 100.0, 100.0, 100.0]))

        train, validation, memory, _prior, train_rows = fit_heldout_memory(
            rows, lag=3, folds=5, validation_fold=4,
            quantile=0.8, min_support=1, min_episodes=1,
        )

        train_ids = {
            stage.episode for episode in train.episodes for stage in episode
        }
        validation_ids = {
            stage.episode
            for episode in validation.episodes for stage in episode
        }
        self.assertEqual(train_ids, {0, 1, 2, 3})
        self.assertEqual(validation_ids, {4})
        self.assertEqual(
            {row["episode"] for row in train_rows}, train_ids,
        )
        self.assertLess(
            train.detector.thresholds[("observation_change", 0)], 1.0,
        )
        self.assertEqual(memory.snapshot()["transition_rows"], 8)
        prediction = memory.predict([()], ((0, 1),))
        self.assertEqual(prediction["episode_support"], 4)

    def test_shrinkage_uses_episode_support_as_pseudo_count_weight(self):
        context_a = (("a",), False, False)
        context_b = (("b",), False, False)
        row = dict(
            episode=1,
            episode_support=1,
            distribution={context_a: 0.9, context_b: 0.1},
            prior_distribution={context_a: 0.1, context_b: 0.9},
        )

        adjusted = apply_option_prior_shrinkage([row], alpha=3.0)[0]

        self.assertAlmostEqual(adjusted["distribution"][context_a], 0.3)
        self.assertAlmostEqual(adjusted["distribution"][context_b], 0.7)
        self.assertAlmostEqual(sum(adjusted["distribution"].values()), 1.0)

    def test_shrinkage_alpha_is_selected_from_nested_train_oof_only(self):
        rows = []
        for episode in range(5):
            rows.extend(make_rows(episode, [0.0, 0.0, 0.0, 0.0]))
        train, validation, _memory, _prior, train_rows = fit_heldout_memory(
            rows, lag=3, folds=5, validation_fold=4,
            min_support=1, min_episodes=1,
        )
        self.assertEqual(
            {
                stage.episode
                for episode in validation.episodes for stage in episode
            },
            {4},
        )

        selection = select_shrinkage_alpha(
            train_rows, lag=3, candidates=[0.0, 2.0], folds=2,
            min_support=1, min_episodes=1, min_oof_episodes=2,
        )

        self.assertEqual(selection["oof_episode_ids"], [0, 1, 2, 3])
        self.assertNotIn(4, selection["oof_episode_ids"])
        self.assertEqual(selection["selected_alpha"], 0.0)

    def test_calibration_gate_scores_uncensored_supported_rows_only(self):
        target = (("successor",), False, False)
        other = (("other",), False, False)
        rows = [
            dict(
                episode=episode,
                censored=False,
                supported=True,
                realized=target,
                distribution={target: 0.8, other: 0.2},
                prior_distribution={target: 0.5, other: 0.5},
            )
            for episode in range(24)
        ]
        rows.append(dict(
            episode=24,
            censored=True,
            supported=True,
            realized=(None, False, True),
            distribution={(None, False, True): 1.0},
            prior_distribution={(None, False, True): 1.0},
        ))

        report = calibration_report(
            rows, bootstrap_samples=200, seed=7,
            min_eval_episodes=20, min_coverage=0.5,
        )

        self.assertEqual(report["validation_uncensored"], 24)
        self.assertEqual(report["validation_censored"], 1)
        self.assertEqual(report["model"]["count"], 24)
        self.assertAlmostEqual(report["brier_skill"], 0.84)
        self.assertGreater(report["nll_gain"], 0.0)
        self.assertTrue(report["gate"]["passed"])

    def test_unknown_predictions_reduce_coverage_and_fail_gate(self):
        target = (("successor",), False, False)
        rows = [
            dict(
                episode=episode,
                censored=False,
                supported=episode == 0,
                realized=target,
                distribution={target: 1.0} if episode == 0 else None,
                prior_distribution={target: 1.0} if episode == 0 else None,
            )
            for episode in range(10)
        ]

        report = calibration_report(
            rows, bootstrap_samples=100, seed=1,
            min_eval_episodes=2, min_coverage=0.5,
        )

        self.assertAlmostEqual(report["supported_coverage"], 0.1)
        self.assertEqual(report["unknown_uncensored"], 9)
        self.assertIn(
            "supported_coverage_below_threshold",
            report["gate"]["reasons"],
        )
        self.assertFalse(report["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
