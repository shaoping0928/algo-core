"""Tests for the event/stage outcome audit."""

import unittest

import numpy as np

from outcome_level_probe import (
    candidate_predicates, clock_dimensions, contrast_report, dedupe_predicates,
    entry_labels, episode_tables, event_mask, event_transitions,
    fold_splits, person_period, stream_stats, survival_curve, time_basis,
    window_metrics,
)


def _table(episode, obs, actions, done):
    return dict(
        episode=episode,
        obs=np.asarray(obs, dtype=float),
        actions=np.asarray(actions, dtype=int),
        done=np.asarray(done, dtype=bool),
    )


class OutcomeLevelProbeTest(unittest.TestCase):
    def test_episode_tables_group_rows(self):
        rows = [
            dict(obs=[0.0, 1.0], history=[1, 0, 0, 0], done=False, episode=7),
            dict(obs=[0.1, 1.1], history=[0, 1, 0, 0], done=True, episode=7),
            dict(obs=[0.2, 1.2], history=[0, 0, 0, 0], done=False, episode=8),
        ]
        tables = episode_tables(rows)
        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0]["episode"], 7)
        self.assertEqual(tables[0]["actions"].tolist(), [1, 0])
        self.assertEqual(tables[0]["done"].tolist(), [False, True])

    def test_clock_dimensions_detects_counters_not_sawtooth(self):
        tables = [
            _table(
                0,
                [
                    [0.0, 0.0, 0.80],
                    [0.1, 1.0, 0.79],
                    [0.25, 2.0, 0.78],
                    [0.3, 3.0, 1.55],
                ],
                [0, 0, 0, 0],
                [False, False, False, False],
            ),
        ]
        # dim 1 is a counter (constant increments), dim 2 is a sawtooth
        # structural coordinate with a reset jump: not a clock.
        self.assertEqual(clock_dimensions(tables, 3), [1])

    def test_event_mask_kinds(self):
        obs = [[0.0], [0.5], [0.05], [-0.2], [0.3]]
        table = _table(0, obs, [0] * 5, [False] * 5)
        stats = stream_stats([table], [0])
        low = event_mask(table, dict(
            name="low", kind="cross_low", dim=0, thr=stats[0]["q10"],
        ))
        self.assertEqual(low.tolist(), [False, False, False, True, False])
        sign = event_mask(table, dict(name="sign", kind="sign", dim=0))
        self.assertEqual(np.flatnonzero(sign).tolist(), [1, 3, 4])
        jump = event_mask(table, dict(
            name="jump", kind="jump", dim=0, thr=0.4,
        ))
        self.assertEqual(np.flatnonzero(jump).tolist(), [1, 2, 4])

    def test_dedupe_keeps_terminal_and_drops_aliases(self):
        predicates = [
            dict(name="terminal", kind="terminal"),
            dict(name="a", kind="jump", dim=0, thr=0.1),
            dict(name="b", kind="jump", dim=0, thr=0.1),
        ]
        masks = [
            [np.asarray([True, False])],
            [np.asarray([True, False])],
            [np.asarray([True, True])],
        ]
        kept, kept_masks = dedupe_predicates(predicates, masks, 0.8, 16)
        self.assertEqual(
            [item["name"] for item in kept], ["terminal", "b"]
        )
        self.assertEqual(len(kept_masks), 2)

    def test_entry_labels_window_and_censor(self):
        mask = [np.asarray([False, False, False, False, True, False])]
        entries = dict(
            row=np.asarray([1, 3, 4]),
            table=np.asarray([0, 0, 0]),
            remaining=np.asarray([4, 2, 1]),
        )
        event_step, duration, full = entry_labels(mask, entries, 3)
        self.assertEqual(event_step.tolist(), [3, 1, 0])
        self.assertEqual(duration.tolist(), [3, 1, 1])
        self.assertEqual(full.tolist(), [3, 1, -1])

    def test_person_period_labels(self):
        phi = np.zeros((3, 1))
        lags = np.zeros((3, 1))
        event_step = np.asarray([3, 0, 2])
        duration = np.asarray([3, 4, 2])
        features, labels = person_period(phi, lags, event_step, duration, 4)
        self.assertEqual(len(labels), 9)
        self.assertEqual(labels[-3:].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(labels[:4].tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_survival_curve_is_monotone_and_contrast_positive(self):
        phi = np.zeros((4, 1))
        lags = np.asarray([[0.0], [1.0], [0.0], [1.0]])
        coefficients = np.zeros(6)
        coefficients[1] = 2.0
        curve = survival_curve(coefficients, phi, lags, 4)
        self.assertTrue(np.all(np.diff(curve, axis=1) >= -1e-12))
        report, _delta = contrast_report(
            coefficients, phi, lags, np.zeros(4, dtype=int), 4, (4,),
            draws=20, seed=0,
        )
        self.assertGreater(report[4]["mean"], 0.0)

    def test_window_metrics_known_and_labels(self):
        curve = np.asarray([[0.1, 0.2], [0.9, 0.9], [0.5, 0.8]])
        event_step = np.asarray([0, 1, 2])
        remaining = np.asarray([5, 5, 0])
        result = window_metrics(curve, event_step, remaining, 2)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["base"], 2.0 / 3.0)
        self.assertEqual(result["auc"], 1.0)

    def test_fold_splits_round_robin(self):
        fold = fold_splits(np.asarray([1, 1, 2, 2, 3]), 2)
        self.assertEqual(fold.tolist(), [0, 0, 1, 1, 0])

    def test_event_transitions_counts_pairs(self):
        masks = [
            [np.asarray([True, False, False])],
            [np.asarray([False, True, False])],
        ]
        result = event_transitions(["a", "b"], masks, [
            dict(obs=np.zeros((3, 1)), actions=np.zeros(3), done=np.zeros(3)),
        ])
        self.assertEqual(result["a"]["b"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
