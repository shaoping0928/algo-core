"""Tests for the P1-B event-level transition / early-warning probe."""

import unittest

import numpy as np

from event_transition_probe import (
    history_features, lead_analysis, modal_successor,
    hazard_next_top1, next_event_sequences,
)


def _table(episode, obs, actions, done):
    return dict(
        episode=episode,
        obs=np.asarray(obs, dtype=float),
        actions=np.asarray(actions, dtype=int),
        done=np.asarray(done, dtype=bool),
    )


class EventTransitionProbeTest(unittest.TestCase):
    def test_lead_analysis_counts_precursors_within_window(self):
        level_mask = [np.asarray([True, False, False, False])]
        terminal = [_table(0, np.zeros((4, 1)), [0] * 4,
                           [False, False, False, True])]
        report = lead_analysis(
            [np.zeros((4, 1), dtype=bool), level_mask],
            ["terminal", "x0_sign"], terminal, w_pre=3,
        )
        row = report["x0_sign"]
        self.assertEqual(row["n_terminal"], 1)
        self.assertEqual(row["coverage"], 1.0)
        self.assertEqual(row["median_lead"], 3)

    def test_history_features_are_causal(self):
        # The level fires at rows 1 and 5; an entry at row 5 observes the
        # row-5 event (same row is known) and nothing after it.
        masks = [[np.asarray([False, True, False, False, False, True])]]
        entries = dict(
            row=np.asarray([5]),
            table=np.asarray([0]),
            remaining=np.asarray([0]),
        )
        features = history_features(masks, ["x0_sign"], entries)
        self.assertAlmostEqual(features[0, 0], 0.0)      # since = 0 / cap
        self.assertAlmostEqual(features[0, 1], 2.0 / 16.0)
        self.assertAlmostEqual(features[0, 2], 1.0)
        masks_far = [[np.asarray([True] + [False] * 5)]]
        features_far = history_features(masks_far, ["x0_sign"], entries)
        self.assertAlmostEqual(features_far[0, 0], 5.0 / 32.0)

    def test_next_event_sequences_merge_same_row_and_add_terminal(self):
        table = _table(0, np.zeros((4, 1)), [0] * 4,
                       [False, False, False, True])
        masks = [
            [np.asarray([False, False, False, True])],   # terminal
            [np.asarray([False, True, True, False])],    # level A
            [np.asarray([False, True, False, False])],   # level B
        ]
        sequences = next_event_sequences(masks, ["t", "a", "b"], [table])
        self.assertEqual(sequences[0], [(1, 1), (2, 1), (3, 0)])

    def test_modal_successor_counts_majority(self):
        sequences = [
            [(0, 1), (2, 0)],
            [(0, 1), (2, 1)],
            [(0, 1), (2, 1)],
        ]
        result = modal_successor(sequences, ["t", "a"])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["accuracy"], 2.0 / 3.0)
        self.assertEqual(result["per_source"]["a"]["modal"], "a")

    def test_hazard_next_top1_scores_shared_entries(self):
        curves = np.asarray([
            [[0.9], [0.1]],   # level 0 (terminal)
            [[0.1], [0.9]],   # level 1
        ])
        sequences = [[(0, 1), (3, 0)], [(1, 1)]]
        entries = dict(table=np.asarray([0, 1]), row=np.asarray([2, 0]))
        result = hazard_next_top1(curves, sequences, ["t", "a"], entries, 1)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["chance"], 0.5)
        self.assertEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
