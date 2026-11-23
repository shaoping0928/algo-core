"""Tests for the B0 decision-miss diagnostic."""

import unittest

import numpy as np

from decision_b0_probe import (
    classify_miss, local_support, ood_reference, summarize, training_index,
)


def _rows(episodes=4, frames=12):
    rows = []
    for episode in range(episodes):
        for t in range(frames):
            obs = np.asarray([
                float(episode) * 10.0 + 0.1 * t,
                float(t % 3) + 0.2 * episode,
            ])
            action = int((t + episode) % 2 == 0)
            history = np.asarray([action, int(t % 2), 0, 0], dtype=float)
            rows.append(dict(
                obs=obs, history=history, episode=episode,
            ))
    return rows


class DecisionB0ProbeTest(unittest.TestCase):
    def test_training_index_reads_action_and_pending(self):
        index = training_index(
            _rows(), 2, 3, np.asarray([1.0, 1.0, 1.0, 1.0])
        )
        self.assertEqual(len(index["states"]), 48)
        self.assertEqual(index["actions"][0], 1)
        self.assertEqual(index["actions"][1], 0)
        self.assertEqual(index["pending"][0, 0], 0)

    def test_support_counts_both_actions_and_episodes(self):
        index = training_index(
            _rows(), 2, 3, np.asarray([1.0, 1.0, 1.0, 1.0]),
            neighbours=8,
        )
        support = local_support(
            index, index["states"][0], np.asarray([0, 1, 0]), neighbours=8,
        )
        self.assertEqual(support["neighbours"], 8)
        self.assertEqual(
            support["rows_0"] + support["rows_1"], support["neighbours"],
        )
        self.assertGreaterEqual(support["episodes_0"], 1)
        self.assertGreaterEqual(support["episodes_1"], 1)

    def test_ood_reference_flags_far_endpoint(self):
        index = training_index(
            _rows(), 2, 3, np.asarray([1.0, 1.0, 1.0, 1.0])
        )
        reference = ood_reference(index)
        self.assertGreater(reference["threshold"], 0.0)
        near = index["states"][0]
        self.assertLess(
            float(np.linalg.norm(
                index["states"] - near, axis=1
            ).min()), reference["threshold"],
        )

    def test_classify_miss_tree(self):
        base = dict(
            same_first_action=False,
            true_action_contrast=2,
            support=dict(
                rows_0=8, rows_1=8, episodes_0=4, episodes_1=4,
            ),
            endpoint_ood=dict(selected=False, oracle_best=False),
            first_action_gap=0.01,
        )
        self.assertEqual(classify_miss(dict(base)), "ranking")
        self.assertEqual(
            classify_miss(dict(base, first_action_gap=0.2)),
            "fine_score_or_challenge",
        )
        self.assertEqual(
            classify_miss(dict(
                base,
                support=dict(
                    rows_0=0, rows_1=8, episodes_0=0, episodes_1=4,
                ),
            )),
            "coverage_support",
        )
        self.assertEqual(
            classify_miss(dict(
                base, endpoint_ood=dict(selected=True, oracle_best=False),
            )),
            "coverage_endpoint",
        )
        self.assertEqual(
            classify_miss(dict(base, same_first_action=True)),
            "suffix_only",
        )
        self.assertEqual(
            classify_miss(dict(base, true_action_contrast=0)),
            "suffix_within_action",
        )

    def test_summarize_counts_relevant_misses(self):
        rows = [
            dict(
                decision_relevant=True, same_first_action=False,
                first_action_gap=0.01, support_admissible=True,
                true_action_contrast=2,
                endpoint_ood=dict(selected=False, oracle_best=False),
                candidate_ood_rate=0.1, candidate_ood_rate_high=0.0,
                candidate_percentile_median=0.5,
                **{"class": "ranking"},
            ),
            dict(
                decision_relevant=True, same_first_action=True,
                first_action_gap=None, support_admissible=False,
                true_action_contrast=0,
                endpoint_ood=dict(selected=False, oracle_best=False),
                candidate_ood_rate=0.2, candidate_ood_rate_high=0.1,
                candidate_percentile_median=0.6,
                **{"class": "suffix_only"},
            ),
            dict(
                decision_relevant=False, same_first_action=False,
                first_action_gap=0.02, support_admissible=True,
                true_action_contrast=0,
                endpoint_ood=dict(selected=False, oracle_best=False),
                candidate_ood_rate=0.0, candidate_ood_rate_high=0.0,
                candidate_percentile_median=0.2,
                **{"class": "not_relevant"},
            ),
        ]
        summary = summarize(rows, 3, 8)
        self.assertEqual(summary["decisions"], 3)
        self.assertEqual(summary["misses"], 2)
        self.assertEqual(summary["action_different_misses"], 1)
        self.assertEqual(summary["same_first_action_misses"], 1)
        self.assertEqual(summary["gap_count"], 1)
        self.assertAlmostEqual(summary["gap_median"], 0.01)


if __name__ == "__main__":
    unittest.main()
