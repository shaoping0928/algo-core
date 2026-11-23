"""Tests for the decision-side endpoint-event veto probe."""

import unittest

import numpy as np

from decision_event_probe import event_quality, select_arm, summarize


class DecisionEventProbeTest(unittest.TestCase):
    def test_baseline_and_veto_select_different_candidates(self):
        scores = np.asarray([[0.1], [0.2], [0.3]])
        mask = np.asarray([True, True, True])
        event = np.asarray([1, 0, 0])
        baseline, _ = select_arm(scores, mask, event, "baseline")
        veto, _ = select_arm(scores, mask, event, "veto-death")
        self.assertEqual(baseline, 0)
        self.assertEqual(veto, 1)

    def test_unknown_veto_and_fallback(self):
        scores = np.asarray([[0.1], [0.2]])
        mask = np.asarray([True, True])
        event = np.asarray([-1, 0])
        unknown, _ = select_arm(
            scores, mask, event, "veto-death-unknown"
        )
        self.assertEqual(unknown, 1)
        fallback, used_fallback = select_arm(
            scores, mask, np.asarray([1, 1]), "veto-death"
        )
        self.assertEqual(fallback, 0)
        self.assertTrue(used_fallback)

    def test_event_quality_counts_unknowns(self):
        result = event_quality(
            np.asarray([1, 0, -1, 1]),
            np.asarray([True, False, True, False]),
        )
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["false_negative"], 0)
        self.assertEqual(result["unknown"], 1)

    def test_summary_reports_improvement_and_vetoed_oracle(self):
        records = [dict(
            baseline=dict(action=0, value=4, regret=1, oracle_best=True,
                          survives=False, fallback=False),
            **{
                "veto-death": dict(
                    action=1, value=5, regret=0, oracle_best=True,
                    survives=True, fallback=False,
                ),
                "veto-death-unknown": dict(
                    action=1, value=5, regret=0, oracle_best=True,
                    survives=True, fallback=False,
                ),
            },
            event=dict(
                candidates=2, unknown=0, true_positive=1,
                false_positive=0, false_negative=0, true_negative=1,
            ),
        )]
        result = summarize(records, horizon=5)
        self.assertEqual(result["decisions"], 1)
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["worsened"], 0)
        self.assertEqual(result["vetoed_oracle_best"], 1)
        self.assertEqual(result["veto-death"]["decision_relevant_misses"], 0)
        self.assertEqual(result["baseline"]["decision_relevant_misses"], 1)


if __name__ == "__main__":
    unittest.main()
