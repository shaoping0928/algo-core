"""Tests for the h6 autopsy failure taxonomy and arm comparison."""

import unittest

from h6_autopsy_probe import classify_row, compare_arms, summarize_autopsy


def _row(**overrides):
    row = dict(
        horizon=6, selected_true=-1, selected_value=7, regret=0,
        remaining=100, oracle_best_score_gap=0.5,
        oracle_best_feasible=True, risk_auc=None,
        risk_by_action={0: 0.1, 1: 0.2},
        true_by_action={0: 7, 1: 7},
        plan_followed=None,
    )
    row.update(overrides)
    return row


class ClassifyTest(unittest.TestCase):
    def test_window_safe_and_far_is_fine(self):
        self.assertEqual(classify_row(_row(), 6), "fine")

    def test_replan_death_when_safe_plan_dies_soon(self):
        row = _row(remaining=4)
        self.assertEqual(classify_row(row, 6), "replan_death")

    def test_post_window_death(self):
        row = _row(remaining=9)
        self.assertEqual(classify_row(row, 6), "post_window_death")

    def test_doomed_when_no_better_candidate(self):
        row = _row(selected_true=2, selected_value=2, regret=0)
        self.assertEqual(classify_row(row, 6), "in_window_doomed")

    def test_missed_subdivides_by_gap(self):
        row = _row(selected_true=2, selected_value=2, regret=2,
                   oracle_best_score_gap=0.01)
        self.assertEqual(classify_row(row, 6), "in_window_missed_near_tie")
        row = _row(selected_true=2, selected_value=2, regret=2,
                   oracle_best_score_gap=0.5)
        self.assertEqual(classify_row(row, 6), "in_window_missed_large_gap")
        row = _row(selected_true=2, selected_value=2, regret=2,
                   oracle_best_feasible=False)
        self.assertEqual(classify_row(row, 6), "in_window_missed_veto")


class SummarizeTest(unittest.TestCase):
    def test_counts_and_imminent_blind(self):
        rows = [
            _row(remaining=2),
            _row(remaining=50),
            _row(selected_true=1, selected_value=1, regret=0,
                 remaining=1),
        ]
        summary = summarize_autopsy(rows, 6)
        self.assertEqual(summary["decisions"], 3)
        self.assertEqual(summary["classes"]["fine"], 1)
        self.assertEqual(summary["classes"]["replan_death"], 1)
        self.assertEqual(summary["classes"]["in_window_doomed"], 1)
        self.assertEqual(summary["in_window_deaths"], 1)
        self.assertEqual(summary["imminent"], 2)
        self.assertEqual(summary["imminent_blind"], 1)

    def test_risk_contrast_counts_lower_risk_surviving(self):
        # action 0 has lower risk and the better true value: contrast correct.
        rows = [_row(risk_by_action={0: 0.1, 1: 0.9},
                     true_by_action={0: 7, 1: 2})]
        summary = summarize_autopsy(rows, 6)
        self.assertEqual(summary["risk_contrast"]["comparable"], 1)
        self.assertEqual(
            summary["risk_contrast"]["risk_contrast_correct"], 1
        )


class CompareTest(unittest.TestCase):
    def test_escape_and_both_dying(self):
        low = [
            # h6 plan dies inside its window at this state.
            _row(eval_seed=120, point=5, remaining=2,
                 selected_true=1, selected_value=1, regret=0),
            # h6 window-safe but the episode dies 3 frames later.
            _row(eval_seed=120, point=9, remaining=3),
        ]
        high = [
            # h8 finds a plan that survives its full window here.
            _row(eval_seed=120, point=5, remaining=2, selected_true=-1),
            # h8's plan also dies inside its own window.
            _row(eval_seed=120, point=9, remaining=3, selected_true=4),
        ]
        result = compare_arms(low, high)
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["high_escapes_low_death"], 1)
        self.assertEqual(result["both_dying"], 1)


if __name__ == "__main__":
    unittest.main()
