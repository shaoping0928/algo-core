"""Generic metric helpers for the candidate-plan quality probe."""
import unittest
from types import SimpleNamespace

import numpy as np

from planner_rollout_quality import (
    action_best,
    action_rank,
    action_score_gap,
    near_status_callback,
    oracle_plan,
    plan_death_counts,
    plan_score_gap,
    probe_config,
    status_callback,
    summarize_margins,
)


class FakeEnv:
    def __init__(self):
        self.t = 0
        self.dead = False

    def step(self, action):
        self.t += 1
        if int(action) == 1:
            self.dead = True

    def obs(self):
        return np.asarray([float(self.t)])


class FakeRules:
    def classify_death(self, states, include_margin=False):
        threshold = -0.5 if include_margin else 0.0
        return np.where(states[:, 0] > threshold, 1, -1)

    def classify(self, states):
        return np.where(states[:, 0] > 0.0, 1, 0)

    def classify_death_status(self, states):
        return np.where(
            states[:, 0] > 0.0,
            1,
            np.where(states[:, 0] > -0.5, 2, -1),
        )


class PlannerRolloutQualityTests(unittest.TestCase):
    def test_probe_config_keeps_seed_and_margin(self):
        config = probe_config(SimpleNamespace(seed=101, margin=0.03))
        self.assertEqual(config["seed"], 101)
        self.assertAlmostEqual(
            config["death_rule_boundary_margin"], 0.03
        )
        self.assertEqual(config["steps"], 1500)

    def test_plan_death_counts(self):
        counts = plan_death_counts(
            predicted=[2, -1, 1, -1], truth=[0, 2, 1, -1]
        )
        self.assertEqual(counts["total"], 4)
        self.assertEqual(counts["true_death"], 3)
        self.assertEqual(counts["predicted_death"], 2)
        self.assertEqual(counts["tp"], 2)
        self.assertEqual(counts["fp"], 0)
        self.assertEqual(counts["fn"], 1)
        self.assertEqual(counts["tn"], 1)
        self.assertEqual(counts["exact"], 1)
        self.assertEqual(counts["within_one"], 1)
        self.assertEqual(counts["false_negative"], 1)
        self.assertEqual(counts["false_positive"], 0)

    def test_action_best_prefers_feasible_survival(self):
        sequences = np.asarray([[0], [0], [1], [1]], dtype=float)
        first = [3, -1, 1, 2]
        feasible = np.asarray([False, True, True, False])
        best = action_best(sequences, first, 4, feasible=feasible)
        self.assertEqual(best, {0: 5, 1: 1})
        unfiltered = action_best(sequences, first, 4)
        self.assertEqual(unfiltered, {0: 5, 1: 2})

    def test_action_rank_uses_feasible_scores(self):
        sequences = np.asarray([[0], [0], [1], [1]], dtype=float)
        scores = np.asarray([[0.5], [0.1], [0.2], [0.05]])
        feasible = np.asarray([False, True, True, False])
        ranked = action_rank(sequences, scores, [0, 1], feasible=feasible)
        self.assertEqual(ranked, {0: 0.1, 1: 0.2})
        unfiltered = action_rank(sequences, scores, [0, 1])
        self.assertEqual(unfiltered, {0: 0.1, 1: 0.05})

    def test_status_callback_keeps_death_side_tristate(self):
        rules = FakeRules()
        states = np.asarray([[-1.0], [1.0]])
        death_side = status_callback(rules, 1)(states, None, 0)
        np.testing.assert_array_equal(death_side, [-1, 1])
        full = status_callback(rules, 1, full=True)(states, None, 0)
        np.testing.assert_array_equal(full, [0, 1])
        margin_rules = FakeRules()
        margin_states = np.asarray([[-0.25], [0.25]])
        envelope = status_callback(
            margin_rules, 1, include_margin=True
        )(margin_states, None, 0)
        np.testing.assert_array_equal(envelope, [1, 1])

    @staticmethod
    def margin_trace():
        return dict(
            scores=np.asarray([[1.0], [3.0], [2.0], [4.0]]),
            feasible=np.asarray([True, True, True, True]),
            finite=np.ones(4, dtype=bool),
        )

    def test_action_score_gap_compares_best_plan_per_action(self):
        trace = self.margin_trace()
        sequences = np.asarray([[0], [0], [1], [1]])
        gap, best = action_score_gap(trace, sequences)
        self.assertAlmostEqual(gap, 1.0)
        self.assertEqual(best, {0: 1.0, 1: 2.0})
        trace["feasible"] = np.asarray([True, True, False, False])
        gap, best = action_score_gap(trace, sequences)
        self.assertIsNone(gap)
        self.assertEqual(best, {0: 1.0})

    def test_plan_score_gap_uses_the_lexicographic_top_two(self):
        trace = dict(
            scores=np.asarray([[2.0, 9.0], [1.0, 5.0], [1.0, 3.0]]),
            feasible=np.ones(3, dtype=bool),
            finite=np.ones(3, dtype=bool),
        )
        self.assertAlmostEqual(plan_score_gap(trace), 0.0)
        trace["scores"] = np.asarray([[2.0], [1.0], [3.0]])
        self.assertAlmostEqual(plan_score_gap(trace), 1.0)

    def test_summarize_margins_stratifies_selected_misses(self):
        rows = [
            dict(action_margin=0.05, selected_miss=True, regret=3.0,
                 selected_value=2.0, oracle_best_value=5.0, horizon=8,
                 oracle_best_feasible=True, oracle_best_score_gap=0.5,
                 oracle_best_rank=3),
            dict(action_margin=0.10, selected_miss=True, regret=2.0,
                 selected_value=3.0, oracle_best_value=5.0, horizon=8,
                 oracle_best_feasible=True, oracle_best_score_gap=0.5,
                 oracle_best_rank=3),
            dict(action_margin=0.20, selected_miss=False, regret=0.0,
                 selected_value=5.0, oracle_best_value=5.0, horizon=8,
                 oracle_best_feasible=True, oracle_best_score_gap=0.0,
                 oracle_best_rank=0),
            dict(action_margin=0.50, selected_miss=False, regret=0.0,
                 selected_value=5.0, oracle_best_value=5.0, horizon=8,
                 oracle_best_feasible=True, oracle_best_score_gap=0.0,
                 oracle_best_rank=0),
            dict(action_margin=None, selected_miss=False, regret=0.0,
                 selected_value=1.0, oracle_best_value=5.0, horizon=8,
                 oracle_best_feasible=True, oracle_best_score_gap=0.0,
                 oracle_best_rank=0),
        ]
        summary = summarize_margins(rows)
        self.assertEqual(summary["decisions"], 5)
        self.assertEqual(summary["usable"], 4)
        self.assertEqual(summary["one_action_only"], 1)
        self.assertEqual(summary["selected_misses"], 2)
        self.assertEqual(summary["decision_relevant_misses"], 2)
        self.assertEqual(summary["unavoidable_misses"], 0)
        self.assertEqual(summary["regret_nonzero"], 2)
        self.assertAlmostEqual(summary["regret_max"], 3.0)
        self.assertEqual(summary["oracle_best_survives"], 0)
        self.assertEqual(summary["selected_survives"], 0)
        self.assertEqual(summary["decision_relevant_oracle_feasible"], 2)
        self.assertEqual(summary["decision_relevant_score_misrank"], 2)
        self.assertEqual(summary["decision_relevant_death_veto"], 0)
        self.assertAlmostEqual(
            summary["mean_score_gap_decision_relevant"], 0.5
        )
        self.assertEqual(summary["oracle_best_feasible"], 5)
        self.assertEqual(summary["oracle_best_ranked_first"], 3)
        self.assertAlmostEqual(summary["miss_margin_mean"], 0.075)
        self.assertAlmostEqual(summary["ok_margin_mean"], 0.35)
        self.assertEqual(len(summary["buckets"]), 4)
        self.assertEqual(
            sum(bucket["misses"] for bucket in summary["buckets"]), 2
        )
        self.assertEqual(summary["buckets"][0]["miss_rate"], 1.0)

    def test_near_status_callback_keeps_three_levels(self):
        rules = FakeRules()
        states = np.asarray([[-1.0], [-0.25], [0.25]])
        status = near_status_callback(rules, 1)(states, None, 0)
        np.testing.assert_array_equal(status, [-1, 2, 1])

    def test_oracle_plan_returns_first_death_and_final_state(self):
        env = FakeEnv()
        first, state, dead = oracle_plan(env, [0, 1, 0])
        self.assertEqual(first, 1)
        self.assertTrue(dead)
        np.testing.assert_array_equal(state, [2.0])
        self.assertEqual(env.t, 0)
        self.assertFalse(env.dead)


if __name__ == "__main__":
    unittest.main()
