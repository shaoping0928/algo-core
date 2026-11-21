"""Generic metric helpers for the candidate-plan quality probe."""
import unittest

import numpy as np

from planner_rollout_quality import (
    action_best,
    action_rank,
    oracle_plan,
    plan_death_counts,
    status_callback,
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
    def classify_death(self, states):
        return np.where(states[:, 0] > 0.0, 1, -1)

    def classify(self, states):
        return np.where(states[:, 0] > 0.0, 1, 0)


class PlannerRolloutQualityTests(unittest.TestCase):
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
