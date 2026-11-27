"""Tests for episode-level validation of stage strategy proposals."""
import unittest

from stage_strategy_credit import StageStrategyCredit


class StageStrategyCreditTests(unittest.TestCase):
    def setUp(self):
        self.credit = StageStrategyCredit()
        self.context = ((("tree", "source"),),)
        self.target = (("tree", "target"),)

    def test_credit_requires_distinct_episode_support(self):
        self.credit.record(self.context, self.target, 10, 1.0)
        self.credit.record(self.context, self.target, 10, 1.0)
        estimate = self.credit.estimate(self.context, self.target)
        self.assertEqual(estimate["episode_support"], 1)
        self.assertEqual(estimate["status"], "unknown")
        self.assertEqual(estimate["priority_multiplier"], 1.0)

    def test_repeated_realized_matches_raise_rank_priority(self):
        self.credit.record(self.context, self.target, 10, 1.0)
        self.credit.record(self.context, self.target, 11, 1.0)
        estimate = self.credit.estimate(self.context, self.target)
        self.assertEqual(estimate["status"], "supported")
        self.assertEqual(estimate["mean_similarity"], 1.0)
        self.assertGreater(estimate["priority_multiplier"], 1.0)
        self.assertLessEqual(estimate["priority_multiplier"], 1.25)

    def test_repeated_realized_mismatches_lower_rank_priority(self):
        self.credit.record(self.context, self.target, 10, 0.0)
        self.credit.record(self.context, self.target, 11, 0.0)
        estimate = self.credit.estimate(self.context, self.target)
        self.assertEqual(estimate["mean_similarity"], 0.0)
        self.assertLess(estimate["priority_multiplier"], 1.0)

    def test_unvalidated_candidate_keeps_neutral_priority(self):
        self.credit.record(self.context, self.target, 10, 1.0)
        self.credit.record(self.context, self.target, 11, 0.0)
        estimate = self.credit.estimate(self.context, self.target)
        self.assertEqual(estimate["priority_multiplier"], 1.0)


if __name__ == "__main__":
    unittest.main()
