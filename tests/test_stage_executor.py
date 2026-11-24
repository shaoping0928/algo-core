"""Tests for the stage-level option executor's value and trace helpers."""
import unittest

import numpy as np

from stage_executor import expand_trace, option_values


class Leaf:
    def __init__(self, option, probability, feedback, outcome,
                 elapsed=1.0):
        self.options = (option,)
        self.probability = float(probability)
        self.feedback_mean = tuple(feedback)
        self.outcome = outcome
        self.elapsed_mean = float(elapsed)


class Result:
    def __init__(self, leaves):
        self.leaves = tuple(leaves)


class StageExecutorHelperTests(unittest.TestCase):
    def test_run_length_trace_expands_in_order(self):
        self.assertEqual(
            expand_trace(((0, 2), (1, 1), (0, 3))),
            [0, 0, 1, 0, 0, 0],
        )

    def test_option_value_prefers_feedback_and_penalises_terminal(self):
        option_a = ((0, 2),)
        option_b = ((1, 1),)
        result = Result([
            Leaf(option_a, 1.0, (0.0, 0.0), "terminal"),
            Leaf(option_b, 1.0, (2.0, 0.0), "open"),
        ])
        rows = option_values(result, terminal_weight=1.0)
        self.assertEqual(rows[0][2], option_b)
        self.assertAlmostEqual(rows[0][0], 2.0)
        self.assertAlmostEqual(rows[1][0], -1.0)

    def test_duration_weight_penalises_long_options(self):
        option = ((0, 1),)
        result = Result([Leaf(option, 1.0, (0.0, 0.0), "open", elapsed=10.0)])
        rows = option_values(result, terminal_weight=0.0,
                             duration_weight=0.1)
        self.assertAlmostEqual(rows[0][0], -1.0)

    def test_probability_weighted_mixture(self):
        option = ((0, 1),)
        result = Result([
            Leaf(option, 0.5, (4.0, 0.0), "open"),
            Leaf(option, 0.5, (0.0, 0.0), "terminal"),
        ])
        rows = option_values(result, terminal_weight=1.0)
        self.assertAlmostEqual(rows[0][0], 0.5 * 4.0 - 0.5 * 1.0)

    def test_no_leaves_yields_no_options(self):
        self.assertEqual(option_values(Result([])), [])


if __name__ == "__main__":
    unittest.main()
