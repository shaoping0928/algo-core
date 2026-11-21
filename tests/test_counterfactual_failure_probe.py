"""Helpers for the counterfactual-failure probe."""
import unittest

import numpy as np

from counterfactual_failure_probe import (
    add_positives,
    dedupe_terminals,
    sample_points,
    terminal_report,
)


class StubRules:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction, dtype=int)
        self.added = []

    def classify_death(self, states):
        return self.prediction[:len(states)]

    def add_examples(self, states, labels):
        self.added.append(
            (len(states), int(np.sum(labels)))
        )


class CounterfactualFailureProbeTests(unittest.TestCase):
    def test_sample_points_includes_the_tail(self):
        points = sample_points(frames=100, branches=4, tail=3)
        self.assertEqual(points, sorted(set(points)))
        self.assertIn(99, points)
        self.assertIn(98, points)
        self.assertIn(97, points)
        self.assertGreaterEqual(min(points), 2)
        self.assertLessEqual(max(points), 99)

    def test_dedupe_terminals_respects_limit(self):
        states = np.asarray([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ])
        unique = dedupe_terminals(states)
        self.assertEqual(unique.shape[0], 4)
        limited = dedupe_terminals(states, limit=2)
        self.assertEqual(limited.shape[0], 2)
        np.testing.assert_allclose(limited[0], [0.0, 0.0])
        np.testing.assert_allclose(limited[-1], [3.0, 3.0])

    def test_terminal_report_uses_death_side(self):
        rules = StubRules([1, -1, 1])
        report = terminal_report(rules, np.zeros((3, 2)))
        self.assertEqual(report["terminals"], 3)
        self.assertEqual(report["covered"], 2)
        self.assertEqual(report["unknown"], 1)
        self.assertAlmostEqual(report["coverage"], 2.0 / 3.0)

    def test_add_positives_caps_and_labels(self):
        rules = StubRules([])
        added = add_positives(
            rules, np.zeros((5, 2)), 3, np.random.RandomState(0)
        )
        self.assertEqual(added, 3)
        self.assertEqual(rules.added, [(3, 3)])
        added = add_positives(
            rules, np.zeros((2, 2)), 5, np.random.RandomState(0)
        )
        self.assertEqual(added, 2)
        self.assertEqual(rules.added[-1], (2, 2))


if __name__ == "__main__":
    unittest.main()
