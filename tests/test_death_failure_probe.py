"""Helpers for the failure-selected refinement probe."""
import unittest

import numpy as np

from death_failure_probe import add_all_terminals, add_random, test_report


class StubRules:
    def __init__(self, report=None):
        self.report = report or {}
        self.added = []

    def coverage_report(self, states, labels):
        return self.report

    def add_examples(self, states, labels):
        self.added.append((len(states), int(np.sum(labels))))


class DeathFailureProbeTests(unittest.TestCase):
    def test_test_report_uses_the_death_side(self):
        rules = StubRules(dict(
            total=100, positive=10, death_covered_positive=8,
            negative=90, death_covered_negative=3,
        ))
        report = test_report(
            rules, np.zeros((100, 2)), np.zeros(100, dtype=int)
        )
        self.assertEqual(report["terminals"], 10)
        self.assertEqual(report["covered"], 8)
        self.assertEqual(report["unknown_terminal"], 2)
        self.assertAlmostEqual(report["coverage"], 0.8)
        self.assertEqual(report["false_alarm"], 3)
        self.assertAlmostEqual(report["false_alarm_rate"], 3.0 / 90.0)

    def test_add_random_adds_the_requested_number(self):
        states = np.arange(20, dtype=float).reshape(10, 2)
        labels = np.asarray([1, 0] * 5)
        rules = StubRules()
        added = add_random(
            rules, states, labels, 3, np.random.RandomState(0)
        )
        self.assertEqual(added, 3)
        self.assertEqual(len(rules.added), 1)
        self.assertEqual(rules.added[0][0], 3)

    def test_add_random_is_a_noop_for_zero(self):
        rules = StubRules()
        added = add_random(
            rules, np.zeros((4, 2)), np.zeros(4, dtype=int), 0,
            np.random.RandomState(0),
        )
        self.assertEqual(added, 0)
        self.assertEqual(rules.added, [])

    def test_add_all_terminals_keeps_only_positives(self):
        states = np.arange(8, dtype=float).reshape(4, 2)
        labels = np.asarray([1, 0, 1, 0])
        rules = StubRules()
        added = add_all_terminals(rules, states, labels)
        self.assertEqual(added, 2)
        self.assertEqual(rules.added, [(2, 2)])


if __name__ == "__main__":
    unittest.main()
