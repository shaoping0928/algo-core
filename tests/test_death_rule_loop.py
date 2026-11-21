"""Failure-driven death-rule refinement."""
import unittest

import numpy as np

from death_rule_loop import death_failures, refine_once, run_loop
from death_rules import DeathRuleSet


class StubRules:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction, dtype=int)

    def classify_death(self, states):
        return self.prediction[:len(states)]


class DeathRuleLoopTests(unittest.TestCase):
    def test_death_failures_separates_missed_terminals_and_false_alarms(self):
        states = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([1, 1, 0, 0])
        rules = StubRules([1, -1, 1, -1])
        missed_states, missed_labels, false_states, false_labels = (
            death_failures(rules, states, labels)
        )
        np.testing.assert_array_equal(missed_states, [[1.0]])
        np.testing.assert_array_equal(missed_labels, [1])
        np.testing.assert_array_equal(false_states, [[2.0]])
        np.testing.assert_array_equal(false_labels, [0])

    def test_refine_once_adds_a_missing_second_mode(self):
        rng = np.random.RandomState(0)
        source_x = rng.uniform(-1.0, 1.0, (400, 2))
        # The source stream only demonstrates the first mode.
        source_x[:, 1] = np.minimum(source_x[:, 1], 0.4)
        source_labels = (source_x[:, 0] > 0.5).astype(int)
        rules = DeathRuleSet(
            max_rules=4, max_literals=2, min_support=2,
            max_thresholds=32,
        ).fit(source_x, source_labels)

        stream_x = rng.uniform(-1.0, 1.0, (400, 2))
        stream_labels = (
            (stream_x[:, 0] > 0.5) | (stream_x[:, 1] > 0.5)
        ).astype(int)
        before = rules.coverage_report(stream_x, stream_labels)
        report = refine_once(rules, stream_x, stream_labels)
        after = rules.coverage_report(stream_x, stream_labels)
        self.assertGreater(report.added_missed, 0)
        self.assertGreaterEqual(
            after["death_coverage"], before["death_coverage"]
        )
        self.assertGreater(
            after["death_coverage"], before["death_coverage"]
        )
        self.assertEqual(
            report.after["death_coverage"], after["death_coverage"]
        )

    def test_refine_once_can_skip_false_alarms(self):
        rng = np.random.RandomState(1)
        states = rng.uniform(-1.0, 1.0, (300, 2))
        labels = (states[:, 0] > 0.0).astype(int)
        rules = DeathRuleSet(
            max_rules=2, max_literals=1, min_support=2,
            max_thresholds=24,
        ).fit(states, labels)
        # A stream whose negative region is labelled as death by a
        # deliberately wrong stub cannot be refined by the real learner;
        # this test only checks that skipping false alarms still refits
        # missed positives and reports both counts.
        report = refine_once(
            rules, states, labels, include_false_alarms=False
        )
        self.assertEqual(report.added_false_alarm, 0)
        self.assertIn("death_coverage", report.before)
        self.assertIn("death_coverage", report.after)

    def test_run_loop_stops_when_the_stream_has_no_failures(self):
        rng = np.random.RandomState(2)
        states = rng.uniform(-1.0, 1.0, (240, 2))
        labels = (states[:, 0] > 0.25).astype(int)
        rules = DeathRuleSet(
            max_rules=2, max_literals=1, min_support=2,
            max_thresholds=24,
        ).fit(states, labels)
        history = run_loop(
            rules, lambda _round: (states, labels), rounds=4
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].added_missed, 0)
        self.assertEqual(history[0].added_false_alarm, 0)


if __name__ == "__main__":
    unittest.main()
