"""Tests for the sparse per-target rule fitter."""

import unittest

import numpy as np

from query_graph import Query, RuleBank, Slot
from sparse_rules import (
    fit_event_rule,
    fit_sparse_scopes,
    forward_select,
    make_rules,
    slot_names,
    standardize,
)


class SparseRulesTest(unittest.TestCase):
    def test_forward_select_recovers_a_sparse_scope(self):
        rng = np.random.RandomState(7)
        matrix = rng.normal(size=(300, 5))
        target = 1.0 + 2.0 * matrix[:, 1] - 0.5 * matrix[:, 3]
        selected, weights, r2 = forward_select(
            standardize(matrix), target, max_terms=4
        )
        self.assertEqual(set(selected), {1, 3})
        self.assertGreater(r2, 0.99)
        self.assertEqual(len(weights), 3)

    def test_fit_sparse_scopes_and_make_rules_predict_next_state(self):
        rng = np.random.RandomState(11)
        observations = rng.normal(size=(400, 3))
        histories = rng.normal(size=(400, 1))
        next_observations = np.zeros((400, 2))
        next_observations[:, 0] = (
            0.5 + 0.7 * observations[:, 0] - 0.2 * histories[:, 0]
        )
        next_observations[:, 1] = 2.0 + 0.3 * observations[:, 2]
        scopes = fit_sparse_scopes(
            observations, histories, next_observations, max_terms=4
        )
        self.assertEqual(len(scopes), 2)
        self.assertIn(0, scopes[0].inputs)
        self.assertEqual(set(scopes[1].inputs), {2})
        rules = make_rules(scopes, observations.shape[1], histories.shape[1])
        bank = RuleBank(rules)
        row = 17
        source = {
            Slot("x0", 0): observations[row, 0],
            Slot("x1", 0): observations[row, 1],
            Slot("x2", 0): observations[row, 2],
            Slot("a0", 0): histories[row, 0],
        }
        result = bank.query(Query(
            target_slots=(Slot("y0", 1), Slot("y1", 1)),
            source=source,
        ))
        self.assertAlmostEqual(
            float(result.values[Slot("y0", 1)]),
            float(next_observations[row, 0]),
            places=6,
        )
        self.assertAlmostEqual(
            float(result.values[Slot("y1", 1)]),
            float(next_observations[row, 1]),
            places=6,
        )

    def test_slot_names_keep_state_and_action_anonymous(self):
        self.assertEqual(
            slot_names(2, 3), ("x0", "x1", "a0", "a1", "a2")
        )

    def test_fit_event_rule_recovers_a_high_side_event(self):
        rng = np.random.RandomState(3)
        observations = rng.normal(size=(400, 4))
        labels = (observations[:, 1] > 0.4).astype(int)
        fitted = fit_event_rule(
            observations, labels, Slot("event", 1), min_support=20,
        )
        self.assertIsNotNone(fitted)
        rule, report = fitted
        self.assertEqual(report["column"], 1)
        self.assertEqual(report["direction"], 1)
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("event", 1),),
            source={Slot("x1", 0): 1.0},
        ))
        self.assertEqual(result.values[Slot("event", 1)], 1)

    def test_fit_event_rule_recovers_a_low_side_event(self):
        rng = np.random.RandomState(4)
        observations = rng.normal(size=(400, 4))
        labels = (observations[:, 2] < -0.3).astype(int)
        fitted = fit_event_rule(
            observations, labels, Slot("event", 1), min_support=20,
        )
        self.assertIsNotNone(fitted)
        rule, report = fitted
        self.assertEqual(report["column"], 2)
        self.assertEqual(report["direction"], -1)
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("event", 1),),
            source={Slot("x2", 0): -1.0},
        ))
        self.assertEqual(result.values[Slot("event", 1)], 1)

    def test_fit_event_rule_abstains_when_labels_are_pure(self):
        observations = np.zeros((100, 2))
        fitted = fit_event_rule(
            observations, np.zeros(100, dtype=int), Slot("event", 1),
        )
        self.assertIsNone(fitted)


if __name__ == "__main__":
    unittest.main()
