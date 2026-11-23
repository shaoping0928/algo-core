"""Tests for the learned component-sign derivation rules."""

import unittest

import numpy as np

from component_derivation import (
    ComponentDerivationBank,
    DifferenceSignRule,
    ThresholdSignRule,
    difference_states,
    evaluate_rule,
    fit_difference_rule,
    fit_threshold_rule,
    make_difference_rule,
    make_threshold_rule,
    threshold_states,
)
from query_graph import Query, RuleBank, Slot, UNKNOWN


class ComponentDerivationTest(unittest.TestCase):
    def test_threshold_rule_recovers_vy_to_dy_sign(self):
        scores = np.asarray([-2.0, -0.5, -0.1, 0.1, 0.5, 2.0])
        labels = np.asarray([-1, -1, -1, 1, 1, 1])
        fitted = fit_threshold_rule(scores, labels)
        self.assertGreaterEqual(fitted.confidence, 1.0)
        rule = make_threshold_rule(
            fitted, Slot("vy", 0), Slot("dy_sign", 1)
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("dy_sign", 1),),
            source={Slot("vy", 0): 0.7},
        ))
        self.assertEqual(result.values[Slot("dy_sign", 1)], 1)

    def test_difference_rule_recovers_sign_of_delta(self):
        previous = np.asarray([0.0, 0.0, 0.0, 0.0])
        current = np.asarray([-1.0, -0.1, 0.1, 1.0])
        labels = np.asarray([-1, -1, 1, 1])
        fitted = fit_difference_rule(current, previous, labels)
        self.assertGreaterEqual(fitted.confidence, 1.0)
        rule = make_difference_rule(
            fitted, Slot("y", 0), Slot("y", -1), Slot("dy_sign", 1)
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("dy_sign", 1),),
            source={Slot("y", 0): 0.4, Slot("y", -1): 0.1},
        ))
        self.assertEqual(result.values[Slot("dy_sign", 1)], 1)

    def test_missing_input_yields_unknown(self):
        rule = ThresholdSignRule(
            "sign", Slot("vy", 0), threshold=0.0, support=10,
            confidence=1.0,
        )
        graph_rule = make_threshold_rule(
            rule, Slot("vy", 0), Slot("dy_sign", 1)
        )
        bank = RuleBank([graph_rule])
        result = bank.query(Query(
            target_slots=(Slot("dy_sign", 1),), source={},
        ))
        self.assertIs(result.values[Slot("dy_sign", 1)], UNKNOWN)
        accuracy, unknown = evaluate_rule(rule, [{}], [1])
        self.assertTrue(np.isnan(accuracy))
        self.assertEqual(unknown, 1.0)

    def test_unknown_rate_and_accuracy_are_counted_separately(self):
        rule = ThresholdSignRule(
            "sign", Slot("vy", 0), threshold=0.0, support=10,
            confidence=1.0,
        )
        states = threshold_states([1.0, -1.0, 0.0], Slot("vy", 0))
        accuracy, unknown = evaluate_rule(rule, states, [1, -1, 0])
        self.assertAlmostEqual(accuracy, 1.0)
        self.assertEqual(unknown, 0.0)
        states_with_missing = states + [{}]
        accuracy, unknown = evaluate_rule(
            rule, states_with_missing, [1, -1, 0, 1]
        )
        self.assertAlmostEqual(accuracy, 1.0)
        self.assertAlmostEqual(unknown, 0.25)

    def test_difference_states_include_previous_slot(self):
        states = difference_states([1.0], [0.5], Slot("y", 0))
        self.assertEqual(states[0][Slot("y", 0)], 1.0)
        self.assertEqual(states[0][Slot("y", -1)], 0.5)

    def test_derivation_bank_abstains_without_a_rule_or_input(self):
        fitted = ThresholdSignRule(
            "sign", Slot("vy", 0), threshold=0.0, support=10,
            confidence=1.0,
        )
        rule = make_threshold_rule(
            fitted, Slot("vy", 0), Slot("dy_sign", 1)
        )
        bank = ComponentDerivationBank()
        self.assertIs(
            bank.predict("missing", {Slot("vy", 0): 1.0}), UNKNOWN
        )
        bank.bind("component", rule)
        self.assertEqual(
            bank.predict("component", {Slot("vy", 0): 1.0}), 1
        )
        self.assertIs(bank.predict("component", {}), UNKNOWN)


if __name__ == "__main__":
    unittest.main()
