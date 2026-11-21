"""Nested expression composition and XOR/NOT rule recovery."""
import unittest

import numpy as np

from death_rules import UNKNOWN, DeathRuleSet
from node_features import Primitive
from rule_expressions import (
    Compare,
    FeatureTerm,
    Literal,
    Nary,
    Not,
    NumericOp,
)


class ExpressionTests(unittest.TestCase):
    def test_nary_and_or_xor_and_not(self):
        rng = np.random.RandomState(0)
        states = rng.uniform(-1.0, 1.0, (400, 3))
        a = Literal(Primitive(0, False, ">", 0.0))
        b = Literal(Primitive(1, False, ">", 0.0))
        c = Literal(Primitive(2, False, ">", 0.0))
        and_mask = Nary("and", (a, b, c)).mask(states)
        self.assertTrue(np.array_equal(
            and_mask, (states[:, 0] > 0) & (states[:, 1] > 0)
            & (states[:, 2] > 0)
        ))
        or_mask = Nary("or", (a, b, c)).mask(states)
        self.assertTrue(np.array_equal(
            or_mask, (states[:, 0] > 0) | (states[:, 1] > 0)
            | (states[:, 2] > 0)
        ))
        xor_mask = Nary("xor", (a, b)).mask(states)
        self.assertTrue(np.array_equal(
            xor_mask, (states[:, 0] > 0) ^ (states[:, 1] > 0)
        ))
        not_mask = Not(a).mask(states)
        self.assertTrue(np.array_equal(not_mask, ~(states[:, 0] > 0)))
        nested = Nary("and", (Not(a), Nary("or", (b, c))))
        self.assertTrue(nested.text().startswith("("))
        self.assertEqual(nested.signature(), (0, 1, 2))

    def test_numeric_term_comparisons(self):
        rng = np.random.RandomState(1)
        states = rng.uniform(-1.0, 1.0, (300, 2))
        difference = NumericOp("diff", (
            FeatureTerm(0), FeatureTerm(1)
        ))
        comparison = Compare(difference, ">", 0.2)
        expected = (states[:, 0] - states[:, 1]) > 0.2
        self.assertTrue(np.array_equal(comparison.mask(states), expected))
        self.assertIn("f0", comparison.text())
        self.assertIn("f1", comparison.text())
        product = NumericOp("product", (
            FeatureTerm(0), FeatureTerm(1)
        ))
        self.assertTrue(np.allclose(
            product.values(states), states[:, 0] * states[:, 1]
        ))

    def test_composition_search_recovers_xor(self):
        rng = np.random.RandomState(2)
        n = 800
        states = rng.uniform(-1.0, 1.0, (n, 2))
        labels = (
            (states[:, 0] > 0.0) ^ (states[:, 1] > 0.0)
        ).astype(int)
        cut = int(0.7 * n)
        rules = DeathRuleSet(
            max_literals=1, min_support=2, max_thresholds=32,
            compose=True, compose_max_base=16, compose_max_pairs=64,
            compose_rules=8,
        ).fit(states[:cut], labels[:cut])
        report = rules.verify(states[cut:], labels[cut:])
        self.assertEqual(report["false_positive"], 0)
        self.assertEqual(report["false_negative"], 0)
        self.assertTrue(any(
            "XOR" in rule.text() or "NOT" in rule.text()
            for rule in rules.rules_for(1)
        ))

    def test_composition_search_recovers_not(self):
        rng = np.random.RandomState(3)
        n = 800
        states = rng.uniform(-1.0, 1.0, (n, 2))
        labels = ((states[:, 0] <= 0.0) & (states[:, 1] > 0.0)).astype(int)
        # Make the target a NOT of the first literal and an AND with the
        # second, but do not hand the learner that structure.
        rules = DeathRuleSet(
            max_literals=1, min_support=2, max_thresholds=32,
            compose=True, compose_max_base=16, compose_max_pairs=64,
            compose_rules=8,
        ).fit(states, labels)
        report = rules.verify(states, labels)
        self.assertLessEqual(report["false_negative"], 2)
        self.assertLessEqual(report["false_positive"], 2)
        self.assertTrue(rules.rules_for(1))

    def test_unknown_when_composition_cannot_separate(self):
        rng = np.random.RandomState(4)
        states = rng.uniform(-1.0, 1.0, (200, 2))
        labels = (rng.rand(200) < 0.5).astype(int)
        rules = DeathRuleSet(
            max_literals=1, min_support=2, max_thresholds=16,
            compose=False,
        ).fit(states, labels)
        report = rules.verify(states, labels)
        self.assertGreater(report["unknown"], 0)
        self.assertEqual(report["false_positive"], 0)
        self.assertEqual(report["false_negative"], 0)


if __name__ == "__main__":
    unittest.main()
