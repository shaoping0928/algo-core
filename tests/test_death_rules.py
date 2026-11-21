"""0/1 death-condition induction and counterexample refinement."""
import unittest

import numpy as np

from death_rules import UNKNOWN, DeathRuleSet


def _band_values(rng, n, low, high, outer=1.0):
    """Values either inside [-low, low] or outside [-outer, -high]."""
    inner = rng.uniform(-low, low, n)
    side = rng.choice([-1.0, 1.0], n)
    outer_values = side * rng.uniform(high, outer, n)
    return np.where(rng.rand(n) < 0.5, inner, outer_values)


class DeathRuleTests(unittest.TestCase):
    def test_recovers_a_two_literal_death_condition(self):
        rng = np.random.RandomState(0)
        n = 2400
        x0 = _band_values(rng, n, 0.12, 0.18)
        x1 = _band_values(rng, n, 0.35, 0.45)
        noise = rng.uniform(-1.0, 1.0, (n, 2))
        states = np.column_stack([x0, x1, noise])
        labels = (
            (np.abs(x0) < 0.15) & (np.abs(x1) > 0.40)
        ).astype(int)
        cut = int(0.7 * n)
        rules = DeathRuleSet(
            max_literals=4, min_support=2, max_thresholds=64
        ).fit(states[:cut], labels[:cut])
        report = rules.verify(states[cut:], labels[cut:])
        self.assertEqual(report["false_positive"], 0)
        self.assertEqual(report["false_negative"], 0)
        self.assertEqual(report["unknown"], 0)
        self.assertGreaterEqual(len(rules.rules_for(1)), 1)
        self.assertTrue(any(
            len(rule.literals) >= 2 for rule in rules.rules_for(1)
        ))
        self.assertTrue(any(
            "f0" in rule.text() and "f1" in rule.text()
            for rule in rules.rules_for(1)
        ))
        death_only = rules.classify_death(states[cut:])
        self.assertTrue(np.all((death_only == 1) | (death_only == UNKNOWN)))
        self.assertTrue(np.all(death_only[labels[cut:] == 1] == 1))

    def test_recovers_a_disjunction_as_separate_rules(self):
        # A deterministic 2-D grid keeps both branches represented in the
        # train and test splits; this test is about rule form, not about
        # extrapolating into an unobserved gap.
        axis = np.linspace(-1.0, 1.0, 61)
        xx, yy = np.meshgrid(axis, axis)
        states = np.column_stack([xx.ravel(), yy.ravel()])
        labels = (
            (states[:, 0] < -0.5)
            | ((states[:, 0] > 0.2) & (states[:, 1] > 0.5))
        ).astype(int)
        order = np.random.RandomState(1).permutation(len(states))
        cut = int(0.7 * len(states))
        train = order[:cut]
        test = order[cut:]
        rules = DeathRuleSet(
            max_literals=4, min_support=2, max_thresholds=64
        ).fit(states[train], labels[train])
        report = rules.verify(states[test], labels[test])
        self.assertEqual(report["false_positive"], 0)
        self.assertEqual(report["false_negative"], 0)
        self.assertGreaterEqual(len(rules.rules_for(1)), 2)

    def test_mixed_leaf_is_unknown_not_a_probability(self):
        rng = np.random.RandomState(2)
        states = rng.uniform(-1.0, 1.0, (400, 2))
        labels = (
            (states[:, 0] > 0.0) ^ (states[:, 1] > 0.0)
        ).astype(int)
        rules = DeathRuleSet(
            max_literals=1, min_support=2
        ).fit(states, labels)
        prediction = rules.classify(states)
        self.assertTrue(np.all(prediction == UNKNOWN))
        report = rules.verify(states, labels)
        self.assertEqual(report["unknown"], len(states))
        self.assertEqual(report["answered"], 0)
        self.assertEqual(rules.rules_, [])

    def test_counterexamples_add_the_missing_variable(self):
        rng = np.random.RandomState(3)
        n = 240
        x0 = rng.uniform(-1.0, 1.0, n)
        x1 = rng.uniform(-1.0, 1.0, n)
        # The third coordinate is the extra variable the first pass never
        # had to use: it is constant until the counterexamples arrive.
        states = np.column_stack([x0, x1, np.zeros(n)])
        labels = (x0 < -0.2).astype(int)
        rules = DeathRuleSet(
            max_literals=4, min_support=2, max_thresholds=48
        ).fit(states, labels)
        # A second basin that the first rule never saw: the third
        # coordinate separates it from the old safe region.
        new_states = np.vstack([
            np.column_stack([
                rng.uniform(0.25, 0.8, 24),
                rng.uniform(0.6, 0.95, 24),
                np.ones(24),
            ]),
            np.column_stack([
                rng.uniform(0.25, 0.8, 24),
                rng.uniform(-0.95, -0.6, 24),
                np.ones(24),
            ]),
        ])
        new_labels = np.asarray([1] * 24 + [0] * 24)
        before = rules.classify(new_states[:24])
        self.assertTrue(np.all(before != 1))
        for state, label in zip(new_states, new_labels):
            rules.refine(state, int(label))
        after_deaths = rules.classify(new_states[:24])
        after_safe = rules.classify(new_states[24:])
        self.assertTrue(np.all(after_deaths == 1))
        self.assertTrue(np.all(after_safe == 0))
        self.assertEqual(len(rules.refinement_log), 48)
        self.assertEqual(rules.refinement_log[-1]["after"], 0)

    def test_coverage_first_grows_a_three_literal_rule(self):
        rng = np.random.RandomState(7)
        n = 1600
        states = rng.uniform(-1.0, 1.0, (n, 3))
        labels = (
            (states[:, 0] > 0.0)
            & (states[:, 1] > 0.0)
            & (states[:, 2] > 0.0)
        ).astype(int)
        rules = DeathRuleSet(
            max_literals=2, min_support=2, max_thresholds=32,
            coverage_first=True,
        ).fit(states, labels)
        coverage = rules.coverage_report()
        self.assertEqual(coverage["death_uncovered_positive"], 0)
        self.assertEqual(coverage["death_coverage"], 1.0)
        self.assertEqual(coverage["death_covered_negative"], 0)
        self.assertTrue(any(
            len(rule.literals) >= 3 for rule in rules.rules_for(1)
        ))

    def test_coverage_first_falls_back_to_xor_composition(self):
        rng = np.random.RandomState(8)
        n = 1200
        states = rng.uniform(-1.0, 1.0, (n, 2))
        labels = (
            (states[:, 0] > 0.0) ^ (states[:, 1] > 0.0)
        ).astype(int)
        rules = DeathRuleSet(
            max_literals=1, min_support=2, max_thresholds=32,
            coverage_first=True, compose_max_base=16,
            compose_max_pairs=64,
        ).fit(states, labels)
        coverage = rules.coverage_report()
        self.assertEqual(coverage["death_uncovered_positive"], 0)
        self.assertEqual(coverage["death_covered_negative"], 0)
        self.assertTrue(any(
            "XOR" in rule.text() for rule in rules.rules_for(1)
        ))

    def test_boundary_margin_returns_unknown_not_safe(self):
        rng = np.random.RandomState(9)
        n = 1200
        states = np.column_stack([
            rng.uniform(-1.0, 1.0, n),
            rng.uniform(-1.0, 1.0, n),
        ])
        labels = (states[:, 0] > 0.5).astype(int)
        rules = DeathRuleSet(
            max_literals=3, min_support=2, max_thresholds=48,
            boundary_margin=0.02,
        ).fit(states, labels)
        probe = np.asarray([
            [0.499, 0.0],
            [-0.5, 0.0],
            [0.8, 0.0],
        ])
        prediction = rules.classify(probe)
        self.assertEqual(prediction[0], UNKNOWN)
        self.assertEqual(prediction[1], 0)
        self.assertEqual(prediction[2], 1)
        death_only = rules.classify_death(probe)
        self.assertEqual(death_only[0], UNKNOWN)
        self.assertEqual(death_only[2], 1)
        death_margin = rules.classify_death(probe, include_margin=True)
        self.assertEqual(death_margin[0], 1)
        self.assertEqual(death_margin[1], UNKNOWN)
        self.assertEqual(death_margin[2], 1)
        coverage = rules.coverage_report()
        self.assertGreater(coverage["safe_blocked_near_death"], 0)
        self.assertEqual(coverage["death_covered_negative"], 0)

    def test_redundant_literal_is_removed(self):
        rng = np.random.RandomState(6)
        n = 1200
        states = rng.uniform(-1.0, 1.0, (n, 3))
        # The third coordinate is always far above the threshold, so it
        # adds nothing to the conjunction.
        labels = (
            (states[:, 0] > 0.0) & (states[:, 1] > 0.0)
        ).astype(int)
        rules = DeathRuleSet(
            max_literals=4, min_support=2, max_thresholds=64
        ).fit(states, labels)
        report = rules.verify(states, labels)
        self.assertEqual(report["false_positive"], 0)
        self.assertEqual(report["false_negative"], 0)
        self.assertFalse(any(
            2 in rule.signature() for rule in rules.rules_for(1)
        ))

    def test_discovers_two_death_modes(self):
        rng = np.random.RandomState(5)
        n = 3000
        states = rng.uniform(-1.0, 1.0, (n, 4))
        # Two disjoint terminal regions with different coordinate
        # signatures.  The learner is not told that there are two.
        mode_a = states[:, 0] < -0.6
        mode_b = (
            (np.abs(states[:, 1]) < 0.15)
            & (states[:, 2] > 0.5)
        )
        labels = (mode_a | mode_b).astype(int)
        cut = int(0.7 * n)
        rules = DeathRuleSet(
            max_literals=4, min_support=2, max_thresholds=96
        ).fit(states[:cut], labels[:cut])
        report = rules.verify(states[cut:], labels[cut:])
        self.assertEqual(report["false_positive"], 0)
        # A little boundary bleed is acceptable; almost every death must
        # be covered.
        self.assertLessEqual(report["false_negative"], 2)
        modes = rules.modes(1)
        self.assertGreaterEqual(len(modes), 2)
        signatures = {mode["signature"] for mode in modes}
        self.assertTrue(any(0 in signature for signature in signatures))
        self.assertTrue(any(
            signature and (1 in signature or 2 in signature)
            for signature in signatures
        ))

    def test_rule_mask_is_the_conjunction_of_its_literals(self):
        rng = np.random.RandomState(4)
        states = rng.uniform(-1.0, 1.0, (600, 3))
        labels = (
            (np.abs(states[:, 0]) < 0.3)
            & (states[:, 2] > 0.2)
        ).astype(int)
        rules = DeathRuleSet(
            max_literals=4, min_support=2, max_thresholds=48
        ).fit(states, labels)
        for rule in rules.rules_for(1):
            mask = rule.mask(states)
            self.assertTrue(np.all(labels[mask] == 1))
            self.assertIn("death", rule.text())


if __name__ == "__main__":
    unittest.main()
