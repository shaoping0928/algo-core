"""Context-bucketed 0/1 death rules."""
import unittest

import numpy as np

from contextual_rules import ContextualDeathRuleSet
from death_rules import UNKNOWN


class ContextualRuleTests(unittest.TestCase):
    def test_context_specific_conditions_beat_a_global_rule(self):
        rng = np.random.RandomState(0)
        n = 2400
        states = rng.uniform(-1.0, 1.0, (n, 2))
        context = (rng.rand(n) < 0.5).astype(int)
        # The same coordinates mean different terminal conditions in the
        # two contexts.  The global state-only rule cannot separate both
        # without false positives; the bucketed set can.
        labels = np.where(
            context == 0, states[:, 0] > 0.4, states[:, 1] > 0.4
        ).astype(int)
        rules = ContextualDeathRuleSet(
            buckets=2, min_positive=2, max_literals=3,
            min_support=2, max_thresholds=48, coverage_first=True,
        ).fit(states, labels, context)
        prediction = rules.classify(states, context)
        self.assertTrue(np.all(prediction != UNKNOWN))
        self.assertEqual(
            int(np.sum((prediction == 1) & (labels == 0))), 0
        )
        self.assertEqual(
            int(np.sum((prediction == 0) & (labels == 1))), 0
        )
        self.assertTrue(all(
            stats["fitted"] for stats in rules.bucket_stats_.values()
        ))

    def test_sparse_bucket_falls_back_to_global(self):
        rng = np.random.RandomState(1)
        n = 800
        states = rng.uniform(-1.0, 1.0, (n, 2))
        context = np.zeros(n)
        context[:20] = 1.0
        labels = (states[:, 0] > 0.5).astype(int)
        labels[:20] = 0  # the rare context has no terminal examples
        rules = ContextualDeathRuleSet(
            buckets=2, min_positive=2, max_literals=3,
            min_support=2, max_thresholds=32,
        ).fit(states, labels, context)
        self.assertFalse(rules.bucket_stats_[1]["fitted"])
        self.assertIs(rules.bucket_rules_.get(1), None)
        probe = np.asarray([[-0.8, 0.0], [0.8, 0.0]])
        prediction = rules.classify(probe, [1.0, 1.0])
        self.assertEqual(prediction[0], 0)
        self.assertEqual(prediction[1], 1)

    def test_refine_adds_a_context_bucket_example(self):
        rng = np.random.RandomState(2)
        n = 600
        states = rng.uniform(-1.0, 1.0, (n, 2))
        context = np.zeros(n)
        labels = (states[:, 0] > 0.5).astype(int)
        rules = ContextualDeathRuleSet(
            buckets=1, min_positive=2, max_literals=3,
            min_support=2, max_thresholds=32,
        ).fit(states, labels, context)
        probe = np.asarray([[0.8, 0.0]])
        before = rules.classify(probe, [1.0])[0]
        self.assertEqual(before, 1)  # global fallback
        rules.refine(probe[0], 1, context=1.0)
        after = rules.classify(probe, [1.0])[0]
        self.assertEqual(after, 1)
        self.assertTrue(rules.refinement_log)

    def test_death_only_never_claims_safe(self):
        rng = np.random.RandomState(3)
        n = 800
        states = rng.uniform(-1.0, 1.0, (n, 2))
        context = (rng.rand(n) < 0.5).astype(int)
        labels = np.where(
            context == 0, states[:, 0] > 0.4, states[:, 1] > 0.4
        ).astype(int)
        rules = ContextualDeathRuleSet(
            buckets=2, min_positive=2, max_literals=3,
            min_support=2, max_thresholds=32,
        ).fit(states, labels, context)
        prediction = rules.classify_death(states, context)
        self.assertTrue(np.all(
            (prediction == 1) | (prediction == UNKNOWN)
        ))
        self.assertTrue(np.all(prediction[labels == 1] == 1))


if __name__ == "__main__":
    unittest.main()
