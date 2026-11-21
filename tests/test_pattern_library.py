"""Feature-dimension restriction and cross-world pattern schemas."""
import unittest

import numpy as np

from death_rules import DeathRuleSet
from pattern_library import PatternLibrary


class PatternLibraryTests(unittest.TestCase):
    def test_feature_dims_restrict_the_rule_language(self):
        rng = np.random.RandomState(0)
        n = 1200
        states = rng.uniform(-1.0, 1.0, (n, 3))
        labels = (states[:, 0] > 0.4).astype(int)
        rules = DeathRuleSet(
            max_literals=3, min_support=2, max_thresholds=32,
            feature_dims=(0,),
        ).fit(states, labels)
        self.assertTrue(rules.rules_)
        self.assertTrue(all(
            0 in rule.signature() and
            set(rule.signature()) <= {0}
            for rule in rules.rules_
        ))

    def test_library_extracts_the_shared_dimensions(self):
        rng = np.random.RandomState(1)
        rules = []
        for _ in range(3):
            states = rng.uniform(-1.0, 1.0, (600, 4))
            labels = (
                ((np.abs(states[:, 0]) < 0.2)
                 & (np.abs(states[:, 1]) > 0.4))
                | (np.abs(states[:, 2]) > 0.85)
            ).astype(int)
            rules.append(DeathRuleSet(
                max_literals=3, min_support=2, max_thresholds=32,
            ).fit(states, labels))
        library = PatternLibrary()
        library.add_rule_sets(rules)
        dims = library.schema_dims(1)
        self.assertTrue(set(dims) <= {0, 1, 2})
        self.assertIn(0, dims)
        self.assertIn(1, dims)
        self.assertIn(2, dims)
        self.assertNotIn(3, dims)
        instantiated = library.instantiate(
            states, labels, max_literals=3, min_support=2,
            max_thresholds=32,
        )
        self.assertTrue(instantiated.rules_)


if __name__ == "__main__":
    unittest.main()
