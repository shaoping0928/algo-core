"""Tests for the stage-type value readout."""
import unittest

import numpy as np

from stage_type_clusters import StageSample
from stage_type_value import StageTypeValue


def sample(episode, profile, outcome, value, key, terminal=False):
    return StageSample(
        episode=episode,
        profile=np.asarray(profile, dtype=float),
        outcome=(outcome,),
        value=float(value),
        terminal=bool(terminal),
        key=tuple(key),
    )


class StageTypeValueTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            sample(0, [1.0, 0.05], "A", 2.0, ("a",)),
            sample(1, [0.98, 0.02], "A", 2.0, ("a",)),
            sample(2, [1.0, 0.10], "A", 2.0, ("a",)),
            sample(3, [0.05, 1.0], "B", -1.0, ("b",), terminal=True),
            sample(4, [0.02, 0.98], "B", -1.0, ("b",), terminal=True),
            sample(5, [0.10, 1.0], "B", -1.0, ("b",), terminal=True),
        ]
        self.validation = [
            sample(10, [1.0, 0.02], "A", 2.0, ("a",)),
            sample(11, [0.03, 1.0], "B", -1.0, ("b",), terminal=True),
        ]

    def fit(self):
        return StageTypeValue.fit(
            self.train, self.validation, min_similarity=0.8,
            min_type_support=2,
        )

    def test_type_values_follow_realized_feedback(self):
        model = self.fit()
        type_a, value_a, _ = model.value(np.asarray([1.0, 0.0]))
        type_b, value_b, _ = model.value(np.asarray([0.0, 1.0]))
        self.assertNotEqual(type_a, type_b)
        self.assertAlmostEqual(value_a, 2.0)
        self.assertAlmostEqual(value_b, -1.0)
        self.assertAlmostEqual(model.normalized[type_a], 1.0)
        self.assertAlmostEqual(model.normalized[type_b], 0.0)

    def test_terminal_rate_is_aggregated(self):
        model = self.fit()
        _type_b, _value, _ = model.value(np.asarray([0.0, 1.0]))
        terminal = [
            rate for type_id, rate in model.terminal_rates.items()
            if model.values[type_id] < 0.0
        ]
        self.assertEqual(terminal, [1.0])

    def test_unknown_profile_gets_a_neutral_penalty(self):
        model = self.fit()
        type_id, value, _ = model.value(np.asarray([0.7071, 0.7071]))
        self.assertIsNone(type_id)
        self.assertIsNone(value)
        self.assertAlmostEqual(
            model.penalty(np.asarray([0.7071, 0.7071])), 0.5,
        )

    def test_root_penalties_map_tree_roots(self):
        class StubTree:
            def chain(self, key):
                return (0, 7) if key == ("a",) else (1, 9)

        model = self.fit()
        penalties = model.root_penalties(StubTree())
        self.assertAlmostEqual(penalties[("tree", "7")], 0.0)
        self.assertAlmostEqual(penalties[("tree", "9")], 1.0)
        self.assertAlmostEqual(penalties[("tree", "base")], 0.5)

    def test_fit_is_deterministic(self):
        first = self.fit()
        second = self.fit()
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(first.values, second.values)


if __name__ == "__main__":
    unittest.main()
