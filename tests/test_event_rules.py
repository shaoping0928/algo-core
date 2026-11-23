"""Tests for wrapping the production 0/1 death rules as event predicates."""

import unittest

import numpy as np

from death_rules import DeathRuleSet
from event_rules import make_death_event_rule
from query_graph import Query, RuleBank, Slot, UNKNOWN


def fitted_rules():
    values = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    labels = (values[:, 0] > 0.5).astype(int)
    return DeathRuleSet(
        max_rules=8, max_literals=1, min_support=2,
        max_thresholds=32,
    ).fit(values, labels)


class EventRulesTest(unittest.TestCase):
    def test_death_rules_wrap_as_a_discrete_event_rule(self):
        rules = fitted_rules()
        rule = make_death_event_rule(
            rules, [Slot("x0", 0)], target_slot=Slot("event", 1),
        )
        bank = RuleBank([rule])
        high = bank.query(Query(
            target_slots=(Slot("event", 1),),
            source={Slot("x0", 0): 1.5},
        ))
        low = bank.query(Query(
            target_slots=(Slot("event", 1),),
            source={Slot("x0", 0): -1.5},
        ))
        self.assertEqual(high.values[Slot("event", 1)], 1)
        self.assertEqual(low.values[Slot("event", 1)], 0)

    def test_missing_slot_gives_unknown_not_safe(self):
        rules = fitted_rules()
        rule = make_death_event_rule(
            rules, [Slot("x0", 0), Slot("x1", 0)],
            target_slot=Slot("event", 1),
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("event", 1),),
            source={Slot("x0", 0): 0.0},
        ))
        self.assertIs(result.values[Slot("event", 1)], UNKNOWN)
        self.assertEqual(result.unknown, 1)


if __name__ == "__main__":
    unittest.main()
