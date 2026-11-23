"""Tests for the query-driven transition graph substrate."""

import unittest

import numpy as np

from query_graph import (
    ACTIVE,
    INACTIVE,
    UNKNOWN,
    Applicability,
    CallablePredictor,
    DtSpec,
    Pattern,
    Query,
    Readout,
    Rule,
    RuleBank,
    Slot,
    TimeWindow,
    make_event_rule,
    make_linear_rule,
)


def linear_rule(name, target, inputs, weights, confidence=1.0,
                applicability=None, group=None):
    return make_linear_rule(
        name=name,
        target_slot=target,
        input_slots=tuple(inputs),
        weights=np.asarray(weights, dtype=float),
        confidence=confidence,
        support=10,
        applicability=applicability,
        group=group,
    )


class QueryGraphTest(unittest.TestCase):
    def test_linear_rule_fills_a_target_slot(self):
        rule = linear_rule(
            "identity", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0]
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 3.0},
        ))
        self.assertEqual(result.values[Slot("x", 1)], 3.0)
        self.assertEqual(result.applied, 1)
        self.assertEqual(result.unknown, 0)

    def test_exact_query_rejects_a_variable_duration_rule(self):
        stage = Rule(
            name="stage",
            pattern_in=Pattern(()),
            pattern_out=Pattern((Slot("stage", 5),)),
            dt=DtSpec.interval(5, 10),
            predictor=None,
        )
        bank = RuleBank([stage])
        result = bank.query(Query(
            target_slots=(Slot("stage", 5),), source={},
        ))
        self.assertIs(result.values[Slot("stage", 5)], UNKNOWN)
        self.assertEqual(result.visited, 0)

    def test_shared_applicability_can_prune_the_whole_query(self):
        gate = Applicability(
            slots=(Slot("x", 0),),
            center=np.asarray([0.0]),
            radius=0.1,
            support=10,
        )
        rule = linear_rule(
            "identity", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0],
        )
        bank = RuleBank([rule], shared_applicability=gate)
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 1.0},
        ))
        self.assertEqual(result.shared_status, INACTIVE)
        self.assertIs(result.values[Slot("x", 1)], UNKNOWN)
        self.assertEqual(result.visited, 0)

    def test_independent_applicability_visits_and_prunes_candidates(self):
        near = Applicability(
            slots=(Slot("x", 0),), center=np.asarray([0.0]),
            radius=0.5, support=10,
        )
        far = Applicability(
            slots=(Slot("x", 0),), center=np.asarray([10.0]),
            radius=0.5, support=10,
        )
        first = linear_rule(
            "near", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0],
            applicability=near,
        )
        second = linear_rule(
            "far", Slot("x", 1), (Slot("x", 0),), [0.0, 2.0],
            applicability=far,
        )
        bank = RuleBank([first, second])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 0.2},
        ))
        self.assertEqual(result.visited, 2)
        self.assertEqual(result.pruned, 1)
        self.assertEqual(result.applied, 1)
        self.assertAlmostEqual(float(result.values[Slot("x", 1)]), 0.2)

    def test_key_node_query_does_not_resolve_unrequested_targets(self):
        x_rule = linear_rule(
            "x", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0]
        )
        y_rule = linear_rule(
            "y", Slot("y", 1), (Slot("y", 0),), [0.0, 1.0]
        )
        bank = RuleBank([x_rule, y_rule])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 2.0, Slot("y", 0): 5.0},
        ))
        self.assertEqual(result.values[Slot("x", 1)], 2.0)
        self.assertNotIn(Slot("y", 1), result.values)
        self.assertEqual(result.visited, 1)

    def test_missing_input_yields_unknown(self):
        rule = linear_rule(
            "needs-z", Slot("x", 1), (Slot("z", 0),), [0.0, 1.0]
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={},
        ))
        self.assertIs(result.values[Slot("x", 1)], UNKNOWN)
        self.assertEqual(result.unknown, 1)
        self.assertEqual(result.applied, 0)

    def test_equal_confidence_conflict_is_unknown(self):
        first = linear_rule(
            "first", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0],
            confidence=0.8,
        )
        second = linear_rule(
            "second", Slot("x", 1), (Slot("x", 0),), [0.0, 2.0],
            confidence=0.8,
        )
        bank = RuleBank([first, second])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 1.0},
        ))
        self.assertIs(result.values[Slot("x", 1)], UNKNOWN)
        self.assertEqual(result.conflicts, 1)

    def test_confidence_resolves_a_conflict(self):
        first = linear_rule(
            "first", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0],
            confidence=0.9,
        )
        second = linear_rule(
            "second", Slot("x", 1), (Slot("x", 0),), [0.0, 2.0],
            confidence=0.5,
        )
        bank = RuleBank([first, second])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 1.0},
        ))
        self.assertEqual(result.values[Slot("x", 1)], 1.0)
        self.assertEqual(result.conflicts, 0)

    def test_recursive_resolution_uses_a_rule_twice(self):
        rule = linear_rule(
            "step", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0]
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("x", 2),),
            source={Slot("x", 0): 4.0},
        ))
        self.assertEqual(result.values[Slot("x", 2)], 4.0)
        self.assertEqual(result.visited, 2)
        self.assertEqual(result.applied, 2)

    def test_recursive_resolution_handles_three_steps(self):
        rule = linear_rule(
            "step", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0]
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("x", 3),),
            source={Slot("x", 0): 4.0},
        ))
        self.assertEqual(result.values[Slot("x", 3)], 4.0)
        self.assertEqual(result.visited, 3)
        self.assertEqual(result.applied, 3)

    def test_exact_dt_mismatch_does_not_fill_a_shorter_target(self):
        rule = linear_rule(
            "two-step", Slot("x", 2), (Slot("x", 0),), [0.0, 1.0]
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 4.0},
        ))
        self.assertIs(result.values[Slot("x", 1)], UNKNOWN)
        self.assertEqual(result.visited, 0)

    def test_shared_group_filters_rules_before_visiting(self):
        gate = Applicability(support=10)
        group_zero = linear_rule(
            "zero", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0],
            group=0,
        )
        group_one = linear_rule(
            "one", Slot("x", 1), (Slot("x", 0),), [0.0, 2.0],
            group=1,
        )
        bank = RuleBank(
            [group_zero, group_one], shared_applicability=gate,
        )
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 1.0},
            group=1,
        ))
        self.assertEqual(result.values[Slot("x", 1)], 2.0)
        self.assertEqual(result.visited, 1)
        self.assertEqual(result.shared_status, ACTIVE)

    def test_event_predicate_returns_a_discrete_flag(self):
        rule = make_event_rule(
            "high", Slot("high", 1), Slot("x", 0), threshold=0.5,
            confidence=1.0, support=10,
        )
        bank = RuleBank([rule])
        high = bank.query(Query(
            target_slots=(Slot("high", 1),),
            source={Slot("x", 0): 1.0},
        ))
        low = bank.query(Query(
            target_slots=(Slot("high", 1),),
            source={Slot("x", 0): 0.0},
        ))
        self.assertEqual(high.values[Slot("high", 1)], 1)
        self.assertEqual(low.values[Slot("high", 1)], 0)

    def test_readout_declares_required_slots_and_abstains(self):
        rule = linear_rule(
            "x", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0]
        )
        readout = Readout(
            "needs-x-and-y",
            required_slots=(Slot("x", 1), Slot("y", 1)),
            function=lambda state: state[Slot("x", 1)],
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 1.0},
            readout=readout,
        ))
        self.assertIs(result.readout, UNKNOWN)
        self.assertTrue(result.readout_unknown)

    def test_readout_runs_when_required_slots_are_filled(self):
        rule = linear_rule(
            "x", Slot("x", 1), (Slot("x", 0),), [0.0, 1.0]
        )
        readout = Readout(
            "x-only",
            required_slots=(Slot("x", 1),),
            function=lambda state: state[Slot("x", 1)] > 0.0,
        )
        bank = RuleBank([rule])
        result = bank.query(Query(
            target_slots=(Slot("x", 1),),
            source={Slot("x", 0): 1.0},
            readout=readout,
        ))
        self.assertTrue(result.readout)
        self.assertFalse(result.readout_unknown)

    def test_dt_spec_interval_and_exact(self):
        self.assertTrue(DtSpec.steps(1).allows(1))
        self.assertFalse(DtSpec.steps(1).allows(2))
        self.assertTrue(DtSpec.interval(5, 10).allows(7))
        self.assertFalse(DtSpec.interval(5, 10).allows(4))

    def test_window_accepts_a_variable_dt_rule_and_marks_it_coarse(self):
        stage = Rule(
            name="stage",
            pattern_in=Pattern((Slot("x", 0),)),
            pattern_out=Pattern((Slot("stage", 0),)),
            dt=DtSpec.interval(5, 10),
            predictor=CallablePredictor(lambda state: 7.0),
        )
        bank = RuleBank([stage])
        target = Slot("stage", 8)
        result = bank.query(Query(
            target_slots=(target,),
            source={Slot("x", 0): 1.0},
            target_windows={target: TimeWindow(5, 10)},
        ))
        self.assertEqual(result.values[target], 7.0)
        self.assertEqual(result.coarse, (target,))
        self.assertIsNone(result.times[target])
        self.assertEqual(result.applied, 1)

    def test_window_rejects_a_rule_outside_the_interval(self):
        stage = Rule(
            name="stage",
            pattern_in=Pattern((Slot("x", 0),)),
            pattern_out=Pattern((Slot("stage", 0),)),
            dt=DtSpec.interval(5, 10),
            predictor=CallablePredictor(lambda state: 7.0),
        )
        bank = RuleBank([stage])
        target = Slot("stage", 8)
        result = bank.query(Query(
            target_slots=(target,),
            source={Slot("x", 0): 1.0},
            target_windows={target: TimeWindow(2, 4)},
        ))
        self.assertIs(result.values[target], UNKNOWN)
        self.assertEqual(result.visited, 0)

    def test_window_prefers_an_exact_rule_over_a_coarse_one(self):
        exact = Rule(
            name="exact",
            pattern_in=Pattern(()),
            pattern_out=Pattern((Slot("stage", 1),)),
            dt=DtSpec.steps(1),
            predictor=CallablePredictor(lambda state: 1.0),
        )
        coarse = Rule(
            name="coarse",
            pattern_in=Pattern(()),
            pattern_out=Pattern((Slot("stage", 0),)),
            dt=DtSpec.interval(5, 10),
            predictor=CallablePredictor(lambda state: 2.0),
        )
        bank = RuleBank([coarse, exact])
        target = Slot("stage", 8)
        result = bank.query(Query(
            target_slots=(target,),
            source={},
            target_windows={target: TimeWindow(5, 10)},
        ))
        self.assertEqual(result.values[target], 1.0)
        self.assertEqual(result.coarse, ())
        self.assertEqual(result.times[target], 8)


if __name__ == "__main__":
    unittest.main()
