"""Tests for the stage-level option executor's value and trace helpers."""
import unittest
from types import SimpleNamespace

import numpy as np

from stage_executor import (
    StageExecutor,
    executable_option_suffix,
    expand_trace,
    option_values,
)
from stage_option_search import StageOptionSearchCapacityError


class Leaf:
    def __init__(self, option, probability, feedback, outcome,
                 elapsed=1.0):
        self.options = (option,)
        self.probability = float(probability)
        self.feedback_mean = tuple(feedback)
        self.outcome = outcome
        self.elapsed_mean = float(elapsed)


class Result:
    def __init__(self, leaves):
        self.leaves = tuple(leaves)
        self.unresolved = ()
        self.complete = True
        self.max_transitions = 1


class StageExecutorHelperTests(unittest.TestCase):
    def test_run_length_trace_expands_in_order(self):
        self.assertEqual(
            expand_trace(((0, 2), (1, 1), (0, 3))),
            [0, 0, 1, 0, 0, 0],
        )

    def test_option_suffix_matches_applied_prefix_and_pending_queue(self):
        suffix = executable_option_suffix(
            ((0, 3), (1, 2)),
            acted_prefix=[0],
            issued_prefix=[1, 0, 0],
            lag=3,
        )
        self.assertEqual(suffix, [1])

    def test_option_suffix_rejects_boundary_or_pending_action_mismatch(self):
        mismatch_boundary = executable_option_suffix(
            ((1, 1), (0, 2), (1, 2)),
            acted_prefix=[0],
            issued_prefix=[1, 0, 0],
            lag=3,
        )
        mismatch_pending = executable_option_suffix(
            ((0, 3), (1, 2)),
            acted_prefix=[0],
            issued_prefix=[0, 1, 0],
            lag=3,
        )
        wrong_queue_length = executable_option_suffix(
            ((0, 3), (1, 2)),
            acted_prefix=[0],
            issued_prefix=[1, 0],
            lag=3,
        )
        self.assertIsNone(mismatch_boundary)
        self.assertIsNone(mismatch_pending)
        self.assertIsNone(wrong_queue_length)

    def test_zero_lag_suffix_only_removes_actions_already_acted(self):
        self.assertEqual(
            executable_option_suffix(
                ((0, 1), (1, 2)), [0], None, lag=0,
            ),
            [1, 1],
        )
        self.assertEqual(
            executable_option_suffix(((0, 1),), [0], None, lag=0),
            [],
        )

    def test_option_value_prefers_feedback_and_penalises_terminal(self):
        option_a = ((0, 2),)
        option_b = ((1, 1),)
        result = Result([
            Leaf(option_a, 1.0, (0.0, 0.0), "terminal"),
            Leaf(option_b, 1.0, (2.0, 0.0), "open"),
        ])
        rows = option_values(result, terminal_weight=1.0)
        self.assertEqual(rows[0][2], option_b)
        self.assertAlmostEqual(rows[0][0], 2.0)
        self.assertAlmostEqual(rows[1][0], -1.0)

    def test_duration_weight_penalises_long_options(self):
        option = ((0, 1),)
        result = Result([Leaf(option, 1.0, (0.0, 0.0), "open", elapsed=10.0)])
        rows = option_values(result, terminal_weight=0.0,
                             duration_weight=0.1)
        self.assertAlmostEqual(rows[0][0], -1.0)

    def test_probability_weighted_mixture(self):
        option = ((0, 1),)
        result = Result([
            Leaf(option, 0.5, (4.0, 0.0), "open"),
            Leaf(option, 0.5, (0.0, 0.0), "terminal"),
        ])
        rows = option_values(result, terminal_weight=1.0)
        self.assertAlmostEqual(rows[0][0], 0.5 * 4.0 - 0.5 * 1.0)

    def test_no_leaves_yields_no_options(self):
        self.assertEqual(option_values(Result([])), [])

    def test_censored_mass_is_counted_as_unsafe(self):
        option = ((0, 1),)
        rows = option_values(Result([
            Leaf(option, 0.75, (4.0, 0.0), "open"),
            Leaf(option, 0.25, (0.0, 0.0), "censored"),
        ]), terminal_weight=1.0)
        self.assertAlmostEqual(rows[0][0], 2.75)
        self.assertAlmostEqual(rows[0][3], 0.25)

    def test_unresolved_mass_forces_abstention(self):
        option = ((0, 1),)
        result = Result([Leaf(option, 0.5, (4.0, 0.0), "open")])
        result.unresolved = (
            SimpleNamespace(options=(option,), probability=0.5),
        )
        self.assertEqual(option_values(result, terminal_weight=1.0), [])

    def test_multistage_value_is_rejected_without_continuation_policy(self):
        result = Result([Leaf(((0, 1),), 1.0, (1.0, 0.0), "open")])
        result.max_transitions = 2
        with self.assertRaisesRegex(ValueError, "max_transitions=1"):
            option_values(result)

    def test_incomplete_enumeration_cannot_be_scored(self):
        result = Result([Leaf(((0, 1),), 1.0, (10.0, 0.0), "open")])
        result.complete = False
        self.assertEqual(option_values(result), [])

    def test_executor_rejects_multistage_search_configuration(self):
        agent = SimpleNamespace(
            stage_experience=SimpleNamespace(detector=object()),
            stage_outcome_memory=object(),
        )
        with self.assertRaisesRegex(ValueError, "max_transitions=1"):
            StageExecutor(agent, max_transitions=2)

    def test_capacity_failure_is_a_safe_fallback(self):
        agent = SimpleNamespace(
            stage_experience=SimpleNamespace(detector=object()),
            stage_outcome_memory=object(),
        )
        executor = StageExecutor(agent)

        class Tracker:
            @staticmethod
            def context():
                return dict(stages=(object(),), history=(("stage",),))

        class Search:
            @staticmethod
            def search(_history):
                raise StageOptionSearchCapacityError("capacity")

        executor.tracker = Tracker()
        executor.search = Search()
        executor.maybe_plan()
        self.assertEqual(executor.active, [])
        self.assertEqual(executor.selected, 0)
        self.assertEqual(executor.snapshot()["search_rejects"], 1)

    def test_incomplete_or_unresolved_result_is_a_safe_fallback(self):
        agent = SimpleNamespace(
            stage_experience=SimpleNamespace(detector=object()),
            stage_outcome_memory=object(),
        )
        executor = StageExecutor(agent)

        class Tracker:
            @staticmethod
            def context():
                return dict(stages=(object(),), history=(("stage",),))

        class Search:
            result = None

            @classmethod
            def search(cls, _history):
                return cls.result

        executor.tracker = Tracker()
        executor.search = Search()
        Search.result = SimpleNamespace(complete=False, unresolved=())
        executor.maybe_plan()
        self.assertEqual(executor.snapshot()["search_rejects"], 1)

        executor.stage_count = 0
        Search.result = SimpleNamespace(
            complete=True,
            unresolved=(SimpleNamespace(probability=0.5),),
        )
        executor.maybe_plan()
        self.assertEqual(executor.snapshot()["search_rejects"], 2)
        self.assertEqual(executor.selected, 0)

    def test_executor_only_selects_an_option_matching_committed_actions(self):
        acted_stage = SimpleNamespace(action_trace=((0, 1),))

        class Tracker:
            @staticmethod
            def context():
                return dict(
                    stages=(acted_stage,),
                    current=acted_stage,
                    history=(("stage",),),
                )

            @staticmethod
            def reset(_episode=None):
                return None

        class Search:
            result = None

            @classmethod
            def search(cls, _history):
                return cls.result

        def result_for(option):
            step = SimpleNamespace(context_depth=2, support=5)
            leaf = SimpleNamespace(
                options=(option,),
                steps=(step,),
                probability=1.0,
                feedback_mean=(1.0, 0.0),
                outcome="open",
                elapsed_mean=1.0,
            )
            return SimpleNamespace(
                leaves=(leaf,), unresolved=(), complete=True,
                max_transitions=1,
            )

        agent = SimpleNamespace(
            stage_experience=SimpleNamespace(detector=object()),
            stage_outcome_memory=object(),
        )
        executor = StageExecutor(agent, lag=3)
        executor.tracker = Tracker()
        executor.search = Search()

        compatible = ((0, 3), (1, 3))
        Search.result = result_for(compatible)
        executor.maybe_plan(prefix=np.asarray([1.0, 0.0, 0.0]))
        self.assertEqual(executor.active, [1, 1])
        self.assertEqual(executor.selected, 1)

        executor.reset()
        executor.tracker = Tracker()
        executor.search = Search()
        Search.result = result_for(((1, 1), (0, 2), (1, 2)))
        executor.maybe_plan(prefix=np.asarray([1.0, 0.0, 0.0]))
        self.assertEqual(executor.active, [])
        self.assertEqual(executor.selected, 0)
        self.assertEqual(executor.snapshot()["alignment_rejects"], 1)

    def test_option_class_is_rejected_when_a_member_prefix_is_incompatible(self):
        acted_stage = SimpleNamespace(action_trace=((0, 1),))
        compatible = ((0, 3), (1, 3))
        incompatible = ((1, 1), (0, 2), (1, 2))
        quotient = SimpleNamespace(
            accepted=True,
            classes={
                0: SimpleNamespace(
                    representative=compatible,
                    members=(compatible, incompatible),
                )
            },
        )

        class Tracker:
            @staticmethod
            def context():
                return dict(
                    stages=(acted_stage,),
                    current=acted_stage,
                    history=(("stage",),),
                )

        class Search:
            @staticmethod
            def search(_history):
                step = SimpleNamespace(context_depth=2, support=5)
                leaf = SimpleNamespace(
                    options=(("option_class", 0),),
                    steps=(step,),
                    probability=1.0,
                    feedback_mean=(1.0, 0.0),
                    outcome="open",
                    elapsed_mean=1.0,
                )
                return SimpleNamespace(
                    leaves=(leaf,), unresolved=(), complete=True,
                    max_transitions=1,
                )

        agent = SimpleNamespace(
            stage_experience=SimpleNamespace(detector=object()),
            stage_outcome_memory=object(),
        )
        executor = StageExecutor(
            agent, lag=3, option_quotient=quotient,
        )
        executor.tracker = Tracker()
        executor.search = Search()
        executor.maybe_plan(prefix=np.asarray([1.0, 0.0, 0.0]))
        self.assertEqual(executor.active, [])
        self.assertEqual(executor.selected, 0)
        self.assertEqual(executor.snapshot()["alignment_rejects"], 1)

    def test_compatible_option_class_can_execute_its_representative(self):
        acted_stage = SimpleNamespace(action_trace=((0, 1),))
        representative = ((0, 3), (1, 3))
        other_member = ((0, 3), (1, 2), (0, 1))
        quotient = SimpleNamespace(
            accepted=True,
            classes={
                0: SimpleNamespace(
                    representative=representative,
                    members=(representative, other_member),
                )
            },
        )

        class Tracker:
            @staticmethod
            def context():
                return dict(
                    stages=(acted_stage,),
                    current=acted_stage,
                    history=(("stage",),),
                )

        class Search:
            @staticmethod
            def search(_history):
                step = SimpleNamespace(context_depth=2, support=5)
                leaf = SimpleNamespace(
                    options=(("option_class", 0),),
                    steps=(step,),
                    probability=1.0,
                    feedback_mean=(1.0, 0.0),
                    outcome="open",
                    elapsed_mean=1.0,
                )
                return SimpleNamespace(
                    leaves=(leaf,), unresolved=(), complete=True,
                    max_transitions=1,
                )

        agent = SimpleNamespace(
            stage_experience=SimpleNamespace(detector=object()),
            stage_outcome_memory=object(),
        )
        executor = StageExecutor(
            agent, lag=3, option_quotient=quotient,
        )
        executor.tracker = Tracker()
        executor.search = Search()
        executor.maybe_plan(prefix=np.asarray([1.0, 0.0, 0.0]))
        self.assertEqual(executor.active, [1, 1])
        self.assertEqual(executor.selected, 1)
        self.assertEqual(executor.snapshot()["alignment_rejects"], 0)


if __name__ == "__main__":
    unittest.main()
