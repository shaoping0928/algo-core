"""Tests for the exhaustive stage-level (non-frame) option search."""
import unittest

import numpy as np

from stage_option_search import (
    StageOptionSearch,
    StageOptionSearchCapacityError,
    distribution_mean,
    distribution_quantile,
    exhaustive_option_search,
    survival_curve,
)
from stage_outcome_memory import (
    StageOccurrence,
    StageOutcomeMemory,
    encode_action_trace,
)


def occurrence(episode, key, actions, duration=2.0, feedback=(0.0, 0.0),
               terminal=False, censored=False):
    return StageOccurrence(
        episode=episode,
        key=tuple(key),
        entry_state=np.asarray([float(len(key))]),
        exit_state=np.asarray([float(len(key)) + 1.0]),
        action_trace=encode_action_trace(actions),
        duration=duration,
        feedback=np.asarray(feedback, dtype=float),
        terminal=terminal,
        censored=censored,
    )


def reference_leaves(memory, context, remaining):
    """Independent enumeration used to check completeness.

    Mirrors the contract, not the implementation: every supported option is
    expanded, every outcome is followed, and a depth cut emits ``open``.
    """
    if remaining <= 0:
        return [((), "open", 1.0, context)]
    rows = []
    for option in sorted(memory.action_options, key=repr):
        prediction = memory.predict(context, option)
        if prediction.get("status") != "supported":
            continue
        if int(prediction.get("context_depth") or 0) < 1:
            continue
        mass = sum(row["probability"] for row in prediction["outcomes"])
        assert abs(mass - 1.0) < 1e-9
        for outcome in prediction["outcomes"]:
            probability = float(outcome["probability"])
            if outcome["terminal"]:
                rows.append(((option,), "terminal", probability, context))
            elif outcome["censored"]:
                rows.append(((option,), "censored", probability, context))
            else:
                for sub in reference_leaves(
                        memory,
                        context + (outcome["destination"],),
                        remaining - 1):
                    rows.append((
                        (option,) + sub[0], sub[1],
                        probability * sub[2], sub[3],
                    ))
    return rows


class StubMemory:
    """Minimal memory with an explicit table; UNKNOWN otherwise."""

    def __init__(self, table, options):
        self.table = dict(table)
        self.action_options = set(options)

    def predict(self, context, option):
        return self.table.get(
            (tuple(context), option),
            dict(
                status="unknown", context=(), context_depth=None,
                outcomes=(),
            ),
        )


def prediction(context, outcomes):
    return dict(
        status="supported",
        context=tuple(context),
        context_depth=len(context),
        outcomes=tuple(outcomes),
    )


def outcome(destination=None, terminal=False, censored=False,
            probability=1.0, duration=2.0, feedback=(0.0, 0.0),
            distribution=None):
    if distribution is None:
        distribution = ((float(duration), 1.0),)
    return dict(
        destination=destination,
        terminal=terminal,
        censored=censored,
        probability=float(probability),
        count=1,
        episode_support=1,
        duration=dict(
            mean=float(duration),
            distribution=tuple(
                (float(time), float(weight))
                for time, weight in distribution
            ),
        ),
        feedback_mean=list(feedback),
    )


class OptionSearchModelTests(unittest.TestCase):
    def setUp(self):
        o0 = encode_action_trace([0])
        o1 = encode_action_trace([1])
        o2 = encode_action_trace([2])
        self.o0, self.o1, self.o2 = o0, o1, o2
        # S --o0--> A (2x) / B (1x) / censored (1x)
        # S --o1--> terminal
        # A --o2--> terminal ; B --o2--> terminal
        self.episodes = [
            [occurrence(0, ("S",), [0], duration=2.0),
             occurrence(0, ("A",), [2], duration=3.0,
                        feedback=(1.0, 0.0), terminal=True)],
            [occurrence(1, ("S",), [0], duration=2.0),
             occurrence(1, ("A",), [2], duration=3.0,
                        feedback=(1.0, 0.0), terminal=True)],
            [occurrence(2, ("S",), [0], duration=2.0),
             occurrence(2, ("B",), [2], duration=5.0,
                        feedback=(0.0, 2.0), terminal=True)],
            [occurrence(3, ("S",), [1], duration=2.0, terminal=True)],
            [occurrence(4, ("S",), [0], duration=2.0, censored=True)],
        ]
        self.memory = StageOutcomeMemory(
            min_support=1, min_episodes=1,
        ).fit(self.episodes)

    def search(self, max_transitions=2, **kwargs):
        search_args = {}
        for key in ("root_elapsed", "root_feedback"):
            if key in kwargs:
                search_args[key] = kwargs.pop(key)
        return StageOptionSearch(
            self.memory, max_transitions=max_transitions, **kwargs,
        ).search([("S",)], **search_args)

    def test_supported_options_keep_the_current_stage(self):
        self.assertEqual(
            self.memory.supported_options([("S",)]),
            (self.o0, self.o1),
        )
        self.assertEqual(
            self.memory.supported_options([("S",), ("A",)]),
            (self.o2,),
        )
        # The root pools every source stage, so o2 is only a marginal there.
        self.assertNotIn(
            self.o2, self.memory.supported_options([("S",)]),
        )
        self.assertIn(
            self.o2, self.memory.supported_options(
                [("S",)], min_context_depth=0,
            ),
        )

    def test_exhaustive_leaves_match_an_independent_enumeration(self):
        result = self.search()
        expected = sorted(
            (options, kind, round(probability, 12), context)
            for options, kind, probability, context
            in reference_leaves(self.memory, (("S",),), 2)
        )
        actual = sorted(
            (leaf.options, leaf.outcome, round(leaf.probability, 12),
             leaf.final_context)
            for leaf in result.leaves
        )
        self.assertEqual(actual, expected)
        self.assertTrue(result.complete)
        self.assertEqual(result.unresolved, ())

    def test_leaf_probabilities_are_exact_products(self):
        result = self.search()
        by_key = {
            (leaf.options, leaf.outcome, leaf.final_context): leaf
            for leaf in result.leaves
        }
        a_leaf = by_key[((self.o0, self.o2), "terminal", (("S",), ("A",)))]
        b_leaf = by_key[((self.o0, self.o2), "terminal", (("S",), ("B",)))]
        censored = by_key[((self.o0,), "censored", (("S",),))]
        terminal = by_key[((self.o1,), "terminal", (("S",),))]
        self.assertAlmostEqual(a_leaf.probability, 2.0 / 4.0)
        self.assertAlmostEqual(b_leaf.probability, 0.25)
        self.assertAlmostEqual(censored.probability, 0.25)
        self.assertAlmostEqual(terminal.probability, 1.0)
        # 0.5 + 0.25 + 0.25 covers the o0 option; o1 is a separate decision.
        self.assertAlmostEqual(
            a_leaf.probability + b_leaf.probability + censored.probability,
            1.0,
        )

    def test_terminal_and_censored_outcomes_stop_expansion(self):
        result = self.search()
        for leaf in result.leaves:
            if leaf.outcome in ("terminal", "censored"):
                self.assertTrue(
                    leaf.steps[-1].terminal or leaf.steps[-1].censored
                )
                self.assertEqual(
                    leaf.final_context, leaf.steps[-1].context,
                )
        self.assertEqual(
            result.counts["leaf_terminal"]
            + result.counts["leaf_censored"]
            + result.counts["leaf_open"],
            len(result.leaves),
        )

    def test_depth_limit_emits_open_leaves(self):
        result = self.search(max_transitions=1)
        self.assertTrue(result.complete)
        self.assertTrue(all(
            leaf.reason == "max_transitions"
            for leaf in result.leaves if leaf.outcome == "open"
        ))
        self.assertEqual(
            {leaf.final_context for leaf in result.leaves
             if leaf.outcome == "open"},
            {(("S",), ("A",)), (("S",), ("B",))},
        )
        self.assertEqual(
            result.counts["leaf_open"], 2,
        )

    def test_elapsed_and_feedback_describe_entered_stages_only(self):
        result = self.search(root_elapsed=7.0, root_feedback=(9.0, 9.0))
        by_key = {
            (leaf.options, leaf.final_context): leaf
            for leaf in result.leaves
        }
        a_leaf = by_key[((self.o0, self.o2), (("S",), ("A",)))]
        b_leaf = by_key[((self.o0, self.o2), (("S",), ("B",)))]
        censored = by_key[((self.o0,), (("S",),))]
        # A entered once (duration 3); the terminal outcome describes S's own
        # stage and must not be added again.
        self.assertAlmostEqual(a_leaf.elapsed_mean, 10.0)
        self.assertAlmostEqual(b_leaf.elapsed_mean, 12.0)
        self.assertAlmostEqual(censored.elapsed_mean, 7.0)
        self.assertEqual(a_leaf.feedback_mean, (10.0, 9.0))
        self.assertEqual(b_leaf.feedback_mean, (9.0, 11.0))
        self.assertEqual(censored.feedback_mean, (9.0, 9.0))

    def test_plans_group_by_option_sequence(self):
        result = self.search()
        plans = {plan.options: plan for plan in result.plans()}
        self.assertEqual(set(plans), {(self.o0,), (self.o0, self.o2),
                                      (self.o1,)})
        self.assertAlmostEqual(plans[(self.o0,)].known_mass, 0.25)
        self.assertAlmostEqual(
            plans[(self.o0, self.o2)].known_mass, 0.75,
        )
        self.assertEqual(
            dict(plans[(self.o0, self.o2)].outcome_mass),
            {"terminal": 0.75},
        )
        self.assertAlmostEqual(
            plans[(self.o0, self.o2)].expected_elapsed,
            (0.5 * 3.0 + 0.25 * 5.0) / 0.75,
        )

    def test_search_is_deterministic(self):
        first = self.search()
        second = self.search()
        self.assertEqual(
            [(leaf.options, leaf.outcome, leaf.probability,
              leaf.final_context) for leaf in first.leaves],
            [(leaf.options, leaf.outcome, leaf.probability,
              leaf.final_context) for leaf in second.leaves],
        )
        self.assertEqual(first.counts["queries"], second.counts["queries"])

    def test_every_edge_mass_is_checked(self):
        result = self.search()
        self.assertGreater(result.counts["edge_mass_checks"], 0)
        self.assertLess(result.counts["max_edge_deviation"], 1e-9)

    def test_snapshot_reports_the_contract(self):
        snapshot = self.search().snapshot()
        self.assertEqual(snapshot["max_transitions"], 2)
        self.assertEqual(snapshot["min_context_depth"], 1)
        self.assertTrue(snapshot["complete"])
        self.assertEqual(snapshot["unresolved_mass"], 0.0)
        self.assertIn("backoff_refused", snapshot["counts"])


class StopTimeDistributionTests(unittest.TestCase):
    def setUp(self):
        o0 = encode_action_trace([0])
        o2 = encode_action_trace([2])
        self.o0, self.o2 = o0, o2
        # S --o0--> A with A lasting 2 or 4 frames; A --o2--> terminal.
        self.episodes = [
            [occurrence(0, ("S",), [0], duration=1.0),
             occurrence(0, ("A",), [2], duration=2.0, terminal=True)],
            [occurrence(1, ("S",), [0], duration=1.0),
             occurrence(1, ("A",), [2], duration=4.0, terminal=True)],
        ]
        self.memory = StageOutcomeMemory(
            min_support=1, min_episodes=1,
        ).fit(self.episodes)

    def search(self, **kwargs):
        return StageOptionSearch(
            self.memory, max_transitions=2,
        ).search([("S",)], **kwargs)

    def test_memory_reports_the_realized_duration_histogram(self):
        prediction = self.memory.predict([("S",)], self.o0)
        row = next(
            row for row in prediction["outcomes"]
            if row["destination"] == ("A",)
        )
        self.assertEqual(
            row["duration"]["distribution"],
            ((2.0, 0.5), (4.0, 0.5)),
        )
        self.assertEqual(row["duration"]["count"], 2)
        self.assertAlmostEqual(row["duration"]["mean"], 3.0)

    def test_elapsed_is_a_convolution_not_a_mean(self):
        result = self.search()
        self.assertEqual(len(result.leaves), 1)
        leaf = result.leaves[0]
        self.assertEqual(leaf.options, (self.o0, self.o2))
        self.assertEqual(
            leaf.elapsed_distribution, ((2.0, 0.5), (4.0, 0.5)),
        )
        self.assertAlmostEqual(leaf.elapsed_mean, 3.0)
        self.assertAlmostEqual(leaf.steps[0].duration_mean, 3.0)
        self.assertEqual(
            leaf.steps[0].duration_distribution,
            ((2.0, 0.5), (4.0, 0.5)),
        )

    def test_root_offset_shifts_the_distribution(self):
        result = self.search(root_elapsed=1.0)
        distribution = result.leaves[0].elapsed_distribution
        self.assertEqual(distribution, ((3.0, 0.5), (5.0, 0.5)))
        self.assertAlmostEqual(distribution_mean(distribution), 4.0)
        self.assertEqual(distribution_quantile(distribution, 0.5), 3.0)
        self.assertEqual(distribution_quantile(distribution, 0.9), 5.0)

    def test_survival_curve_reports_exceedance_mass(self):
        self.assertEqual(
            survival_curve(((2.0, 0.5), (4.0, 0.5))),
            ((2.0, 0.5), (4.0, 0.0)),
        )

    def test_plan_elapsed_distribution_is_normalized(self):
        plan = self.search().plans()[0]
        self.assertAlmostEqual(
            sum(probability for _time, probability
                in plan.elapsed_distribution),
            1.0,
        )
        self.assertAlmostEqual(plan.expected_elapsed, 3.0)

    def test_multiple_outcomes_convolve_per_branch(self):
        o0 = encode_action_trace([0])
        memory = StubMemory(
            {
                ((("S",),), o0): prediction(
                    (("S",),),
                    [
                        outcome(destination=("A",), probability=0.5,
                                distribution=((1.0, 1.0),)),
                        outcome(destination=("B",), probability=0.5,
                                distribution=((2.0, 1.0),)),
                    ],
                ),
            },
            [o0],
        )
        result = exhaustive_option_search(
            memory, [("S",)], max_transitions=1,
        )
        by_context = {
            leaf.final_context: leaf.elapsed_distribution
            for leaf in result.leaves
        }
        self.assertEqual(by_context[(("S",), ("A",))], ((1.0, 1.0),))
        self.assertEqual(by_context[(("S",), ("B",))], ((2.0, 1.0),))


class OptionSearchContractTests(unittest.TestCase):
    def test_unsupported_child_is_unresolved_and_not_renormalized(self):
        o0 = encode_action_trace([0])
        memory = StubMemory(
            {
                ((("S",),), o0): prediction(
                    (("S",),),
                    [outcome(destination=("A",), probability=1.0)],
                ),
            },
            [o0],
        )
        result = exhaustive_option_search(
            memory, [("S",)], max_transitions=2,
        )
        self.assertEqual(result.leaves, ())
        self.assertEqual(len(result.unresolved), 1)
        node = result.unresolved[0]
        self.assertEqual(node.context, (("S",), ("A",)))
        self.assertEqual(node.options, (o0,))
        self.assertAlmostEqual(node.probability, 1.0)
        self.assertEqual(node.reason, "no_supported_option")
        self.assertAlmostEqual(result.unresolved_mass, 1.0)

    def test_unknown_option_is_never_fabricated(self):
        o0 = encode_action_trace([0])
        o9 = encode_action_trace([9])
        memory = StubMemory(
            {
                ((("S",),), o0): prediction(
                    (("S",),),
                    [outcome(destination=("A",), probability=1.0)],
                ),
                ((("S",), ("A",)), o0): prediction(
                    (("S",), ("A",)),
                    [outcome(terminal=True, probability=1.0)],
                ),
            },
            [o0, o9],
        )
        result = exhaustive_option_search(
            memory, [("S",)], max_transitions=2,
        )
        self.assertNotIn(o9, [leaf.options[-1] for leaf in result.leaves])
        self.assertEqual(result.counts["unknown_options"], 2)
        self.assertEqual(result.leaves[0].outcome, "terminal")

    def test_root_backoff_is_refused_by_default(self):
        o0 = encode_action_trace([0])

        class RootOnlyMemory:
            action_options = {o0}

            def predict(self, _context, _option):
                return prediction(
                    (), [outcome(terminal=True, probability=1.0)],
                )

        memory = RootOnlyMemory()
        strict = exhaustive_option_search(
            memory, [("S",)], max_transitions=1,
        )
        self.assertEqual(strict.leaves, ())
        self.assertEqual(strict.counts["backoff_refused"], 1)
        loose = exhaustive_option_search(
            memory, [("S",)], max_transitions=1, min_context_depth=0,
        )
        self.assertEqual(loose.leaves[0].outcome, "terminal")

    def test_edge_mass_must_sum_to_one(self):
        o0 = encode_action_trace([0])
        memory = StubMemory(
            {
                ((("S",),), o0): prediction(
                    (("S",),),
                    [
                        outcome(destination=("A",), probability=0.6),
                        outcome(destination=("B",), probability=0.3),
                    ],
                ),
            },
            [o0],
        )
        with self.assertRaises(ValueError):
            exhaustive_option_search(memory, [("S",)], max_transitions=1)

    def test_missing_duration_distribution_is_rejected(self):
        o0 = encode_action_trace([0])
        bare = dict(
            destination=("A",),
            terminal=False,
            censored=False,
            probability=1.0,
            count=1,
            episode_support=1,
            duration=dict(mean=2.0),
            feedback_mean=[0.0, 0.0],
        )
        memory = StubMemory(
            {((("S",),), o0): prediction((("S",),), [bare])},
            [o0],
        )
        with self.assertRaises(ValueError):
            exhaustive_option_search(memory, [("S",)], max_transitions=1)

    def test_duration_distribution_must_sum_to_one(self):
        o0 = encode_action_trace([0])
        memory = StubMemory(
            {
                ((("S",),), o0): prediction(
                    (("S",),),
                    [outcome(
                        destination=("A",), probability=1.0,
                        distribution=((2.0, 0.5), (4.0, 0.4)),
                    )],
                ),
            },
            [o0],
        )
        with self.assertRaises(ValueError):
            exhaustive_option_search(memory, [("S",)], max_transitions=1)

    def test_elapsed_support_capacity_is_not_truncated(self):
        o0 = encode_action_trace([0])
        distribution = tuple(
            (float(time), 1.0 / 20.0) for time in range(1, 21)
        )
        memory = StubMemory(
            {
                ((("S",),), o0): prediction(
                    (("S",),),
                    [outcome(
                        destination=("A",), probability=1.0,
                        distribution=distribution,
                    )],
                ),
            },
            [o0],
        )
        with self.assertRaises(StageOptionSearchCapacityError):
            exhaustive_option_search(
                memory, [("S",)], max_transitions=1,
                max_elapsed_support=5,
            )

    def test_capacity_guard_raises_instead_of_dropping_branches(self):
        o0 = encode_action_trace([0])
        o1 = encode_action_trace([1])

        class CyclicMemory:
            action_options = {o0, o1}

            def predict(self, context, _option):
                return prediction(context, [
                    outcome(destination=("A",), probability=0.5),
                    outcome(destination=("B",), probability=0.5),
                ])

        memory = CyclicMemory()
        with self.assertRaises(StageOptionSearchCapacityError):
            exhaustive_option_search(
                memory, [("S",)], max_transitions=6, max_leaves=4,
            )
        truncated = exhaustive_option_search(
            memory, [("S",)], max_transitions=6, max_leaves=4,
            allow_truncation=True,
        )
        self.assertFalse(truncated.complete)


if __name__ == "__main__":
    unittest.main()
