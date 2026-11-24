"""Tests for predictive-equivalence classes over realized action options."""
import unittest

import numpy as np

from predictive_option_quotient import FeedbackBins, OptionQuotient
from stage_option_search import StageOptionSearch
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


class IdenticalOptionMergeTests(unittest.TestCase):
    def setUp(self):
        self.o_a = encode_action_trace([0, 0])
        self.o_b = encode_action_trace([1, 1])
        self.o_c = encode_action_trace([2, 2])
        self.o_a_stage = encode_action_trace([3])
        self.episodes = [
            [occurrence(0, ("S",), [0, 0], duration=2.0),
             occurrence(0, ("A",), [3], duration=3.0,
                        feedback=(1.0, 0.0), terminal=True)],
            [occurrence(1, ("S",), [1, 1], duration=2.0),
             occurrence(1, ("A",), [3], duration=3.0,
                        feedback=(1.0, 0.0), terminal=True)],
            [occurrence(2, ("S",), [2, 2], duration=2.0, terminal=True)],
        ]
        self.memory = StageOutcomeMemory(
            min_support=1, min_episodes=1,
        ).fit(self.episodes)
        self.quotient = OptionQuotient.fit(
            self.episodes, self.episodes,
            prior=4.0, tolerance=0.05,
            min_support=1, min_episodes=1,
        )

    def test_identical_successor_distributions_merge(self):
        self.assertTrue(self.quotient.accepted)
        self.assertEqual(len(self.quotient.classes), 3)
        class_a = self.quotient.class_for(("S",), self.o_a)
        class_b = self.quotient.class_for(("S",), self.o_b)
        class_c = self.quotient.class_for(("S",), self.o_c)
        self.assertIsNotNone(class_a)
        self.assertEqual(class_a, class_b)
        self.assertNotEqual(class_a, class_c)
        members = self.quotient.classes[class_a].members
        self.assertEqual(set(members), {self.o_a, self.o_b})

    def test_class_distribution_preserves_successor_and_stop_time(self):
        class_id = self.quotient.class_for(("S",), self.o_a)
        prediction = self.quotient.predict_class(class_id)
        self.assertEqual(prediction["status"], "supported")
        self.assertEqual(prediction["context_depth"], 1)
        self.assertEqual(len(prediction["outcomes"]), 1)
        row = prediction["outcomes"][0]
        self.assertEqual(row["destination"], ("A",))
        self.assertAlmostEqual(row["probability"], 1.0)
        self.assertEqual(
            row["duration"]["distribution"], ((3.0, 1.0),),
        )
        self.assertEqual(row["feedback_mean"], [1.0, 0.0])

    def test_unseen_trace_is_unknown(self):
        self.assertIsNone(self.quotient.class_for(
            ("S",), encode_action_trace([9, 9]),
        ))
        result = self.quotient.predict(
            ("S",), encode_action_trace([9, 9]),
        )
        self.assertEqual(result["status"], "unknown")

    def test_class_mode_search_expands_one_branch_per_class(self):
        search = StageOptionSearch(
            self.memory, max_transitions=1,
            option_quotient=self.quotient,
        )
        result = search.search([("S",)])
        self.assertEqual(result.snapshot()["option_mode"], "class")
        self.assertEqual(result.counts["option_vocabulary"], 3)
        self.assertEqual(len(result.leaves), 2)
        outcomes = sorted(leaf.outcome for leaf in result.leaves)
        self.assertEqual(outcomes, ["open", "terminal"])
        for leaf in result.leaves:
            self.assertEqual(len(leaf.options), 1)
            token = leaf.options[0]
            self.assertEqual(token[0], "option_class")
            self.assertIsNotNone(leaf.steps[0].option_class)
        open_leaf = next(
            leaf for leaf in result.leaves if leaf.outcome == "open"
        )
        self.assertEqual(open_leaf.elapsed_distribution, ((3.0, 1.0),))

    def test_rejected_quotient_is_refused(self):
        rejected = OptionQuotient(
            prior=4.0, tolerance=0.05,
            min_support=1, min_episodes=1,
        )
        with self.assertRaises(ValueError):
            StageOptionSearch(
                self.memory, max_transitions=1,
                option_quotient=rejected,
            )


class FeedbackHistogramTests(unittest.TestCase):
    def test_bins_partition_the_training_range(self):
        values = [np.asarray([float(value)]) for value in range(10)]
        bins = FeedbackBins.fit(values, bins=4)
        self.assertEqual(bins.dimensions, 1)
        self.assertEqual(bins.index(-1.0, 0), 0)
        self.assertEqual(bins.index(100.0, 0), len(bins.edges[0]) - 2)
        counts = bins.counts(values, 0)
        self.assertEqual(float(counts.sum()), 10.0)
        self.assertEqual(
            counts[bins.index(9.0, 0)], counts.max(),
        )

    def test_histogram_measures_a_mixture_more_cheaply_than_a_gaussian(self):
        o_a = encode_action_trace([0, 0])
        o_b = encode_action_trace([1, 1])
        episodes = [
            [occurrence(0, ("S",), [0, 0], duration=2.0),
             occurrence(0, ("A",), [3], duration=3.0,
                        feedback=(0.0, 0.0), terminal=True)],
            [occurrence(1, ("S",), [1, 1], duration=2.0),
             occurrence(1, ("A",), [3], duration=3.0,
                        feedback=(10.0, 0.0), terminal=True)],
        ]
        quotient = OptionQuotient.fit(
            episodes, episodes,
            prior=4.0, tolerance=0.7,
            min_support=1, min_episodes=1,
            acceptance_tolerance=1.0,
        )
        self.assertEqual(len(quotient.merge_log), 1)
        # A 0-vs-10 feedback split costs ~log(2) per sample as a histogram;
        # as a single Gaussian it would cost several nats.
        self.assertGreater(quotient.merge_log[0]["delta"], 0.5)
        self.assertLess(quotient.merge_log[0]["delta"], 0.8)
        self.assertEqual(quotient.class_for(("S",), o_a),
                         quotient.class_for(("S",), o_b))
        self.assertTrue(quotient.accepted)

    def test_snapshot_reports_the_feedback_model(self):
        o_a = encode_action_trace([0])
        episodes = [
            [occurrence(0, ("S",), [0], duration=2.0),
             occurrence(0, ("A",), [3], duration=3.0,
                        feedback=(1.0, 0.0), terminal=True)],
        ]
        quotient = OptionQuotient.fit(
            episodes, episodes, prior=4.0, tolerance=0.05,
            min_support=1, min_episodes=1, feedback_bins=4,
        )
        snapshot = quotient.snapshot()
        self.assertEqual(snapshot["feedback_model"], "histogram")
        self.assertEqual(snapshot["feedback_bins"], 4)
        self.assertEqual(snapshot["feedback_dimensions"], 2)


class DivergentOptionTests(unittest.TestCase):
    def setUp(self):
        self.o_a = encode_action_trace([0, 0])
        self.o_b = encode_action_trace([1, 1])
        episodes = []
        for index in range(5):
            episodes.append([
                occurrence(index, ("S",), [0, 0], duration=2.0),
                occurrence(index, ("A",), [3], duration=3.0,
                           feedback=(1.0, 0.0), terminal=True),
            ])
        for index in range(5, 10):
            episodes.append([
                occurrence(index, ("S",), [1, 1], duration=2.0),
                occurrence(index, ("B",), [3], duration=3.0,
                           feedback=(1.0, 0.0), terminal=True),
            ])
        self.episodes = episodes
        self.memory = StageOutcomeMemory(
            min_support=1, min_episodes=1,
        ).fit(episodes)

    def test_default_tolerance_keeps_different_successors_apart(self):
        quotient = OptionQuotient.fit(
            self.episodes, self.episodes,
            prior=4.0, tolerance=0.05,
            min_support=1, min_episodes=1,
        )
        self.assertEqual(len(quotient.classes), 4)
        self.assertNotEqual(
            quotient.class_for(("S",), self.o_a),
            quotient.class_for(("S",), self.o_b),
        )

    def test_aggressive_merge_is_rejected_by_the_held_out_gate(self):
        quotient = OptionQuotient.fit(
            self.episodes, self.episodes,
            prior=4.0, tolerance=1.0,
            min_support=1, min_episodes=1,
        )
        self.assertEqual(len(quotient.classes), 3)
        self.assertFalse(quotient.accepted)
        self.assertLess(quotient.acceptance_delta, -0.2)
        with self.assertRaises(ValueError):
            StageOptionSearch(
                self.memory, max_transitions=1,
                option_quotient=quotient,
            )

    def test_fit_is_deterministic(self):
        first = OptionQuotient.fit(
            self.episodes, self.episodes,
            prior=4.0, tolerance=1.0,
            min_support=1, min_episodes=1,
        )
        second = OptionQuotient.fit(
            self.episodes, self.episodes,
            prior=4.0, tolerance=1.0,
            min_support=1, min_episodes=1,
        )
        self.assertEqual(
            {key: value.members for key, value in first.classes.items()},
            {key: value.members for key, value in second.classes.items()},
        )
        self.assertEqual(first.snapshot(), second.snapshot())


if __name__ == "__main__":
    unittest.main()
