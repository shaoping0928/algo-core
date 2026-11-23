"""Tests for the generic, signal-delimited stage outcome memory."""
import unittest

import numpy as np

from stage_outcome_memory import (
    GenericKeySignalDetector,
    UNKNOWN,
    KeySignal,
    StageOccurrence,
    StageOutcomeMemory,
    encode_action_trace,
    segment_episode,
    stage_context_path,
)


def occurrence(episode, key, actions, duration=2.0, feedback=(0.0, 0.0),
               terminal=False, censored=False):
    entry = float(len(key))
    return StageOccurrence(
        episode=episode,
        key=tuple(key),
        entry_state=np.asarray([entry]),
        exit_state=np.asarray([entry + 1.0]),
        action_trace=encode_action_trace(actions),
        duration=duration,
        feedback=np.asarray(feedback, dtype=float),
        terminal=terminal,
        censored=censored,
    )


class SignalSegmentationTests(unittest.TestCase):
    def test_detector_requires_training_episodes(self):
        with self.assertRaises(ValueError):
            GenericKeySignalDetector().fit([])

    def test_detector_learns_thresholds_from_training_episodes_only(self):
        detector = GenericKeySignalDetector(quantile=0.75)
        train = dict(
            observations=np.asarray([[0.0], [0.1], [0.2], [0.3], [4.3]]),
            feedback=np.zeros((4, 1)),
            prediction_error=np.asarray([0.0, 0.0, 0.1, 0.0]),
        )
        detector.fit([train])
        fitted_thresholds = dict(detector.thresholds)

        heldout_obs = np.asarray([[0.0], [0.1], [0.2], [0.3], [4.3]])
        heldout_error = np.asarray([0.0, 0.0, 0.0, 3.0])
        signals = detector.transform_episode(
            heldout_obs,
            feedback=np.zeros((4, 1)),
            prediction_error=heldout_error,
        )
        at_spike = {signal.channel for signal in signals[4]}
        self.assertIn(("observation_change", 0), at_spike)
        self.assertIn(("prediction_error", 0), at_spike)
        self.assertEqual(detector.thresholds, fitted_thresholds)
        self.assertEqual(signals[1], ())

    def test_sustained_key_signal_creates_one_boundary_on_rising_edge(self):
        observations = np.arange(6, dtype=float)[:, None]
        actions = [0, 0, 1, 1, 0]
        feedback = np.asarray([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.0, 1.0], [1.0, 0.0],
        ])
        alpha = KeySignal(0, "signal-a", polarity=1)
        beta = KeySignal(2, "signal-b", polarity=-1)
        signals = [
            [alpha], [KeySignal(1, "signal-a", polarity=1)],
            [KeySignal(2, "signal-a", polarity=1), beta],
            [], [KeySignal(4, "signal-a", polarity=1)], [],
        ]
        stages = segment_episode(
            episode=9,
            observations=observations,
            actions=actions,
            feedback=feedback,
            key_signals=signals,
            terminal=True,
        )
        self.assertEqual([stage.duration for stage in stages], [2.0, 2.0, 1.0])
        self.assertEqual(stages[0].key, (("signal-a", 1),))
        # The sustained A pulse does not re-trigger at step 2; only new B does.
        self.assertEqual(stages[1].key, (("signal-b", -1),))
        self.assertEqual(stages[2].key, (("signal-a", 1),))
        self.assertEqual(stages[1].signal_strengths, (1.0,))
        self.assertEqual(stages[0].action_trace, ((0, 2),))
        self.assertEqual(stages[1].action_trace, ((1, 2),))
        self.assertTrue(stages[-1].terminal)
        self.assertFalse(stages[-1].censored)
        np.testing.assert_array_equal(stages[0].feedback, [2.0, 0.0])

    def test_right_censoring_is_recorded_on_the_final_stage(self):
        stages = segment_episode(
            episode=2,
            observations=np.asarray([[0.0], [1.0], [2.0]]),
            actions=[0, 1],
            feedback=np.asarray([[0.2], [0.3]]),
            key_signals=[[], [KeySignal(1, "change")], []],
            censored=True,
        )
        self.assertEqual(len(stages), 2)
        self.assertFalse(stages[-1].terminal)
        self.assertTrue(stages[-1].censored)


class StageOutcomeMemoryTests(unittest.TestCase):
    def test_context_depth_cap_can_select_root_only(self):
        self.assertEqual(
            stage_context_path([("A",), ("B",)], max_depth=0),
            ((),),
        )

    def test_context_path_runs_from_specific_history_to_root(self):
        self.assertEqual(
            stage_context_path([("P",), ("A",), ("B",)]),
            (
                (("P",), ("A",), ("B",)),
                (("A",), ("B",)),
                (("B",),),
                (),
            ),
        )

    def setUp(self):
        self.option = encode_action_trace([0, 0])
        self.episodes = [
            [
                occurrence(0, ("X",), [1]),
                occurrence(0, ("A",), [0, 0], duration=4,
                           feedback=(1.0, 0.0)),
                occurrence(0, ("B",), [1], censored=True),
            ],
            [
                occurrence(1, ("Y",), [1]),
                occurrence(1, ("A",), [0, 0], duration=6,
                           feedback=(0.0, 1.0)),
                occurrence(1, ("B",), [1], censored=True),
            ],
            [
                occurrence(2, ("Y",), [1]),
                occurrence(2, ("A",), [0, 0], duration=8,
                           feedback=(0.0, 1.0)),
                occurrence(2, ("C",), [1], censored=True),
            ],
        ]

    def test_backoff_predicts_only_observed_sibling_transitions(self):
        memory = StageOutcomeMemory(
            min_support=2, min_episodes=2,
        ).fit(self.episodes)
        result = memory.predict(
            stage_history=[("X",), ("A",)],
            action_trace=self.option,
        )
        self.assertEqual(result["status"], "supported")
        # The exact X->A context has one episode; it backs off to A.
        self.assertEqual(result["context"], (("A",),))
        self.assertEqual(result["context_depth"], 1)
        self.assertEqual(result["support"], 3)
        self.assertEqual(result["episode_support"], 3)
        self.assertEqual(set(result["siblings"]), {("B",), ("C",)})
        by_destination = {
            row["destination"]: row for row in result["outcomes"]
        }
        self.assertAlmostEqual(by_destination[("B",)]["probability"], 2 / 3)
        self.assertAlmostEqual(by_destination[("C",)]["probability"], 1 / 3)
        self.assertAlmostEqual(by_destination[("B",)]["duration"]["mean"], 5.0)
        self.assertEqual(by_destination[("B",)]["feedback_mean"], [0.5, 0.5])

    def test_action_order_is_preserved_and_unsupported_option_is_unknown(self):
        memory = StageOutcomeMemory(
            min_support=1, min_episodes=1,
        ).fit(self.episodes)
        self.assertEqual(
            encode_action_trace([0, 0, 1, 1, 0]),
            ((0, 2), (1, 2), (0, 1)),
        )
        result = memory.predict(
            stage_history=[("A",)],
            action_trace=encode_action_trace([1, 1]),
        )
        self.assertEqual(result["status"], UNKNOWN)
        self.assertEqual(result["outcomes"], ())

    def test_repeated_rows_in_one_episode_do_not_pass_episode_gate(self):
        # Repeated occurrences from one episode are pooled together but only
        # contribute one independent episode to the support gate.
        episode = [
            occurrence(5, ("A",), [0, 0]),
            occurrence(5, ("B",), [1]),
            occurrence(5, ("A",), [0, 0]),
            occurrence(5, ("B",), [1], censored=True),
        ]
        memory = StageOutcomeMemory(
            min_support=2, min_episodes=2,
        ).fit([episode])
        result = memory.predict(
            stage_history=[("A",)],
            action_trace=self.option,
        )
        self.assertEqual(result["status"], UNKNOWN)

    def test_terminal_and_censored_outcomes_remain_distinct(self):
        episodes = [
            [occurrence(10, ("A",), [0], terminal=True)],
            [occurrence(11, ("A",), [0], censored=True)],
        ]
        memory = StageOutcomeMemory(
            min_support=2, min_episodes=2,
        ).fit(episodes)
        result = memory.predict(
            stage_history=[("A",)],
            action_trace=encode_action_trace([0]),
        )
        self.assertEqual(result["status"], "supported")
        self.assertAlmostEqual(result["terminal_rate"], 0.5)
        self.assertAlmostEqual(result["censored_rate"], 0.5)
        self.assertEqual(
            {(row["terminal"], row["censored"])
             for row in result["outcomes"]},
            {(True, False), (False, True)},
        )

    def test_no_training_data_returns_unknown(self):
        result = StageOutcomeMemory().fit([]).predict(
            stage_history=[("A",)],
            action_trace=encode_action_trace([0]),
        )
        self.assertEqual(result["status"], UNKNOWN)


if __name__ == "__main__":
    unittest.main()
