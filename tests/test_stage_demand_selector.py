"""Tests for the world-feedback stage-demand selector."""
import unittest
from unittest.mock import patch

import numpy as np

from stage_demand_selector import (
    StageDemandSelector,
    StageOutcomeReadout,
    remaining_net_feedback,
)
from stage_outcome_memory import (
    StageOccurrence,
    StageOutcomeMemory,
    encode_action_trace,
)


OPTION = encode_action_trace([0, 0])


def stage(episode, key, feedback, duration=2.0, terminal=False,
          action_trace=OPTION):
    return StageOccurrence(
        episode=episode,
        key=(key,),
        entry_state=np.zeros(2),
        exit_state=np.ones(2),
        action_trace=action_trace,
        duration=duration,
        feedback=np.asarray(feedback, dtype=float),
        terminal=terminal,
    )


def positive_episodes(count=4):
    episodes = []
    episode = 0
    for _ in range(count):
        episodes.append([
            stage(episode, "S", (0.0, 0.0)),
            stage(episode, "A", (2.0, 0.0)),
            stage(episode, "A_TAIL", (1.0, 0.0), terminal=True),
        ])
        episode += 1
    for _ in range(count):
        episodes.append([
            stage(episode, "S", (0.0, 0.0)),
            stage(episode, "B", (0.0, 2.0)),
            stage(episode, "B_TAIL", (0.0, 1.0), terminal=True),
        ])
        episode += 1
    return episodes


class StageDemandSelectorTests(unittest.TestCase):
    def setUp(self):
        self.episodes = positive_episodes()
        self.readout = StageOutcomeReadout.fit(
            self.episodes, folds=2, ridge=1.0,
        )
        self.selector = StageDemandSelector(
            self.readout, min_outcome_episodes=2, min_samples=4,
        )
        self.memory = StageOutcomeMemory(
            min_support=2, min_episodes=2,
        ).fit(self.episodes)

    def test_readout_is_fit_on_fold_out_candidates(self):
        self.assertGreaterEqual(self.readout.samples, 4)
        self.assertGreater(self.readout.target_std, 0.0)

    def test_readout_training_uses_the_online_pooled_query(self):
        option_a = encode_action_trace([0, 0])
        option_b = encode_action_trace([1, 1])
        episodes = []
        for episode in range(8):
            destination = "A" if episode < 4 else "B"
            option = option_a if episode < 4 else option_b
            reward = (2.0, 0.0) if destination == "A" else (0.0, 2.0)
            episodes.append([
                stage(episode, "S", (0.0, 0.0), action_trace=option),
                stage(episode, destination, reward),
                stage(
                    episode, "TAIL", (0.0, 0.0), terminal=True,
                ),
            ])
        with patch.object(
            StageOutcomeMemory, "predict",
            side_effect=AssertionError("readout must use pooled queries"),
        ):
            readout = StageOutcomeReadout.fit(episodes, folds=2)
        self.assertEqual(readout.samples, 16)

        memory = StageOutcomeMemory(
            min_support=2, min_episodes=2,
        ).fit(episodes)
        pooled = memory.predict_pooled([("S",)])
        exact = memory.predict([("S",)], option_a)
        self.assertTrue(pooled["option_pooled"])
        self.assertFalse(exact["option_pooled"])
        self.assertGreater(pooled["support"], exact["support"])
        selector = StageDemandSelector(
            readout, min_outcome_episodes=2, min_samples=4,
        )
        demand, report = selector.select(pooled, [("S",)])
        self.assertIsNotNone(demand)
        self.assertEqual(report["status"], "supported")

    def test_selector_picks_the_higher_learned_feedback_sibling(self):
        prediction = self.memory.predict(
            [("S",)], OPTION,
        )
        self.assertEqual(prediction["status"], "supported")
        demand, report = self.selector.select(
            prediction, [("S",)], OPTION,
        )
        self.assertEqual(report["status"], "supported")
        self.assertIsNotNone(demand)
        self.assertEqual(demand.target_stage, ("A",))
        self.assertEqual(demand.stage_history, (("S",),))
        self.assertEqual(demand.action_trace, OPTION)
        self.assertGreater(report["value"], report["runner_up_value"])
        self.assertGreater(report["margin"], 0.0)
        self.assertGreater(demand.demand_strength, 0.0)
        self.assertEqual(
            demand.transition_episode_support,
            report["candidates"][0]["episode_support"],
        )
        self.assertEqual(
            report["demand_strength"], demand.demand_strength,
        )
        self.assertFalse(report["affects_value"])
        self.assertEqual(
            report["provenance"], "world_feedback_outcome_readout"
        )

    def test_terminal_siblings_are_never_demanded(self):
        episodes = []
        for episode in range(4):
            episodes.append([
                stage(episode, "S", (0.0, 0.0)),
                stage(episode, "A", (0.5, 0.0)),
                stage(episode, "T", (9.0, 0.0), terminal=True),
            ])
        memory = StageOutcomeMemory(
            min_support=2, min_episodes=2,
        ).fit(episodes)
        prediction = memory.predict([("S",)], OPTION)
        demand, report = self.selector.select(
            prediction, [("S",)], OPTION,
        )
        self.assertIsNotNone(demand)
        self.assertEqual(demand.target_stage, ("A",))
        self.assertTrue(all(
            not candidate["destination"] == ("T",)
            for candidate in report["candidates"]
        ))

    def test_terminal_only_context_returns_none(self):
        episodes = []
        for episode in range(4):
            episodes.append([
                stage(episode, "S", (0.0, 0.0)),
                stage(episode, "T", (1.0, 0.0), terminal=True),
            ])
        memory = StageOutcomeMemory(
            min_support=2, min_episodes=2,
        ).fit(episodes)
        prediction = memory.predict([("S",)], OPTION)
        demand, report = self.selector.select(
            prediction, [("S",)], OPTION,
        )
        self.assertIsNone(demand)
        self.assertEqual(
            report["reason"], "no_supported_nonterminal_sibling"
        )

    def test_empty_readout_returns_unknown(self):
        readout = StageOutcomeReadout.fit([], folds=2)
        selector = StageDemandSelector(readout)
        prediction = self.memory.predict([("S",)], OPTION)
        demand, report = selector.select(
            prediction, [("S",)], OPTION,
        )
        self.assertIsNone(demand)
        self.assertEqual(report["reason"], "readout_unsupported")

    def test_demand_uses_the_exact_scored_candidate_row(self):
        def outcome(probability, feedback):
            return dict(
                destination=("A",),
                terminal=False,
                censored=False,
                destination_terminal_rate=0.0,
                destination_censored_rate=0.0,
                probability=probability,
                support=4,
                episode_support=2,
                duration=dict(mean=2.0, q90=2.0),
                feedback_mean=list(feedback),
                feedback_variance=[0.0, 0.0],
            )

        prediction = dict(
            status="supported",
            support=8,
            resolution=1,
            outcomes=(
                outcome(0.1, (0.0, 0.0)),
                outcome(0.9, (2.0, 0.0)),
            ),
        )
        demand, report = self.selector.select(
            prediction, [("S",)], OPTION,
        )
        self.assertIsNotNone(demand)
        self.assertEqual(
            demand.transition_probability,
            report["candidates"][0]["probability"],
        )
        self.assertEqual(demand.stage_history, (("S",),))
        self.assertEqual(demand.action_trace, OPTION)

    def test_selection_does_not_mutate_the_fitted_objects(self):
        before = self.selector.snapshot()
        prediction = self.memory.predict([("S",)], OPTION)
        self.selector.select(prediction, [("S",)], OPTION)
        self.assertEqual(self.selector.snapshot(), before)

    def test_remaining_feedback_is_realized_not_counterfactual(self):
        stages = self.episodes[0]
        self.assertAlmostEqual(remaining_net_feedback(stages, 0), 3.0)
        self.assertAlmostEqual(remaining_net_feedback(stages, 1), 1.0)
        self.assertAlmostEqual(remaining_net_feedback(stages, 2), 0.0)


if __name__ == "__main__":
    unittest.main()
