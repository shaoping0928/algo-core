"""Tests for demand-conditioned search ordering, separate from value."""
import unittest

import numpy as np

from stage_demand_heuristic import (
    DemandAssociationSample,
    StageDemand,
    StageDemandHeuristic,
)
from stage_outcome_memory import (
    KeySignal,
    StageOccurrence,
    encode_action_trace,
    segment_episode,
)


class StageDemandHeuristicTests(unittest.TestCase):
    def setUp(self):
        option = encode_action_trace([0, 0, 1])
        self.option = option
        self.samples = []
        for episode in range(5):
            self.samples.append(DemandAssociationSample(
                episode=episode,
                stage_history=(("source",),),
                active_features=("toward-target",),
                destination_stage=("target",),
                action_trace=option,
            ))
        for episode in range(5, 7):
            self.samples.append(DemandAssociationSample(
                episode=episode,
                stage_history=(("source",),),
                active_features=("other-direction",),
                destination_stage=("target",),
                action_trace=option,
            ))
        for episode in range(7, 10):
            self.samples.append(DemandAssociationSample(
                episode=episode,
                stage_history=(("source",),),
                active_features=("other-direction",),
                destination_stage=("other",),
                action_trace=option,
            ))
        self.memory = StageDemandHeuristic(
            min_support=2, min_episodes=2, prior=1.0,
        ).fit(self.samples)
        self.demand = StageDemand(
            target_stage=("target",),
            stage_history=(("source",),),
            action_trace=option,
            transition_probability=0.8,
            transition_support=5,
            transition_resolution=1,
        )

    def test_partial_association_ranks_features_for_demand(self):
        toward = self.memory.score(("toward-target",), self.demand)
        away = self.memory.score(("other-direction",), self.demand)
        self.assertEqual(toward["status"], "supported")
        self.assertGreater(toward["priority"], away["priority"])
        self.assertFalse(toward["affects_value"])
        self.assertFalse(toward["action_conditioned"])
        self.assertEqual(toward["provenance"], "observed_cooccurrence")

    def test_target_demand_can_be_created_from_a_sibling_prediction(self):
        demand = StageDemand.from_transition_prediction(
            prediction=dict(
                status="supported",
                support=8,
                resolution=2,
                outcomes=(
                    dict(
                        destination=("target",), terminal=False,
                        censored=False, probability=0.75,
                    ),
                ),
            ),
            target_stage=("target",),
            stage_history=(("source",),),
            action_trace=self.option,
        )
        self.assertEqual(demand.target_stage, ("target",))
        self.assertEqual(demand.transition_probability, 0.75)
        with self.assertRaises(ValueError):
            StageDemand.from_transition_prediction(
                prediction=dict(status="unknown", outcomes=()),
                target_stage=("target",),
                stage_history=(("source",),),
                action_trace=self.option,
            )

    def test_stage_occurrences_create_pointwise_realized_samples(self):
        stages = segment_episode(
            episode=44,
            observations=np.asarray([[0.0], [0.5], [1.0]]),
            actions=[0, 1],
            feedback=np.asarray([[0.0], [0.2]]),
            key_signals=[
                [KeySignal(0, "entry")],
                [KeySignal(1, "next-stage")],
                [],
            ],
            terminal=True,
            active_features=[("source-feature",), ("later-feature",)],
        )
        samples = DemandAssociationSample.from_stage_episodes([stages])
        self.assertEqual(len(samples), 1)
        self.assertEqual(
            samples[0].active_features, ("source-feature",)
        )
        self.assertEqual(samples[0].destination_stage, (("next-stage", 0),))
        # Future executed actions are not attached to earlier-state samples.
        self.assertIsNone(samples[0].action_trace)

    def test_context_backoff_preserves_the_selected_resolution(self):
        samples = [
            DemandAssociationSample(
                episode=0,
                stage_history=(("rare-prefix",), ("source",)),
                active_features=("toward-target",),
                destination_stage=("target",),
                action_trace=self.option,
            )
        ] + [
            DemandAssociationSample(
                episode=episode,
                stage_history=(("other-prefix",), ("source",)),
                active_features=("toward-target",),
                destination_stage=("target",),
                action_trace=self.option,
            )
            for episode in range(1, 4)
        ]
        memory = StageDemandHeuristic(
            min_support=2, min_episodes=2, prior=1.0,
        ).fit(samples)
        demand = StageDemand(
            target_stage=("target",),
            stage_history=(("rare-prefix",), ("source",)),
            action_trace=self.option,
        )
        result = memory.score(("toward-target",), demand)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["context"], (("source",),))
        self.assertEqual(result["resolution"], 1)

    def test_unseen_feature_or_option_returns_unknown(self):
        unseen_feature = self.memory.score(("unseen",), self.demand)
        self.assertEqual(unseen_feature["status"], "unknown")
        unseen_option = StageDemand(
            target_stage=("target",),
            stage_history=(("source",),),
            association_action_trace=encode_action_trace([1, 1, 1]),
        )
        self.assertEqual(
            self.memory.score(("toward-target",), unseen_option)["status"],
            "unknown",
        )

    def test_repeated_samples_in_one_episode_do_not_pass_support_gate(self):
        samples = [
            DemandAssociationSample(
                episode=21,
                stage_history=(("source",),),
                active_features=("toward-target",),
                destination_stage=("target",),
                action_trace=self.option,
            )
            for _ in range(5)
        ]
        memory = StageDemandHeuristic(
            min_support=2, min_episodes=2,
        ).fit(samples)
        result = memory.score(("toward-target",), self.demand)
        self.assertEqual(result["status"], "unknown")

    def test_prefix_samples_contain_only_actions_seen_so_far(self):
        source = StageOccurrence(
            episode=7,
            key=(("source",),),
            entry_state=np.zeros(1),
            exit_state=np.ones(1),
            action_trace=encode_action_trace([0, 1, 0]),
            duration=3.0,
            feedback=np.zeros(1),
            feature_snapshots=(
                (("f0",),), (("f1",),), (("f2",),),
            ),
            active_features=(("f0",),),
        )
        destination = StageOccurrence(
            episode=7,
            key=(("destination",),),
            entry_state=np.ones(1),
            exit_state=np.ones(1),
            action_trace=encode_action_trace([1]),
            duration=1.0,
            feedback=np.zeros(1),
            terminal=True,
        )
        samples = DemandAssociationSample.from_stage_prefixes(
            [[source, destination]]
        )
        self.assertEqual(len(samples), 3)
        self.assertEqual(
            [sample.action_trace for sample in samples],
            [
                None,
                encode_action_trace([0]),
                encode_action_trace([0, 1]),
            ],
        )
        self.assertEqual(
            [sample.active_features for sample in samples],
            [(("f0",),), (("f1",),), (("f2",),)],
        )
        self.assertTrue(all(
            sample.destination_stage == (("destination",),)
            for sample in samples
        ))

    def test_option_backoff_uses_a_shorter_known_prefix_suffix(self):
        samples = []
        for trace in (encode_action_trace([1]), encode_action_trace([0, 1])):
            for episode in range(3):
                samples.append(DemandAssociationSample(
                    episode=episode,
                    stage_history=(("source",),),
                    active_features=("feature",),
                    destination_stage=("target",),
                    action_trace=trace,
                ))
        memory = StageDemandHeuristic(
            min_support=1, min_episodes=1, prior=1.0,
        ).fit(samples)
        demand = StageDemand(
            target_stage=("target",),
            stage_history=(("source",),),
            association_action_trace=encode_action_trace([0, 0, 1]),
        )
        result = memory.score(("feature",), demand)
        self.assertEqual(result["status"], "supported")
        self.assertTrue(result["action_conditioned"])
        unknown = StageDemand(
            target_stage=("target",),
            stage_history=(("source",),),
            association_action_trace=encode_action_trace([0, 0, 0]),
        )
        # The pooled table is never used for an option-conditioned query.
        self.assertNotEqual(
            memory.score(("feature",), unknown)["status"], "supported",
        )


if __name__ == "__main__":
    unittest.main()
