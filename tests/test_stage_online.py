"""Tests for the online stage context and demand provider."""
import unittest

import numpy as np

from stage_demand_heuristic import (
    StageDemand,
    StageDemandAlternatives,
)
from stage_experience import StageExperienceBatch
from stage_online import OnlineStageDemandProvider, OnlineStageTracker


class FakeAgent:
    def __init__(self, lag=2):
        self.lag = int(lag)
        self.scales = np.ones(2 + self.lag)

    def _branch_active_features(self, obs):
        return ()

    def _nearest(self, state):
        return "p0", 0.0


def make_episode(episode=100, lag=2):
    zero = np.zeros(2)
    spike = np.asarray([5.0, 0.0])
    ys = [0.0, 6.0, 16.0, 16.0, 16.0]
    actions = [0, 1, 0, 1]
    feedback = [zero, zero, spike, zero]
    rows = []
    for step, action in enumerate(actions):
        issued = 1 - int(action)
        history = [float(issued), 0.0, float(action)][:lag + 1]
        last = step == len(actions) - 1
        rows.append(dict(
            episode=int(episode),
            obs=np.asarray([ys[step], 0.0]),
            next_obs=np.asarray([ys[step + 1], 0.0]),
            history=np.asarray(history, dtype=float),
            signals=np.asarray(feedback[step], dtype=float),
            terminal=False,
            done=False,
        ))
        rows[-1]["terminal"] = bool(last)
        rows[-1]["done"] = bool(last)
    return rows


class StubMemory:
    def predict_pooled(self, history):
        return dict(status="supported", outcomes=(), support=4)


class StubSelector:
    def __init__(self):
        self.calls = 0

    def select(self, prediction, history, action_trace=None,
               association_action_trace=None):
        self.calls += 1
        demand = StageDemand(
            target_stage=("destination",),
            stage_history=history,
            action_trace=None,
            association_action_trace=association_action_trace,
            transition_probability=0.6,
            transition_support=4,
            transition_resolution=1,
        )
        return demand, dict(status="supported")


class StubTree:
    def chain(self, key):
        if key == ("destination",):
            return (0,)
        if key == ():
            return (0,)
        if key and str(key[0]).startswith("destination-"):
            return (int(str(key[0]).split("-")[-1]),)
        return ()


class StubHeuristic:
    pass


class SequenceSelector:
    def __init__(self):
        self.calls = 0

    def select(self, prediction, history, action_trace=None,
               association_action_trace=None):
        index = self.calls
        self.calls += 1
        return StageDemand(
            target_stage=("destination-%d" % index,),
            stage_history=history,
            association_action_trace=association_action_trace,
            transition_probability=0.8,
            transition_support=prediction["support"],
            transition_episode_support=4,
            demand_strength=0.7,
        ), dict(status="supported")


class ProposalSelector:
    def __init__(self, value_confidence=0.9):
        self.readout = type("Readout", (), {"target_std": 1.0})()
        self.min_outcome_episodes = 2
        self.value_confidence = float(value_confidence)

    def select(self, _prediction, history, action_trace=None,
               association_action_trace=None):
        demand = StageDemand(
            target_stage=("destination-0",),
            stage_history=history,
            association_action_trace=association_action_trace,
            transition_probability=0.6,
            transition_support=4,
            transition_episode_support=4,
            demand_strength=0.7,
        )
        return demand, dict(
            status="supported",
            candidates=(
                dict(
                    destination=("destination-0",), value=2.0,
                    probability=0.6, support=4, episode_support=4,
                ),
                dict(
                    destination=("destination-1",), value=1.5,
                    probability=0.3, support=3, episode_support=3,
                ),
            ),
            value_confidence=self.value_confidence,
        )


class SequenceTracker:
    def __init__(self, contexts, episode=100):
        self.contexts = list(contexts)
        self.episode = episode

    def context(self):
        return self.contexts.pop(0)

    def observe_transition(self, _transition):
        return True

    def reset(self, episode=None):
        self.episode = episode
        return self


class OnlineStageTests(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.rows = make_episode()
        self.experience = StageExperienceBatch.fit(
            self.rows, self.agent, lag=2, quantile=0.6,
            feature_source="prototype",
        )

    def test_tracker_context_uses_only_observed_transitions(self):
        tracker = OnlineStageTracker(self.experience.detector, lag=2)
        for row in self.rows[:2]:
            self.assertTrue(tracker.observe_transition(row))
        partial = tracker.context()
        self.assertIsNotNone(partial)
        self.assertLessEqual(
            sum(count for _action, count in partial["prefix"]),
            2,
        )
        for row in self.rows[2:]:
            tracker.observe_transition(row)
        full = tracker.context()
        self.assertEqual(
            full["history"],
            tuple(
                stage.key for stage in self.experience.episodes[0]
            ),
        )
        self.assertEqual(
            full["prefix"],
            self.experience.episodes[0][-1].action_trace,
        )
        count = sum(value for _action, value in full["prefix"])
        self.assertGreater(count, 0)

    def test_tracker_reset_drops_the_previous_episode(self):
        tracker = OnlineStageTracker(self.experience.detector, lag=2)
        for row in self.rows:
            tracker.observe_transition(row)
        self.assertIsNotNone(tracker.context())
        tracker.reset()
        self.assertIsNone(tracker.context())

    def test_provider_maps_target_to_tree_root_and_causal_prefix(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = StubSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        for row in self.rows:
            provider.observe_transition(row)
        request = provider(agent=self.agent, observation=None, prefix=(), actions=())
        self.assertIsNotNone(request)
        demand, heuristic = request
        self.assertIs(heuristic, self.agent.stage_tree_heuristic)
        self.assertEqual(demand.target_stage, (("tree", "0"),))
        self.assertEqual(
            demand.association_action_trace,
            self.experience.episodes[0][-1].action_trace,
        )
        self.assertIsNone(demand.action_trace)
        self.assertGreaterEqual(provider.snapshot()["counts"]["supported"], 1)

    def test_provider_confidence_gate_returns_unknown(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = StubSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(
            self.agent, min_target_probability=0.7,
        )
        for row in self.rows:
            provider.observe_transition(row)
        self.assertIsNone(provider(
            agent=self.agent, observation=None, prefix=(), actions=(),
        ))
        self.assertEqual(
            provider.snapshot()["counts"]["low_target_probability"], 1,
        )
        support_provider = OnlineStageDemandProvider(
            self.agent, min_prediction_support=5,
        )
        for row in self.rows:
            support_provider.observe_transition(row)
        self.assertIsNone(support_provider(
            agent=self.agent, observation=None, prefix=(), actions=(),
        ))
        self.assertEqual(
            support_provider.snapshot()["counts"][
                "low_prediction_support"
            ],
            1,
        )

    def test_provider_requires_fitted_components(self):
        self.agent.stage_experience = self.experience
        with self.assertRaises(ValueError):
            OnlineStageDemandProvider(self.agent)

    def test_provider_holds_one_demand_until_the_next_stage_boundary(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        selector = SequenceSelector()
        self.agent.stage_demand_selector = selector
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "hybrid"
        provider = OnlineStageDemandProvider(self.agent)
        provider.tracker = SequenceTracker([
            dict(
                stages=(object(),), history=(("source",),),
                prefix=((0, 1),),
            ),
            dict(
                stages=(object(),), history=(("source-updated",),),
                prefix=((0, 2),),
            ),
            dict(
                stages=(object(), object()),
                history=(("source",), ("next",)),
                prefix=((1, 1),),
            ),
        ])

        first = provider(self.agent, np.zeros(2), np.zeros(2), ())
        repeated = provider(self.agent, np.zeros(2), np.zeros(2), ())
        next_stage = provider(self.agent, np.zeros(2), np.zeros(2), ())

        self.assertEqual(first[0].target_stage, (("tree", "0"),))
        self.assertIs(repeated[0], first[0])
        self.assertEqual(next_stage[0].target_stage, (("tree", "1"),))
        self.assertEqual(selector.calls, 2)
        counts = provider.snapshot()["counts"]
        self.assertEqual(counts["intent_started"], 2)
        self.assertEqual(counts["intent_reused"], 1)
        self.assertEqual(counts["intent_released_stage_boundary"], 1)

    def test_provider_records_realized_target_match_once_at_stage_boundary(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = SequenceSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
            dict(
                stages=(
                    type("Stage", (), {"key": ("source",)})(),
                    type(
                        "Stage", (), {"key": ("destination-0",)}
                    )(),
                ),
                history=(("source",), ("destination-0",)),
                prefix=((1, 1),),
            ),
        ], episode=701)

        provider(self.agent, np.zeros(2), np.zeros(2), ())
        provider(self.agent, np.zeros(2), np.zeros(2), ())

        self.assertEqual(provider.strategy_credit.snapshot()["observations"], 1)
        self.assertEqual(provider.last_validation["episode"], 701)
        self.assertEqual(provider.last_validation["similarity"], 1.0)
        self.assertEqual(
            provider.last_validation["reason"], "target_stage_match",
        )

    def test_provider_limits_proposals_to_supported_stage_slate(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = ProposalSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(
            self.agent, max_strategy_proposals=2,
        )
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
        ], episode=705)

        demand_set, _heuristic = provider(
            self.agent, np.zeros(2), np.zeros(2), (),
        )

        self.assertIsInstance(demand_set, StageDemandAlternatives)
        self.assertEqual(len(demand_set.demands), 2)
        self.assertEqual(
            demand_set.target_stages,
            ((("tree", "0"),), (("tree", "1"),)),
        )

    def test_realized_secondary_proposal_gets_credit_without_false_negative(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = ProposalSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
            dict(
                stages=(
                    type("Stage", (), {"key": ("source",)})(),
                    type(
                        "Stage", (), {"key": ("destination-1",)}
                    )(),
                ),
                history=(("source",), ("destination-1",)),
                prefix=((1, 1),),
            ),
        ], episode=706)

        provider(self.agent, np.zeros(2), np.zeros(2), ())
        provider(self.agent, np.zeros(2), np.zeros(2), ())

        source_history = ((("tree", "base"),),)
        primary = (("tree", "0"),)
        secondary = (("tree", "1"),)
        self.assertEqual(
            provider.strategy_credit.estimate(
                source_history, primary,
            )["mean_similarity"],
            0.0,
        )
        self.assertEqual(
            provider.strategy_credit.estimate(
                source_history, secondary,
            )["mean_similarity"],
            1.0,
        )
        self.assertEqual(
            provider.last_validation["matched_proposals"],
            (secondary,),
        )

    def test_provider_uses_prior_episode_credit_only_for_rank_strength(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = SequenceSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        source_history = ((("tree", "base"),),)
        target = (("tree", "0"),)
        provider.strategy_credit.record(source_history, target, 1, 1.0)
        provider.strategy_credit.record(source_history, target, 2, 1.0)
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
        ], episode=703)

        demand, _heuristic = provider(
            self.agent, np.zeros(2), np.zeros(2), (),
        )

        self.assertGreater(demand.demand_strength, 0.7)
        self.assertEqual(
            demand.demand_strength,
            provider.last["demand_strength"],
        )

    def test_stable_primary_and_clear_value_margin_collapse_the_slate(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = ProposalSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        source_history = ((("tree", "base"),),)
        target = (("tree", "0"),)
        provider.strategy_credit.record(source_history, target, 1, 1.0)
        provider.strategy_credit.record(source_history, target, 2, 1.0)
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
        ], episode=707)

        demand, _heuristic = provider(
            self.agent, np.zeros(2), np.zeros(2), (),
        )

        self.assertIsInstance(demand, StageDemand)
        self.assertEqual(provider.last["proposal_trigger"], "primary_stable")
        self.assertEqual(provider.last["proposal_limit"], 1)

    def test_primary_mismatch_expands_the_strategy_slate(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = ProposalSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        source_history = ((("tree", "base"),),)
        target = (("tree", "0"),)
        provider.strategy_credit.record(source_history, target, 1, 0.0)
        provider.strategy_credit.record(source_history, target, 2, 0.0)
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
        ], episode=708)

        demand_set, _heuristic = provider(
            self.agent, np.zeros(2), np.zeros(2), (),
        )

        self.assertIsInstance(demand_set, StageDemandAlternatives)
        self.assertEqual(provider.last["proposal_trigger"], "primary_mismatch")
        self.assertEqual(provider.last["proposal_limit"], 3)

    def test_intermediate_value_confidence_keeps_two_proposals(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = ProposalSelector(
            value_confidence=0.6,
        )
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        source_history = ((("tree", "base"),),)
        target = (("tree", "0"),)
        provider.strategy_credit.record(source_history, target, 1, 1.0)
        provider.strategy_credit.record(source_history, target, 2, 1.0)
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
        ], episode=709)

        demand_set, _heuristic = provider(
            self.agent, np.zeros(2), np.zeros(2), (),
        )

        self.assertIsInstance(demand_set, StageDemandAlternatives)
        self.assertEqual(provider.last["proposal_limit"], 2)
        self.assertEqual(len(demand_set.demands), 2)

    def test_provider_reset_uses_environment_episode_id(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = StubSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)

        provider.reset_episode(0, 1200007)

        self.assertEqual(provider.tracker.episode, 1200007)

    def test_strategy_credit_is_persistent_on_runtime(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = StubSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        first = OnlineStageDemandProvider(self.agent)
        second = OnlineStageDemandProvider(self.agent)

        self.assertIs(first.strategy_credit, second.strategy_credit)

    def test_terminal_transition_is_a_real_strategy_mismatch(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = SequenceSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
        ], episode=702)
        provider(self.agent, np.zeros(2), np.zeros(2), ())

        provider.observe_transition(dict(done=True, terminal=True))

        self.assertEqual(provider.strategy_credit.snapshot()["observations"], 1)
        self.assertEqual(provider.last_validation["similarity"], 0.0)
        self.assertEqual(
            provider.last_validation["reason"], "terminal_before_target",
        )
        self.assertIsNone(provider.snapshot()["active_intent"])

    def test_episode_reset_censors_unresolved_intent_without_credit(self):
        self.agent.stage_experience = self.experience
        self.agent.stage_outcome_memory = StubMemory()
        self.agent.stage_demand_selector = SequenceSelector()
        self.agent.stage_tree = StubTree()
        self.agent.stage_tree_heuristic = StubHeuristic()
        self.agent.stage_demand_feature_source = "prototype"
        provider = OnlineStageDemandProvider(self.agent)
        provider.tracker = SequenceTracker([
            dict(
                stages=(type("Stage", (), {"key": ("source",)})(),),
                history=(("source",),), prefix=((0, 1),),
            ),
        ], episode=703)
        provider(self.agent, np.zeros(2), np.zeros(2), ())

        provider.reset_episode(0, 704)

        self.assertEqual(provider.strategy_credit.snapshot()["observations"], 0)
        self.assertEqual(
            provider.snapshot()["counts"]["intent_censored_episode_reset"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
