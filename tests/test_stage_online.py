"""Tests for the online stage context and demand provider."""
import unittest

import numpy as np

from stage_demand_heuristic import StageDemand
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
        return ()


class StubHeuristic:
    pass


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

    def test_provider_requires_fitted_components(self):
        self.agent.stage_experience = self.experience
        with self.assertRaises(ValueError):
            OnlineStageDemandProvider(self.agent)


if __name__ == "__main__":
    unittest.main()
