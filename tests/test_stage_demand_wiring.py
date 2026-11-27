"""Tests for wiring stage demands into the generic beam rank channel."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stage_demand_heuristic import StageDemand
from substrate_nodes import (
    decision_action,
    evaluate,
    make_node_demand_rank_step,
    make_node_hybrid_rank_step,
    make_node_prototype_rank_step,
    make_node_root_rank_step,
    policy_result,
)


class FakeRuntime:
    obs_dim = 1
    scales = np.ones(1)
    feature_ids = ["aligned", "other"]
    _feature_all = ["aligned", "other"]

    @staticmethod
    def _feature_mask_table(observations):
        observations = np.asarray(observations, dtype=float)
        return [
            observations[:, 0] >= 0.5,
            observations[:, 0] < 0.5,
        ]


class FakeHeuristic:
    @staticmethod
    def score(features, _demand):
        if "aligned" in features:
            return dict(status="supported", priority=1.5)
        return dict(status="unknown", priority=None)


class FakeEnv:
    def __init__(self):
        self.t = 0
        self.dead = False
        self.passed = 0
        self.feedback_calls = 0

    def obs(self):
        return np.asarray([float(self.t)])

    def step(self, _action):
        self.t += 1
        self.dead = self.t >= 1

    def feedback(self):
        self.feedback_calls += 1
        return dict(dopamine=0.2, pain=0.1)


class FakeDemandProvider:
    def __init__(self, demand, heuristic):
        self.demand = demand
        self.heuristic = heuristic
        self.reset_args = None
        self.requests = []
        self.transitions = []

    def reset_episode(self, episode, seed):
        self.reset_args = (episode, seed)

    def __call__(self, _agent, observation, prefix, actions):
        self.requests.append((
            np.asarray(observation).copy(),
            np.asarray(prefix).copy(),
            tuple(actions),
        ))
        return self.demand, self.heuristic

    def observe_transition(self, transition):
        self.transitions.append(transition)


class StageDemandWiringTests(unittest.TestCase):
    def test_runtime_features_feed_the_separate_rank_channel(self):
        demand = StageDemand(
            target_stage=("target",),
            stage_history=(("source",),),
        )
        stats = {}
        rank_step = make_node_demand_rank_step(
            FakeRuntime(), demand, FakeHeuristic(), stats=stats,
        )
        states = np.asarray([[0.0], [1.0], [0.25]])
        priorities = rank_step(
            states, np.zeros((3, 2)), step=0,
        )
        np.testing.assert_array_equal(priorities, [0.0, 1.5, 0.0])
        self.assertEqual(stats["candidates"], 3)
        self.assertEqual(stats["supported"], 1)
        self.assertEqual(stats["nonzero"], 1)
        self.assertEqual(stats["unknown"], 2)

    def test_runtime_prototypes_feed_the_separate_rank_channel(self):
        class PrototypeRuntime:
            obs_dim = 1
            lag = 1
            scales = np.ones(2)

            @staticmethod
            def _nearest(state):
                return (
                    "p_aligned" if float(state[0]) >= 0.5 else "p_other",
                    0.0,
                )

        class PrototypeHeuristic:
            @staticmethod
            def score(features, _demand):
                if ("prototype", "p_aligned") in features:
                    return dict(status="supported", priority=1.5)
                return dict(status="unknown", priority=None)

        demand = StageDemand(
            target_stage=(("tree", "0"),),
            stage_history=((("tree", "0"),),),
            association_action_trace=((0, 1),),
        )
        rank_step = make_node_prototype_rank_step(
            PrototypeRuntime(), demand, PrototypeHeuristic(),
            pending_prefix=[0.0], lag=1,
        )
        states = np.asarray([[0.0], [1.0], [0.25]])
        sequences = np.zeros((3, 2))
        priorities = rank_step(states, sequences, step=0)
        np.testing.assert_array_equal(priorities, [0.0, 1.5, 0.0])

    def test_hybrid_ranker_combines_namespaced_branch_and_prototype_features(self):
        class HybridRuntime:
            obs_dim = 1
            lag = 1
            scales = np.ones(2)
            value_feature_ids = ["value_alignment"]
            _feature_all = ["value_alignment"]

            @staticmethod
            def _branch_active_features(observation):
                return ("f_high",) if float(observation[0]) >= 0.5 else ()

            @staticmethod
            def _feature_mask_table(observations):
                values = np.asarray(observations, dtype=float)
                return [values[:, 0] >= 0.5]

            @staticmethod
            def _nearest(state):
                return (
                    "p_high" if float(state[0]) >= 0.5 else "p_low",
                    0.0,
                )

        class HybridHeuristic:
            @staticmethod
            def score(features, _demand):
                if (("branch", "f_high") in features
                        and ("prototype", "p_high") in features
                        and ("value", "value_alignment") in features):
                    return dict(status="supported", priority=2.0)
                return dict(status="unknown", priority=None)

        demand = StageDemand(
            target_stage=(("tree", "0"),),
            stage_history=((("tree", "0"),),),
        )
        rank_step = make_node_hybrid_rank_step(
            HybridRuntime(), demand, HybridHeuristic(),
            pending_prefix=[0.0], lag=1,
        )
        priorities = rank_step(
            np.asarray([[0.0], [1.0]]),
            np.zeros((2, 2)),
            step=0,
        )
        np.testing.assert_array_equal(priorities, [0.0, 2.0])

    def test_root_resolver_features_feed_the_rank_channel(self):
        class RootResolver:
            @staticmethod
            def resolve(_agent, observation, _pending):
                if float(observation[0]) >= 0.5:
                    return dict(status="supported", key=(("stage",),))
                return dict(status="unknown", key=None)

        class RootTree:
            @staticmethod
            def chain(key):
                return (7,) if key == (("stage",),) else ()

        class RootHeuristic:
            @staticmethod
            def score(features, _demand):
                if ("tree", "7") in features:
                    return dict(status="supported", priority=2.0)
                return dict(status="unknown", priority=None)

        class RootRuntime:
            obs_dim = 1
            lag = 1
            scales = np.ones(2)
            stage_resolver = RootResolver()
            stage_tree = RootTree()

        demand = StageDemand(
            target_stage=(("tree", "7"),),
            stage_history=((("tree", "7"),),),
            association_action_trace=((0, 1),),
        )
        rank_step = make_node_root_rank_step(
            RootRuntime(), demand, RootHeuristic(),
            pending_prefix=[0.0], lag=1,
        )
        states = np.asarray([[0.0], [1.0], [0.25]])
        sequences = np.zeros((3, 2))
        priorities = rank_step(states, sequences, step=0)
        np.testing.assert_array_equal(priorities, [0.0, 2.0, 0.0])

    def test_unknown_root_rank_does_not_use_nearest_prototype_fallback(self):
        class UnknownResolver:
            @staticmethod
            def resolve(_agent, _observation, _pending):
                return dict(status="unknown", key=None)

        class RootTree:
            @staticmethod
            def chain(_key):
                return (7,)

        class RootHeuristic:
            @staticmethod
            def score(features, _demand):
                return dict(
                    status="supported" if ("tree", "7") in features
                    else "unknown",
                    priority=5.0 if ("tree", "7") in features else None,
                )

        class RootRuntime:
            obs_dim = 1
            lag = 1
            scales = np.ones(2)
            stage_resolver = UnknownResolver()
            stage_tree = RootTree()
            stage_prototype_roots = {"p0": ("tree", "7")}

            @staticmethod
            def _nearest(_state):
                return "p0", 0.0

        demand = StageDemand(
            target_stage=(("tree", "7"),),
            stage_history=((("tree", "7"),),),
        )
        rank_step = make_node_root_rank_step(
            RootRuntime(), demand, RootHeuristic(),
            pending_prefix=[0.0], lag=1,
        )
        priorities = rank_step(
            np.asarray([[1.0], [2.0]]),
            np.zeros((2, 2)),
            step=0,
        )
        np.testing.assert_array_equal(priorities, [0.0, 0.0])

    def test_policy_result_builds_ranker_from_demand_and_runtime_features(self):
        class Env:
            @staticmethod
            def obs():
                return np.asarray([0.0])

        class Heuristic:
            @staticmethod
            def score(features, _demand):
                if "aligned" in features:
                    return dict(status="supported", priority=2.0)
                return dict(status="unknown", priority=None)

        captured = {}

        def fake_beam_search(**kwargs):
            captured["rank_step"] = kwargs["rank_step"]
            return object()

        with patch("substrate_nodes.beam_search", side_effect=fake_beam_search):
            result = policy_result(
                Env(), None, FakeRuntime(), prefix=[0.0],
                horizon=2, lag=1, beam_width=2,
                stage_demand=StageDemand(
                    target_stage=("target",),
                    stage_history=(("source",),),
                ),
                demand_heuristic=Heuristic(),
            )
        self.assertIsNotNone(result)
        priorities = captured["rank_step"](
            np.asarray([[0.0], [1.0]]),
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            0,
        )
        np.testing.assert_array_equal(priorities, [0.0, 2.0])

    def test_policy_result_passes_rank_mode_and_margin(self):
        class Env:
            @staticmethod
            def obs():
                return np.asarray([0.0])

        captured = {}

        def fake_beam_search(**kwargs):
            captured.update(kwargs)
            return object()

        callback = lambda _states, _sequences, _step: np.zeros(1)
        with patch("substrate_nodes.beam_search", side_effect=fake_beam_search):
            policy_result(
                Env(), None, FakeRuntime(), prefix=[0.0],
                horizon=1, lag=1, beam_width=2,
                rank_step=callback, rank_mode="tie", rank_margin=0.05,
            )
        self.assertEqual(captured["rank_mode"], "tie")
        self.assertEqual(captured["rank_margin"], 0.05)
        self.assertEqual(captured["rank_group_quota"], 0)

    def test_policy_result_selects_chunked_search_only_when_requested(self):
        class Env:
            @staticmethod
            def obs():
                return np.asarray([0.0])

        captured = {}

        def fake_chunked_search(**kwargs):
            captured.update(kwargs)
            return object()

        with patch(
            "substrate_nodes.chunked_beam_search",
            side_effect=fake_chunked_search,
        ):
            policy_result(
                Env(), None, FakeRuntime(), prefix=[0.0],
                horizon=6, lag=1, beam_width=2,
                search_chunk_size=3,
            )
        self.assertEqual(captured["chunk_size"], 3)
        self.assertEqual(captured["rank_group_quota"], 1)

    def test_policy_result_requires_beam_for_demand_ranking(self):
        class Env:
            @staticmethod
            def obs():
                return np.asarray([0.0])

        with self.assertRaisesRegex(ValueError, "requires beam search"):
            policy_result(
                Env(), None, FakeRuntime(), prefix=[0.0],
                horizon=1, lag=1, beam_width=0,
                rank_step=lambda _states, _sequences, _step: np.zeros(1),
            )

    def test_decision_action_passes_the_rank_only_callback(self):
        callback = lambda _states, _sequences, _step: np.zeros(1)
        with patch(
            "substrate_nodes.policy_result",
            return_value=SimpleNamespace(actions=np.asarray([1.0])),
        ) as policy:
            result = decision_action(
                None, None, None, prefix=[0.0], horizon=1, lag=1,
                decision="search", beam_width=2, rank_step=callback,
            )
        self.assertEqual(result, 1)
        self.assertIs(policy.call_args.kwargs["rank_step"], callback)

    def test_evaluate_provider_sees_and_learns_only_real_transitions(self):
        env = FakeEnv()
        demand = StageDemand(
            target_stage=("target",),
            stage_history=(("source",),),
        )
        provider = FakeDemandProvider(demand, FakeHeuristic())
        agent = SimpleNamespace(obs_dim=1)
        with patch("substrate_nodes.decision_action", return_value=1) as call:
            frames, actions, transitions = evaluate(
                seed=3,
                world=None,
                agent=agent,
                cap=1,
                horizon=1,
                lag=1,
                episode=4,
                env_factory=lambda _seed, spacing=None: env,
                beam_width=2,
                stage_demand_provider=provider,
            )
        self.assertEqual((frames, actions, transitions), (1, [1], []))
        self.assertEqual(provider.reset_args, (4, 3))
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(len(provider.transitions), 1)
        realized = provider.transitions[0]
        self.assertEqual(realized["history"].tolist(), [1.0, 0.0])
        self.assertEqual(realized["signals"].tolist(), [0.2, 0.1])
        self.assertTrue(realized["done"])
        self.assertIs(call.call_args.kwargs["stage_demand"], demand)
        self.assertIs(
            call.call_args.kwargs["demand_heuristic"], provider.heuristic
        )
        self.assertEqual(env.feedback_calls, 1)


if __name__ == "__main__":
    unittest.main()
