"""Tests for the non-temporal (endogenous) state transition."""
import unittest

import numpy as np

from node_graph import (
    CONCEPT,
    DERIVED,
    MEMBERSHIP,
    MEMORY,
    NodeGraph,
)
from static_transition import (
    StaticState,
    endogenous_transition,
    integrate_episode,
    seed_state,
)


def build_graph():
    graph = NodeGraph()
    graph.add_node(CONCEPT, "a")
    graph.add_node(CONCEPT, "b")
    graph.add_node(CONCEPT, "c")
    graph.add_node(MEMORY, "m")
    graph.add_node(DERIVED, "d")
    graph.add_association("a", "b", weight=3.0)
    graph.add_membership("m", "a")
    graph.edges[("m", "a", MEMBERSHIP)].weight = 1.0
    graph.add_derivation("d", "a")
    graph.add_transition("a", "c", weight=5.0)
    return graph


class EndogenousTransitionTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_graph()

    def state(self, seeds, **kwargs):
        return endogenous_transition(
            self.graph, seeds=seeds, max_steps=1, decay=1.0,
            min_strength=0.01, **kwargs,
        )

    def test_association_membership_and_derivation_are_traversed(self):
        state = self.state(["a"])
        strengths = state.strengths()
        self.assertAlmostEqual(strengths["a"], 1.0)
        self.assertAlmostEqual(strengths["b"], 0.75)
        self.assertAlmostEqual(strengths["m"], 1.0)
        self.assertAlmostEqual(strengths["d"], 1.0)
        kinds = state.by_kind()
        self.assertEqual(kinds["memory"], ("m",))
        self.assertEqual(kinds["derived"], ("d",))

    def test_transition_edges_are_not_followed(self):
        state = self.state(["a"])
        self.assertNotIn("c", state.strengths())
        self.assertNotIn(
            "transition",
            dict(state.edge_counts),
        )

    def test_membership_is_retrieved_not_expanded(self):
        state = self.state(["m"])
        self.assertEqual(set(state.strengths()), {"m"})

    def test_derivation_is_forward_only(self):
        state = self.state(["d"])
        self.assertEqual(set(state.strengths()), {"d"})

    def test_depth_zero_is_the_seed_only(self):
        state = seed_state(self.graph, seeds=["a"])
        self.assertEqual(set(state.strengths()), {"a"})
        self.assertTrue(state.converged)

    def test_decay_scales_each_hop(self):
        state = endogenous_transition(
            self.graph, seeds=["a"], max_steps=1, decay=0.5,
            min_strength=0.01,
        )
        self.assertAlmostEqual(state.strengths()["b"], 0.375)

    def test_min_strength_drops_weak_nodes(self):
        state = endogenous_transition(
            self.graph, seeds=["a"], max_steps=1, decay=0.1,
            min_strength=0.2,
        )
        self.assertEqual(set(state.strengths()), {"a"})

    def test_two_hops_reach_associated_memory(self):
        self.graph.add_node(CONCEPT, "x")
        self.graph.add_association("m", "x", weight=1.0)
        state = endogenous_transition(
            self.graph, seeds=["a"], max_steps=2, decay=1.0,
            min_strength=0.01,
        )
        self.assertIn("x", state.strengths())
        self.assertFalse(state.converged)
        self.assertEqual(state.snapshot()["nodes"], 5)

    def test_convergence_is_reported(self):
        state = endogenous_transition(
            self.graph, seeds=["a"], max_steps=5, decay=1.0,
            min_strength=0.01,
        )
        self.assertTrue(state.converged)

    def test_state_change_tokens_are_signed(self):
        first = seed_state(self.graph, seeds=["a"])
        second = seed_state(self.graph, seeds=["b"])
        tokens = set(second.change(first, min_change=0.5))
        self.assertEqual(tokens, {("a", -1), ("b", 1)})

    def test_unknown_seed_is_rejected(self):
        with self.assertRaises(KeyError):
            endogenous_transition(self.graph, seeds=["missing"])

    def test_integrate_episode_is_sequential_and_deterministic(self):
        states = integrate_episode(
            self.graph, [["a"], ["b"], []], max_steps=1,
        )
        self.assertEqual(len(states), 3)
        self.assertEqual(states[2].nodes, ())
        again = integrate_episode(
            self.graph, [["a"], ["b"], []], max_steps=1,
        )
        self.assertEqual(
            [state.snapshot() for state in states],
            [state.snapshot() for state in again],
        )


class AgentSeedingTests(unittest.TestCase):
    def test_state_is_encoded_with_the_runtime_scales(self):
        graph = build_graph()

        class StubAgent:
            obs_dim = 1
            scales = np.asarray([2.0])

            def _active_feature_ids(self, encoded):
                return ["a"] if encoded[0] > 0.5 else []

        agent = StubAgent()
        agent.graph = graph
        state = endogenous_transition(
            agent, state=[2.0], max_steps=0,
        )
        self.assertEqual(state.seeds, ("a",))
        empty = endogenous_transition(
            agent, state=[0.5], max_steps=0,
        )
        self.assertEqual(empty.nodes, ())

    def test_empty_seed_returns_an_empty_state(self):
        class StubAgent:
            obs_dim = 1
            scales = np.asarray([1.0])
            graph = build_graph()

            def _active_feature_ids(self, _encoded):
                return []

        state = endogenous_transition(StubAgent(), state=[0.0])
        self.assertIsInstance(state, StaticState)
        self.assertEqual(state.nodes, ())


if __name__ == "__main__":
    unittest.main()
