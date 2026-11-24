"""Tests for learned continuous strengths on the non-temporal graph."""
import unittest

from node_graph import CONCEPT, MEMBERSHIP, MEMORY, NodeGraph
from static_graph_strengths import (
    StaticGraphStrengths,
    learn_static_strengths,
)
from static_transition import endogenous_transition


class AssociationLearningTests(unittest.TestCase):
    def test_jaccard_strength_and_support_gate(self):
        states = [
            {"a", "b"},
            {"a", "b"},
            {"a"},
            {"b", "c"},
        ]
        learned = learn_static_strengths(states, min_support=2)
        self.assertAlmostEqual(learned.association("a", "b"), 0.5)
        self.assertEqual(
            learned.association("b", "c"), 0.0,
        )
        self.assertEqual(learned.support[("a", "b")], 2)
        self.assertEqual(learned.snapshot()["associations"], 1)

    def test_association_is_symmetric(self):
        learned = learn_static_strengths(
            [{"a", "b"}, {"a", "b"}], min_support=2,
        )
        self.assertAlmostEqual(learned.association("a", "b"), 1.0)
        self.assertAlmostEqual(learned.association("b", "a"), 1.0)

    def test_membership_defaults_to_a_structural_binding(self):
        graph = NodeGraph()
        graph.add_node(CONCEPT, "a")
        graph.add_node(MEMORY, "m")
        graph.add_membership("m", "a")
        self.assertEqual(graph.edges[("m", "a", MEMBERSHIP)].weight, 0.0)
        learned = learn_static_strengths(
            [{"a", "a"}, {"a", "a"}], graph=graph, min_support=1,
        )
        self.assertAlmostEqual(learned.member_strength("m", "a"), 1.0)
        # Learning is read-only: the runtime edge keeps its original weight.
        self.assertEqual(graph.edges[("m", "a", MEMBERSHIP)].weight, 0.0)

    def test_explicit_membership_mapping_wins(self):
        learned = learn_static_strengths(
            [{"a"}], min_support=1,
            membership={("m", "a"): 0.25},
        )
        self.assertAlmostEqual(learned.member_strength("m", "a"), 0.25)


class OverlayTraversalTests(unittest.TestCase):
    def setUp(self):
        self.graph = NodeGraph()
        for node in ("a", "b", "c"):
            self.graph.add_node(CONCEPT, node)

    def overlay(self):
        return StaticGraphStrengths(
            associations={("a", "b"): 0.8, ("b", "c"): 0.7},
            membership={},
        )

    def test_overlay_reaches_associations_the_graph_never_stored(self):
        without = endogenous_transition(
            self.graph, seeds=["a"], max_steps=2, decay=1.0,
            min_strength=0.01,
        )
        self.assertEqual(set(without.strengths()), {"a"})
        with_overlay = endogenous_transition(
            self.graph, seeds=["a"], max_steps=2, decay=1.0,
            min_strength=0.01, strengths=self.overlay(),
        )
        strengths = with_overlay.strengths()
        self.assertAlmostEqual(strengths["b"], 0.8)
        self.assertAlmostEqual(strengths["c"], 0.56)
        self.assertEqual(
            dict(with_overlay.edge_counts)["association"], 3,
        )

    def test_overlay_depth_is_bounded(self):
        one_hop = endogenous_transition(
            self.graph, seeds=["a"], max_steps=1, decay=1.0,
            min_strength=0.01, strengths=self.overlay(),
        )
        self.assertEqual(set(one_hop.strengths()), {"a", "b"})
        self.assertIn("c", endogenous_transition(
            self.graph, seeds=["a"], max_steps=2, decay=1.0,
            min_strength=0.01, strengths=self.overlay(),
        ).strengths())

    def test_normalization_spreads_mass_over_neighbours(self):
        overlay = StaticGraphStrengths(
            associations={("a", "b"): 0.6, ("a", "c"): 0.2},
            normalized=True,
        )
        values = dict(overlay.associations_of("a"))
        self.assertAlmostEqual(values["b"], 0.75)
        self.assertAlmostEqual(values["c"], 0.25)
        self.assertAlmostEqual(sum(values.values()), 1.0)
        raw = StaticGraphStrengths(
            associations={("a", "b"): 0.6, ("a", "c"): 0.2},
            normalized=False,
        )
        self.assertAlmostEqual(raw.association("a", "b"), 0.6)

    def test_graph_weights_are_untouched(self):
        before = {
            key: edge.weight for key, edge in self.graph.edges.items()
        }
        endogenous_transition(
            self.graph, seeds=["a"], max_steps=2,
            strengths=self.overlay(),
        )
        after = {
            key: edge.weight for key, edge in self.graph.edges.items()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
