import unittest

import numpy as np

from graph_memory import PrototypeGraphMemory
from node_graph import (
    ASSOCIATION,
    EDGE,
    MEMORY,
    TRANSITION_EDGE,
    NodeGraph,
)


class NodeGraphTests(unittest.TestCase):
    def test_association_has_no_edge_node(self):
        graph = NodeGraph()
        a = graph.add_node("concept")
        b = graph.add_node("concept")
        edge = graph.add_association(a, b)
        self.assertEqual(edge.kind, ASSOCIATION)
        self.assertIsNone(edge.edge_node)
        self.assertEqual(graph.snapshot()["edge_nodes"], 0)

    def test_coactivation_forms_association_and_recall(self):
        graph = NodeGraph()
        a = graph.add_node("concept")
        b = graph.add_node("concept")
        graph.activate([a, b], [1.0, 1.0])
        graph.observe_association([a, b], rate=0.5)
        graph.observe_association([a, b], rate=0.5)
        edge = graph.edges[(a, b, ASSOCIATION)]
        self.assertGreater(edge.weight, 0.0)
        recalled = graph.recall([a], steps=1)
        self.assertGreater(recalled[b], 0.0)

    def test_transition_creates_edge_node_and_participates_in_memory(self):
        graph = NodeGraph()
        a = graph.add_node("concept")
        b = graph.add_node("concept")
        graph.activate([a, b], [1.0, 1.0])
        edge = graph.observe_transition(a, b, rate=0.5)
        self.assertIsNotNone(edge.edge_node)
        edge_node = graph.nodes[edge.edge_node]
        self.assertEqual(edge_node.kind, EDGE)
        context = graph.add_node("concept")
        graph.add_association(edge.edge_node, context)
        memory = graph.add_memory([a, edge.edge_node, b])
        self.assertEqual(graph.nodes[memory].kind, MEMORY)
        self.assertIn(edge.edge_node, graph.nodes[memory].members)

    def test_derivation_depth_is_bounded(self):
        graph = NodeGraph()
        source = graph.add_node("concept")
        first = graph.add_derived(source, max_depth=2)
        second = graph.add_derived(first, max_depth=2)
        with self.assertRaises(ValueError):
            graph.add_derived(second, max_depth=2)

    def test_prune_removes_low_weight_association(self):
        graph = NodeGraph()
        a = graph.add_node("concept")
        b = graph.add_node("concept")
        graph.add_association(a, b, weight=1e-6)
        removed = graph.prune(min_association=1e-3)
        self.assertEqual(len(removed), 1)
        self.assertNotIn((a, b, ASSOCIATION), graph.edges)
        self.assertNotIn((a, b, ASSOCIATION), graph.outgoing[a])
        self.assertNotIn((a, b, ASSOCIATION), graph.incoming[a])

    def test_transition_edge_node_can_be_forgotten(self):
        graph = NodeGraph()
        a = graph.add_node("concept")
        b = graph.add_node("concept")
        edge = graph.observe_transition(a, b, rate=0.5)
        edge_node = edge.edge_node
        graph.nodes[edge_node].trace = 0.0
        removed = graph.prune(min_edge_trace=1e-4)
        self.assertIn((a, b, TRANSITION_EDGE), removed)
        self.assertNotIn(edge_node, graph.nodes)

    def test_prune_removes_edges_and_membership_referencing_edge_node(self):
        graph = NodeGraph()
        a = graph.add_node("concept")
        b = graph.add_node("concept")
        c = graph.add_node("concept")
        edge = graph.observe_transition(a, b, rate=0.5)
        edge_node = edge.edge_node
        graph.nodes[edge_node].trace = 0.0
        association = graph.add_association(edge_node, c, weight=0.5)
        memory = graph.add_memory([edge_node, c])
        graph.prune(min_edge_trace=1e-4)

        self.assertNotIn(edge_node, graph.nodes)
        self.assertNotIn(association.key(), graph.edges)
        self.assertNotIn(edge_node, graph.nodes[memory].members)
        for node_id in (edge_node, c):
            self.assertNotIn(association.key(),
                             graph.outgoing.get(node_id, []))
            self.assertNotIn(association.key(),
                             graph.incoming.get(node_id, []))
        # Recall must not walk an edge whose endpoint no longer exists.
        graph.recall([c], steps=1)

    def test_prototype_graph_memory_creates_memories_and_edge_nodes(self):
        rows = []
        for episode in range(4):
            for step in range(6):
                rows.append(dict(
                    obs=np.asarray([float(step), float(episode)], dtype=float),
                    terminal=step == 5,
                    episode=episode,
                ))
        memory = PrototypeGraphMemory(clusters=3, seed=2).fit(rows)
        snapshot = memory.snapshot()
        self.assertGreater(snapshot["edge_nodes"], 0)
        self.assertGreater(snapshot["memories"], 0)
        risk = memory.risk(rows)
        self.assertEqual(risk.shape, (len(rows),))
        self.assertTrue(np.isfinite(risk).all())
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        direct = memory.risk_states(observations[:5])
        self.assertEqual(direct.shape, (5,))
        self.assertTrue(np.isfinite(direct).all())
        values = memory.value_states(observations[:5])
        self.assertEqual(values.shape, (5,))
        self.assertTrue(np.isfinite(values).all())

    def test_prototype_graph_memory_fit_is_repeatable(self):
        rows = [
            dict(
                obs=np.asarray([float(step), 0.0], dtype=float),
                next_obs=np.asarray([float(step + 1), 0.0], dtype=float),
                terminal=step == 5,
                episode=0,
            )
            for step in range(6)
        ] + [
            dict(
                obs=np.asarray([float(step), 1.0], dtype=float),
                next_obs=np.asarray([float(step + 1), 1.0], dtype=float),
                terminal=step == 5,
                episode=1,
            )
            for step in range(6)
        ]
        memory = PrototypeGraphMemory(clusters=3, seed=2)
        first = memory.fit(rows).snapshot()
        second = memory.fit(rows).snapshot()
        self.assertEqual(first, second)

    def test_transition_edge_values_accumulate_pain(self):
        rows = []
        cycle = ((0.0, 1.0, True), (1.0, 2.0, False),
                 (2.0, 3.0, False), (3.0, 0.0, False))
        for _ in range(20):
            for obs, next_obs, terminal in cycle:
                rows.append(dict(
                    obs=np.asarray([obs], dtype=float),
                    next_obs=np.asarray([next_obs], dtype=float),
                    terminal=terminal,
                    episode=0,
                ))
        memory = PrototypeGraphMemory(clusters=4, seed=3, rate=0.2).fit(rows)
        danger = memory.prototype(np.asarray([[0.0]]))[0]
        successor = memory.prototype(np.asarray([[1.0]]))[0]
        safe = memory.prototype(np.asarray([[1.0]]))[0]
        safe_successor = memory.prototype(np.asarray([[2.0]]))[0]
        danger_node = memory.graph.edge_node(
            "concept_%d" % danger, "concept_%d" % successor
        )
        safe_node = memory.graph.edge_node(
            "concept_%d" % safe, "concept_%d" % safe_successor
        )
        self.assertIsNotNone(danger_node)
        self.assertIsNotNone(safe_node)
        self.assertGreater(memory.graph.nodes[danger_node].value, 0.7)
        self.assertLess(memory.graph.nodes[safe_node].value, 0.3)


if __name__ == "__main__":
    unittest.main()
