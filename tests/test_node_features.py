import unittest

import numpy as np

from node_features import Composite, FeatureRuleBank, Primitive
from node_graph import CONCEPT, NodeGraph
from node_runtime import NodeRuntime


class FeatureRuleTests(unittest.TestCase):
    def test_primitive_and_composite_masks(self):
        x = np.asarray([
            [0.0, 0.5], [0.01, 0.5], [0.0, 0.0], [0.5, 0.0],
        ])
        near = Primitive(0, True, "<", 0.05)
        outside = Composite("or", (
            Primitive(1, False, ">", 0.2),
            Primitive(1, False, "<", -0.2),
        ))
        collision = Composite("and", (near, outside))
        np.testing.assert_array_equal(
            collision.mask(x), [True, True, False, False]
        )

    def test_bank_recovers_and_composition(self):
        rng = np.random.RandomState(7)
        x = rng.uniform(-1.0, 1.0, (2400, 4))
        y = (
            (np.abs(x[:, 0]) < 0.08) & (np.abs(x[:, 1]) > 0.30)
        ).astype(float)
        train = np.arange(1700)
        test = np.arange(1700, 2400)
        bank = FeatureRuleBank()
        rules = bank.induce(x[train], y[train], x[test], y[test])
        self.assertTrue(rules)
        self.assertGreaterEqual(rules[0].test_precision, 0.8)
        positive = bank.activate(x[y > 0][:20])
        negative = bank.activate(x[y == 0][:20])
        self.assertGreater(positive.mean(), negative.mean())

    def test_runtime_rules_raise_predicted_pain(self):
        rng = np.random.RandomState(11)
        x = rng.uniform(-1.0, 1.0, (1200, 2))
        y = (
            (np.abs(x[:, 0]) < 0.08) & (np.abs(x[:, 1]) > 0.30)
        ).astype(float)
        runtime = NodeRuntime(
            obs_dim=2, lag=0, scales=np.ones(2),
            birth_threshold=0.01, max_nodes=10, exploration=0.0,
        )
        runtime.induce_rules(x[:900], y[:900], x[900:], y[900:])
        states = np.asarray([[0.0, 0.5], [0.5, 0.0]])
        risk = runtime.pain_signal_batch(states)
        self.assertGreater(risk[0], risk[1])

    def test_feature_node_predicate_survives_serialization(self):
        graph = NodeGraph()
        predicate = {
            "kind": "composite", "op": "and",
            "children": [
                {"kind": "primitive", "dim": 0, "magnitude": True,
                 "op": "<", "threshold": 0.1},
                {"kind": "primitive", "dim": 1, "magnitude": False,
                 "op": ">", "threshold": 0.2},
            ],
        }
        node_id = graph.add_node(
            CONCEPT, "feature_test", role="feature",
            predicate=predicate,
        )
        restored = NodeGraph.from_state(graph.to_state())
        self.assertEqual(restored.nodes[node_id].predicate, predicate)
        self.assertEqual(restored.nodes[node_id].role, "feature")

    def test_feature_nodes_join_memory_and_action_edges(self):
        rng = np.random.RandomState(13)
        x = rng.uniform(-1.0, 1.0, (1200, 2))
        y = (
            (np.abs(x[:, 0]) < 0.08) & (np.abs(x[:, 1]) > 0.30)
        ).astype(float)
        runtime = NodeRuntime(
            obs_dim=2, lag=0, scales=np.ones(2),
            birth_threshold=0.01, max_nodes=10, exploration=0.0,
        )
        runtime.induce_rules(x[:900], y[:900], x[900:], y[900:])
        runtime.install_feature_rules()
        self.assertTrue(runtime.feature_ids)
        state = np.asarray([0.0, 0.5])
        active = runtime._active_feature_ids(state)
        self.assertTrue(active)
        feature_id = active[0]

        runtime.learn_transition(
            state, 0, np.asarray([0.5, 0.0]), [0.0, 1.0]
        )
        edges = runtime.graph.action_edges_from(feature_id, 0)
        self.assertTrue(edges)
        memories = runtime.memories_for([feature_id])
        self.assertTrue(memories)
        memory_id = next(iter(memories))
        self.assertIn(feature_id, runtime.graph.nodes[memory_id].members)
        self.assertGreater(
            runtime.pain_signal_batch(state.reshape(1, -1))[0], 0.0
        )


if __name__ == "__main__":
    unittest.main()
