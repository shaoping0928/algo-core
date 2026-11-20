import unittest

import numpy as np

from node_features import (
    Composite,
    FeatureRule,
    FeatureRuleBank,
    Primitive,
    expand_observation,
)
from node_graph import CONCEPT, DERIVATION, NodeGraph
from node_runtime import NodeRuntime


class FeatureRuleTests(unittest.TestCase):
    def test_pairwise_difference_coordinates(self):
        x = np.asarray([[1.0, 0.0], [0.0, 2.0]])
        expanded = expand_observation(x)
        self.assertEqual(expanded.shape, (2, 3))
        np.testing.assert_allclose(expanded[:, 2], [1.0, -2.0])
        # A rule over the difference coordinate is induced and read back.
        rng = np.random.RandomState(3)
        data = rng.uniform(-1.0, 1.0, (600, 2))
        target = ((data[:, 0] - data[:, 1]) > 0.5).astype(float)
        train = np.arange(450)
        test = np.arange(450, 600)
        bank = FeatureRuleBank(min_z=1.0, min_lift=1.5,
                               pairwise_features=True)
        rules = bank.induce(
            data[train], target[train], data[test], target[test]
        )
        self.assertTrue(rules)
        dims = set()
        def collect(feature):
            if isinstance(feature, Primitive):
                dims.add(feature.dim)
            for child in getattr(feature, "children", ()):
                collect(child)
        for rule in rules:
            collect(rule.feature)
        self.assertIn(2, dims)

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

    def test_coverage_gates_drop_rare_extreme_rules(self):
        x = np.linspace(0.0, 1.0, 1000).reshape(-1, 1)
        # A high-precision rule that only fires on 2% of the states.
        rare = (x[:, 0] > 0.98).astype(float)
        train = np.arange(0, 1000, 2)
        test = np.arange(1, 1000, 2)
        off = FeatureRuleBank(min_z=1.0, min_lift=1.5)
        self.assertTrue(off.induce(x[train], rare[train], x[test], rare[test]))
        rule_gate = FeatureRuleBank(
            min_z=1.0, min_lift=1.5, min_coverage=0.1
        )
        self.assertFalse(rule_gate.induce(
            x[train], rare[train], x[test], rare[test]
        ))
        primitive_gate = FeatureRuleBank(
            min_z=1.0, min_lift=1.5, min_primitive_coverage=0.1
        )
        self.assertFalse(primitive_gate.induce(
            x[train], rare[train], x[test], rare[test]
        ))
        # A rule that covers a large share of states still passes.
        common = (x[:, 0] > 0.5).astype(float)
        self.assertTrue(rule_gate.induce(
            x[train], common[train], x[test], common[test]
        ))
        # An explicit cross-world coverage stream takes precedence: the
        # same rule passes on the training split but is rejected when an
        # independent stream never reaches its region.
        split_gate = FeatureRuleBank(
            min_z=1.0, min_lift=1.5, min_coverage=0.01
        )
        self.assertTrue(split_gate.induce(
            x[train], rare[train], x[test], rare[test]
        ))
        cross_gate = FeatureRuleBank(
            min_z=1.0, min_lift=1.5, min_coverage=0.01
        )
        self.assertFalse(cross_gate.induce(
            x[train], rare[train], x[test], rare[test],
            coverage_x=np.zeros((100, 1)),
        ))

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

    def test_shared_primitive_nodes_are_reused(self):
        runtime = NodeRuntime(
            obs_dim=2, lag=0, scales=np.ones(2),
            birth_threshold=0.01, max_nodes=10, exploration=0.0,
        )
        shared = Primitive(0, True, "<", 0.2)
        first = FeatureRule(shared, 5, 4, 0.8, 2, 2, 1.0, 0.75)
        second = FeatureRule(
            Composite("and", (shared, Primitive(1, False, ">", 0.3))),
            5, 4, 0.8, 2, 2, 1.0, 0.75,
        )
        runtime.rules.rules = [first, second]
        runtime.install_feature_rules()
        self.assertEqual(len(runtime.feature_ids), 2)
        composite = runtime.feature_ids[1]
        child_ids = runtime._feature_children[composite]
        # The primitive root is reused as the composite's child instead of
        # being stored twice.
        self.assertIn(runtime.feature_ids[0], child_ids)
        self.assertEqual(len(runtime._feature_all), len(set(runtime._feature_all)))

    def test_existing_memory_is_indexed_by_features(self):
        runtime = NodeRuntime(
            obs_dim=2, lag=0, scales=np.ones(2),
            birth_threshold=0.01, max_nodes=10, exploration=0.0,
        )
        state = np.asarray([0.0, 0.5])
        next_state = np.asarray([0.5, 0.0])
        runtime.learn_transition(state, 0, next_state, [0.0, 1.0])
        rule = FeatureRule(
            Primitive(0, True, "<", 0.2), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        runtime.rules.rules = [rule]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        runtime.learn_transition(state, 0, next_state, [0.0, 1.0])
        memories = runtime.memories_for([feature_id])
        self.assertTrue(memories)
        memory_id = next(iter(memories))
        self.assertIn(feature_id, runtime.graph.nodes[memory_id].members)

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
        self.assertGreaterEqual(len(runtime._feature_all), len(runtime.feature_ids))
        composite_roots = [
            node_id for node_id in runtime.feature_ids
            if runtime._feature_children.get(node_id)
        ]
        self.assertTrue(composite_roots)
        root = composite_roots[0]
        for child in runtime._feature_children[root]:
            self.assertIn(
                (root, child, DERIVATION), runtime.graph.edges
            )
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
