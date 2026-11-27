"""Unit tests for BidirectionalVirtualNodeDAG (forward Post, backward Pre, DAG derivation, bridge)."""
from __future__ import annotations

import unittest


class VirtualNodeDAGTests(unittest.TestCase):
    def test_bidirectional_post_pre_intersection_difference_and_bridge(self):
        from virtual_node_dag import BidirectionalVirtualNodeDAG

        # Synthetic task-agnostic transition graph:
        #   op A (1) -> {10, 11, 12, 20, 21}   (specific {10,11,12} + shared {20,21})
        #   op B (2) -> {15, 16, 20, 21}       (specific {15,16} + shared {20,21})
        #   state 10 -> goal 30; state 15 -> goal 30
        #   background noise symbols 40..69 so lift > 1.5 is clear
        pairs = []
        for _ in range(15):
            for dst in (10, 11, 12, 20, 21):
                pairs.append((1, dst))
            for dst in (15, 16, 20, 21):
                pairs.append((2, dst))
            pairs.append((10, 30))
            pairs.append((15, 30))
            for s in range(40, 70):
                pairs.append((s, 40 + ((s + 1) % 30)))

        dag = BidirectionalVirtualNodeDAG(
            vocab_size=80,
            min_trigger_count=5,
            min_pair_count=3,
            min_lift=1.5,
            min_shared_members=2,
            containment_threshold=0.5,
        ).fit_transitions(pairs)

        post1 = dag.forward_node(1)
        post2 = dag.forward_node(2)
        pre30 = dag.backward_node(30)
        self.assertIsNotNone(post1)
        self.assertIsNotNone(post2)
        self.assertIsNotNone(pre30)
        self.assertTrue({10, 11, 12, 20, 21}.issubset(post1.members))
        self.assertTrue({10, 15}.issubset(pre30.members))

        # Intersection: shared consequences {20, 21}
        shared = dag.derive_intersection(post1.node_id, post2.node_id)
        self.assertIsNotNone(shared)
        self.assertEqual(shared.members, frozenset({20, 21}))
        self.assertEqual(shared.parents, {post1.node_id, post2.node_id})

        # Difference purification: Post(1) \ Shared = {10, 11, 12}
        pure1 = dag.derive_difference(post1.node_id, shared.node_id)
        self.assertIsNotNone(pure1)
        self.assertEqual(pure1.members, frozenset({10, 11, 12}))

        # Bidirectional bridge lookup: Post(1) ∩ Pre(30) = {10}
        bridges = dag.bridge_lookup(1, 30)
        self.assertEqual([m for m, _w in bridges], [10])

        # Reverse membership lookup
        containers = dag.containing_virtual_nodes(10, direction="forward")
        self.assertIn(post1.node_id, [n.node_id for n in containers])


if __name__ == "__main__":
    unittest.main()
