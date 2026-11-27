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

    def test_automatic_bisimulation_refinement_splits_divergent_members(self):
        from virtual_node_dag import BidirectionalVirtualNodeDAG

        # Trigger 1 transitions to 4 members {10, 11, 20, 21}:
        #   (1, 10) and (1, 11) continue to {30, 31}   (behavior group A)
        #   (1, 20) and (1, 21) continue to {40, 41}   (behavior group B)
        seq = []
        for _ in range(20):
            seq.extend([1, 10, 30, 50, 51, 52])
            seq.extend([1, 11, 31, 53, 54, 55])
            seq.extend([1, 20, 40, 56, 57, 58])
            seq.extend([1, 21, 41, 59, 60, 61])
        dag = BidirectionalVirtualNodeDAG(
            vocab_size=70, min_trigger_count=5, min_pair_count=3, min_lift=1.2,
        ).fit_sequence(seq)

        post1 = dag.forward_node(1)
        self.assertIsNotNone(post1)
        children = dag.refine_virtual_node(
            post1.node_id, max_splits=3, min_member_support=5, min_split_dissimilarity=0.5,
        )
        self.assertGreaterEqual(len(children), 2)
        # Group A {10, 11} and Group B {20, 21} must be separated into distinct child virtual nodes
        child_sets = [c.members for c in children]
        self.assertTrue(any(10 in s and 20 not in s for s in child_sets))
        self.assertTrue(any(20 in s and 10 not in s for s in child_sets))


if __name__ == "__main__":
    unittest.main()
