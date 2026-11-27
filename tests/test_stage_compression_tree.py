"""Unit tests for SelfDiscoveredRoleExtractor and HierarchicalStageTree drill-down."""
from __future__ import annotations

import unittest
import numpy as np


class StageCompressionTreeTests(unittest.TestCase):
    def test_role_extractor_discovers_ordered_roles_and_tree_drills_down(self):
        from compression_tree import HierarchicalStageTree, SelfDiscoveredRoleExtractor
        from sequence_stage import HierarchicalSequenceStageModel

        # Two distinct repeating regimes to create both homogeneous & heterogeneous parents
        seg_a = np.tile(np.array([1, 2, 3, 4, 5], dtype=np.int32), 20)
        seg_b = np.tile(np.array([6, 7, 8, 9, 10], dtype=np.int32), 20)
        stream = np.concatenate([seg_a, seg_b, seg_a, seg_b])

        model = HierarchicalSequenceStageModel(
            vocab_size=12, feature_dim=16, window=3, ema_alpha=0.10,
            n_regimes=4, min_dwell=8, hysteresis=2, ctx_len=4, chunk_len=100, seed=7,
        ).fit(stream)
        chain, _obs, h_slow = model.compress_to_stage_chain(stream)
        surp = model._compute_local_surprises(stream)
        spans = [
            (stream[st.start : st.end + 1], surp[st.start : st.end + 1], st.mean_state)
            for st in chain
        ]
        role_ext = SelfDiscoveredRoleExtractor(n_roles=3).fit_from_spans(
            spans, model.feature_map.embeddings_, model.feature_map.salience_,
        )
        self.assertEqual(len(role_ext.role_profiles_), 3)
        # Canonical ordering by relative position in stage
        self.assertLessEqual(
            role_ext.role_profiles_[0]["pos_frac"],
            role_ext.role_profiles_[2]["pos_frac"],
        )

        tree = HierarchicalStageTree.build_from_stage_chain(
            chain, stream, surp,
            model.feature_map.embeddings_, model.feature_map.salience_,
            role_ext, branch_factor=3,
        )
        self.assertGreaterEqual(tree.max_level, 1)
        self.assertLess(len(tree.roots), len(tree.leaves))

        # Coarse threshold -> hits parent directly with fewer visited nodes than total leaves
        hits_coarse, stats_coarse = tree.attend(
            h_slow[50], conf_threshold=0.05, rel_threshold=0.0, top_k=2,
        )
        self.assertGreater(len(hits_coarse), 0)
        self.assertGreater(stats_coarse["parent_hits"], 0)
        self.assertEqual(stats_coarse["drill_downs"], 0)

        # Strict threshold -> drills down to finer child nodes
        hits_detail, stats_detail = tree.attend(
            h_slow[50], conf_threshold=0.999, rel_threshold=0.0, top_k=2,
        )
        self.assertGreater(len(hits_detail), 0)
        self.assertGreater(stats_detail["drill_downs"], 0)
        self.assertEqual(hits_detail[0][0].level, 0)

    def test_learned_drill_down_policy_and_continuous_tree_isomorphism(self):
        from types import SimpleNamespace
        from compression_tree import HierarchicalStageTree, LearnedDrillDownPolicy

        # Synthetic continuous-control stages (task-agnostic, zero text symbols)
        stages = []
        for i in range(12):
            regime = 0 if i < 6 else 1
            base = np.array([1.0, 0.2, -0.1, 0.0]) if regime == 0 else np.array([-1.0, 0.8, 0.5, 0.1])
            jitter = np.array([0.05 * (i % 3), 0.0, 0.0, 0.0])
            stages.append(SimpleNamespace(
                regime=regime,
                start=i * 10,
                end=(i + 1) * 10 - 1,
                duration=10,
                entry_state=base + jitter,
                mean_state=base + jitter,
                delta_state=np.array([0.05, 0.0, 0.0, 0.0]),
            ))
        tree = HierarchicalStageTree.build_from_continuous_stages(stages, branch_factor=3)
        self.assertGreaterEqual(tree.max_level, 1)
        self.assertLess(len(tree.roots), len(tree.leaves))

        # Train LearnedDrillDownPolicy: low-conf node gets high drill reward, high-conf gets 0 gain
        pol = LearnedDrillDownPolicy(step_cost=0.05, lr=0.3)
        low_node = SimpleNamespace(level=1, confidence=0.2, children=(1, 2, 3))
        high_node = SimpleNamespace(level=1, confidence=0.95, children=(1, 2, 3))
        for _ in range(40):
            pol.update(low_node, 0.5, 2, 0.2, reward_parent=0.2, reward_drill=0.9, extra_nodes=3)
            pol.update(high_node, 0.9, 2, 0.2, reward_parent=0.9, reward_drill=0.9, extra_nodes=3)
        self.assertTrue(pol.should_drill(low_node, 0.5, 2, 0.2))
        self.assertFalse(pol.should_drill(high_node, 0.9, 2, 0.2))

        # Attend using the learned mental-action policy (no fixed conf_threshold!)
        hits, stats = tree.attend(np.array([1.0, 0.2, -0.1, 0.0]), policy=pol, top_k=2)
        self.assertGreater(len(hits), 0)
        self.assertLessEqual(stats["nodes_visited"], len(tree.leaves))


if __name__ == "__main__":
    unittest.main()
