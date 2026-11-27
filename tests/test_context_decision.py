"""Unit tests for ContextDrivenDecisionController unifying all context reserves."""
from __future__ import annotations

import unittest
import numpy as np


class ContextDrivenDecisionTests(unittest.TestCase):
    def test_controller_unifies_class_policy_virtual_dag_temp_memory_and_stage_tree(self):
        from class_policy import ClassLevelSequencePolicy
        from compression_tree import HierarchicalStageTree, LearnedDrillDownPolicy, SelfDiscoveredRoleExtractor
        from context_decision import ContextDrivenDecisionController
        from sequence_stage import HierarchicalSequenceStageModel
        from temp_context import TemporaryContextMemory
        from virtual_node_dag import BidirectionalVirtualNodeDAG

        # Structured synthetic stream with two recurring contexts and a 2-step bridge:
        #   context A: 10 -> 11 -> 12 -> 13 -> 14
        #   context B: 20 -> 21 -> 22 -> 23 -> 24
        seq = []
        for _ in range(40):
            seq.extend([10, 11, 12, 13, 14, 1, 2])
            seq.extend([20, 21, 22, 23, 24, 1, 2])
        tokens = np.asarray(seq, dtype=np.int32)

        pol = ClassLevelSequencePolicy(
            vocab_size=30, n_classes=6, n_singletons=2, feature_dim=8, window=3, seed=7,
        ).fit(tokens)
        dag = BidirectionalVirtualNodeDAG(
            vocab_size=30, min_trigger_count=5, min_pair_count=3, min_lift=1.2,
        ).fit_sequence(tokens)
        tcm = TemporaryContextMemory(
            vocab_size=30, assoc_window=6, top_assoc_per_sym=8, min_cooc=2,
        ).fit_associations(tokens, stride=1)

        # Build compression tree + learned drill policy over the stream
        stage_model = HierarchicalSequenceStageModel(
            vocab_size=30, feature_dim=8, window=3, n_regimes=4,
            min_dwell=4, hysteresis=1, ctx_len=3, chunk_len=60, seed=7,
        ).fit(tokens)
        chain, _obs, h_slow = stage_model.compress_to_stage_chain(tokens[:120])
        surp = stage_model._compute_local_surprises(tokens[:120])
        role_ext = SelfDiscoveredRoleExtractor(n_roles=3).fit_from_spans(
            [(tokens[st.start : st.end + 1], surp[st.start : st.end + 1], st.mean_state) for st in chain],
            stage_model.feature_map.embeddings_, stage_model.feature_map.salience_,
        )
        tree = HierarchicalStageTree.build_from_stage_chain(
            chain, tokens[:120], surp,
            stage_model.feature_map.embeddings_, stage_model.feature_map.salience_,
            role_ext, branch_factor=3,
        )
        drill_pol = LearnedDrillDownPolicy(step_cost=0.01)

        ctrl = ContextDrivenDecisionController(
            pol, dag, tcm, stage_tree=tree, drill_policy=drill_pol,
            active_inject_weight=0.3, bridge_weight=0.2,
        )
        tcm.reset()
        for sym in (10, 11, 12):
            tcm.observe(sym)
        cand, probs, info = ctrl.decide_distribution([10, 11, 12], h_slow[10], k=4)
        self.assertEqual(cand[0], 13)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertGreater(info["tree_visited"], 0.0)


if __name__ == "__main__":
    unittest.main()
