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

    def test_learned_class_context_gate_updates_without_hardcoded_rules(self):
        from context_decision import LearnedClassContextGate

        gate = LearnedClassContextGate(n_classes=10, lr=0.8)
        self.assertAlmostEqual(gate.class_gate(2), 0.5)
        self.assertAlmostEqual(gate.class_gate(7), 0.5)
        for _ in range(15):
            gate.update(2, -1.0)
            gate.update(7, +1.0)
        self.assertLess(gate.class_gate(2), 0.2)
        self.assertGreater(gate.class_gate(7), 0.8)

    def test_external_vs_self_observation_prevents_episodic_echo_and_pins_root(self):
        from temp_context import TemporaryContextMemory

        seq = []
        for _ in range(30):
            seq.extend([10, 11, 12, 13, 1, 2])
            seq.extend([20, 21, 22, 23, 1, 2])
        tokens = np.asarray(seq, dtype=np.int32)
        tcm = TemporaryContextMemory(vocab_size=30, assoc_window=5, min_cooc=2).fit_associations(tokens, stride=1)

        tcm.reset()
        for sym in (10, 11, 12):
            tcm.observe(sym, source="external")
        ep_ext = tcm.episodic_cache_probs([10, 11])
        self.assertIn(12, ep_ext)
        self.assertEqual(float(tcm._root_field.max()), 0.0)

        # Switching to source="self" must auto-pin external root field and NOT write self-echoes to episodic cache
        for sym in (20, 21, 20, 21):
            tcm.observe(sym, source="self")
        self.assertGreater(float(tcm._root_field.max()), 0.0)
        ep_self = tcm.episodic_cache_probs([20])
        self.assertNotIn(21, ep_self)

    def test_information_progress_lookahead_avoids_absorbing_sink(self):
        from class_policy import ClassLevelSequencePolicy
        from context_decision import ContextDrivenDecisionController
        from temp_context import TemporaryContextMemory
        from virtual_node_dag import BidirectionalVirtualNodeDAG

        # Symbol 1 is a high-frequency absorbing sink (1 -> 1 -> 1 -> 1 -> 1),
        # while 1 -> 10 -> 11 -> 12 is an informative continuation.
        seq = []
        for _ in range(60):
            seq.extend([1, 1, 1, 1, 1, 10, 11, 12, 13, 2])
        tokens = np.asarray(seq, dtype=np.int32)
        pol = ClassLevelSequencePolicy(
            vocab_size=20, n_classes=6, n_singletons=2, feature_dim=8, window=3, seed=5,
        ).fit(tokens)
        dag = BidirectionalVirtualNodeDAG(
            vocab_size=20, min_trigger_count=5, min_pair_count=3, min_lift=1.1,
        ).fit_sequence(tokens)
        tcm = TemporaryContextMemory(vocab_size=20, assoc_window=5, min_cooc=2).fit_associations(tokens, stride=1)
        tcm.reset()
        for sym in (10, 11, 1):
            tcm.observe(sym, source="external")

        ctrl = ContextDrivenDecisionController(
            pol, dag, tcm,
            active_inject_weight=0.25,
            bridge_weight=0.15,
            lookahead_depth=2,
            lookahead_beam=4,
            lookahead_pmi_alpha=0.45,
            lookahead_stagnation_weight=0.80,
        )
        slow = pol.update_slow_state(np.zeros(8, dtype=float), 10)
        slow = pol.update_slow_state(slow, 1)
        acts, probs, info = ctrl.decide_with_lookahead([10, 11, 1], slow, k=4, depth=2, beam_width=4)
        self.assertNotEqual(acts[0], 1, f"Lookahead fell into absorbing sink 1: acts={acts}, probs={probs}")
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_stage_tree_root_slots_persist_after_state_drift(self):
        from class_policy import ClassLevelSequencePolicy
        from compression_tree import HierarchicalStageTree, LearnedDrillDownPolicy, SelfDiscoveredRoleExtractor
        from context_decision import ContextDrivenDecisionController
        from sequence_stage import HierarchicalSequenceStageModel
        from temp_context import TemporaryContextMemory
        from virtual_node_dag import BidirectionalVirtualNodeDAG

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
        tcm = TemporaryContextMemory(vocab_size=30, assoc_window=6, min_cooc=2).fit_associations(tokens, stride=1)

        stage_model = HierarchicalSequenceStageModel(
            vocab_size=30, feature_dim=8, window=3, n_regimes=4,
            min_dwell=4, hysteresis=1, ctx_len=3, chunk_len=60, seed=7,
        ).fit(tokens)
        prompt = np.asarray([10, 11, 12, 13, 14, 1, 2, 10, 11, 12, 13, 14], dtype=np.int32)
        chain, _obs, _h = stage_model.compress_to_stage_chain(prompt)
        surp = stage_model._compute_local_surprises(prompt)
        role_ext = SelfDiscoveredRoleExtractor(n_roles=3).fit_from_spans(
            [(prompt[st.start : st.end + 1], surp[st.start : st.end + 1], st.mean_state) for st in chain],
            stage_model.feature_map.embeddings_, stage_model.feature_map.salience_,
        )
        tree = HierarchicalStageTree.build_from_stage_chain(
            chain, prompt, surp,
            stage_model.feature_map.embeddings_, stage_model.feature_map.salience_,
            role_ext, branch_factor=3,
        )
        ctrl = ContextDrivenDecisionController(
            pol, dag, tcm, stage_tree=tree, drill_policy=LearnedDrillDownPolicy(step_cost=0.01),
        )
        self.assertIsNotNone(ctrl._anchor_state)
        self.assertTrue(len(ctrl._tree_root_slots) > 0)
        self.assertGreater(float(tcm._root_field.max()), 0.0)
        # Even if slow_state drifts to opposite vector, root slots remain active
        strengths, _ = ctrl._collect_active_node_strengths(-ctrl._anchor_state)
        self.assertTrue(any(sym in strengths for sym in ctrl._tree_root_slots))


if __name__ == "__main__":
    unittest.main()
