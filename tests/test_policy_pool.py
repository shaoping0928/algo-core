"""Unit tests for PolicyPopulationManager (birth, shadow validation, promotion, eviction) and lookahead."""
from __future__ import annotations

import unittest
import numpy as np


class PolicyPoolTests(unittest.TestCase):
    def test_population_manager_births_promotes_and_evicts(self):
        from policy_pool import PolicyPopulationManager

        mgr = PolicyPopulationManager(
            max_pool_size=5,
            birth_every=10,
            evict_every=10,
            min_trials_promote=15,
            min_trials_evict=10,
            depth_cost=0.005,
            max_lookahead_depth=3,
            seed=7,
        )
        # Synthetic oracle: deeper lookahead (depth >= 2) + gated content injection yields highest reward
        rng = np.random.RandomState(7)

        def eval_spec(spec):
            base = 0.85 if (spec.lookahead_depth >= 2 and spec.content_class_gate) else 0.25
            r = float(np.clip(base + rng.normal(0.0, 0.05), 0.0, 1.0))
            return r, (r > 0.5), (r > 0.3)

        for _ in range(80):
            mgr.record_step_validation(eval_spec)

        snap = mgr.snapshot()
        self.assertGreater(snap["total_births"], 4)
        self.assertGreater(snap["total_promotions"], 0)
        self.assertGreater(snap["total_evictions"], 0)
        self.assertLessEqual(snap["pool_size"], 6)
        self.assertGreaterEqual(snap["active_depth"], 2)
        self.assertTrue(snap["active_gate"])

    def test_context_decision_lookahead_depth2_runs_cleanly(self):
        from class_policy import ClassLevelSequencePolicy
        from context_decision import ContextDrivenDecisionController
        from temp_context import TemporaryContextMemory
        from virtual_node_dag import BidirectionalVirtualNodeDAG

        seq = []
        for _ in range(30):
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

        ctrl = ContextDrivenDecisionController(
            pol, dag, tcm, content_class_gate=True, lookahead_depth=2, lookahead_beam=3,
        )
        tcm.reset()
        for s in (10, 11, 12):
            tcm.observe(s)
        cand, probs, info = ctrl.decide_with_lookahead([10, 11, 12], k=4)
        self.assertEqual(cand[0], 13)
        self.assertEqual(info["lookahead_depth"], 2.0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_policy_pool_evolves_structural_memory_weights_via_controller(self):
        from class_policy import ClassLevelSequencePolicy
        from context_decision import ContextDrivenDecisionController
        from policy_pool import PolicyPopulationManager, apply_spec_to_controller
        from structural_memory import HierarchicalChunkMemory, StructuralFrameMemory
        from temp_context import TemporaryContextMemory
        from virtual_node_dag import BidirectionalVirtualNodeDAG

        # Delayed paired-cue stream beyond class_ctx_len=4:
        #   10 -> [4 random noise steps] -> 20 vs 11 -> [4 random noise steps] -> 21
        rng = np.random.RandomState(5)
        seq = []
        for _ in range(240):
            op, cl = (10, 20) if rng.rand() < 0.5 else (11, 21)
            seq.append(op)
            seq.extend(rng.randint(1, 9, size=4).tolist())
            seq.append(cl)
            seq.extend(rng.randint(1, 9, size=6).tolist())
        tokens = np.asarray(seq, dtype=np.int32)

        pol = ClassLevelSequencePolicy(
            vocab_size=26, n_classes=6, n_singletons=2, feature_dim=8, window=3, seed=5,
        ).fit(tokens)
        dag = BidirectionalVirtualNodeDAG(
            vocab_size=26, min_trigger_count=5, min_pair_count=3, min_lift=1.1,
        ).fit_sequence(tokens)
        tcm = TemporaryContextMemory(vocab_size=26, assoc_window=8, min_cooc=2).fit_associations(tokens, stride=1)
        cm = HierarchicalChunkMemory(
            vocab_size=26, max_levels=2, max_chunks_per_level=32, min_pair_count=10, min_pmi=1.2,
        ).fit(tokens)
        fm = StructuralFrameMemory(
            vocab_size=26, min_gap=2, max_gap=8, min_pair_count=15, min_pmi=1.2,
        ).fit(tokens)

        ctrl = ContextDrivenDecisionController(
            pol, dag, tcm, chunk_memory=cm, frame_memory=fm,
        )
        mgr = PolicyPopulationManager(
            max_pool_size=6, birth_every=15, evict_every=15,
            min_trials_promote=12, min_trials_evict=12, max_lookahead_depth=1, seed=5,
        )
        # Shadow-validate on delayed closure points across 4 random filler steps
        eval_cases = []
        for i in range(40):
            op, cl = (10, 20) if i % 2 == 0 else (11, 21)
            filler = rng.randint(1, 9, size=4).tolist()
            eval_cases.append(([op] + filler, cl))
        for hist, tgt in eval_cases:
            tcm.reset()
            for s in hist:
                tcm.observe(s, source="external")

            def _eval(spec):
                apply_spec_to_controller(spec, ctrl)
                acts, _, _ = ctrl.decide_distribution(hist, None, k=4)
                h1 = bool(acts and acts[0] == tgt)
                h4 = bool(tgt in acts)
                return (1.0 if h1 else (0.2 if h4 else 0.0)), h1, h4

            mgr.record_step_validation(_eval)

        snap = mgr.snapshot()
        self.assertGreater(snap["total_promotions"], 0)
        # Winning evolved policy must enable higher-order structural memory!
        self.assertGreater(snap["active_frame"] + snap["active_chunk"], 0.15)


if __name__ == "__main__":
    unittest.main()
