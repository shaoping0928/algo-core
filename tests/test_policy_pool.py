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


if __name__ == "__main__":
    unittest.main()
