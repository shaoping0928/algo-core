"""Unit tests for higher-order structural memory (HierarchicalChunkMemory & StructuralFrameMemory)."""
from __future__ import annotations

import unittest
import numpy as np


class HigherOrderStructuralMemoryTests(unittest.TestCase):
    def test_hierarchical_chunk_memory_resolves_shared_suffix_macro_options(self):
        from structural_memory import HierarchicalChunkMemory

        # Two macro-options share the exact same 2-symbol suffix (11, 12):
        #   Macro A: [10, 11, 12] -> Macro X: [20, 21, 22]
        #   Macro B: [15, 11, 12] -> Macro Y: [30, 31, 32]
        # An order-1/order-2 atomic model cannot distinguish what follows [11, 12],
        # whereas recursive chunking discovers [10,11,12] vs [15,11,12] and resolves it!
        rng = np.random.RandomState(0)
        seq = []
        for _ in range(120):
            if rng.rand() < 0.5:
                seq.extend([10, 11, 12, 20, 21, 22, 1])
            else:
                seq.extend([15, 11, 12, 30, 31, 32, 1])
            # Add occasional noise symbol so PMI of true macros is strongly positive
            seq.extend([int(rng.randint(2, 8)), int(rng.randint(2, 8))])
        tokens = np.asarray(seq, dtype=np.int32)

        cm = HierarchicalChunkMemory(
            vocab_size=40, max_levels=2, max_chunks_per_level=64,
            min_pair_count=10, min_pmi=1.2, max_chunk_len=4, chunk_ctx_len=2,
        ).fit(tokens)

        self.assertGreater(cm.n_total_units, 40)
        acts_a, probs_a, _ = cm.predict_next_atoms([1, 10, 11, 12], k=4)
        acts_b, probs_b, _ = cm.predict_next_atoms([1, 15, 11, 12], k=4)
        self.assertEqual(acts_a[0], 20, f"Expected 20 after [10,11,12], got {acts_a}")
        self.assertEqual(acts_b[0], 30, f"Expected 30 after [15,11,12], got {acts_b}")
        # Intra-chunk completion: inside [20, 21] -> 22
        acts_in, _, _ = cm.predict_next_atoms([10, 11, 12, 20, 21], k=4)
        self.assertEqual(acts_in[0], 22)

    def test_structural_frame_memory_discovers_paired_delayed_dependencies_and_pops(self):
        from structural_memory import StructuralFrameMemory

        # Non-language Key -> [random 3..6 filler] -> Lock paired dependency task
        rng = np.random.RandomState(42)
        pairs = [(10, 20), (11, 21), (12, 22)]
        seq = []
        for _ in range(300):
            op, cl = pairs[int(rng.randint(len(pairs)))]
            seq.append(op)
            gap = int(rng.randint(3, 7))
            seq.extend(rng.randint(1, 9, size=gap).tolist())
            seq.append(cl)
            seq.extend(rng.randint(1, 9, size=4).tolist())
        tokens = np.asarray(seq, dtype=np.int32)

        fm = StructuralFrameMemory(
            vocab_size=30, min_gap=2, max_gap=10, min_pair_count=15,
            min_pmi=1.2, min_asymmetry=0.65,
        ).fit(tokens)

        for op, cl in pairs:
            self.assertIn(op, fm.frames_by_opener_)
            self.assertEqual(fm.frames_by_opener_[op][0].closer, cl)

        # Online pushdown register: open 11, advance 4 random steps -> predicts 21, then pops on 21!
        fm.reset()
        for sym in (11, 3, 5, 2, 7):
            fm.observe(sym)
        pred_open = fm.predict_closers_for_history()
        self.assertIn(21, pred_open)
        self.assertNotIn(20, pred_open)

        # Closing with 21 must pop the frame cleanly (zero lingering expectation!)
        fm.observe(21)
        pred_closed = fm.predict_closers_for_history()
        self.assertEqual(pred_closed, {})

    def test_controller_solves_delayed_paired_frame_with_structural_memory(self):
        from class_policy import ClassLevelSequencePolicy
        from context_decision import ContextDrivenDecisionController
        from structural_memory import HierarchicalChunkMemory, StructuralFrameMemory
        from temp_context import TemporaryContextMemory
        from virtual_node_dag import BidirectionalVirtualNodeDAG

        rng = np.random.RandomState(7)
        pairs = [(10, 20), (11, 21)]
        seq = []
        for _ in range(250):
            op, cl = pairs[int(rng.randint(2))]
            seq.append(op)
            seq.extend(rng.randint(1, 9, size=4).tolist())
            seq.append(cl)
            seq.extend(rng.randint(1, 9, size=6).tolist())
        tokens = np.asarray(seq, dtype=np.int32)

        pol = ClassLevelSequencePolicy(
            vocab_size=26, n_classes=6, n_singletons=2, feature_dim=8, window=3, seed=7,
        ).fit(tokens)
        dag = BidirectionalVirtualNodeDAG(
            vocab_size=26, min_trigger_count=5, min_pair_count=3, min_lift=1.1,
        ).fit_sequence(tokens)
        tcm = TemporaryContextMemory(vocab_size=26, assoc_window=8, min_cooc=3).fit_associations(tokens, stride=1)
        cm = HierarchicalChunkMemory(
            vocab_size=26, max_levels=2, min_pair_count=8, min_pmi=1.2,
        ).fit(tokens)
        fm = StructuralFrameMemory(
            vocab_size=26, min_gap=2, max_gap=8, min_pair_count=15, min_pmi=1.2,
        ).fit(tokens)

        ctrl = ContextDrivenDecisionController(
            pol, dag, tcm,
            chunk_memory=cm, chunk_weight=0.35,
            frame_memory=fm, frame_weight=0.60,
        )
        tcm.reset()
        hist = [10, 2, 4, 6, 3]
        for s in hist:
            tcm.observe(s, source="external")
        acts, probs, info = ctrl.decide_distribution(hist, None, k=4)
        self.assertEqual(acts[0], 20, f"Expected delayed closer 20, got {acts} ({info})")
        self.assertGreater(info["frame_closers"], 0.0)

    def test_chunk_class_abstraction_generalizes_to_unseen_skip_slot_composition(self):
        from structural_memory import HierarchicalChunkMemory

        # Three interchangeable 2-atom middle macro-options: (10,11), (12,13), (14,15)
        # Training ONLY sees:
        #   Anchor 5 + { (10,11), (12,13) } -> Goal (20,21)
        #   Anchor 6 + { (10,11), (12,13), (14,15) } -> Goal (22,23)
        #   Shared neutral frame 7 + { (10,11), (12,13), (14,15) } -> 8
        # NEVER SEEN IN TRAINING: Anchor 5 + (14,15) -> must generalize via skip-slot schema to 20!
        rng = np.random.RandomState(13)
        seq = []
        for _ in range(180):
            r = rng.rand()
            if r < 0.30:
                mid = [10, 11] if rng.rand() < 0.5 else [12, 13]
                seq.extend([1, 5] + mid + [20, 21, 1])
            elif r < 0.65:
                mid = [[10, 11], [12, 13], [14, 15]][int(rng.randint(3))]
                seq.extend([1, 6] + mid + [22, 23, 1])
            else:
                mid = [[10, 11], [12, 13], [14, 15]][int(rng.randint(3))]
                seq.extend([1, 7] + mid + [8, 1])
            seq.extend([int(rng.randint(2, 5))])
        tokens = np.asarray(seq, dtype=np.int32)

        cm = HierarchicalChunkMemory(
            vocab_size=30, max_levels=1, max_chunks_per_level=32,
            min_pair_count=10, min_pmi=1.2, max_chunk_len=2, chunk_ctx_len=2,
            enable_chunk_classes=True, n_chunk_classes=6, n_singleton_chunks=6,
            min_schema_count=5, seed=7,
        ).fit(tokens)

        self.assertIsNotNone(cm.chunk_to_class_)
        self.assertGreater(len(cm.skip_schemas_), 0)
        # All three middle macros should be discovered chunks
        for pair in ((10, 11), (12, 13), (14, 15)):
            self.assertIn(pair, cm.atoms_to_chunk_)
        # Zero-shot unseen composition [1, 5, 14, 15] -> skip-slot schema (5, CC_MID) boosts 20!
        acts_zs, probs_zs, info_zs = cm.predict_next_atoms([1, 5, 14, 15], k=6)
        self.assertIn(20, acts_zs[:3], f"Expected zero-shot schema target 20 in top-3, got {acts_zs} ({info_zs})")
        self.assertGreater(info_zs["class_mass"], 0.0)


if __name__ == "__main__":
    unittest.main()
