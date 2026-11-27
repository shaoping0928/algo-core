"""Unit tests for OnlineSymbolFeatureMap and HierarchicalSequenceStageModel."""
from __future__ import annotations

import unittest
import numpy as np


class SequenceStageTests(unittest.TestCase):
    def test_online_symbol_feature_map_groups_cooccurring_symbols(self):
        from sequence_stage import OnlineSymbolFeatureMap

        # Two disjoint synthetic topic streams with distinct symbol sets
        topic_a = np.tile(np.array([1, 2, 3, 4, 5], dtype=np.int32), 40)
        topic_b = np.tile(np.array([6, 7, 8, 9, 10], dtype=np.int32), 40)
        stream = np.concatenate([topic_a, topic_b, topic_a, topic_b])

        fm = OnlineSymbolFeatureMap(vocab_size=12, dim=16, window=3, seed=7).fit(stream)
        h = fm.encode_stream(stream, ema_alpha=0.15)
        self.assertEqual(h.shape, (len(stream), 16))
        # Context vectors inside topic_a vs topic_b must be separable
        mean_a = h[40:180].mean(axis=0)
        mean_b = h[240:380].mean(axis=0)
        self.assertGreater(float(np.linalg.norm(mean_a - mean_b)), 0.1)

    def test_hierarchical_sequence_stage_model_compresses_and_proposes(self):
        from sequence_stage import HierarchicalSequenceStageModel

        topic_a = np.tile(np.array([1, 2, 3, 4, 5], dtype=np.int32), 40)
        topic_b = np.tile(np.array([6, 7, 8, 9, 10], dtype=np.int32), 40)
        stream = np.concatenate([topic_a, topic_b, topic_a, topic_b])

        model = HierarchicalSequenceStageModel(
            vocab_size=12, feature_dim=16, window=3, ema_alpha=0.15,
            n_regimes=4, min_dwell=5, hysteresis=2, ctx_len=4, chunk_len=100, seed=7,
        ).fit(stream)
        self.assertEqual(model.stage_symbol_probs_.shape, (4, 12))
        self.assertEqual(model.stage_transitions_.shape, (4, 4))

        h = np.zeros(16, dtype=np.float32)
        for tok in [1, 2, 3, 4]:
            h = model.update_slow_state(h, tok)
        cand, probs, sid = model.propose_with_stage([1, 2, 3, 4], h, k=4, stage_beta=0.35)
        self.assertEqual(cand[0], 5)
        self.assertGreater(float(probs[0]), 0.0)
        self.assertGreaterEqual(sid, 0)


if __name__ == "__main__":
    unittest.main()
