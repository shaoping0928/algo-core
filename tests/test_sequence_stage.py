"""Unit tests for dynamic progression features and hierarchical stage-chain compression."""
from __future__ import annotations

import unittest
import numpy as np


class SequenceStageTests(unittest.TestCase):
    def test_online_symbol_feature_map_progression_stream(self):
        from sequence_stage import OnlineSymbolFeatureMap

        topic_a = np.tile(np.array([1, 2, 3, 4, 5], dtype=np.int32), 40)
        topic_b = np.tile(np.array([6, 7, 8, 9, 10], dtype=np.int32), 40)
        stream = np.concatenate([topic_a, topic_b, topic_a, topic_b])

        fm = OnlineSymbolFeatureMap(vocab_size=12, dim=16, window=3, seed=7).fit(stream)
        obs, h_slow, nov = fm.encode_progression_stream(stream)
        self.assertEqual(obs.shape, (len(stream), 12))
        self.assertEqual(h_slow.shape, (len(stream), 16))
        self.assertEqual(nov.shape, (len(stream),))
        # Topic switch at index 200 should cause a higher novelty/shift than steady repetition
        self.assertGreater(float(obs[205, 1]), float(obs[100, 1]))

    def test_hierarchical_sequence_stage_model_compresses_and_predicts_stages(self):
        from sequence_stage import HierarchicalSequenceStageModel

        topic_a = np.tile(np.array([1, 2, 3, 4, 5], dtype=np.int32), 40)
        topic_b = np.tile(np.array([6, 7, 8, 9, 10], dtype=np.int32), 40)
        stream = np.concatenate([topic_a, topic_b, topic_a, topic_b])

        model = HierarchicalSequenceStageModel(
            vocab_size=12, feature_dim=16, window=3, ema_alpha=0.10,
            n_regimes=4, min_dwell=8, hysteresis=2, ctx_len=4, chunk_len=100, seed=7,
        ).fit(stream)
        self.assertEqual(model.stage_symbol_probs_.shape, (4, 12))
        self.assertEqual(model.stage_trans_probs_.shape, (4, 4))

        chain, obs, _h = model.compress_to_stage_chain(stream[:150])
        self.assertGreater(len(chain), 0)
        self.assertLess(len(chain), 150)
        self.assertEqual(chain[0].entry_state.shape, (16,))
        self.assertEqual(chain[0].delta_state.shape, (16,))

        nxt_stage, prob = model.predict_next_stage(chain[0].regime)
        self.assertGreaterEqual(nxt_stage, 0)
        self.assertGreater(prob, 0.0)

        cand, probs, sid = model.propose_with_stage([1, 2, 3, 4], obs[20], k=4, stage_beta=0.25)
        self.assertEqual(cand[0], 5)
        self.assertGreater(float(probs[0]), 0.0)
        self.assertGreaterEqual(sid, 0)


if __name__ == "__main__":
    unittest.main()
