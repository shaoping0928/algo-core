"""Tests for sparse inverted-index retrieval, Top-K action pruning, and V=64 seq task."""
from __future__ import annotations

import unittest
import numpy as np


class SparseHeuristicSeqTests(unittest.TestCase):
    def test_sparse_action_index_inverted_postings_propose_top_k(self):
        from sparse_index import SparseFeatureActionIndex

        idx = SparseFeatureActionIndex(tuple(range(64)), quantize_step=0.1)
        s0 = np.array([0.2, 0.4, 0.8])
        s1 = np.array([0.9, 0.1, 0.3])
        for _ in range(5):
            idx.update(s0, action=42, dopamine=1.0, pain=0.0)
            idx.update(s0, action=7, dopamine=0.0, pain=1.0)
            idx.update(s1, action=19, dopamine=1.0, pain=0.0)
        cand = idx.propose_one(s0, k=4)
        self.assertEqual(len(cand), 4)
        self.assertEqual(cand[0], 42)
        self.assertNotIn(7, cand[:1])
        batch = idx.propose_batch(np.vstack([s0, s1]), k=4)
        self.assertEqual(batch.shape, (2, 4))
        self.assertEqual(int(batch[0, 0]), 42)
        self.assertEqual(int(batch[1, 0]), 19)

    def test_sparse_prototype_retriever_jaccard(self):
        from sparse_index import SparsePrototypeRetriever

        ret = SparsePrototypeRetriever()
        p0 = ret.add([(0, 1), (1, 2), (5, 3)])
        p1 = ret.add([(0, 1), (9, 4)])
        hits = ret.query_top_k([(0, 1), (1, 2), (5, 3)], k=2)
        self.assertEqual(hits[0][0], p0)
        self.assertAlmostEqual(hits[0][1], 1.0)
        self.assertEqual(hits[1][0], p1)

    def test_chained_search_prunes_64_actions_to_top_k(self):
        from evaluator import run_seed

        row = run_seed(
            dict(
                env="seq64",
                seed=100,
                steps=500,
                coverage=0.25,
                cap=30,
                max_nodes=120,
                epochs=1,
                lag=1,
                eval_seeds=[200],
                eval_episodes=1,
                decision="chained",
                chain_adaptive=True,
                chain_chunk_size=2,
                chain_hops=2,
                chain_max_hops=3,
                chain_beam_width=2,
                chain_wide_beam_width=3,
            )
        )
        self.assertEqual(len(row["evals"]), 1)
        self.assertEqual(row["frames"], 30)


if __name__ == "__main__":
    unittest.main()
