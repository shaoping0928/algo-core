"""Unit tests for multi-core sharding, 16GB MemoryBudgetGovernor, HierarchicalOnlineArbitrator, and checkpoint save/load."""
from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np


class ScalabilityAndArbitrationTests(unittest.TestCase):
    def test_parallel_sharded_sparse_index_is_exact_match_with_serial(self):
        from sparse_index import SparseFeatureActionIndex

        rng = np.random.RandomState(11)
        tokens = rng.randint(1, 40, size=150_000, dtype=np.int32)
        idx1 = SparseFeatureActionIndex(tuple(range(40)), prior_weight=0.0).fit_token_stream(
            tokens, ctx_len=3, n_workers=1,
        )
        idx4 = SparseFeatureActionIndex(tuple(range(40)), prior_weight=0.0).fit_token_stream(
            tokens, ctx_len=3, n_workers=4,
        )
        self.assertEqual(idx1.total_updates, idx4.total_updates)
        self.assertEqual(set(idx1._key_totals.keys()), set(idx4._key_totals.keys()))
        for k in list(idx1._key_totals.keys())[:200]:
            self.assertAlmostEqual(idx1._key_totals[k], idx4._key_totals[k], places=6)

    def test_memory_budget_governor_convex_pressure_and_stronger_eviction(self):
        from memory_governor import MemoryBudgetGovernor, current_rss_bytes
        from policy_pool import PolicyPopulationManager
        from sparse_index import SparseFeatureActionIndex

        # Live OS process RSS must be readable and positive (> 5 MB)
        self.assertGreater(current_rss_bytes(), 5 * 1024 * 1024)

        gb = 1024 ** 3
        gov = MemoryBudgetGovernor(max_bytes=16 * gb, soft_ratio=0.50, hard_ratio=0.92)
        self.assertEqual(gov.pressure(rss_bytes=6 * gb), 0.0)
        p_mid = gov.pressure(rss_bytes=11 * gb)
        p_high = gov.pressure(rss_bytes=14 * gb)
        p_max = gov.pressure(rss_bytes=15 * gb)
        self.assertGreater(p_mid, 0.0)
        self.assertGreater(p_high, p_mid)
        self.assertGreaterEqual(p_max, p_high)
        self.assertEqual(p_max, 1.0)

        # Verify higher pressure evicts strictly more postings & pool members
        rng = np.random.RandomState(3)
        toks = rng.randint(1, 50, size=6000, dtype=np.int32)
        idx_lo = SparseFeatureActionIndex(tuple(range(50)), prior_weight=0.0).fit_token_stream(toks, ctx_len=3)
        idx_hi = SparseFeatureActionIndex(tuple(range(50)), prior_weight=0.0).fit_token_stream(toks, ctx_len=3)
        ev_lo = idx_lo.prune_under_pressure(0.20)
        ev_hi = idx_hi.prune_under_pressure(0.95)
        self.assertGreater(ev_hi, ev_lo)

        mgr = PolicyPopulationManager(max_pool_size=8, seed=1)
        for _ in range(4):
            mgr.spawn_candidate()
        self.assertEqual(len(mgr.pool), 8)
        ev_pool = mgr.prune_under_pressure(0.95)
        self.assertGreater(ev_pool, 0)
        self.assertLessEqual(len(mgr.pool), 3)

    def test_hierarchical_online_arbitrator_resists_single_episode_hijack_and_learns_consensus(self):
        from hierarchical_arbitrator import HierarchicalOnlineArbitrator

        arb = HierarchicalOnlineArbitrator(n_regimes=3, lr=0.6, shrinkage=0.04, episode_window=30)
        w_init = arb.weights_for_regime(2).copy()

        # Single outlier episode favoring expert 3 only shifts weight by a bounded step
        for _ in range(30):
            arb.record_step([{1: 0.9}, {2: 0.9}, {3: 0.9}, {9: 0.95}], 0.85, target=9)
        w_after_one = arb.weights_for_regime(2).copy()
        self.assertLess(w_after_one[3], 0.35)

        # Six consistent cross-episodes where expert 2 (chunk_grammar) wins cleanly shifts consensus to expert 2
        for _ep in range(6):
            for _ in range(30):
                arb.record_step([{1: 0.9}, {2: 0.9}, {7: 0.95}, {9: 0.9}], 0.85, target=7)
        w_final = arb.weights_for_regime(2)
        self.assertGreater(w_final[2], w_init[2] + 0.10)
        self.assertEqual(int(np.argmax(w_final)), 2)

    def test_substrate_bundle_parallel_fit_and_checkpoint_roundtrip(self):
        from substrate_bundle import StructuralSubstrateBundle

        rng = np.random.RandomState(42)
        seq = []
        for _ in range(120):
            seq.extend([1, 10, 11, 12, 20, 21, 1, int(rng.randint(2, 8))])
            seq.extend([1, 15, 11, 12, 22, 23, 1, int(rng.randint(2, 8))])
        tokens = np.asarray(seq, dtype=np.int32)
        itos = [f"s{i}" for i in range(30)]
        vocab = {s: i for i, s in enumerate(itos)}

        bundle = StructuralSubstrateBundle.fit_parallel(
            tokens, vocab_size=30, vocab=vocab, itos=itos,
            n_workers=4, n_classes=6, n_singletons=2, feature_dim=8,
            max_chunks_per_level=32, n_chunk_classes=6, build_stage_model=True, seed=7,
        )
        ctrl1 = bundle.make_controller(use_arbitrator=True)
        bundle.temp_memory.reset()
        for s in (1, 10, 11, 12):
            bundle.temp_memory.observe(s, source="external")
        acts1, probs1, _ = ctrl1.decide_distribution([1, 10, 11, 12], None, k=4)

        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "test_bundle.pkl.gz")
            save_info = bundle.save(ckpt)
            self.assertTrue(os.path.exists(ckpt))
            self.assertGreater(save_info["size_mb"], 0.0)

            reloaded = StructuralSubstrateBundle.load(ckpt)
            ctrl2 = reloaded.make_controller(use_arbitrator=True)
            reloaded.temp_memory.reset()
            for s in (1, 10, 11, 12):
                reloaded.temp_memory.observe(s, source="external")
            acts2, probs2, _ = ctrl2.decide_distribution([1, 10, 11, 12], None, k=4)
            self.assertEqual(acts1, acts2)
            self.assertTrue(np.allclose(probs1, probs2, atol=1e-6))

    def test_bitpacked_index_matches_shard_count_and_streaming_chunk_ingest(self):
        from sparse_index import SparseFeatureActionIndex
        from substrate_bundle import StructuralSubstrateBundle

        rng = np.random.RandomState(99)
        tokens = rng.randint(1, 50, size=25_000, dtype=np.int32)
        # Bit-packed path (default for small vocab + ctx_len=4)
        idx_bp = SparseFeatureActionIndex(tuple(range(50)), prior_weight=0.0).fit_token_stream(
            tokens, ctx_len=4, n_workers=4, min_high_order_count=1,
        )
        # Direct Python shard reference
        sh_p, sh_kt, sh_gc = SparseFeatureActionIndex._count_shard(tokens.tolist(), 4, len(tokens), 4)
        self.assertEqual(idx_bp.total_updates, sum(sh_gc.values()))
        self.assertEqual(set(idx_bp._key_totals.keys()), set(sh_kt.keys()))
        for fk in list(sh_kt.keys())[:200]:
            self.assertAlmostEqual(idx_bp._key_totals[fk], sh_kt[fk], places=6)

        # Streaming chunk ingestion on StructuralSubstrateBundle
        seq1 = np.tile(np.asarray([1, 2, 10, 11, 12, 20, 21], dtype=np.int32), 80)
        seq2 = np.tile(np.asarray([1, 2, 10, 11, 12, 20, 21], dtype=np.int32), 80)
        bun = StructuralSubstrateBundle.fit_parallel(
            seq1, vocab_size=30, n_workers=2, n_classes=6, n_singletons=2,
            feature_dim=8, max_chunks_per_level=24, n_chunk_classes=6, build_stage_model=False, seed=3,
        )
        tot_before = bun.class_policy.class_index.total_updates
        rep = bun.ingest_stream_chunk(seq2, n_workers=2, min_high_order_count=1)
        self.assertEqual(rep["total_tokens"], len(seq1) + len(seq2))
        self.assertGreater(bun.class_policy.class_index.total_updates, tot_before)


if __name__ == "__main__":
    unittest.main()
