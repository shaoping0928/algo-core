"""Unit tests for multi-core sharding, 16GB MemoryBudgetGovernor, HierarchicalOnlineArbitrator, and checkpoint save/load."""
from __future__ import annotations

import os
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class ScalabilityAndArbitrationTests(unittest.TestCase):
    def test_cached_sparse_proposal_matches_uncached_and_returns_independent_results(self):
        from sparse_index import SparseFeatureActionIndex

        index = SparseFeatureActionIndex(range(8)).fit_token_stream([1, 2, 3, 1, 2, 4], ctx_len=2)
        keys = [(-1, 2), (12, 1, 2)]
        expected, expected_probs = index.propose_with_probs(None, 4, sparse_keys=keys)
        with patch.object(index, "_propose_interpolated_ad", wraps=index._propose_interpolated_ad) as query:
            ranked, probs = index.propose_cached(keys, 4)
            self.assertEqual(ranked, expected)
            np.testing.assert_array_equal(probs, expected_probs)
            ranked.clear()
            probs.fill(-1)
            ranked, probs = index.propose_cached(keys, 4)
            self.assertEqual(ranked, expected)
            np.testing.assert_array_equal(probs, expected_probs)
            self.assertEqual(query.call_count, 1)
            index.propose_cached(keys, 2)
            index.propose_cached(keys, 4, discount=0.25)
            self.assertEqual(query.call_count, 3)

        reloaded = pickle.loads(pickle.dumps(index))
        ranked, probs = reloaded.propose_cached(keys, 4)
        self.assertEqual(ranked, expected)
        np.testing.assert_array_equal(probs, expected_probs)

    def test_cached_sparse_proposal_refreshes_after_index_mutations(self):
        from sparse_index import SparseFeatureActionIndex

        index = SparseFeatureActionIndex(range(8)).fit_token_stream([1, 2, 3, 1, 2, 4], ctx_len=2)
        keys = [(-1, 2), (12, 1, 2)]

        def check_refresh(previous_probs):
            ranked, probs = index.propose_cached(keys, 4)
            expected, expected_probs = index.propose_with_probs(None, 4, sparse_keys=keys)
            self.assertEqual(ranked, expected)
            np.testing.assert_array_equal(probs, expected_probs)
            self.assertFalse(np.array_equal(probs, previous_probs))
            return probs

        _, before = index.propose_cached(keys, 4)
        self.assertGreater(index.prune_under_pressure(0.20), 0)
        before = check_refresh(before)
        index.update(None, 5, dopamine=1.0, sparse_keys=keys)
        before = check_refresh(before)
        index.fit_token_stream([1, 2, 6] * 4, ctx_len=2)
        check_refresh(before)

    def test_packed_sparse_proposal_preserves_scores_ties_and_fallback(self):
        from sparse_index import SparseFeatureActionIndex

        index = SparseFeatureActionIndex(range(512))
        short, long, weighted = (-1, 2), (12, 1, 2), (12, 3, 2)
        for action in range(1, 401):
            index.update(None, action, dopamine=1.0, sparse_keys=[short])
        for action in range(3, 401, 3):
            for _ in range(action % 5 + 1):
                index.update(None, action, dopamine=1.0, sparse_keys=[long])
        index.update(None, 3, dopamine=0.5, sparse_keys=[weighted])

        scalar = index._propose_interpolated_ad
        with patch.object(index, "_propose_interpolated_ad", wraps=scalar) as fallback:
            for keys, k, discount in (([short], 12, 0.5), ([short, long], 25, 0.75),
                                     ([short, long], 7, 0.0), ([short, long], 12, 2.5)):
                with self.subTest(keys=keys, k=k, discount=discount):
                    expected, expected_probs = scalar(keys, k, discount)
                    actual, probs = index.propose_cached(keys, k, discount)
                    self.assertEqual(actual, expected)
                    np.testing.assert_array_equal(probs, expected_probs)
            fallback.assert_not_called()

            for keys, k in (([short, weighted], 10), ([short], 480), ([(-1, 8)], 24)):
                expected, expected_probs = scalar(keys, k)
                actual, probs = index.propose_cached(keys, k)
                self.assertEqual(actual, expected)
                np.testing.assert_array_equal(probs, expected_probs)
            self.assertEqual(fallback.call_count, 3)

        for _ in range(10):
            index.update(None, 401, dopamine=1.0, sparse_keys=[short])
        expected, expected_probs = scalar([short], 12)
        actual, probs = index.propose_cached([short], 12)
        self.assertEqual(actual, expected)
        self.assertIn(401, actual)
        np.testing.assert_array_equal(probs, expected_probs)

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

    def test_marginal_loo_arbitrator_and_self_mode_echo_damping(self):
        from hierarchical_arbitrator import HierarchicalOnlineArbitrator
        from temp_context import TemporaryContextMemory

        arb = HierarchicalOnlineArbitrator(n_regimes=3, lr=0.45, episode_window=20)
        # Expert 2 (chunk_grammar) is sparse but high-precision when active;
        # Expert 1 (working_memory) only echoes broad distribution (similar to base_policy).
        for _ in range(120):
            experts = [
                {1: 0.40, 2: 0.35, 3: 0.25},  # base_policy
                {1: 0.42, 2: 0.33, 3: 0.25},  # working_memory (redundant)
                {3: 0.90, 1: 0.10},           # chunk_grammar (high marginal precision on tgt=3)
                {},                           # frame_register
            ]
            arb.record_step(experts, 0.30, target=3, expert_confidences=(1.0, 1.0, 0.9, 0.3), k=4)
        arb.consolidate_episode()
        w_ext = arb.weights_for_regime(0)
        self.assertGreater(w_ext[2], w_ext[1])  # chunk_grammar outranks redundant working_memory

        # In open-loop self_mode, working_memory weight is damped relative to structural experts
        _, _, w_ext_blend = arb.blend_experts(
            [{1: 0.5, 2: 0.5}, {1: 0.9, 2: 0.1}, {2: 0.9, 1: 0.1}, {}],
            0.30, self_mode=False,
        )
        _, _, w_self_blend = arb.blend_experts(
            [{1: 0.5, 2: 0.5}, {1: 0.9, 2: 0.1}, {2: 0.9, 1: 0.1}, {}],
            0.30, self_mode=True,
        )
        self.assertLess(w_self_blend[1], w_ext_blend[1])
        self.assertGreater(w_self_blend[2], w_ext_blend[2])

        # Episodic prompt cache must decay during self-generated steps & non-content self-emissions
        # must adapt circuit V_diff downward without hard-coded symbol bans.
        tcm = TemporaryContextMemory(vocab_size=30, cache_decay=0.80)
        tcm.enable_circuit_field(use_gpu=False)
        for s in (2, 3, 4, 2, 3, 4):
            tcm.observe(s, source="external")
        ep_before = dict(tcm.episodic_cache_probs([2, 3]))
        self.assertIn(4, ep_before)
        for _ in range(15):
            tcm.observe(2, source="self")
            tcm.observe(3, source="self")
        ep_after = tcm.episodic_cache_probs([2, 3])
        self.assertLess(ep_after.get(4, 0.0), ep_before[4])
        # Repeated self-emission of symbol 2 drives its circuit differentiator V_diff < 0
        self.assertLess(float(tcm.circuit_differentiator()[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
