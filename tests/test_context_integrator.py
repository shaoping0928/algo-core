import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for dp, _, _ in os.walk(os.path.join(ROOT, "core")):
    if "__pycache__" not in dp:
        sys.path.insert(0, dp)

from context_integrator import ContextualIntegrator
from substrate_bundle import StructuralSubstrateBundle


class ContextualIntegratorTests(unittest.TestCase):
    def test_multi_cue_intersection_disambiguation_and_no_amnesia(self):
        # Synthetic two-domain stream where cue 10 is ambiguous between domain 1 (11 -> 20,21)
        # and domain 2 (12 -> 30,31), separated by long stretches of neutral filler (1..5).
        rng = np.random.RandomState(42)
        seq = []
        for _ in range(120):
            # Domain 1 episode: cues 10, 11 followed 15-35 steps later by targets 20, 21
            seq.extend([10, 1, 2, 11] + rng.randint(1, 6, size=18).tolist() + [20, 3, 21, 4, 22])
            # Domain 2 episode: cues 10, 12 followed 15-35 steps later by targets 30, 31
            seq.extend([10, 2, 3, 12] + rng.randint(1, 6, size=18).tolist() + [30, 4, 31, 1, 32])
        arr = np.asarray(seq, dtype=np.int32)

        ci = ContextualIntegrator(
            n_units=40, vocab_size=40, forward_window=36, min_cooc=5, min_pmi=0.4,
        ).fit_unit_stream(arr, stride=1, max_centers=50_000)

        # 1. Domain 1 context followed by 40 neutral filler steps -> early cues MUST survive!
        ci.reset()
        ci.observe_units([10, 11] + [1, 2, 3, 4] * 10, source="external")
        llr1 = ci.contextual_atom_llr()
        self.assertGreater(llr1.get(20, 0.0), llr1.get(30, 0.0))
        self.assertGreater(llr1.get(20, 0.0), 0.5)

        # 2. Domain 2 context disambiguates cue 10 toward target 30 instead of 20!
        ci.reset()
        ci.observe_units([10, 12] + [1, 2, 3, 4] * 10, source="external")
        llr2 = ci.contextual_atom_llr()
        self.assertGreater(llr2.get(30, 0.0), llr2.get(20, 0.0))

        # 3. Realization adaptation: once target 30 is self-emitted repeatedly, its drive drops
        #    while unexpressed domain-2 target 31 stays elevated!
        s30_before = llr2[30]
        s31_before = llr2.get(31, 0.0)
        for _ in range(3):
            ci.observe_units([30], source="self")
        llr2_after = ci.contextual_atom_llr()
        self.assertLess(llr2_after.get(30, 0.0), s30_before)
        self.assertGreater(llr2_after.get(31, 0.0), llr2_after.get(30, 0.0) * 0.8)
        self.assertGreater(s31_before, 0.0)

    def test_bundle_integrates_context_across_long_boundary(self):
        # Construct a 2-topic stream with distinct episodes separated by neutral gaps,
        # where both topics share the exact same local boundary [2, 3, 4, 5] before
        # diverging to topic-A continuation (20, 21) vs topic-B continuation (30, 31).
        rng = np.random.RandomState(7)
        seq = []
        for ep in range(160):
            # Inter-episode gap of neutral tokens (1..4) so Topic A & Topic B do not bleed
            seq.extend(rng.randint(1, 5, size=55).tolist())
            if ep % 2 == 0:
                seq.extend([10, 11, 1, 12, 13, 2, 3, 4, 1, 2, 3, 4, 5, 20, 21, 6, 7])
            else:
                seq.extend([14, 15, 1, 16, 17, 2, 3, 4, 1, 2, 3, 4, 5, 30, 31, 6, 7])
        arr = np.asarray(seq, dtype=np.int32)

        bundle = StructuralSubstrateBundle.fit_parallel(
            arr, vocab_size=40, n_workers=2, n_classes=12, n_singletons=4,
            feature_dim=8, max_chunks_per_level=32, n_chunk_classes=8,
            build_stage_model=False, post_fit_noise_prune=False, seed=42,
        )
        self.assertIsNotNone(bundle.context_integrator)
        ctrl = bundle.make_controller(use_arbitrator=False)

        # Prompt A ends with identical local context [2, 3, 4, 5] as Prompt B!
        prompt_a = [10, 11, 1, 12, 13, 2, 3, 4, 1, 2, 3, 4, 5]
        prompt_b = [14, 15, 1, 16, 17, 2, 3, 4, 1, 2, 3, 4, 5]

        bundle.temp_memory.reset()
        ha = np.zeros(8, dtype=float)
        for s in prompt_a:
            bundle.temp_memory.observe(s, source="external")
            ha = bundle.class_policy.update_slow_state(ha, s)
        acts_a, _, _ = ctrl.decide_distribution(prompt_a[-4:], ha, k=4)

        bundle.temp_memory.reset()
        hb = np.zeros(8, dtype=float)
        for s in prompt_b:
            bundle.temp_memory.observe(s, source="external")
            hb = bundle.class_policy.update_slow_state(hb, s)
        acts_b, _, _ = ctrl.decide_distribution(prompt_b[-4:], hb, k=4)

        # Identical last 4 tokens [2, 3, 4, 5], yet long-range integrated context selects 20 vs 30!
        self.assertEqual(acts_a[0], 20)
        self.assertEqual(acts_b[0], 30)

    def test_stateful_context_policy_kogge_stone_inplace_and_long_range_readout(self):
        from stateful_context_policy import StatefulContextPolicy, _kogge_stone_ema

        # 1. Verify vectorized Kogge-Stone EMA is bit-exact against serial recurrence
        rng = np.random.RandomState(123)
        X = rng.randn(512, 16).astype(np.float32)
        h0 = rng.randn(16).astype(np.float32)
        for g in (0.60, 0.90, 0.982):
            H_ks = _kogge_stone_ema(X, g, h0)
            H_ref = np.zeros_like(X)
            prev = h0.copy()
            for t in range(len(X)):
                prev = g * prev + (1.0 - g) * X[t]
                H_ref[t] = prev
            self.assertTrue(np.allclose(H_ks, H_ref, atol=2e-4))

        # 2. Verify standalone StatefulContextPolicy (ZERO n-gram lookup!) resolves
        #    two different macro-contexts ending with identical 6-step local bridge!
        seq = []
        for ep in range(200):
            seq.extend(rng.randint(1, 5, size=20).tolist())
            if ep % 2 == 0:
                seq.extend([10, 11, 12, 10, 11, 1, 2, 3, 4, 5, 6, 20, 21])
            else:
                seq.extend([15, 16, 17, 15, 16, 1, 2, 3, 4, 5, 6, 30, 31])
        arr = np.asarray(seq, dtype=np.int32)
        emb = rng.randn(40, 24).astype(np.float32)
        emb[0] = 0.0

        sp = StatefulContextPolicy(vocab_size=40, embed_dim=24, gate_dim=64, lr=0.025).fit_stream(
            arr, emb, sample_stride=1, max_samples=50_000,
        )
        # In-place mutation check: observe() keeps the SAME PersistentContextState object
        sp.reset_state()
        st_ref = sp.state
        for s in [10, 11, 12, 10, 11, 1, 2, 3, 4, 5, 6]:
            sp.observe(s, source="external")
        self.assertIs(sp.state, st_ref)
        pred_a = sp.predict_symbol_map(k=5)

        # Forked lookahead child modifies copy in-place without corrupting parent state
        snap_before = sp.state.h_fast.copy()
        child_st = sp.fork_step(sp.state, 20, is_self=True)
        self.assertTrue(np.allclose(sp.state.h_fast, snap_before))
        self.assertFalse(np.allclose(child_st.h_fast, snap_before))

        sp.reset_state()
        for s in [15, 16, 17, 15, 16, 1, 2, 3, 4, 5, 6]:
            sp.observe(s, source="external")
        pred_b = sp.predict_symbol_map(k=5)

        # Identical last 6 tokens [1, 2, 3, 4, 5, 6], yet persistent node state cleanly predicts 20 vs 30!
        self.assertGreater(pred_a.get(20, 0.0), pred_a.get(30, 0.0))
        self.assertGreater(pred_b.get(30, 0.0), pred_b.get(20, 0.0))

        # 3. Verify 2nd-order nonlinear XOR/bilinear context-gated transition interaction:
        #    (Ctx A, Local L1) -> 25,  (Ctx A, Local L2) -> 26
        #    (Ctx B, Local L1) -> 26,  (Ctx B, Local L2) -> 25
        #    Provably impossible for any 1st-order linear additive model, cleanly solved by
        #    our 2nd-order context gate q = tanh(x_loc P_loc) * sigmoid(z_ctx P_ctx)!
        xor_seq = []
        for ep in range(320):
            xor_seq.extend(rng.randint(1, 5, size=14).tolist())
            m = ep % 4
            if m == 0:
                xor_seq.extend([10, 11, 10, 11, 1, 2, 8, 25, 1])   # Ctx A + L1(8) -> 25
            elif m == 1:
                xor_seq.extend([10, 11, 10, 11, 1, 2, 9, 26, 1])   # Ctx A + L2(9) -> 26
            elif m == 2:
                xor_seq.extend([15, 16, 15, 16, 1, 2, 8, 26, 1])   # Ctx B + L1(8) -> 26
            else:
                xor_seq.extend([15, 16, 15, 16, 1, 2, 9, 25, 1])   # Ctx B + L2(9) -> 25
        sp_xor = StatefulContextPolicy(
            vocab_size=40, embed_dim=24, gate_dim=64, lr=0.025, seed=7,
        ).fit_stream(np.asarray(xor_seq, dtype=np.int32), emb, sample_stride=1, max_samples=50_000, epochs=25)

        for prefix, want, avoid in [
            ([10, 11, 10, 11, 1, 2, 8], 25, 26),
            ([10, 11, 10, 11, 1, 2, 9], 26, 25),
            ([15, 16, 15, 16, 1, 2, 8], 26, 25),
            ([15, 16, 15, 16, 1, 2, 9], 25, 26),
        ]:
            sp_xor.reset_state()
            for tok in prefix:
                sp_xor.observe(tok, source="external")
            p_map = sp_xor.predict_symbol_map(k=6)
            self.assertGreater(p_map.get(want, 0.0), p_map.get(avoid, 0.0))
            lbf = sp_xor.context_log_bayes_factor()
            self.assertIsNotNone(lbf)
            self.assertGreater(float(lbf[want]), float(lbf[avoid]))

    def test_selective_associative_scan_and_gpu_cpu_training_parity(self):
        from stateful_context_policy import StatefulContextPolicy, _selective_associative_scan

        # 1. Exact parity of parallel associative selective scan vs serial recurrence
        rng = np.random.RandomState(55)
        X = rng.randn(400, 16).astype(np.float32)
        wg = rng.uniform(0.01, 1.0, size=400).astype(np.float32)
        h0 = rng.randn(16).astype(np.float32)
        H_par = _selective_associative_scan(X, wg, base_alpha=0.22, h0=h0)
        H_ref = np.zeros_like(X)
        prev = h0.copy()
        for t in range(len(X)):
            w = float(np.clip(0.22 * wg[t], 0.0, 0.95))
            prev = (1.0 - w) * prev + w * X[t]
            H_ref[t] = prev
        self.assertTrue(np.allclose(H_par, H_ref, atol=2e-4))

        # 2. Exact CPU vs OpenCL GPU training parity on identical batch & seed
        emb = rng.randn(64, 16).astype(np.float32)
        emb[0] = 0.0
        sp_cpu = StatefulContextPolicy(vocab_size=64, embed_dim=16, gate_dim=32, batch_size=64, use_gpu=False, seed=9)
        sp_gpu = StatefulContextPolicy(vocab_size=64, embed_dim=16, gate_dim=32, batch_size=64, use_gpu=True, seed=9)
        sp_cpu.configure_embeddings(emb)
        sp_gpu.configure_embeddings(emb)
        if sp_gpu._ensure_gpu_trainer(64):
            Xl = rng.randn(64, sp_cpu.loc_dim).astype(np.float32) * 0.2
            Zc = rng.randn(64, sp_cpu.ctx_dim).astype(np.float32) * 0.2
            tc = rng.randint(1, 64, size=64, dtype=np.int32)
            tp = rng.randint(1, 64, size=64, dtype=np.int32)
            yb = rng.randint(1, 64, size=64, dtype=np.int32)
            for _ in range(3):
                lc = sp_cpu._cpu_train_batch(Xl, Zc, tc, tp, yb)
                lg = sp_gpu._gpu_train_batch(Xl, Zc, tc, tp, yb, read_loss=True)
                self.assertAlmostEqual(lc, lg, places=3)
            sp_gpu._sync_weights_from_gpu()
            for pname in ("E_dyn_", "P_loc_", "P_ctx_", "P_h2_", "P_ctx2_", "W_q_"):
                self.assertTrue(
                    np.allclose(getattr(sp_cpu, pname), getattr(sp_gpu, pname), atol=3e-4),
                    msg=f"GPU/CPU mismatch on {pname}",
                )
            sp_gpu.close_gpu()

    def test_whole_chunk_commitment_prevents_half_chunk_splice(self):
        # Verify ContextualIntegrator commits to completing multi-symbol chunk (20, 21)
        # once its context-promoted entry atom 20 is emitted in self-mode, and clears after 21.
        ci = ContextualIntegrator(n_units=45, vocab_size=40, forward_window=20, min_cooc=2, min_pmi=0.2)
        chunk_to_atoms = {i: (i,) for i in range(40)}
        chunk_to_atoms[40] = (20, 21)
        ci._configure_unit_metadata(chunk_to_atoms)
        ci.forward_graph_ = {10: [(40, 2.0)], 11: [(40, 2.0)]}
        ci.unit_salience_ = np.ones(45, dtype=np.float32)
        ci.is_salient_ = np.ones(45, dtype=bool)
        ci.is_salient_[0] = False
        ci._fitted = True

        ci.reset()
        ci.observe_units([10, 11], source="external")
        ci.update_chunk_commitment(20, is_self=True)
        nxt_a, c_str = ci.committed_continuation()
        self.assertEqual(nxt_a, 21)
        self.assertGreater(c_str, 0.5)

        # Emitting expected atom 21 completes and clears the commitment
        ci.update_chunk_commitment(21, is_self=True)
        nxt_after, _ = ci.committed_continuation()
        self.assertEqual(nxt_after, 0)


if __name__ == "__main__":
    unittest.main()
