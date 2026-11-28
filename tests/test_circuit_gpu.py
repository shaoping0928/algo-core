"""Unit tests for OpenCL GPU backend and ElectronicCircuitNodeField (R/C/Integrator/Differentiator)."""
from __future__ import annotations

import unittest
import numpy as np


class CircuitFieldAndGPUTests(unittest.TestCase):
    def test_circuit_element_physics_capacitor_integrator_differentiator(self):
        from circuit_field import ElectronicCircuitNodeField

        cf = ElectronicCircuitNodeField(
            n_nodes=8, batch_size=1, dt=0.3, init_resistance=5.0, init_capacitance=2.0,
            alpha_int=0.15, use_gpu=False,
        )
        drive_on = np.zeros((1, 8), dtype=np.float32)
        drive_on[0, 3] = 2.0
        drive_off = np.zeros((1, 8), dtype=np.float32)

        # Onset: capacitor charges, differentiator spikes positively
        vc1, vi1, vd1, _ = cf.step(drive_on, n_ticks=1)
        self.assertGreater(float(vc1[0, 3]), 0.0)
        self.assertGreater(float(vd1[0, 3]), 0.0)
        self.assertGreater(float(vc1[0, 3]), float(vi1[0, 3]))

        # Steady charging: integrator catches up, differentiator decays below initial onset relative to V_cap
        for _ in range(25):
            vc_s, vi_s, vd_s, _ = cf.step(drive_on, n_ticks=1)
        self.assertGreater(float(vi_s[0, 3]), float(vi1[0, 3]))

        # Step-off: capacitor discharges through resistor R, differentiator flips NEGATIVE (trailing edge)
        for _ in range(6):
            vc_off, vi_off, vd_off, _ = cf.step(drive_off, n_ticks=1)
        self.assertLess(float(vc_off[0, 3]), float(vc_s[0, 3]))
        self.assertLess(float(vd_off[0, 3]), 0.0)

    def test_opencl_gpu_matches_cpu_numerically_on_coupled_circuit_field(self):
        from circuit_field import ElectronicCircuitNodeField
        from opencl_gpu import OpenCLAccelerator

        acc = OpenCLAccelerator()
        if not acc.available:
            self.skipTest("No OpenCL GPU available on this host")

        rng = np.random.RandomState(7)
        n_nodes, batch_size = 512, 8
        assoc = {
            i: [(int((i + step) % n_nodes), float(rng.uniform(0.1, 0.8))) for step in (1, 7, 19)]
            for i in range(n_nodes)
        }
        drive = rng.randn(batch_size, n_nodes).astype(np.float32)

        cf_cpu = ElectronicCircuitNodeField(n_nodes=n_nodes, batch_size=batch_size, use_gpu=False)
        cf_gpu = ElectronicCircuitNodeField(n_nodes=n_nodes, batch_size=batch_size, use_gpu=True, accelerator=acc)
        try:
            cf_cpu.set_sparse_coupling(assoc)
            cf_gpu.set_sparse_coupling(assoc)
            c_cap, c_int, c_diff, c_out = cf_cpu.step(drive, n_ticks=5)
            g_cap, g_int, g_diff, g_out = cf_gpu.step(drive, n_ticks=5)
            self.assertTrue(np.allclose(c_cap, g_cap, atol=1e-4))
            self.assertTrue(np.allclose(c_int, g_int, atol=1e-4))
            self.assertTrue(np.allclose(c_diff, g_diff, atol=1e-4))
            self.assertTrue(np.allclose(c_out, g_out, atol=1e-4))
        finally:
            cf_gpu.close()

    def test_temp_memory_circuit_differentiator_adapts_emitted_and_propels_unexpressed(self):
        import pickle
        from temp_context import TemporaryContextMemory

        # Domain A: symbols 10, 11, 12, 13 co-occur closely; frequent glue symbols 1, 2, 3
        seq = []
        for _ in range(60):
            seq.extend([1, 2, 3, 1, 2, 3, 10, 11, 12, 13, 1, 2, 3])
            seq.extend([1, 2, 3, 1, 2, 3, 20, 21, 22, 23, 1, 2, 3])
        tokens = np.asarray(seq, dtype=np.int32)
        tcm = TemporaryContextMemory(vocab_size=30, assoc_window=6, min_cooc=2).fit_associations(tokens, stride=1)
        cf = tcm.enable_circuit_field(use_gpu=True)

        try:
            tcm.reset()
            # External context activates 10 & 11 (charging unexpressed neighbors 12, 13)
            for s in (10, 11):
                tcm.observe(s, source="external")
            vd_ext = tcm.circuit_differentiator()
            self.assertIsNotNone(vd_ext)
            self.assertGreater(float(vd_ext[12]), 0.0)

            # Repeated self-emission of 10 integrates realized charge into V_int[10],
            # driving V_diff[10] negative while keeping unexpressed neighbor 12 positive!
            for _ in range(4):
                tcm.observe(10, source="self")
            vd_self = tcm.circuit_differentiator()
            self.assertLess(float(vd_self[10]), 0.0)
            self.assertGreater(float(vd_self[12]), float(vd_self[10]))

            # Verify pickle round-trip safety (GPU buffers cleanly stripped and re-bound)
            blob = pickle.dumps(tcm)
            tcm2 = pickle.loads(blob)
            self.assertIsNotNone(tcm2.circuit_field)
            self.assertAlmostEqual(float(tcm2.circuit_differentiator()[10]), float(vd_self[10]), places=5)
            tcm2.circuit_field.close()
        finally:
            cf.close()

    def test_gpu_beam_engine_matches_cpu_across_multi_depth_forks(self):
        from circuit_field import ElectronicCircuitNodeField
        from opencl_gpu import OpenCLAccelerator

        acc = OpenCLAccelerator()
        if not acc.available:
            self.skipTest("No OpenCL GPU available on this host")

        rng = np.random.RandomState(19)
        n_nodes = 256
        assoc = {
            i: [(int((i + s) % n_nodes), float(rng.uniform(0.1, 0.7))) for s in (1, 5, 17)]
            for i in range(n_nodes)
        }
        vcap0 = rng.uniform(0.0, 1.0, size=(1, n_nodes)).astype(np.float32)
        vint0 = rng.uniform(0.0, 0.5, size=(1, n_nodes)).astype(np.float32)

        cf_cpu = ElectronicCircuitNodeField(n_nodes, batch_size=1, use_gpu=False)
        cf_gpu = ElectronicCircuitNodeField(n_nodes, batch_size=1, use_gpu=True, accelerator=acc)
        try:
            cf_cpu.set_sparse_coupling(assoc)
            cf_gpu.set_sparse_coupling(assoc)
            eng_c = cf_cpu.get_beam_engine(max_branches=32)
            eng_g = cf_gpu.get_beam_engine(max_branches=32)
            eng_c.seed_root(vcap0, vint0)
            eng_g.seed_root(vcap0, vint0)

            # Depth 1: 1 -> 8 branches
            p1 = [0] * 8
            a1 = [10, 20, 30, 40, 50, 60, 70, 80]
            s1 = [1.2, 0.0, 0.9, 1.5, 0.0, 1.1, 0.8, 1.3]
            vd_c1, vo_c1 = eng_c.fork_and_step(p1, a1, s1, n_parent_active=1)
            vd_g1, vo_g1 = eng_g.fork_and_step(p1, a1, s1, n_parent_active=1)
            self.assertTrue(np.allclose(vd_c1, vd_g1, atol=1e-4))
            self.assertTrue(np.allclose(vo_c1, vo_g1, atol=1e-4))

            # Depth 2: 8 -> 24 branches
            p2 = [i % 8 for i in range(24)]
            a2 = [int((i * 7 + 3) % n_nodes) for i in range(24)]
            s2 = [float(0.0 if i % 3 == 0 else 1.0) for i in range(24)]
            vd_c2, vo_c2 = eng_c.fork_and_step(p2, a2, s2, n_parent_active=8)
            vd_g2, vo_g2 = eng_g.fork_and_step(p2, a2, s2, n_parent_active=8)
            self.assertTrue(np.allclose(vd_c2, vd_g2, atol=1e-4))
            self.assertTrue(np.allclose(vo_c2, vo_g2, atol=1e-4))
        finally:
            cf_gpu.close()

    def test_decide_with_lookahead_runs_multi_branch_circuit_on_gpu(self):
        from class_policy import ClassLevelSequencePolicy
        from context_decision import ContextDrivenDecisionController
        from temp_context import TemporaryContextMemory
        from virtual_node_dag import BidirectionalVirtualNodeDAG

        seq = []
        for _ in range(40):
            seq.extend([1, 2, 3, 1, 2, 3, 10, 11, 12, 13, 14, 1, 2])
            seq.extend([1, 2, 3, 1, 2, 3, 20, 21, 22, 23, 24, 1, 2])
        tokens = np.asarray(seq, dtype=np.int32)
        pol = ClassLevelSequencePolicy(vocab_size=30, n_classes=6, n_singletons=3, feature_dim=8, window=3, seed=7).fit(tokens)
        dag = BidirectionalVirtualNodeDAG(vocab_size=30, min_trigger_count=5, min_pair_count=3, min_lift=1.1).fit_sequence(tokens)
        tcm = TemporaryContextMemory(vocab_size=30, assoc_window=6, min_cooc=2).fit_associations(tokens, stride=1)
        cf = tcm.enable_circuit_field(use_gpu=True)
        try:
            ctrl = ContextDrivenDecisionController(pol, dag, tcm, lookahead_depth=3, lookahead_beam=4)
            tcm.reset()
            for s in (10, 11, 12):
                tcm.observe(s, source="external")
            acts, probs, info = ctrl.decide_with_lookahead([10, 11, 12], k=4, depth=3, beam_width=4)
            self.assertEqual(acts[0], 13)
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
            self.assertEqual(info["lookahead_depth"], 3.0)
            if cf.gpu is not None:
                self.assertEqual(info["beam_circuit_gpu"], 1.0)
        finally:
            cf.close()


if __name__ == "__main__":
    unittest.main()
