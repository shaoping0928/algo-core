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


if __name__ == "__main__":
    unittest.main()
