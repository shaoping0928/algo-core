"""Dynamical Electronic-Element Node Field (Resistor / Capacitor / Integrator / Differentiator).

Models every node ``i`` in a batched spreading network (shape ``[B, N]``) with
physically unified derived electronic elements:
  - **Resistor (R)**: leakage resistance controlling exponential decay conductance ``1/R``
  - **Capacitor (C)**: charge-storing potential ``V_cap`` (short-term working memory)
  - **Integrator (Int)**: slow low-pass accumulated potential ``V_int`` (long-range trend)
  - **Differentiator (Diff)**: high-pass change signal ``V_diff = V_cap - V_int``
    (instantaneous onset, novelty/surprise edge, and discharge detector)
  - **Sparse CSR Coupling Network**: optional sparse synaptic/associative graph
    ``I_spread = W_csr @ V_cap`` coupling nodes dynamically across steps.

Executes identically on GPU (via ``OpenCLAccelerator``, e.g., AMD RX 9070 XT)
or CPU (NumPy), with persistent device buffers so multi-step rollouts incur
zero per-step PCIe overhead.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencl_gpu import OpenCLAccelerator


class ElectronicCircuitNodeField:
    """GPU/CPU-accelerated dynamical RC + Integrator + Differentiator node field."""

    def __init__(
        self,
        n_nodes: int,
        *,
        batch_size: int = 1,
        dt: float = 0.25,
        init_resistance: float = 4.0,
        init_capacitance: float = 2.5,
        alpha_int: float = 0.12,
        w_cap: float = 0.60,
        w_int: float = 0.25,
        w_diff: float = 0.35,
        coupling_gain: float = 0.30,
        use_gpu: bool = True,
        accelerator: Optional[OpenCLAccelerator] = None,
    ):
        self.n_nodes = max(int(n_nodes), 2)
        self.batch_size = max(int(batch_size), 1)
        self.dt = float(dt)
        self.alpha_int = float(np.clip(alpha_int, 1e-4, 1.0))
        self.w_cap = float(w_cap)
        self.w_int = float(w_int)
        self.w_diff = float(w_diff)
        self.coupling_gain = float(coupling_gain)

        # Per-node R and C parameters (learnable/adaptable per symbol/node)
        self.R = np.full(self.n_nodes, float(init_resistance), dtype=np.float32)
        self.C = np.full(self.n_nodes, float(init_capacitance), dtype=np.float32)

        # Host states [B, N]
        self.V_cap = np.zeros((self.batch_size, self.n_nodes), dtype=np.float32)
        self.V_int = np.zeros((self.batch_size, self.n_nodes), dtype=np.float32)
        self.V_diff = np.zeros((self.batch_size, self.n_nodes), dtype=np.float32)
        self.V_out = np.zeros((self.batch_size, self.n_nodes), dtype=np.float32)

        # Optional sparse CSR coupling graph
        self._csr_row_ptr: Optional[np.ndarray] = None
        self._csr_col_idx: Optional[np.ndarray] = None
        self._csr_vals: Optional[np.ndarray] = None
        self._use_gpu_pref: bool = bool(use_gpu)

        # OpenCL GPU backend setup
        self.gpu: Optional[OpenCLAccelerator] = None
        self.backend: str = "cpu"
        self._bufs: Dict[str, object] = {}
        if use_gpu:
            acc = accelerator if accelerator is not None else OpenCLAccelerator()
            if acc.available:
                self.gpu = acc
                self.backend = f"opencl:{acc.device_name}"
                self._alloc_gpu_state_buffers()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["gpu"] = None
        state["_bufs"] = {}
        state["backend"] = "cpu"
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.gpu = None
        self._bufs = {}
        self.backend = "cpu"
        if getattr(self, "_use_gpu_pref", False):
            acc = OpenCLAccelerator()
            if acc.available:
                self.gpu = acc
                self.backend = f"opencl:{acc.device_name}"
                self._alloc_gpu_state_buffers()
                if self._csr_row_ptr is not None and self._csr_col_idx is not None and self._csr_vals is not None:
                    self._bufs["row_ptr"] = acc.create_buffer_from_array(self._csr_row_ptr, read_only=True)
                    self._bufs["col_idx"] = acc.create_buffer_from_array(self._csr_col_idx, read_only=True)
                    self._bufs["vals"] = acc.create_buffer_from_array(self._csr_vals, read_only=True)

    def _alloc_gpu_state_buffers(self) -> None:
        assert self.gpu is not None
        self._release_gpu_buffers()
        g = self.gpu
        z = np.zeros((self.batch_size, self.n_nodes), dtype=np.float32)
        self._bufs["R"] = g.create_buffer_from_array(self.R, read_only=True)
        self._bufs["C"] = g.create_buffer_from_array(self.C, read_only=True)
        self._bufs["I_in"] = g.create_buffer_from_array(z)
        self._bufs["I_spread"] = g.create_buffer_from_array(z)
        self._bufs["V_cap"] = g.create_buffer_from_array(self.V_cap)
        self._bufs["V_int"] = g.create_buffer_from_array(self.V_int)
        self._bufs["V_diff"] = g.create_buffer_from_array(self.V_diff)
        self._bufs["V_out"] = g.create_buffer_from_array(self.V_out)

    def _release_gpu_buffers(self) -> None:
        if self.gpu is not None:
            for b in self._bufs.values():
                self.gpu.release_buffer(b)
        self._bufs.clear()

    def set_sparse_coupling(self, assoc_graph: Dict[int, List[Tuple[int, float]]]) -> None:
        """Compile a sparse adjacency dict ``src -> [(dst, w), ...]`` into CSR (on GPU if enabled)."""
        # Build transpose CSR so row ``dst`` sums incoming ``w * V_cap[src]``
        incoming: List[List[Tuple[int, float]]] = [[] for _ in range(self.n_nodes)]
        for src, edges in assoc_graph.items():
            si = int(src)
            if 0 <= si < self.n_nodes:
                for dst, w in edges:
                    di = int(dst)
                    if 0 <= di < self.n_nodes and float(w) != 0.0:
                        incoming[di].append((si, float(w)))

        row_ptr = [0]
        col_idx: List[int] = []
        vals: List[float] = []
        for dst in range(self.n_nodes):
            for si, w in incoming[dst]:
                col_idx.append(si)
                vals.append(w * self.coupling_gain)
            row_ptr.append(len(col_idx))

        self._csr_row_ptr = np.asarray(row_ptr, dtype=np.int32)
        self._csr_col_idx = np.asarray(col_idx if col_idx else [0], dtype=np.int32)
        self._csr_vals = np.asarray(vals if vals else [0.0], dtype=np.float32)

        if self.gpu is not None:
            for key in ("row_ptr", "col_idx", "vals"):
                if key in self._bufs:
                    self.gpu.release_buffer(self._bufs.pop(key))
            self._bufs["row_ptr"] = self.gpu.create_buffer_from_array(self._csr_row_ptr, read_only=True)
            self._bufs["col_idx"] = self.gpu.create_buffer_from_array(self._csr_col_idx, read_only=True)
            self._bufs["vals"] = self.gpu.create_buffer_from_array(self._csr_vals, read_only=True)

    def reset(self) -> None:
        self.V_cap.fill(0.0)
        self.V_int.fill(0.0)
        self.V_diff.fill(0.0)
        self.V_out.fill(0.0)
        if self.gpu is not None:
            for k in ("V_cap", "V_int", "V_diff", "V_out"):
                self.gpu.write_buffer(self._bufs[k], self.V_cap)

    def _cpu_spmv(self, x: np.ndarray) -> np.ndarray:
        if self._csr_row_ptr is None or self._csr_col_idx is None or self._csr_vals is None:
            return np.zeros_like(x)
        out = np.zeros_like(x)
        rp = self._csr_row_ptr
        ci = self._csr_col_idx
        vl = self._csr_vals
        for row in range(self.n_nodes):
            s, e = int(rp[row]), int(rp[row + 1])
            if s < e:
                out[:, row] = x[:, ci[s:e]] @ vl[s:e]
        return out

    def step(
        self,
        drive_current: np.ndarray,
        *,
        n_ticks: int = 1,
        readback: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance all ``[B, N]`` circuit nodes by ``n_ticks`` dynamical steps."""
        inp = np.ascontiguousarray(drive_current, dtype=np.float32).reshape(self.batch_size, self.n_nodes)
        ticks = max(int(n_ticks), 1)
        has_graph = (self._csr_row_ptr is not None)

        if self.gpu is not None:
            g = self.gpu
            b_iin = self._bufs["I_in"]
            g.write_buffer(b_iin, inp)
            n_tot = self.batch_size * self.n_nodes
            for tk in range(ticks):
                cur_in = b_iin
                if has_graph and tk > 0:
                    g.launch_spmv_batch(
                        self.batch_size, self.n_nodes,
                        self._bufs["row_ptr"], self._bufs["col_idx"], self._bufs["vals"],
                        self._bufs["V_cap"], self._bufs["I_spread"], sync=False,
                    )
                    cur_in = self._bufs["I_spread"]
                g.launch_circuit_step(
                    n_tot, self.n_nodes,
                    self.dt, self.alpha_int, self.w_cap, self.w_int, self.w_diff,
                    cur_in, self._bufs["R"], self._bufs["C"],
                    self._bufs["V_cap"], self._bufs["V_int"], self._bufs["V_diff"], self._bufs["V_out"],
                    sync=False,
                )
            g.finish()
            if readback:
                g.read_buffer(self._bufs["V_cap"], self.V_cap)
                g.read_buffer(self._bufs["V_int"], self.V_int)
                g.read_buffer(self._bufs["V_diff"], self.V_diff)
                g.read_buffer(self._bufs["V_out"], self.V_out)
            return self.V_cap, self.V_int, self.V_diff, self.V_out

        # Exact NumPy CPU reference path
        r = np.maximum(self.R, 1e-4)[None, :]
        c = np.maximum(self.C, 1e-4)[None, :]
        decay = np.maximum(1.0 - (self.dt / (r * c)), 0.0)
        inv_c = self.dt / c
        for tk in range(ticks):
            cur = inp if (tk == 0 or not has_graph) else self._cpu_spmv(self.V_cap)
            vc_new = decay * self.V_cap + inv_c * cur
            vi_new = (1.0 - self.alpha_int) * self.V_int + self.alpha_int * vc_new
            vd_new = vc_new - vi_new
            vout = np.tanh(self.w_cap * vc_new + self.w_int * vi_new + self.w_diff * vd_new)
            self.V_cap = vc_new.astype(np.float32, copy=False)
            self.V_int = vi_new.astype(np.float32, copy=False)
            self.V_diff = vd_new.astype(np.float32, copy=False)
            self.V_out = vout.astype(np.float32, copy=False)
        return self.V_cap, self.V_int, self.V_diff, self.V_out

    def step_single_event(
        self,
        symbol: int,
        salience: float,
        *,
        is_self: bool = False,
        assoc_neighbors: Sequence[Tuple[int, float]] = (),
        realize_gain: float = 0.65,
    ) -> None:
        """Fast O(deg + N) single-stream RC/Integrator/Differentiator update for online steps.

        - On ``is_self=False`` (external input): injects charge into ``V_cap[sym]`` and
          associative neighbors, charging the working-memory capacitor field.
        - On ``is_self=True`` (self-generated emission): integrates realized charge into
          ``V_int[sym]`` (so ``V_diff[sym] = V_cap[sym] - V_int[sym]`` naturally adapts
          downward once expressed) while spreading fresh positive drive to unexpressed
          associative neighbors (keeping ``V_diff[nb] > 0`` for forward topic progression!).
        """
        sym = int(symbol)
        if sym <= 0 or sym >= self.n_nodes:
            return
        r = np.maximum(self.R, 1e-4)
        c = np.maximum(self.C, 1e-4)
        decay = np.maximum(1.0 - (self.dt / (r * c)), 0.0)
        inv_c = self.dt / c

        vc = self.V_cap[0] * decay
        vi = self.V_int[0] * (1.0 - self.alpha_int) + self.alpha_int * vc

        sal = float(salience)
        if not is_self:
            vc[sym] += float(inv_c[sym]) * 0.70 * sal
            for nb, w in assoc_neighbors:
                ni = int(nb)
                if 0 < ni < self.n_nodes:
                    vc[ni] += float(inv_c[ni]) * float(w) * sal
        else:
            # Self-emitted node: charge enters slow integrator (realization / adaptation)
            # while associative neighbors receive fresh transient drive in capacitor!
            vi[sym] += float(realize_gain) * max(sal, 0.5)
            vc[sym] *= 0.72
            for nb, w in assoc_neighbors:
                ni = int(nb)
                if 0 < ni < self.n_nodes and ni != sym:
                    vc[ni] += float(inv_c[ni]) * 0.45 * float(w) * sal

        vd = vc - vi
        vout = np.tanh(self.w_cap * vc + self.w_int * vi + self.w_diff * vd)
        self.V_cap[0] = vc
        self.V_int[0] = vi
        self.V_diff[0] = vd
        self.V_out[0] = vout

    def sync_host_to_gpu(self) -> None:
        """Push current host circuit states to GPU buffers (for batched GPU lookahead rollouts)."""
        if self.gpu is not None and self._bufs:
            for k, arr in (("V_cap", self.V_cap), ("V_int", self.V_int), ("V_diff", self.V_diff), ("V_out", self.V_out)):
                self.gpu.write_buffer(self._bufs[k], arr)

    def close(self) -> None:
        self._release_gpu_buffers()
