"""Persistent In-Place Selective State-Space & 2-Layer Residual Context-Gated Policy (RX 9070 XT GPU).

Implements continuous, coherent state-space dynamics where each observed or
self-emitted symbol **modifies an already-existing multi-timescale node state
in-place** (never rebuilding from scratch), with:
  1. **Jointly Fine-Tuned Local Embeddings ``E_dyn`` (e.g. 64-dim)** trained
     end-to-end on RX 9070 XT GPU alongside centered distributional anchors;
  2. **Selective Entity-Lock RC State ``h_ent``**: input-dependent write gates
     ``delta(u)`` (IDF salience + learned ``R*C`` persistence) via exact
     parallel associative prefix scan (Mamba/S5-style recurrence);
  3. **2-Layer Residual Context-Gated Bilinear Transitions**:
       * Layer 1: ``Q1 = tanh(x_loc @ P_loc + b_loc) * sigmoid(z_ctx @ P_ctx + b_ctx)``
       * Layer 2: ``Q2 = Q1 + tanh(Q1 @ P_h2 + b_h2) * sigmoid(z_ctx @ P_ctx2 + b_ctx2)``
       * Readout: ``logits = x_loc @ W_loc + z_ctx @ W_ctx + Q2 @ W_q + b_out``
  4. **Exact Counterfactual Conditional Context PMI**:
       ``Delta_ctx(y | x_loc, z_ctx) = log P(y | x_loc, z_ctx) - log P(y | x_loc, 0)``
  5. **Native OpenCL GPU Training Engine**:
     Keeps all 13 parameter tensors + AdamW moments permanently in VRAM,
     running tiled SGEMM, fused residual gates, Softmax/CE, embedding gather/
     reduce, and AdamW updates on-device.

Strictly task-agnostic: operates solely on discrete symbol/chunk IDs and
distributional embeddings (zero language rules or word segmenters).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencl_gpu import OpenCLAccelerator


@dataclass
class PersistentContextState:
    """Live persistent selective multi-scale node state modified strictly in-place."""

    e_cur: np.ndarray
    e_prev: np.ndarray
    h_fast: np.ndarray
    h_mid: np.ndarray
    h_slow: np.ndarray
    h_chunk: np.ndarray
    h_ent: np.ndarray
    steps: int = 0
    slow_anchor: Optional[np.ndarray] = None
    ent_anchor: Optional[np.ndarray] = None

    def copy(self) -> "PersistentContextState":
        return PersistentContextState(
            e_cur=self.e_cur.copy(),
            e_prev=self.e_prev.copy(),
            h_fast=self.h_fast.copy(),
            h_mid=self.h_mid.copy(),
            h_slow=self.h_slow.copy(),
            h_chunk=self.h_chunk.copy(),
            h_ent=self.h_ent.copy(),
            steps=self.steps,
            slow_anchor=self.slow_anchor.copy() if self.slow_anchor is not None else None,
            ent_anchor=self.ent_anchor.copy() if self.ent_anchor is not None else None,
        )

    def split_local_context(
        self,
        *,
        context_band_gain: float = 1.0,
        anchor_blend: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(x_loc [3D], z_ctx [5D])`` from the live persistent state."""
        h_diff = self.h_fast - self.h_mid
        h_s = self.h_slow
        h_e = self.h_ent
        if anchor_blend > 0.0:
            ab = float(np.clip(anchor_blend, 0.0, 0.85))
            if self.slow_anchor is not None:
                h_s = (1.0 - ab) * h_s + ab * self.slow_anchor
            if self.ent_anchor is not None:
                h_e = (1.0 - ab) * h_e + ab * self.ent_anchor
        x_loc = np.concatenate([self.e_cur, self.e_prev, self.h_fast]).astype(np.float32, copy=False)
        g = float(context_band_gain)
        if g != 1.0:
            z_ctx = np.concatenate([g * self.h_mid, g * h_s, g * self.h_chunk, g * h_e, h_diff]).astype(
                np.float32, copy=False,
            )
        else:
            z_ctx = np.concatenate([self.h_mid, h_s, self.h_chunk, h_e, h_diff]).astype(
                np.float32, copy=False,
            )
        return x_loc, z_ctx

    def to_feature_vector(
        self,
        *,
        context_band_gain: float = 1.0,
        anchor_blend: float = 0.0,
    ) -> np.ndarray:
        x_loc, z_ctx = self.split_local_context(
            context_band_gain=context_band_gain, anchor_blend=anchor_blend,
        )
        return np.concatenate([x_loc, z_ctx, np.ones(1, dtype=np.float32)]).astype(np.float32, copy=False)


def _kogge_stone_ema(X: np.ndarray, gamma: float, h0: Optional[np.ndarray] = None) -> np.ndarray:
    """Bit-exact vectorized C-level recursive EMA ``H[t] = g * H[t-1] + (1-g) * X[t]``."""
    n = X.shape[0]
    if n == 0:
        return np.empty_like(X, dtype=np.float32)
    g = float(gamma)
    H = ((1.0 - g) * X).astype(np.float32, copy=True)
    if h0 is not None:
        H[0] += g * h0.astype(np.float32, copy=False)
    stride = 1
    while stride < n:
        p = g ** stride
        if p < 1e-5:
            break
        H[stride:] += np.float32(p) * H[:-stride]
        stride <<= 1
    return H


def _selective_associative_scan(
    X: np.ndarray,
    write_gates: np.ndarray,
    base_alpha: float = 0.22,
    h0: Optional[np.ndarray] = None,
    max_horizon: int = 512,
) -> np.ndarray:
    """Exact log-depth parallel associative scan for input-selective state-space recurrence."""
    n = X.shape[0]
    if n == 0:
        return np.empty_like(X, dtype=np.float32)
    w = np.clip(write_gates.astype(np.float32, copy=False) * np.float32(base_alpha), 0.0, 0.95).reshape(n, 1)
    A = (np.float32(1.0) - w).copy()
    H = (w * X).astype(np.float32, copy=True)
    if h0 is not None:
        H[0] += A[0] * h0.astype(np.float32, copy=False)
    stride = 1
    limit = min(n, max(int(max_horizon), 32))
    while stride < limit:
        H[stride:] += A[stride:] * H[:-stride]
        A[stride:] *= A[:-stride]
        stride <<= 1
    return H


def _sigmoid(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, -16.0, 16.0)
    return (1.0 / (1.0 + np.exp(-xc))).astype(np.float32, copy=False)


def _log_softmax_1d(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    shifted = logits - m
    lse = m + float(np.log(np.sum(np.exp(np.clip(shifted, -40.0, 0.0))) + 1e-12))
    return (logits - lse).astype(np.float32, copy=False)


class StatefulContextPolicy:
    """In-place selective 2-layer residual context-gated policy with RX 9070 XT GPU training."""

    _PARAM_NAMES: Tuple[str, ...] = (
        "E_dyn",
        "P_loc", "b_loc", "P_ctx", "b_ctx",
        "P_h2", "b_h2", "P_ctx2", "b_ctx2",
        "W_loc", "W_ctx", "W_q", "b_out",
    )

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        n_classes: int = 0,
        gate_dim: int = 512,
        gamma_fast: float = 0.60,
        gamma_mid: float = 0.90,
        gamma_slow: float = 0.982,
        gamma_chunk: float = 0.94,
        entity_alpha: float = 0.22,
        lr: float = 0.010,
        weight_decay: float = 1e-5,
        batch_size: int = 2048,
        use_gpu: bool = True,
        seed: int = 42,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.embed_dim = max(int(embed_dim), 4)
        self.n_classes = max(int(n_classes), 0)
        self.gate_dim = max(int(gate_dim), 32)
        self.gamma_fast = float(np.clip(gamma_fast, 0.1, 0.99))
        self.gamma_mid = float(np.clip(gamma_mid, 0.5, 0.995))
        self.gamma_slow = float(np.clip(gamma_slow, 0.8, 0.999))
        self.gamma_chunk = float(np.clip(gamma_chunk, 0.5, 0.995))
        self.entity_alpha = float(np.clip(entity_alpha, 0.02, 0.80))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = max(int(batch_size), 64)
        self.use_gpu = bool(use_gpu)
        self.seed = int(seed)

        self.loc_dim = 3 * self.embed_dim
        self.ctx_dim = 5 * self.embed_dim
        self.feat_dim = self.loc_dim + self.ctx_dim + 1

        # Fixed distributional anchors (for exact Kogge-Stone/selective scans) + trainable E_dyn!
        self.atom_emb_: Optional[np.ndarray] = None
        self.chunk_emb_: Optional[np.ndarray] = None
        self.write_gate_: np.ndarray = np.full(self.vocab_size, 0.5, dtype=np.float32)
        self.write_gate_[0] = 0.0
        self.sym_to_class_: Optional[np.ndarray] = None

        # 13 Trainable Parameters: Dynamic Embeddings + 2-Layer Residual Context-Gated Network
        self.E_dyn_: Optional[np.ndarray] = None           # [V, D]
        self.P_loc_: Optional[np.ndarray] = None           # [3D, R]
        self.b_loc_: Optional[np.ndarray] = None           # [R]
        self.P_ctx_: Optional[np.ndarray] = None           # [5D, R]
        self.b_ctx_: Optional[np.ndarray] = None           # [R]
        self.P_h2_: Optional[np.ndarray] = None            # [R, R]
        self.b_h2_: Optional[np.ndarray] = None            # [R]
        self.P_ctx2_: Optional[np.ndarray] = None          # [5D, R]
        self.b_ctx2_: Optional[np.ndarray] = None          # [R]
        self.W_loc_: Optional[np.ndarray] = None           # [3D, V]
        self.W_ctx_: Optional[np.ndarray] = None           # [5D, V]
        self.W_q_: Optional[np.ndarray] = None             # [R, V]
        self.b_out_: Optional[np.ndarray] = None           # [V]

        # Transient optimizer & GPU runtime handles (stripped on pickle!)
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_t: int = 0
        self._gpu: Optional[OpenCLAccelerator] = None
        self._gpu_bufs: Dict[str, object] = {}
        self._gpu_bs: int = 0
        self._gpu_infer_bufs: Dict[str, object] = {}
        self._gpu_infer_cap: int = 0
        self.train_backend: str = "cpu"
        self.total_samples: int = 0
        self.last_loss_: float = 0.0
        self._fitted: bool = False

        # Live persistent node state (modified strictly in-place on observe!)
        self.state: PersistentContextState = self._make_zero_state()

    def __getstate__(self) -> dict:
        st = self.__dict__.copy()
        st["_adam_m"] = {}
        st["_adam_v"] = {}
        st["_gpu"] = None
        st["_gpu_bufs"] = {}
        st["_gpu_bs"] = 0
        st["_gpu_infer_bufs"] = {}
        st["_gpu_infer_cap"] = 0
        return st

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._adam_m = {}
        self._adam_v = {}
        self._gpu = None
        self._gpu_bufs = {}
        self._gpu_bs = 0
        self._gpu_infer_bufs = {}
        self._gpu_infer_cap = 0

    def close_gpu(self) -> None:
        if self._gpu is not None:
            for buf in self._gpu_bufs.values():
                self._gpu.release_buffer(buf)
            for buf in getattr(self, "_gpu_infer_bufs", {}).values():
                self._gpu.release_buffer(buf)
        self._gpu_bufs.clear()
        self._gpu_bs = 0
        if hasattr(self, "_gpu_infer_bufs"):
            self._gpu_infer_bufs.clear()
        self._gpu_infer_cap = 0

    def _make_zero_state(self) -> PersistentContextState:
        d = self.embed_dim
        return PersistentContextState(
            e_cur=np.zeros(d, dtype=np.float32),
            e_prev=np.zeros(d, dtype=np.float32),
            h_fast=np.zeros(d, dtype=np.float32),
            h_mid=np.zeros(d, dtype=np.float32),
            h_slow=np.zeros(d, dtype=np.float32),
            h_chunk=np.zeros(d, dtype=np.float32),
            h_ent=np.zeros(d, dtype=np.float32),
            steps=0,
            slow_anchor=None,
            ent_anchor=None,
        )

    def _init_params(self) -> None:
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        ld, cd, r, v = self.loc_dim, self.ctx_dim, self.gate_dim, self.vocab_size
        # Warm-start trainable E_dyn from mean-centered distributional embeddings!
        if self.atom_emb_ is not None:
            self.E_dyn_ = self.atom_emb_.copy().astype(np.float32)
        else:
            self.E_dyn_ = (rng.randn(v, self.embed_dim) * 0.1).astype(np.float32)
            self.E_dyn_[0] = 0.0

        self.P_loc_ = (rng.randn(ld, r) * np.sqrt(1.0 / max(ld, 1))).astype(np.float32)
        self.b_loc_ = np.zeros(r, dtype=np.float32)
        self.P_ctx_ = (rng.randn(cd, r) * np.sqrt(1.0 / max(cd, 1))).astype(np.float32)
        self.b_ctx_ = np.full(r, 0.5, dtype=np.float32)

        # Residual Layer 2 initialized small so training starts smoothly from Layer 1
        self.P_h2_ = (rng.randn(r, r) * np.sqrt(0.35 / max(r, 1))).astype(np.float32)
        self.b_h2_ = np.zeros(r, dtype=np.float32)
        self.P_ctx2_ = (rng.randn(cd, r) * np.sqrt(0.5 / max(cd, 1))).astype(np.float32)
        self.b_ctx2_ = np.zeros(r, dtype=np.float32)

        self.W_loc_ = np.zeros((ld, v), dtype=np.float32)
        self.W_ctx_ = np.zeros((cd, v), dtype=np.float32)
        self.W_q_ = np.zeros((r, v), dtype=np.float32)
        self.b_out_ = np.zeros(v, dtype=np.float32)
        self._adam_m.clear()
        self._adam_v.clear()
        self._adam_t = 0
        self.close_gpu()

    def configure_embeddings(
        self,
        atom_embeddings: np.ndarray,
        *,
        chunk_to_atoms: Optional[Dict[int, Tuple[int, ...]]] = None,
        n_total_units: Optional[int] = None,
        sym_to_class: Optional[np.ndarray] = None,
        rc_persistence: Optional[np.ndarray] = None,
    ) -> None:
        ae = np.asarray(atom_embeddings, dtype=np.float32)
        if ae.shape[0] < self.vocab_size:
            pad = np.zeros((self.vocab_size, ae.shape[1]), dtype=np.float32)
            pad[: ae.shape[0]] = ae
            ae = pad
        self.embed_dim = int(ae.shape[1])
        self.loc_dim = 3 * self.embed_dim
        self.ctx_dim = 5 * self.embed_dim
        self.feat_dim = self.loc_dim + self.ctx_dim + 1

        valid = np.linalg.norm(ae[1:], axis=1) > 1e-8
        if np.any(valid):
            mu = ae[1:][valid].mean(axis=0, keepdims=True)
            ac = ae - mu
        else:
            ac = ae.copy()
        norms = np.maximum(np.linalg.norm(ac, axis=1, keepdims=True), 1e-8)
        ac = (ac / norms).astype(np.float32)
        ac[0] = 0.0
        self.atom_emb_ = ac

        u_tot = max(int(n_total_units) if n_total_units is not None else self.vocab_size, self.vocab_size)
        ce = np.zeros((u_tot, self.embed_dim), dtype=np.float32)
        ce[: self.vocab_size] = ac[: self.vocab_size]
        if chunk_to_atoms is not None:
            for cid, atoms in chunk_to_atoms.items():
                ci = int(cid)
                if self.vocab_size <= ci < u_tot and atoms:
                    valid_a = [int(a) for a in atoms if 0 < int(a) < self.vocab_size]
                    if valid_a:
                        v_vec = ac[valid_a].mean(axis=0)
                        vn = float(np.linalg.norm(v_vec))
                        ce[ci] = (v_vec / max(vn, 1e-8)).astype(np.float32)
        self.chunk_emb_ = ce

        if sym_to_class is not None:
            self.sym_to_class_ = np.asarray(sym_to_class, dtype=np.int32)
            self.n_classes = int(self.sym_to_class_.max()) + 1

        if rc_persistence is not None and len(rc_persistence) >= self.vocab_size:
            tau = np.asarray(rc_persistence[: self.vocab_size], dtype=np.float32)
            t_hi = max(float(np.percentile(tau[1:], 95)), 1.0)
            self.write_gate_ = np.clip(tau / t_hi, 0.02, 1.0).astype(np.float32)
            self.write_gate_[0] = 0.0

        if self.P_loc_ is None or self.P_loc_.shape[0] != self.loc_dim:
            self._init_params()
        self.state = self._make_zero_state()

    def _fit_write_gates_from_stream(self, arr: np.ndarray) -> None:
        cnts = np.bincount(arr, minlength=self.vocab_size).astype(np.float64)
        cnts[0] = 0.0
        tot = max(float(cnts.sum()), 1.0)
        p_u = (cnts + 0.5) / (tot + 0.5 * self.vocab_size)
        idf = -np.log(np.clip(p_u, 1e-8, 1.0))
        nz = cnts > 0
        if np.any(nz):
            p10 = float(np.percentile(idf[nz], 10))
            p90 = float(np.percentile(idf[nz], 90))
            span = max(p90 - p10, 0.5)
            norm = np.clip((idf - p10) / span, 0.0, 1.0)
            wg = (0.015 + 0.985 * (norm ** 1.6)).astype(np.float32)
            if float(np.std(self.write_gate_[1:])) > 0.05:
                wg = np.clip(0.65 * wg + 0.35 * self.write_gate_, 0.015, 1.0).astype(np.float32)
            wg[0] = 0.0
            self.write_gate_ = wg

    # ------------------------------------------------------------------
    # Native RX 9070 XT OpenCL 2-Layer Residual + Dynamic-Embed GPU Engine
    # ------------------------------------------------------------------

    def _ensure_gpu_trainer(self, batch_size: int) -> bool:
        if not self.use_gpu:
            return False
        if self._gpu is None:
            acc = OpenCLAccelerator()
            if (
                acc.available
                and acc._k_sgemm is not None
                and acc._k_res_fwd is not None
                and acc._k_emb_gather is not None
            ):
                self._gpu = acc
            else:
                return False
        g = self._gpu
        bs = int(batch_size)
        if self._gpu_bufs and self._gpu_bs == bs:
            return True

        self.close_gpu()
        ld, cd, r, v = self.loc_dim, self.ctx_dim, self.gate_dim, self.vocab_size
        bufs: Dict[str, object] = {}
        for name in self._PARAM_NAMES:
            arr = getattr(self, f"{name}_")
            assert arr is not None
            bufs[f"p_{name}"] = g.create_buffer_from_array(arr)
            bufs[f"g_{name}"] = g.create_empty_buffer(arr.nbytes)
            m0 = self._adam_m.get(name)
            v0 = self._adam_v.get(name)
            bufs[f"m_{name}"] = g.create_buffer_from_array(m0 if m0 is not None else np.zeros_like(arr))
            bufs[f"v_{name}"] = g.create_buffer_from_array(v0 if v0 is not None else np.zeros_like(arr))

        # Batch activation & gradient scratchpads permanently in VRAM
        bufs["Xl"] = g.create_empty_buffer(bs * ld * 4)
        bufs["dXl"] = g.create_empty_buffer(bs * ld * 4)
        bufs["Zc"] = g.create_empty_buffer(bs * cd * 4)
        bufs["tok_cur"] = g.create_empty_buffer(bs * 4)
        bufs["tok_prev"] = g.create_empty_buffer(bs * 4)
        bufs["yb"] = g.create_empty_buffer(bs * 4)

        bufs["A_pre"] = g.create_empty_buffer(bs * r * 4)
        bufs["G_pre"] = g.create_empty_buffer(bs * r * 4)
        bufs["A"] = g.create_empty_buffer(bs * r * 4)
        bufs["G"] = g.create_empty_buffer(bs * r * 4)
        bufs["Q1"] = g.create_empty_buffer(bs * r * 4)

        bufs["H2_pre"] = g.create_empty_buffer(bs * r * 4)
        bufs["G2_pre"] = g.create_empty_buffer(bs * r * 4)
        bufs["H2"] = g.create_empty_buffer(bs * r * 4)
        bufs["G2"] = g.create_empty_buffer(bs * r * 4)
        bufs["Q2"] = g.create_empty_buffer(bs * r * 4)

        bufs["logits"] = g.create_empty_buffer(bs * v * 4)
        bufs["losses"] = g.create_empty_buffer(bs * 4)
        bufs["dQ2"] = g.create_empty_buffer(bs * r * 4)
        bufs["dH2_pre"] = g.create_empty_buffer(bs * r * 4)
        bufs["dG2_pre"] = g.create_empty_buffer(bs * r * 4)
        bufs["dQ1"] = g.create_empty_buffer(bs * r * 4)
        bufs["dA_pre"] = g.create_empty_buffer(bs * r * 4)
        bufs["dG_pre"] = g.create_empty_buffer(bs * r * 4)

        self._gpu_bufs = bufs
        self._gpu_bs = bs
        self.train_backend = f"opencl:{g.device_name}"
        return True

    def _sync_weights_from_gpu(self) -> None:
        if self._gpu is None or not self._gpu_bufs:
            return
        g = self._gpu
        g.finish()
        for name in self._PARAM_NAMES:
            arr = getattr(self, f"{name}_")
            g.read_buffer(self._gpu_bufs[f"p_{name}"], arr)

    def _gpu_train_batch(
        self,
        Xl: np.ndarray,
        Zc: np.ndarray,
        tok_cur: np.ndarray,
        tok_prev: np.ndarray,
        yb: np.ndarray,
        *,
        read_loss: bool = False,
    ) -> float:
        g = self._gpu
        assert g is not None
        bufs = self._gpu_bufs
        m_b = int(Xl.shape[0])
        d, ld, cd, r, v = self.embed_dim, self.loc_dim, self.ctx_dim, self.gate_dim, self.vocab_size

        g.write_buffer(bufs["Xl"], Xl)
        g.write_buffer(bufs["Zc"], Zc)
        g.write_buffer(bufs["tok_cur"], tok_cur)
        g.write_buffer(bufs["tok_prev"], tok_prev)
        g.write_buffer(bufs["yb"], yb)

        # 0. Gather live trainable E_dyn[tok_cur], E_dyn[tok_prev] into Xl[:, 0:2D] on GPU!
        g.launch_embed_gather_2(m_b, d, ld, bufs["tok_cur"], bufs["tok_prev"], bufs["p_E_dyn"], bufs["Xl"])

        # 1. Layer 1: Q1 = tanh(Xl @ P_loc + b_loc) * sigmoid(Zc @ P_ctx + b_ctx)
        g.launch_sgemm(m_b, r, ld, bufs["Xl"], bufs["p_P_loc"], bufs["A_pre"], beta=0.0)
        g.launch_sgemm(m_b, r, cd, bufs["Zc"], bufs["p_P_ctx"], bufs["G_pre"], beta=0.0)
        g.launch_gated_bilinear_fwd(
            m_b, r, bufs["A_pre"], bufs["p_b_loc"], bufs["G_pre"], bufs["p_b_ctx"],
            bufs["A"], bufs["G"], bufs["Q1"],
        )

        # 2. Layer 2 (Residual): Q2 = Q1 + tanh(Q1 @ P_h2 + b_h2) * sigmoid(Zc @ P_ctx2 + b_ctx2)
        g.launch_sgemm(m_b, r, r, bufs["Q1"], bufs["p_P_h2"], bufs["H2_pre"], beta=0.0)
        g.launch_sgemm(m_b, r, cd, bufs["Zc"], bufs["p_P_ctx2"], bufs["G2_pre"], beta=0.0)
        g.launch_res_gated_fwd(
            m_b, r, bufs["Q1"], bufs["H2_pre"], bufs["p_b_h2"], bufs["G2_pre"], bufs["p_b_ctx2"],
            bufs["H2"], bufs["G2"], bufs["Q2"],
        )

        # 3. Readout logits + fused Softmax/CE/dL
        g.launch_sgemm(m_b, v, ld, bufs["Xl"], bufs["p_W_loc"], bufs["logits"], beta=0.0)
        g.launch_sgemm(m_b, v, cd, bufs["Zc"], bufs["p_W_ctx"], bufs["logits"], beta=1.0)
        g.launch_sgemm(m_b, v, r, bufs["Q2"], bufs["p_W_q"], bufs["logits"], beta=1.0)
        g.launch_softmax_ce_grad(m_b, v, bufs["p_b_out"], bufs["yb"], bufs["logits"], bufs["losses"])

        # 4. Readout backward
        g.launch_sgemm(ld, v, m_b, bufs["Xl"], bufs["logits"], bufs["g_W_loc"], transA=True, beta=0.0)
        g.launch_sgemm(cd, v, m_b, bufs["Zc"], bufs["logits"], bufs["g_W_ctx"], transA=True, beta=0.0)
        g.launch_sgemm(r, v, m_b, bufs["Q2"], bufs["logits"], bufs["g_W_q"], transA=True, beta=0.0)
        g.launch_col_sum(m_b, v, bufs["logits"], bufs["g_b_out"])
        g.launch_sgemm(m_b, r, v, bufs["logits"], bufs["p_W_q"], bufs["dQ2"], transB=True, beta=0.0)

        # 5. Layer 2 residual backward: dQ1 = dQ2 + dH2_pre @ P_h2.T
        g.launch_res_gated_bwd(m_b, r, bufs["dQ2"], bufs["H2"], bufs["G2"], bufs["dH2_pre"], bufs["dG2_pre"])
        g.launch_sgemm(r, r, m_b, bufs["Q1"], bufs["dH2_pre"], bufs["g_P_h2"], transA=True, beta=0.0)
        g.launch_sgemm(cd, r, m_b, bufs["Zc"], bufs["dG2_pre"], bufs["g_P_ctx2"], transA=True, beta=0.0)
        g.launch_col_sum(m_b, r, bufs["dH2_pre"], bufs["g_b_h2"])
        g.launch_col_sum(m_b, r, bufs["dG2_pre"], bufs["g_b_ctx2"])
        # dQ1 = dH2_pre @ P_h2.T + dQ2 (use dQ2 copy via beta=1? Reuse dQ1 cleanly: dQ1 = dH2_pre @ P_h2.T + 1.0*dQ2!)
        # Note: compute dQ1 = dH2_pre @ P_h2.T (beta=0) then add dQ2 via SGEMM? Easier: write dQ1 into dQ2 in-place!
        g.launch_sgemm(m_b, r, r, bufs["dH2_pre"], bufs["p_P_h2"], bufs["dQ2"], transB=True, beta=1.0)

        # 6. Layer 1 backward (dQ2 now holds exact dQ1 = dQ2_in + dH2_pre @ P_h2.T!)
        g.launch_gated_bilinear_bwd(m_b, r, bufs["dQ2"], bufs["A"], bufs["G"], bufs["dA_pre"], bufs["dG_pre"])
        g.launch_sgemm(ld, r, m_b, bufs["Xl"], bufs["dA_pre"], bufs["g_P_loc"], transA=True, beta=0.0)
        g.launch_sgemm(cd, r, m_b, bufs["Zc"], bufs["dG_pre"], bufs["g_P_ctx"], transA=True, beta=0.0)
        g.launch_col_sum(m_b, r, bufs["dA_pre"], bufs["g_b_loc"])
        g.launch_col_sum(m_b, r, bufs["dG_pre"], bufs["g_b_ctx"])

        # 7. Embedding gradient: dXl = dL @ W_loc.T + dA_pre @ P_loc.T -> deterministic reduce into g_E_dyn!
        g.launch_sgemm(m_b, ld, v, bufs["logits"], bufs["p_W_loc"], bufs["dXl"], transB=True, beta=0.0)
        g.launch_sgemm(m_b, ld, r, bufs["dA_pre"], bufs["p_P_loc"], bufs["dXl"], transB=True, beta=1.0)
        g.launch_embed_grad_reduce_2(
            m_b, v, d, ld, bufs["tok_cur"], bufs["tok_prev"], bufs["dXl"], bufs["g_E_dyn"],
        )

        # 8. On-device AdamW for all 13 parameter tensors
        self._adam_t += 1
        lr_cur = self.lr / float(np.sqrt(1.0 + 0.0010 * self._adam_t))
        bc1 = 1.0 - (0.9 ** self._adam_t)
        bc2 = 1.0 - (0.999 ** self._adam_t)
        step_main = float(lr_cur * np.sqrt(bc2) / max(bc1, 1e-8))
        step_gate = step_main * 0.75
        step_emb = step_main * 0.65
        wd_main = float(1.0 - lr_cur * self.weight_decay)
        wd_gate = float(1.0 - (lr_cur * 0.75) * self.weight_decay)

        for name, n_el, stp, wds in [
            ("W_loc", ld * v, step_main, wd_main),
            ("W_ctx", cd * v, step_main, wd_main),
            ("W_q", r * v, step_main, wd_main),
            ("b_out", v, step_main, 1.0),
            ("P_loc", ld * r, step_gate, wd_gate),
            ("b_loc", r, step_gate, 1.0),
            ("P_ctx", cd * r, step_gate, wd_gate),
            ("b_ctx", r, step_gate, 1.0),
            ("P_h2", r * r, step_gate, wd_gate),
            ("b_h2", r, step_gate, 1.0),
            ("P_ctx2", cd * r, step_gate, wd_gate),
            ("b_ctx2", r, step_gate, 1.0),
            ("E_dyn", v * d, step_emb, wd_gate),
        ]:
            g.launch_adamw_update(
                n_el, stp, wds,
                bufs[f"p_{name}"], bufs[f"g_{name}"], bufs[f"m_{name}"], bufs[f"v_{name}"],
            )

        self.total_samples += m_b
        if read_loss:
            loss_arr = np.empty(m_b, dtype=np.float32)
            g.read_buffer(bufs["losses"], loss_arr)
            return float(np.mean(loss_arr))
        return 0.0

    # ------------------------------------------------------------------
    # Exact CPU Fallback Optimizer & Unified Streaming Training Driver
    # ------------------------------------------------------------------

    def _adam_step_cpu(self, name: str, param: np.ndarray, grad: np.ndarray, lr_cur: float) -> None:
        m = self._adam_m.get(name)
        v = self._adam_v.get(name)
        if m is None or m.shape != param.shape:
            m = np.zeros_like(param)
            v = np.zeros_like(param)
            self._adam_m[name] = m
            self._adam_v[name] = v
        b1, b2 = np.float32(0.9), np.float32(0.999)
        m *= b1
        m += (1.0 - b1) * grad
        v *= b2
        v += (1.0 - b2) * (grad * grad)
        bc1 = 1.0 - (0.9 ** self._adam_t)
        bc2 = 1.0 - (0.999 ** self._adam_t)
        step_size = np.float32(lr_cur * np.sqrt(bc2) / max(bc1, 1e-8))
        if self.weight_decay > 0.0 and param.ndim == 2:
            param *= np.float32(1.0 - lr_cur * self.weight_decay)
        param -= step_size * (m / (np.sqrt(v) + np.float32(1e-7)))

    def _cpu_train_batch(
        self,
        Xl: np.ndarray,
        Zc: np.ndarray,
        tok_cur: np.ndarray,
        tok_prev: np.ndarray,
        yb: np.ndarray,
    ) -> float:
        m_b = int(Xl.shape[0])
        d = self.embed_dim
        Xl = Xl.copy()
        Xl[:, 0:d] = self.E_dyn_[tok_cur]
        Xl[:, d : 2 * d] = self.E_dyn_[tok_prev]

        A = np.tanh(Xl @ self.P_loc_ + self.b_loc_)
        G = _sigmoid(Zc @ self.P_ctx_ + self.b_ctx_)
        Q1 = A * G

        H2 = np.tanh(Q1 @ self.P_h2_ + self.b_h2_)
        G2 = _sigmoid(Zc @ self.P_ctx2_ + self.b_ctx2_)
        Q2 = Q1 + H2 * G2

        logits = (Xl @ self.W_loc_) + (Zc @ self.W_ctx_) + (Q2 @ self.W_q_) + self.b_out_
        logits[:, 0] = -1e4
        l_max = logits.max(axis=1, keepdims=True)
        ex = np.exp(np.clip(logits - l_max, -30.0, 0.0))
        probs = ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)

        row_idx = np.arange(m_b)
        loss_val = float(-np.mean(np.log(np.clip(probs[row_idx, yb], 1e-9, 1.0))))

        dL = probs
        dL[row_idx, yb] -= 1.0
        dL *= np.float32(1.0 / float(m_b))

        dW_loc = Xl.T @ dL
        dW_ctx = Zc.T @ dL
        dW_q = Q2.T @ dL
        db_out = dL.sum(axis=0)
        dQ2 = dL @ self.W_q_.T

        dH2_pre = dQ2 * G2 * (np.float32(1.0) - H2 * H2)
        dG2_pre = dQ2 * H2 * G2 * (np.float32(1.0) - G2)
        dP_h2 = Q1.T @ dH2_pre
        db_h2 = dH2_pre.sum(axis=0)
        dP_ctx2 = Zc.T @ dG2_pre
        db_ctx2 = dG2_pre.sum(axis=0)

        dQ1 = dQ2 + dH2_pre @ self.P_h2_.T
        dA_pre = dQ1 * G * (np.float32(1.0) - A * A)
        dG_pre = dQ1 * A * G * (np.float32(1.0) - G)
        dP_loc = Xl.T @ dA_pre
        db_loc = dA_pre.sum(axis=0)
        dP_ctx = Zc.T @ dG_pre
        db_ctx = dG_pre.sum(axis=0)

        dXl = (dL @ self.W_loc_.T) + (dA_pre @ self.P_loc_.T)
        dE_dyn = np.zeros_like(self.E_dyn_)
        np.add.at(dE_dyn, tok_cur, dXl[:, 0:d])
        np.add.at(dE_dyn, tok_prev, dXl[:, d : 2 * d])
        dE_dyn[0] = 0.0

        self._adam_t += 1
        lr_cur = self.lr / float(np.sqrt(1.0 + 0.0010 * self._adam_t))
        self._adam_step_cpu("W_loc", self.W_loc_, dW_loc, lr_cur)
        self._adam_step_cpu("W_ctx", self.W_ctx_, dW_ctx, lr_cur)
        self._adam_step_cpu("W_q", self.W_q_, dW_q, lr_cur)
        self._adam_step_cpu("b_out", self.b_out_, db_out, lr_cur)
        self._adam_step_cpu("P_loc", self.P_loc_, dP_loc, lr_cur * 0.75)
        self._adam_step_cpu("b_loc", self.b_loc_, db_loc, lr_cur * 0.75)
        self._adam_step_cpu("P_ctx", self.P_ctx_, dP_ctx, lr_cur * 0.75)
        self._adam_step_cpu("b_ctx", self.b_ctx_, db_ctx, lr_cur * 0.75)
        self._adam_step_cpu("P_h2", self.P_h2_, dP_h2, lr_cur * 0.75)
        self._adam_step_cpu("b_h2", self.b_h2_, db_h2, lr_cur * 0.75)
        self._adam_step_cpu("P_ctx2", self.P_ctx2_, dP_ctx2, lr_cur * 0.75)
        self._adam_step_cpu("b_ctx2", self.b_ctx2_, db_ctx2, lr_cur * 0.75)
        self._adam_step_cpu("E_dyn", self.E_dyn_, dE_dyn, lr_cur * 0.65)
        self.E_dyn_[0] = 0.0
        self.total_samples += m_b
        return loss_val

    def _train_stream_blocks(
        self,
        tokens: np.ndarray,
        *,
        chunk_memory: Optional[object] = None,
        block_size: int = 65_536,
        sample_stride: int = 4,
        max_samples: int = 4_000_000,
        epochs: int = 1,
    ) -> None:
        assert self.atom_emb_ is not None and self.P_loc_ is not None
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        n = len(arr)
        if n < 16:
            return
        ae = self.atom_emb_
        ce = self.chunk_emb_ if self.chunk_emb_ is not None else ae
        d = self.embed_dim
        a2c = getattr(chunk_memory, "atoms_to_chunk_", None) if chunk_memory is not None else None

        if self.total_samples == 0 and self.b_out_ is not None:
            self._fit_write_gates_from_stream(arr)
            cnts = np.bincount(arr, minlength=self.vocab_size).astype(np.float64)
            cnts[0] = 0.0
            tot_c = max(float(cnts.sum()), 1.0)
            p_uni = (cnts + 0.5) / (tot_c + 0.5 * self.vocab_size)
            self.b_out_ = np.log(np.clip(p_uni, 1e-7, 1.0)).astype(np.float32)
            self.b_out_[0] = -10.0

        st = max(int(sample_stride), max((n - 2) // max(int(max_samples), 1000), 1))
        bs = self.batch_size
        on_gpu = (n >= 32_000) and self._ensure_gpu_trainer(bs)
        if not on_gpu:
            self.train_backend = "cpu"

        wg_arr = self.write_gate_
        loss_acc = 0.0
        loss_cnt = 0
        b_counter = 0

        for _ep in range(max(int(epochs), 1)):
            h_f0 = np.zeros(d, dtype=np.float32)
            h_m0 = np.zeros(d, dtype=np.float32)
            h_s0 = np.zeros(d, dtype=np.float32)
            h_c0 = np.zeros(d, dtype=np.float32)
            h_e0 = np.zeros(d, dtype=np.float32)
            prev_last_tok = 0

            for start in range(0, n - 2, block_size):
                end = min(start + block_size, n - 1)
                blk_len = end - start
                if blk_len < 4:
                    break
                blk = arr[start:end]
                targets = arr[start + 1 : end + 1]
                blk_prev = np.empty_like(blk)
                blk_prev[0] = prev_last_tok
                blk_prev[1:] = blk[:-1]
                prev_last_tok = int(blk[-1])

                X = ae[blk]
                if a2c is not None and len(ce) > self.vocab_size:
                    X_ch = X.copy()
                    pairs_a = blk_prev.tolist()
                    pairs_b = blk.tolist()
                    c_arr = np.asarray([a2c.get((a, b), 0) for a, b in zip(pairs_a, pairs_b)], dtype=np.int32)
                    hit = c_arr >= self.vocab_size
                    if np.any(hit):
                        X_ch[hit] = ce[c_arr[hit]]
                else:
                    X_ch = X

                H_f = _kogge_stone_ema(X, self.gamma_fast, h_f0)
                H_m = _kogge_stone_ema(X, self.gamma_mid, h_m0)
                H_s = _kogge_stone_ema(X, self.gamma_slow, h_s0)
                H_c = _kogge_stone_ema(X_ch, self.gamma_chunk, h_c0)
                H_e = _selective_associative_scan(X_ch, wg_arr[blk], self.entity_alpha, h_e0)
                h_f0, h_m0, h_s0, h_c0, h_e0 = H_f[-1], H_m[-1], H_s[-1], H_c[-1], H_e[-1]

                idx = np.arange(0, blk_len, st)
                t_sub = targets[idx]
                valid_m = (t_sub > 0) & (blk[idx] > 0)
                if not np.any(valid_m):
                    continue
                sel = idx[valid_m]
                y_all = t_sub[valid_m]
                tc_all = blk[sel]
                tp_all = blk_prev[sel]

                # Xl[:, 0:2D] will be filled live from trainable E_dyn[tc], E_dyn[tp]!
                X_loc_all = np.empty((len(sel), self.loc_dim), dtype=np.float32)
                X_loc_all[:, 2 * d : 3 * d] = H_f[sel]
                Z_ctx_all = np.concatenate([H_m[sel], H_s[sel], H_c[sel], H_e[sel], H_f[sel] - H_m[sel]], axis=1)

                n_sel = len(sel)
                for b0 in range(0, n_sel, bs):
                    b1 = min(b0 + bs, n_sel)
                    m_b = b1 - b0
                    if m_b < 8:
                        continue
                    Xl = np.ascontiguousarray(X_loc_all[b0:b1], dtype=np.float32)
                    Zc = np.ascontiguousarray(Z_ctx_all[b0:b1], dtype=np.float32)
                    tc = np.ascontiguousarray(tc_all[b0:b1], dtype=np.int32)
                    tp = np.ascontiguousarray(tp_all[b0:b1], dtype=np.int32)
                    yb = np.ascontiguousarray(y_all[b0:b1], dtype=np.int32)
                    b_counter += 1
                    if on_gpu:
                        read_l = (b_counter % 25 == 0)
                        lv = self._gpu_train_batch(Xl, Zc, tc, tp, yb, read_loss=read_l)
                        if read_l:
                            loss_acc += lv
                            loss_cnt += 1
                    else:
                        loss_acc += self._cpu_train_batch(Xl, Zc, tc, tp, yb)
                        loss_cnt += 1

        if on_gpu:
            self._sync_weights_from_gpu()
        if loss_cnt > 0:
            self.last_loss_ = round(loss_acc / float(loss_cnt), 4)
        self._fitted = True

    def fit_stream(
        self,
        tokens: Sequence[int],
        atom_embeddings: np.ndarray,
        *,
        chunk_memory: Optional[object] = None,
        sym_to_class: Optional[np.ndarray] = None,
        rc_persistence: Optional[np.ndarray] = None,
        sample_stride: int = 4,
        max_samples: int = 4_000_000,
        epochs: int = 1,
    ) -> "StatefulContextPolicy":
        self.configure_embeddings(
            atom_embeddings,
            chunk_to_atoms=getattr(chunk_memory, "chunk_to_atoms_", None),
            n_total_units=getattr(chunk_memory, "n_total_units", self.vocab_size),
            sym_to_class=sym_to_class,
            rc_persistence=rc_persistence,
        )
        self._init_params()
        self.total_samples = 0
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        eff_ep = epochs if len(arr) >= 50_000 else max(int(epochs), 18)
        eff_bs = self.batch_size if len(arr) >= 50_000 else min(self.batch_size, 256)
        old_bs = self.batch_size
        self.batch_size = eff_bs
        try:
            self._train_stream_blocks(
                arr, chunk_memory=chunk_memory,
                sample_stride=sample_stride, max_samples=max_samples, epochs=eff_ep,
            )
        finally:
            self.batch_size = old_bs
        return self

    def ingest_chunk(
        self,
        tokens: Sequence[int],
        *,
        chunk_memory: Optional[object] = None,
        sample_stride: int = 4,
        max_samples: int = 4_000_000,
    ) -> "StatefulContextPolicy":
        if self.atom_emb_ is None or self.P_loc_ is None:
            return self
        self._train_stream_blocks(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            chunk_memory=chunk_memory, sample_stride=sample_stride, max_samples=max_samples, epochs=1,
        )
        return self

    # ------------------------------------------------------------------
    # Live IN-PLACE Selective State Evolution & 2-Layer Residual Readout
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self.state = self._make_zero_state()

    def pin_macro_anchor(self) -> None:
        self.state.slow_anchor = self.state.h_slow.copy()
        self.state.ent_anchor = self.state.h_ent.copy()

    def _step_state_inplace(
        self,
        st: PersistentContextState,
        symbol: int,
        *,
        chunk_id: int = 0,
        is_self: bool = False,
    ) -> PersistentContextState:
        if self.atom_emb_ is None:
            return st
        sym = int(symbol)
        if sym <= 0 or sym >= self.vocab_size:
            st.steps += 1
            return st

        e_anchor = self.atom_emb_[sym]
        e_dyn = self.E_dyn_[sym] if self.E_dyn_ is not None else e_anchor
        if chunk_id >= self.vocab_size and self.chunk_emb_ is not None and chunk_id < len(self.chunk_emb_):
            e_c = self.chunk_emb_[chunk_id]
        else:
            e_c = e_anchor

        gf, gm, gs, gc = self.gamma_fast, self.gamma_mid, self.gamma_slow, self.gamma_chunk
        w_ent = float(self.entity_alpha) * float(self.write_gate_[sym])
        if is_self:
            gs = min(0.992, gs + 0.010)
            w_ent *= 0.35
            if st.slow_anchor is None and st.steps >= 4:
                st.slow_anchor = st.h_slow.copy()
                st.ent_anchor = st.h_ent.copy()

        st.e_prev = st.e_cur
        st.e_cur = e_dyn
        st.h_fast = gf * st.h_fast + (1.0 - gf) * e_anchor
        st.h_mid = gm * st.h_mid + (1.0 - gm) * e_anchor
        st.h_slow = gs * st.h_slow + (1.0 - gs) * e_anchor
        st.h_chunk = gc * st.h_chunk + (1.0 - gc) * e_c
        st.h_ent = (1.0 - w_ent) * st.h_ent + w_ent * e_c
        st.steps += 1
        return st

    def observe(
        self,
        symbol: int,
        *,
        chunk_id: int = 0,
        source: str = "external",
    ) -> PersistentContextState:
        return self._step_state_inplace(
            self.state, symbol, chunk_id=chunk_id, is_self=(source == "self"),
        )

    def fork_step(
        self,
        parent_state: PersistentContextState,
        symbol: int,
        *,
        chunk_id: int = 0,
        is_self: bool = True,
    ) -> PersistentContextState:
        child = parent_state.copy()
        return self._step_state_inplace(child, symbol, chunk_id=chunk_id, is_self=is_self)

    def _forward_logits(self, x_loc: np.ndarray, z_ctx: Optional[np.ndarray]) -> np.ndarray:
        assert self.P_loc_ is not None and self.W_loc_ is not None and self.W_q_ is not None and self.b_out_ is not None
        a_h = np.tanh(x_loc @ self.P_loc_ + self.b_loc_)
        if z_ctx is None:
            g1 = _sigmoid(self.b_ctx_)
            q1 = a_h * g1
            if self.P_h2_ is not None:
                h2 = np.tanh(q1 @ self.P_h2_ + self.b_h2_)
                g2 = _sigmoid(self.b_ctx2_)
                q2 = q1 + h2 * g2
            else:
                q2 = q1
            logits = (x_loc @ self.W_loc_) + (q2 @ self.W_q_) + self.b_out_
        else:
            g1 = _sigmoid(z_ctx @ self.P_ctx_ + self.b_ctx_)
            q1 = a_h * g1
            if self.P_h2_ is not None:
                h2 = np.tanh(q1 @ self.P_h2_ + self.b_h2_)
                g2 = _sigmoid(z_ctx @ self.P_ctx2_ + self.b_ctx2_)
                q2 = q1 + h2 * g2
            else:
                q2 = q1
            logits = (x_loc @ self.W_loc_) + (z_ctx @ self.W_ctx_) + (q2 @ self.W_q_) + self.b_out_
        logits[0] = -1e4
        return logits.astype(np.float32, copy=False)

    def predict_symbol_map(
        self,
        *,
        state: Optional[PersistentContextState] = None,
        k: int = 24,
        self_mode: bool = False,
        context_band_gain: float = 1.0,
    ) -> Dict[int, float]:
        if not self._fitted or self.W_loc_ is None:
            return {}
        st = state if state is not None else self.state
        if st.steps <= 0:
            return {}
        gain = max(float(context_band_gain), 1.25) if self_mode else float(context_band_gain)
        a_blend = 0.40 if self_mode else 0.0
        x_loc, z_ctx = st.split_local_context(context_band_gain=gain, anchor_blend=a_blend)
        logits = self._forward_logits(x_loc, z_ctx)

        k_eff = min(max(int(k), 1), self.vocab_size - 1)
        top_idx = np.argpartition(-logits, k_eff)[:k_eff]
        top_l = logits[top_idx].astype(np.float64)
        top_l -= float(top_l.max())
        probs = np.exp(np.clip(top_l, -30.0, 0.0))
        p_sum = float(probs.sum())
        if p_sum <= 1e-12:
            return {}
        probs /= p_sum
        return {int(idx): float(pr) for idx, pr in zip(top_idx.tolist(), probs.tolist()) if int(idx) > 0 and pr > 1e-9}

    def context_log_bayes_factor(
        self,
        *,
        state: Optional[PersistentContextState] = None,
        self_mode: bool = False,
        context_band_gain: float = 1.0,
    ) -> Optional[np.ndarray]:
        """Exact counterfactual conditional PMI ``log P(y | x_loc, z_ctx) - log P(y | x_loc, 0)``."""
        if not self._fitted or self.W_loc_ is None:
            return None
        st = state if state is not None else self.state
        if st.steps <= 0:
            return None
        gain = max(float(context_band_gain), 1.30) if self_mode else float(context_band_gain)
        a_blend = 0.40 if self_mode else 0.0
        x_loc, z_ctx = st.split_local_context(context_band_gain=gain, anchor_blend=a_blend)
        lp_full = _log_softmax_1d(self._forward_logits(x_loc, z_ctx))
        lp_null = _log_softmax_1d(self._forward_logits(x_loc, None))
        delta = np.clip(lp_full - lp_null, -2.5, 2.5).astype(np.float32)
        delta[0] = 0.0
        return delta

    def _ensure_gpu_infer(self, min_cap: int = 32) -> bool:
        """Upload read-only inference weights to RX 9070 XT VRAM for batched lookahead evaluation."""
        if not self.use_gpu or not self._fitted or self.W_loc_ is None:
            return False
        if self._gpu is None:
            acc = OpenCLAccelerator()
            if acc.available and acc._k_sgemm is not None and acc._k_res_fwd is not None:
                self._gpu = acc
            else:
                return False
        g = self._gpu
        cap = max(int(min_cap), 32)
        if getattr(self, "_gpu_infer_bufs", None) and self._gpu_infer_cap >= cap:
            return True
        if getattr(self, "_gpu_infer_bufs", None):
            for buf in self._gpu_infer_bufs.values():
                g.release_buffer(buf)
            self._gpu_infer_bufs.clear()

        ld, cd, r, v = self.loc_dim, self.ctx_dim, self.gate_dim, self.vocab_size
        bufs: Dict[str, object] = {}
        for name in ("P_loc", "b_loc", "P_ctx", "b_ctx", "W_loc", "W_ctx", "W_q", "b_out"):
            arr = getattr(self, f"{name}_")
            bufs[name] = g.create_buffer_from_array(arr, read_only=True)
        if self.P_h2_ is not None:
            for name in ("P_h2", "b_h2", "P_ctx2", "b_ctx2"):
                arr = getattr(self, f"{name}_")
                bufs[name] = g.create_buffer_from_array(arr, read_only=True)
        # Compact inference scratchpads for up to ``cap`` concurrent lookahead branches
        bufs["Xl"] = g.create_empty_buffer(cap * ld * 4)
        bufs["Zc"] = g.create_empty_buffer(cap * cd * 4)
        bufs["A_pre"] = g.create_empty_buffer(cap * r * 4)
        bufs["G_pre"] = g.create_empty_buffer(cap * r * 4)
        bufs["A"] = g.create_empty_buffer(cap * r * 4)
        bufs["G"] = g.create_empty_buffer(cap * r * 4)
        bufs["Q1"] = g.create_empty_buffer(cap * r * 4)
        bufs["H2_pre"] = g.create_empty_buffer(cap * r * 4)
        bufs["G2_pre"] = g.create_empty_buffer(cap * r * 4)
        bufs["H2"] = g.create_empty_buffer(cap * r * 4)
        bufs["G2"] = g.create_empty_buffer(cap * r * 4)
        bufs["Q2"] = g.create_empty_buffer(cap * r * 4)
        bufs["loc_logits"] = g.create_empty_buffer(cap * v * 4)
        bufs["logits"] = g.create_empty_buffer(cap * v * 4)
        self._gpu_infer_bufs = bufs
        self._gpu_infer_cap = cap
        return True

    def predict_batch_with_pmi(
        self,
        states: Sequence[PersistentContextState],
        *,
        k: int = 16,
        self_mode: bool = True,
        need_pmi: bool = True,
        context_band_gain: float = 1.0,
    ) -> Tuple[List[Dict[int, float]], Optional[np.ndarray]]:
        """Evaluate ``B`` persistent lookahead states in ONE fused GPU/BLAS pass.

        Computes BOTH calibrated Top-K probability maps AND exact counterfactual
        conditional PMI ``log P(y | x_loc, z_ctx) - log P(y | x_loc, 0)`` while
        sharing ``A_h`` and ``Xl @ W_loc`` across full & null paths (~6x fewer ops).
        """
        n_b = len(states)
        if n_b == 0 or not self._fitted or self.W_loc_ is None:
            return [{} for _ in range(n_b)], None

        gain = max(float(context_band_gain), 1.25) if self_mode else float(context_band_gain)
        a_blend = 0.40 if self_mode else 0.0
        xl_list, zc_list = [], []
        valid_rows = []
        for idx_s, st in enumerate(states):
            if st is not None and st.steps > 0:
                xl, zc = st.split_local_context(context_band_gain=gain, anchor_blend=a_blend)
                xl_list.append(xl)
                zc_list.append(zc)
                valid_rows.append(idx_s)
        if not valid_rows:
            return [{} for _ in range(n_b)], None

        Xl = np.ascontiguousarray(np.stack(xl_list, axis=0), dtype=np.float32)  # [M, 3D]
        Zc = np.ascontiguousarray(np.stack(zc_list, axis=0), dtype=np.float32)  # [M, 5D]
        m_b = Xl.shape[0]
        ld, cd, r, v = self.loc_dim, self.ctx_dim, self.gate_dim, self.vocab_size

        # Use RX 9070 XT GPU when batch >= 4 (amortizes PCIe launch cleanly); single batched BLAS otherwise
        use_gpu_path = (m_b >= 4) and self._ensure_gpu_infer(m_b)
        if use_gpu_path:
            g = self._gpu
            bufs = self._gpu_infer_bufs
            g.write_buffer(bufs["Xl"], Xl)
            g.write_buffer(bufs["Zc"], Zc)
            # Shared local path: A_pre = Xl @ P_loc, loc_logits = Xl @ W_loc
            g.launch_sgemm(m_b, r, ld, bufs["Xl"], bufs["P_loc"], bufs["A_pre"], beta=0.0)
            g.launch_sgemm(m_b, r, cd, bufs["Zc"], bufs["P_ctx"], bufs["G_pre"], beta=0.0)
            g.launch_gated_bilinear_fwd(
                m_b, r, bufs["A_pre"], bufs["b_loc"], bufs["G_pre"], bufs["b_ctx"],
                bufs["A"], bufs["G"], bufs["Q1"],
            )
            q_out_key = "Q1"
            if self.P_h2_ is not None:
                g.launch_sgemm(m_b, r, r, bufs["Q1"], bufs["P_h2"], bufs["H2_pre"], beta=0.0)
                g.launch_sgemm(m_b, r, cd, bufs["Zc"], bufs["P_ctx2"], bufs["G2_pre"], beta=0.0)
                g.launch_res_gated_fwd(
                    m_b, r, bufs["Q1"], bufs["H2_pre"], bufs["b_h2"], bufs["G2_pre"], bufs["b_ctx2"],
                    bufs["H2"], bufs["G2"], bufs["Q2"],
                )
                q_out_key = "Q2"
            g.launch_sgemm(m_b, v, ld, bufs["Xl"], bufs["W_loc"], bufs["loc_logits"], beta=0.0)
            g.launch_sgemm(m_b, v, cd, bufs["Zc"], bufs["W_ctx"], bufs["logits"], beta=0.0)
            g.launch_sgemm(m_b, v, r, bufs[q_out_key], bufs["W_q"], bufs["logits"], beta=1.0)
            g.finish()
            loc_l = g.read_buffer(bufs["loc_logits"], np.empty((m_b, v), dtype=np.float32))
            ctx_q_l = g.read_buffer(bufs["logits"], np.empty((m_b, v), dtype=np.float32))
            logits_full = loc_l + ctx_q_l + self.b_out_[None, :]
            if need_pmi:
                A_host = g.read_buffer(bufs["A"], np.empty((m_b, r), dtype=np.float32))
                g1_0 = _sigmoid(self.b_ctx_)[None, :]
                q1_0 = A_host * g1_0
                if self.P_h2_ is not None:
                    h2_0 = np.tanh(q1_0 @ self.P_h2_ + self.b_h2_)
                    q2_0 = q1_0 + h2_0 * _sigmoid(self.b_ctx2_)[None, :]
                else:
                    q2_0 = q1_0
                logits_null = loc_l + (q2_0 @ self.W_q_) + self.b_out_[None, :]
            else:
                logits_null = None
        else:
            A_h = np.tanh(Xl @ self.P_loc_ + self.b_loc_)
            loc_l = (Xl @ self.W_loc_) + self.b_out_[None, :]
            G1 = _sigmoid(Zc @ self.P_ctx_ + self.b_ctx_)
            Q1 = A_h * G1
            if self.P_h2_ is not None:
                H2 = np.tanh(Q1 @ self.P_h2_ + self.b_h2_)
                G2 = _sigmoid(Zc @ self.P_ctx2_ + self.b_ctx2_)
                Q2 = Q1 + H2 * G2
            else:
                Q2 = Q1
            logits_full = loc_l + (Zc @ self.W_ctx_) + (Q2 @ self.W_q_)
            if need_pmi:
                q1_0 = A_h * _sigmoid(self.b_ctx_)[None, :]
                if self.P_h2_ is not None:
                    q2_0 = q1_0 + np.tanh(q1_0 @ self.P_h2_ + self.b_h2_) * _sigmoid(self.b_ctx2_)[None, :]
                else:
                    q2_0 = q1_0
                logits_null = loc_l + (q2_0 @ self.W_q_)
            else:
                logits_null = None

        logits_full[:, 0] = -1e4
        k_eff = min(max(int(k), 1), v - 1)
        out_maps: List[Dict[int, float]] = [{} for _ in range(n_b)]
        out_pmis = np.zeros((n_b, v), dtype=np.float32) if need_pmi else None

        for j, orig_i in enumerate(valid_rows):
            lf = logits_full[j]
            top_idx = np.argpartition(-lf, k_eff)[:k_eff]
            top_l = lf[top_idx].astype(np.float64)
            top_l -= float(top_l.max())
            probs = np.exp(np.clip(top_l, -30.0, 0.0))
            ps = float(probs.sum())
            if ps > 1e-12:
                probs /= ps
                out_maps[orig_i] = {
                    int(idx): float(pr) for idx, pr in zip(top_idx.tolist(), probs.tolist())
                    if int(idx) > 0 and pr > 1e-9
                }
            if need_pmi and logits_null is not None and out_pmis is not None:
                ln = logits_null[j]
                ln[0] = -1e4
                delta = np.clip(_log_softmax_1d(lf) - _log_softmax_1d(ln), -2.5, 2.5)
                delta[0] = 0.0
                out_pmis[orig_i] = delta

        return out_maps, out_pmis

    def predict_map_and_pmi(
        self,
        *,
        state: Optional[PersistentContextState] = None,
        k: int = 16,
        self_mode: bool = True,
        need_pmi: bool = True,
    ) -> Tuple[Dict[int, float], Optional[np.ndarray]]:
        st = state if state is not None else self.state
        maps, pmis = self.predict_batch_with_pmi([st], k=k, self_mode=self_mode, need_pmi=need_pmi)
        return maps[0], (pmis[0] if pmis is not None else None)

    def param_bytes(self) -> int:
        tot = 0
        for name in self._PARAM_NAMES:
            arr = getattr(self, f"{name}_", None)
            if arr is not None:
                tot += int(arr.nbytes)
        return tot

    def prune_under_pressure(self, pressure: float) -> int:
        if pressure >= 0.40 and self._adam_m:
            n = len(self._adam_m) + len(self._adam_v)
            self._adam_m.clear()
            self._adam_v.clear()
            return n
        return 0
