"""End-to-End RX 9070 XT GPU-Trained Learnable Gated RC State-Space Policy (1- & 2-Layer BPTT).

Upgrades fixed-decay multi-scale EMA states into a **fully learnable, input- and
state-dependent 1- or 2-Layer Residual Gated-RC State-Space Recurrent Network**
trained end-to-end with exact Backpropagation Through Time (BPTT) natively on
OpenCL GPUs (AMD RX 9070 XT) with VRAM-resident zero-copy token streaming:

  - **Layer 1 (Fast/Mid Syntactic & Phrasal RC Dynamics)** ``s1_t in R^H``:
      ``z1_t = sigmoid(x_t @ W_x1 + s1_{t-1} @ U_s1 + b_x1 + b_s1)``
      ``r1_t = sigmoid(...)``
      ``h1~_t = tanh(x_t @ W_xh1 + r1_t * (s1_{t-1} @ U_sh1 + b_sh1) + b_xh1)``
      ``s1_t = (1 - z1_t) * s1_{t-1} + z1_t * h1~_t``       (strictly IN-PLACE!)

  - **Layer 2 (Optional Slow Macro-Context / Discourse RC Dynamics, ``num_layers=2``)** ``s2_t in R^H``:
      Takes ``s1_t`` as input with slow-capacitor write-gate initialization
      (``b_s2[:H] in [-4.0, -0.5]``), producing a **residual skip readout**:
      ``logits_t = (s1_t + s2_t) @ W_out + b_out``

  - **Self-Learned High-tau Slow-Channel Selective Anchor Gating (Task-Agnostic)**:
      During external prompt observation, tracks each top-layer channel's own
      learned write gate ``z_ema``. At ``pin_macro_anchor()``, anchors ONLY the
      high-tau slow-capacitor channels (small ``z_ema``) while leaving fast
      syntactic channels (large ``z_ema``) 100% unconstrained during ``self_mode``.

Strictly task-agnostic: operates solely on discrete symbol ID streams with zero
language rules, zero word segmenters, and zero repetition bans.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencl_gpu import OpenCLAccelerator


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -16.0, 16.0)))).astype(np.float32, copy=False)


def _log_softmax_1d(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    lse = m + float(np.log(np.sum(np.exp(np.clip(logits - m, -40.0, 0.0))) + 1e-12))
    return (logits - lse).astype(np.float32, copy=False)


@dataclass
class LearnableSSMState:
    """Live persistent 1- or 2-layer gated RC node state modified strictly in-place."""

    s: np.ndarray
    s2: Optional[np.ndarray] = None
    steps: int = 0
    s_anchor: Optional[np.ndarray] = None
    slow_anchor_w: Optional[np.ndarray] = None
    z_ema_top: Optional[np.ndarray] = None

    def copy(self) -> "LearnableSSMState":
        return LearnableSSMState(
            s=self.s.copy(),
            s2=self.s2.copy() if self.s2 is not None else None,
            steps=self.steps,
            s_anchor=self.s_anchor.copy() if self.s_anchor is not None else None,
            slow_anchor_w=self.slow_anchor_w.copy() if self.slow_anchor_w is not None else None,
            z_ema_top=self.z_ema_top.copy() if self.z_ema_top is not None else None,
        )


class LearnableStateSpacePolicy:
    """End-to-end GPU-trained 1- or 2-layer residual gated-RC state-space policy."""

    _PARAM_NAMES_1L: Tuple[str, ...] = ("E", "W_x", "b_x", "U_s", "b_s", "W_out", "b_out")
    _PARAM_NAMES_2L: Tuple[str, ...] = (
        "E", "W_x", "b_x", "U_s", "b_s",
        "W_x2", "b_x2", "U_s2", "b_s2",
        "W_out", "b_out",
    )

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 512,
        *,
        num_layers: int = 1,
        bptt_len: int = 48,
        batch_streams: int = 256,
        lr: float = 0.004,
        min_lr_ratio: float = 0.15,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.5,
        slow_anchor_strength: float = 0.42,
        use_gpu: bool = True,
        seed: int = 42,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.embed_dim = max(int(embed_dim), 8)
        self.hidden_dim = max(int(hidden_dim), 32)
        self.num_layers = 2 if int(num_layers) >= 2 else 1
        self.bptt_len = max(int(bptt_len), 8)
        self.batch_streams = max(int(batch_streams), 8)
        self.lr = float(lr)
        self.min_lr_ratio = float(np.clip(min_lr_ratio, 0.02, 1.0))
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.slow_anchor_strength = float(np.clip(slow_anchor_strength, 0.0, 0.85))
        self.use_gpu = bool(use_gpu)
        self.seed = int(seed)

        # Layer 1 & shared parameters
        self.E_: Optional[np.ndarray] = None       # [V, D]
        self.W_x_: Optional[np.ndarray] = None     # [D, 3H]
        self.b_x_: Optional[np.ndarray] = None     # [3H]
        self.U_s_: Optional[np.ndarray] = None     # [H, 3H]
        self.b_s_: Optional[np.ndarray] = None     # [3H]
        # Layer 2 parameters (used when num_layers == 2)
        self.W_x2_: Optional[np.ndarray] = None    # [H, 3H]
        self.b_x2_: Optional[np.ndarray] = None    # [3H]
        self.U_s2_: Optional[np.ndarray] = None    # [H, 3H]
        self.b_s2_: Optional[np.ndarray] = None    # [3H]
        # Shared residual readout
        self.W_out_: Optional[np.ndarray] = None   # [H, V]
        self.b_out_: Optional[np.ndarray] = None   # [V]

        # Transient runtime & optimizer state (stripped on pickle)
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_t: int = 0
        self._gpu: Optional[OpenCLAccelerator] = None
        self._gpu_bufs: Dict[str, object] = {}
        self._gpu_bt: Tuple[int, int] = (0, 0)
        self.train_backend: str = "cpu"
        self.vram_allocated_mb: float = 0.0
        self.total_tokens_trained: int = 0
        self.last_loss_: float = 0.0
        self._fitted: bool = False

        # Live persistent node state (modified strictly in-place!)
        self.state: LearnableSSMState = self._make_zero_state()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return self._PARAM_NAMES_2L if self.num_layers == 2 else self._PARAM_NAMES_1L

    def __getstate__(self) -> dict:
        st = self.__dict__.copy()
        st["_adam_m"] = {}
        st["_adam_v"] = {}
        st["_gpu"] = None
        st["_gpu_bufs"] = {}
        st["_gpu_bt"] = (0, 0)
        return st

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "num_layers"):
            self.num_layers = 1
        if not hasattr(self, "min_lr_ratio"):
            self.min_lr_ratio = 0.15
        if not hasattr(self, "slow_anchor_strength"):
            self.slow_anchor_strength = 0.42
        if not hasattr(self, "vram_allocated_mb"):
            self.vram_allocated_mb = 0.0
        self._adam_m = {}
        self._adam_v = {}
        self._gpu = None
        self._gpu_bufs = {}
        self._gpu_bt = (0, 0)

    def close_gpu(self) -> None:
        if self._gpu is not None and self._gpu_bufs:
            for buf in self._gpu_bufs.values():
                self._gpu.release_buffer(buf)
        self._gpu_bufs.clear()
        self._gpu_bt = (0, 0)
        self.vram_allocated_mb = 0.0

    def _make_zero_state(self) -> LearnableSSMState:
        return LearnableSSMState(
            s=np.zeros(self.hidden_dim, dtype=np.float32),
            s2=(np.zeros(self.hidden_dim, dtype=np.float32) if self.num_layers == 2 else None),
            steps=0,
            s_anchor=None,
            slow_anchor_w=None,
            z_ema_top=None,
        )

    def init_params(self, init_embeddings: Optional[np.ndarray] = None) -> None:
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        v, d, h = self.vocab_size, self.embed_dim, self.hidden_dim
        if init_embeddings is not None and init_embeddings.shape[0] >= v and init_embeddings.shape[1] == d:
            ae = init_embeddings[:v].astype(np.float32).copy()
            valid = np.linalg.norm(ae[1:], axis=1) > 1e-8
            if np.any(valid):
                ae -= ae[1:][valid].mean(axis=0, keepdims=True)
            ae /= np.maximum(np.linalg.norm(ae, axis=1, keepdims=True), 1e-8)
            ae *= 0.65
            ae[0] = 0.0
            self.E_ = ae
        else:
            self.E_ = (rng.randn(v, d) * np.sqrt(1.0 / max(d, 1))).astype(np.float32)
            self.E_[0] = 0.0

        # Layer 1: fast/mid multi-timescale write-gate initialization
        self.W_x_ = (rng.randn(d, 3 * h) * np.sqrt(1.0 / max(d, 1))).astype(np.float32)
        self.b_x_ = np.zeros(3 * h, dtype=np.float32)
        self.U_s_ = (rng.randn(h, 3 * h) * np.sqrt(0.5 / max(h, 1))).astype(np.float32)
        self.b_s_ = np.zeros(3 * h, dtype=np.float32)
        if self.num_layers == 2:
            self.b_s_[:h] = np.linspace(-2.0, 0.8, h, dtype=np.float32)
            # Layer 2: slow macro-context capacitor write-gate initialization!
            self.W_x2_ = (rng.randn(h, 3 * h) * np.sqrt(0.5 / max(h, 1))).astype(np.float32)
            self.b_x2_ = np.zeros(3 * h, dtype=np.float32)
            self.U_s2_ = (rng.randn(h, 3 * h) * np.sqrt(0.5 / max(h, 1))).astype(np.float32)
            self.b_s2_ = np.zeros(3 * h, dtype=np.float32)
            self.b_s2_[:h] = np.linspace(-4.0, -0.5, h, dtype=np.float32)
        else:
            self.b_s_[:h] = np.linspace(-2.5, 0.5, h, dtype=np.float32)
            self.W_x2_ = self.b_x2_ = self.U_s2_ = self.b_s2_ = None

        out_scale = np.sqrt(0.35 / max(h, 1)) if self.num_layers == 2 else np.sqrt(0.5 / max(h, 1))
        self.W_out_ = (rng.randn(h, v) * out_scale).astype(np.float32)
        self.b_out_ = np.zeros(v, dtype=np.float32)
        self._adam_m.clear()
        self._adam_v.clear()
        self._adam_t = 0
        self.close_gpu()

    def param_bytes(self) -> int:
        tot = 0
        for name in self._param_names:
            arr = getattr(self, f"{name}_", None)
            if arr is not None:
                tot += int(arr.nbytes)
        return tot

    def _lr_for_step(self, step_t: int, total_steps: int = 0) -> float:
        if total_steps > 0:
            # Smooth warmup (first 2% or 40 steps) + cosine annealing to min_lr_ratio * lr
            warm = min(max(total_steps // 50, 20), 80)
            if step_t <= warm:
                return float(self.lr * (float(step_t) / float(max(warm, 1))))
            prog = min(float(step_t - warm) / float(max(total_steps - warm, 1)), 1.0)
            cos_f = 0.5 * (1.0 + math.cos(math.pi * prog))
            return float(self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos_f))
        return float(self.lr / math.sqrt(1.0 + 0.0005 * step_t))

    # ------------------------------------------------------------------
    # Native RX 9070 XT OpenCL 1- & 2-Layer BPTT Engine (VRAM-Resident!)
    # ------------------------------------------------------------------

    def _ensure_gpu_trainer(self, B: int, T: int) -> bool:
        if not self.use_gpu:
            return False
        if self._gpu is None:
            acc = OpenCLAccelerator()
            if (
                acc.available
                and acc._k_sgemm is not None
                and acc._k_gru_fwd is not None
                and acc._k_gru_bwd is not None
                and acc._k_emb_gather_1 is not None
                and acc._k_zero_buf is not None
                and acc._k_vec_add is not None
            ):
                self._gpu = acc
            else:
                return False
        g = self._gpu
        if self._gpu_bufs and self._gpu_bt == (B, T):
            return True

        self.close_gpu()
        v, d, h = self.vocab_size, self.embed_dim, self.hidden_dim
        bt = B * T
        bufs: Dict[str, object] = {}
        vram_bytes = 0

        def _alloc_arr(key: str, arr: np.ndarray) -> None:
            nonlocal vram_bytes
            bufs[key] = g.create_buffer_from_array(arr)
            vram_bytes += int(arr.nbytes)

        def _alloc_empty(key: str, nbytes: int) -> None:
            nonlocal vram_bytes
            nb = max(int(nbytes), 4)
            bufs[key] = g.create_empty_buffer(nb)
            vram_bytes += nb

        for name in self._param_names:
            arr = getattr(self, f"{name}_")
            assert arr is not None
            _alloc_arr(f"p_{name}", arr)
            _alloc_empty(f"g_{name}", arr.nbytes)
            m0 = self._adam_m.get(name)
            v0 = self._adam_v.get(name)
            _alloc_arr(f"m_{name}", m0 if m0 is not None else np.zeros_like(arr))
            _alloc_arr(f"v_{name}", v0 if v0 is not None else np.zeros_like(arr))

        # Shared token & Layer-1 sequence buffers permanently in dedicated GPU VRAM
        _alloc_empty("tok_in", bt * 4)
        _alloc_empty("tok_tgt", bt * 4)
        _alloc_empty("X_all", bt * d * 4)
        _alloc_empty("dX_all", bt * d * 4)

        _alloc_empty("Xg_all", bt * 3 * h * 4)
        _alloc_empty("dXg_all", bt * 3 * h * 4)
        _alloc_empty("Sg_all", bt * 3 * h * 4)
        _alloc_empty("dSg_all", bt * 3 * h * 4)
        _alloc_empty("S_prev_all", bt * h * 4)
        _alloc_empty("S_all", bt * h * 4)
        _alloc_empty("cache_zrh", bt * 3 * h * 4)
        _alloc_arr("s_cur", np.zeros((B, h), dtype=np.float32))

        if self.num_layers == 2:
            _alloc_empty("Xg2_all", bt * 3 * h * 4)
            _alloc_empty("dXg2_all", bt * 3 * h * 4)
            _alloc_empty("Sg2_all", bt * 3 * h * 4)
            _alloc_empty("dSg2_all", bt * 3 * h * 4)
            _alloc_empty("S2_prev_all", bt * h * 4)
            _alloc_empty("S2_all", bt * h * 4)
            _alloc_empty("cache2_zrh", bt * 3 * h * 4)
            _alloc_arr("s2_cur", np.zeros((B, h), dtype=np.float32))
            _alloc_empty("S_out_all", bt * h * 4)
            _alloc_empty("dS1_from_l2", bt * h * 4)
            _alloc_empty("dS1_total", bt * h * 4)

        _alloc_empty("Sg_t", B * 3 * h * 4)
        _alloc_empty("dSg_t", B * 3 * h * 4)
        _alloc_empty("dS_out_all", bt * h * 4)
        _alloc_empty("dS_carry", B * h * 4)
        _alloc_empty("s_prev_t", B * h * 4)
        _alloc_empty("logits", bt * v * 4)
        _alloc_empty("losses", bt * 4)

        self._gpu_bufs = bufs
        self._gpu_bt = (B, T)
        self.vram_allocated_mb = round(vram_bytes / (1024.0 * 1024.0), 1)
        self.train_backend = f"opencl:{g.device_name}"
        return True

    def _sync_weights_from_gpu(self) -> None:
        if self._gpu is None or not self._gpu_bufs:
            return
        g = self._gpu
        g.finish()
        for name in self._param_names:
            arr = getattr(self, f"{name}_")
            g.read_buffer(self._gpu_bufs[f"p_{name}"], arr)

    def _gpu_bptt_step(
        self,
        tok_in: Optional[np.ndarray],      # [T, B] int32 OR None if already sliced on GPU
        tok_tgt: Optional[np.ndarray],     # [T, B] int32 OR None if already sliced on GPU
        *,
        B: int = 0,
        T: int = 0,
        reset_hidden: bool = False,
        read_loss: bool = False,
        total_steps: int = 0,
    ) -> float:
        g = self._gpu
        assert g is not None
        bufs = self._gpu_bufs
        if tok_in is not None:
            T, B = tok_in.shape
            g.write_buffer(bufs["tok_in"], np.ascontiguousarray(tok_in, dtype=np.int32))
            g.write_buffer(bufs["tok_tgt"], np.ascontiguousarray(tok_tgt, dtype=np.int32))
        bt = B * T
        v, d, h = self.vocab_size, self.embed_dim, self.hidden_dim

        if reset_hidden:
            g.launch_zero_buf(B * h, bufs["s_cur"])
            if self.num_layers == 2:
                g.launch_zero_buf(B * h, bufs["s2_cur"])

        # 1. Fused embedding gather & Layer-1 input projection for ALL T*B tokens in ONE GEMM
        g.launch_embed_gather_1(bt, d, bufs["tok_in"], bufs["p_E"], bufs["X_all"])
        g.launch_sgemm(bt, 3 * h, d, bufs["X_all"], bufs["p_W_x"], bufs["Xg_all"], beta=0.0)

        # 2. Layer-1 forward recurrent scan t = 0..T-1
        for t in range(T):
            g.launch_copy_slice(B * h, t, 1, bufs["S_prev_all"], bufs["s_cur"])
            g.launch_sgemm(B, 3 * h, h, bufs["s_cur"], bufs["p_U_s"], bufs["Sg_t"], beta=0.0)
            g.launch_copy_slice(B * 3 * h, t, 1, bufs["Sg_all"], bufs["Sg_t"])
            g.launch_gru_step_fwd(
                B, h, t,
                bufs["Xg_all"], bufs["Sg_t"], bufs["p_b_x"], bufs["p_b_s"],
                bufs["s_cur"], bufs["S_all"], bufs["cache_zrh"],
            )
            g.launch_copy_slice(B * h, t, 0, bufs["S_all"], bufs["s_cur"])

        # 2b. Optional Layer-2 forward recurrent scan + Residual Skip S_out = S1 + S2
        if self.num_layers == 2:
            g.launch_sgemm(bt, 3 * h, h, bufs["S_all"], bufs["p_W_x2"], bufs["Xg2_all"], beta=0.0)
            for t in range(T):
                g.launch_copy_slice(B * h, t, 1, bufs["S2_prev_all"], bufs["s2_cur"])
                g.launch_sgemm(B, 3 * h, h, bufs["s2_cur"], bufs["p_U_s2"], bufs["Sg_t"], beta=0.0)
                g.launch_copy_slice(B * 3 * h, t, 1, bufs["Sg2_all"], bufs["Sg_t"])
                g.launch_gru_step_fwd(
                    B, h, t,
                    bufs["Xg2_all"], bufs["Sg_t"], bufs["p_b_x2"], bufs["p_b_s2"],
                    bufs["s2_cur"], bufs["S2_all"], bufs["cache2_zrh"],
                )
                g.launch_copy_slice(B * h, t, 0, bufs["S2_all"], bufs["s2_cur"])
            g.launch_vec_add(bt * h, bufs["S_all"], bufs["S2_all"], bufs["S_out_all"])
            s_out_buf = bufs["S_out_all"]
        else:
            s_out_buf = bufs["S_all"]

        # 3. Fused output logits + Softmax/CE/dL + output grad for ALL T*B tokens
        g.launch_sgemm(bt, v, h, s_out_buf, bufs["p_W_out"], bufs["logits"], beta=0.0)
        g.launch_softmax_ce_grad(bt, v, bufs["p_b_out"], bufs["tok_tgt"], bufs["logits"], bufs["losses"])
        g.launch_sgemm(h, v, bt, s_out_buf, bufs["logits"], bufs["g_W_out"], transA=True, beta=0.0)
        g.launch_col_sum(bt, v, bufs["logits"], bufs["g_b_out"])
        g.launch_sgemm(bt, h, v, bufs["logits"], bufs["p_W_out"], bufs["dS_out_all"], transB=True, beta=0.0)

        # 4a. Optional Layer-2 exact BPTT t = T-1..0 + residual gradient accumulation into Layer 1
        if self.num_layers == 2:
            g.launch_zero_buf(B * h, bufs["dS_carry"])
            for t in range(T - 1, -1, -1):
                g.launch_copy_slice(B * h, t, 0, bufs["S2_prev_all"], bufs["s_prev_t"])
                g.launch_gru_step_bwd(
                    B, h, t,
                    bufs["dS_out_all"], bufs["dS_carry"], bufs["cache2_zrh"],
                    bufs["s_prev_t"], bufs["Sg2_all"], bufs["p_b_s2"],
                    bufs["dXg2_all"], bufs["dSg2_all"], bufs["dS_carry"],
                )
                g.launch_copy_slice(B * 3 * h, t, 0, bufs["dSg2_all"], bufs["dSg_t"])
                g.launch_sgemm(B, h, 3 * h, bufs["dSg_t"], bufs["p_U_s2"], bufs["dS_carry"], transB=True, beta=1.0)

            g.launch_sgemm(h, 3 * h, bt, bufs["S_all"], bufs["dXg2_all"], bufs["g_W_x2"], transA=True, beta=0.0)
            g.launch_col_sum(bt, 3 * h, bufs["dXg2_all"], bufs["g_b_x2"])
            g.launch_sgemm(h, 3 * h, bt, bufs["S2_prev_all"], bufs["dSg2_all"], bufs["g_U_s2"], transA=True, beta=0.0)
            g.launch_col_sum(bt, 3 * h, bufs["dSg2_all"], bufs["g_b_s2"])
            g.launch_sgemm(bt, h, 3 * h, bufs["dXg2_all"], bufs["p_W_x2"], bufs["dS1_from_l2"], transB=True, beta=0.0)
            # Residual skip gradient: dS1 = dS_out + dS1_from_l2
            g.launch_vec_add(bt * h, bufs["dS_out_all"], bufs["dS1_from_l2"], bufs["dS1_total"])
            dS1_buf = bufs["dS1_total"]
        else:
            dS1_buf = bufs["dS_out_all"]

        # 4b. Layer-1 exact BPTT t = T-1..0
        g.launch_zero_buf(B * h, bufs["dS_carry"])
        for t in range(T - 1, -1, -1):
            g.launch_copy_slice(B * h, t, 0, bufs["S_prev_all"], bufs["s_prev_t"])
            g.launch_gru_step_bwd(
                B, h, t,
                dS1_buf, bufs["dS_carry"], bufs["cache_zrh"],
                bufs["s_prev_t"], bufs["Sg_all"], bufs["p_b_s"],
                bufs["dXg_all"], bufs["dSg_all"], bufs["dS_carry"],
            )
            g.launch_copy_slice(B * 3 * h, t, 0, bufs["dSg_all"], bufs["dSg_t"])
            g.launch_sgemm(B, h, 3 * h, bufs["dSg_t"], bufs["p_U_s"], bufs["dS_carry"], transB=True, beta=1.0)

        # 5. Layer-1 weight & embedding gradients
        g.launch_sgemm(d, 3 * h, bt, bufs["X_all"], bufs["dXg_all"], bufs["g_W_x"], transA=True, beta=0.0)
        g.launch_col_sum(bt, 3 * h, bufs["dXg_all"], bufs["g_b_x"])
        g.launch_sgemm(h, 3 * h, bt, bufs["S_prev_all"], bufs["dSg_all"], bufs["g_U_s"], transA=True, beta=0.0)
        g.launch_col_sum(bt, 3 * h, bufs["dSg_all"], bufs["g_b_s"])
        g.launch_sgemm(bt, d, 3 * h, bufs["dXg_all"], bufs["p_W_x"], bufs["dX_all"], transB=True, beta=0.0)
        g.launch_embed_grad_reduce_1(bt, v, d, bufs["tok_in"], bufs["dX_all"], bufs["g_E"])

        # 6. Gradient clipping + Fused on-device AdamW
        self._adam_t += 1
        lr_cur = self._lr_for_step(self._adam_t, total_steps=total_steps)
        bc1 = 1.0 - (0.9 ** self._adam_t)
        bc2 = 1.0 - (0.999 ** self._adam_t)
        step_sz = float(lr_cur * np.sqrt(bc2) / max(bc1, 1e-8))
        wd_s = float(1.0 - lr_cur * self.weight_decay)

        specs = [
            ("E", v * d, True),
            ("W_x", d * 3 * h, True),
            ("b_x", 3 * h, False),
            ("U_s", h * 3 * h, True),
            ("b_s", 3 * h, False),
        ]
        if self.num_layers == 2:
            specs.extend([
                ("W_x2", h * 3 * h, True),
                ("b_x2", 3 * h, False),
                ("U_s2", h * 3 * h, True),
                ("b_s2", 3 * h, False),
            ])
        specs.extend([
            ("W_out", h * v, True),
            ("b_out", v, False),
        ])
        for name, n_el, use_wd in specs:
            g.launch_clip_grad(n_el, self.grad_clip, bufs[f"g_{name}"])
            g.launch_adamw_update(
                n_el, step_sz, (wd_s if use_wd else 1.0),
                bufs[f"p_{name}"], bufs[f"g_{name}"], bufs[f"m_{name}"], bufs[f"v_{name}"],
            )

        self.total_tokens_trained += bt
        if read_loss:
            l_arr = np.empty(bt, dtype=np.float32)
            g.read_buffer(bufs["losses"], l_arr)
            if tok_tgt is not None:
                n_val = int(np.count_nonzero((tok_tgt > 0) & (tok_tgt < v)))
            else:
                # Non-zero losses correspond to valid non-<UNK> targets
                n_val = int(np.count_nonzero(l_arr > 0.0))
            return float(np.sum(l_arr) / float(max(n_val, 1)))
        return 0.0

    # ------------------------------------------------------------------
    # Exact CPU Reference BPTT (for unit tests & CPU fallback)
    # ------------------------------------------------------------------

    def _adam_step_cpu(self, name: str, param: np.ndarray, grad: np.ndarray, lr_cur: float) -> None:
        grad = np.clip(grad, -self.grad_clip, self.grad_clip)
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

    @staticmethod
    def _cpu_gru_layer_fwd(
        X_in: np.ndarray,       # [T, B, Din]
        W_x: np.ndarray,
        b_x: np.ndarray,
        U_s: np.ndarray,
        b_s: np.ndarray,
        s_init: np.ndarray,     # [B, H]
    ):
        T, B, Din = X_in.shape
        h = U_s.shape[0]
        Xg_all = (X_in.reshape(T * B, Din) @ W_x).reshape(T, B, 3 * h)
        S_prev = np.zeros((T, B, h), dtype=np.float32)
        S_all = np.zeros((T, B, h), dtype=np.float32)
        Sg_all = np.zeros((T, B, 3 * h), dtype=np.float32)
        Z_all = np.zeros((T, B, h), dtype=np.float32)
        R_all = np.zeros((T, B, h), dtype=np.float32)
        Ht_all = np.zeros((T, B, h), dtype=np.float32)
        s_cur = s_init.copy()
        for t in range(T):
            S_prev[t] = s_cur
            sg = s_cur @ U_s
            Sg_all[t] = sg
            z = _sigmoid(Xg_all[t, :, :h] + sg[:, :h] + b_x[:h] + b_s[:h])
            r = _sigmoid(Xg_all[t, :, h : 2 * h] + sg[:, h : 2 * h] + b_x[h : 2 * h] + b_s[h : 2 * h])
            sg_h = sg[:, 2 * h :] + b_s[2 * h :]
            ht = np.tanh(Xg_all[t, :, 2 * h :] + b_x[2 * h :] + r * sg_h)
            s_cur = (1.0 - z) * s_cur + z * ht
            S_all[t] = s_cur
            Z_all[t], R_all[t], Ht_all[t] = z, r, ht
        return S_all, s_cur, (S_prev, Sg_all, Z_all, R_all, Ht_all)

    @staticmethod
    def _cpu_gru_layer_bwd(
        X_in: np.ndarray,
        dS_out: np.ndarray,     # [T, B, H]
        W_x: np.ndarray,
        U_s: np.ndarray,
        b_s: np.ndarray,
        cache,
    ):
        S_prev, Sg_all, Z_all, R_all, Ht_all = cache
        T, B, h = dS_out.shape
        bt = T * B
        Din = X_in.shape[2]
        dXg_all = np.zeros((T, B, 3 * h), dtype=np.float32)
        dSg_all = np.zeros((T, B, 3 * h), dtype=np.float32)
        dS_carry = np.zeros((B, h), dtype=np.float32)
        for t in range(T - 1, -1, -1):
            ds = dS_out[t] + dS_carry
            z, r, ht = Z_all[t], R_all[t], Ht_all[t]
            sp = S_prev[t]
            sg_h = Sg_all[t, :, 2 * h :] + b_s[2 * h :]
            dz_pre = ds * (ht - sp) * z * (1.0 - z)
            dh_pre = ds * z * (1.0 - ht * ht)
            dr_pre = dh_pre * sg_h * r * (1.0 - r)
            dsg_h = dh_pre * r
            dXg_all[t, :, :h] = dz_pre
            dXg_all[t, :, h : 2 * h] = dr_pre
            dXg_all[t, :, 2 * h :] = dh_pre
            dSg_all[t, :, :h] = dz_pre
            dSg_all[t, :, h : 2 * h] = dr_pre
            dSg_all[t, :, 2 * h :] = dsg_h
            dS_carry = ds * (1.0 - z) + dSg_all[t] @ U_s.T
        dXg_flat = dXg_all.reshape(bt, 3 * h)
        dSg_flat = dSg_all.reshape(bt, 3 * h)
        dW_x = X_in.reshape(bt, Din).T @ dXg_flat
        db_x = dXg_flat.sum(axis=0)
        dU_s = S_prev.reshape(bt, h).T @ dSg_flat
        db_s = dSg_flat.sum(axis=0)
        dX_in = (dXg_flat @ W_x.T).reshape(T, B, Din)
        return dX_in, dW_x, db_x, dU_s, db_s

    def _cpu_bptt_step(
        self,
        tok_in: np.ndarray,
        tok_tgt: np.ndarray,
        s_init: Optional[np.ndarray] = None,
        s2_init: Optional[np.ndarray] = None,
        total_steps: int = 0,
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        T, B = tok_in.shape
        bt = B * T
        v, d, h = self.vocab_size, self.embed_dim, self.hidden_dim

        X_flat = np.where(tok_in.reshape(bt, 1) > 0, self.E_[tok_in.reshape(bt)], 0.0).astype(np.float32)
        X_in = X_flat.reshape(T, B, d)

        s1_0 = np.zeros((B, h), dtype=np.float32) if s_init is None else s_init
        S1_all, s1_cur, cache1 = self._cpu_gru_layer_fwd(X_in, self.W_x_, self.b_x_, self.U_s_, self.b_s_, s1_0)

        if self.num_layers == 2:
            s2_0 = np.zeros((B, h), dtype=np.float32) if s2_init is None else s2_init
            S2_all, s2_cur, cache2 = self._cpu_gru_layer_fwd(S1_all, self.W_x2_, self.b_x2_, self.U_s2_, self.b_s2_, s2_0)
            S_out = S1_all + S2_all
        else:
            s2_cur, cache2 = None, None
            S_out = S1_all

        S_flat = S_out.reshape(bt, h)
        logits = S_flat @ self.W_out_ + self.b_out_[None, :]
        logits[:, 0] = -1e4
        l_max = logits.max(axis=1, keepdims=True)
        ex = np.exp(np.clip(logits - l_max, -30.0, 0.0))
        probs = ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)

        y_flat = tok_tgt.reshape(bt)
        valid_mask = (y_flat > 0) & (y_flat < v)
        rows = np.arange(bt)
        row_losses = np.zeros(bt, dtype=np.float32)
        n_val = int(np.count_nonzero(valid_mask))
        if n_val > 0:
            row_losses[valid_mask] = -np.log(np.clip(probs[rows[valid_mask], y_flat[valid_mask]], 1e-9, 1.0))
        loss_val = float(np.sum(row_losses) / float(max(n_val, 1)))

        dL = probs
        if n_val > 0:
            dL[rows[valid_mask], y_flat[valid_mask]] -= 1.0
        dL[~valid_mask] = 0.0
        dL *= np.float32(1.0 / float(bt))

        dW_out = S_flat.T @ dL
        db_out = dL.sum(axis=0)
        dS_out = (dL @ self.W_out_.T).reshape(T, B, h)

        if self.num_layers == 2:
            dS1_from_l2, dW_x2, db_x2, dU_s2, db_s2 = self._cpu_gru_layer_bwd(
                S1_all, dS_out, self.W_x2_, self.U_s2_, self.b_s2_, cache2,
            )
            dS1_tot = dS_out + dS1_from_l2
        else:
            dS1_tot = dS_out

        dX_in, dW_x, db_x, dU_s, db_s = self._cpu_gru_layer_bwd(
            X_in, dS1_tot, self.W_x_, self.U_s_, self.b_s_, cache1,
        )
        dE = np.zeros_like(self.E_)
        np.add.at(dE, tok_in.reshape(bt), dX_in.reshape(bt, d))
        dE[0] = 0.0

        self._adam_t += 1
        lr_cur = self._lr_for_step(self._adam_t, total_steps=total_steps)
        self._adam_step_cpu("E", self.E_, dE, lr_cur)
        self._adam_step_cpu("W_x", self.W_x_, dW_x, lr_cur)
        self._adam_step_cpu("b_x", self.b_x_, db_x, lr_cur)
        self._adam_step_cpu("U_s", self.U_s_, dU_s, lr_cur)
        self._adam_step_cpu("b_s", self.b_s_, db_s, lr_cur)
        if self.num_layers == 2:
            self._adam_step_cpu("W_x2", self.W_x2_, dW_x2, lr_cur)
            self._adam_step_cpu("b_x2", self.b_x2_, db_x2, lr_cur)
            self._adam_step_cpu("U_s2", self.U_s2_, dU_s2, lr_cur)
            self._adam_step_cpu("b_s2", self.b_s2_, db_s2, lr_cur)
        self._adam_step_cpu("W_out", self.W_out_, dW_out, lr_cur)
        self._adam_step_cpu("b_out", self.b_out_, db_out, lr_cur)
        self.E_[0] = 0.0
        self.total_tokens_trained += bt
        return loss_val, s1_cur, s2_cur

    def fit_stream(
        self,
        tokens: Sequence[int],
        *,
        init_embeddings: Optional[np.ndarray] = None,
        epochs: int = 1,
        log_every: int = 0,
        total_steps_hint: int = 0,
    ) -> "LearnableStateSpacePolicy":
        """Train across a raw token stream via VRAM-resident parallel-stream BPTT."""
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        if self.E_ is None:
            self.init_params(init_embeddings=init_embeddings)
            cnts = np.bincount(arr, minlength=self.vocab_size).astype(np.float64)
            cnts[0] = 0.0
            p_u = (cnts + 0.5) / (max(float(cnts.sum()), 1.0) + 0.5 * self.vocab_size)
            self.b_out_ = np.log(np.clip(p_u, 1e-7, 1.0)).astype(np.float32)
            self.b_out_[0] = -10.0
        return self.ingest_chunk(arr, epochs=epochs, log_every=log_every, total_steps_hint=total_steps_hint)

    def ingest_chunk(
        self,
        tokens: Sequence[int],
        *,
        epochs: int = 1,
        log_every: int = 0,
        total_steps_hint: int = 0,
    ) -> "LearnableStateSpacePolicy":
        """Stream-train BPTT with the ENTIRE token array resident in dedicated GPU VRAM."""
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        n = len(arr)
        T = min(self.bptt_len, max(n // 16, 4))
        B = min(self.batch_streams, max(n // (T * 4), 1))
        if n < B * (T + 1):
            return self
        if self.E_ is None:
            self.init_params()

        stream_len = n // B
        n_steps = (stream_len - 1) // T
        if n_steps <= 0:
            return self
        trimmed = np.ascontiguousarray(arr[: B * stream_len].reshape(B, stream_len), dtype=np.int32)
        n_ep = max(int(epochs), 1)
        tot_steps = int(total_steps_hint) if total_steps_hint > 0 else (n_ep * n_steps)

        on_gpu = (B * T >= 512) and self._ensure_gpu_trainer(B, T)
        if not on_gpu:
            self.train_backend = "cpu"

        # Upload entire token chunk ONCE into dedicated GPU VRAM (zero PCIe traffic inside loop!)
        b_full_stream = None
        if on_gpu and self._gpu is not None and self._gpu._k_slice_tokens is not None:
            b_full_stream = self._gpu.create_buffer_from_array(trimmed, read_only=True)
            self.vram_allocated_mb = round(
                self.vram_allocated_mb + trimmed.nbytes / (1024.0 * 1024.0), 1,
            )

        loss_acc = 0.0
        loss_cnt = 0
        step_i = 0

        try:
            for ep_i in range(n_ep):
                s1_cpu = np.zeros((B, self.hidden_dim), dtype=np.float32)
                s2_cpu = np.zeros((B, self.hidden_dim), dtype=np.float32) if self.num_layers == 2 else None
                for k_step in range(n_steps):
                    s0 = k_step * T
                    step_i += 1
                    if on_gpu and b_full_stream is not None:
                        read_l = (step_i % 50 == 0) or (k_step == n_steps - 1)
                        self._gpu.launch_slice_tokens(
                            B, T, stream_len, s0, b_full_stream,
                            self._gpu_bufs["tok_in"], self._gpu_bufs["tok_tgt"],
                        )
                        lv = self._gpu_bptt_step(
                            None, None, B=B, T=T,
                            reset_hidden=(k_step == 0), read_loss=read_l, total_steps=tot_steps,
                        )
                        if read_l:
                            loss_acc += lv
                            loss_cnt += 1
                            if log_every > 0 and step_i % log_every == 0:
                                lr_now = self._lr_for_step(self._adam_t, total_steps=tot_steps)
                                print(
                                    f"    [BPTT ep={ep_i+1}/{n_ep} step {step_i}/{n_ep*n_steps}] "
                                    f"tokens={self.total_tokens_trained:,} lr={lr_now:.5f} loss={lv:.3f}",
                                    flush=True,
                                )
                    else:
                        t_in = np.ascontiguousarray(trimmed[:, s0 : s0 + T].T, dtype=np.int32)
                        t_tg = np.ascontiguousarray(trimmed[:, s0 + 1 : s0 + T + 1].T, dtype=np.int32)
                        if on_gpu:
                            read_l = (step_i % 50 == 0) or (k_step == n_steps - 1)
                            lv = self._gpu_bptt_step(
                                t_in, t_tg, reset_hidden=(k_step == 0), read_loss=read_l, total_steps=tot_steps,
                            )
                            if read_l:
                                loss_acc += lv
                                loss_cnt += 1
                        else:
                            lv, s1_cpu, s2_cpu = self._cpu_bptt_step(
                                t_in, t_tg, s_init=s1_cpu, s2_init=s2_cpu, total_steps=tot_steps,
                            )
                            loss_acc += lv
                            loss_cnt += 1
        finally:
            if b_full_stream is not None and self._gpu is not None:
                self._gpu.release_buffer(b_full_stream)
                self.vram_allocated_mb = max(
                    round(self.vram_allocated_mb - trimmed.nbytes / (1024.0 * 1024.0), 1), 0.0,
                )

        if on_gpu:
            self._sync_weights_from_gpu()
        if loss_cnt > 0:
            self.last_loss_ = round(loss_acc / float(loss_cnt), 4)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Live IN-PLACE State Evolution & Self-Learned Slow-Channel Anchoring
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self.state = self._make_zero_state()

    def pin_macro_anchor(self) -> None:
        """Freeze the prompt's slow macro-context state & self-learned high-tau channel mask."""
        top_s = self.state.s2 if (self.num_layers == 2 and self.state.s2 is not None) else self.state.s
        self.state.s_anchor = top_s.copy()
        if self.state.z_ema_top is not None:
            # Purely task-agnostic: channels with small learned write-gate z_ema (high tau)
            # are the self-discovered slow macro-topic capacitors; fast syntactic channels (z >= 0.55) get 0.0!
            slowness = np.clip((0.55 - self.state.z_ema_top) / 0.45, 0.0, 1.0) ** 2
            self.state.slow_anchor_w = (self.slow_anchor_strength * slowness).astype(np.float32)
        else:
            self.state.slow_anchor_w = np.full(self.hidden_dim, 0.08, dtype=np.float32)

    def _step_state_inplace(
        self,
        st: LearnableSSMState,
        symbol: int,
        *,
        is_self: bool = False,
    ) -> LearnableSSMState:
        """Modify an EXISTING ``LearnableSSMState`` strictly in-place (zero state rebuild!)."""
        if self.E_ is None or self.W_x_ is None or self.U_s_ is None:
            return st
        sym = int(symbol)
        h = self.hidden_dim
        x = self.E_[sym] if (0 < sym < self.vocab_size) else np.zeros(self.embed_dim, dtype=np.float32)

        # Layer 1 in-place step (always 100% unconstrained for fluent syntax!)
        xg1 = x @ self.W_x_ + self.b_x_
        sg1 = st.s @ self.U_s_ + self.b_s_
        z1 = _sigmoid(xg1[:h] + sg1[:h])
        r1 = _sigmoid(xg1[h : 2 * h] + sg1[h : 2 * h])
        ht1 = np.tanh(xg1[2 * h :] + r1 * sg1[2 * h :])
        st.s = ((np.float32(1.0) - z1) * st.s + z1 * ht1).astype(np.float32, copy=False)

        z_top = z1
        if self.num_layers == 2 and self.W_x2_ is not None and self.U_s2_ is not None:
            if st.s2 is None:
                st.s2 = np.zeros(h, dtype=np.float32)
            xg2 = st.s @ self.W_x2_ + self.b_x2_
            sg2 = st.s2 @ self.U_s2_ + self.b_s2_
            z2 = _sigmoid(xg2[:h] + sg2[:h])
            r2 = _sigmoid(xg2[h : 2 * h] + sg2[h : 2 * h])
            ht2 = np.tanh(xg2[2 * h :] + r2 * sg2[2 * h :])
            st.s2 = ((np.float32(1.0) - z2) * st.s2 + z2 * ht2).astype(np.float32, copy=False)
            z_top = z2
            # Gentle in-place high-tau capacitor tether during self_mode prevents 100-step macro drift
            if is_self and st.s_anchor is not None and st.slow_anchor_w is not None:
                tether = np.float32(0.08) * st.slow_anchor_w
                st.s2 = ((np.float32(1.0) - tether) * st.s2 + tether * st.s_anchor).astype(np.float32, copy=False)

        if not is_self:
            if st.z_ema_top is None:
                st.z_ema_top = z_top.copy()
            else:
                st.z_ema_top = (np.float32(0.92) * st.z_ema_top + np.float32(0.08) * z_top).astype(np.float32, copy=False)

        st.steps += 1
        return st

    def observe(self, symbol: int, *, chunk_id: int = 0, source: str = "external") -> LearnableSSMState:
        del chunk_id
        return self._step_state_inplace(self.state, symbol, is_self=(source == "self"))

    def fork_step(
        self,
        parent_state: LearnableSSMState,
        symbol: int,
        *,
        chunk_id: int = 0,
        is_self: bool = True,
    ) -> LearnableSSMState:
        del chunk_id
        child = parent_state.copy()
        return self._step_state_inplace(child, symbol, is_self=is_self)

    def _effective_readout_vec(self, st: LearnableSSMState, *, self_mode: bool) -> np.ndarray:
        if self.num_layers == 2 and st.s2 is not None:
            s2_eff = st.s2
            if self_mode and st.s_anchor is not None:
                w = st.slow_anchor_w if st.slow_anchor_w is not None else np.float32(0.10)
                s2_eff = (np.float32(1.0) - w) * s2_eff + w * st.s_anchor
            return (st.s + s2_eff).astype(np.float32, copy=False)
        s_eff = st.s
        if self_mode and st.s_anchor is not None:
            w = st.slow_anchor_w if st.slow_anchor_w is not None else np.float32(0.08)
            s_eff = (np.float32(1.0) - w) * s_eff + w * st.s_anchor
        return s_eff

    def predict_symbol_map(
        self,
        *,
        state: Optional[LearnableSSMState] = None,
        k: int = 24,
        self_mode: bool = False,
    ) -> Dict[int, float]:
        maps, _ = self.predict_batch_with_pmi(
            [state if state is not None else self.state], k=k, self_mode=self_mode, need_pmi=False,
        )
        return maps[0]

    def predict_map_and_pmi(
        self,
        *,
        state: Optional[LearnableSSMState] = None,
        k: int = 16,
        self_mode: bool = True,
        need_pmi: bool = False,
    ) -> Tuple[Dict[int, float], Optional[np.ndarray]]:
        st = state if state is not None else self.state
        maps, pmis = self.predict_batch_with_pmi([st], k=k, self_mode=self_mode, need_pmi=need_pmi)
        return maps[0], (pmis[0] if pmis is not None else None)

    def predict_batch_with_pmi(
        self,
        states: Sequence[LearnableSSMState],
        *,
        k: int = 16,
        self_mode: bool = True,
        need_pmi: bool = False,
    ) -> Tuple[List[Dict[int, float]], Optional[np.ndarray]]:
        """Ultra-fast batched readout of ``B`` persistent states (``[B, H] @ [H, V]`` < 0.2 ms!)."""
        n_b = len(states)
        if n_b == 0 or not self._fitted or self.W_out_ is None or self.b_out_ is None:
            return [{} for _ in range(n_b)], None
        s_list, valid_rows = [], []
        for idx_s, st in enumerate(states):
            if st is not None and st.steps > 0:
                s_list.append(self._effective_readout_vec(st, self_mode=self_mode))
                valid_rows.append(idx_s)
        if not valid_rows:
            return [{} for _ in range(n_b)], None

        S_mat = np.stack(s_list, axis=0)
        logits = (S_mat @ self.W_out_) + self.b_out_[None, :]
        logits[:, 0] = -1e4

        v = self.vocab_size
        k_eff = min(max(int(k), 1), v - 1)
        out_maps: List[Dict[int, float]] = [{} for _ in range(n_b)]
        out_pmis = np.zeros((n_b, v), dtype=np.float32) if need_pmi else None
        if need_pmi:
            b_mask = self.b_out_.copy()
            b_mask[0] = -1e4
            lp_base = _log_softmax_1d(b_mask)
        else:
            lp_base = None

        for j, orig_i in enumerate(valid_rows):
            lf = logits[j]
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
            if need_pmi and lp_base is not None and out_pmis is not None:
                delta = np.clip(_log_softmax_1d(lf) - lp_base, -2.5, 2.5)
                delta[0] = 0.0
                out_pmis[orig_i] = delta

        return out_maps, out_pmis

    def context_log_bayes_factor(
        self,
        *,
        state: Optional[LearnableSSMState] = None,
        self_mode: bool = False,
    ) -> Optional[np.ndarray]:
        _, pmi = self.predict_map_and_pmi(state=state, k=8, self_mode=self_mode, need_pmi=True)
        return pmi

    def prune_under_pressure(self, pressure: float) -> int:
        if pressure >= 0.40 and self._adam_m:
            n = len(self._adam_m) + len(self._adam_v)
            self._adam_m.clear()
            self._adam_v.clear()
            return n
        return 0
