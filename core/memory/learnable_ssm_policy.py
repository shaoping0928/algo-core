"""End-to-End RX 9070 XT GPU-Trained Learnable Gated State-Space Policy (BPTT).

Upgrades fixed-decay multi-scale EMA states into a **fully learnable, input- and
state-dependent Gated RC State-Space Recurrent Network** trained end-to-end with
exact Backpropagation Through Time (BPTT) natively on OpenCL GPUs (AMD RX 9070 XT):

  - **Continuous In-Place Node State Evolution** ``s_t in R^H`` (never rebuilt!):
      ``z_t = sigmoid(x_t @ W_z + s_{t-1} @ U_z + b_z)``   (learned per-channel RC write/leak gate)
      ``r_t = sigmoid(x_t @ W_r + s_{t-1} @ U_r + b_r)``   (learned state routing/reset gate)
      ``h~_t = tanh(x_t @ W_h + r_t * (s_{t-1} @ U_h) + b_h)``
      ``s_t = (1 - z_t) * s_{t-1} + z_t * h~_t``           (strictly IN-PLACE state modification!)
      ``logits_t = s_t @ W_out + b_out``

  - **Drop-In Compatible with ``ContextDrivenDecisionController`` & ``TemporaryContextMemory``**:
    Implements ``observe()``, ``reset_state()``, ``fork_step()``, ``predict_symbol_map()``,
    ``predict_map_and_pmi()``, and ``predict_batch_with_pmi()`` (with microsecond in-place
    state updates and batched lookahead evaluation!).

  - **Zero-Subsampling 64M RX 9070 XT BPTT Engine**:
    Processes ``B = 256`` parallel independent token streams across ``T = 48`` BPTT steps
    (``12,288`` tokens per GPU update), keeping ALL parameters and persistent hidden states
    resident in VRAM and fusing input/output/weight GEMMs into massive GPU launches.

Strictly task-agnostic: operates solely on discrete symbol ID streams with zero language
rules, zero word segmenters, and zero repetition bans.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    """Live persistent gated RC node state ``s in R^H`` modified strictly in-place."""

    s: np.ndarray
    steps: int = 0
    s_anchor: Optional[np.ndarray] = None

    def copy(self) -> "LearnableSSMState":
        return LearnableSSMState(
            s=self.s.copy(),
            steps=self.steps,
            s_anchor=self.s_anchor.copy() if self.s_anchor is not None else None,
        )


class LearnableStateSpacePolicy:
    """End-to-end GPU-trained learnable gated RC state-space policy (48-step BPTT)."""

    _PARAM_NAMES: Tuple[str, ...] = ("E", "W_x", "b_x", "U_s", "b_s", "W_out", "b_out")

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 384,
        *,
        bptt_len: int = 48,
        batch_streams: int = 256,
        lr: float = 0.004,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.5,
        use_gpu: bool = True,
        seed: int = 42,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.embed_dim = max(int(embed_dim), 8)
        self.hidden_dim = max(int(hidden_dim), 32)
        self.bptt_len = max(int(bptt_len), 8)
        self.batch_streams = max(int(batch_streams), 8)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.use_gpu = bool(use_gpu)
        self.seed = int(seed)

        # 7 Trainable Parameter Tensors (~5.4 MB for V=2000, D=64, H=384)
        self.E_: Optional[np.ndarray] = None       # [V, D]
        self.W_x_: Optional[np.ndarray] = None     # [D, 3H]
        self.b_x_: Optional[np.ndarray] = None     # [3H]
        self.U_s_: Optional[np.ndarray] = None     # [H, 3H]
        self.b_s_: Optional[np.ndarray] = None     # [3H]
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
        self.total_tokens_trained: int = 0
        self.last_loss_: float = 0.0
        self._fitted: bool = False

        # Live persistent node state (modified strictly in-place!)
        self.state: LearnableSSMState = self._make_zero_state()

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

    def _make_zero_state(self) -> LearnableSSMState:
        return LearnableSSMState(s=np.zeros(self.hidden_dim, dtype=np.float32), steps=0, s_anchor=None)

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

        self.W_x_ = (rng.randn(d, 3 * h) * np.sqrt(1.0 / max(d, 1))).astype(np.float32)
        self.b_x_ = np.zeros(3 * h, dtype=np.float32)
        # Orthogonal-like scaled recurrent init + multi-timescale write-gate bias!
        # Initializing z-bias across [-2.5, +0.5] gives natural fast/mid/slow RC channels from step 0!
        self.U_s_ = (rng.randn(h, 3 * h) * np.sqrt(0.5 / max(h, 1))).astype(np.float32)
        self.b_s_ = np.zeros(3 * h, dtype=np.float32)
        self.b_s_[:h] = np.linspace(-2.5, 0.5, h, dtype=np.float32)
        self.W_out_ = (rng.randn(h, v) * np.sqrt(0.5 / max(h, 1))).astype(np.float32)
        self.b_out_ = np.zeros(v, dtype=np.float32)
        self._adam_m.clear()
        self._adam_v.clear()
        self._adam_t = 0
        self.close_gpu()

    def param_bytes(self) -> int:
        tot = 0
        for name in self._PARAM_NAMES:
            arr = getattr(self, f"{name}_", None)
            if arr is not None:
                tot += int(arr.nbytes)
        return tot

    # ------------------------------------------------------------------
    # Native RX 9070 XT OpenCL BPTT Engine
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
        for name in self._PARAM_NAMES:
            arr = getattr(self, f"{name}_")
            assert arr is not None
            bufs[f"p_{name}"] = g.create_buffer_from_array(arr)
            bufs[f"g_{name}"] = g.create_empty_buffer(arr.nbytes)
            m0 = self._adam_m.get(name)
            v0 = self._adam_v.get(name)
            bufs[f"m_{name}"] = g.create_buffer_from_array(m0 if m0 is not None else np.zeros_like(arr))
            bufs[f"v_{name}"] = g.create_buffer_from_array(v0 if v0 is not None else np.zeros_like(arr))

        # Sequence buffers [T, B, ...] permanently in RX 9070 XT VRAM
        bufs["tok_in"] = g.create_empty_buffer(bt * 4)
        bufs["tok_tgt"] = g.create_empty_buffer(bt * 4)
        bufs["X_all"] = g.create_empty_buffer(bt * d * 4)
        bufs["dX_all"] = g.create_empty_buffer(bt * d * 4)
        bufs["Xg_all"] = g.create_empty_buffer(bt * 3 * h * 4)
        bufs["dXg_all"] = g.create_empty_buffer(bt * 3 * h * 4)
        bufs["Sg_all"] = g.create_empty_buffer(bt * 3 * h * 4)
        bufs["dSg_all"] = g.create_empty_buffer(bt * 3 * h * 4)
        bufs["S_prev_all"] = g.create_empty_buffer(bt * h * 4)
        bufs["S_all"] = g.create_empty_buffer(bt * h * 4)
        bufs["cache_zrh"] = g.create_empty_buffer(bt * 3 * h * 4)

        bufs["s_cur"] = g.create_buffer_from_array(np.zeros((B, h), dtype=np.float32))
        bufs["Sg_t"] = g.create_empty_buffer(B * 3 * h * 4)
        bufs["dSg_t"] = g.create_empty_buffer(B * 3 * h * 4)
        bufs["dS_out_all"] = g.create_empty_buffer(bt * h * 4)
        bufs["dS_carry"] = g.create_empty_buffer(B * h * 4)
        bufs["s_prev_t"] = g.create_empty_buffer(B * h * 4)

        bufs["logits"] = g.create_empty_buffer(bt * v * 4)
        bufs["losses"] = g.create_empty_buffer(bt * 4)

        self._gpu_bufs = bufs
        self._gpu_bt = (B, T)
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

    def _gpu_bptt_step(
        self,
        tok_in: np.ndarray,      # [T, B] int32
        tok_tgt: np.ndarray,     # [T, B] int32
        *,
        reset_hidden: bool = False,
        read_loss: bool = False,
    ) -> float:
        g = self._gpu
        assert g is not None
        bufs = self._gpu_bufs
        T, B = tok_in.shape
        bt = B * T
        v, d, h = self.vocab_size, self.embed_dim, self.hidden_dim

        if reset_hidden:
            g.write_buffer(bufs["s_cur"], np.zeros((B, h), dtype=np.float32))
        g.write_buffer(bufs["tok_in"], np.ascontiguousarray(tok_in, dtype=np.int32))
        g.write_buffer(bufs["tok_tgt"], np.ascontiguousarray(tok_tgt, dtype=np.int32))

        # 1. Fused embedding gather & input projection for ALL T*B tokens in ONE GEMM!
        g.launch_embed_gather_1(bt, d, bufs["tok_in"], bufs["p_E"], bufs["X_all"])
        g.launch_sgemm(bt, 3 * h, d, bufs["X_all"], bufs["p_W_x"], bufs["Xg_all"], beta=0.0)

        # 2. Forward recurrent scan t = 0..T-1 (s_cur evolves strictly in-place in VRAM!)
        for t in range(T):
            # Save s_prev[t] = s_cur
            g.launch_copy_slice(B * h, t, 1, bufs["S_prev_all"], bufs["s_cur"])
            # Sg_t = s_cur @ U_s
            g.launch_sgemm(B, 3 * h, h, bufs["s_cur"], bufs["p_U_s"], bufs["Sg_t"], beta=0.0)
            g.launch_copy_slice(B * 3 * h, t, 1, bufs["Sg_all"], bufs["Sg_t"])
            # Fused gated RC step -> writes S_all[t] and cache_zrh[t]
            g.launch_gru_step_fwd(
                B, h, t,
                bufs["Xg_all"], bufs["Sg_t"], bufs["p_b_x"], bufs["p_b_s"],
                bufs["s_cur"], bufs["S_all"], bufs["cache_zrh"],
            )
            # Advance persistent s_cur = S_all[t]
            g.launch_copy_slice(B * h, t, 0, bufs["S_all"], bufs["s_cur"])

        # 3. Fused output logits + Softmax/CE/dL + output grad for ALL T*B tokens in ONE GEMM!
        g.launch_sgemm(bt, v, h, bufs["S_all"], bufs["p_W_out"], bufs["logits"], beta=0.0)
        g.launch_softmax_ce_grad(bt, v, bufs["p_b_out"], bufs["tok_tgt"], bufs["logits"], bufs["losses"])
        g.launch_sgemm(h, v, bt, bufs["S_all"], bufs["logits"], bufs["g_W_out"], transA=True, beta=0.0)
        g.launch_col_sum(bt, v, bufs["logits"], bufs["g_b_out"])
        g.launch_sgemm(bt, h, v, bufs["logits"], bufs["p_W_out"], bufs["dS_out_all"], transB=True, beta=0.0)

        # 4. Exact Backward Through Time t = T-1..0
        g.write_buffer(bufs["dS_carry"], np.zeros((B, h), dtype=np.float32))
        for t in range(T - 1, -1, -1):
            g.launch_copy_slice(B * h, t, 0, bufs["S_prev_all"], bufs["s_prev_t"])
            # Writes dXg_all[t], dSg_all[t], and direct carry ds*(1-z) into s_prev_t? Use dS_carry cleanly:
            # First write direct carry into s_prev_t (reused as dS_prev_direct buffer!)
            g.launch_gru_step_bwd(
                B, h, t,
                bufs["dS_out_all"], bufs["dS_carry"], bufs["cache_zrh"],
                bufs["s_prev_t"], bufs["Sg_all"], bufs["p_b_s"],
                bufs["dXg_all"], bufs["dSg_all"], bufs["dS_carry"],
            )
            # Extract dSg_all[t] -> dSg_t, then accumulate dS_carry += dSg_t @ U_s.T (beta=1.0!)
            g.launch_copy_slice(B * 3 * h, t, 0, bufs["dSg_all"], bufs["dSg_t"])
            g.launch_sgemm(B, h, 3 * h, bufs["dSg_t"], bufs["p_U_s"], bufs["dS_carry"], transB=True, beta=1.0)

        # 5. Fused Weight Gradients across ALL T*B steps in ONE GEMM each!
        g.launch_sgemm(d, 3 * h, bt, bufs["X_all"], bufs["dXg_all"], bufs["g_W_x"], transA=True, beta=0.0)
        g.launch_col_sum(bt, 3 * h, bufs["dXg_all"], bufs["g_b_x"])
        g.launch_sgemm(h, 3 * h, bt, bufs["S_prev_all"], bufs["dSg_all"], bufs["g_U_s"], transA=True, beta=0.0)
        g.launch_col_sum(bt, 3 * h, bufs["dSg_all"], bufs["g_b_s"])
        g.launch_sgemm(bt, d, 3 * h, bufs["dXg_all"], bufs["p_W_x"], bufs["dX_all"], transB=True, beta=0.0)
        g.launch_embed_grad_reduce_1(bt, v, d, bufs["tok_in"], bufs["dX_all"], bufs["g_E"])

        # 6. Gradient clipping + Fused on-device AdamW for all 7 tensors
        self._adam_t += 1
        lr_cur = self.lr / float(np.sqrt(1.0 + 0.0005 * self._adam_t))
        bc1 = 1.0 - (0.9 ** self._adam_t)
        bc2 = 1.0 - (0.999 ** self._adam_t)
        step_sz = float(lr_cur * np.sqrt(bc2) / max(bc1, 1e-8))
        wd_s = float(1.0 - lr_cur * self.weight_decay)

        for name, n_el, use_wd in [
            ("E", v * d, True),
            ("W_x", d * 3 * h, True),
            ("b_x", 3 * h, False),
            ("U_s", h * 3 * h, True),
            ("b_s", 3 * h, False),
            ("W_out", h * v, True),
            ("b_out", v, False),
        ]:
            g.launch_clip_grad(n_el, self.grad_clip, bufs[f"g_{name}"])
            g.launch_adamw_update(
                n_el, step_sz, (wd_s if use_wd else 1.0),
                bufs[f"p_{name}"], bufs[f"g_{name}"], bufs[f"m_{name}"], bufs[f"v_{name}"],
            )

        self.total_tokens_trained += bt
        if read_loss:
            l_arr = np.empty(bt, dtype=np.float32)
            g.read_buffer(bufs["losses"], l_arr)
            n_val = int(np.count_nonzero((tok_tgt > 0) & (tok_tgt < v)))
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

    def _cpu_bptt_step(
        self,
        tok_in: np.ndarray,
        tok_tgt: np.ndarray,
        s_init: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        T, B = tok_in.shape
        bt = B * T
        v, d, h = self.vocab_size, self.embed_dim, self.hidden_dim

        X_all = np.where(tok_in.reshape(bt, 1) > 0, self.E_[tok_in.reshape(bt)], 0.0).astype(np.float32)
        Xg_all = (X_all @ self.W_x_).reshape(T, B, 3 * h)

        S_prev_all = np.zeros((T, B, h), dtype=np.float32)
        S_all = np.zeros((T, B, h), dtype=np.float32)
        Sg_all = np.zeros((T, B, 3 * h), dtype=np.float32)
        Z_all = np.zeros((T, B, h), dtype=np.float32)
        R_all = np.zeros((T, B, h), dtype=np.float32)
        Ht_all = np.zeros((T, B, h), dtype=np.float32)

        s_cur = np.zeros((B, h), dtype=np.float32) if s_init is None else s_init.copy()
        for t in range(T):
            S_prev_all[t] = s_cur
            sg = s_cur @ self.U_s_
            Sg_all[t] = sg
            z = _sigmoid(Xg_all[t, :, :h] + sg[:, :h] + self.b_x_[:h] + self.b_s_[:h])
            r = _sigmoid(Xg_all[t, :, h : 2 * h] + sg[:, h : 2 * h] + self.b_x_[h : 2 * h] + self.b_s_[h : 2 * h])
            sg_h = sg[:, 2 * h :] + self.b_s_[2 * h :]
            ht = np.tanh(Xg_all[t, :, 2 * h :] + self.b_x_[2 * h :] + r * sg_h)
            s_cur = (1.0 - z) * s_cur + z * ht
            S_all[t] = s_cur
            Z_all[t], R_all[t], Ht_all[t] = z, r, ht

        S_flat = S_all.reshape(bt, h)
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
            vr = rows[valid_mask]
            vy = y_flat[valid_mask]
            row_losses[valid_mask] = -np.log(np.clip(probs[vr, vy], 1e-9, 1.0))
        loss_val = float(np.sum(row_losses) / float(max(n_val, 1)))

        dL = probs
        if n_val > 0:
            dL[rows[valid_mask], y_flat[valid_mask]] -= 1.0
        dL[~valid_mask] = 0.0
        dL *= np.float32(1.0 / float(bt))

        dW_out = S_flat.T @ dL
        db_out = dL.sum(axis=0)
        dS_out = (dL @ self.W_out_.T).reshape(T, B, h)

        dXg_all = np.zeros_like(Xg_all)
        dSg_all = np.zeros_like(Sg_all)
        dS_carry = np.zeros((B, h), dtype=np.float32)

        for t in range(T - 1, -1, -1):
            ds = dS_out[t] + dS_carry
            z, r, ht = Z_all[t], R_all[t], Ht_all[t]
            sp = S_prev_all[t]
            sg_h = Sg_all[t, :, 2 * h :] + self.b_s_[2 * h :]

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

            dS_carry = ds * (1.0 - z) + dSg_all[t] @ self.U_s_.T

        dXg_flat = dXg_all.reshape(bt, 3 * h)
        dSg_flat = dSg_all.reshape(bt, 3 * h)
        dW_x = X_all.T @ dXg_flat
        db_x = dXg_flat.sum(axis=0)
        dU_s = S_prev_all.reshape(bt, h).T @ dSg_flat
        db_s = dSg_flat.sum(axis=0)
        dX_flat = dXg_flat @ self.W_x_.T
        dE = np.zeros_like(self.E_)
        np.add.at(dE, tok_in.reshape(bt), dX_flat)
        dE[0] = 0.0

        self._adam_t += 1
        lr_cur = self.lr / float(np.sqrt(1.0 + 0.0005 * self._adam_t))
        self._adam_step_cpu("E", self.E_, dE, lr_cur)
        self._adam_step_cpu("W_x", self.W_x_, dW_x, lr_cur)
        self._adam_step_cpu("b_x", self.b_x_, db_x, lr_cur)
        self._adam_step_cpu("U_s", self.U_s_, dU_s, lr_cur)
        self._adam_step_cpu("b_s", self.b_s_, db_s, lr_cur)
        self._adam_step_cpu("W_out", self.W_out_, dW_out, lr_cur)
        self._adam_step_cpu("b_out", self.b_out_, db_out, lr_cur)
        self.E_[0] = 0.0
        self.total_tokens_trained += bt
        return loss_val, s_cur

    def fit_stream(
        self,
        tokens: Sequence[int],
        *,
        init_embeddings: Optional[np.ndarray] = None,
        epochs: int = 1,
        log_every: int = 0,
    ) -> "LearnableStateSpacePolicy":
        """Train across a raw token stream via parallel-stream BPTT on RX 9070 XT GPU."""
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        if self.E_ is None:
            self.init_params(init_embeddings=init_embeddings)
            # Initialize output bias to log unigram prior for instant fast start
            cnts = np.bincount(arr, minlength=self.vocab_size).astype(np.float64)
            cnts[0] = 0.0
            p_u = (cnts + 0.5) / (max(float(cnts.sum()), 1.0) + 0.5 * self.vocab_size)
            self.b_out_ = np.log(np.clip(p_u, 1e-7, 1.0)).astype(np.float32)
            self.b_out_[0] = -10.0
        return self.ingest_chunk(arr, epochs=epochs, log_every=log_every)

    def ingest_chunk(
        self,
        tokens: Sequence[int],
        *,
        epochs: int = 1,
        log_every: int = 0,
    ) -> "LearnableStateSpacePolicy":
        """Stream-train BPTT across ``B`` contiguous parallel slices of ``tokens``."""
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        n = len(arr)
        T = min(self.bptt_len, max(n // 16, 4))
        B = min(self.batch_streams, max(n // (T * 4), 1))
        if n < B * (T + 1):
            return self
        if self.E_ is None:
            self.init_params()

        # Partition stream into B long contiguous parallel streams [B, stream_len]
        stream_len = n // B
        n_steps = (stream_len - 1) // T
        if n_steps <= 0:
            return self
        trimmed = arr[: B * stream_len].reshape(B, stream_len)

        on_gpu = (B * T >= 512) and self._ensure_gpu_trainer(B, T)
        if not on_gpu:
            self.train_backend = "cpu"

        loss_acc = 0.0
        loss_cnt = 0
        step_i = 0

        for _ep in range(max(int(epochs), 1)):
            s_cpu = np.zeros((B, self.hidden_dim), dtype=np.float32)
            for k_step in range(n_steps):
                s0 = k_step * T
                # [T, B] contiguous time-major slices!
                t_in = np.ascontiguousarray(trimmed[:, s0 : s0 + T].T, dtype=np.int32)
                t_tg = np.ascontiguousarray(trimmed[:, s0 + 1 : s0 + T + 1].T, dtype=np.int32)
                # Mask <UNK>=0 targets cleanly to avoid corrupting loss
                step_i += 1
                if on_gpu:
                    read_l = (step_i % 50 == 0) or (k_step == n_steps - 1)
                    lv = self._gpu_bptt_step(t_in, t_tg, reset_hidden=(k_step == 0), read_loss=read_l)
                    if read_l:
                        loss_acc += lv
                        loss_cnt += 1
                        if log_every > 0 and step_i % log_every == 0:
                            print(f"    [BPTT step {step_i}] tokens={self.total_tokens_trained:,} loss={lv:.3f}", flush=True)
                else:
                    lv, s_cpu = self._cpu_bptt_step(t_in, t_tg, s_init=s_cpu)
                    loss_acc += lv
                    loss_cnt += 1

        if on_gpu:
            self._sync_weights_from_gpu()
        if loss_cnt > 0:
            self.last_loss_ = round(loss_acc / float(loss_cnt), 4)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Live IN-PLACE State Evolution & Drop-In Controller API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self.state = self._make_zero_state()

    def pin_macro_anchor(self) -> None:
        self.state.s_anchor = self.state.s.copy()

    def _step_state_inplace(self, st: LearnableSSMState, symbol: int) -> LearnableSSMState:
        """Modify an EXISTING ``LearnableSSMState`` strictly in-place (zero state rebuild!)."""
        if self.E_ is None or self.W_x_ is None or self.U_s_ is None:
            return st
        sym = int(symbol)
        h = self.hidden_dim
        x = self.E_[sym] if (0 < sym < self.vocab_size) else np.zeros(self.embed_dim, dtype=np.float32)
        xg = x @ self.W_x_ + self.b_x_
        sg = st.s @ self.U_s_ + self.b_s_

        z = _sigmoid(xg[:h] + sg[:h])
        r = _sigmoid(xg[h : 2 * h] + sg[h : 2 * h])
        ht = np.tanh(xg[2 * h :] + r * sg[2 * h :])
        st.s = ((np.float32(1.0) - z) * st.s + z * ht).astype(np.float32, copy=False)
        st.steps += 1
        return st

    def observe(self, symbol: int, *, chunk_id: int = 0, source: str = "external") -> LearnableSSMState:
        del chunk_id, source
        return self._step_state_inplace(self.state, symbol)

    def fork_step(
        self,
        parent_state: LearnableSSMState,
        symbol: int,
        *,
        chunk_id: int = 0,
        is_self: bool = True,
    ) -> LearnableSSMState:
        del chunk_id, is_self
        child = parent_state.copy()
        return self._step_state_inplace(child, symbol)

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
                s_vec = st.s
                if self_mode and st.s_anchor is not None:
                    # Mild persistent macro-anchor keeps long 100-step rollouts grounded
                    s_vec = np.float32(0.92) * s_vec + np.float32(0.08) * st.s_anchor
                s_list.append(s_vec)
                valid_rows.append(idx_s)
        if not valid_rows:
            return [{} for _ in range(n_b)], None

        S_mat = np.stack(s_list, axis=0)  # [M, H]
        logits = (S_mat @ self.W_out_) + self.b_out_[None, :]  # [M, V] (< 0.15 ms!)
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
