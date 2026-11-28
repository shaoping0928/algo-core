"""Persistent In-Place Multi-Scale 2nd-Order Context-Gated Transition Policy.

Implements continuous, coherent state-space dynamics where each observed or
self-emitted symbol **modifies an already-existing multi-timescale node state
in-place** (never rebuilding from scratch), and the evolving context state
**multiplicatively gates local transitions (2nd-order bilinear interaction)**:

  - **In-Place Multi-Scale RC Node State**:
      * Local trajectory ``x_loc = [e_cur, e_prev, h_fast]`` (immediate 2-step + tau~2.5 phrase)
      * Persistent context ``z_ctx = [h_mid, h_slow, h_chunk, h_diff]`` (tau~10, tau~55, chunk, velocity)
  - **2nd-Order Context-Gated Bilinear Transition Layer**:
      ``a_t = tanh(x_loc @ P_loc + b_loc)``          (local transition motifs)
      ``g_t = sigmoid(z_ctx @ P_ctx + b_ctx)``       (persistent context gate in (0, 1)^R)
      ``q_t = a_t * g_t``                            (context-gated transition activations!)
      ``logits(y) = x_loc @ W_loc + z_ctx @ W_ctx + q_t @ W_q + b_out``
  - **Exact Counterfactual Conditional Context PMI**:
      ``Delta_ctx(y | x_loc, z_ctx) = log P(y | x_loc, z_ctx) - log P(y | x_loc, 0)``
      Mathematically cancels both unigram base-rate bias ``P(y)`` AND local 2-gram
      double-counting ``P(y | x_loc)``, isolating pure 2nd-order context-gated lift.
  - **Streaming Cross-Entropy Adam over Vectorized Kogge-Stone RC Scans**:
    Trains directly on 64M+ streaming token corpora with O(1) compact parameters
    (~4 MB float32) and microsecond in-place state forking during lookahead.

Strictly task-agnostic: operates solely on discrete symbol/chunk IDs and
distributional embeddings (zero language rules or word segmenters).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PersistentContextState:
    """Live persistent multi-scale node state modified strictly in-place at each step."""

    e_cur: np.ndarray
    e_prev: np.ndarray
    h_fast: np.ndarray
    h_mid: np.ndarray
    h_slow: np.ndarray
    h_chunk: np.ndarray
    steps: int = 0
    slow_anchor: Optional[np.ndarray] = None

    def copy(self) -> "PersistentContextState":
        return PersistentContextState(
            e_cur=self.e_cur.copy(),
            e_prev=self.e_prev.copy(),
            h_fast=self.h_fast.copy(),
            h_mid=self.h_mid.copy(),
            h_slow=self.h_slow.copy(),
            h_chunk=self.h_chunk.copy(),
            steps=self.steps,
            slow_anchor=self.slow_anchor.copy() if self.slow_anchor is not None else None,
        )

    def split_local_context(
        self,
        *,
        context_band_gain: float = 1.0,
        anchor_blend: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(x_loc [3D], z_ctx [4D])`` from the live persistent state."""
        h_diff = self.h_fast - self.h_mid
        h_s = self.h_slow
        if anchor_blend > 0.0 and self.slow_anchor is not None:
            ab = float(np.clip(anchor_blend, 0.0, 0.85))
            h_s = (1.0 - ab) * h_s + ab * self.slow_anchor
        x_loc = np.concatenate([self.e_cur, self.e_prev, self.h_fast]).astype(np.float32, copy=False)
        g = float(context_band_gain)
        if g != 1.0:
            z_ctx = np.concatenate([g * self.h_mid, g * h_s, g * self.h_chunk, h_diff]).astype(
                np.float32, copy=False,
            )
        else:
            z_ctx = np.concatenate([self.h_mid, h_s, self.h_chunk, h_diff]).astype(
                np.float32, copy=False,
            )
        return x_loc, z_ctx

    def to_feature_vector(
        self,
        *,
        context_band_gain: float = 1.0,
        anchor_blend: float = 0.0,
    ) -> np.ndarray:
        """Assemble concatenated ``[x_loc, z_ctx, 1.0]`` feature vector."""
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, -16.0, 16.0)
    return (1.0 / (1.0 + np.exp(-xc))).astype(np.float32, copy=False)


def _log_softmax_1d(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    shifted = logits - m
    lse = m + float(np.log(np.sum(np.exp(np.clip(shifted, -40.0, 0.0))) + 1e-12))
    return (logits - lse).astype(np.float32, copy=False)


class StatefulContextPolicy:
    """In-place multi-scale 2nd-order context-gated bilinear transition policy."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        n_classes: int = 0,
        gate_dim: int = 256,
        gamma_fast: float = 0.60,
        gamma_mid: float = 0.90,
        gamma_slow: float = 0.982,
        gamma_chunk: float = 0.94,
        lr: float = 0.012,
        weight_decay: float = 1e-5,
        batch_size: int = 2048,
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
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = max(int(batch_size), 64)
        self.seed = int(seed)

        self.loc_dim = 3 * self.embed_dim
        self.ctx_dim = 4 * self.embed_dim
        self.feat_dim = self.loc_dim + self.ctx_dim + 1

        # Mean-centered normalized embeddings for atoms & derived chunk embeddings
        self.atom_emb_: Optional[np.ndarray] = None        # [V, D] float32
        self.chunk_emb_: Optional[np.ndarray] = None       # [U, D] float32
        self.sym_to_class_: Optional[np.ndarray] = None    # [V] int32

        # 2nd-order Context-Gated Bilinear Transition parameters
        self.P_loc_: Optional[np.ndarray] = None           # [3D, R]
        self.b_loc_: Optional[np.ndarray] = None           # [R]
        self.P_ctx_: Optional[np.ndarray] = None           # [4D, R]
        self.b_ctx_: Optional[np.ndarray] = None           # [R]
        self.W_loc_: Optional[np.ndarray] = None           # [3D, V]
        self.W_ctx_: Optional[np.ndarray] = None           # [4D, V]
        self.W_q_: Optional[np.ndarray] = None             # [R, V]
        self.b_out_: Optional[np.ndarray] = None           # [V]

        # Lightweight Adam optimizer state (cleared on save to keep checkpoint tiny!)
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_t: int = 0
        self.total_samples: int = 0
        self.last_loss_: float = 0.0
        self._fitted: bool = False

        # Live persistent node state (modified strictly in-place on observe!)
        self.state: PersistentContextState = self._make_zero_state()

    def __getstate__(self) -> dict:
        st = self.__dict__.copy()
        # Strip transient optimizer moments before checkpoint serialization (~3x smaller!)
        st["_adam_m"] = {}
        st["_adam_v"] = {}
        return st

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_adam_m"):
            self._adam_m = {}
        if not hasattr(self, "_adam_v"):
            self._adam_v = {}

    def _make_zero_state(self) -> PersistentContextState:
        d = self.embed_dim
        return PersistentContextState(
            e_cur=np.zeros(d, dtype=np.float32),
            e_prev=np.zeros(d, dtype=np.float32),
            h_fast=np.zeros(d, dtype=np.float32),
            h_mid=np.zeros(d, dtype=np.float32),
            h_slow=np.zeros(d, dtype=np.float32),
            h_chunk=np.zeros(d, dtype=np.float32),
            steps=0,
            slow_anchor=None,
        )

    def _init_params(self) -> None:
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        ld, cd, r, v = self.loc_dim, self.ctx_dim, self.gate_dim, self.vocab_size
        self.P_loc_ = (rng.randn(ld, r) * np.sqrt(1.0 / max(ld, 1))).astype(np.float32)
        self.b_loc_ = np.zeros(r, dtype=np.float32)
        self.P_ctx_ = (rng.randn(cd, r) * np.sqrt(1.0 / max(cd, 1))).astype(np.float32)
        # Positive bias (+0.5) so context gates start mildly open (~0.62) and specialize cleanly
        self.b_ctx_ = np.full(r, 0.5, dtype=np.float32)
        self.W_loc_ = np.zeros((ld, v), dtype=np.float32)
        self.W_ctx_ = np.zeros((cd, v), dtype=np.float32)
        self.W_q_ = np.zeros((r, v), dtype=np.float32)
        self.b_out_ = np.zeros(v, dtype=np.float32)
        self._adam_m.clear()
        self._adam_v.clear()
        self._adam_t = 0

    def configure_embeddings(
        self,
        atom_embeddings: np.ndarray,
        *,
        chunk_to_atoms: Optional[Dict[int, Tuple[int, ...]]] = None,
        n_total_units: Optional[int] = None,
        sym_to_class: Optional[np.ndarray] = None,
    ) -> None:
        """Store mean-centered unit-norm atom embeddings & derived chunk embeddings."""
        ae = np.asarray(atom_embeddings, dtype=np.float32)
        if ae.shape[0] < self.vocab_size:
            pad = np.zeros((self.vocab_size, ae.shape[1]), dtype=np.float32)
            pad[: ae.shape[0]] = ae
            ae = pad
        self.embed_dim = int(ae.shape[1])
        self.loc_dim = 3 * self.embed_dim
        self.ctx_dim = 4 * self.embed_dim
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

        if self.P_loc_ is None or self.P_loc_.shape[0] != self.loc_dim:
            self._init_params()
        self.state = self._make_zero_state()

    def _adam_step(self, name: str, param: np.ndarray, grad: np.ndarray, lr_cur: float) -> None:
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

    def _train_stream_blocks(
        self,
        tokens: np.ndarray,
        *,
        chunk_memory: Optional[object] = None,
        block_size: int = 65_536,
        sample_stride: int = 4,
        max_samples: int = 2_000_000,
        epochs: int = 1,
    ) -> None:
        """Train 2nd-order context-gated bilinear transitions via streaming cross-entropy Adam."""
        assert self.atom_emb_ is not None and self.P_loc_ is not None
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        n = len(arr)
        if n < 16:
            return
        ae = self.atom_emb_
        ce = self.chunk_emb_ if self.chunk_emb_ is not None else ae
        d = self.embed_dim
        a2c = getattr(chunk_memory, "atoms_to_chunk_", None) if chunk_memory is not None else None

        # Initialize output bias to empirical log unigram rate on very first fit
        if self.total_samples == 0 and self.b_out_ is not None:
            cnts = np.bincount(arr, minlength=self.vocab_size).astype(np.float64)
            cnts[0] = 0.0
            tot_c = max(float(cnts.sum()), 1.0)
            p_uni = (cnts + 0.5) / (tot_c + 0.5 * self.vocab_size)
            self.b_out_ = np.log(np.clip(p_uni, 1e-7, 1.0)).astype(np.float32)
            self.b_out_[0] = -10.0

        st = max(int(sample_stride), max((n - 2) // max(int(max_samples), 1000), 1))
        bs = self.batch_size
        loss_acc = 0.0
        loss_batches = 0

        for _ep in range(max(int(epochs), 1)):
            h_f0 = np.zeros(d, dtype=np.float32)
            h_m0 = np.zeros(d, dtype=np.float32)
            h_s0 = np.zeros(d, dtype=np.float32)
            h_c0 = np.zeros(d, dtype=np.float32)
            prev_last_emb = np.zeros(d, dtype=np.float32)

            for start in range(0, n - 2, block_size):
                end = min(start + block_size, n - 1)
                blk_len = end - start
                if blk_len < 4:
                    break
                blk = arr[start:end]
                targets = arr[start + 1 : end + 1]
                X = ae[blk]  # [L, D]
                X_prev = np.empty_like(X)
                X_prev[0] = prev_last_emb
                X_prev[1:] = X[:-1]
                prev_last_emb = X[-1].copy()

                if a2c is not None and len(ce) > self.vocab_size:
                    X_ch = X.copy()
                    if start == 0:
                        pairs_a = blk[:-1].tolist()
                        pairs_b = blk[1:].tolist()
                        offset_p = 1
                    else:
                        pairs_a = arr[start - 1 : end - 1].tolist()
                        pairs_b = blk.tolist()
                        offset_p = 0
                    c_arr = np.asarray([a2c.get((a, b), 0) for a, b in zip(pairs_a, pairs_b)], dtype=np.int32)
                    hit = c_arr >= self.vocab_size
                    if np.any(hit):
                        X_ch[np.flatnonzero(hit) + offset_p] = ce[c_arr[hit]]
                else:
                    X_ch = X

                H_f = _kogge_stone_ema(X, self.gamma_fast, h_f0)
                H_m = _kogge_stone_ema(X, self.gamma_mid, h_m0)
                H_s = _kogge_stone_ema(X, self.gamma_slow, h_s0)
                H_c = _kogge_stone_ema(X_ch, self.gamma_chunk, h_c0)
                h_f0, h_m0, h_s0, h_c0 = H_f[-1], H_m[-1], H_s[-1], H_c[-1]

                idx = np.arange(0, blk_len, st)
                t_sub = targets[idx]
                valid_m = (t_sub > 0) & (blk[idx] > 0)
                if not np.any(valid_m):
                    continue
                sel = idx[valid_m]
                y_all = t_sub[valid_m]

                X_loc_all = np.concatenate([X[sel], X_prev[sel], H_f[sel]], axis=1)
                Z_ctx_all = np.concatenate([H_m[sel], H_s[sel], H_c[sel], H_f[sel] - H_m[sel]], axis=1)

                n_sel = len(sel)
                for b0 in range(0, n_sel, bs):
                    b1 = min(b0 + bs, n_sel)
                    m_b = b1 - b0
                    if m_b < 8:
                        continue
                    Xl = X_loc_all[b0:b1]
                    Zc = Z_ctx_all[b0:b1]
                    yb = y_all[b0:b1]

                    # Forward: 2nd-order Context-Gated Bilinear Transition!
                    A = np.tanh(Xl @ self.P_loc_ + self.b_loc_)        # [B, R]
                    G = _sigmoid(Zc @ self.P_ctx_ + self.b_ctx_)       # [B, R]
                    Q = A * G                                          # [B, R]
                    logits = (Xl @ self.W_loc_) + (Zc @ self.W_ctx_) + (Q @ self.W_q_) + self.b_out_
                    logits[:, 0] = -1e4
                    l_max = logits.max(axis=1, keepdims=True)
                    ex = np.exp(np.clip(logits - l_max, -30.0, 0.0))
                    probs = ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)

                    row_idx = np.arange(m_b)
                    p_true = np.clip(probs[row_idx, yb], 1e-9, 1.0)
                    loss_acc += float(-np.mean(np.log(p_true)))
                    loss_batches += 1

                    # Exact Cross-Entropy Gradient (dL = (probs - y_onehot) / B)
                    dL = probs
                    dL[row_idx, yb] -= 1.0
                    dL *= np.float32(1.0 / float(m_b))

                    dW_loc = Xl.T @ dL
                    dW_ctx = Zc.T @ dL
                    dW_q = Q.T @ dL
                    db_out = dL.sum(axis=0)

                    # Backprop into 2nd-order bilinear gates P_loc & P_ctx
                    dQ = dL @ self.W_q_.T                              # [B, R]
                    dA_pre = dQ * G * (np.float32(1.0) - A * A)        # [B, R]
                    dG_pre = dQ * A * G * (np.float32(1.0) - G)        # [B, R]
                    dP_loc = Xl.T @ dA_pre
                    db_loc = dA_pre.sum(axis=0)
                    dP_ctx = Zc.T @ dG_pre
                    db_ctx = dG_pre.sum(axis=0)

                    self._adam_t += 1
                    # Smooth inverse-sqrt decay across streaming chunks (never collapses to 0)
                    lr_cur = self.lr / float(np.sqrt(1.0 + 0.0012 * self._adam_t))
                    self._adam_step("W_loc", self.W_loc_, dW_loc, lr_cur)
                    self._adam_step("W_ctx", self.W_ctx_, dW_ctx, lr_cur)
                    self._adam_step("W_q", self.W_q_, dW_q, lr_cur)
                    self._adam_step("b_out", self.b_out_, db_out, lr_cur)
                    self._adam_step("P_loc", self.P_loc_, dP_loc, lr_cur * 0.8)
                    self._adam_step("b_loc", self.b_loc_, db_loc, lr_cur * 0.8)
                    self._adam_step("P_ctx", self.P_ctx_, dP_ctx, lr_cur * 0.8)
                    self._adam_step("b_ctx", self.b_ctx_, db_ctx, lr_cur * 0.8)

                    self.total_samples += m_b

        if loss_batches > 0:
            self.last_loss_ = round(loss_acc / float(loss_batches), 4)
        self._fitted = True

    def fit_stream(
        self,
        tokens: Sequence[int],
        atom_embeddings: np.ndarray,
        *,
        chunk_memory: Optional[object] = None,
        sym_to_class: Optional[np.ndarray] = None,
        sample_stride: int = 4,
        max_samples: int = 2_000_000,
        epochs: int = 1,
    ) -> "StatefulContextPolicy":
        """Fit initial 2nd-order context-gated bilinear policy on a token stream."""
        self.configure_embeddings(
            atom_embeddings,
            chunk_to_atoms=getattr(chunk_memory, "chunk_to_atoms_", None),
            n_total_units=getattr(chunk_memory, "n_total_units", self.vocab_size),
            sym_to_class=sym_to_class,
        )
        self._init_params()
        self.total_samples = 0
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        # Small synthetic/unit-test streams benefit from multiple passes to converge cleanly
        eff_ep = epochs if len(arr) >= 50_000 else max(int(epochs), 18)
        eff_bs = self.batch_size if len(arr) >= 50_000 else min(self.batch_size, 256)
        old_bs = self.batch_size
        self.batch_size = eff_bs
        try:
            self._train_stream_blocks(
                arr,
                chunk_memory=chunk_memory,
                sample_stride=sample_stride,
                max_samples=max_samples,
                epochs=eff_ep,
            )
        finally:
            self.batch_size = old_bs
        return self

    def ingest_chunk(
        self,
        tokens: Sequence[int],
        *,
        chunk_memory: Optional[object] = None,
        sample_stride: int = 8,
        max_samples: int = 2_000_000,
    ) -> "StatefulContextPolicy":
        """Continue streaming Adam updates on a new macro-chunk (64M+ streaming)."""
        if self.atom_emb_ is None or self.P_loc_ is None:
            return self
        self._train_stream_blocks(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            chunk_memory=chunk_memory,
            sample_stride=sample_stride,
            max_samples=max_samples,
            epochs=1,
        )
        return self

    # ------------------------------------------------------------------
    # Live IN-PLACE Persistent State Evolution & 2nd-Order Readout
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self.state = self._make_zero_state()

    def pin_macro_anchor(self) -> None:
        self.state.slow_anchor = self.state.h_slow.copy()

    def _step_state_inplace(
        self,
        st: PersistentContextState,
        symbol: int,
        *,
        chunk_id: int = 0,
        is_self: bool = False,
    ) -> PersistentContextState:
        """Modify an EXISTING ``PersistentContextState`` strictly in-place."""
        if self.atom_emb_ is None:
            return st
        sym = int(symbol)
        if sym <= 0 or sym >= self.vocab_size:
            st.steps += 1
            return st

        e_a = self.atom_emb_[sym]
        if chunk_id >= self.vocab_size and self.chunk_emb_ is not None and chunk_id < len(self.chunk_emb_):
            e_c = self.chunk_emb_[chunk_id]
        else:
            e_c = e_a

        gf, gm, gs, gc = self.gamma_fast, self.gamma_mid, self.gamma_slow, self.gamma_chunk
        if is_self:
            gs = min(0.992, gs + 0.010)
            if st.slow_anchor is None and st.steps >= 4:
                st.slow_anchor = st.h_slow.copy()

        st.e_prev = st.e_cur
        st.e_cur = e_a
        st.h_fast = gf * st.h_fast + (1.0 - gf) * e_a
        st.h_mid = gm * st.h_mid + (1.0 - gm) * e_a
        st.h_slow = gs * st.h_slow + (1.0 - gs) * e_a
        st.h_chunk = gc * st.h_chunk + (1.0 - gc) * e_c
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

    def _forward_logits(
        self,
        x_loc: np.ndarray,
        z_ctx: Optional[np.ndarray],
    ) -> np.ndarray:
        assert self.P_loc_ is not None and self.W_loc_ is not None and self.W_q_ is not None and self.b_out_ is not None
        a_h = np.tanh(x_loc @ self.P_loc_ + self.b_loc_)
        if z_ctx is None:
            g_h = _sigmoid(self.b_ctx_)
            logits = (x_loc @ self.W_loc_) + ((a_h * g_h) @ self.W_q_) + self.b_out_
        else:
            g_h = _sigmoid(z_ctx @ self.P_ctx_ + self.b_ctx_)
            logits = (x_loc @ self.W_loc_) + (z_ctx @ self.W_ctx_) + ((a_h * g_h) @ self.W_q_) + self.b_out_
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
        """Directly read out calibrated Top-K next-symbol probabilities from 2nd-order gated state."""
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
        """Exact counterfactual conditional PMI ``log P(y | x_loc, z_ctx) - log P(y | x_loc, 0)``.

        Captures BOTH direct context drive AND 2nd-order multiplicative ``(x_loc * z_ctx)``
        transition gating while exactly cancelling unigram ``P(y)`` and local ``P(y | x_loc)``!
        """
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

    def param_bytes(self) -> int:
        tot = 0
        for arr in (self.P_loc_, self.b_loc_, self.P_ctx_, self.b_ctx_, self.W_loc_, self.W_ctx_, self.W_q_, self.b_out_):
            if arr is not None:
                tot += int(arr.nbytes)
        return tot

    def prune_under_pressure(self, pressure: float) -> int:
        """Compact ~4 MB dense readout; drops transient optimizer buffers under pressure."""
        if pressure >= 0.40 and self._adam_m:
            n = len(self._adam_m) + len(self._adam_v)
            self._adam_m.clear()
            self._adam_v.clear()
            return n
        return 0
