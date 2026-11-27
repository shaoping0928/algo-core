"""Task-agnostic online symbol context features and hierarchical stage compression.

Bridges discrete symbol streams (e.g., character sequences without any word
segmenter) directly into ``core/stage/summarizer.py`` (``StageRegimeSummarizer``):
  1. ``OnlineSymbolFeatureMap`` builds low-dimensional distributional symbol
     vectors via single-pass sparse Random Indexing over local co-occurrence
     windows, weighted by self-supervised inverse-frequency salience.
  2. Salience-weighted EMA produces a smooth slow-timescale context trajectory
     ``h_t`` suitable for ``StageRegimeSummarizer`` compression (with
     ``min_dwell`` and ``hysteresis``) into high-level stage nodes.
  3. ``HierarchicalSequenceStageModel`` combines fast ``SparseFeatureActionIndex``
     (Interpolated AD) with Bayes/lift conditioning on the active high-level
     stage node ``S_t`` and learns macro stage-to-stage transitions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparse_index import SparseFeatureActionIndex
from summarizer import StageRegimeSummarizer


class OnlineSymbolFeatureMap:
    """Self-supervised low-dim distributional features for discrete symbols."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 24,
        window: int = 6,
        nnz_per_symbol: int = 4,
        seed: int = 0,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.dim = max(int(dim), 4)
        self.window = max(int(window), 1)
        self.nnz_per_symbol = max(1, min(int(nnz_per_symbol), self.dim))
        self.seed = int(seed)
        self.embeddings_: Optional[np.ndarray] = None
        self.salience_: Optional[np.ndarray] = None
        self.global_probs_: Optional[np.ndarray] = None

    def fit(self, tokens: Sequence[int]) -> "OnlineSymbolFeatureMap":
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        n = len(arr)
        if n < 2:
            raise ValueError("need at least 2 symbols to fit OnlineSymbolFeatureMap")
        v = self.vocab_size
        counts = np.bincount(np.clip(arr, 0, v - 1), minlength=v).astype(float)
        # Inverse-frequency salience (zero hard-coded stopwords)
        sal = np.log((float(n) + 1.0) / (counts + 1.0))
        sal[0] = 0.0  # <UNK> carries no topic salience
        mean_sal = float(sal[counts > 0].mean()) if np.any(counts > 0) else 1.0
        self.salience_ = sal / max(mean_sal, 1e-6)
        self.global_probs_ = (counts + 1.0) / (float(n) + float(v))

        # Deterministic sparse random index vectors in {-1, 0, +1}^D
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        rand_idx = np.zeros((v, self.dim), dtype=np.float32)
        for sym in range(1, v):
            cols = rng.choice(self.dim, size=self.nnz_per_symbol, replace=False)
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.nnz_per_symbol)
            rand_idx[sym, cols] = signs

        # Vectorized window co-occurrence accumulation: O(window * N * D)
        weighted_seq = rand_idx[np.clip(arr, 0, v - 1)] * self.salience_[np.clip(arr, 0, v - 1)][:, None].astype(np.float32)
        csum = np.zeros((n + 1, self.dim), dtype=np.float64)
        np.cumsum(weighted_seq, axis=0, out=csum[1:])
        w = self.window
        idx_arr = np.arange(n)
        lo = np.maximum(0, idx_arr - w)
        hi = np.minimum(n, idx_arr + w + 1)
        ctx_vec = (csum[hi] - csum[lo]) - weighted_seq

        emb = np.zeros((v, self.dim), dtype=np.float64)
        np.add.at(emb, np.clip(arr, 0, v - 1), ctx_vec)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = np.where(norms > 1e-8, emb / np.where(norms > 1e-8, norms, 1.0), 0.0)
        emb[0] = 0.0
        self.embeddings_ = emb.astype(np.float32)
        return self

    def encode_stream(
        self,
        tokens: Sequence[int],
        ema_alpha: float = 0.06,
    ) -> np.ndarray:
        """Compute salience-weighted slow EMA context trajectory ``(len(tokens), dim)``."""
        if self.embeddings_ is None or self.salience_ is None:
            raise RuntimeError("fit OnlineSymbolFeatureMap before encoding")
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        n = len(arr)
        out = np.empty((n, self.dim), dtype=np.float32)
        emb = self.embeddings_
        sal = self.salience_
        base_a = float(ema_alpha)
        h = np.zeros(self.dim, dtype=np.float32)
        for i in range(n):
            sym = int(arr[i])
            if sym > 0:
                a = min(base_a * float(sal[sym]), 0.5)
                h = (1.0 - a) * h + a * emb[sym]
            out[i] = h
        return out


class HierarchicalSequenceStageModel:
    """Two-level model: StageRegimeSummarizer macro stages + AD n-gram fast index."""

    def __init__(
        self,
        vocab_size: int,
        *,
        feature_dim: int = 24,
        window: int = 6,
        ema_alpha: float = 0.06,
        n_regimes: int = 16,
        min_dwell: int = 12,
        hysteresis: int = 3,
        ctx_len: int = 5,
        chunk_len: int = 400,
        seed: int = 0,
    ):
        self.vocab_size = int(vocab_size)
        self.ema_alpha = float(ema_alpha)
        self.ctx_len = int(ctx_len)
        self.chunk_len = max(int(chunk_len), 32)
        self.feature_map = OnlineSymbolFeatureMap(
            vocab_size=vocab_size, dim=feature_dim, window=window, seed=seed,
        )
        self.summarizer = StageRegimeSummarizer(
            n_regimes=int(n_regimes),
            min_dwell=int(min_dwell),
            hysteresis=int(hysteresis),
            partition="kmeans",
            strip_pure_drift=False,
        )
        self.fast_index = SparseFeatureActionIndex(
            tuple(range(vocab_size)), prior_weight=0.0,
        )
        self.stage_symbol_probs_: Optional[np.ndarray] = None
        self.stage_transitions_: Optional[np.ndarray] = None
        self.stage_counts_: Optional[np.ndarray] = None
        self.avg_dwell_: float = 0.0

    def fit(
        self,
        tokens: Sequence[int],
        *,
        max_stage_samples: int = 200000,
    ) -> "HierarchicalSequenceStageModel":
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        # 1. Fit fast character-level Interpolated AD index
        self.fast_index.fit_token_stream(arr, ctx_len=self.ctx_len)
        # 2. Fit online distributional symbol features
        self.feature_map.fit(arr)
        # 3. Encode slow EMA trajectory (bounded prefix for fast clustering)
        stage_arr = arr[: min(len(arr), int(max_stage_samples))]
        h_seq = self.feature_map.encode_stream(stage_arr, ema_alpha=self.ema_alpha)
        # Slice into episodes for StageRegimeSummarizer
        episodes = []
        cl = self.chunk_len
        for start in range(0, len(h_seq) - cl, cl):
            seg_h = h_seq[start : start + cl]
            episodes.append(dict(
                observations=seg_h,
                actions=np.zeros(len(seg_h) - 1, dtype=int),
                feedback=np.zeros((len(seg_h) - 1, 2), dtype=float),
                terminal=False,
            ))
        if not episodes:
            episodes.append(dict(
                observations=h_seq,
                actions=np.zeros(len(h_seq) - 1, dtype=int),
                feedback=np.zeros((len(h_seq) - 1, 2), dtype=float),
                terminal=False,
            ))
        self.summarizer.fit_episodes(episodes)

        # 4. Compress chunks via StageRegimeSummarizer -> learn per-stage distributions & macro transitions
        k = len(self.summarizer.centers_)
        stage_sym_counts = np.ones((k, self.vocab_size), dtype=float)
        stage_trans = np.zeros((k, k), dtype=float)
        dwell_lengths: List[int] = []

        # Assign stages across full encoded stream in batches + hysteresis/dwell stats on episodes
        for ep_idx, ep in enumerate(episodes):
            stages = self.summarizer.segment_episode(
                episode=ep_idx,
                observations=ep["observations"],
                actions=ep["actions"],
                feedback=ep["feedback"],
            )
            for s_i, st in enumerate(stages):
                dwell_lengths.append(int(st.duration))
                if s_i + 1 < len(stages):
                    stage_trans[int(st.regime), int(stages[s_i + 1].regime)] += 1.0

        # Vectorized stage assignment over stage_arr to collect per-stage symbol counts
        z = self.summarizer._standardize(h_seq)
        c_norm = np.einsum("ij,ij->i", self.summarizer.centers_, self.summarizer.centers_)[None, :]
        c_t_neg2 = np.ascontiguousarray(-2.0 * self.summarizer.centers_.T)
        z_norm = np.einsum("ij,ij->i", z, z)[:, None]
        d2 = z @ c_t_neg2 + z_norm + c_norm
        regimes = np.argmin(d2, axis=1)
        np.add.at(stage_sym_counts, (regimes, np.clip(stage_arr, 0, self.vocab_size - 1)), 1.0)

        self.stage_counts_ = stage_sym_counts.sum(axis=1)
        self.stage_symbol_probs_ = stage_sym_counts / self.stage_counts_[:, None]
        self.stage_transitions_ = stage_trans
        self.avg_dwell_ = float(np.mean(dwell_lengths)) if dwell_lengths else 0.0
        return self

    def resolve_stage(self, h_vec: np.ndarray) -> int:
        info = self.summarizer.resolve(np.asarray(h_vec, dtype=float))
        return int(info["regime"])

    def update_slow_state(self, h_vec: np.ndarray, sym: int) -> np.ndarray:
        sym_i = int(sym)
        if sym_i <= 0 or sym_i >= self.vocab_size:
            return h_vec
        a = min(self.ema_alpha * float(self.feature_map.salience_[sym_i]), 0.5)
        return ((1.0 - a) * h_vec + a * self.feature_map.embeddings_[sym_i]).astype(np.float32)

    def propose_with_stage(
        self,
        ctx_tokens: Sequence[int],
        h_vec: np.ndarray,
        *,
        k: int = 8,
        stage_beta: float = 0.5,
        pool_k: int = 32,
    ) -> Tuple[List[int], np.ndarray, int]:
        """Fuse fast Interpolated AD with Bayes lift from active high-level stage node."""
        c = [int(x) for x in ctx_tokens[-self.ctx_len :]]
        keys: List[Tuple[int, ...]] = [(-1, c[-1])]
        for order in range(2, len(c) + 1):
            keys.append((10 + order,) + tuple(c[-order:]))

        cand, ad_probs = self.fast_index.propose_with_probs(
            None, k=max(int(pool_k), int(k)), sparse_keys=keys,
        )
        stage_id = self.resolve_stage(h_vec)
        if float(stage_beta) <= 0.0 or self.stage_symbol_probs_ is None:
            return cand[:k], ad_probs[:k], stage_id

        st_probs = self.stage_symbol_probs_[stage_id]
        gl_probs = self.feature_map.global_probs_
        beta = float(stage_beta)
        fused = np.empty(len(cand), dtype=float)
        for i, a in enumerate(cand):
            if a == 0:
                fused[i] = ad_probs[i]
            else:
                lift = float(st_probs[a]) / max(float(gl_probs[a]), 1e-8)
                fused[i] = float(ad_probs[i]) * (lift ** beta)
        order_idx = np.argsort(-fused)[:k]
        top_cand = [cand[int(j)] for j in order_idx]
        top_probs = fused[order_idx]
        return top_cand, top_probs, stage_id

    def top_stage_symbols(
        self, stage_id: int, itos: Sequence[str], top_n: int = 12,
    ) -> List[str]:
        """Inspect most distinctive symbols (by lift) compressed into ``stage_id``."""
        if self.stage_symbol_probs_ is None or self.feature_map.global_probs_ is None:
            return []
        st_p = self.stage_symbol_probs_[int(stage_id)]
        gl_p = self.feature_map.global_probs_
        # Require minimum stage probability so ultra-rare noise chars don't dominate
        score = np.where(st_p > 5e-4, st_p / np.maximum(gl_p, 1e-8), 0.0)
        score[0] = 0.0
        idxs = np.argsort(-score)[:top_n]
        return [itos[int(i)] for i in idxs if score[int(i)] > 1.0]
