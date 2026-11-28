"""Task-agnostic dynamic progression features and hierarchical stage-chain compression.

Compresses long discrete symbol streams (e.g., pure character sequences without
any word segmenter) into chains of variable-duration macro stage nodes using
``StageRegimeSummarizer``:
  1. ``OnlineSymbolFeatureMap`` learns low-dimensional distributional symbol
     vectors via single-pass sparse Random Indexing weighted by self-supervised
     inverse-frequency salience.
  2. ``encode_progression_stream`` extracts dynamic process signals at each step:
     - fast-vs-slow divergence ``||h_fast - h_slow||`` (local shift velocity)
     - displacement from episode entry ``||h_slow - h_entry||`` (progress)
     - local self-supervised prediction surprise ``-log2 P_AD(s_t | ctx)``
     - semantic novelty ``1 - cos(E[s_t], h_slow_{t-1})``
     plus directional shift projections, so ``StageRegimeSummarizer`` discovers
     *process/discourse stages* (opening, elaboration, transition, novelty spike)
     rather than merely static topic clusters.
  3. ``CompressedStageNode`` summarizes each compressed span by
     ``(regime, start, end, duration, entry_state, mean_state, delta_state,
     mean_surprise, mean_novelty, key_symbols)``.
  4. ``HierarchicalSequenceStageModel`` learns macro stage-to-stage transitions
     ``P(S_{m+1} | S_m)`` and uses the target stage's progression direction to
     guide candidate selection on top of fast Interpolated AD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparse_index import SparseFeatureActionIndex
from summarizer import StageRegimeSummarizer


@dataclass(frozen=True)
class CompressedStageNode:
    """One compressed high-level stage node summarizing a span of raw symbols."""

    regime: int
    start: int
    end: int
    duration: int
    entry_state: np.ndarray
    mean_state: np.ndarray
    delta_state: np.ndarray
    mean_surprise: float
    mean_novelty: float
    key_symbols: Tuple[int, ...]


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
        clipped = np.clip(arr, 0, v - 1)
        counts = np.bincount(clipped, minlength=v).astype(float)
        # Self-supervised inverse-frequency salience (zero hard-coded stopwords)
        sal = np.log((float(n) + 1.0) / (counts + 1.0))
        sal[0] = 0.0
        mean_sal = float(sal[counts > 0].mean()) if np.any(counts > 0) else 1.0
        self.salience_ = sal / max(mean_sal, 1e-6)
        self.global_probs_ = (counts + 1.0) / (float(n) + float(v))

        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        rand_idx = np.zeros((v, self.dim), dtype=np.float32)
        for sym in range(1, v):
            cols = rng.choice(self.dim, size=self.nnz_per_symbol, replace=False)
            signs = rng.choice(
                np.array([-1.0, 1.0], dtype=np.float32),
                size=self.nnz_per_symbol,
            )
            rand_idx[sym, cols] = signs

        weighted_seq = rand_idx[clipped] * self.salience_[clipped][:, None].astype(np.float32)
        csum = np.zeros((n + 1, self.dim), dtype=np.float64)
        np.cumsum(weighted_seq, axis=0, out=csum[1:])
        w = self.window
        idx_arr = np.arange(n)
        lo = np.maximum(0, idx_arr - w)
        hi = np.minimum(n, idx_arr + w + 1)
        ctx_vec = (csum[hi] - csum[lo]) - weighted_seq

        emb = np.zeros((v, self.dim), dtype=np.float64)
        np.add.at(emb, clipped, ctx_vec)
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

    def encode_progression_stream(
        self,
        tokens: Sequence[int],
        *,
        surprises: Optional[np.ndarray] = None,
        alpha_fast: float = 0.25,
        alpha_slow: float = 0.04,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute dynamic progression features ``(obs, h_slow, novelty)`` for stage summarization.

        Each row of ``obs`` has shape ``(4 + min(8, dim),)`` containing:
          - ``[0]`` smoothed local prediction surprise (boundary/predictability)
          - ``[1]`` fast-vs-slow semantic shift magnitude ``||h_fast - h_slow||``
          - ``[2]`` progress displacement from chunk start ``||h_slow - h_entry||``
          - ``[3]`` smoothed symbol novelty ``1 - cos(E[s_t], h_slow_{t-1})``
          - ``[4:]`` compact directional shift projection ``(h_fast - h_slow)[:p]``
        """
        if self.embeddings_ is None or self.salience_ is None:
            raise RuntimeError("fit OnlineSymbolFeatureMap before encoding")
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        n = len(arr)
        p_dim = min(8, self.dim)
        obs = np.empty((n, 4 + p_dim), dtype=np.float32)
        h_slow_all = np.empty((n, self.dim), dtype=np.float32)
        nov_all = np.empty(n, dtype=np.float32)

        emb = self.embeddings_
        sal = self.salience_
        af = float(alpha_fast)
        asl = float(alpha_slow)
        h_fast = np.zeros(self.dim, dtype=np.float32)
        h_slow = np.zeros(self.dim, dtype=np.float32)
        h_entry: Optional[np.ndarray] = None
        surp_ema = 0.0
        nov_ema = 0.5

        for i in range(n):
            sym = int(arr[i])
            if sym > 0:
                e = emb[sym]
                h_norm = float(np.linalg.norm(h_slow))
                if h_norm > 1e-6:
                    cos_sim = float(np.dot(e, h_slow)) / h_norm
                    nov = float(np.clip(1.0 - cos_sim, 0.0, 2.0))
                else:
                    nov = 1.0
                s_w = float(sal[sym])
                a_f = min(af * s_w, 0.6)
                a_s = min(asl * s_w, 0.35)
                h_fast = (1.0 - a_f) * h_fast + a_f * e
                h_slow = (1.0 - a_s) * h_slow + a_s * e
                if h_entry is None:
                    h_entry = h_slow.copy()
            else:
                nov = 0.5
            s_val = float(surprises[i]) if surprises is not None else 0.0
            surp_ema = 0.8 * surp_ema + 0.2 * s_val
            nov_ema = 0.85 * nov_ema + 0.15 * nov
            diff = h_fast - h_slow
            shift_mag = float(np.linalg.norm(diff))
            prog_mag = float(np.linalg.norm(h_slow - h_entry)) if h_entry is not None else 0.0

            obs[i, 0] = surp_ema
            obs[i, 1] = shift_mag
            obs[i, 2] = prog_mag
            obs[i, 3] = nov_ema
            obs[i, 4:] = diff[:p_dim]
            h_slow_all[i] = h_slow
            nov_all[i] = nov_ema

        return obs, h_slow_all, nov_all


class HierarchicalSequenceStageModel:
    """Two-level stage-summarization model: dynamic macro stage chains + fast AD index."""

    def __init__(
        self,
        vocab_size: int,
        *,
        feature_dim: int = 24,
        window: int = 6,
        ema_alpha: float = 0.04,
        n_regimes: int = 12,
        min_dwell: int = 16,
        hysteresis: int = 4,
        event_quantile: float = 0.99,
        ctx_len: int = 5,
        chunk_len: int = 400,
        seed: int = 0,
    ):
        self.vocab_size = int(vocab_size)
        self.ema_alpha = float(ema_alpha)
        self.ctx_len = int(ctx_len)
        self.chunk_len = max(int(chunk_len), 48)
        self.feature_map = OnlineSymbolFeatureMap(
            vocab_size=vocab_size, dim=feature_dim, window=window, seed=seed,
        )
        self.summarizer = StageRegimeSummarizer(
            n_regimes=int(n_regimes),
            min_dwell=int(min_dwell),
            hysteresis=int(hysteresis),
            event_quantile=float(event_quantile),
            partition="kmeans",
            strip_pure_drift=False,
        )
        self.fast_index = SparseFeatureActionIndex(
            tuple(range(vocab_size)), prior_weight=0.0,
        )
        self.stage_symbol_probs_: Optional[np.ndarray] = None
        self.stage_transitions_: Optional[np.ndarray] = None
        self.stage_trans_probs_: Optional[np.ndarray] = None
        self.stage_counts_: Optional[np.ndarray] = None
        self.stage_profiles_: Dict[int, Dict[str, float]] = {}
        self.stage_mean_delta_: Optional[np.ndarray] = None
        self.avg_dwell_: float = 0.0

    def _compute_local_surprises(self, tokens: np.ndarray) -> np.ndarray:
        """Compute fast self-supervised character surprise ``-log2 P_1gram(s_t | s_{t-1})``."""
        n = len(tokens)
        surp = np.zeros(n, dtype=np.float32)
        postings = self.fast_index._postings
        totals = self.fast_index._key_totals
        for i in range(1, n):
            fk = (-1, int(tokens[i - 1]))
            tot = totals.get(fk, 0.0)
            if tot > 0.0:
                bucket = postings[fk].get(int(tokens[i]))
                if bucket is None:
                    cnt = 0.0
                elif isinstance(bucket, list):
                    cnt = float(bucket[0])
                else:
                    cnt = float(bucket)
                p = (cnt + 0.5) / (tot + 1.0)
            else:
                p = 1.0 / float(self.vocab_size)
            surp[i] = float(-np.log2(max(p, 1e-6)))
        return surp

    def _top_key_symbols(self, span_tokens: np.ndarray, top_n: int = 3) -> Tuple[int, ...]:
        """Extract the highest-salience unique symbols inside a compressed span."""
        sal = self.feature_map.salience_
        if sal is None or len(span_tokens) == 0:
            return ()
        uniq = np.unique(span_tokens[span_tokens > 0])
        if len(uniq) == 0:
            return ()
        order = np.argsort(-sal[uniq])[:top_n]
        return tuple(int(uniq[int(j)]) for j in order)

    def compress_to_stage_chain(
        self,
        tokens: Sequence[int],
    ) -> Tuple[List[CompressedStageNode], np.ndarray, np.ndarray]:
        """Compress a raw symbol sequence into a chain of ``CompressedStageNode`` objects."""
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        surp = self._compute_local_surprises(arr)
        obs, h_slow, nov = self.feature_map.encode_progression_stream(
            arr, surprises=surp, alpha_slow=self.ema_alpha,
        )
        stages = self.summarizer.segment_episode(
            episode=0,
            observations=obs,
            actions=np.zeros(max(len(obs) - 1, 0), dtype=int),
            feedback=np.zeros((max(len(obs) - 1, 0), 2), dtype=float),
        )
        chain: List[CompressedStageNode] = []
        for st in stages:
            s0 = int(st.start)
            s1 = min(int(st.end), len(arr) - 1)
            span_tok = arr[s0 : s1 + 1]
            chain.append(CompressedStageNode(
                regime=int(st.regime),
                start=s0,
                end=s1,
                duration=int(st.duration),
                entry_state=h_slow[s0].copy(),
                mean_state=h_slow[s0 : s1 + 1].mean(axis=0),
                delta_state=(h_slow[s1] - h_slow[s0]).astype(np.float32),
                mean_surprise=float(surp[s0 : s1 + 1].mean()),
                mean_novelty=float(nov[s0 : s1 + 1].mean()),
                key_symbols=self._top_key_symbols(span_tok, top_n=3),
            ))
        return chain, obs, h_slow

    def fit(
        self,
        tokens: Sequence[int],
        *,
        max_stage_samples: int = 200000,
    ) -> "HierarchicalSequenceStageModel":
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        # 1. Fit fast character-level Interpolated AD index
        self.fast_index.fit_token_stream(arr, ctx_len=self.ctx_len)
        # 2. Fit self-supervised distributional symbol features
        self.feature_map.fit(arr)
        # 3. Build dynamic progression trajectories chunk-by-chunk for StageRegimeSummarizer
        stage_arr = arr[: min(len(arr), int(max_stage_samples))]
        cl = self.chunk_len
        episodes = []
        chunk_slices: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
        for start in range(0, len(stage_arr) - cl, cl):
            seg = stage_arr[start : start + cl]
            surp = self._compute_local_surprises(seg)
            obs, h_slow, nov = self.feature_map.encode_progression_stream(
                seg, surprises=surp, alpha_slow=self.ema_alpha,
            )
            episodes.append(dict(
                observations=obs,
                actions=np.zeros(len(obs) - 1, dtype=int),
                feedback=np.zeros((len(obs) - 1, 2), dtype=float),
                terminal=False,
            ))
            chunk_slices.append((start, start + cl, surp, nov))
        if not episodes:
            surp = self._compute_local_surprises(stage_arr)
            obs, _h_slow, nov = self.feature_map.encode_progression_stream(
                stage_arr, surprises=surp, alpha_slow=self.ema_alpha,
            )
            episodes.append(dict(
                observations=obs,
                actions=np.zeros(len(obs) - 1, dtype=int),
                feedback=np.zeros((len(obs) - 1, 2), dtype=float),
                terminal=False,
            ))
            chunk_slices.append((0, len(stage_arr), surp, nov))
        self.summarizer.fit_episodes(episodes)

        # 4. Compress chunks into macro stage chains & learn stage transitions + progression profiles
        k = len(self.summarizer.centers_)
        stage_sym_counts = np.ones((k, self.vocab_size), dtype=float)
        stage_trans = np.zeros((k, k), dtype=float)
        surp_sum = np.zeros(k, dtype=float)
        nov_sum = np.zeros(k, dtype=float)
        shift_sum = np.zeros(k, dtype=float)
        prog_sum = np.zeros(k, dtype=float)
        seg_count = np.zeros(k, dtype=float)
        dwell_sum = np.zeros(k, dtype=float)
        dwell_lengths: List[int] = []

        for ep_idx, (ep, (c0, c1, surp, nov)) in enumerate(zip(episodes, chunk_slices)):
            stages = self.summarizer.segment_episode(
                episode=ep_idx,
                observations=ep["observations"],
                actions=ep["actions"],
                feedback=ep["feedback"],
            )
            seg_tok = stage_arr[c0:c1]
            ep_obs = ep["observations"]
            for s_i, st in enumerate(stages):
                r = int(st.regime)
                s0 = int(st.start)
                s1 = min(int(st.end), len(seg_tok) - 1)
                dur = int(st.duration)
                dwell_lengths.append(dur)
                seg_count[r] += 1.0
                dwell_sum[r] += float(dur)
                surp_sum[r] += float(surp[s0 : s1 + 1].mean())
                nov_sum[r] += float(nov[s0 : s1 + 1].mean())
                shift_sum[r] += float(ep_obs[s0 : s1 + 1, 1].mean())
                prog_sum[r] += float(ep_obs[s0 : s1 + 1, 2].mean())
                np.add.at(
                    stage_sym_counts[r],
                    np.clip(seg_tok[s0 : s1 + 1], 0, self.vocab_size - 1),
                    1.0,
                )
                if s_i + 1 < len(stages):
                    stage_trans[r, int(stages[s_i + 1].regime)] += 1.0

        self.stage_counts_ = stage_sym_counts.sum(axis=1)
        self.stage_symbol_probs_ = stage_sym_counts / self.stage_counts_[:, None]
        self.stage_transitions_ = stage_trans
        row_sums = stage_trans.sum(axis=1, keepdims=True)
        self.stage_trans_probs_ = (stage_trans + 0.1) / (row_sums + 0.1 * float(k))
        self.avg_dwell_ = float(np.mean(dwell_lengths)) if dwell_lengths else 0.0
        self.stage_profiles_ = {}
        for r in range(k):
            denom = max(seg_count[r], 1.0)
            self.stage_profiles_[r] = dict(
                segments=int(seg_count[r]),
                avg_dwell=float(dwell_sum[r] / denom),
                mean_surprise=float(surp_sum[r] / denom),
                mean_novelty=float(nov_sum[r] / denom),
                mean_shift=float(shift_sum[r] / denom),
                mean_progress=float(prog_sum[r] / denom),
            )
        return self

    def predict_next_stage(self, current_regime: int) -> Tuple[int, float]:
        """Predict most likely next macro stage node from ``stage_trans_probs_``."""
        if self.stage_trans_probs_ is None:
            raise RuntimeError("fit HierarchicalSequenceStageModel before predicting")
        probs = self.stage_trans_probs_[int(current_regime)]
        nxt = int(np.argmax(probs))
        return nxt, float(probs[nxt])

    def update_slow_state(self, h_vec: np.ndarray, sym: int) -> np.ndarray:
        sym_i = int(sym)
        if sym_i <= 0 or sym_i >= self.vocab_size:
            return h_vec
        a = min(self.ema_alpha * float(self.feature_map.salience_[sym_i]), 0.35)
        return ((1.0 - a) * h_vec + a * self.feature_map.embeddings_[sym_i]).astype(np.float32)

    def propose_with_stage(
        self,
        ctx_tokens: Sequence[int],
        obs_vec: np.ndarray,
        *,
        k: int = 8,
        stage_beta: float = 0.25,
        pool_k: int = 32,
    ) -> Tuple[List[int], np.ndarray, int]:
        """Fuse fast Interpolated AD with active dynamic stage node conditioning."""
        c = [int(x) for x in ctx_tokens[-self.ctx_len :]]
        keys: List[Tuple[int, ...]] = [(-1, c[-1])]
        for order in range(2, len(c) + 1):
            keys.append((10 + order,) + tuple(c[-order:]))

        cand, ad_probs = self.fast_index.propose_with_probs(
            None, k=max(int(pool_k), int(k)), sparse_keys=keys,
        )
        stage_id = int(self.summarizer.resolve(np.asarray(obs_vec, dtype=float))["regime"])
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
