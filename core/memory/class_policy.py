"""Task-agnostic feature/class-level sequence policy (no verbatim surface rails).

Instead of relying on brittle high-order surface n-grams (which only work when
exact memorized training spans recur and otherwise cause verbatim rail-copying),
this module learns:
  1. Self-discovered symbol classes ``C(s) in [0, n_classes)`` by clustering
     each symbol's distributional co-occurrence vector ``E[s]`` (zero word
     segmenter, zero hard-coded POS tags).
  2. A stable class-level transition policy ``P(C_t | C_{t-k}..C_{t-1})`` via
     Interpolated Absolute Discounting over abstract class sequences.
  3. Context-conditioned intra-class emission ``P(s_t | C_t, s_slow, local_1_2gram)``
     combining class membership, slow semantic alignment, and minimal local
     character-pairing legality (<=2-gram), preventing 4/5-gram verbatim copying.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sequence_stage import OnlineSymbolFeatureMap
from sparse_index import SparseFeatureActionIndex
from summarizer import _fit_deterministic_centroids


class ClassLevelSequencePolicy:
    """Feature/class-level transition policy with self-discovered symbol classes."""

    def __init__(
        self,
        vocab_size: int,
        *,
        n_classes: int = 128,
        n_singletons: int = 64,
        feature_dim: int = 32,
        window: int = 4,
        class_ctx_len: int = 4,
        local_ctx_len: int = 2,
        ema_alpha: float = 0.06,
        seed: int = 0,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.n_singletons = max(0, min(int(n_singletons), self.vocab_size - 2))
        self.n_content_classes = max(int(n_classes), 4)
        self.n_classes = 1 + self.n_singletons + self.n_content_classes
        self.class_ctx_len = max(int(class_ctx_len), 1)
        self.local_ctx_len = max(int(local_ctx_len), 1)
        self.ema_alpha = float(ema_alpha)
        self.seed = int(seed)

        self.feature_map = OnlineSymbolFeatureMap(
            vocab_size=vocab_size, dim=feature_dim, window=window, seed=seed,
        )
        self.class_index = SparseFeatureActionIndex(
            tuple(range(self.n_classes)), prior_weight=0.0,
        )
        # Only short local (<=2-gram) pairing for basic character-pair legality,
        # intentionally excluding 4/5-gram verbatim surface rails.
        self.local_index = SparseFeatureActionIndex(
            tuple(range(self.vocab_size)), prior_weight=0.0,
        )
        self.sym_to_class_: Optional[np.ndarray] = None
        self.class_centers_: Optional[np.ndarray] = None
        self.class_members_: Dict[int, np.ndarray] = {}
        self.class_emit_probs_: Dict[int, np.ndarray] = {}

    def fit(
        self,
        tokens: Sequence[int],
        *,
        n_workers: int = 1,
        min_high_order_count: int = 1,
    ) -> "ClassLevelSequencePolicy":
        arr = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1,
        )
        # 1. Learn distributional features for every symbol + mean-centered isotropic view
        #    (removes shared embedding-cone anisotropy so cosine discriminates topics!)
        self.feature_map.fit(arr)
        emb = self.feature_map.embeddings_
        counts = np.bincount(arr, minlength=self.vocab_size).astype(float)
        valid_e = (np.arange(self.vocab_size) > 0) & (counts > 0)
        if np.any(valid_e):
            self.emb_mean_ = emb[valid_e].mean(axis=0).astype(np.float32)
        else:
            self.emb_mean_ = np.zeros(emb.shape[1], dtype=np.float32)
        ec = emb.astype(np.float32) - self.emb_mean_[None, :]
        ec_n = np.maximum(np.linalg.norm(ec, axis=1, keepdims=True), 1e-8)
        self.emb_centered_norm_ = (ec / ec_n).astype(np.float32)
        self.emb_centered_norm_[0] = 0.0

        # 2. Data-driven split: top-frequency skeleton symbols become singleton
        #    classes so they do not form a giant black-hole cluster; remaining
        #    content symbols cluster into semantic classes via embeddings.
        valid_syms = np.flatnonzero((np.arange(self.vocab_size) > 0) & (counts > 0))
        freq_order = valid_syms[np.argsort(-counts[valid_syms])]
        n_sing = min(self.n_singletons, max(len(freq_order) - 2, 0))
        singleton_syms = freq_order[:n_sing]
        content_syms = freq_order[n_sing:]

        sym_to_cls = np.zeros(self.vocab_size, dtype=np.int32)
        for idx_s, sym in enumerate(singleton_syms):
            sym_to_cls[int(sym)] = idx_s + 1  # 1..n_sing

        k_cls = min(self.n_content_classes, max(len(content_syms), 1))
        if len(content_syms) > 0:
            centers = _fit_deterministic_centroids(
                emb[content_syms].astype(float), k_cls, max_iter=35,
            )
            self.class_centers_ = centers.astype(np.float32)
            c_norm = np.einsum("ij,ij->i", centers, centers)[None, :]
            e_cont = emb[content_syms].astype(float)
            e_norm = np.einsum("ij,ij->i", e_cont, e_cont)[:, None]
            d2 = e_cont @ (-2.0 * centers.T) + e_norm + c_norm
            assigned = np.argmin(d2, axis=1) + 1 + n_sing
            sym_to_cls[content_syms] = assigned.astype(np.int32)
        else:
            self.class_centers_ = np.zeros((0, emb.shape[1]), dtype=np.float32)

        self.sym_to_class_ = sym_to_cls

        # Store per-class member list and smoothed intra-class emission P(s | C)
        self.class_members_ = {}
        self.class_emit_probs_ = {}
        max_cid = int(sym_to_cls.max())
        for cid in range(1, max_cid + 1):
            members = np.flatnonzero(sym_to_cls == cid)
            if len(members) == 0:
                continue
            m_counts = counts[members] + 0.5
            self.class_members_[cid] = members.astype(np.int32)
            self.class_emit_probs_[cid] = (m_counts / float(m_counts.sum())).astype(float)

        # 3. Fit class-level transition policy P(C_t | C_{t-4}..C_{t-1})
        class_stream = sym_to_cls[arr]
        self.class_index.fit_token_stream(
            class_stream, ctx_len=self.class_ctx_len,
            n_workers=n_workers, min_high_order_count=min_high_order_count,
        )

        # 4. Fit short local pairing index (<=2-gram only, no 4/5-gram surface rails)
        self.local_index.fit_token_stream(
            arr, ctx_len=self.local_ctx_len,
            n_workers=n_workers, min_high_order_count=min_high_order_count,
        )
        return self

    def ingest_chunk(
        self,
        tokens: Sequence[int],
        *,
        n_workers: int = 4,
        min_high_order_count: int = 2,
    ) -> "ClassLevelSequencePolicy":
        """Incrementally ingest a new stream chunk into class & local transition indices."""
        if self.sym_to_class_ is None:
            return self.fit(tokens, n_workers=n_workers)
        arr = np.clip(np.asarray(tokens, dtype=np.int32).reshape(-1), 0, self.vocab_size - 1)
        if len(arr) <= self.class_ctx_len:
            return self
        class_stream = self.sym_to_class_[arr]
        self.class_index.fit_token_stream(
            class_stream, ctx_len=self.class_ctx_len,
            n_workers=n_workers, min_high_order_count=min_high_order_count,
        )
        self.local_index.fit_token_stream(
            arr, ctx_len=self.local_ctx_len,
            n_workers=n_workers, min_high_order_count=min_high_order_count,
        )
        return self

    def prune_under_pressure(self, pressure: float) -> int:
        """Prune sparse transition postings under memory pressure."""
        ev = self.class_index.prune_under_pressure(pressure)
        ev += self.local_index.prune_under_pressure(pressure * 0.75)
        return ev

    def predict_next_classes(
        self,
        ctx_tokens: Sequence[int],
        *,
        top_c: int = 8,
    ) -> Tuple[List[int], np.ndarray]:
        """Predict next abstract symbol classes from recent class sequence."""
        if self.sym_to_class_ is None:
            raise RuntimeError("fit ClassLevelSequencePolicy before predicting")
        c_toks = [int(self.sym_to_class_[min(max(int(x), 0), self.vocab_size - 1)]) for x in ctx_tokens[-self.class_ctx_len :]]
        keys: List[Tuple[int, ...]] = [(-1, c_toks[-1])]
        for order in range(2, len(c_toks) + 1):
            keys.append((10 + order,) + tuple(c_toks[-order:]))
        return self.class_index.propose_with_probs(None, k=top_c, sparse_keys=keys)

    def propose_next_symbols(
        self,
        ctx_tokens: Sequence[int],
        slow_state: Optional[np.ndarray] = None,
        *,
        k: int = 8,
        top_c: int = 12,
        local_weight: float = 0.5,
        semantic_gamma: float = 1.5,
    ) -> Tuple[List[int], np.ndarray]:
        """Predict next symbols via class-level policy * intra-class context emission.

        Combines:
          - Class-level structural transition: ``P(C_t | C_{t-4}..C_{t-1})``
          - Semantic-state aligned intra-class emission: ``P(s | C_t) * exp(gamma * cos(E[s], slow_state))``
          - Optional short local pairing legality (<=2-gram): ``P_local(s | s_{t-2}, s_{t-1})``
        """
        if self.sym_to_class_ is None or self.feature_map.embeddings_ is None:
            raise RuntimeError("fit ClassLevelSequencePolicy before predicting")
        cls_ids, cls_probs = self.predict_next_classes(ctx_tokens, top_c=top_c)
        emb = getattr(self, "emb_centered_norm_", None)
        if emb is None:
            emb = self.feature_map.embeddings_

        # Precompute normalized slow-state direction for semantic emission
        g_u = None
        if slow_state is not None:
            gn = float(np.linalg.norm(slow_state))
            if gn > 1e-6:
                g_u = (np.asarray(slow_state, dtype=float) / gn)

        # Short local (<=2-gram) pairing distribution for basic character-pair legality
        local_map: Dict[int, float] = {}
        if float(local_weight) > 0.0:
            lc = [int(x) for x in ctx_tokens[-self.local_ctx_len :]]
            lkeys: List[Tuple[int, ...]] = [(-1, lc[-1])]
            for order in range(2, len(lc) + 1):
                lkeys.append((10 + order,) + tuple(lc[-order:]))
            lcand, lprobs = self.local_index.propose_with_probs(None, k=64, sparse_keys=lkeys)
            for a, p in zip(lcand, lprobs):
                if a > 0 and p > 0.0:
                    local_map[int(a)] = float(p)

        # Score candidate symbols strictly through class-level policy
        sym_scores: Dict[int, float] = {}
        lw = float(local_weight)
        gam = float(semantic_gamma)
        for cid, cp in zip(cls_ids, cls_probs):
            if cid <= 0 or cp <= 0.0 or cid not in self.class_members_:
                continue
            members = self.class_members_[cid]
            emit_p = self.class_emit_probs_[cid]
            if g_u is not None and gam > 0.0:
                sims = emb[members] @ g_u
                weights = emit_p * np.exp(gam * sims)
                w_sum = float(weights.sum())
                if w_sum > 0.0:
                    weights /= w_sum
            else:
                weights = emit_p
            # Keep top members per active class for fast sparse scoring
            if len(members) > 16:
                top_m = np.argpartition(-weights, 15)[:16]
            else:
                top_m = np.arange(len(members))
            for idx_m in top_m:
                sym = int(members[int(idx_m)])
                p_cls_sym = float(cp) * float(weights[int(idx_m)])
                if lw > 0.0:
                    p_loc = local_map.get(sym, 1e-4)
                    # Geometric interpolation: class policy + local pairing legality
                    score = (p_cls_sym ** (1.0 - lw)) * (p_loc ** lw)
                else:
                    score = p_cls_sym
                if score > sym_scores.get(sym, 0.0):
                    sym_scores[sym] = score

        ranked = sorted(sym_scores.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[:k]
        if not ranked:
            return [1], np.array([1.0], dtype=float)
        acts = [a for a, _ in ranked]
        probs = np.asarray([p for _, p in ranked], dtype=float)
        p_sum = float(probs.sum())
        if p_sum > 0.0:
            probs /= p_sum
        return acts, probs

    def update_slow_state(self, h_vec: np.ndarray, sym: int) -> np.ndarray:
        sym_i = int(sym)
        if sym_i <= 0 or sym_i >= self.vocab_size:
            return h_vec
        a = min(self.ema_alpha * float(self.feature_map.salience_[sym_i]), 0.4)
        e_mat = getattr(self, "emb_centered_norm_", None)
        if e_mat is None:
            e_mat = self.feature_map.embeddings_
        return ((1.0 - a) * h_vec + a * e_mat[sym_i]).astype(np.float32)

    def inspect_classes(self, itos: Sequence[str], n_show: int = 12, top_m: int = 12) -> List[Tuple[int, int, List[str]]]:
        """Return ``[(class_id, size, top_member_chars)]`` sorted by class size."""
        out = []
        for cid, members in self.class_members_.items():
            ep = self.class_emit_probs_[cid]
            order = np.argsort(-ep)[:top_m]
            chars = [itos[int(members[int(j)])] for j in order]
            out.append((cid, len(members), chars))
        out.sort(key=lambda item: item[1], reverse=True)
        return out[:n_show]
