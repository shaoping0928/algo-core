"""Task-agnostic Temporary Context (Working Memory) mechanism.

Provides an episode-scoped fast working memory that adapts online within the
active context without modifying long-term weights:
  1. **Long-range associative spreading field**: from training data, learns
     sparse high-PMI long-range symbol associations (window +/- 25 steps,
     weighted by self-supervised inverse-frequency salience). During inference,
     high-salience anchors observed in the active context spread activation
     into a live working-memory field ``context_field[sym]``.
  2. **Episodic local cache**: records exact transitions ``(prev -> sym)`` and
     salient entities encountered earlier inside the *current* passage/episode
     (with exponential recency decay), enabling immediate online adaptation to
     novel entities/names/terms appearing in unseen held-out streams.
  3. **Zero task-specific assumptions**: operates solely on discrete symbol IDs
     and resets cleanly at episode boundaries via ``reset()``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class TemporaryContextMemory:
    """Episode-scoped working memory combining associative spreading + episodic cache."""

    def __init__(
        self,
        vocab_size: int,
        *,
        assoc_window: int = 24,
        top_assoc_per_sym: int = 32,
        min_cooc: int = 3,
        anchor_decay: float = 0.97,
        cache_decay: float = 0.992,
        max_anchors: int = 24,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.assoc_window = max(int(assoc_window), 2)
        self.top_assoc_per_sym = max(int(top_assoc_per_sym), 4)
        self.min_cooc = max(int(min_cooc), 1)
        self.anchor_decay = float(anchor_decay)
        self.cache_decay = float(cache_decay)
        self.max_anchors = max(int(max_anchors), 4)

        self.salience_: Optional[np.ndarray] = None
        self.assoc_graph_: Dict[int, List[Tuple[int, float]]] = {}
        self._salience_cutoff: float = 0.0

        # Temporary episode state (reset per passage/episode)
        self._anchors: Dict[int, float] = {}
        self._ep_unigram: Dict[int, float] = {}
        self._ep_bigram: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._ep_trigram: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
        self._field: np.ndarray = np.zeros(self.vocab_size, dtype=float)
        self._root_field: np.ndarray = np.zeros(self.vocab_size, dtype=float)
        self._recent: List[int] = []
        self._steps: int = 0

    def fit_associations(
        self,
        tokens: Sequence[int],
        *,
        stride: int = 2,
        max_tokens: int = 1000000,
    ) -> "TemporaryContextMemory":
        """Build sparse high-PMI long-range associative graph from a symbol stream."""
        arr = np.clip(
            np.asarray(tokens[: max(int(max_tokens), 2)], dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        n = len(arr)
        v = self.vocab_size
        counts = np.bincount(arr, minlength=v).astype(float)
        sal = np.log((float(n) + 1.0) / (counts + 1.0))
        sal[0] = 0.0
        nz_sal = sal[counts > 0]
        mean_sal = float(nz_sal.mean()) if len(nz_sal) > 0 else 1.0
        self.salience_ = sal / max(mean_sal, 1e-6)
        # Only track high-salience content symbols as anchors (top ~65% salience, skips ubiquitous glue)
        self._salience_cutoff = float(np.quantile(self.salience_[counts > 0], 0.15)) if len(nz_sal) > 0 else 0.5
        # Specifically skip top-40 ultra-frequent skeleton tokens from spreading noise
        top_freq_cut = np.argsort(-counts)[: min(40, max(v // 10, 1))]
        is_content = np.ones(v, dtype=bool)
        is_content[0] = False
        is_content[top_freq_cut] = False
        self._is_content = is_content

        # Fast strided window co-occurrence over content symbols only
        cooc: Dict[int, Counter] = defaultdict(Counter)
        w = self.assoc_window
        st = max(int(stride), 1)
        content_pos = np.flatnonzero(is_content[arr])
        # Subsample centers by stride for fast O(N) build
        centers = content_pos[::st]
        for idx_i in range(len(centers)):
            pos = int(centers[idx_i])
            a = int(arr[pos])
            lo = max(0, pos - w)
            hi = min(n, pos + w + 1)
            win = arr[lo:hi]
            # Count unique content neighbors in this window (document/paragraph co-occurrence)
            uniq_nb = np.unique(win[is_content[win]])
            ca = cooc[a]
            for b in uniq_nb:
                bi = int(b)
                if bi != a:
                    ca[bi] += 1

        # Convert to sparse normalized positive-PMI edges
        p_global = (counts + 1.0) / (float(n) + float(v))
        self.assoc_graph_ = {}
        for a, cmap in cooc.items():
            a_tot = float(sum(cmap.values()))
            if a_tot < float(self.min_cooc):
                continue
            scored = []
            for b, cnt in cmap.items():
                if cnt < self.min_cooc:
                    continue
                pmi = float(np.log((float(cnt) / a_tot) / max(float(p_global[b]), 1e-12)))
                if pmi > 0.5:
                    scored.append((pmi * float(np.log1p(cnt)), int(b), pmi))
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[: self.top_assoc_per_sym]
            max_w = max(w_val for w_val, _, _ in top)
            self.assoc_graph_[a] = [(b, float(w_val / max(max_w, 1e-8))) for w_val, b, _ in top]
        return self

    def reset(self) -> None:
        """Clear all temporary episode state."""
        self._anchors.clear()
        self._ep_unigram.clear()
        self._ep_bigram.clear()
        self._ep_trigram.clear()
        self._field.fill(0.0)
        self._root_field.fill(0.0)
        self._recent.clear()
        self._steps = 0

    def pin_root_context(self) -> None:
        """Snapshot the current active field as persistent episode root summary."""
        np.copyto(self._root_field, self._field)

    def effective_field(self, root_blend: float = 0.45) -> np.ndarray:
        """Return live decaying working-memory field blended with pinned root summary."""
        r_max = float(self._root_field.max())
        if r_max <= 1e-6:
            return self._field
        l_max = float(self._field.max())
        rb = float(np.clip(root_blend, 0.0, 1.0))
        live_norm = self._field / l_max if l_max > 1e-6 else self._field
        root_norm = self._root_field / r_max
        return (1.0 - rb) * live_norm + rb * root_norm

    def observe(self, symbol: int) -> None:
        """Ingest one observed symbol into temporary working memory."""
        sym = int(symbol)
        if sym <= 0 or sym >= self.vocab_size:
            self._recent.append(0)
            self._steps += 1
            return

        # 1. Decay temporary episodic cache & anchor field
        ad = self.anchor_decay
        cd = self.cache_decay
        self._field *= ad
        if self._anchors:
            for k in list(self._anchors.keys()):
                self._anchors[k] *= ad
                if self._anchors[k] < 1e-3:
                    del self._anchors[k]

        # 2. Update episodic local cache (within-context unigram, bigram, trigram)
        if len(self._recent) >= 1 and self._recent[-1] > 0:
            p1 = self._recent[-1]
            bmap = self._ep_bigram[p1]
            for k in list(bmap.keys()):
                bmap[k] *= cd
            bmap[sym] = bmap.get(sym, 0.0) + 1.0
        if len(self._recent) >= 2 and self._recent[-1] > 0 and self._recent[-2] > 0:
            key2 = (self._recent[-2], self._recent[-1])
            tmap = self._ep_trigram[key2]
            for k in list(tmap.keys()):
                tmap[k] *= cd
            tmap[sym] = tmap.get(sym, 0.0) + 1.0
        self._ep_unigram[sym] = self._ep_unigram.get(sym, 0.0) * cd + 1.0

        # 3. Spread activation if sym is a salient content anchor
        is_c = getattr(self, "_is_content", None)
        sal = float(self.salience_[sym]) if self.salience_ is not None else 1.0
        if is_c is None or bool(is_c[sym]):
            self._anchors[sym] = min(self._anchors.get(sym, 0.0) + sal, 5.0)
            self._field[sym] += 0.5 * sal
            for nb, w in self.assoc_graph_.get(sym, ()):
                self._field[nb] += w * sal
            # Prune anchors to bounded capacity
            if len(self._anchors) > self.max_anchors:
                keep = sorted(self._anchors.items(), key=lambda kv: kv[1], reverse=True)[: self.max_anchors]
                self._anchors = dict(keep)

        self._recent.append(sym)
        if len(self._recent) > 8:
            self._recent = self._recent[-8:]
        self._steps += 1

    def episodic_cache_probs(self, ctx_tokens: Sequence[int]) -> Dict[int, float]:
        """Return online probability distribution from the current episode's local cache."""
        probs: Dict[int, float] = {}
        # Prefer exact episodic trigram -> episodic bigram -> episodic salient unigram
        if len(ctx_tokens) >= 2:
            key2 = (int(ctx_tokens[-2]), int(ctx_tokens[-1]))
            tmap = self._ep_trigram.get(key2)
            if tmap:
                tot = float(sum(tmap.values()))
                if tot > 0.0:
                    for s, w in tmap.items():
                        probs[s] = probs.get(s, 0.0) + 0.65 * (w / tot)
        if len(ctx_tokens) >= 1:
            p1 = int(ctx_tokens[-1])
            bmap = self._ep_bigram.get(p1)
            if bmap:
                tot = float(sum(bmap.values()))
                if tot > 0.0:
                    for s, w in bmap.items():
                        probs[s] = probs.get(s, 0.0) + 0.35 * (w / tot)
        return probs

    def apply_to_candidates(
        self,
        ctx_tokens: Sequence[int],
        candidates: Sequence[int],
        base_probs: np.ndarray,
        *,
        field_beta: float = 0.6,
        cache_lambda: float = 0.25,
        extra_pool_k: int = 8,
    ) -> Tuple[List[int], np.ndarray]:
        """Re-rank & augment base candidates using active temporary working memory.

        1. Injects high-probability episodic cache matches (novel entities/terms
           seen earlier in this exact passage) into the candidate pool.
        2. Modulates candidate probabilities by the spreading-activation
           working-memory field ``self._field[sym]`` (keeping output grounded
           in the current context's active semantic field).
        """
        c_list = [int(a) for a in candidates]
        p_map: Dict[int, float] = {int(a): float(p) for a, p in zip(c_list, base_probs)}

        # Inject episodic local cache probabilities (interpolated with cache_lambda)
        ep_probs = self.episodic_cache_probs(ctx_tokens)
        cl = float(cache_lambda) if ep_probs else 0.0
        if cl > 0.0:
            for a in list(p_map.keys()):
                p_map[a] *= (1.0 - cl)
            for s, ep_p in ep_probs.items():
                p_map[s] = p_map.get(s, 0.0) + cl * ep_p

        # Modulate candidate probabilities using live + pinned root working-memory field
        eff_field = self.effective_field(root_blend=0.45)
        f_max = float(eff_field.max())
        if float(field_beta) > 0.0 and f_max > 1e-6:
            inv_f = 1.0 / f_max
            fb = float(field_beta)
            for a in list(p_map.keys()):
                if a > 0:
                    act_norm = float(eff_field[a]) * inv_f
                    p_map[a] *= float(np.exp(fb * act_norm))

        ranked = sorted(p_map.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[: max(len(c_list), extra_pool_k)]
        acts = [a for a, _ in ranked]
        probs = np.asarray([p for _, p in ranked], dtype=float)
        p_sum = float(probs.sum())
        if p_sum > 0.0:
            probs /= p_sum
        return acts, probs

    def active_field_top(self, itos: Sequence[str], top_n: int = 12) -> List[Tuple[str, float]]:
        """Inspect the top currently activated symbols in temporary working memory."""
        f_max = float(self._field.max())
        if f_max <= 1e-6:
            return []
        idxs = np.argsort(-self._field)[:top_n]
        return [(itos[int(i)], float(self._field[int(i)] / f_max)) for i in idxs if self._field[int(i)] > 0.0]
