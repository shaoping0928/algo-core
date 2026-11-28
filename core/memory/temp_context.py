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
        max_compounds: int = 1500,
        min_compound_pmi: float = 2.5,
        corroboration_singleton_scale: float = 0.22,
        max_recent: int = 96,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.assoc_window = max(int(assoc_window), 2)
        self.top_assoc_per_sym = max(int(top_assoc_per_sym), 4)
        self.min_cooc = max(int(min_cooc), 1)
        self.anchor_decay = float(anchor_decay)
        self.cache_decay = float(cache_decay)
        self.max_anchors = max(int(max_anchors), 4)
        self.max_compounds = max(int(max_compounds), 0)
        self.min_compound_pmi = float(min_compound_pmi)
        self.corroboration_singleton_scale = float(np.clip(corroboration_singleton_scale, 0.0, 1.0))
        self.max_recent = max(int(max_recent), 16)

        self.salience_: Optional[np.ndarray] = None
        self.assoc_graph_: Dict[int, List[Tuple[int, float]]] = {}
        self.assoc_graph_compound_: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        self._salience_cutoff: float = 0.0

        # Temporary episode state (reset per passage/episode)
        self._anchors: Dict[Tuple[int, ...], float] = {}
        self._ep_unigram: Dict[int, float] = {}
        self._ep_bigram: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._ep_trigram: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
        self._direct_field: np.ndarray = np.zeros(self.vocab_size, dtype=float)
        self._field: np.ndarray = np.zeros(self.vocab_size, dtype=float)
        self._root_field: np.ndarray = np.zeros(self.vocab_size, dtype=float)
        self._recent: List[int] = []
        self._steps: int = 0
        self._last_source: str = "external"
        self._field_dirty: bool = False
        self.circuit_field = None

    def enable_circuit_field(
        self,
        *,
        use_gpu: bool = True,
        fit_tokens: Optional[Sequence[int]] = None,
        dt: float = 0.25,
        init_resistance: float = 4.5,
        init_capacitance: float = 2.2,
        alpha_int: float = 0.12,
        w_cap: float = 0.55,
        w_int: float = 0.15,
        w_diff: float = 0.50,
    ):
        """Attach a GPU/CPU ``ElectronicCircuitNodeField`` (R/C/Integrator/Differentiator)."""
        from circuit_field import ElectronicCircuitNodeField

        cf = ElectronicCircuitNodeField(
            n_nodes=self.vocab_size,
            batch_size=1,
            dt=dt,
            init_resistance=init_resistance,
            init_capacitance=init_capacitance,
            alpha_int=alpha_int,
            w_cap=w_cap,
            w_int=w_int,
            w_diff=w_diff,
            use_gpu=use_gpu,
        )
        if self.assoc_graph_:
            cf.set_sparse_coupling(self.assoc_graph_)
        if fit_tokens is not None:
            cf.fit_rc_from_stream(fit_tokens)
        self.circuit_field = cf
        return cf

    def circuit_differentiator(self) -> Optional[np.ndarray]:
        """Return high-pass differentiator potential ``V_diff = V_cap - V_int`` if enabled."""
        cf = getattr(self, "circuit_field", None)
        if cf is None:
            return None
        return cf.V_diff[0]

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

        # Data-driven high-PMI 2-symbol compound chunk discovery (zero task/language rules)
        self.assoc_graph_compound_ = {}
        if self.max_compounds > 0 and n >= 4:
            both_c = is_content[arr[:-1]] & is_content[arr[1:]]
            pair_pos = np.flatnonzero(both_c)
            if len(pair_pos) > 0:
                a_p = arr[pair_pos].astype(np.int64)
                b_p = arr[pair_pos + 1].astype(np.int64)
                codes = a_p * int(v) + b_p
                uniq_codes, code_cnts = np.unique(codes, return_counts=True)
                min_pair_cnt = max(self.min_cooc * 2, 6)
                keep_mask = code_cnts >= min_pair_cnt
                cand_pairs = []
                inv_n = 1.0 / max(float(n - 1), 1.0)
                for code_val, cnt_val in zip(uniq_codes[keep_mask].tolist(), code_cnts[keep_mask].tolist()):
                    ai = int(code_val // v)
                    bi = int(code_val % v)
                    if ai == bi:
                        continue
                    pmi_ab = float(np.log((float(cnt_val) * inv_n) / max(float(p_global[ai] * p_global[bi]), 1e-14)))
                    if pmi_ab >= self.min_compound_pmi:
                        cand_pairs.append((pmi_ab * float(np.log1p(cnt_val)), (ai, bi)))
                cand_pairs.sort(key=lambda item: item[0], reverse=True)
                top_compounds = set(pair for _, pair in cand_pairs[: self.max_compounds])
                if top_compounds:
                    comp_cooc: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
                    # Scan occurrences of discovered compound pairs (strided for speed)
                    for p_idx in pair_pos[::st]:
                        pos = int(p_idx)
                        pair_k = (int(arr[pos]), int(arr[pos + 1]))
                        if pair_k not in top_compounds:
                            continue
                        lo = max(0, pos - w)
                        hi = min(n, pos + w + 2)
                        win = arr[lo:hi]
                        uniq_nb = np.unique(win[is_content[win]])
                        ck = comp_cooc[pair_k]
                        a0, b0 = pair_k
                        for nb in uniq_nb:
                            ni = int(nb)
                            if ni != a0 and ni != b0:
                                ck[ni] += 1
                    for pair_k, cmap in comp_cooc.items():
                        c_tot = float(sum(cmap.values()))
                        if c_tot < float(self.min_cooc):
                            continue
                        sc_list = []
                        for nb, cnt in cmap.items():
                            if cnt < self.min_cooc:
                                continue
                            pmi_nb = float(np.log((float(cnt) / c_tot) / max(float(p_global[nb]), 1e-12)))
                            if pmi_nb > 0.5:
                                sc_list.append((pmi_nb * float(np.log1p(cnt)), int(nb)))
                        if not sc_list:
                            continue
                        sc_list.sort(key=lambda item: item[0], reverse=True)
                        top_nb = sc_list[: self.top_assoc_per_sym]
                        mw = max(wv for wv, _ in top_nb)
                        self.assoc_graph_compound_[pair_k] = [
                            (nb, float(wv / max(mw, 1e-8))) for wv, nb in top_nb
                        ]
        return self

    @property
    def recent_history(self) -> List[int]:
        return self._recent

    def _rebuild_corroborated_field(self) -> None:
        """Compute live spreading field with multi-anchor corroboration (POMDP disambiguation)."""
        self._field.fill(0.0)
        # Direct sensory observations always carry unattenuated self-evidence
        np.copyto(self._field, self._direct_field)
        if not self._anchors:
            self._field_dirty = False
            return

        spread_sum: Dict[int, float] = {}
        spread_sup: Dict[int, int] = {}
        n_anchors = len(self._anchors)
        for anchor_key, a_weight in self._anchors.items():
            if len(anchor_key) == 2:
                edges = self.assoc_graph_compound_.get(anchor_key, ())
            else:
                edges = self.assoc_graph_.get(anchor_key[0], ())
            for nb, edge_w in edges:
                spread_sum[nb] = spread_sum.get(nb, 0.0) + float(a_weight) * float(edge_w)
                spread_sup[nb] = spread_sup.get(nb, 0) + 1

        attenuate_single = (n_anchors >= 3)
        s_scale = self.corroboration_singleton_scale
        for nb, val in spread_sum.items():
            sup = spread_sup.get(nb, 1)
            if attenuate_single and sup <= 1 and self._direct_field[nb] <= 1e-8:
                # Single lone cue with zero corroboration from rest of context -> attenuate
                self._field[nb] += val * s_scale
            elif sup >= 2:
                # Corroborated by multiple distinct cues -> full strength + mild consensus boost
                bonus = min(1.0 + 0.15 * float(sup - 2), 1.45)
                self._field[nb] += val * bonus
            else:
                self._field[nb] += val
        self._field_dirty = False

    def reset(self) -> None:
        """Clear all temporary episode state."""
        self._anchors.clear()
        self._ep_unigram.clear()
        self._ep_bigram.clear()
        self._ep_trigram.clear()
        self._direct_field.fill(0.0)
        self._field.fill(0.0)
        self._root_field.fill(0.0)
        self._recent.clear()
        self._steps = 0
        self._last_source = "external"
        self._field_dirty = False
        cf = getattr(self, "circuit_field", None)
        if cf is not None:
            cf.reset()

    def pin_root_context(self) -> None:
        """Snapshot the current active field as persistent episode root summary."""
        if self._field_dirty:
            self._rebuild_corroborated_field()
        np.copyto(self._root_field, self._field)

    def pin_stage_anchors(self, anchor_weights: Dict[int, float], *, spread_scale: float = 0.5) -> None:
        """Merge persistent hierarchical stage-tree anchor slots into ``_root_field``."""
        if not anchor_weights:
            return
        max_w = max((float(v) for v in anchor_weights.values() if float(v) > 0.0), default=0.0)
        if max_w <= 1e-8:
            return
        base_scale = max(float(self._root_field.max()), float(self._field.max()), 1.0)
        for sym, w in anchor_weights.items():
            s = int(sym)
            if s <= 0 or s >= self.vocab_size or float(w) <= 0.0:
                continue
            wn = (float(w) / max_w) * base_scale
            self._root_field[s] += 0.6 * wn
            if spread_scale > 0.0:
                for nb, edge_w in self.assoc_graph_.get(s, ()):
                    self._root_field[nb] += float(spread_scale) * float(edge_w) * wn

    def effective_field(self, root_blend: float = 0.45) -> np.ndarray:
        """Return live decaying working-memory field blended with pinned root summary."""
        if self._field_dirty:
            self._rebuild_corroborated_field()
        r_max = float(self._root_field.max())
        if r_max <= 1e-6:
            return self._field
        l_max = float(self._field.max())
        rb = float(np.clip(root_blend, 0.0, 1.0))
        live_norm = self._field / l_max if l_max > 1e-6 else self._field
        root_norm = self._root_field / r_max
        blended = (1.0 - rb) * live_norm + rb * root_norm
        cf = getattr(self, "circuit_field", None)
        if cf is not None:
            c_pos = np.maximum(cf.V_out[0], 0.0)
            c_max = float(c_pos.max())
            if c_max > 1e-6:
                blended = 0.65 * blended + 0.35 * (c_pos / c_max)
        return blended

    def observe(
        self,
        symbol: int,
        *,
        source: str = "external",
        self_spread_weight: float = 0.25,
    ) -> None:
        """Ingest one observed symbol into temporary working memory.

        Args:
            symbol: Observed discrete symbol ID.
            source: ``"external"`` for real input/ground-truth observations
                (updates episodic cache + full spreading activation), or
                ``"self"`` for self-generated rollouts/continuations (auto-pins
                the external root summary on transition, skips episodic-cache
                self-echo reinforcement, and spreads damped activation).
            self_spread_weight: Damping multiplier for spreading activation
                when ``source == "self"``.
        """
        is_self = (source == "self")
        if is_self and self._last_source != "self" and float(self._root_field.max()) <= 1e-6:
            # Automatically preserve the external prompt's working-memory summary
            self.pin_root_context()
        self._last_source = "self" if is_self else "external"

        sym = int(symbol)
        if sym <= 0 or sym >= self.vocab_size:
            self._recent.append(0)
            if len(self._recent) > self.max_recent:
                self._recent = self._recent[-self.max_recent :]
            self._steps += 1
            return

        # 1. Decay temporary episodic cache & anchor field
        ad = self.anchor_decay
        cd = self.cache_decay
        self._direct_field *= ad
        if self._anchors:
            for k in list(self._anchors.keys()):
                self._anchors[k] *= ad
                if self._anchors[k] < 1e-3:
                    del self._anchors[k]
        self._field_dirty = True

        # 2. Update episodic local cache ONLY from verified external observations
        #    (prevents self-generated tokens from creating episodic echo loops)
        prev1 = self._recent[-1] if len(self._recent) >= 1 else 0
        if not is_self:
            if prev1 > 0:
                bmap = self._ep_bigram[prev1]
                for k in list(bmap.keys()):
                    bmap[k] *= cd
                bmap[sym] = bmap.get(sym, 0.0) + 1.0
            if len(self._recent) >= 2 and prev1 > 0 and self._recent[-2] > 0:
                key2 = (self._recent[-2], prev1)
                tmap = self._ep_trigram[key2]
                for k in list(tmap.keys()):
                    tmap[k] *= cd
                tmap[sym] = tmap.get(sym, 0.0) + 1.0
            self._ep_unigram[sym] = self._ep_unigram.get(sym, 0.0) * cd + 1.0

        # 3. Register salient content anchor (prefer discovered high-PMI 2-symbol compound!)
        is_c = getattr(self, "_is_content", None)
        sal = float(self.salience_[sym]) if self.salience_ is not None else 1.0
        spread_mul = float(np.clip(self_spread_weight, 0.0, 1.0)) if is_self else 1.0
        if (is_c is None or bool(is_c[sym])) and spread_mul > 0.0:
            self._direct_field[sym] += 0.55 * spread_mul * sal
            pair_k = (prev1, sym) if prev1 > 0 else None
            if pair_k is not None and pair_k in self.assoc_graph_compound_:
                # Replace/upgrade isolated prev1 cue with its tight compound chunk (prev1, sym)
                prev_sal = float(self.salience_[prev1]) if self.salience_ is not None else 1.0
                comp_sal = 0.5 * (prev_sal + sal) * 1.25
                self._anchors.pop((prev1,), None)
                self._anchors[pair_k] = min(self._anchors.get(pair_k, 0.0) + spread_mul * comp_sal, 6.0)
            else:
                self._anchors[(sym,)] = min(self._anchors.get((sym,), 0.0) + spread_mul * sal, 5.0)
            # Prune anchors to bounded capacity
            if len(self._anchors) > self.max_anchors:
                keep = sorted(self._anchors.items(), key=lambda kv: kv[1], reverse=True)[: self.max_anchors]
                self._anchors = dict(keep)

            # Advance electronic circuit node field (R/C/Integrator/Differentiator)
            cf = getattr(self, "circuit_field", None)
            if cf is not None:
                edges = (
                    self.assoc_graph_compound_[pair_k]
                    if (pair_k is not None and pair_k in self.assoc_graph_compound_)
                    else self.assoc_graph_.get(sym, ())
                )
                cf.step_single_event(sym, sal, is_self=is_self, assoc_neighbors=edges)

        self._recent.append(sym)
        if len(self._recent) > self.max_recent:
            self._recent = self._recent[-self.max_recent :]
        self._steps += 1
        self._rebuild_corroborated_field()

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
        novelty_gated_cache: bool = False,
    ) -> Tuple[List[int], np.ndarray]:
        """Re-rank & augment base candidates using active temporary working memory.

        1. Injects high-probability episodic cache matches (novel entities/terms
           seen earlier in this exact passage) into the candidate pool. When
           ``novelty_gated_cache=True``, episodic cache mass is weighted by
           long-term surprise ``(1 - min(4 * p_base, 1))`` so novel episode
           bindings get full recall while already-known transitions are not
           double-counted alongside a recency-lag kernel.
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
            if novelty_gated_cache:
                added_mass = 0.0
                gated_ep: Dict[int, float] = {}
                for s, ep_p in ep_probs.items():
                    base_p = p_map.get(s, 0.0)
                    nov = max(1.0 - min(4.0 * base_p, 1.0), 0.0)
                    w_add = cl * float(ep_p) * nov
                    if w_add > 1e-8:
                        gated_ep[s] = w_add
                        added_mass += w_add
                if added_mass > 0.0:
                    scale_down = max(1.0 - min(added_mass, cl), 0.5)
                    for a in list(p_map.keys()):
                        p_map[a] *= scale_down
                    for s, w_add in gated_ep.items():
                        p_map[s] = p_map.get(s, 0.0) + w_add
            else:
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
        if self._field_dirty:
            self._rebuild_corroborated_field()
        f_max = float(self._field.max())
        if f_max <= 1e-6:
            return []
        idxs = np.argsort(-self._field)[:top_n]
        return [(itos[int(i)], float(self._field[int(i)] / f_max)) for i in idxs if self._field[int(i)] > 0.0]

    def prune_under_pressure(self, pressure: float) -> int:
        """Accelerate forgetting and evict low-weight episodic/associative entries under memory pressure."""
        p = float(np.clip(pressure, 0.0, 1.0))
        if p <= 0.0:
            return 0
        evicted = 0
        # 1. Stronger decay / anchor pruning as pressure rises
        keep_anchors = max(int(round(self.max_anchors * (1.0 - 0.65 * p))), 4)
        if len(self._anchors) > keep_anchors:
            evicted += len(self._anchors) - keep_anchors
            top_a = sorted(self._anchors.items(), key=lambda kv: kv[1], reverse=True)[:keep_anchors]
            self._anchors = dict(top_a)
            self._field_dirty = True

        # 2. Prune weak episodic bigram/trigram cache entries
        min_ep_w = 0.25 + 1.75 * p
        for bmap_owner in (self._ep_bigram, self._ep_trigram):
            for key in list(bmap_owner.keys()):
                inner = bmap_owner[key]
                for s in list(inner.keys()):
                    if inner[s] < min_ep_w:
                        del inner[s]
                        evicted += 1
                if not inner:
                    del bmap_owner[key]

        # 3. Under high pressure (>=0.5), trim tail of long-term associative edges & compounds
        if p >= 0.50:
            keep_edges = max(int(round(self.top_assoc_per_sym * (1.0 - 0.55 * p))), 4)
            for a in list(self.assoc_graph_.keys()):
                edges = self.assoc_graph_[a]
                if len(edges) > keep_edges:
                    evicted += len(edges) - keep_edges
                    self.assoc_graph_[a] = edges[:keep_edges]
            if p >= 0.70 and len(self.assoc_graph_compound_) > 128:
                keep_c = max(int(len(self.assoc_graph_compound_) * (1.0 - 0.5 * p)), 128)
                keys = list(self.assoc_graph_compound_.keys())
                for k in keys[keep_c:]:
                    del self.assoc_graph_compound_[k]
                    evicted += 1
        return evicted
