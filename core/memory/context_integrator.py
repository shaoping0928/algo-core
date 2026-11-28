"""Task-Agnostic Multi-Cue Long-Range Contextual Integrator.

Replaces short-horizon exponential-decay bag-of-symbols context with true
multi-anchor contextual integration over hierarchical structural units (chunks
+ salient atoms):

  1. **Directional Long-Range Contextual PMI Graph (W = 64 units)**:
     Learns forward contextual log-likelihood ratios
     ``LLR_long(target_unit | cue_unit) = log(P(target within next W | cue) / P(target))``
     directly over discovered structural chunks (eliminating single-symbol
     homograph ambiguity) and supports fast streaming incremental updates.
  2. **Persistent Episode Context Slot Bank (Zero Short-Horizon Amnesia)**:
     Maintains salient multi-scale context anchors across the entire episode
     weighted by self-information (IDF salience) rather than fast exponential
     decay (e.g. ``0.85^40 = 0.0015``), so early-prompt concepts stay active.
  3. **Multi-Cue Bayesian Intersection (Product-of-Experts)**:
     Computes joint contextual support across all active episode slots with
     super-linear corroboration when ``>= 2`` distinct context anchors agree on
     a candidate unit, suppressing lone-cue associative noise.
  4. **Realization / Coverage Adaptation**:
     Tracks which contextually predicted units have already been expressed in
     open-loop self-generation, gently shifting residual contextual drive to
     unexpressed high-intersection targets so rollouts progress through a
     coherent train of thought rather than looping or drifting off-topic.

Completely task-agnostic: operates strictly on integer unit/symbol ID streams
with zero language rules, zero word segmenters, and zero repetition bans.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ContextualIntegrator:
    """Multi-cue Bayesian long-range contextual integrator over structural units."""

    def __init__(
        self,
        n_units: int,
        vocab_size: int,
        *,
        forward_window: int = 48,
        top_targets_per_cue: int = 48,
        min_cooc: int = 4,
        min_pmi: float = 0.85,
        max_episode_slots: int = 40,
        slot_decay: float = 0.992,
        realize_decay: float = 0.96,
        singleton_scale: float = 0.28,
    ):
        self.n_units = max(int(n_units), 2)
        self.vocab_size = max(int(vocab_size), 2)
        self.forward_window = max(int(forward_window), 8)
        self.top_targets_per_cue = max(int(top_targets_per_cue), 8)
        self.min_cooc = max(int(min_cooc), 2)
        self.min_pmi = float(min_pmi)
        self.max_episode_slots = max(int(max_episode_slots), 8)
        self.slot_decay = float(np.clip(slot_decay, 0.90, 1.0))
        self.realize_decay = float(np.clip(realize_decay, 0.80, 0.999))
        self.singleton_scale = float(np.clip(singleton_scale, 0.05, 1.0))

        # Learned unit statistics & directional forward PMI graph:
        #   cue_unit_id -> [(target_unit_id, llr_weight), ...]
        self.unit_salience_: Optional[np.ndarray] = None
        self.is_salient_: Optional[np.ndarray] = None
        self.forward_graph_: Dict[int, List[Tuple[int, float]]] = {}
        self.unit_entry_atom_: np.ndarray = np.arange(self.n_units, dtype=np.int32)
        self.unit_len_: np.ndarray = np.ones(self.n_units, dtype=np.int32)
        self.unit_atoms_: Dict[int, Tuple[int, ...]] = {}

        # Accumulators for streaming incremental ingestion
        self._unit_counts: np.ndarray = np.zeros(self.n_units, dtype=np.float64)
        self._total_positions: float = 0.0
        self._cooc_counts: Dict[int, Counter] = defaultdict(Counter)
        self._cue_windows: np.ndarray = np.zeros(self.n_units, dtype=np.float64)
        self._fitted: bool = False

        # Live episode state (reset per episode)
        self._slots: Dict[int, float] = {}            # persistent salient episode slots
        self._slot_age: Dict[int, int] = {}           # step index when slot was last seen
        self._realized: Dict[int, float] = {}         # realization count in self-mode
        self._last_ext_atom: int = 0                  # for upgrading prefix atom -> completed chunk
        self._mismatch_streak: int = 0                # empirical topic-shift detector in continuous streams
        self._committed_atoms: Tuple[int, ...] = ()   # active whole-chunk commitment (prevents half-chunk splices!)
        self._commit_pos: int = 0
        self._commit_strength: float = 0.0
        self._step: int = 0
        self._dirty: bool = True
        self._cached_unit_scores: Dict[int, float] = {}
        self._cached_unit_support: Dict[int, int] = {}
        self._cached_atom_scores: Dict[int, float] = {}
        self._cached_atom_top_unit: Dict[int, int] = {}

    def _configure_unit_metadata(
        self,
        chunk_to_atoms: Optional[Dict[int, Tuple[int, ...]]] = None,
    ) -> None:
        entry = np.zeros(self.n_units, dtype=np.int32)
        ulen = np.ones(self.n_units, dtype=np.int32)
        for u in range(min(self.vocab_size, self.n_units)):
            entry[u] = u
            ulen[u] = 1
        self.unit_atoms_ = {}
        if chunk_to_atoms is not None:
            for cid, atoms in chunk_to_atoms.items():
                ci = int(cid)
                if 0 <= ci < self.n_units and atoms:
                    entry[ci] = int(atoms[0])
                    ulen[ci] = len(atoms)
                    if len(atoms) >= 2:
                        self.unit_atoms_[ci] = tuple(int(a) for a in atoms)
        self.unit_entry_atom_ = entry
        self.unit_len_ = ulen

    def _accumulate_stream(
        self,
        unit_stream: np.ndarray,
        *,
        stride: int = 3,
        max_centers: int = 600_000,
    ) -> None:
        """Accumulate directional forward-window chunk co-occurrences in O(N/stride)."""
        arr = np.clip(np.asarray(unit_stream, dtype=np.int32).reshape(-1), 0, self.n_units - 1)
        n = len(arr)
        if n < 8:
            return
        self._unit_counts += np.bincount(arr, minlength=self.n_units).astype(np.float64)
        self._unit_counts[0] = 0.0
        self._total_positions += float(n)

        # Scale-invariant salience mask: exclude only the ultra-frequent skeleton atoms
        # (top ~2% of active atoms, capped at 45) while keeping all content atoms & chunks.
        counts = self._unit_counts
        is_sal = np.ones(self.n_units, dtype=bool)
        is_sal[0] = False
        atom_counts = counts[: self.vocab_size]
        n_active_atoms = int(np.sum(atom_counts > 0))
        k_stop = min(45, max(n_active_atoms // 18, 0))
        if k_stop > 0:
            top_stop = np.argsort(-atom_counts)[:k_stop]
            is_sal[top_stop] = False
        # In large vocabularies, also exclude ultra-common glue chunks (>1.2% of stream)
        tot_c = max(float(counts.sum()), 1.0)
        if int(np.sum(counts > 0)) >= 120:
            is_sal[counts > 0.012 * tot_c] = False
        self.is_salient_ = is_sal

        sal_pos = np.flatnonzero(is_sal[arr])
        if len(sal_pos) < 4:
            return
        eff_stride = max(int(stride), len(sal_pos) // max(int(max_centers), 1000), 1)
        centers = sal_pos[::eff_stride]

        w = self.forward_window
        cooc = self._cooc_counts
        cue_w = self._cue_windows
        for idx_c in range(len(centers)):
            pos = int(centers[idx_c])
            u = int(arr[pos])
            hi = min(n, pos + w + 1)
            if hi <= pos + 1:
                continue
            win = arr[pos + 1 : hi]
            sal_win = win[is_sal[win]]
            if len(sal_win) == 0:
                continue
            uniq_t = np.unique(sal_win)
            cue_w[u] += 1.0
            cu = cooc[u]
            for tv in uniq_t.tolist():
                if tv != u:
                    cu[tv] += 1

    def _rebuild_forward_graph(self) -> None:
        """Convert accumulated directional co-occurrences into calibrated positive-PMI LLR edges."""
        counts = self._unit_counts
        tot_c = max(float(counts.sum()), 1.0)
        p_u = (counts + 1.0) / (tot_c + float(self.n_units))
        # IDF salience in [0.5, 4.0], giving multi-symbol chunks an extra specificity bonus
        idf = -np.log(np.clip(p_u, 1e-8, 1.0))
        idf_norm = np.clip(idf / max(float(np.median(idf[counts > 0])) if np.any(counts > 0) else 1.0, 1e-6), 0.35, 3.0)
        chunk_bonus = np.where(self.unit_len_ >= 2, 1.35, 1.0)
        self.unit_salience_ = (idf_norm * chunk_bonus).astype(np.float32)
        self.unit_salience_[0] = 0.0

        # Expected occurrence rate of target v in a forward window of length W
        w_eff = float(self.forward_window)
        p_win = 1.0 - np.power(np.clip(1.0 - p_u, 1e-9, 1.0), w_eff)
        p_win = np.clip(p_win, 1e-7, 0.99)

        graph: Dict[int, List[Tuple[int, float]]] = {}
        min_c = self.min_cooc
        min_pmi = self.min_pmi
        top_k = self.top_targets_per_cue

        for u, cmap in self._cooc_counts.items():
            cw = float(self._cue_windows[u])
            if cw < float(min_c):
                continue
            inv_cw = 1.0 / cw
            scored = []
            for v, cnt in cmap.items():
                if cnt < min_c:
                    continue
                # Smoothed conditional forward-window probability P(v in next W | u)
                p_cond = (float(cnt) + 0.5) / (cw + 2.0)
                pmi = float(np.log(p_cond / float(p_win[v])))
                if pmi >= min_pmi:
                    # Reliability-damped PMI LLR (prevents rare count=4 spikes from dominating)
                    rel = float(cnt) / (float(cnt) + 6.0)
                    llr = float(np.clip(pmi * rel, 0.0, 4.5))
                    # Rank by info * log-support so both specific & reliable links are kept
                    rank_sc = llr * float(np.log1p(cnt)) * (1.25 if self.unit_len_[v] >= 2 else 1.0)
                    scored.append((rank_sc, int(v), llr))
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            graph[int(u)] = [(v, llr) for _, v, llr in scored[:top_k]]

        self.forward_graph_ = graph
        self._fitted = True

    def fit_unit_stream(
        self,
        unit_stream: Sequence[int],
        *,
        chunk_to_atoms: Optional[Dict[int, Tuple[int, ...]]] = None,
        stride: int = 3,
        max_centers: int = 600_000,
        release_accumulators: bool = False,
    ) -> "ContextualIntegrator":
        """Fit directional forward contextual PMI graph on a parsed chunk/unit stream."""
        self._configure_unit_metadata(chunk_to_atoms)
        self._accumulate_stream(
            np.asarray(unit_stream, dtype=np.int32),
            stride=stride,
            max_centers=max_centers,
        )
        self._rebuild_forward_graph()
        if release_accumulators:
            self._cooc_counts.clear()
        return self

    def ingest_unit_chunk(
        self,
        unit_stream: Sequence[int],
        *,
        stride: int = 4,
        max_centers: int = 300_000,
    ) -> "ContextualIntegrator":
        """Incrementally accumulate directional context links from a new streaming macro-chunk."""
        self._accumulate_stream(
            np.asarray(unit_stream, dtype=np.int32),
            stride=stride,
            max_centers=max_centers,
        )
        self._rebuild_forward_graph()
        return self

    def prune_under_pressure(self, pressure: float) -> int:
        """Trim raw co-occurrence accumulators and lowest-LLR graph edges under memory pressure."""
        p = float(np.clip(pressure, 0.0, 1.0))
        if p <= 0.0:
            return 0
        evicted = 0
        # Drop raw streaming accumulators first once pressure rises (graph itself is compact!)
        if p >= 0.25 and self._cooc_counts:
            for u in list(self._cooc_counts.keys()):
                evicted += len(self._cooc_counts[u])
            self._cooc_counts.clear()
        if p >= 0.50 and self.forward_graph_:
            keep_k = max(int(round(self.top_targets_per_cue * (1.0 - 0.55 * p))), 12)
            for u in list(self.forward_graph_.keys()):
                edges = self.forward_graph_[u]
                if len(edges) > keep_k:
                    evicted += len(edges) - keep_k
                    self.forward_graph_[u] = edges[:keep_k]
        return evicted

    # ------------------------------------------------------------------
    # Live episode context integration (persistent slots + Bayesian PoE)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear live episode context slots and realization state."""
        self._slots.clear()
        self._slot_age.clear()
        self._realized.clear()
        self._last_ext_atom = 0
        self._mismatch_streak = 0
        self._committed_atoms = ()
        self._commit_pos = 0
        self._commit_strength = 0.0
        self._step = 0
        self._dirty = True
        self._cached_unit_scores.clear()
        self._cached_unit_support.clear()
        self._cached_atom_scores.clear()
        self._cached_atom_top_unit.clear()

    def observe_units(
        self,
        units: Sequence[int],
        *,
        source: str = "external",
        replaced_atoms: Sequence[int] = (),
    ) -> None:
        """Update persistent episode context slots from observed structural units."""
        is_self = (source == "self")
        is_sal = self.is_salient_
        sal_arr = self.unit_salience_
        sd = self.slot_decay
        rd = self.realize_decay

        # When a newly-completed multi-symbol chunk subsumes its just-observed
        # constituent prefix atom(s), upgrade/remove those ambiguous prefix slots.
        if not is_self and replaced_atoms:
            for ra in replaced_atoms:
                rai = int(ra)
                if rai in self._slots and (self._step - self._slot_age.get(rai, -999)) <= 3:
                    del self._slots[rai]
                    self._slot_age.pop(rai, None)
                    self._dirty = True

        for raw_u in units:
            u = int(raw_u)
            self._step += 1
            if u <= 0 or u >= self.n_units:
                continue

            # Empirical topic-shift detection during continuous external reading:
            # if incoming salient units repeatedly fall outside the current context's
            # predicted multi-cue field, gently accelerate decay of stale prior-article slots.
            eff_sd = sd
            if not is_self and len(self._slots) >= 6 and (is_sal is None or bool(is_sal[u])):
                if self._dirty:
                    self._recompute_intersection()
                in_ctx = (u in self._slots) or (u in self._cached_unit_scores)
                if in_ctx:
                    self._mismatch_streak = 0
                else:
                    self._mismatch_streak += 1
                    if self._mismatch_streak >= 5:
                        eff_sd = 0.86

            # Very slow slot decay inside a coherent episode (0.992^60 = 0.62 -> zero 10-char amnesia!)
            if self._slots:
                for sk in list(self._slots.keys()):
                    self._slots[sk] *= eff_sd
                    if self._slots[sk] < 0.08:
                        del self._slots[sk]
                        self._slot_age.pop(sk, None)

            if self._realized:
                for rk in list(self._realized.keys()):
                    self._realized[rk] *= rd
                    if self._realized[rk] < 0.05:
                        del self._realized[rk]

            if is_self:
                # Self-emitted unit: mark as realized (shifts residual contextual drive
                # toward unexpressed multi-cue targets) and add only mild slot update
                # so the model does not hijack its own external topic anchor!
                self._realized[u] = min(self._realized.get(u, 0.0) + 1.0, 5.0)
                ea = int(self.unit_entry_atom_[u])
                if ea > 0 and ea != u:
                    self._realized[ea] = min(self._realized.get(ea, 0.0) + 0.65, 5.0)
                if (is_sal is None or bool(is_sal[u])) and u in self.forward_graph_:
                    s_w = float(sal_arr[u]) if sal_arr is not None else 1.0
                    self._slots[u] = min(self._slots.get(u, 0.0) + 0.14 * s_w, 4.0)
                    self._slot_age[u] = self._step
            else:
                # Verified external observation: register salient chunk/atom into persistent slots
                if is_sal is None or bool(is_sal[u]):
                    s_w = float(sal_arr[u]) if sal_arr is not None else 1.0
                    self._slots[u] = min(self._slots.get(u, 0.0) + s_w, 5.0)
                    self._slot_age[u] = self._step
                    if u < self.vocab_size:
                        self._last_ext_atom = u

            if len(self._slots) > self.max_episode_slots:
                ranked_s = sorted(self._slots.items(), key=lambda kv: kv[1], reverse=True)[
                    : self.max_episode_slots
                ]
                keep_keys = set(k for k, _ in ranked_s)
                self._slots = dict(ranked_s)
                for sk in list(self._slot_age.keys()):
                    if sk not in keep_keys:
                        del self._slot_age[sk]
            self._dirty = True

    def _recompute_intersection(self) -> None:
        """Compute Multi-Cue Bayesian Intersection across all persistent episode slots."""
        self._cached_unit_scores.clear()
        self._cached_unit_support.clear()
        self._cached_atom_scores.clear()
        self._cached_atom_top_unit.clear()
        self._dirty = False

        if not self._fitted or not self._slots:
            return

        # Normalize slot weights so total active mass is bounded & invariant to prompt length
        slot_items = list(self._slots.items())
        n_slots = len(slot_items)
        max_sw = max(w for _, w in slot_items)
        if max_sw <= 1e-8:
            return
        inv_sw = 1.0 / max_sw

        raw_llr: Dict[int, float] = {}
        sup_cnt: Dict[int, int] = {}
        fg = self.forward_graph_

        for u, sw in slot_items:
            edges = fg.get(u)
            if not edges:
                continue
            w_u = float(sw) * inv_sw
            for v, llr in edges:
                raw_llr[v] = raw_llr.get(v, 0.0) + w_u * float(llr)
                sup_cnt[v] = sup_cnt.get(v, 0) + 1

        if not raw_llr:
            return

        require_corrob = (n_slots >= 3)
        s_scale = self.singleton_scale
        u_scores: Dict[int, float] = {}
        a_scores: Dict[int, float] = {}
        a_top_u: Dict[int, int] = {}
        entry_arr = self.unit_entry_atom_
        ulen_arr = self.unit_len_
        realized = self._realized

        for v, sum_llr in raw_llr.items():
            k_sup = sup_cnt[v]
            if require_corrob and k_sup <= 1:
                # Lone-cue association attenuated strongly -> kills homograph noise!
                eff = sum_llr * s_scale
            else:
                # Multi-cue intersection: logarithmic saturation with corroboration boost
                # e.g. 2 cues -> 1.0x, 3 cues -> 1.25x, 5+ cues -> 1.6x
                corrob_mul = 1.0 + 0.25 * float(np.log1p(max(k_sup - 1, 0)))
                eff = float(np.log1p(sum_llr)) * corrob_mul
            if ulen_arr[v] >= 2:
                eff *= 1.15

            # Realization damping: once emitted in self-mode, shift drive to fresh targets
            r_v = realized.get(v, 0.0)
            if r_v > 0.0:
                eff *= float(np.exp(-0.55 * r_v))

            if eff <= 0.05:
                continue
            u_scores[v] = eff

            # Project chunk score onto its entry atom (taking max corroborated unit per atom)
            ea = int(entry_arr[v])
            if ea > 0:
                r_a = realized.get(ea, 0.0)
                eff_a = eff * float(np.exp(-0.35 * r_a)) if r_a > 0.0 else eff
                if eff_a > a_scores.get(ea, 0.0):
                    a_scores[ea] = eff_a
                    a_top_u[ea] = v

        self._cached_unit_scores = u_scores
        self._cached_unit_support = sup_cnt
        self._cached_atom_scores = a_scores
        self._cached_atom_top_unit = a_top_u

    def contextual_unit_scores(self, top_k: int = 32) -> List[Tuple[int, float, int]]:
        """Return top-K multi-cue corroborated units as ``(unit_id, score, n_supporting_slots)``."""
        if self._dirty:
            self._recompute_intersection()
        if not self._cached_unit_scores:
            return []
        ranked = sorted(self._cached_unit_scores.items(), key=lambda kv: kv[1], reverse=True)[
            : max(int(top_k), 1)
        ]
        return [(u, sc, self._cached_unit_support.get(u, 1)) for u, sc in ranked]

    def contextual_atom_llr(self) -> Dict[int, float]:
        """Return dict mapping entry atom_id -> multi-cue contextual LLR score."""
        if self._dirty:
            self._recompute_intersection()
        return self._cached_atom_scores

    def update_chunk_commitment(self, emitted_atom: int, *, is_self: bool = False) -> None:
        """Advance or initiate whole-chunk commitment when a context-promoted entry atom fires.

        Prevents half-chunk prefix splices (e.g. starting chunk ``(a, b)`` because context
        demanded it, then drifting into ``(a, x)`` on the next step). Purely task-agnostic!
        """
        a = int(emitted_atom)
        if not is_self or a <= 0:
            self._committed_atoms = ()
            self._commit_pos = 0
            self._commit_strength = 0.0
            return

        # 1. If already inside an active multi-symbol chunk commitment, advance or clear
        if self._committed_atoms and self._commit_pos < len(self._committed_atoms):
            if a == self._committed_atoms[self._commit_pos]:
                self._commit_pos += 1
                if self._commit_pos >= len(self._committed_atoms):
                    self._committed_atoms = ()
                    self._commit_pos = 0
                    self._commit_strength = 0.0
                return
            # Trajectory diverged -> release old commitment cleanly
            self._committed_atoms = ()
            self._commit_pos = 0
            self._commit_strength = 0.0

        # 2. Check if ``a`` was promoted by a high-confidence multi-symbol context chunk
        if self._dirty:
            self._recompute_intersection()
        top_u = self._cached_atom_top_unit.get(a, 0)
        llr_a = self._cached_atom_scores.get(a, 0.0)
        u_atoms_map = getattr(self, "unit_atoms_", None)
        if top_u >= self.vocab_size and llr_a >= 0.75 and u_atoms_map is not None:
            atoms = u_atoms_map.get(top_u)
            if atoms and len(atoms) >= 2 and atoms[0] == a:
                self._committed_atoms = atoms
                self._commit_pos = 1
                self._commit_strength = float(min(llr_a, 2.5))

    def committed_continuation(self) -> Tuple[int, float]:
        """Return ``(expected_next_atom, commitment_strength)`` if mid-chunk, else ``(0, 0.0)``."""
        if self._committed_atoms and self._commit_pos < len(self._committed_atoms):
            return int(self._committed_atoms[self._commit_pos]), float(self._commit_strength)
        return 0, 0.0

    def active_slots_snapshot(self, top_k: int = 16) -> List[Tuple[int, float]]:
        """Return currently retained persistent episode context slots."""
        return sorted(self._slots.items(), key=lambda kv: kv[1], reverse=True)[: max(int(top_k), 1)]
