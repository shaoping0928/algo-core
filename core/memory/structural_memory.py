"""Task-agnostic Higher-Order Structural Memory:
  1. ``HierarchicalChunkMemory`` — Recursive MDL/PMI chunk induction (macro-options /
     motifs / multi-symbol phrases) with chunk-to-chunk transition & intra-chunk
     completion memory.
  2. ``StructuralFrameMemory`` — Directional non-adjacent open->close dependency
     discovery (key->door, brackets, cause->effect frames) with an online
     pushdown working register and learned gap-timing curves.

Zero language rules, zero word segmenters, and zero task priors: both modules
operate purely on discrete symbol/action ID streams.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparse_index import SparseFeatureActionIndex
from summarizer import _fit_deterministic_centroids


class HierarchicalChunkMemory:
    """Multi-scale recursive chunk induction + abstract chunk-class grammar & skip-slot schemas."""

    def __init__(
        self,
        vocab_size: int,
        *,
        max_levels: int = 2,
        max_chunks_per_level: int = 1200,
        min_pair_count: int = 20,
        min_pmi: float = 2.0,
        max_chunk_len: int = 4,
        chunk_ctx_len: int = 2,
        enable_chunk_classes: bool = True,
        n_chunk_classes: int = 160,
        n_singleton_chunks: int = 32,
        chunk_class_ctx_len: int = 3,
        chunk_embed_dim: int = 24,
        min_schema_count: int = 8,
        seed: int = 0,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.max_levels = max(int(max_levels), 1)
        self.max_chunks_per_level = max(int(max_chunks_per_level), 16)
        self.min_pair_count = max(int(min_pair_count), 3)
        self.min_pmi = float(min_pmi)
        self.max_chunk_len = max(int(max_chunk_len), 2)
        self.chunk_ctx_len = max(int(chunk_ctx_len), 1)
        self.enable_chunk_classes = bool(enable_chunk_classes)
        self.n_chunk_classes = max(int(n_chunk_classes), 4)
        self.n_singleton_chunks = max(int(n_singleton_chunks), 0)
        self.chunk_class_ctx_len = max(int(chunk_class_ctx_len), 1)
        self.chunk_embed_dim = max(int(chunk_embed_dim), 8)
        self.min_schema_count = max(int(min_schema_count), 3)
        self.seed = int(seed)

        # Chunk IDs: 0..vocab_size-1 are atomic symbols; >= vocab_size are discovered chunks
        self.n_total_units: int = self.vocab_size
        self.chunk_to_atoms_: Dict[int, Tuple[int, ...]] = {
            i: (i,) for i in range(self.vocab_size)
        }
        self.atoms_to_chunk_: Dict[Tuple[int, ...], int] = {
            (i,): i for i in range(self.vocab_size)
        }
        self.level_merges_: List[Dict[Tuple[int, int], int]] = []
        # Prefix index: atom tuple prefix -> list of (next_atom, full_chunk_id)
        self.prefix_to_chunks_: Dict[Tuple[int, ...], List[Tuple[int, int]]] = defaultdict(list)
        self.chunk_counts_: np.ndarray = np.zeros(self.vocab_size, dtype=float)
        self.chunk_index: Optional[SparseFeatureActionIndex] = None

        # Higher-order abstract Chunk-Class Grammar (Non-Terminals) & Skip-Slot Schemas
        self.n_total_chunk_classes_: int = 0
        self.chunk_role_vecs_: Optional[np.ndarray] = None
        self.chunk_to_class_: Optional[np.ndarray] = None
        self.class_to_chunks_: Dict[int, np.ndarray] = {}
        self.class_entry_atom_probs_: Dict[int, List[Tuple[int, float]]] = {}
        self.chunk_class_index: Optional[SparseFeatureActionIndex] = None
        self.skip_schemas_: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        self._fitted: bool = False

    def _merge_stream_once(
        self,
        stream: np.ndarray,
        pair_map: Dict[Tuple[int, int], int],
    ) -> np.ndarray:
        """Fast greedy left-to-right non-overlapping merge for one hierarchy level."""
        if not pair_map or len(stream) < 2:
            return stream
        s_list = stream.tolist()
        n = len(s_list)
        out: List[int] = []
        i = 0
        get = pair_map.get
        while i < n:
            if i + 1 < n:
                merged = get((s_list[i], s_list[i + 1]))
                if merged is not None:
                    out.append(merged)
                    i += 2
                    continue
            out.append(s_list[i])
            i += 1
        return np.asarray(out, dtype=np.int32)

    def fit(self, tokens: Sequence[int], *, n_workers: int = 1) -> "HierarchicalChunkMemory":
        """Recursively discover high-PMI structural chunks and fit chunk-level transitions."""
        arr = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        if len(arr) < 4:
            return self

        self.chunk_to_atoms_ = {i: (i,) for i in range(self.vocab_size)}
        self.atoms_to_chunk_ = {(i,): i for i in range(self.vocab_size)}
        self.level_merges_ = []
        next_cid = self.vocab_size
        cur_stream = arr

        for _lvl in range(self.max_levels):
            n_cur = len(cur_stream)
            if n_cur < 4:
                break
            n_units = next_cid
            u_counts = np.bincount(cur_stream, minlength=n_units).astype(float)
            u_counts[0] = 0.0
            tot_u = max(float(u_counts.sum()), 1.0)
            p_u = (u_counts + 1.0) / (tot_u + float(n_units))

            # Vectorized adjacent pair counting (skip <UNK>=0)
            valid = (cur_stream[:-1] > 0) & (cur_stream[1:] > 0)
            if not np.any(valid):
                break
            a_arr = cur_stream[:-1][valid].astype(np.int64)
            b_arr = cur_stream[1:][valid].astype(np.int64)
            stride_base = int(n_units)
            codes = a_arr * stride_base + b_arr
            uniq_codes, code_cnts = np.unique(codes, return_counts=True)
            keep = code_cnts >= self.min_pair_count
            if not np.any(keep):
                break

            inv_pairs = 1.0 / max(float(len(a_arr)), 1.0)
            scored = []
            for code_val, cnt_val in zip(uniq_codes[keep].tolist(), code_cnts[keep].tolist()):
                u = int(code_val // stride_base)
                v = int(code_val % stride_base)
                atoms_uv = self.chunk_to_atoms_[u] + self.chunk_to_atoms_[v]
                if len(atoms_uv) > self.max_chunk_len:
                    continue
                if atoms_uv in self.atoms_to_chunk_:
                    continue
                pmi = float(np.log((float(cnt_val) * inv_pairs) / max(float(p_u[u] * p_u[v]), 1e-14)))
                if pmi >= self.min_pmi:
                    # MDL compression gain proxy: frequency * positive PMI
                    mdl_gain = float(cnt_val) * pmi
                    scored.append((mdl_gain, u, v, atoms_uv))

            if not scored:
                break
            scored.sort(key=lambda item: item[0], reverse=True)
            pair_map: Dict[Tuple[int, int], int] = {}
            for _gain, u, v, atoms_uv in scored[: self.max_chunks_per_level]:
                cid = next_cid
                next_cid += 1
                pair_map[(u, v)] = cid
                self.chunk_to_atoms_[cid] = atoms_uv
                self.atoms_to_chunk_[atoms_uv] = cid

            if not pair_map:
                break
            self.level_merges_.append(pair_map)
            cur_stream = self._merge_stream_once(cur_stream, pair_map)

        self.n_total_units = next_cid
        self.chunk_counts_ = np.bincount(cur_stream, minlength=self.n_total_units).astype(float)

        # Build atom-prefix -> (next_atom, chunk_id) lookup for intra-chunk completion
        self.prefix_to_chunks_ = defaultdict(list)
        for cid in range(self.vocab_size, self.n_total_units):
            atoms = self.chunk_to_atoms_[cid]
            for pref_len in range(1, len(atoms)):
                pref = atoms[:pref_len]
                nxt_a = atoms[pref_len]
                self.prefix_to_chunks_[pref].append((nxt_a, cid))

        # Fit higher-order SparseFeatureActionIndex over compressed chunk stream
        self.chunk_index = SparseFeatureActionIndex(
            tuple(range(self.n_total_units)), prior_weight=0.0,
        )
        self.chunk_index.fit_token_stream(cur_stream, ctx_len=self.chunk_ctx_len, n_workers=n_workers)

        # Fit abstract Chunk-Class Grammar (Non-Terminals) + Skip-Slot Construction Schemas
        self._fit_chunk_class_abstraction(cur_stream, n_workers=n_workers)
        self._fitted = True
        return self

    def _fit_chunk_class_abstraction(self, cur_stream: np.ndarray, *, n_workers: int = 1) -> None:
        """Cluster discovered chunks by left/right distributional context into abstract classes."""
        self.chunk_to_class_ = None
        self.chunk_class_index = None
        self.class_to_chunks_ = {}
        self.class_entry_atom_probs_ = {}
        self.skip_schemas_ = {}
        self.n_total_chunk_classes_ = 0
        if not self.enable_chunk_classes or len(cur_stream) < 8:
            return

        u_tot = self.n_total_units
        dim = self.chunk_embed_dim
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        r_left = (rng.randn(u_tot, dim) / np.sqrt(float(dim))).astype(np.float32)
        r_right = (rng.randn(u_tot, dim) / np.sqrt(float(dim))).astype(np.float32)
        r_left[0] = 0.0
        r_right[0] = 0.0

        # Vectorized left/right neighbor distributional projection (exact Harris equivalence)
        valid = (cur_stream[:-1] > 0) & (cur_stream[1:] > 0)
        if not np.any(valid):
            return
        lhs = cur_stream[:-1][valid]
        rhs = cur_stream[1:][valid]
        left_acc = np.zeros((u_tot, dim), dtype=np.float32)
        right_acc = np.zeros((u_tot, dim), dtype=np.float32)
        np.add.at(left_acc, rhs, r_left[lhs])
        np.add.at(right_acc, lhs, r_right[rhs])

        ln = np.maximum(np.linalg.norm(left_acc, axis=1, keepdims=True), 1e-8)
        rn = np.maximum(np.linalg.norm(right_acc, axis=1, keepdims=True), 1e-8)
        role_vecs = np.concatenate([left_acc / ln, right_acc / rn], axis=1)
        self.chunk_role_vecs_ = role_vecs

        counts = self.chunk_counts_
        active = np.flatnonzero((np.arange(u_tot) > 0) & (counts >= 2.0))
        if len(active) < 4:
            return
        freq_order = active[np.argsort(-counts[active])]
        n_sing = min(self.n_singleton_chunks, max(len(freq_order) - 4, 0))
        sing_units = freq_order[:n_sing]
        clust_units = freq_order[n_sing:]

        c2cls = np.zeros(u_tot, dtype=np.int32)
        for idx_s, uid in enumerate(sing_units):
            c2cls[int(uid)] = idx_s + 1  # 1..n_sing

        k_cls = min(self.n_chunk_classes, max(len(clust_units) // 2, 1))
        if len(clust_units) > 0:
            vecs = role_vecs[clust_units].astype(float)
            centers = _fit_deterministic_centroids(vecs, k_cls, max_iter=25)
            c_norm = np.einsum("ij,ij->i", centers, centers)[None, :]
            v_norm = np.einsum("ij,ij->i", vecs, vecs)[:, None]
            d2 = vecs @ (-2.0 * centers.T) + v_norm + c_norm
            assigned = np.argmin(d2, axis=1) + 1 + n_sing
            c2cls[clust_units] = assigned.astype(np.int32)

        # Assign any ultra-rare count==1 units to nearest centroid so every seen unit has a class
        rare_units = np.flatnonzero((np.arange(u_tot) > 0) & (counts > 0.0) & (c2cls == 0))
        if len(rare_units) > 0 and len(clust_units) > 0:
            rvecs = role_vecs[rare_units].astype(float)
            v_n = np.einsum("ij,ij->i", rvecs, rvecs)[:, None]
            d2r = rvecs @ (-2.0 * centers.T) + v_n + c_norm
            c2cls[rare_units] = (np.argmin(d2r, axis=1) + 1 + n_sing).astype(np.int32)

        self.chunk_to_class_ = c2cls
        max_cc = int(c2cls.max())
        self.n_total_chunk_classes_ = max_cc

        # Precompute class -> member chunks & projected entry-atom probabilities P(entry_atom | CC)
        for cc in range(1, max_cc + 1):
            members = np.flatnonzero(c2cls == cc)
            if len(members) == 0:
                continue
            self.class_to_chunks_[cc] = members.astype(np.int32)
            atom_mass: Dict[int, float] = {}
            tot_m = 0.0
            for cid in members.tolist():
                w = float(counts[cid]) + 0.5
                a0 = self.chunk_to_atoms_[cid][0]
                if a0 > 0:
                    atom_mass[a0] = atom_mass.get(a0, 0.0) + w
                    tot_m += w
            if tot_m > 0.0:
                ranked_a = sorted(atom_mass.items(), key=lambda kv: kv[1], reverse=True)[:32]
                self.class_entry_atom_probs_[cc] = [(a, float(w / tot_m)) for a, w in ranked_a]

        # Fit higher-order Chunk-Class Transition Index: P(CC_k | CC_{k-3}, CC_{k-2}, CC_{k-1})
        cc_stream = c2cls[cur_stream]
        self.chunk_class_index = SparseFeatureActionIndex(
            tuple(range(max_cc + 1)), prior_weight=0.0,
        )
        self.chunk_class_index.fit_token_stream(cc_stream, ctx_len=self.chunk_class_ctx_len, n_workers=n_workers)

        # Fit Skip-Slot Construction Schemas: (prev2_anchor_unit, prev1_slot_class) -> next_unit
        if len(cur_stream) >= 4:
            top_anchors = set(freq_order[: min(500, len(freq_order))].tolist())
            schema_acc: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
            s_arr = cur_stream.tolist()
            cc_arr = cc_stream.tolist()
            for i in range(2, len(s_arr)):
                u2 = s_arr[i - 2]
                if u2 not in top_anchors:
                    continue
                c1 = cc_arr[i - 1]
                u0 = s_arr[i]
                if c1 > 0 and u0 > 0:
                    schema_acc[(u2, c1)][u0] += 1
            for skey, cmap in schema_acc.items():
                tot_s = sum(cmap.values())
                if tot_s < self.min_schema_count:
                    continue
                top_nxt = cmap.most_common(8)
                self.skip_schemas_[skey] = [
                    (int(u), float(cnt) / float(tot_s)) for u, cnt in top_nxt if cnt >= 2
                ]

    def parse_sequence(self, tokens: Sequence[int]) -> List[int]:
        """Parse an atomic symbol sequence into discovered multi-scale chunk IDs."""
        cur = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        for pair_map in self.level_merges_:
            cur = self._merge_stream_once(cur, pair_map)
        return cur.tolist()

    @staticmethod
    def _chunk_keys(c_ctx: Sequence[int]) -> List[Tuple[int, ...]]:
        if not c_ctx:
            return []
        keys: List[Tuple[int, ...]] = [(-1, int(c_ctx[-1]))]
        for order in range(2, len(c_ctx) + 1):
            keys.append((10 + order,) + tuple(int(x) for x in c_ctx[-order:]))
        return keys

    def predict_next_atoms(
        self,
        recent_tokens: Sequence[int],
        *,
        slow_state: Optional[np.ndarray] = None,
        atom_embeddings: Optional[np.ndarray] = None,
        local_prob_fn=None,
        k: int = 16,
        top_chunks: int = 24,
    ) -> Tuple[List[int], np.ndarray, Dict[str, float]]:
        """Predict next atomic symbol via concrete chunks, abstract chunk-classes, skip schemas, and intra-chunk completion."""
        if not self._fitted or self.chunk_index is None or not recent_tokens:
            return [], np.zeros(0, dtype=float), dict(intra_mass=0.0, inter_mass=0.0, class_mass=0.0, chunk_conf=0.0)

        tail = [int(x) for x in recent_tokens[-28:]]
        chunks = self.parse_sequence(tail)
        if not chunks:
            return [], np.zeros(0, dtype=float), dict(intra_mass=0.0, inter_mass=0.0, class_mass=0.0, chunk_conf=0.0)

        prev_atom = tail[-1] if tail else 0
        score_map: Dict[int, float] = {}
        inter_mass = 0.0
        intra_mass = 0.0
        class_mass = 0.0
        has_ho_inter = False
        has_ho_intra = False
        has_ho_class = False

        # 1. Inter-chunk concrete transition: P(next_chunk | prev_chunks) -> entry atom
        c_ctx = chunks[-self.chunk_ctx_len :]
        c_keys = self._chunk_keys(c_ctx)
        if len(c_ctx) >= 2 and c_keys[-1] in self.chunk_index._postings:
            has_ho_inter = True
        c_cand, c_probs = self.chunk_index.propose_with_probs(
            None, k=top_chunks, sparse_keys=c_keys,
        )
        for cid, pr in zip(c_cand, c_probs):
            ci = int(cid)
            p_val = float(pr)
            if ci <= 0 or p_val <= 0.0:
                continue
            entry_atom = self.chunk_to_atoms_[ci][0]
            if entry_atom > 0:
                score_map[entry_atom] = score_map.get(entry_atom, 0.0) + p_val
                inter_mass += p_val

        # 2. Abstract Chunk-Class Grammar: P(CC_k | CC_{k-3}, CC_{k-2}, CC_{k-1}) -> entry atom
        #    Generalizes zero-shot across unseen chunk combinations (6-12 atomic step horizon!)
        if self.chunk_class_index is not None and self.chunk_to_class_ is not None:
            cc_tail = [
                int(self.chunk_to_class_[c])
                for c in chunks[-self.chunk_class_ctx_len :]
                if 0 <= c < len(self.chunk_to_class_)
            ]
            if cc_tail and cc_tail[-1] > 0:
                cc_keys = self._chunk_keys(cc_tail)
                if len(cc_tail) >= 2 and cc_keys[-1] in self.chunk_class_index._postings:
                    has_ho_class = True
                cc_cand, cc_probs = self.chunk_class_index.propose_with_probs(
                    None, k=12, sparse_keys=cc_keys,
                )
                g_u = None
                if slow_state is not None and atom_embeddings is not None:
                    gn = float(np.linalg.norm(slow_state))
                    if gn > 1e-6:
                        g_u = np.asarray(slow_state, dtype=float) / gn
                cc_scale = 0.55 if not has_ho_inter else 0.35
                for cc_id, cc_p in zip(cc_cand, cc_probs):
                    cci = int(cc_id)
                    p_cc = float(cc_p)
                    if cci <= 0 or p_cc <= 0.0:
                        continue
                    atom_dist = self.class_entry_atom_probs_.get(cci)
                    if not atom_dist:
                        continue
                    for a0, p_a in atom_dist[:16]:
                        w = cc_scale * p_cc * p_a
                        if local_prob_fn is not None and prev_atom > 0:
                            p_loc, lift = local_prob_fn(prev_atom, a0)
                            if p_loc > 0.0:
                                w *= (0.5 + 0.5 * min(float(lift), 3.0))
                            else:
                                w *= 0.25
                        if g_u is not None and atom_embeddings is not None and a0 < len(atom_embeddings):
                            ea = atom_embeddings[a0]
                            en = float(np.linalg.norm(ea))
                            if en > 1e-6:
                                cos_s = float(np.dot(ea, g_u) / en)
                                w *= float(np.exp(0.6 * cos_s))
                        if w > 1e-8:
                            score_map[a0] = score_map.get(a0, 0.0) + w
                            class_mass += w

            # 2b. Skip-Slot Construction Schemas: (prev2_anchor, CC(prev1)) -> next_chunk
            if len(chunks) >= 2 and self.skip_schemas_:
                u2 = chunks[-2]
                c1 = int(self.chunk_to_class_[chunks[-1]]) if chunks[-1] < len(self.chunk_to_class_) else 0
                sk_list = self.skip_schemas_.get((u2, c1))
                if sk_list:
                    has_ho_class = True
                    for u_nxt, p_sk in sk_list:
                        a0 = self.chunk_to_atoms_[u_nxt][0]
                        if a0 > 0:
                            w_sk = 0.40 * float(p_sk)
                            score_map[a0] = score_map.get(a0, 0.0) + w_sk
                            class_mass += w_sk

        # 3. Intra-chunk completion: check if trailing 1..(max_chunk_len-1) atoms
        #    form a prefix of a multi-atom chunk conditioned on concrete OR abstract class context!
        max_pref = min(self.max_chunk_len - 1, len(tail))
        for pref_len in range(1, max_pref + 1):
            pref = tuple(tail[-pref_len:])
            if any(x <= 0 for x in pref):
                continue
            conts = self.prefix_to_chunks_.get(pref)
            if not conts:
                continue
            before_tail = tail[:-pref_len]
            b_prob_map: Dict[int, float] = {}
            b_cc_map: Dict[int, float] = {}
            if before_tail:
                before_chunks = self.parse_sequence(before_tail)
                b_ctx = before_chunks[-self.chunk_ctx_len :]
                b_cand, b_probs = self.chunk_index.propose_with_probs(
                    None, k=max(top_chunks, 32), sparse_keys=self._chunk_keys(b_ctx),
                )
                b_prob_map = {int(c): float(p) for c, p in zip(b_cand, b_probs)}
                if self.chunk_class_index is not None and self.chunk_to_class_ is not None:
                    bcc = [
                        int(self.chunk_to_class_[c])
                        for c in before_chunks[-self.chunk_class_ctx_len :]
                        if 0 <= c < len(self.chunk_to_class_)
                    ]
                    if bcc and bcc[-1] > 0:
                        bc_c, bc_p = self.chunk_class_index.propose_with_probs(
                            None, k=16, sparse_keys=self._chunk_keys(bcc),
                        )
                        b_cc_map = {int(cc): float(p) for cc, p in zip(bc_c, bc_p)}

            tot_c_prior = max(float(sum(self.chunk_counts_[cid] for _, cid in conts)), 1.0)
            pref_bonus = 1.0 + 0.45 * float(pref_len - 1)
            for nxt_atom, cid in conts:
                if nxt_atom <= 0:
                    continue
                p_ctx_c = b_prob_map.get(cid, 0.0)
                p_ctx_cc = 0.0
                if self.chunk_to_class_ is not None and cid < len(self.chunk_to_class_):
                    p_ctx_cc = b_cc_map.get(int(self.chunk_to_class_[cid]), 0.0)
                if p_ctx_c > 0.0:
                    w_add = pref_bonus * p_ctx_c
                    has_ho_intra = True
                elif p_ctx_cc > 0.0 and pref_len >= 2:
                    # Abstract chunk-class confirmation for unseen context + known multi-atom chunk
                    w_add = 0.45 * pref_bonus * p_ctx_cc * (float(self.chunk_counts_[cid]) / tot_c_prior)
                    has_ho_intra = True
                elif pref_len >= 2:
                    w_add = 0.10 * pref_bonus * (float(self.chunk_counts_[cid]) / tot_c_prior)
                    has_ho_intra = True
                else:
                    w_add = 0.0
                if w_add > 1e-8:
                    score_map[nxt_atom] = score_map.get(nxt_atom, 0.0) + w_add
                    intra_mass += w_add

        if not score_map:
            return [], np.zeros(0, dtype=float), dict(intra_mass=0.0, inter_mass=0.0, class_mass=0.0, chunk_conf=0.0)

        is_multi_atom_prev = (len(self.chunk_to_atoms_.get(chunks[-1], ())) >= 2)
        chunk_conf = 0.12
        if has_ho_inter:
            chunk_conf += 0.40
        if has_ho_class:
            chunk_conf += 0.25
        if has_ho_intra:
            chunk_conf += 0.30
        if is_multi_atom_prev:
            chunk_conf += 0.20
        chunk_conf = float(np.clip(chunk_conf, 0.0, 1.0))

        ranked = sorted(score_map.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[: max(int(k), 1)]
        acts = [a for a, _ in ranked]
        probs = np.asarray([p for _, p in ranked], dtype=float)
        p_sum = float(probs.sum())
        if p_sum > 0.0:
            probs /= p_sum
        return acts, probs, dict(
            intra_mass=intra_mass, inter_mass=inter_mass, class_mass=class_mass, chunk_conf=chunk_conf,
        )

    def inspect_chunk_classes(self, itos: Sequence[str], top_classes: int = 12) -> List[Tuple[int, List[str]]]:
        """Inspect multi-member abstract chunk classes discovered purely from distributional roles."""
        rows = []
        for cc, members in self.class_to_chunks_.items():
            if len(members) < 3:
                continue
            # Prefer classes containing multi-atom chunks
            multi = [int(m) for m in members if int(m) >= self.vocab_size]
            if len(multi) < 2:
                continue
            multi.sort(key=lambda cid: float(self.chunk_counts_[cid]), reverse=True)
            tot_cnt = float(sum(self.chunk_counts_[cid] for cid in multi[:8]))
            sample_txt = ["".join(itos[a] for a in self.chunk_to_atoms_[cid]) for cid in multi[:7]]
            rows.append((tot_cnt, cc, sample_txt))
        rows.sort(reverse=True)
        return [(cc, txt) for _, cc, txt in rows[:top_classes]]

    def prune_under_pressure(self, pressure: float) -> int:
        """Evict low-utility chunk postings and tail skip-slot schemas under memory pressure."""
        p = float(np.clip(pressure, 0.0, 1.0))
        if p <= 0.0:
            return 0
        evicted = 0
        if self.chunk_index is not None:
            evicted += self.chunk_index.prune_under_pressure(p)
        if self.chunk_class_index is not None:
            evicted += self.chunk_class_index.prune_under_pressure(p * 0.7)
        if self.skip_schemas_ and p >= 0.30:
            keep_frac = max(1.0 - 0.75 * p, 0.15)
            keep_n = max(int(len(self.skip_schemas_) * keep_frac), 64)
            if len(self.skip_schemas_) > keep_n:
                # Keep schemas with highest peak transition confidence
                ranked = sorted(
                    self.skip_schemas_.items(),
                    key=lambda kv: max((pr for _, pr in kv[1]), default=0.0),
                    reverse=True,
                )
                evicted += len(self.skip_schemas_) - keep_n
                self.skip_schemas_ = dict(ranked[:keep_n])
        return evicted

    def top_discovered_chunks(self, itos: Sequence[str], top_n: int = 20) -> List[Tuple[str, int]]:
        """Inspect the highest-count discovered multi-symbol chunks."""
        rows = []
        for cid in range(self.vocab_size, self.n_total_units):
            cnt = int(self.chunk_counts_[cid])
            if cnt <= 0:
                continue
            atoms = self.chunk_to_atoms_[cid]
            text = "".join(itos[a] if 0 <= a < len(itos) else "?" for a in atoms)
            rows.append((cnt, text))
        rows.sort(reverse=True)
        return [(txt, cnt) for cnt, txt in rows[:top_n]]


@dataclass
class FrameSpec:
    """Learned directional non-adjacent dependency ``opener ... [gap] ... closer``."""

    opener: int
    closer: int
    pmi: float
    asymmetry: float
    mean_gap: float
    std_gap: float
    strength: float


class StructuralFrameMemory:
    """Discovers directional non-adjacent frames (open->close / key->lock) with an online register."""

    def __init__(
        self,
        vocab_size: int,
        *,
        min_gap: int = 2,
        max_gap: int = 18,
        min_pair_count: int = 15,
        min_pmi: float = 1.6,
        min_asymmetry: float = 0.62,
        max_closers_per_opener: int = 6,
        max_active_frames: int = 12,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.min_gap = max(int(min_gap), 1)
        self.max_gap = max(int(max_gap), self.min_gap + 1)
        self.min_pair_count = max(int(min_pair_count), 3)
        self.min_pmi = float(min_pmi)
        self.min_asymmetry = float(min_asymmetry)
        self.max_closers_per_opener = max(int(max_closers_per_opener), 1)
        self.max_active_frames = max(int(max_active_frames), 2)

        self.frames_by_opener_: Dict[int, List[FrameSpec]] = {}
        self._active: List[Tuple[int, int]] = []  # list of (opener_sym, age)
        self._fitted: bool = False

    def fit(
        self,
        tokens: Sequence[int],
        *,
        max_samples: int = 250000,
    ) -> "StructuralFrameMemory":
        """Discover directional non-adjacent dependencies ``opener -> closer`` + gap timing."""
        arr = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        n = len(arr)
        if n <= self.max_gap + 2:
            return self

        counts = np.bincount(arr, minlength=self.vocab_size).astype(float)
        counts[0] = 0.0
        tot = max(float(counts.sum()), 1.0)
        p_glob = (counts + 1.0) / (tot + float(self.vocab_size))

        # Count directional pairs (a -> b) within [min_gap, max_gap] + gap moments
        fwd_cnt: Dict[Tuple[int, int], int] = defaultdict(int)
        fwd_gap_sum: Dict[Tuple[int, int], float] = defaultdict(float)
        fwd_gap_sq: Dict[Tuple[int, int], float] = defaultdict(float)
        opener_trials: Dict[int, int] = defaultdict(int)

        span = n - self.max_gap - 1
        stride = max(span // max(int(max_samples), 1000), 1)
        g_lo = self.min_gap
        g_hi = self.max_gap
        arr_list = arr.tolist()

        for t in range(0, span, stride):
            a = arr_list[t]
            if a <= 0:
                continue
            opener_trials[a] += 1
            seen_b = set()
            for d in range(g_lo, g_hi + 1):
                b = arr_list[t + d]
                if b <= 0 or b == a or b in seen_b:
                    continue
                seen_b.add(b)
                key = (a, b)
                fwd_cnt[key] += 1
                fd = float(d)
                fwd_gap_sum[key] += fd
                fwd_gap_sq[key] += fd * fd

        # Probability that symbol b appears at least once in a random window of width W
        w_len = float(g_hi - g_lo + 1)
        by_opener: Dict[int, List[FrameSpec]] = defaultdict(list)
        for (a, b), cnt in fwd_cnt.items():
            if cnt < self.min_pair_count:
                continue
            rev = fwd_cnt.get((b, a), 0)
            asym = float(cnt) / float(cnt + rev)
            if asym < self.min_asymmetry:
                continue
            mean_g = fwd_gap_sum[(a, b)] / float(cnt)
            var_g = max(fwd_gap_sq[(a, b)] / float(cnt) - mean_g * mean_g, 0.25)
            std_g = float(np.sqrt(var_g))
            n_a = float(opener_trials.get(a, 1))
            p_cond = float(cnt) / max(n_a, 1.0)
            # Effective timing window reflects how tightly b concentrates around mean_gap
            w_eff = min(max(2.0 * std_g + 1.0, 1.0), w_len)
            p_null = 1.0 - float(np.power(max(1.0 - float(p_glob[b]), 1e-9), w_eff))
            pmi = float(np.log(max(p_cond, 1e-12) / max(p_null, 1e-12)))
            if pmi < self.min_pmi:
                continue
            strength = pmi * asym * min(float(np.log1p(cnt)) / 5.0, 1.5)
            by_opener[a].append(FrameSpec(
                opener=a,
                closer=b,
                pmi=pmi,
                asymmetry=asym,
                mean_gap=mean_g,
                std_gap=std_g,
                strength=strength,
            ))

        self.frames_by_opener_ = {}
        for a, specs in by_opener.items():
            specs.sort(key=lambda sp: sp.strength, reverse=True)
            self.frames_by_opener_[a] = specs[: self.max_closers_per_opener]

        self._fitted = True
        return self

    def reset(self) -> None:
        self._active.clear()

    def observe(self, symbol: int) -> None:
        """Advance online structural frame register by one observed/emitted symbol."""
        sym = int(symbol)
        # 1. Age existing frames and pop any that are closed by ``sym`` or expired
        next_active: List[Tuple[int, int]] = []
        for op, age in self._active:
            new_age = age + 1
            if new_age > self.max_gap:
                continue
            specs = self.frames_by_opener_.get(op, ())
            if any(sp.closer == sym for sp in specs):
                # Frame closed cleanly -> pop!
                continue
            next_active.append((op, new_age))
        self._active = next_active

        # 2. If ``sym`` is a learned directional opener, push it onto the register
        if sym > 0 and sym in self.frames_by_opener_:
            # Refresh if same opener already active
            self._active = [(op, age) for op, age in self._active if op != sym]
            self._active.append((sym, 0))
            if len(self._active) > self.max_active_frames:
                self._active = self._active[-self.max_active_frames :]

    def predict_closers_for_history(
        self,
        recent_history: Optional[Sequence[int]] = None,
    ) -> Dict[int, float]:
        """Compute active expected closer scores from either the live register or a rolled history."""
        if not self._fitted:
            return {}
        if recent_history is not None:
            # Stateless replay over trailing max_gap+2 symbols (safe for lookahead branching!)
            reg: List[Tuple[int, int]] = []
            tail = [int(x) for x in recent_history[-(self.max_gap + 2) :]]
            for sym in tail:
                nxt_reg: List[Tuple[int, int]] = []
                for op, age in reg:
                    na = age + 1
                    if na > self.max_gap:
                        continue
                    if any(sp.closer == sym for sp in self.frames_by_opener_.get(op, ())):
                        continue
                    nxt_reg.append((op, na))
                reg = nxt_reg
                if sym > 0 and sym in self.frames_by_opener_:
                    reg = [(op, age) for op, age in reg if op != sym]
                    reg.append((sym, 0))
                    if len(reg) > self.max_active_frames:
                        reg = reg[-self.max_active_frames :]
        else:
            reg = self._active

        if not reg:
            return {}

        scores: Dict[int, float] = {}
        for op, age in reg:
            # Next symbol will land at gap = age + 1
            next_gap = float(age + 1)
            if next_gap < float(self.min_gap) or next_gap > float(self.max_gap):
                continue
            for sp in self.frames_by_opener_.get(op, ()):
                sigma = max(sp.std_gap, 1.5)
                timing = float(np.exp(-0.5 * ((next_gap - sp.mean_gap) / sigma) ** 2))
                val = sp.strength * (0.25 + 0.75 * timing)
                if val > scores.get(sp.closer, 0.0):
                    scores[sp.closer] = val
        return scores

    def top_frames(self, itos: Sequence[str], top_n: int = 20) -> List[Tuple[str, str, float, float, float]]:
        """Inspect strongest discovered directional frames: (opener, closer, pmi, asym, mean_gap)."""
        all_specs = [sp for specs in self.frames_by_opener_.values() for sp in specs]
        all_specs.sort(key=lambda sp: sp.strength, reverse=True)
        out = []
        for sp in all_specs[:top_n]:
            o_s = itos[sp.opener] if 0 <= sp.opener < len(itos) else "?"
            c_s = itos[sp.closer] if 0 <= sp.closer < len(itos) else "?"
            out.append((o_s, c_s, round(sp.pmi, 2), round(sp.asymmetry, 2), round(sp.mean_gap, 1)))
        return out
