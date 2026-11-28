"""Task-agnostic sparse inverted index for prototype retrieval and Top-K action pruning.

Replaces dense O(|V|) action branching and O(N_proto * D) distance scans with
sparse active-feature inverted postings:
  - ``SparseFeatureActionIndex``: maps active quantized/sparse state features to
    candidate action scores from experience, proposing Top-K actions in time
    proportional to active-feature postings rather than vocabulary size |V|.
  - ``SparsePrototypeRetriever``: inverted-index sparse cosine/Jaccard retrieval
    for high-dimensional sparse representations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import heapq
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def extract_sparse_features(
    state: np.ndarray,
    *,
    quantize_step: float = 0.25,
    include_nonzero_ids: bool = False,
) -> Tuple[Tuple[int, int], ...]:
    """Extract deterministic sparse feature keys ``(dim, bucket)`` from a state vector."""
    arr = np.asarray(state, dtype=float).reshape(-1)
    if include_nonzero_ids:
        nz = np.flatnonzero(np.abs(arr) > 1e-12)
        return tuple((int(d), int(round(float(arr[d]) / quantize_step))) for d in nz)
    step = max(float(quantize_step), 1e-6)
    return tuple((int(d), int(round(float(v) / step))) for d, v in enumerate(arr))


class SparseFeatureActionIndex:
    """Inverted-postings heuristic action proposer for large discrete action spaces."""

    def __init__(
        self,
        action_space: Sequence[int],
        *,
        quantize_step: float = 0.25,
        include_nonzero_ids: bool = False,
        prior_weight: float = 0.25,
    ):
        self.action_space: Tuple[int, ...] = tuple(int(a) for a in action_space)
        self.quantize_step = float(quantize_step)
        self.include_nonzero_ids = bool(include_nonzero_ids)
        self.prior_weight = float(prior_weight)
        self._postings: Dict[Tuple[int, ...], Dict[int, List[float]]] = defaultdict(dict)
        self._key_totals: Dict[Tuple[int, ...], float] = defaultdict(float)
        self._global_counts: Counter = Counter()
        self._global_scores: Dict[int, float] = defaultdict(float)
        self.total_updates = 0

    def update(
        self,
        state: np.ndarray,
        action: int,
        dopamine: float = 0.0,
        pain: float = 0.0,
        sparse_keys: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> None:
        """Record one observed ``(state, action, feedback)`` into inverted postings."""
        act = int(action)
        score = float(dopamine) - float(pain)
        feats = (
            tuple(sparse_keys)
            if sparse_keys is not None
            else extract_sparse_features(
                state,
                quantize_step=self.quantize_step,
                include_nonzero_ids=self.include_nonzero_ids,
            )
        )
        for fk in feats:
            bucket = self._postings[fk].get(act)
            if bucket is None:
                self._postings[fk][act] = [1.0, score]
            else:
                bucket[0] += 1.0
                bucket[1] += score
            self._key_totals[fk] += 1.0
        self._global_counts[act] += 1
        self._global_scores[act] += score
        self.total_updates += 1

    @staticmethod
    def _count_shard(
        s_list: List[int],
        start: int,
        end: int,
        ctx_len: int,
    ) -> Tuple[Dict[Tuple[int, ...], Dict[int, float]], Dict[Tuple[int, ...], float], Counter]:
        """Fast thread-local counting of nested suffix keys on slice ``[start, end)``."""
        postings: Dict[Tuple[int, ...], Dict[int, float]] = {}
        key_totals: Dict[Tuple[int, ...], float] = {}
        g_counts: Counter = Counter()
        c = ctx_len
        for pos in range(start, end):
            act = s_list[pos]
            g_counts[act] += 1
            # Order 1
            fk1 = (-1, s_list[pos - 1])
            d1 = postings.get(fk1)
            if d1 is None:
                postings[fk1] = {act: 1.0}
                key_totals[fk1] = 1.0
            else:
                d1[act] = d1.get(act, 0.0) + 1.0
                key_totals[fk1] += 1.0
            # Orders 2..ctx_len
            if c >= 2:
                fk2 = (12, s_list[pos - 2], s_list[pos - 1])
                d2 = postings.get(fk2)
                if d2 is None:
                    postings[fk2] = {act: 1.0}
                    key_totals[fk2] = 1.0
                else:
                    d2[act] = d2.get(act, 0.0) + 1.0
                    key_totals[fk2] += 1.0
            if c >= 3:
                fk3 = (13, s_list[pos - 3], s_list[pos - 2], s_list[pos - 1])
                d3 = postings.get(fk3)
                if d3 is None:
                    postings[fk3] = {act: 1.0}
                    key_totals[fk3] = 1.0
                else:
                    d3[act] = d3.get(act, 0.0) + 1.0
                    key_totals[fk3] += 1.0
            if c >= 4:
                for order in range(4, c + 1):
                    fko = (10 + order,) + tuple(s_list[pos - order : pos])
                    do = postings.get(fko)
                    if do is None:
                        postings[fko] = {act: 1.0}
                        key_totals[fko] = 1.0
                    else:
                        do[act] = do.get(act, 0.0) + 1.0
                        key_totals[fko] += 1.0
        return postings, key_totals, g_counts

    def fit_token_stream(
        self,
        tokens: Sequence[int],
        *,
        ctx_len: int = 4,
        n_workers: int = 1,
    ) -> "SparseFeatureActionIndex":
        """Directly index nested suffix n-gram keys -> next token (supports no-GIL multi-core sharding)."""
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        n = len(arr)
        c = max(int(ctx_len), 1)
        if n <= c:
            return self
        s_list = arr.tolist()
        nw = max(int(n_workers), 1)
        if nw <= 1 or (n - c) < 100_000:
            shard_results = [self._count_shard(s_list, c, n, c)]
        else:
            nw = min(nw, max(os.cpu_count() or 1, 1), 12)
            span = n - c
            step = (span + nw - 1) // nw
            ranges = [
                (c + i * step, min(c + (i + 1) * step, n))
                for i in range(nw)
                if c + i * step < n
            ]
            with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
                futs = [ex.submit(self._count_shard, s_list, s0, s1, c) for s0, s1 in ranges]
                shard_results = [f.result() for f in futs]

        # Exact lossless merge of shard counts into self._postings / self._key_totals
        post_main = self._postings
        kt_main = self._key_totals
        gc_main = self._global_counts
        gs_main = self._global_scores
        for sh_post, sh_kt, sh_gc in shard_results:
            for fk, d_act in sh_post.items():
                dst = post_main[fk]
                for act, cnt in d_act.items():
                    bkt = dst.get(act)
                    if bkt is None:
                        dst[act] = [cnt, cnt]
                    else:
                        bkt[0] += cnt
                        bkt[1] += cnt
                kt_main[fk] += sh_kt[fk]
            for act, cnt in sh_gc.items():
                gc_main[act] += cnt
                gs_main[act] += float(cnt)
                self.total_updates += int(cnt)
        return self

    def prune_under_pressure(self, pressure: float) -> int:
        """Evict lowest-count sparse postings proportionally to memory pressure ``p in [0, 1]``."""
        p = float(np.clip(pressure, 0.0, 1.0))
        if p <= 0.0 or not self._postings:
            return 0
        # Convex eviction threshold: gentle (min_cnt=1) at low pressure, aggressive at high pressure
        min_cnt = 1.0 if p < 0.25 else (2.0 if p < 0.55 else (4.0 if p < 0.80 else 8.0))
        evicted = 0
        for fk in list(self._postings.keys()):
            # Preserve order-1 keys longer than high-order tail keys
            is_high_order = (len(fk) >= 3 and fk[0] >= 12)
            thresh = min_cnt if is_high_order else max(min_cnt * 0.5, 1.0)
            post = self._postings[fk]
            removed_mass = 0.0
            for act in list(post.keys()):
                if post[act][0] <= thresh:
                    removed_mass += post[act][0]
                    del post[act]
                    evicted += 1
            if not post:
                del self._postings[fk]
                self._key_totals.pop(fk, None)
            elif removed_mass > 0.0:
                self._key_totals[fk] = max(self._key_totals.get(fk, 0.0) - removed_mass, 0.0)
        return evicted

    def _propose_interpolated_ad(
        self,
        sparse_keys: Sequence[Tuple[int, ...]],
        k_req: int,
        discount: float = 0.5,
    ) -> Tuple[List[int], np.ndarray]:
        """Exact top-down Interpolated Absolute Discounting over nested suffix keys."""
        # Extract nested suffix keys ordered from highest order down to order 1.
        ordered_keys: List[Tuple[int, Tuple[int, ...]]] = []
        for fk in sparse_keys:
            if len(fk) >= 2 and fk[0] == -1:
                ordered_keys.append((1, fk))
            elif len(fk) >= 3 and fk[0] >= 12:
                ordered_keys.append((len(fk) - 1, fk))
        ordered_keys.sort(key=lambda item: item[0], reverse=True)

        probs: Dict[int, float] = defaultdict(float)
        weight = 1.0
        d = float(discount)
        for _order, fk in ordered_keys:
            post = self._postings.get(fk)
            tot = self._key_totals.get(fk, 0.0)
            if not post or tot <= 0.0:
                continue
            inv_tot = 1.0 / tot
            for act, (cnt, s_sum) in post.items():
                eff = max(cnt - d, 0.0)
                if eff > 0.0:
                    pos_rate = max(s_sum / cnt, 0.0)
                    if pos_rate > 0.0:
                        probs[act] += weight * (eff * inv_tot) * pos_rate
            # Exact backoff mass passed down to shorter suffix contexts.
            n_types = float(len(post))
            weight *= min((d * n_types) * inv_tot, 1.0)
            if weight <= 1e-12:
                break

        # Fold remaining backoff mass into global unigram distribution.
        if weight > 1e-12 and self._global_counts:
            g_tot = float(sum(self._global_counts.values()))
            if g_tot > 0.0:
                inv_g = 1.0 / g_tot
                for act in list(probs.keys()):
                    gc = self._global_counts.get(act, 0)
                    if gc > 0:
                        pos_rate = max(self._global_scores[act] / float(gc), 0.0)
                        probs[act] += weight * (float(gc) * inv_g) * pos_rate
                if len(probs) < k_req:
                    for act, gc in self._global_counts.most_common(k_req * 2):
                        if act not in probs and gc > 0:
                            pos_rate = max(self._global_scores[act] / float(gc), 0.0)
                            probs[act] = weight * (float(gc) * inv_g) * pos_rate

        if probs:
            ranked = sorted(
                probs.keys(),
                key=lambda a: (probs[a], -a),
                reverse=True,
            )[:k_req]
            p_vals = np.asarray([probs[a] for a in ranked], dtype=float)
        else:
            ranked = []
            p_vals = np.zeros(0, dtype=float)

        if len(ranked) < k_req:
            seen = set(ranked)
            extra: List[int] = []
            for a in self.action_space:
                if a not in seen:
                    extra.append(a)
                    seen.add(a)
                    if len(ranked) + len(extra) == k_req:
                        break
            ranked.extend(extra)
            p_vals = np.concatenate([p_vals, np.zeros(len(extra), dtype=float)])

        return ranked, p_vals

    def propose_with_probs(
        self,
        state: Optional[np.ndarray],
        k: int,
        sparse_keys: Optional[Sequence[Tuple[int, ...]]] = None,
        discount: float = 0.5,
    ) -> Tuple[List[int], np.ndarray]:
        """Return ``(top_k_actions, scores_or_probs)`` for a single query."""
        k_req = min(max(int(k), 1), len(self.action_space))
        if sparse_keys is not None:
            return self._propose_interpolated_ad(sparse_keys, k_req, discount=discount)
        ranked = self.propose_one(state, k_req, sparse_keys=None)
        return ranked, np.full(len(ranked), 1.0 / max(len(ranked), 1), dtype=float)

    def fit_transitions(self, transitions: Sequence[dict], lag: int = 0) -> "SparseFeatureActionIndex":
        """Bulk-index a list of transition dicts from ``collect``."""
        del lag
        for row in transitions:
            hist = np.asarray(row.get("history", ()), dtype=float).reshape(-1)
            if hist.size == 0:
                continue
            act = int(round(float(hist[0])))
            sig = row.get("signals", (0.0, 0.0))
            self.update(
                row["obs"], act, dopamine=float(sig[0]), pain=float(sig[1]),
                sparse_keys=row.get("sparse_keys"),
            )
        return self

    def propose_one(
        self,
        state: Optional[np.ndarray],
        k: int,
        sparse_keys: Optional[Sequence[Tuple[int, ...]]] = None,
        discount: float = 0.5,
    ) -> List[int]:
        """Return Top-K candidate actions for a single state via sparse postings."""
        k_req = min(max(int(k), 1), len(self.action_space))
        if sparse_keys is not None:
            ranked, _ = self._propose_interpolated_ad(
                sparse_keys, k_req, discount=discount,
            )
            return ranked
        if k_req >= len(self.action_space):
            return list(self.action_space)
        feats = extract_sparse_features(
            state,
            quantize_step=self.quantize_step,
            include_nonzero_ids=self.include_nonzero_ids,
        )
        scores: Dict[int, float] = defaultdict(float)
        counts: Dict[int, float] = defaultdict(float)
        for fk in feats:
            post = self._postings.get(fk)
            if not post:
                continue
            for act, (cnt, s_sum) in post.items():
                counts[act] += cnt
                scores[act] += s_sum

        if scores:
            ranked = sorted(
                scores.keys(),
                key=lambda a: (
                    (scores[a] / max(counts[a], 1.0))
                    + self.prior_weight * np.log1p(counts[a]),
                    -a,
                ),
                reverse=True,
            )[:k_req]
        else:
            ranked = []

        if len(ranked) < k_req:
            seen = set(ranked)
            if self._global_counts:
                global_ranked = sorted(
                    self._global_counts.keys(),
                    key=lambda a: (
                        self._global_scores[a] / max(self._global_counts[a], 1),
                        self._global_counts[a],
                        -a,
                    ),
                    reverse=True,
                )
                for a in global_ranked:
                    if a not in seen:
                        ranked.append(a)
                        seen.add(a)
                        if len(ranked) == k_req:
                            break
            if len(ranked) < k_req:
                for a in self.action_space:
                    if a not in seen:
                        ranked.append(a)
                        seen.add(a)
                        if len(ranked) == k_req:
                            break
        return ranked

    def propose_batch(self, states: np.ndarray, k: int) -> np.ndarray:
        """Return ``(len(states), k_eff)`` integer/float action matrix."""
        arr = np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        k_req = min(max(int(k), 1), len(self.action_space))
        out = np.empty((len(arr), k_req), dtype=float)
        for i in range(len(arr)):
            out[i] = self.propose_one(arr[i], k_req)
        return out


class SparsePrototypeRetriever:
    """Inverted-index sparse prototype retriever (Jaccard / cosine over active keys)."""

    def __init__(self):
        self._proto_feats: List[frozenset] = []
        self._postings: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def add(self, features: Iterable[Tuple[int, int]]) -> int:
        fset = frozenset(features)
        pid = len(self._proto_feats)
        self._proto_feats.append(fset)
        for fk in fset:
            self._postings[fk].append(pid)
        return pid

    def query_top_k(
        self, features: Iterable[Tuple[int, int]], k: int = 5,
    ) -> List[Tuple[int, float]]:
        qset = frozenset(features)
        if not qset or not self._proto_feats:
            return []
        overlap: Counter = Counter()
        for fk in qset:
            for pid in self._postings.get(fk, ()):
                overlap[pid] += 1
        if not overlap:
            return []
        q_len = len(qset)
        scored = []
        for pid, inter in overlap.items():
            union = q_len + len(self._proto_feats[pid]) - inter
            sim = float(inter) / float(max(union, 1))
            scored.append((sim, -pid, pid))
        top = heapq.nlargest(max(int(k), 1), scored)
        return [(pid, sim) for sim, _neg_id, pid in top]
