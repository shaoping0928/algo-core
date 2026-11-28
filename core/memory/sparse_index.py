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
from functools import lru_cache
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


def _ordered_suffix_keys(keys: Sequence[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    suffixes = []
    for key in keys:
        if len(key) >= 2 and key[0] == -1:
            suffixes.append((1, key))
        elif len(key) >= 3 and key[0] >= 12:
            suffixes.append((len(key) - 1, key))
    suffixes.sort(key=lambda item: item[0], reverse=True)
    return [key for _, key in suffixes]


class _PackedSuffixQuery:
    """Vectorized discounted backoff for broad count postings over dense action IDs."""

    def __init__(self, index: SparseFeatureActionIndex):
        self._postings = index._postings
        self._totals = index._key_totals
        self._size = len(index.action_space)
        self._row = lru_cache(maxsize=256)(self._pack_row)
        self._global_mass = np.zeros(self._size, dtype=np.float64)
        self._global_rate = np.zeros(self._size, dtype=np.float64)
        total = float(sum(index._global_counts.values()))
        if total > 0.0:
            inv_total = 1.0 / total
            for action, count in index._global_counts.items():
                if 0 <= action < self._size and count > 0:
                    self._global_mass[action] = float(count) * inv_total
                    self._global_rate[action] = max(index._global_scores[action] / float(count), 0.0)
        self._finite_global_rates = bool(np.isfinite(self._global_rate).all())

    def _pack_row(self, key: Tuple[int, ...]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        post = self._postings[key]
        try:
            actions = np.fromiter(post, dtype=np.intp, count=len(post))
        except OverflowError:
            return None
        if actions.min() < 0 or actions.max() >= self._size:
            return None
        try:
            counts = np.fromiter(post.values(), dtype=np.float64, count=len(post))
        except (TypeError, ValueError):
            # Reward-bearing buckets use the scalar reference path.
            return None
        return actions, counts

    def propose(
        self, keys: Sequence[Tuple[int, ...]], k: int, discount: float,
    ) -> Optional[Tuple[List[int], np.ndarray]]:
        if not self._finite_global_rates or not 0.0 <= discount < float("inf"):
            return None
        rows = []
        for key in _ordered_suffix_keys(keys):
            post = self._postings.get(key)
            total = self._totals.get(key, 0.0)
            if post and total > 0.0:
                rows.append((key, len(post), 1.0 / total))
        if sum(count for _, count, _ in rows) < 256:
            return None

        scores = np.zeros(self._size, dtype=np.float64)
        seen = np.zeros(self._size, dtype=bool)
        weight = 1.0
        for key, count, inv_total in rows:
            packed = self._row(key)
            if packed is None:
                return None
            actions, counts = packed
            discounted = counts - discount
            keep = discounted > 0.0
            selected = actions[keep]
            scores[selected] += weight * (discounted[keep] * inv_total)
            seen[selected] = True
            weight *= min((discount * float(count)) * inv_total, 1.0)
            if weight <= 1e-12:
                break

        active = np.flatnonzero(seen)
        if active.size < k:
            # Sparse queries can introduce new actions from the unigram fallback.
            return None
        values = scores[active]
        if weight > 1e-12:
            values += (weight * self._global_mass[active]) * self._global_rate[active]

        if active.size > k:
            cutoff = np.partition(values, -k)[-k]
            above = np.flatnonzero(values > cutoff)
            # At the cutoff, choose the smallest action IDs, like the scalar sort.
            ties = np.flatnonzero(values == cutoff)[: k - len(above)]
            top = np.concatenate((above, ties))
            active, values = active[top], values[top]
        ordered = np.lexsort((active, -values))
        return active[ordered].tolist(), values[ordered]


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
        self._postings: Dict[Tuple[int, ...], Dict[int, object]] = defaultdict(dict)
        self._key_totals: Dict[Tuple[int, ...], float] = defaultdict(float)
        self._global_counts: Counter = Counter()
        self._global_scores: Dict[int, float] = defaultdict(float)
        self.total_updates = 0

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_cached_proposal", None)
        return state

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
            post = self._postings[fk]
            bucket = post.get(act)
            if bucket is None:
                post[act] = 1.0 if score == 1.0 else [1.0, score]
            elif isinstance(bucket, list):
                bucket[0] += 1.0
                bucket[1] += score
            elif score == 1.0:
                post[act] = float(bucket) + 1.0
            else:
                bval = float(bucket)
                post[act] = [bval + 1.0, bval + score]
            self._key_totals[fk] += 1.0
        self._global_counts[act] += 1
        self._global_scores[act] += score
        self.total_updates += 1
        self.__dict__.pop("_cached_proposal", None)

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

    @staticmethod
    def _bitpack_count_order(
        arr: np.ndarray,
        c: int,
        order: int,
        bits: int,
        min_cnt: int,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Vectorized C-level bit-packed n-gram counting via ``np.unique`` (releases GIL)."""
        n = len(arr)
        codes = arr[c - order : n - order].astype(np.int64)
        for step in range(1, order + 1):
            codes = (codes << bits) | arr[c - order + step : n - order + step].astype(np.int64)
        uniq, cnts = np.unique(codes, return_counts=True)
        if min_cnt > 1:
            mask = cnts >= min_cnt
            uniq = uniq[mask]
            cnts = cnts[mask]
        return order, uniq, cnts

    def _fit_bitpacked(
        self,
        arr: np.ndarray,
        c: int,
        bits: int,
        n_workers: int,
        min_high_order_count: int,
    ) -> None:
        """Fast lossless int64 bit-packed ingestion (20x-50x faster than Python loops)."""
        n = len(arr)
        mask_val = (1 << bits) - 1
        # Update global counts via vectorized bincount
        targets = arr[c:]
        g_bins = np.bincount(targets)
        nz = np.flatnonzero(g_bins)
        gc_main = self._global_counts
        gs_main = self._global_scores
        for act, cnt in zip(nz.tolist(), g_bins[nz].tolist()):
            gc_main[act] += cnt
            gs_main[act] += float(cnt)
            self.total_updates += int(cnt)

        orders = list(range(1, c + 1))
        if n_workers > 1 and c > 1:
            with ThreadPoolExecutor(max_workers=min(n_workers, c)) as ex:
                futs = [
                    ex.submit(
                        self._bitpack_count_order,
                        arr, c, o, bits,
                        min_high_order_count if o >= 2 else 1,
                    )
                    for o in orders
                ]
                results = [f.result() for f in futs]
        else:
            results = [
                self._bitpack_count_order(
                    arr, c, o, bits,
                    min_high_order_count if o >= 2 else 1,
                )
                for o in orders
            ]

        post_main = self._postings
        kt_main = self._key_totals
        for order, uniq, cnts in results:
            if len(uniq) == 0:
                continue
            acts = (uniq & mask_val).tolist()
            cnt_list = cnts.tolist()
            if order == 1:
                p1s = ((uniq >> bits) & mask_val).tolist()
                for p1, act, cnt in zip(p1s, acts, cnt_list):
                    fk = (-1, p1)
                    dst = post_main[fk]
                    fc = float(cnt)
                    bkt = dst.get(act)
                    if bkt is None:
                        dst[act] = fc
                    elif isinstance(bkt, list):
                        bkt[0] += fc
                        bkt[1] += fc
                    else:
                        dst[act] = float(bkt) + fc
                    kt_main[fk] += fc
            elif order == 2:
                p1s = ((uniq >> bits) & mask_val).tolist()
                p2s = ((uniq >> (2 * bits)) & mask_val).tolist()
                for p2, p1, act, cnt in zip(p2s, p1s, acts, cnt_list):
                    fk = (12, p2, p1)
                    dst = post_main[fk]
                    fc = float(cnt)
                    bkt = dst.get(act)
                    if bkt is None:
                        dst[act] = fc
                    elif isinstance(bkt, list):
                        bkt[0] += fc
                        bkt[1] += fc
                    else:
                        dst[act] = float(bkt) + fc
                    kt_main[fk] += fc
            elif order == 3:
                p1s = ((uniq >> bits) & mask_val).tolist()
                p2s = ((uniq >> (2 * bits)) & mask_val).tolist()
                p3s = ((uniq >> (3 * bits)) & mask_val).tolist()
                for p3, p2, p1, act, cnt in zip(p3s, p2s, p1s, acts, cnt_list):
                    fk = (13, p3, p2, p1)
                    dst = post_main[fk]
                    fc = float(cnt)
                    bkt = dst.get(act)
                    if bkt is None:
                        dst[act] = fc
                    elif isinstance(bkt, list):
                        bkt[0] += fc
                        bkt[1] += fc
                    else:
                        dst[act] = float(bkt) + fc
                    kt_main[fk] += fc
            elif order == 4:
                p1s = ((uniq >> bits) & mask_val).tolist()
                p2s = ((uniq >> (2 * bits)) & mask_val).tolist()
                p3s = ((uniq >> (3 * bits)) & mask_val).tolist()
                p4s = ((uniq >> (4 * bits)) & mask_val).tolist()
                for p4, p3, p2, p1, act, cnt in zip(p4s, p3s, p2s, p1s, acts, cnt_list):
                    fk = (14, p4, p3, p2, p1)
                    dst = post_main[fk]
                    fc = float(cnt)
                    bkt = dst.get(act)
                    if bkt is None:
                        dst[act] = fc
                    elif isinstance(bkt, list):
                        bkt[0] += fc
                        bkt[1] += fc
                    else:
                        dst[act] = float(bkt) + fc
                    kt_main[fk] += fc
            else:
                tag = 10 + order
                shifts = [bits * s for s in range(order, 0, -1)]
                u_list = uniq.tolist()
                for uv, act, cnt in zip(u_list, acts, cnt_list):
                    fk = (tag,) + tuple((uv >> sh) & mask_val for sh in shifts)
                    dst = post_main[fk]
                    fc = float(cnt)
                    bkt = dst.get(act)
                    if bkt is None:
                        dst[act] = fc
                    elif isinstance(bkt, list):
                        bkt[0] += fc
                        bkt[1] += fc
                    else:
                        dst[act] = float(bkt) + fc
                    kt_main[fk] += fc

    def fit_token_stream(
        self,
        tokens: Sequence[int],
        *,
        ctx_len: int = 4,
        n_workers: int = 1,
        min_high_order_count: int = 1,
    ) -> "SparseFeatureActionIndex":
        """Directly index nested suffix n-gram keys -> next token (bit-packed C + no-GIL multi-core)."""
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        n = len(arr)
        c = max(int(ctx_len), 1)
        if n <= c:
            return self

        self.__dict__.pop("_cached_proposal", None)
        min_v = int(arr.min())
        max_v = int(arr.max())
        bits = max(max_v.bit_length(), 1)
        if min_v >= 0 and (c + 1) * bits <= 62:
            self._fit_bitpacked(arr, c, bits, max(int(n_workers), 1), max(int(min_high_order_count), 1))
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
                        dst[act] = float(cnt)
                    elif isinstance(bkt, list):
                        bkt[0] += cnt
                        bkt[1] += cnt
                    else:
                        dst[act] = float(bkt) + float(cnt)
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
        # Order-aware convex eviction: preserve lower-order backbone while trimming high-order tails
        if p < 0.25:
            t_ord1, t_ord2, t_ord3, t_ord4 = 0.0, 1.0, 1.0, 1.0
        elif p < 0.55:
            t_ord1, t_ord2, t_ord3, t_ord4 = 1.0, 1.0, 2.0, 2.0
        elif p < 0.80:
            t_ord1, t_ord2, t_ord3, t_ord4 = 1.0, 2.0, 3.0, 4.0
        else:
            t_ord1, t_ord2, t_ord3, t_ord4 = 2.0, 3.0, 5.0, 8.0

        evicted = 0
        new_postings: Dict[Tuple[int, ...], Dict[int, object]] = defaultdict(dict)
        new_totals: Dict[Tuple[int, ...], float] = defaultdict(float)
        kt_old = self._key_totals
        for fk, post in self._postings.items():
            if len(fk) >= 3 and fk[0] >= 12:
                order = fk[0] - 10
                thresh = t_ord2 if order == 2 else (t_ord3 if order == 3 else t_ord4)
            else:
                thresh = t_ord1
            if thresh <= 0.0:
                new_postings[fk] = post
                new_totals[fk] = kt_old.get(fk, 0.0)
                continue
            kept: Dict[int, object] = {}
            removed_mass = 0.0
            for act, bkt in post.items():
                cnt = float(bkt[0]) if isinstance(bkt, list) else float(bkt)
                if cnt <= thresh:
                    removed_mass += cnt
                    evicted += 1
                else:
                    kept[act] = bkt
            if kept:
                # Re-insert compacted inner dict only if entries were removed, else reuse
                new_postings[fk] = kept if removed_mass > 0.0 else post
                new_totals[fk] = max(kt_old.get(fk, 0.0) - removed_mass, 0.0)
        if evicted > 0:
            # Replace outer dicts cleanly to eliminate CPython hash-table tombstones
            self._postings = new_postings
            self._key_totals = new_totals
            self.__dict__.pop("_cached_proposal", None)
        return evicted

    def _propose_interpolated_ad(
        self,
        sparse_keys: Sequence[Tuple[int, ...]],
        k_req: int,
        discount: float = 0.5,
    ) -> Tuple[List[int], np.ndarray]:
        """Exact top-down Interpolated Absolute Discounting over nested suffix keys."""
        probs: Dict[int, float] = defaultdict(float)
        weight = 1.0
        d = float(discount)
        for fk in _ordered_suffix_keys(sparse_keys):
            post = self._postings.get(fk)
            tot = self._key_totals.get(fk, 0.0)
            if not post or tot <= 0.0:
                continue
            inv_tot = 1.0 / tot
            for act, bkt in post.items():
                if isinstance(bkt, list):
                    cnt, s_sum = float(bkt[0]), float(bkt[1])
                    eff = max(cnt - d, 0.0)
                    if eff > 0.0:
                        pos_rate = max(s_sum / cnt, 0.0)
                        if pos_rate > 0.0:
                            probs[act] += weight * (eff * inv_tot) * pos_rate
                else:
                    cnt = float(bkt)
                    eff = max(cnt - d, 0.0)
                    if eff > 0.0:
                        probs[act] += weight * (eff * inv_tot)
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

    def propose_cached(
        self,
        sparse_keys: Sequence[Tuple[int, ...]],
        k: int,
        discount: float = 0.5,
    ) -> Tuple[List[int], np.ndarray]:
        """Reuse suffix queries across lookahead branches on the same fitted index."""
        query = getattr(self, "_cached_proposal", None)
        if query is None:
            query = self.__dict__.setdefault(
                "_cached_proposal", self._build_cached_proposal()
            )
        k_req = min(max(int(k), 1), len(self.action_space))
        ranked, probs = query(tuple(sparse_keys), k_req, float(discount))
        # Preserve the uncached API's caller-owned results.
        return ranked.copy(), probs.copy()

    def _build_cached_proposal(self):
        scalar = self._propose_interpolated_ad
        size = len(self.action_space)
        if not 256 <= size <= 65_536 or self.action_space != tuple(range(size)):
            return lru_cache(maxsize=4096)(scalar)

        packed = _PackedSuffixQuery(self)

        @lru_cache(maxsize=4096)
        def query(keys, k, discount):
            result = packed.propose(keys, k, discount)
            return scalar(keys, k, discount) if result is None else result

        return query

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
            for act, bkt in post.items():
                if isinstance(bkt, list):
                    counts[act] += float(bkt[0])
                    scores[act] += float(bkt[1])
                else:
                    val = float(bkt)
                    counts[act] += val
                    scores[act] += val

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
