"""Task-agnostic Self-Learned Recency-Lag & Count-Saturation Kernel.

Learns from raw discrete event/symbol/action streams whether a symbol ``s``
or transition ``(prev -> s)`` / ``(prev2, prev1 -> s)`` that occurred ``lag``
steps earlier (and ``count`` times inside the recent saturation window) is
statistically **excited** (hold-action, state persistence, periodic cycle,
first bursty recurrence) or **inhibited / saturated** (refractory period,
inhibition of return, diminishing marginal recurrence after repeated clauses).

Zero sign prior and zero language assumptions:
  - Under an i.i.d. / memoryless Markov stream, expected hits equal observed
    hits across every ``(lag_bin, count_bin)``, so all log-ratios converge to 0.
  - On action-persistence or periodic tasks, high counts stay strongly positive.
  - On bursty natural streams, count=1 recurrence learns positive burstiness
    while higher-order/higher-count repetitions saturate smoothly toward 0/negative.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


DEFAULT_LAG_BINS: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (2, 4),
    (5, 12),
    (13, 32),
    (33, 80),
)


class LearnedRecencyLagKernel:
    """Unbiased empirical recency-lag + count-saturation log-ratio modulator."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 1,
        *,
        lag_bins: Sequence[Tuple[int, int]] = DEFAULT_LAG_BINS,
        sat_window: int = 36,
        n_count_bins: int = 3,
        prior_strength: float = 8.0,
        max_abs_log_ratio: float = 2.5,
        class_weight: float = 0.65,
        transition_weight: float = 0.75,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.n_classes = max(int(n_classes), 1)
        self.lag_bins: Tuple[Tuple[int, int], ...] = tuple((int(lo), int(hi)) for lo, hi in lag_bins)
        self.n_bins = len(self.lag_bins)
        self.max_lag = max((hi for _, hi in self.lag_bins), default=1)
        self.sat_window = min(max(int(sat_window), 8), self.max_lag)
        self.n_count_bins = max(int(n_count_bins), 2)  # cnt = 1, 2, >=3
        self.prior_strength = float(prior_strength)
        self.max_abs_log_ratio = float(max_abs_log_ratio)
        self.class_weight = float(class_weight)
        self.transition_weight = float(transition_weight)

        self._lag_to_bin = np.full(self.max_lag + 1, -1, dtype=np.int32)
        for b_idx, (lo, hi) in enumerate(self.lag_bins):
            for l_val in range(max(lo, 1), min(hi, self.max_lag) + 1):
                self._lag_to_bin[l_val] = b_idx

        # Observed and null-expected counts
        self._obs_class = np.zeros((self.n_classes + 1, self.n_bins), dtype=float)
        self._exp_class = np.zeros((self.n_classes + 1, self.n_bins), dtype=float)
        self._obs_global = np.zeros(self.n_bins, dtype=float)
        self._exp_global = np.zeros(self.n_bins, dtype=float)
        # 2-step transition recurrence conditioned on (lag_bin, count_bin)
        self._obs_trans = np.zeros((self.n_bins, self.n_count_bins), dtype=float)
        self._exp_trans = np.zeros((self.n_bins, self.n_count_bins), dtype=float)
        # 3-step transition recurrence conditioned on count_bin inside sat_window
        self._obs_trans3 = np.zeros(self.n_count_bins, dtype=float)
        self._exp_trans3 = np.zeros(self.n_count_bins, dtype=float)

        # Learned log-ratio tables (0.0 = unbiased neutral)
        self.class_log_ratio_: np.ndarray = np.zeros((self.n_classes + 1, self.n_bins), dtype=float)
        self.global_log_ratio_: np.ndarray = np.zeros(self.n_bins, dtype=float)
        self.trans_log_ratio_: np.ndarray = np.zeros(self.n_bins, dtype=float)
        self.trans_count_log_ratio_: np.ndarray = np.zeros((self.n_bins, self.n_count_bins), dtype=float)
        self.trans3_count_log_ratio_: np.ndarray = np.zeros(self.n_count_bins, dtype=float)

        self.sym_to_class_: np.ndarray = np.zeros(self.vocab_size, dtype=np.int32)
        self._p_in_class: np.ndarray = np.zeros(self.vocab_size, dtype=float)
        self._fitted: bool = False

    def lag_to_bin(self, lag: int) -> int:
        lv = int(lag)
        if lv <= 0 or lv > self.max_lag:
            return -1
        return int(self._lag_to_bin[lv])

    def _recompute_tables(self) -> None:
        ps = self.prior_strength
        cap = self.max_abs_log_ratio
        g_ratio = (self._obs_global + ps) / (self._exp_global + ps)
        self.global_log_ratio_ = np.clip(np.log(np.maximum(g_ratio, 1e-6)), -cap, cap)

        c_obs = self._obs_class + ps * (self._obs_global[None, :] + 1.0) / np.maximum(
            self._exp_global[None, :] + 1.0, 1e-6
        )
        c_exp = self._exp_class + ps
        c_ratio = c_obs / np.maximum(c_exp, 1e-6)
        self.class_log_ratio_ = np.clip(np.log(np.maximum(c_ratio, 1e-6)), -cap, cap)

        # Aggregate 2-step lag ratio + count-conditioned 2-step ratio
        t_obs_agg = self._obs_trans.sum(axis=1)
        t_exp_agg = self._exp_trans.sum(axis=1)
        t_ratio = (t_obs_agg + ps) / (t_exp_agg + ps)
        self.trans_log_ratio_ = np.clip(np.log(np.maximum(t_ratio, 1e-6)), -cap, cap)

        tc_ratio = (self._obs_trans + ps) / (self._exp_trans + ps)
        self.trans_count_log_ratio_ = np.clip(np.log(np.maximum(tc_ratio, 1e-6)), -cap, cap)

        # Count-conditioned 3-step transition ratio (captures clause/pattern saturation)
        t3_ratio = (self._obs_trans3 + ps) / (self._exp_trans3 + ps)
        self.trans3_count_log_ratio_ = np.clip(np.log(np.maximum(t3_ratio, 1e-6)), -cap, cap)

    def fit(
        self,
        tokens: Sequence[int],
        *,
        sym_to_class: Optional[np.ndarray] = None,
        local_index=None,
        max_samples: int = 120000,
    ) -> "LearnedRecencyLagKernel":
        """Estimate unbiased symbol-lag and count-conditioned transition-lag log-ratios."""
        arr = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        n = len(arr)
        if n <= self.max_lag + 3:
            return self

        if sym_to_class is not None:
            stc = np.asarray(sym_to_class, dtype=np.int32).reshape(-1)
            self.sym_to_class_ = np.zeros(self.vocab_size, dtype=np.int32)
            lim = min(len(stc), self.vocab_size)
            self.sym_to_class_[:lim] = np.clip(stc[:lim], 0, self.n_classes)
        else:
            self.sym_to_class_ = np.ones(self.vocab_size, dtype=np.int32)
            self.sym_to_class_[0] = 0

        counts = np.bincount(arr, minlength=self.vocab_size).astype(float)
        counts[0] = 0.0
        class_totals = np.zeros(self.n_classes + 1, dtype=float)
        for s in range(1, self.vocab_size):
            c = int(self.sym_to_class_[s])
            class_totals[c] += counts[s]
        self._p_in_class = np.zeros(self.vocab_size, dtype=float)
        for s in range(1, self.vocab_size):
            c = int(self.sym_to_class_[s])
            if class_totals[c] > 0.0:
                self._p_in_class[s] = counts[s] / class_totals[c]

        local_post = getattr(local_index, "_postings", None) if local_index is not None else None
        local_tot = getattr(local_index, "_key_totals", None) if local_index is not None else None
        fallback_bg: Optional[Dict[int, Dict[int, float]]] = None
        fallback_tot: Optional[Dict[int, float]] = None
        if local_post is None or local_tot is None:
            fallback_bg = defaultdict(dict)
            fallback_tot = defaultdict(float)
            valid = (arr[:-1] > 0) & (arr[1:] > 0)
            for a_i, b_i in zip(arr[:-1][valid].tolist(), arr[1:][valid].tolist()):
                d = fallback_bg[a_i]
                d[b_i] = d.get(b_i, 0.0) + 1.0
                fallback_tot[a_i] += 1.0

        # Build fast strided trigram baseline for exact 3-step null expectation
        tg_map: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
        tg_tot: Dict[Tuple[int, int], float] = defaultdict(float)
        tg_stride = max(n // 500000, 1)
        v3 = (arr[:-2:tg_stride] > 0) & (arr[1:-1:tg_stride] > 0) & (arr[2::tg_stride] > 0)
        for a_i, b_i, c_i in zip(
            arr[:-2:tg_stride][v3].tolist(),
            arr[1:-1:tg_stride][v3].tolist(),
            arr[2::tg_stride][v3].tolist(),
        ):
            k2 = (a_i, b_i)
            d3 = tg_map[k2]
            d3[c_i] = d3.get(c_i, 0.0) + 1.0
            tg_tot[k2] += 1.0

        def _bigram_prob(prev_s: int, next_s: int) -> float:
            if local_post is not None and local_tot is not None:
                key = (-1, prev_s)
                tot = local_tot.get(key, 0.0)
                if tot <= 0.0:
                    return 0.0
                post = local_post.get(key)
                if not post:
                    return 0.0
                bkt = post.get(next_s)
                if bkt is None:
                    return 0.0
                cnt = float(bkt[0]) if isinstance(bkt, list) else float(bkt)
                return cnt / float(tot)
            assert fallback_bg is not None and fallback_tot is not None
            tot = fallback_tot.get(prev_s, 0.0)
            if tot <= 0.0:
                return 0.0
            return fallback_bg.get(prev_s, {}).get(next_s, 0.0) / tot

        def _trigram_prob(p2_s: int, p1_s: int, next_s: int) -> float:
            k2 = (p2_s, p1_s)
            tot = tg_tot.get(k2, 0.0)
            if tot <= 0.0:
                return _bigram_prob(p1_s, next_s)
            return tg_map[k2].get(next_s, 0.0) / tot

        self._obs_class.fill(0.0)
        self._exp_class.fill(0.0)
        self._obs_global.fill(0.0)
        self._exp_global.fill(0.0)
        self._obs_trans.fill(0.0)
        self._exp_trans.fill(0.0)
        self._obs_trans3.fill(0.0)
        self._exp_trans3.fill(0.0)

        ml = self.max_lag
        sw = self.sat_window
        nc = self.n_count_bins
        span = n - ml - 2
        stride = max(span // max(int(max_samples), 1000), 1)
        stc = self.sym_to_class_
        pic = self._p_in_class
        l2b = self._lag_to_bin

        for t in range(ml + 2, n, stride):
            tgt = int(arr[t])
            if tgt <= 0:
                continue
            c_tgt = int(stc[tgt])
            prev1 = int(arr[t - 1])
            prev2 = int(arr[t - 2])

            seen_sym_class = set()
            tgt_found = False
            trans2_first_bin: Dict[int, int] = {}
            trans2_cnt: Dict[int, int] = {}
            trans3_cnt: Dict[int, int] = {}

            for lag in range(1, ml + 1):
                s_past = int(arr[t - lag])
                if s_past <= 0:
                    continue
                b_idx = int(l2b[lag])
                if b_idx < 0:
                    continue

                # 1. Symbol-level recency within target class
                if int(stc[s_past]) == c_tgt and s_past not in seen_sym_class:
                    seen_sym_class.add(s_past)
                    p_u = float(pic[s_past])
                    self._exp_class[c_tgt, b_idx] += p_u
                    self._exp_global[b_idx] += p_u
                    if not tgt_found and s_past == tgt:
                        self._obs_class[c_tgt, b_idx] += 1.0
                        self._obs_global[b_idx] += 1.0
                        tgt_found = True

                # 2. 2-step transition recurrence + count inside sat_window
                if prev1 > 0 and int(arr[t - lag - 1]) == prev1:
                    if s_past not in trans2_first_bin:
                        trans2_first_bin[s_past] = b_idx
                    if lag <= sw:
                        trans2_cnt[s_past] = trans2_cnt.get(s_past, 0) + 1
                    elif s_past not in trans2_cnt:
                        trans2_cnt[s_past] = 1
                    # 3. 3-step transition recurrence inside sat_window
                    if prev2 > 0 and lag <= sw and int(arr[t - lag - 2]) == prev2:
                        trans3_cnt[s_past] = trans3_cnt.get(s_past, 0) + 1

            for u, b_idx in trans2_first_bin.items():
                p_tr = _bigram_prob(prev1, u)
                if p_tr > 0.0:
                    cb = min(max(trans2_cnt.get(u, 1) - 1, 0), nc - 1)
                    self._exp_trans[b_idx, cb] += p_tr
                    if u == tgt:
                        self._obs_trans[b_idx, cb] += 1.0

            for u, c3 in trans3_cnt.items():
                p_t3 = _trigram_prob(prev2, prev1, u)
                if p_t3 > 0.0:
                    cb3 = min(max(c3 - 1, 0), nc - 1)
                    self._exp_trans3[cb3] += p_t3
                    if u == tgt:
                        self._obs_trans3[cb3] += 1.0

        self._recompute_tables()
        self._fitted = True
        return self

    def modulate_scores(
        self,
        recent_history: Sequence[int],
        score_map: Dict[int, float],
        *,
        strength: float = 1.0,
        excitation_scale: float = 1.0,
    ) -> Dict[int, float]:
        """Multiply candidate scores by learned lag + count-saturation likelihood ratios.

        ``excitation_scale`` scales only positive self-excitation (``delta_log > 0``)
        while preserving data-learned negative refractory/saturation inhibition at
        full strength — preventing runaway positive-feedback loops during open-loop
        self-generation without altering external-observation inference.
        """
        if not self._fitted or not recent_history or not score_map or float(strength) <= 0.0:
            return score_map

        hist = [int(x) for x in recent_history[-self.max_lag :]]
        hlen = len(hist)
        prev1 = hist[-1] if hlen >= 1 else 0
        prev2 = hist[-2] if hlen >= 2 else 0
        l2b = self._lag_to_bin
        sw = self.sat_window
        nc = self.n_count_bins

        sym_bin: Dict[int, int] = {}
        trans2_bin: Dict[int, int] = {}
        trans2_cnt: Dict[int, int] = {}
        trans3_cnt: Dict[int, int] = {}

        for lag in range(1, min(hlen, self.max_lag) + 1):
            idx = hlen - lag
            s = hist[idx]
            if s <= 0:
                continue
            b = int(l2b[lag])
            if b < 0:
                continue
            if s not in sym_bin:
                sym_bin[s] = b
            if prev1 > 0 and idx >= 1 and hist[idx - 1] == prev1:
                if s not in trans2_bin:
                    trans2_bin[s] = b
                if lag <= sw:
                    trans2_cnt[s] = trans2_cnt.get(s, 0) + 1
                elif s not in trans2_cnt:
                    trans2_cnt[s] = 1
                if prev2 > 0 and lag <= sw and idx >= 2 and hist[idx - 2] == prev2:
                    trans3_cnt[s] = trans3_cnt.get(s, 0) + 1

        if not sym_bin and not trans2_bin:
            return score_map

        cw = self.class_weight * float(strength)
        tw = self.transition_weight * float(strength)
        stc = self.sym_to_class_
        clr = self.class_log_ratio_
        tclr = self.trans_count_log_ratio_
        t3lr = self.trans3_count_log_ratio_

        out: Dict[int, float] = {}
        for a, sc in score_map.items():
            ai = int(a)
            delta_log = 0.0
            sb = sym_bin.get(ai, -1)
            if sb >= 0 and 0 < ai < self.vocab_size:
                c = int(stc[ai])
                delta_log += cw * float(clr[c, sb])

            # Use finer-grained 3-step saturation ratio when (prev2, prev1 -> a) recurs;
            # otherwise fall back to count-conditioned 2-step lag ratio (zero double stacking).
            c3 = trans3_cnt.get(ai, 0)
            if c3 > 0:
                cb3 = min(c3 - 1, nc - 1)
                delta_log += tw * float(t3lr[cb3])
            else:
                tb = trans2_bin.get(ai, -1)
                if tb >= 0:
                    cb2 = min(max(trans2_cnt.get(ai, 1) - 1, 0), nc - 1)
                    delta_log += tw * float(tclr[tb, cb2])

            if delta_log != 0.0:
                if delta_log > 0.0 and excitation_scale != 1.0:
                    delta_log *= float(np.clip(excitation_scale, 0.0, 1.0))
                delta_log = float(np.clip(delta_log, -3.5, 3.5))
                out[ai] = float(sc) * float(np.exp(delta_log))
            else:
                out[ai] = float(sc)
        return out

    def snapshot(self) -> dict:
        return dict(
            fitted=bool(self._fitted),
            lag_bins=list(self.lag_bins),
            global_log_ratio=[round(float(v), 3) for v in self.global_log_ratio_],
            trans_log_ratio=[round(float(v), 3) for v in self.trans_log_ratio_],
            trans3_count_log_ratio=[round(float(v), 3) for v in self.trans3_count_log_ratio_],
        )
