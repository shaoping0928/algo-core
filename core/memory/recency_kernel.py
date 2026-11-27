"""Task-agnostic Self-Learned Recency-Lag Kernel (Hawkes / Renewal Modulation).

Learns from raw discrete event/symbol/action streams whether a symbol ``s``
(or transition ``(prev, s)``) that occurred ``lag`` steps earlier in the recent
history is statistically **excited** (positive log-ratio: hold-action, state
persistence, periodic recurrence, topic continuity) or **inhibited** (negative
log-ratio: synaptic refractory period, inhibition of return, phrase non-repetition).

Crucially, this module carries **zero sign prior** and **zero language assumptions**:
  - Under an i.i.d. / memoryless Markov stream, expected hits equal observed hits,
    so all learned log-ratios converge to ``0.0``.
  - On action-persistence or cyclic tasks, it automatically learns **positive**
    weights at the task's natural persistence/period lags.
  - On natural sequences with short-range refractoriness and mid-range recurrence,
    it automatically learns **negative** weights at short lags and **positive**
    weights at longer lags.
"""
from __future__ import annotations

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
    """Unbiased empirical & online-updatable recency-lag log-ratio modulator."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 1,
        *,
        lag_bins: Sequence[Tuple[int, int]] = DEFAULT_LAG_BINS,
        prior_strength: float = 8.0,
        max_abs_log_ratio: float = 2.5,
        class_weight: float = 0.65,
        transition_weight: float = 0.85,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.n_classes = max(int(n_classes), 1)
        self.lag_bins: Tuple[Tuple[int, int], ...] = tuple((int(lo), int(hi)) for lo, hi in lag_bins)
        self.n_bins = len(self.lag_bins)
        self.max_lag = max((hi for _, hi in self.lag_bins), default=1)
        self.prior_strength = float(prior_strength)
        self.max_abs_log_ratio = float(max_abs_log_ratio)
        self.class_weight = float(class_weight)
        self.transition_weight = float(transition_weight)

        # Fast lag -> bin_id lookup table for lags 1..max_lag
        self._lag_to_bin = np.full(self.max_lag + 1, -1, dtype=np.int32)
        for b_idx, (lo, hi) in enumerate(self.lag_bins):
            for l_val in range(max(lo, 1), min(hi, self.max_lag) + 1):
                self._lag_to_bin[l_val] = b_idx

        # Observed and null-expected hit counts (zero prior bias -> ratio = 1.0 -> log = 0.0)
        self._obs_class = np.zeros((self.n_classes + 1, self.n_bins), dtype=float)
        self._exp_class = np.zeros((self.n_classes + 1, self.n_bins), dtype=float)
        self._obs_global = np.zeros(self.n_bins, dtype=float)
        self._exp_global = np.zeros(self.n_bins, dtype=float)
        self._obs_trans = np.zeros(self.n_bins, dtype=float)
        self._exp_trans = np.zeros(self.n_bins, dtype=float)

        # Learned log-ratio tables (0.0 = neutral)
        self.class_log_ratio_: np.ndarray = np.zeros((self.n_classes + 1, self.n_bins), dtype=float)
        self.global_log_ratio_: np.ndarray = np.zeros(self.n_bins, dtype=float)
        self.trans_log_ratio_: np.ndarray = np.zeros(self.n_bins, dtype=float)

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

        # Hierarchical shrinkage: class log-ratio shrinks smoothly toward global when sparse
        c_obs = self._obs_class + ps * (self._obs_global[None, :] + 1.0) / np.maximum(
            self._exp_global[None, :] + 1.0, 1e-6
        )
        c_exp = self._exp_class + ps
        c_ratio = c_obs / np.maximum(c_exp, 1e-6)
        self.class_log_ratio_ = np.clip(np.log(np.maximum(c_ratio, 1e-6)), -cap, cap)

        t_ratio = (self._obs_trans + ps) / (self._exp_trans + ps)
        self.trans_log_ratio_ = np.clip(np.log(np.maximum(t_ratio, 1e-6)), -cap, cap)

    def fit(
        self,
        tokens: Sequence[int],
        *,
        sym_to_class: Optional[np.ndarray] = None,
        local_index=None,
        max_samples: int = 120000,
    ) -> "LearnedRecencyLagKernel":
        """Estimate unbiased symbol-lag and transition-lag log-ratios from a symbol stream."""
        arr = np.clip(
            np.asarray(tokens, dtype=np.int32).reshape(-1),
            0,
            self.vocab_size - 1,
        )
        n = len(arr)
        if n <= self.max_lag + 2:
            return self

        if sym_to_class is not None:
            stc = np.asarray(sym_to_class, dtype=np.int32).reshape(-1)
            self.sym_to_class_ = np.zeros(self.vocab_size, dtype=np.int32)
            lim = min(len(stc), self.vocab_size)
            self.sym_to_class_[:lim] = np.clip(stc[:lim], 0, self.n_classes)
        else:
            self.sym_to_class_ = np.ones(self.vocab_size, dtype=np.int32)
            self.sym_to_class_[0] = 0

        # Empirical intra-class probability P(s | class(s))
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

        # Build fast bigram lookup for transition-lag null expectation if local_index not provided
        local_post = getattr(local_index, "_postings", None) if local_index is not None else None
        local_tot = getattr(local_index, "_key_totals", None) if local_index is not None else None
        fallback_bg: Optional[Dict[int, Dict[int, float]]] = None
        fallback_tot: Optional[Dict[int, float]] = None
        if local_post is None or local_tot is None:
            fallback_bg = {}
            fallback_tot = {}
            valid = (arr[:-1] > 0) & (arr[1:] > 0)
            p_arr = arr[:-1][valid]
            n_arr = arr[1:][valid]
            for a_i, b_i in zip(p_arr.tolist(), n_arr.tolist()):
                d = fallback_bg.get(a_i)
                if d is None:
                    d = {}
                    fallback_bg[a_i] = d
                d[b_i] = d.get(b_i, 0.0) + 1.0
                fallback_tot[a_i] = fallback_tot.get(a_i, 0.0) + 1.0

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
                return float(bkt[0]) / float(tot) if bkt is not None else 0.0
            assert fallback_bg is not None and fallback_tot is not None
            tot = fallback_tot.get(prev_s, 0.0)
            if tot <= 0.0:
                return 0.0
            return fallback_bg.get(prev_s, {}).get(next_s, 0.0) / tot

        self._obs_class.fill(0.0)
        self._exp_class.fill(0.0)
        self._obs_global.fill(0.0)
        self._exp_global.fill(0.0)
        self._obs_trans.fill(0.0)
        self._exp_trans.fill(0.0)

        ml = self.max_lag
        span = n - ml
        stride = max(span // max(int(max_samples), 1000), 1)
        stc = self.sym_to_class_
        pic = self._p_in_class
        l2b = self._lag_to_bin

        for t in range(ml, n, stride):
            tgt = int(arr[t])
            if tgt <= 0:
                continue
            c_tgt = int(stc[tgt])
            prev = int(arr[t - 1])

            # Single backward scan over recent window [t-ml .. t-1]:
            # tracks first (most recent) lag of symbols in c_tgt and transitions (prev -> u)
            seen_sym_class = set()
            seen_trans = set()
            tgt_found = False
            trans_tgt_found = False

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

                # 2. Transition-level recency: did (prev -> next_past) occur at this lag?
                if prev > 0 and lag >= 1 and (t - lag - 1) >= 0 and int(arr[t - lag - 1]) == prev:
                    if s_past not in seen_trans:
                        seen_trans.add(s_past)
                        p_tr = _bigram_prob(prev, s_past)
                        if p_tr > 0.0:
                            self._exp_trans[b_idx] += p_tr
                            if not trans_tgt_found and s_past == tgt:
                                self._obs_trans[b_idx] += 1.0
                                trans_tgt_found = True

        self._recompute_tables()
        self._fitted = True
        return self

    def modulate_scores(
        self,
        recent_history: Sequence[int],
        score_map: Dict[int, float],
        *,
        strength: float = 1.0,
    ) -> Dict[int, float]:
        """Multiply candidate probabilities by learned recency-lag likelihood ratios.

        Applies both:
          - symbol-class lag ratio ``exp(class_weight * class_log_ratio_[c, bin])``
          - transition lag ratio ``exp(transition_weight * trans_log_ratio_[bin])``
        Strictly data-driven: excites when learned ratio > 0, inhibits when < 0,
        and leaves unseen candidates completely untouched.
        """
        if not self._fitted or not recent_history or not score_map or float(strength) <= 0.0:
            return score_map

        hist = [int(x) for x in recent_history[-self.max_lag :]]
        hlen = len(hist)
        prev = hist[-1] if hlen >= 1 else 0
        l2b = self._lag_to_bin

        # Collect most-recent lag bin for every symbol & (prev -> sym) transition in history
        sym_bin: Dict[int, int] = {}
        trans_bin: Dict[int, int] = {}
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
            if prev > 0 and idx >= 1 and hist[idx - 1] == prev and s not in trans_bin:
                trans_bin[s] = b

        if not sym_bin and not trans_bin:
            return score_map

        cw = self.class_weight * float(strength)
        tw = self.transition_weight * float(strength)
        stc = self.sym_to_class_
        clr = self.class_log_ratio_
        tlr = self.trans_log_ratio_

        out: Dict[int, float] = {}
        for a, sc in score_map.items():
            ai = int(a)
            delta_log = 0.0
            sb = sym_bin.get(ai, -1)
            if sb >= 0 and 0 < ai < self.vocab_size:
                c = int(stc[ai])
                delta_log += cw * float(clr[c, sb])
            tb = trans_bin.get(ai, -1)
            if tb >= 0:
                delta_log += tw * float(tlr[tb])
            if delta_log != 0.0:
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
        )
