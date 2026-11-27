"""Calibrated macro-action regime transition models and online continuation updaters."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from failure_memory import TrunkPointFailureMemory
from summarizer import RegimeStage, StageRegimeSummarizer, UNKNOWN


def _dict_to_pmf_array(dist: Mapping[int, float]) -> np.ndarray:
    if not dist:
        return np.asarray([1.0], dtype=float)
    max_t = max(int(t) for t in dist)
    arr = np.zeros(max_t + 1, dtype=float)
    for t, p in dist.items():
        if p > 0.0:
            arr[int(t)] += float(p)
    return arr


def _convolve_arrays(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    max_support: int = 512,
) -> np.ndarray:
    """Fast exact discrete convolution of two non-negative integer PMF arrays."""
    conv = np.convolve(arr_a, arr_b)
    if len(conv) > max_support:
        tail = float(conv[max_support - 1:].sum())
        conv = conv[:max_support].copy()
        conv[-1] = tail
    return conv


def _add_scaled_pmf(target: Optional[np.ndarray], src: np.ndarray, scale: float) -> np.ndarray:
    if target is None:
        return src * scale
    if len(src) <= len(target):
        target[:len(src)] += src * scale
        return target
    expanded = np.zeros(len(src), dtype=float)
    expanded[:len(target)] = target
    expanded += src * scale
    return expanded


class CalibratedRegimeTransitionModel:
    """Controllable stage transition model with nested OOF episode shrinkage.

    Learns ``(regime_context, action_profile) -> (outcome, duration, feedback)``
    strictly from realized transitions. Supports both action-conditioned and
    action-pooled queries, and causal multi-step rollout over ``k`` stages with
    discrete stop-time distribution convolution.
    """

    def __init__(
        self,
        max_depth: int = 2,
        min_episodes: int = 2,
        alphas: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
        oof_folds: int = 4,
    ):
        if int(max_depth) < 1:
            raise ValueError("max_depth must be positive")
        if int(min_episodes) < 1:
            raise ValueError("min_episodes must be positive")
        self.max_depth = int(max_depth)
        self.min_episodes = int(min_episodes)
        self.alphas = tuple(float(a) for a in alphas)
        if not self.alphas or any(a < 0.0 for a in self.alphas):
            raise ValueError("alphas must be non-empty and non-negative")
        self.oof_folds = int(oof_folds)
        self.fitted = False

    @staticmethod
    def realized_outcome(
        stages: Sequence[RegimeStage], index: int,
    ) -> Tuple[str, Optional[int]]:
        stage = stages[index]
        if stage.terminal:
            return ("terminal", None)
        if stage.censored:
            return ("censored", None)
        if index + 1 < len(stages):
            return ("regime", int(stages[index + 1].regime))
        return ("censored", None)

    def _accumulate(
        self,
        episodes: Sequence[Sequence[RegimeStage]],
        use_action: bool,
    ):
        counts = defaultdict(Counter)
        ep_sets = defaultdict(set)
        durations = defaultdict(Counter)
        net_fb_sums = defaultdict(float)
        net_fb_counts = defaultdict(int)

        def record(key, out, ep_id, dur, net_fb):
            counts[key][out] += 1
            ep_sets[key].add(ep_id)
            durations[key][int(dur)] += 1
            net_fb_sums[key] += float(net_fb)
            net_fb_counts[key] += 1

        for stages in episodes:
            for idx, stage in enumerate(stages):
                out = self.realized_outcome(stages, idx)
                if out[0] == "censored":
                    continue
                fb = np.asarray(stage.feedback, dtype=float).reshape(-1)
                net_fb = float(fb[0] - fb[1]) if fb.size >= 2 else (
                    float(fb[0]) if fb.size else 0.0
                )
                hist = tuple(int(s.regime) for s in stages[:idx + 1])
                act = stage.action_profile if use_action else None
                for depth in range(min(len(hist), self.max_depth), -1, -1):
                    ctx = hist[-depth:] if depth else ()
                    record((ctx, act), out, stage.episode, stage.duration, net_fb)
                if use_action:
                    record(((), None), out, stage.episode, stage.duration, net_fb)
                    for depth in range(min(len(hist), self.max_depth), 0, -1):
                        ctx = hist[-depth:]
                        record(
                            (ctx, None), out, stage.episode, stage.duration, net_fb,
                        )
        return counts, ep_sets, durations, net_fb_sums, net_fb_counts

    @staticmethod
    def _normalize(counter: Counter) -> dict:
        total = sum(counter.values())
        if total <= 0:
            return {}
        return {key: float(val) / float(total) for key, val in counter.items()}

    def _predict_internal(
        self,
        counts,
        ep_sets,
        durations,
        net_fb_sums,
        net_fb_counts,
        history: Sequence[int],
        action_profile: Optional[Tuple[Hashable, ...]],
        alpha: float,
        use_action: bool,
    ) -> dict:
        hist = tuple(int(r) for r in history)
        act = action_profile if (use_action and action_profile is not None) else None
        prior_key = (
            ((), act)
            if (act is not None and len(ep_sets.get(((), act), ())) >= 1)
            else ((), None)
        )
        prior_dist = self._normalize(counts[prior_key])
        if not prior_dist:
            return dict(
                status=UNKNOWN, distribution={}, context=(),
                context_depth=0, episode_support=0, action_conditioned=False,
                duration_distribution={}, expected_net_feedback=0.0,
            )

        chosen_key = None
        chosen_depth = 0
        chosen_act_cond = False
        if act is not None:
            for depth in range(min(len(hist), self.max_depth), 0, -1):
                ctx = hist[-depth:]
                key = (ctx, act)
                if len(ep_sets.get(key, ())) >= self.min_episodes:
                    chosen_key = key
                    chosen_depth = depth
                    chosen_act_cond = True
                    break
        if chosen_key is None:
            for depth in range(min(len(hist), self.max_depth), 0, -1):
                ctx = hist[-depth:]
                key = (ctx, None)
                if len(ep_sets.get(key, ())) >= self.min_episodes:
                    chosen_key = key
                    chosen_depth = depth
                    chosen_act_cond = False
                    break

        if chosen_key is None:
            dur_dist = self._normalize(durations[prior_key])
            exp_fb = net_fb_sums[prior_key] / max(net_fb_counts[prior_key], 1)
            return dict(
                status="supported",
                distribution=prior_dist,
                context=(),
                context_depth=0,
                episode_support=len(ep_sets.get(prior_key, ())),
                action_conditioned=False,
                duration_distribution=dur_dist,
                expected_net_feedback=float(exp_fb),
            )

        ep_sup = len(ep_sets[chosen_key])
        ctx_dist = self._normalize(counts[chosen_key])
        weight = 1.0 if alpha == 0.0 else float(ep_sup) / float(ep_sup + alpha)
        # Always include full-support prior_dist in the shrinkage target so
        # unseen context-specific outcomes never suffer zero-probability NLL blowups.
        all_outcomes = set(ctx_dist).union(prior_dist)
        shrunk = {
            outcome: (
                weight * ctx_dist.get(outcome, 0.0)
                + (1.0 - weight) * prior_dist.get(outcome, 0.0)
            )
            for outcome in all_outcomes
        }
        # Shrink duration distribution and expected net feedback with the same weight
        ctx_dur = self._normalize(durations[chosen_key])
        pri_dur = self._normalize(durations[prior_key])
        dur_dist = {
            t: weight * ctx_dur.get(t, 0.0) + (1.0 - weight) * pri_dur.get(t, 0.0)
            for t in set(ctx_dur).union(pri_dur)
        }
        ctx_fb = net_fb_sums[chosen_key] / max(net_fb_counts[chosen_key], 1)
        pri_fb = net_fb_sums[prior_key] / max(net_fb_counts[prior_key], 1)
        exp_fb = weight * ctx_fb + (1.0 - weight) * pri_fb
        return dict(
            status="supported",
            distribution=shrunk,
            context=chosen_key[0],
            context_depth=int(chosen_depth),
            episode_support=int(ep_sup),
            action_conditioned=bool(chosen_act_cond),
            duration_distribution=dur_dist,
            expected_net_feedback=float(exp_fb),
        )

    def fit(
        self,
        train_episodes: Sequence[Sequence[RegimeStage]],
        use_action: bool = True,
    ) -> "CalibratedRegimeTransitionModel":
        """Fit counts and select episode shrinkage alpha via nested train OOF."""
        episodes = [tuple(ep) for ep in train_episodes if ep]
        self.use_action = bool(use_action)
        if not episodes:
            self.alpha_ = self.alphas[0]
            self.counts_ = defaultdict(Counter)
            self.ep_sets_ = defaultdict(set)
            self.durations_ = defaultdict(Counter)
            self.net_fb_sums_ = defaultdict(float)
            self.net_fb_counts_ = defaultdict(int)
            self.oof_nll_by_alpha_ = {}
            self.fitted = True
            return self

        k_folds = max(2, min(self.oof_folds, len(episodes)))
        oof_scores = {}
        if len(episodes) >= 2 and len(self.alphas) > 1:
            fold_splits = [
                (
                    [ep for i, ep in enumerate(episodes) if i % k_folds != f],
                    [ep for i, ep in enumerate(episodes) if i % k_folds == f],
                )
                for f in range(k_folds)
            ]
            accumulated = [
                (self._accumulate(tr, use_action=self.use_action), va)
                for tr, va in fold_splits if tr and va
            ]
            for alpha in self.alphas:
                nlls = []
                for (cnt, eps, durs, fbs, fbc), va in accumulated:
                    for stages in va:
                        for idx, stage in enumerate(stages):
                            out = self.realized_outcome(stages, idx)
                            if out[0] == "censored":
                                continue
                            hist = tuple(int(s.regime) for s in stages[:idx + 1])
                            pred = self._predict_internal(
                                cnt, eps, durs, fbs, fbc, hist,
                                stage.action_profile, alpha, self.use_action,
                            )
                            if pred["status"] != "supported":
                                continue
                            p = max(pred["distribution"].get(out, 0.0), 1e-6)
                            nlls.append(-math.log(p))
                oof_scores[alpha] = (
                    float(np.mean(nlls)) if nlls else float("inf")
                )
            self.alpha_ = min(self.alphas, key=lambda a: (oof_scores[a], a))
        else:
            self.alpha_ = self.alphas[0]

        self.oof_nll_by_alpha_ = oof_scores
        (
            self.counts_,
            self.ep_sets_,
            self.durations_,
            self.net_fb_sums_,
            self.net_fb_counts_,
        ) = self._accumulate(episodes, use_action=self.use_action)
        self.fitted = True
        return self

    def predict(
        self,
        history: Sequence[int],
        action_profile: Optional[Tuple[Hashable, ...]] = None,
    ) -> dict:
        if not self.fitted:
            raise RuntimeError("fit CalibratedRegimeTransitionModel before predict")
        return self._predict_internal(
            self.counts_, self.ep_sets_, self.durations_,
            self.net_fb_sums_, self.net_fb_counts_,
            history, action_profile, self.alpha_, self.use_action,
        )

    def prior(
        self, action_profile: Optional[Tuple[Hashable, ...]] = None,
    ) -> dict:
        if not self.fitted:
            raise RuntimeError("fit CalibratedRegimeTransitionModel before prior")
        act = (
            action_profile
            if (self.use_action and action_profile is not None)
            else None
        )
        key = (
            ((), act)
            if (act is not None and len(self.ep_sets_.get(((), act), ())) >= 1)
            else ((), None)
        )
        return self._normalize(self.counts_[key])

    def rollout(
        self,
        history: Sequence[int],
        steps: int,
        action_profiles: Optional[Sequence[Optional[Tuple[Hashable, ...]]]] = None,
    ) -> dict:
        """Strictly causal k-stage rollout with stop-time convolution.

        ``action_profiles`` may be None (action-pooled rollout) or a sequence
        of ``steps`` macro-action profiles to evaluate a multi-stage plan.
        Probability mass is strictly conserved:
        ``sum(regime_distribution.values()) + terminal_probability + unknown_mass == 1.0``.
        """
        if not self.fitted:
            raise RuntimeError("fit CalibratedRegimeTransitionModel before rollout")
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be positive")
        if action_profiles is not None and len(action_profiles) != steps:
            raise ValueError("action_profiles length must match steps")

        init_hist = tuple(int(r) for r in history)[-self.max_depth:]
        active_mass = {init_hist: 1.0}
        # Track elapsed-time PMF as compact 1D NumPy arrays for fast convolution
        active_time_pmf = {init_hist: np.asarray([1.0], dtype=float)}
        terminated_pmf = None
        unknown_pmf = None

        terminal_prob = 0.0
        unknown_mass = 0.0
        expected_net_feedback = 0.0
        root_fallback_rate = 0.0
        pred_cache = {}

        for step_idx in range(steps):
            act = None if action_profiles is None else action_profiles[step_idx]
            next_mass = defaultdict(float)
            next_time_pmf = {}
            for h, mass in active_mass.items():
                if mass <= 1e-12:
                    continue
                ckey = (h, act)
                cached = pred_cache.get(ckey)
                if cached is None:
                    pred = self.predict(h, act)
                    dur_arr = (
                        _dict_to_pmf_array(pred["duration_distribution"])
                        if pred["status"] == "supported" else None
                    )
                    cached = (pred, dur_arr)
                    pred_cache[ckey] = cached
                pred, dur_arr = cached
                h_time = active_time_pmf[h]
                if pred["status"] != "supported":
                    unknown_mass += mass
                    unknown_pmf = _add_scaled_pmf(unknown_pmf, h_time, mass)
                    continue
                if pred["context_depth"] == 0:
                    root_fallback_rate += mass / float(steps)
                expected_net_feedback += mass * pred["expected_net_feedback"]
                convolved = _convolve_arrays(h_time, dur_arr)
                for (kind, dest), prob in pred["distribution"].items():
                    if prob <= 0.0:
                        continue
                    branch_mass = mass * prob
                    if kind == "terminal":
                        terminal_prob += branch_mass
                        terminated_pmf = _add_scaled_pmf(
                            terminated_pmf, convolved, branch_mass,
                        )
                    elif kind == "regime":
                        nxt_h = (h + (int(dest),))[-self.max_depth:]
                        next_mass[nxt_h] += branch_mass
                        next_time_pmf[nxt_h] = _add_scaled_pmf(
                            next_time_pmf.get(nxt_h), convolved, branch_mass,
                        )
                    else:
                        unknown_mass += branch_mass
                        unknown_pmf = _add_scaled_pmf(
                            unknown_pmf, convolved, branch_mass,
                        )
            active_mass = dict(next_mass)
            active_time_pmf = {
                h: arr / active_mass[h]
                for h, arr in next_time_pmf.items()
                if active_mass[h] > 0.0
            }

        regime_dist = defaultdict(float)
        total_pmf = None
        for h, mass in active_mass.items():
            regime_dist[h[-1]] += mass
            total_pmf = _add_scaled_pmf(total_pmf, active_time_pmf[h], mass)
        if terminated_pmf is not None:
            total_pmf = _add_scaled_pmf(total_pmf, terminated_pmf, 1.0)
        if unknown_pmf is not None:
            total_pmf = _add_scaled_pmf(total_pmf, unknown_pmf, 1.0)
        if total_pmf is None:
            total_pmf = np.asarray([1.0], dtype=float)

        nz = np.flatnonzero(total_pmf > 0.0)
        elapsed_dist = tuple((int(t), float(total_pmf[t])) for t in nz)
        expected_elapsed = float(np.dot(nz, total_pmf[nz])) if len(nz) else 0.0
        return dict(
            steps=steps,
            regime_distribution=dict(regime_dist),
            terminal_probability=float(terminal_prob),
            unknown_mass=float(unknown_mass),
            expected_net_feedback=float(expected_net_feedback),
            expected_elapsed_frames=expected_elapsed,
            elapsed_distribution=elapsed_dist,
            root_fallback_rate=float(root_fallback_rate),
        )

    def controllability_report(
        self, n_regimes: int, tv_threshold: float = 0.25,
    ) -> dict:
        """Measure how many regimes have >=2 supported action profiles with TV >= threshold."""
        profiles = sorted(
            {
                act for (ctx, act), eps in self.ep_sets_.items()
                if act is not None and len(eps) >= self.min_episodes
            },
            key=repr,
        )
        multi_profile_regimes = 0
        controllable_regimes = 0
        max_tvs = []
        for regime in range(int(n_regimes)):
            dists = []
            for prof in profiles:
                if len(self.ep_sets_.get(((regime,), prof), ())) >= self.min_episodes:
                    pred = self.predict((regime,), prof)
                    if pred["status"] == "supported" and pred["action_conditioned"]:
                        dists.append(pred["distribution"])
            if len(dists) >= 2:
                multi_profile_regimes += 1
                mtv = max(
                    0.5 * sum(
                        abs(dists[i].get(k, 0.0) - dists[j].get(k, 0.0))
                        for k in set(dists[i]).union(dists[j])
                    )
                    for i in range(len(dists))
                    for j in range(i + 1, len(dists))
                )
                max_tvs.append(mtv)
                if mtv >= float(tv_threshold):
                    controllable_regimes += 1
        return dict(
            n_regimes=int(n_regimes),
            supported_profiles=tuple(profiles),
            multi_profile_regimes=int(multi_profile_regimes),
            controllable_regimes=int(controllable_regimes),
            mean_max_tv=(
                float(np.mean(max_tvs)) if max_tvs else 0.0
            ),
        )

    def continuation_values(
        self,
        n_regimes: int,
        horizon_stages: int = 4,
        discount: float = 0.95,
        mode: str = "optimal",
        min_action_episodes: int = 3,
        lcb_z: float = 1.0,
    ) -> dict:
        """Finite-horizon dynamic programming over the calibrated regime model.

        Computes per-regime k-stage survival probability, discounted cumulative
        survival across stages ``1..k`` (normalized to ``[0, 1]``), and
        expected net feedback using only OOF-shrunk transition statistics.
        ``mode="pooled"`` evaluates the observed behavior distribution;
        ``mode="optimal"`` maximizes over supported macro-action profiles;
        ``mode="lcb"`` selects macro-action profiles using a conservative
        lower-confidence penalty ``lcb_z / sqrt(episode_support)`` anchored
        against the pooled baseline to curb small-sample optimism while
        exploiting well-supported controllable transitions.
        """
        if not self.fitted:
            raise RuntimeError("fit CalibratedRegimeTransitionModel first")
        if mode not in ("pooled", "optimal", "lcb"):
            raise ValueError("mode must be 'pooled', 'optimal', or 'lcb'")
        k_steps = int(horizon_stages)
        if k_steps < 1:
            raise ValueError("horizon_stages must be positive")
        gamma = float(discount)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if float(lcb_z) < 0.0:
            raise ValueError("lcb_z must be non-negative")
        n_reg = int(n_regimes)
        min_act_ep = max(int(self.min_episodes), int(min_action_episodes))

        profiles = sorted(
            {
                act for (_ctx, act), eps in self.ep_sets_.items()
                if act is not None and len(eps) >= min_act_ep
            },
            key=repr,
        )

        # Precompute depth-1 shrunk transitions for pooled & supported profiles
        transitions = {}
        regime_support = {}
        for r in range(n_reg):
            pool_pred = self.predict((r,), None)
            regime_support[r] = int(
                len(self.ep_sets_.get(((r,), None), ()))
            )
            cands = [(None, pool_pred, regime_support[r])]
            if mode in ("optimal", "lcb") and self.use_action:
                for prof in profiles:
                    n_ep = int(len(self.ep_sets_.get(((r,), prof), ())))
                    if n_ep >= min_act_ep:
                        pred = self.predict((r,), prof)
                        if (pred["status"] == "supported"
                                and pred["action_conditioned"]):
                            cands.append((prof, pred, n_ep))
            transitions[r] = cands

        surv = np.ones(n_reg, dtype=float)
        cum_stages = np.zeros(n_reg, dtype=float)
        ret = np.zeros(n_reg, dtype=float)
        best_profile = [None] * n_reg
        norm_factor = sum(gamma ** h for h in range(k_steps))

        for _ in range(k_steps):
            new_surv = np.zeros(n_reg, dtype=float)
            new_cum = np.zeros(n_reg, dtype=float)
            new_ret = np.zeros(n_reg, dtype=float)
            for r in range(n_reg):
                cands = transitions[r]
                if mode == "pooled":
                    _prof, pred, _nep = cands[0]
                    if pred["status"] != "supported":
                        new_surv[r] = 0.0
                        new_cum[r] = 0.0
                        new_ret[r] = 0.0
                        continue
                    dist = pred["distribution"]
                    s_val = sum(
                        prob * surv[dest]
                        for (kind, dest), prob in dist.items()
                        if kind == "regime" and 0 <= dest < n_reg
                    )
                    c_val = sum(
                        prob * (1.0 + gamma * cum_stages[dest])
                        for (kind, dest), prob in dist.items()
                        if kind == "regime" and 0 <= dest < n_reg
                    )
                    r_val = pred["expected_net_feedback"] + gamma * sum(
                        prob * ret[dest]
                        for (kind, dest), prob in dist.items()
                        if kind == "regime" and 0 <= dest < n_reg
                    )
                    new_surv[r] = float(s_val)
                    new_cum[r] = float(c_val)
                    new_ret[r] = float(r_val)
                elif mode == "lcb":
                    best_obj = -float("inf")
                    best_tuple = (0.0, 0.0, 0.0)
                    chosen = None
                    for prof, pred, n_ep in cands:
                        if pred["status"] != "supported":
                            continue
                        dist = pred["distribution"]
                        s_val = sum(
                            prob * surv[dest]
                            for (kind, dest), prob in dist.items()
                            if kind == "regime" and 0 <= dest < n_reg
                        )
                        c_val = sum(
                            prob * (1.0 + gamma * cum_stages[dest])
                            for (kind, dest), prob in dist.items()
                            if kind == "regime" and 0 <= dest < n_reg
                        )
                        r_val = pred["expected_net_feedback"] + gamma * sum(
                            prob * ret[dest]
                            for (kind, dest), prob in dist.items()
                            if kind == "regime" and 0 <= dest < n_reg
                        )
                        pen = (
                            0.0 if prof is None
                            else float(lcb_z) / math.sqrt(max(int(n_ep), 1))
                        )
                        obj = (c_val / max(norm_factor, 1e-9)) - pen
                        if obj > best_obj:
                            best_obj = float(obj)
                            best_tuple = (float(s_val), float(c_val), float(r_val))
                            chosen = prof
                    new_surv[r], new_cum[r], new_ret[r] = best_tuple
                    best_profile[r] = chosen
                else:
                    best_s = -1.0
                    best_c = 0.0
                    best_r = -float("inf")
                    chosen = None
                    for prof, pred, _nep in cands:
                        if pred["status"] != "supported":
                            continue
                        dist = pred["distribution"]
                        s_val = sum(
                            prob * surv[dest]
                            for (kind, dest), prob in dist.items()
                            if kind == "regime" and 0 <= dest < n_reg
                        )
                        c_val = sum(
                            prob * (1.0 + gamma * cum_stages[dest])
                            for (kind, dest), prob in dist.items()
                            if kind == "regime" and 0 <= dest < n_reg
                        )
                        r_val = pred["expected_net_feedback"] + gamma * sum(
                            prob * ret[dest]
                            for (kind, dest), prob in dist.items()
                            if kind == "regime" and 0 <= dest < n_reg
                        )
                        # Maximize survival first, break ties by expected return
                        if (s_val, r_val) > (best_s, best_r):
                            best_s = float(s_val)
                            best_c = float(c_val)
                            best_r = float(r_val)
                            chosen = prof
                    new_surv[r] = max(0.0, best_s)
                    new_cum[r] = max(0.0, best_c)
                    new_ret[r] = 0.0 if best_r == -float("inf") else best_r
                    best_profile[r] = chosen
            surv = new_surv
            cum_stages = new_cum
            ret = new_ret

        cum_norm = cum_stages / max(norm_factor, 1e-9)

        # Compute prior/baseline averages weighted by regime episode support
        weights = np.asarray(
            [max(regime_support[r], 0) for r in range(n_reg)], dtype=float,
        )
        if weights.sum() > 0:
            base_surv = float(np.dot(weights, surv) / weights.sum())
            base_cum = float(np.dot(weights, cum_norm) / weights.sum())
            base_ret = float(np.dot(weights, ret) / weights.sum())
        else:
            base_surv = float(surv.mean()) if n_reg else 0.0
            base_cum = float(cum_norm.mean()) if n_reg else 0.0
            base_ret = float(ret.mean()) if n_reg else 0.0

        return dict(
            mode=mode,
            horizon_stages=k_steps,
            discount=gamma,
            lcb_z=float(lcb_z),
            survival=tuple(float(v) for v in surv),
            cum_survival=tuple(float(v) for v in cum_norm),
            expected_return=tuple(float(v) for v in ret),
            best_profile=tuple(best_profile),
            regime_support=tuple(int(regime_support[r]) for r in range(n_reg)),
            baseline_survival=base_surv,
            baseline_cum_survival=base_cum,
            baseline_return=base_ret,
        )


class RegimeContinuationEvaluator:
    """Read-only endpoint continuation risk/value scorer with explicit OOD gate.

    Maps a batch of candidate states through a frozen ``StageRegimeSummarizer``
    and returns the multi-step continuation hazard ``1 - value(regime)`` using
    either hard nearest-regime lookup (``soft_top_k=1``) or smooth Gaussian
    kernel interpolation over the ``soft_top_k`` nearest supported regimes.
    Supports a decision-level OOD coverage gate (``min_decision_coverage``):
    when fewer than that fraction of a decision's candidate endpoints are
    supported, ``as_stage_value_callback`` abstains cleanly (returning zeros)
    so partial-OOD contrast never distorts fine-level planning.
    """

    def __init__(
        self,
        summarizer: StageRegimeSummarizer,
        values: Mapping,
        min_regime_episodes: int = 2,
        min_decision_coverage: float = 0.0,
        ood_policy: str = "baseline",
        value_key: str = "survival",
        soft_top_k: int = 1,
        kernel_tau: float = 0.7,
    ):
        if not summarizer.fitted:
            raise ValueError("summarizer must be fitted")
        if not 0.0 <= float(min_decision_coverage) <= 1.0:
            raise ValueError("min_decision_coverage must be in [0, 1]")
        if ood_policy not in ("baseline", "neutral_mean", "conservative"):
            raise ValueError(
                "ood_policy must be 'baseline', 'neutral_mean', or 'conservative'"
            )
        if value_key not in ("survival", "cum_survival"):
            raise ValueError("value_key must be 'survival' or 'cum_survival'")
        if int(soft_top_k) < 1:
            raise ValueError("soft_top_k must be positive")
        if float(kernel_tau) <= 0.0:
            raise ValueError("kernel_tau must be positive")
        self.summarizer = summarizer
        self.values = dict(values)
        self.min_regime_episodes = int(min_regime_episodes)
        self.min_decision_coverage = float(min_decision_coverage)
        self.ood_policy = str(ood_policy)
        self.value_key = str(value_key)
        self.soft_top_k = int(soft_top_k)
        self.kernel_tau = float(kernel_tau)
        key = (
            self.value_key
            if self.value_key in self.values
            else "survival"
        )
        base_key = (
            "baseline_cum_survival"
            if key == "cum_survival" and "baseline_cum_survival" in self.values
            else "baseline_survival"
        )
        self._survival = np.asarray(self.values[key], dtype=float)
        self._support = np.asarray(self.values["regime_support"], dtype=int)
        self._sup_mask = self._support >= self.min_regime_episodes
        self._baseline_survival = float(self.values[base_key])
        self._risk = 1.0 - self._survival
        self._baseline_risk = 1.0 - self._baseline_survival
        self._center_norm = np.einsum(
            "ij,ij->i", summarizer.centers_, summarizer.centers_,
        )[None, :]
        self._center_t_neg2 = np.ascontiguousarray(-2.0 * summarizer.centers_.T)
        self._ood_radius_sq = float(summarizer.ood_radius_) ** 2
        base_scale = float(getattr(summarizer, "kernel_scale_", 1.0))
        self._kernel_sigma2 = max(base_scale * (self.kernel_tau ** 2), 1e-6)

    def _squared_distances(self, obs_view: np.ndarray) -> np.ndarray:
        z = self.summarizer._standardize(obs_view)
        z_norm = np.einsum("ij,ij->i", z, z)[:, None]
        d2 = z @ self._center_t_neg2
        d2 += z_norm
        d2 += self._center_norm
        np.maximum(d2, 0.0, out=d2)
        return d2

    def continuation_risk_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(risk_array, supported_mask)`` for a batch of observations.

        Unsupported / OOD observations receive ``baseline_risk`` (neutral relative
        to the prior) and ``supported_mask=False`` so callers never treat OOD
        states as artificially safe.
        """
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        n = len(obs)
        if n == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=bool)
        obs_view = obs[:, :self.summarizer.obs_dim_]
        d2 = self._squared_distances(obs_view)
        regimes = np.argmin(d2, axis=1)
        min_d2 = d2[np.arange(n), regimes]
        finite = np.isfinite(obs_view).all(axis=1) & np.isfinite(min_d2)
        supported = (
            finite
            & (min_d2 <= self._ood_radius_sq)
            & self._sup_mask[regimes]
        )
        n_sup = int(np.sum(self._sup_mask))
        if self.soft_top_k <= 1 or n_sup <= 1:
            raw_risks = self._risk[regimes]
        else:
            k = d2.shape[1]
            m = min(self.soft_top_k, n_sup)
            d2_sup = np.where(self._sup_mask[None, :], d2, np.inf)
            if m < k:
                idx = np.argpartition(d2_sup, m - 1, axis=1)[:, :m]
                d2_sub = np.take_along_axis(d2_sup, idx, axis=1)
                r_sub = self._risk[idx]
            else:
                d2_sub = d2_sup
                r_sub = np.broadcast_to(self._risk[None, :], (n, k))
            d2_min = np.min(d2_sub, axis=1, keepdims=True)
            logits = np.where(
                np.isfinite(d2_sub),
                -0.5 * (d2_sub - d2_min) / self._kernel_sigma2,
                -np.inf,
            )
            weights = np.exp(logits)
            w_sum = weights.sum(axis=1, keepdims=True)
            weights = np.where(
                w_sum > 0.0,
                weights / np.where(w_sum > 0.0, w_sum, 1.0),
                0.0,
            )
            raw_risks = (weights * r_sub).sum(axis=1)
        risks = np.where(supported, raw_risks, self._baseline_risk)
        return risks, supported

    def decision_penalties(self, observations: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Score one decision's candidate batch with coverage-gated abstention.

        Returns ``(penalties, coverage, active)``. When ``coverage`` falls
        below ``min_decision_coverage`` (or no candidate is supported), returns
        zeros and ``active=False`` so the fine planner decides unimpeded.
        """
        risks, supported = self.continuation_risk_batch(observations)
        n = len(risks)
        if n == 0:
            return risks, 0.0, False
        coverage = float(supported.mean())
        if not np.any(supported) or coverage < self.min_decision_coverage:
            return np.zeros(n, dtype=float), coverage, False
        if self.ood_policy == "neutral_mean" and not np.all(supported):
            fill = float(risks[supported].mean())
            risks = np.where(supported, risks, fill)
        elif self.ood_policy == "conservative" and not np.all(supported):
            fill = float(risks[supported].max())
            risks = np.where(supported, risks, fill)
        return risks, coverage, True

    def as_stage_value_callback(self, stats: Optional[dict] = None):
        """Return a ``stage_value(states, sequences) -> penalties`` callback."""
        def callback(states, _sequences):
            penalties, coverage, active = self.decision_penalties(states)
            if stats is not None:
                stats["decisions"] = int(stats.get("decisions", 0)) + 1
                stats["active"] = int(stats.get("active", 0)) + int(active)
                stats["abstained"] = int(stats.get("abstained", 0)) + int(not active)
                stats["coverage_sum"] = float(stats.get("coverage_sum", 0.0)) + coverage
            return penalties
        return callback

    def snapshot(self) -> dict:
        return dict(
            summarizer=self.summarizer.snapshot(),
            mode=self.values.get("mode"),
            horizon_stages=int(self.values.get("horizon_stages", 0)),
            baseline_survival=self._baseline_survival,
            min_regime_episodes=self.min_regime_episodes,
            min_decision_coverage=self.min_decision_coverage,
            ood_policy=self.ood_policy,
            value_key=self.value_key,
            soft_top_k=self.soft_top_k,
            kernel_tau=self.kernel_tau,
            supported_regimes=int(np.sum(self._support >= self.min_regime_episodes)),
        )


class OnlineRegimeContinuationUpdater:
    """Fixed-partition online updater for regime transitions and continuation values.

    Keeps the fitted ``StageRegimeSummarizer`` partition frozen (preventing
    early partition-drift shocks when on-policy trajectories are much longer
    than bootstrap trajectories), incrementally segments each realized online
    episode once, maintains a bounded tail window of online episodes alongside
    the bootstrap episodes, and periodically refits only
    ``CalibratedRegimeTransitionModel`` and ``RegimeContinuationEvaluator``.
    Optionally attaches a ``TrunkPointFailureMemory`` for localized online
    terminal-boundary hazard calibration.
    """

    def __init__(
        self,
        summarizer: StageRegimeSummarizer,
        base_episodes: Sequence[Sequence[RegimeStage]],
        horizon_stages: int = 8,
        mode: str = "lcb",
        min_action_episodes: int = 4,
        lcb_z: float = 1.0,
        max_depth: int = 2,
        min_episodes: int = 2,
        min_regime_episodes: int = 2,
        min_decision_coverage: float = 0.80,
        ood_policy: str = "neutral_mean",
        value_key: str = "cum_survival",
        soft_top_k: int = 4,
        kernel_tau: float = 0.7,
        update_every: int = 2,
        max_online_frames: int = 12000,
        initial_evaluator: Optional[RegimeContinuationEvaluator] = None,
        failure_memory: Optional[TrunkPointFailureMemory] = None,
        failure_memory_weight: float = 0.0,
        var_sigma: float = 0.75,
        min_var_similarity: float = 0.35,
        stage_abstain_lcb: float = 0.0,
        stage_abstain_floor: float = 0.35,
    ):
        if not summarizer.fitted:
            raise ValueError("summarizer must be fitted")
        if int(update_every) < 1:
            raise ValueError("update_every must be positive")
        if int(max_online_frames) < 0:
            raise ValueError("max_online_frames must be non-negative")
        if float(failure_memory_weight) < 0.0:
            raise ValueError("failure_memory_weight must be non-negative")
        if float(var_sigma) <= 0.0:
            raise ValueError("var_sigma must be positive")
        if not 0.0 <= float(stage_abstain_lcb) <= 1.0:
            raise ValueError("stage_abstain_lcb must be in [0, 1]")
        self.summarizer = summarizer
        self._base_episodes = tuple(
            tuple(ep) for ep in base_episodes if ep
        )
        self.horizon_stages = int(horizon_stages)
        self.mode = str(mode)
        self.min_action_episodes = int(min_action_episodes)
        self.lcb_z = float(lcb_z)
        self.max_depth = int(max_depth)
        self.min_episodes = int(min_episodes)
        self.min_regime_episodes = int(min_regime_episodes)
        self.min_decision_coverage = float(min_decision_coverage)
        self.ood_policy = str(ood_policy)
        self.value_key = str(value_key)
        self.soft_top_k = int(soft_top_k)
        self.kernel_tau = float(kernel_tau)
        self.update_every = int(update_every)
        self.max_online_frames = int(max_online_frames)
        self.failure_memory = failure_memory
        self.failure_memory_weight = float(failure_memory_weight)
        self.var_sigma = float(var_sigma)
        self.min_var_similarity = float(min_var_similarity)
        self.stage_abstain_lcb = float(stage_abstain_lcb)
        self.stage_abstain_gate = None
        if self.stage_abstain_lcb > 0.0:
            from node_variables import SourceAbstainGate
            self.stage_abstain_gate = SourceAbstainGate(
                warmup=5, decay=0.90, target_lcb=self.stage_abstain_lcb,
                floor_weight=float(stage_abstain_floor), z=1.0,
            )

        self._online_buffer: List[
            Tuple[Tuple[Tuple[RegimeStage, ...], ...], int, Optional[np.ndarray]]
        ] = []
        self._online_frames = 0
        self.episodes_seen = 0
        self.updates = 0
        self.active_var_signature: Optional[np.ndarray] = None
        if initial_evaluator is not None:
            self.evaluator = initial_evaluator
        else:
            self.evaluator = self._build_evaluator(self._base_episodes)

    def _build_evaluator(
        self, stage_episodes: Sequence[Sequence[RegimeStage]],
    ) -> RegimeContinuationEvaluator:
        model = CalibratedRegimeTransitionModel(
            max_depth=self.max_depth,
            min_episodes=self.min_episodes,
        ).fit(stage_episodes, use_action=True)
        values = model.continuation_values(
            len(self.summarizer.centers_),
            horizon_stages=self.horizon_stages,
            mode=self.mode,
            min_action_episodes=self.min_action_episodes,
            lcb_z=self.lcb_z,
        )
        return RegimeContinuationEvaluator(
            self.summarizer,
            values,
            min_regime_episodes=self.min_regime_episodes,
            min_decision_coverage=self.min_decision_coverage,
            ood_policy=self.ood_policy,
            value_key=self.value_key,
            soft_top_k=self.soft_top_k,
            kernel_tau=self.kernel_tau,
        )

    def refit(
        self, var_signature: Optional[np.ndarray] = None,
    ) -> RegimeContinuationEvaluator:
        """Refit transition counts and continuation values on frozen regimes.

        When ``var_signature`` is provided, only online episodes whose inferred
        world-variable signature matches within ``min_var_similarity`` are
        combined with the shared class-node bootstrap prior.
        """
        self.active_var_signature = (
            None if var_signature is None
            else np.asarray(var_signature, dtype=float).reshape(-1).copy()
        )
        combined: List[Sequence[RegimeStage]] = list(self._base_episodes)
        if var_signature is None:
            for chunk_eps, _n_frames, _ep_sig in self._online_buffer:
                combined.extend(chunk_eps)
        else:
            from node_variables import variable_similarity
            for chunk_eps, _n_frames, ep_sig in self._online_buffer:
                if (
                    variable_similarity(
                        var_signature, ep_sig, sigma=self.var_sigma,
                    )
                    >= self.min_var_similarity
                ):
                    combined.extend(chunk_eps)
        self.evaluator = self._build_evaluator(combined)
        self.updates += 1
        return self.evaluator

    def record_episode_rows(
        self,
        rows: Sequence[Mapping],
        lag: int = 0,
        var_signature: Optional[np.ndarray] = None,
    ) -> bool:
        """Ingest one realized episode's rows and refit on cadence.

        Returns ``True`` when a transition/value refit ran on this call.
        """
        if not rows:
            return False
        if (
            self.stage_abstain_gate is not None
            and bool(rows[-1].get("done", False))
            and len(rows) >= 25
        ):
            obs_arr = np.asarray(
                [r["obs"] for r in rows], dtype=float,
            )
            sr, _sup = self.evaluator.continuation_risk_batch(obs_arr)
            pre_mean = float(np.mean(sr[-15:]))
            safe_mean = float(np.mean(sr[:-25]))
            self.stage_abstain_gate.observe(pre_mean > safe_mean)
        new_eps = self.summarizer.transform_rows(rows, lag=int(lag))
        if not new_eps:
            return False
        sig_copy = (
            None if var_signature is None
            else np.asarray(var_signature, dtype=float).reshape(-1).copy()
        )
        if self.failure_memory is not None:
            self.failure_memory.ingest_rows(
                rows, include_failures=True, var_signature=sig_copy,
            )
        n_frames = len(rows)
        self._online_buffer.append((new_eps, n_frames, sig_copy))
        self._online_frames += n_frames
        if self.max_online_frames > 0:
            while (
                self._online_frames > self.max_online_frames
                and len(self._online_buffer) > 1
            ):
                _dropped_eps, dropped_frames, _dropped_sig = (
                    self._online_buffer.pop(0)
                )
                self._online_frames -= dropped_frames
        self.episodes_seen += 1
        shifted = False
        if sig_copy is not None and self.active_var_signature is not None:
            from node_variables import variable_similarity
            shifted = (
                variable_similarity(
                    sig_copy, self.active_var_signature, sigma=self.var_sigma,
                )
                < self.min_var_similarity
            )
        if shifted or (self.episodes_seen % self.update_every == 0):
            self.refit(var_signature=sig_copy)
            return True
        return False

    def as_stage_value_callback(
        self,
        stats: Optional[dict] = None,
        var_signature_getter=None,
    ):
        """Return a live callback delegating to the latest ``self.evaluator``."""
        def callback(states, _sequences):
            penalties, coverage, active = self.evaluator.decision_penalties(states)
            if stats is not None:
                stats["decisions"] = int(stats.get("decisions", 0)) + 1
                stats["active"] = int(stats.get("active", 0)) + int(active)
                stats["abstained"] = int(stats.get("abstained", 0)) + int(not active)
                stats["coverage_sum"] = float(stats.get("coverage_sum", 0.0)) + coverage
            if (
                self.failure_memory is not None
                and self.failure_memory_weight > 0.0
            ):
                cur_sig = (
                    var_signature_getter()
                    if var_signature_getter is not None
                    else self.active_var_signature
                )
                penalties = (
                    penalties
                    + self.failure_memory_weight
                    * self.failure_memory.hazard_batch(
                        states, var_signature=cur_sig,
                    )
                )
            if self.stage_abstain_gate is not None:
                penalties = self.stage_abstain_gate.weight() * penalties
            return penalties
        return callback

    def snapshot(self) -> dict:
        return dict(
            episodes_seen=int(self.episodes_seen),
            updates=int(self.updates),
            update_every=int(self.update_every),
            max_online_frames=int(self.max_online_frames),
            online_episodes_buffered=int(
                sum(len(chunk) for chunk, _, _ in self._online_buffer)
            ),
            online_frames_buffered=int(self._online_frames),
            failure_memory_weight=float(self.failure_memory_weight),
            failure_memory=(
                self.failure_memory.snapshot()
                if self.failure_memory is not None else None
            ),
            stage_abstain_gate=(
                self.stage_abstain_gate.snapshot()
                if self.stage_abstain_gate is not None else None
            ),
            evaluator=self.evaluator.snapshot(),
        )

