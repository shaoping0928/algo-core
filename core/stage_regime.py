"""Task-agnostic stage regime summarization and calibrated multi-step transitions.

This module addresses three structural deficiencies in long-horizon stage modeling:

1. **Unbounded constant-drift dimensions**: pure deterministic counter channels
   (where ``delta == c != 0`` within numerical noise across all training steps)
   push long evaluation episodes outside Euclidean prototype spheres.
   :func:`detect_pure_drift_dims` identifies and strips them from state-regime
   similarity without hardcoding any coordinate index.
2. **Regime-based stage summarization**: instead of identifying a stage solely
   by an instantaneous boundary-trigger token, :class:`StageRegimeSummarizer`
   combines event boundaries (:class:`GenericKeySignalDetector`) with
   hysteresis-smoothed state-regime clustering over non-drift observations,
   preserving long-horizon state information while remaining causally
   resolvable online.
3. **Controllable macro-action transitions & multi-step rollout**: instead of
   fragmenting transition counts by exact bit-for-bit RLE action strings,
   :class:`CalibratedRegimeTransitionModel` conditions transitions on compact
   duty-cycle/early-impulse action profiles with nested train-only episode-OOF
   shrinkage and supports causal k-stage rollout with discrete stop-time
   convolution.

All thresholds, drift masks, centroids, OOD radii, action quantiles, and
shrinkage weights are fitted strictly on training episodes and frozen during
evaluation. No task geometry or semantics enter this module.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stage_experience import _episode_frames
from stage_outcome_memory import (
    GenericKeySignalDetector,
    KeySignal,
    stage_key_from_signals,
)


UNKNOWN = "unknown"


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_native(item) for item in value)
    return value


def detect_pure_drift_dims(
    observation_sequences: Sequence[np.ndarray],
    min_deviation: float = 1e-5,
    noise_scale: float = 1e-5,
) -> Tuple[int, ...]:
    """Identify observation channels with strictly constant non-zero per-step drift.

    A channel is flagged iff its median per-step increment ``c`` satisfies
    ``|c| > noise_floor`` and every observed transition stays within
    ``noise_floor`` of ``c`` (zero reversals, zero resets, zero jumps). Such
    channels behave as unbounded episode clocks and cause systematic OOD
    rejection whenever an evaluation episode outlasts training episodes.
    """
    deltas_by_dim = defaultdict(list)
    for seq in observation_sequences:
        obs = np.asarray(seq, dtype=float)
        if obs.ndim == 1:
            obs = obs[:, None]
        if len(obs) < 2:
            continue
        deltas = np.diff(obs, axis=0)
        for dim in range(deltas.shape[1]):
            deltas_by_dim[dim].extend(deltas[:, dim].tolist())
    drift_dims = []
    for dim, values in sorted(deltas_by_dim.items()):
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        median_step = float(np.median(arr))
        scale = max(float(np.max(np.abs(arr))), 1.0)
        noise_floor = max(float(min_deviation), float(noise_scale) * scale)
        if (abs(median_step) > noise_floor
                and float(np.max(np.abs(arr - median_step))) <= noise_floor):
            drift_dims.append(int(dim))
    return tuple(drift_dims)


def _fit_deterministic_centroids(
    matrix: np.ndarray,
    n_clusters: int,
    max_iter: int = 40,
) -> np.ndarray:
    """Deterministic furthest-point initialization + Lloyd refinement in NumPy."""
    matrix = np.asarray(matrix, dtype=float)
    n_rows, width = matrix.shape
    if n_rows == 0:
        raise ValueError("cannot fit regime centroids on empty data")
    k = max(1, min(int(n_clusters), n_rows))
    mean = matrix.mean(axis=0)
    first = int(np.argmin(np.sum((matrix - mean) ** 2, axis=1)))
    centers = [matrix[first].copy()]
    min_d2 = np.sum((matrix - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(min_d2))
        centers.append(matrix[nxt].copy())
        d2 = np.sum((matrix - centers[-1]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    centers = np.asarray(centers, dtype=float)
    for _ in range(int(max_iter)):
        d2 = np.sum((matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        updated = centers.copy()
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                updated[idx] = matrix[mask].mean(axis=0)
        if np.allclose(updated, centers, atol=1e-6):
            break
        centers = updated
    return centers


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


def _fit_survival_tree_centroids(
    standardized: np.ndarray,
    targets: np.ndarray,
    episode_ids: np.ndarray,
    n_regimes: int,
    min_leaf_episodes: int = 4,
    min_leaf_samples: int = 25,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Deterministic greedy binary partition minimizing remaining-life SSE.

    Purely self-supervised and task-agnostic: splits standardized observations
    only along axes that reduce within-leaf variance of log remaining steps to
    termination, subject to episode-diversity and sample-count guards. Because
    each split picks one axis at a time, duplicated or collinear channels never
    receive double weight and low-signal channels are naturally ignored.
    """
    n, d = standardized.shape
    k = max(1, min(int(n_regimes), n))
    if k <= 1 or d == 0:
        return standardized.mean(axis=0, keepdims=True), ()

    min_ep = max(1, int(min_leaf_episodes))
    min_n = max(1, int(min_leaf_samples))
    qs = np.linspace(0.1, 0.9, 13)

    def _best_split(idx: np.ndarray):
        if len(idx) < 2 * min_n:
            return None
        y_sub = targets[idx]
        parent_sse = float(np.sum((y_sub - y_sub.mean()) ** 2))
        if parent_sse <= 1e-12:
            return None
        ep_sub = episode_ids[idx]
        best_gain = 1e-9
        best_spec = None
        for dim in range(d):
            col = standardized[idx, dim]
            cands = np.unique(np.quantile(col, qs))
            for thr in cands:
                l_mask = col <= thr
                n_l = int(l_mask.sum())
                n_r = len(idx) - n_l
                if n_l < min_n or n_r < min_n:
                    continue
                if len(np.unique(ep_sub[l_mask])) < min_ep:
                    continue
                if len(np.unique(ep_sub[~l_mask])) < min_ep:
                    continue
                yl = y_sub[l_mask]
                yr = y_sub[~l_mask]
                sse = float(
                    np.sum((yl - yl.mean()) ** 2)
                    + np.sum((yr - yr.mean()) ** 2)
                )
                gain = parent_sse - sse
                if gain > best_gain:
                    best_gain = gain
                    best_spec = (dim, idx[l_mask], idx[~l_mask])
        return None if best_spec is None else (best_gain, best_spec)

    nodes = [np.arange(n, dtype=int)]
    leaves = [0]
    split_dims: List[int] = []
    cache = {0: _best_split(nodes[0])}

    while len(leaves) < k:
        best_leaf = None
        best_gain = 1e-9
        for lid in leaves:
            res = cache.get(lid)
            if res is not None and res[0] > best_gain:
                best_gain = res[0]
                best_leaf = lid
        if best_leaf is None:
            break
        _, (dim, l_idx, r_idx) = cache[best_leaf]
        split_dims.append(int(dim))
        l_id = len(nodes)
        r_id = len(nodes) + 1
        nodes.append(l_idx)
        nodes.append(r_idx)
        leaves.remove(best_leaf)
        leaves.extend([l_id, r_id])
        cache[l_id] = _best_split(l_idx)
        cache[r_id] = _best_split(r_idx)

    centers = np.asarray(
        [standardized[nodes[lid]].mean(axis=0) for lid in leaves],
        dtype=float,
    )
    return centers, tuple(split_dims)


@dataclass(frozen=True)
class RegimeStage:
    """One variable-duration macro stage combining regime state and events."""

    episode: Hashable
    regime: int
    event_entered: bool
    event_key: Tuple[Hashable, ...]
    start: int
    end: int
    duration: int
    entry_state: np.ndarray
    mean_state: np.ndarray
    delta_state: np.ndarray
    actions: Tuple[int, ...]
    action_profile: Tuple[Hashable, ...]
    feedback: np.ndarray
    terminal: bool
    censored: bool


class StageRegimeSummarizer:
    """Self-supervised regime state summarizer and event-aware segmenter."""

    def __init__(
        self,
        n_regimes: int = 16,
        min_dwell: int = 6,
        hysteresis: int = 2,
        event_quantile: float = 0.98,
        ood_quantile: float = 0.995,
        strip_pure_drift: bool = True,
        partition: str = "kmeans",
        min_leaf_episodes: int = 4,
        min_leaf_samples: int = 25,
    ):
        if int(n_regimes) < 1:
            raise ValueError("n_regimes must be positive")
        if int(min_dwell) < 1:
            raise ValueError("min_dwell must be positive")
        if int(hysteresis) < 1:
            raise ValueError("hysteresis must be positive")
        if not 0.5 < float(ood_quantile) <= 1.0:
            raise ValueError("ood_quantile must be in (0.5, 1.0]")
        if partition not in ("kmeans", "survival_tree"):
            raise ValueError("partition must be 'kmeans' or 'survival_tree'")
        self.n_regimes = int(n_regimes)
        self.min_dwell = int(min_dwell)
        self.hysteresis = int(hysteresis)
        self.event_quantile = float(event_quantile)
        self.ood_quantile = float(ood_quantile)
        self.strip_pure_drift = bool(strip_pure_drift)
        self.partition = str(partition)
        self.min_leaf_episodes = int(min_leaf_episodes)
        self.min_leaf_samples = int(min_leaf_samples)
        self.fitted = False

    def _active_matrix(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            return obs[self.keep_dims_]
        return obs[:, self.keep_dims_]

    def _standardize(self, observations: np.ndarray) -> np.ndarray:
        return (self._active_matrix(observations) - self.mean_) / self.scale_

    def fit_episodes(self, episodes: Sequence[Mapping]) -> "StageRegimeSummarizer":
        """Fit drift mask, event detector, centroids, OOD radius, and action bins."""
        if not episodes:
            raise ValueError("at least one training episode is required")
        obs_seqs = [
            np.asarray(ep["observations"], dtype=float) for ep in episodes
        ]
        obs_dim = obs_seqs[0].shape[1]
        if any(seq.ndim != 2 or seq.shape[1] != obs_dim for seq in obs_seqs):
            raise ValueError("all episodes must share observation width")

        self.obs_dim_ = int(obs_dim)
        if self.strip_pure_drift:
            self.drift_dims_ = detect_pure_drift_dims(obs_seqs)
        else:
            self.drift_dims_ = ()
        keep = [d for d in range(obs_dim) if d not in self.drift_dims_]
        if not keep:
            # Degenerate all-counter input: keep all dims rather than crashing
            keep = list(range(obs_dim))
            self.drift_dims_ = ()
        self.keep_dims_ = np.asarray(keep, dtype=int)

        self.detector_ = GenericKeySignalDetector(
            quantile=self.event_quantile,
        ).fit([
            dict(observations=ep["observations"], feedback=ep.get("feedback"))
            for ep in episodes
        ])

        stacked = np.vstack(obs_seqs)
        active = self._active_matrix(stacked)
        self.mean_ = active.mean(axis=0)
        sd = active.std(axis=0)
        self.scale_ = np.where(sd > 1e-8, sd, 1.0)
        standardized = (active - self.mean_) / self.scale_

        self.split_dims_ = ()
        if self.partition == "survival_tree":
            z_list, y_list, ep_list = [], [], []
            for e_idx, ep in enumerate(episodes):
                if not bool(ep.get("terminal", False)):
                    continue
                seq = np.asarray(ep["observations"], dtype=float)
                n_t = max(len(seq) - 1, 1)
                z_ep = (self._active_matrix(seq[:n_t]) - self.mean_) / self.scale_
                rem = np.arange(n_t, 0, -1, dtype=float)
                z_list.append(z_ep)
                y_list.append(np.log1p(rem))
                ep_list.append(np.full(n_t, e_idx, dtype=int))
            if z_list:
                centers, split_local = _fit_survival_tree_centroids(
                    np.vstack(z_list),
                    np.concatenate(y_list),
                    np.concatenate(ep_list),
                    self.n_regimes,
                    min_leaf_episodes=self.min_leaf_episodes,
                    min_leaf_samples=self.min_leaf_samples,
                )
                if len(centers) >= 2:
                    self.centers_ = centers
                    self.split_dims_ = tuple(
                        int(self.keep_dims_[d]) for d in split_local
                    )
                else:
                    self.centers_ = _fit_deterministic_centroids(
                        standardized, self.n_regimes,
                    )
            else:
                self.centers_ = _fit_deterministic_centroids(
                    standardized, self.n_regimes,
                )
        else:
            self.centers_ = _fit_deterministic_centroids(
                standardized, self.n_regimes,
            )
        c_norm = np.einsum("ij,ij->i", self.centers_, self.centers_)[None, :]
        z_norm = np.einsum("ij,ij->i", standardized, standardized)[:, None]
        d2_all = standardized @ (-2.0 * self.centers_.T)
        d2_all += z_norm
        d2_all += c_norm
        np.maximum(d2_all, 0.0, out=d2_all)
        d2 = np.min(d2_all, axis=1)
        self.ood_radius_ = float(np.quantile(np.sqrt(d2), self.ood_quantile))
        if d2_all.shape[1] >= 2:
            second_d2 = np.partition(d2_all, 1, axis=1)[:, 1]
            self.kernel_scale_ = max(float(np.median(second_d2)), 1e-6)
        else:
            self.kernel_scale_ = max(float(np.median(d2)), 1e-6)
        # Mark fitted before internal first-pass segmentation for action quantiles
        self.rate_q50_ = 0.25
        self.fitted = True

        # Fit duty-cycle split point strictly on training segmented stages
        nonzero_rates = []
        for ep in episodes:
            stages = self.segment_episode(
                episode=ep.get("episode", 0),
                observations=ep["observations"],
                actions=ep["actions"],
                feedback=ep["feedback"],
                terminal=bool(ep.get("terminal", False)),
                censored=bool(ep.get("censored", False)),
            )
            for stage in stages:
                if stage.actions:
                    rate = float(np.mean(np.asarray(stage.actions) != 0))
                    if rate > 0.0:
                        nonzero_rates.append(rate)
        if nonzero_rates:
            self.rate_q50_ = float(np.quantile(nonzero_rates, 0.50))
        return self

    def fit_rows(self, rows: Sequence[Mapping], lag: int) -> "StageRegimeSummarizer":
        """Fit directly from realized transition dicts."""
        frames, _, _ = _episode_frames(
            rows, SimpleNamespace(lag=int(lag)), lag=int(lag),
            with_features=False,
        )
        if not frames:
            raise ValueError("no usable episodes in rows")
        return self.fit_episodes([
            dict(
                episode=fr.episode,
                observations=fr.observations,
                actions=fr.actions,
                feedback=fr.feedback,
                terminal=fr.terminal,
                censored=fr.censored,
            )
            for fr in frames
        ])

    def resolve(self, observation: np.ndarray) -> dict:
        """Map one continuous observation to a regime id with explicit OOD gate."""
        if not self.fitted:
            raise RuntimeError("fit StageRegimeSummarizer before resolve")
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.size != self.obs_dim_ or not np.isfinite(obs).all():
            return dict(
                status=UNKNOWN, regime=None, distance=float("inf"),
                in_radius=False, reason="invalid_observation",
            )
        z = self._standardize(obs)
        dists = np.sqrt(np.sum((self.centers_ - z[None, :]) ** 2, axis=1))
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        in_radius = dist <= self.ood_radius_
        if not in_radius:
            return dict(
                status=UNKNOWN, regime=idx, distance=dist,
                in_radius=False, reason="outside_radius",
            )
        return dict(
            status="supported", regime=idx, distance=dist,
            in_radius=True, reason=None,
        )

    def summarize_action_profile(
        self, actions: Sequence[int],
    ) -> Tuple[Hashable, ...]:
        """Compact controllable stage behavior profile (<=5 tokens).

        Encodes non-zero action duty-cycle tier (idle / low / high) and
        whether any non-zero action occurred in the first half of the stage.
        Works for binary or discrete action spaces with action 0 as default.
        """
        if not actions:
            return ("idle", 0)
        arr = np.asarray(actions, dtype=float).reshape(-1)
        active = arr != 0.0
        rate = float(active.mean())
        if rate <= 0.0:
            return ("idle", 0)
        tier = "low" if rate <= self.rate_q50_ else "high"
        half = max(1, len(active) // 2)
        early = int(bool(np.any(active[:half])))
        return (tier, early)

    def segment_episode(
        self,
        episode: Hashable,
        observations: np.ndarray,
        actions: Sequence[int],
        feedback: np.ndarray,
        terminal: bool = False,
        censored: bool = False,
    ) -> Tuple[RegimeStage, ...]:
        """Causally segment one episode into event-aware macro regime stages."""
        if not self.fitted:
            raise RuntimeError("fit StageRegimeSummarizer before segmenting")
        obs = np.asarray(observations, dtype=float)
        fb = np.asarray(feedback, dtype=float)
        if fb.ndim == 1:
            fb = fb[:, None]
        n_trans = len(actions)
        if len(obs) != n_trans + 1 or len(fb) != n_trans:
            raise ValueError("observations/actions/feedback lengths mismatch")
        if n_trans == 0:
            return ()

        Z = self._standardize(obs)
        d2 = np.sum((Z[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2)
        regimes = np.argmin(d2, axis=1)

        signals = self.detector_.transform_episode(obs, feedback=fb)
        hard_keys = [()] * len(obs)
        prev_tokens = set()
        for t in range(len(obs)):
            tokens = {sig.token for sig in signals[t]}
            new_tokens = tokens - prev_tokens
            if t > 0 and new_tokens:
                hard_keys[t] = stage_key_from_signals(
                    [sig for sig in signals[t] if sig.token in new_tokens]
                )
            prev_tokens = tokens

        boundaries = [0]
        boundary_events = [()]
        cur_regime = int(regimes[0])
        cur_start = 0
        pending_regime = None
        pending_count = 0

        for t in range(1, n_trans):
            r = int(regimes[t])
            if hard_keys[t]:
                boundaries.append(t)
                boundary_events.append(hard_keys[t])
                cur_regime = r
                cur_start = t
                pending_regime = None
                pending_count = 0
                continue
            if r == cur_regime:
                pending_regime = None
                pending_count = 0
                continue
            if r == pending_regime:
                pending_count += 1
            else:
                pending_regime = r
                pending_count = 1
            if ((t - cur_start) >= self.min_dwell
                    and pending_count >= self.hysteresis):
                boundaries.append(t)
                boundary_events.append(())
                cur_regime = r
                cur_start = t
                pending_regime = None
                pending_count = 0

        stages = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else n_trans
            if end <= start:
                continue
            is_last = idx == len(boundaries) - 1
            acts = tuple(int(a) for a in actions[start:end])
            seg_obs = obs[start:end + 1]
            stages.append(RegimeStage(
                episode=_native(episode),
                regime=int(regimes[start]),
                event_entered=bool(boundary_events[idx]),
                event_key=boundary_events[idx],
                start=int(start),
                end=int(end),
                duration=int(end - start),
                entry_state=seg_obs[0].copy(),
                mean_state=seg_obs.mean(axis=0),
                delta_state=seg_obs[-1] - seg_obs[0],
                actions=acts,
                action_profile=self.summarize_action_profile(acts),
                feedback=fb[start:end].sum(axis=0),
                terminal=bool(terminal and is_last),
                censored=bool(censored and is_last),
            ))
        return tuple(stages)

    def transform_rows(
        self, rows: Sequence[Mapping], lag: int,
    ) -> Tuple[Tuple[RegimeStage, ...], ...]:
        """Segment realized transition rows using frozen parameters."""
        if not self.fitted:
            raise RuntimeError("fit StageRegimeSummarizer before transform")
        frames, _, _ = _episode_frames(
            rows, SimpleNamespace(lag=int(lag)), lag=int(lag),
            with_features=False,
        )
        return tuple(
            self.segment_episode(
                episode=fr.episode,
                observations=fr.observations,
                actions=fr.actions,
                feedback=fr.feedback,
                terminal=fr.terminal,
                censored=fr.censored,
            )
            for fr in frames
        )

    def snapshot(self) -> dict:
        if not self.fitted:
            return dict(fitted=False)
        return dict(
            fitted=True,
            partition=self.partition,
            n_regimes=int(len(self.centers_)),
            min_dwell=self.min_dwell,
            hysteresis=self.hysteresis,
            obs_dim=self.obs_dim_,
            drift_dims=self.drift_dims_,
            keep_dims=tuple(int(d) for d in self.keep_dims_),
            split_dims=tuple(getattr(self, "split_dims_", ())),
            ood_radius=float(self.ood_radius_),
            kernel_scale=float(getattr(self, "kernel_scale_", 1.0)),
            rate_q50=float(self.rate_q50_),
            event_channels=len(self.detector_.thresholds),
        )


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


class TrunkPointFailureMemory:
    """Trunk-indexed smooth local terminal-failure memory (strictly task-agnostic).

    Stores ONLY true world-feedback terminal states (``done=True`` ``next_obs``)
    alongside a bounded background reservoir of surviving states inside each
    frozen coarse trunk leaf of ``StageRegimeSummarizer``.  Quiet trunk leaves
    with zero observed failures stay compressed at zero query cost.  When a
    candidate state falls inside a trunk leaf that has recorded failures (and
    within the summarizer's OOD radius), returns the smooth local Gaussian
    kernel hazard ratio:

        ``k_fail(z) / (k_fail(z) + k_safe(z) + prior)``

    This avoids both precursor-label corridor smear (never marks normal
    pre-terminal flight as hazardous) and hard axis-aligned split cliffs, while
    allowing repeated online blind spots to accumulate calibrated hazard.
    """

    def __init__(
        self,
        summarizer: StageRegimeSummarizer,
        sigma_mult: float = 0.25,
        prior: float = 4.0,
        safe_stride: int = 2,
        max_fail_per_leaf: int = 120,
        max_safe_per_leaf: int = 400,
        var_sigma: float = 0.75,
    ):
        if not summarizer.fitted:
            raise ValueError("summarizer must be fitted")
        if float(sigma_mult) <= 0.0:
            raise ValueError("sigma_mult must be positive")
        if float(prior) < 0.0:
            raise ValueError("prior must be non-negative")
        if int(safe_stride) < 1:
            raise ValueError("safe_stride must be positive")
        if int(max_fail_per_leaf) < 1 or int(max_safe_per_leaf) < 1:
            raise ValueError("per-leaf reservoir limits must be positive")
        if float(var_sigma) <= 0.0:
            raise ValueError("var_sigma must be positive")
        self.summarizer = summarizer
        self.sigma_mult = float(sigma_mult)
        self.prior = float(prior)
        self.safe_stride = int(safe_stride)
        self.max_fail_per_leaf = int(max_fail_per_leaf)
        self.max_safe_per_leaf = int(max_safe_per_leaf)
        self.var_sigma = float(var_sigma)

        self.n_trunk = len(summarizer.centers_)
        self._center_norm = np.einsum(
            "ij,ij->i", summarizer.centers_, summarizer.centers_,
        )[None, :]
        self._center_t_neg2 = np.ascontiguousarray(-2.0 * summarizer.centers_.T)
        self._ood_radius_sq = float(summarizer.ood_radius_) ** 2
        base_scale = float(getattr(summarizer, "kernel_scale_", 1.0))
        self._sigma2 = max(base_scale * (self.sigma_mult ** 2), 1e-4)

        self._fail_lists: List[List[np.ndarray]] = [[] for _ in range(self.n_trunk)]
        self._fail_sigs: List[List[Optional[np.ndarray]]] = [
            [] for _ in range(self.n_trunk)
        ]
        self._safe_lists: List[List[np.ndarray]] = [[] for _ in range(self.n_trunk)]
        self._fail_arrs: List[Optional[np.ndarray]] = [None] * self.n_trunk
        self._safe_arrs: List[Optional[np.ndarray]] = [None] * self.n_trunk
        self.total_failures = 0

    def _route_trunk_batch(
        self, obs_view: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.summarizer._standardize(obs_view)
        z_norm = np.einsum("ij,ij->i", z, z)[:, None]
        d2 = z @ self._center_t_neg2
        d2 += z_norm
        d2 += self._center_norm
        np.maximum(d2, 0.0, out=d2)
        trunk = np.argmin(d2, axis=1)
        min_d2 = d2[np.arange(len(obs_view)), trunk]
        in_rad = np.isfinite(z).all(axis=1) & (min_d2 <= self._ood_radius_sq)
        return z, trunk, in_rad

    def _rebuild_leaf(self, tid: int) -> None:
        fl = self._fail_lists[tid]
        sl = self._safe_lists[tid]
        self._fail_arrs[tid] = np.asarray(fl, dtype=float) if fl else None
        self._safe_arrs[tid] = np.asarray(sl, dtype=float) if sl else None

    def ingest_rows(
        self,
        rows: Sequence[Mapping],
        include_failures: bool = True,
        var_signature: Optional[np.ndarray] = None,
    ) -> int:
        """Ingest realized transition rows into trunk-indexed fail/safe reservoirs.

        Returns the number of terminal failures newly recorded on this call.
        """
        if not rows:
            return 0
        by_ep: Dict[Hashable, List[Mapping]] = {}
        for row in rows:
            by_ep.setdefault(row.get("episode", 0), []).append(row)

        touched = set()
        added_failures = 0
        obs_dim = self.summarizer.obs_dim_
        sig_copy = (
            None if var_signature is None
            else np.asarray(var_signature, dtype=float).reshape(-1).copy()
        )
        for seq in by_ep.values():
            if not seq:
                continue
            length = len(seq)
            terminal = bool(seq[-1].get("done", False))
            obs_list = [
                np.asarray(item["obs"], dtype=float)[:obs_dim] for item in seq
            ]
            if terminal:
                obs_list.append(
                    np.asarray(seq[-1]["next_obs"], dtype=float)[:obs_dim]
                )
            obs_arr = np.asarray(obs_list, dtype=float)
            z_arr, trunk_arr, in_rad = self._route_trunk_batch(obs_arr)

            if terminal and include_failures:
                idx_term = len(obs_list) - 1
                if in_rad[idx_term]:
                    tid = int(trunk_arr[idx_term])
                    self._fail_lists[tid].append(z_arr[idx_term].copy())
                    self._fail_sigs[tid].append(sig_copy)
                    if len(self._fail_lists[tid]) > self.max_fail_per_leaf:
                        self._fail_lists[tid].pop(0)
                        self._fail_sigs[tid].pop(0)
                    touched.add(tid)
                    self.total_failures += 1
                    added_failures += 1

            safe_end = max(0, length - 2) if terminal else length
            for idx in range(0, safe_end, self.safe_stride):
                if not in_rad[idx]:
                    continue
                tid = int(trunk_arr[idx])
                self._safe_lists[tid].append(z_arr[idx].copy())
                if len(self._safe_lists[tid]) > self.max_safe_per_leaf:
                    self._safe_lists[tid].pop(0)
                touched.add(tid)

        for tid in touched:
            self._rebuild_leaf(tid)
        return added_failures

    def hazard_batch(
        self,
        states: np.ndarray,
        var_signature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return localized smooth terminal-hazard ratio in ``[0, 1)``."""
        arr = np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = len(arr)
        if n == 0 or self.total_failures == 0:
            return np.zeros(n, dtype=float)
        z_arr, trunk_arr, in_rad = self._route_trunk_batch(
            arr[:, :self.summarizer.obs_dim_],
        )
        out = np.zeros(n, dtype=float)
        inv_2s2 = -0.5 / self._sigma2
        sim_fn = None
        if var_signature is not None:
            from node_variables import variable_similarity
            sim_fn = variable_similarity
        for tid in np.unique(trunk_arr):
            f_arr = self._fail_arrs[int(tid)]
            if f_arr is None:
                continue
            mask = (trunk_arr == tid) & in_rad
            if not np.any(mask):
                continue
            zq = z_arr[mask]
            df2 = np.sum((zq[:, None, :] - f_arr[None, :, :]) ** 2, axis=2)
            if sim_fn is not None:
                vw = np.asarray(
                    [
                        sim_fn(var_signature, fs, sigma=self.var_sigma)
                        for fs in self._fail_sigs[int(tid)]
                    ],
                    dtype=float,
                )
                kf = (np.exp(inv_2s2 * df2) * vw[None, :]).sum(axis=1)
            else:
                kf = np.exp(inv_2s2 * df2).sum(axis=1)
            s_arr = self._safe_arrs[int(tid)]
            if s_arr is not None:
                ds2 = np.sum((zq[:, None, :] - s_arr[None, :, :]) ** 2, axis=2)
                ks = np.exp(inv_2s2 * ds2).sum(axis=1)
            else:
                ks = np.zeros(len(zq), dtype=float)
            out[mask] = kf / (kf + ks + self.prior)
        return out

    def snapshot(self) -> dict:
        active_leaves = sum(1 for arr in self._fail_arrs if arr is not None)
        return dict(
            sigma_mult=self.sigma_mult,
            prior=self.prior,
            total_failures=int(self.total_failures),
            buffered_failures=int(sum(len(x) for x in self._fail_lists)),
            buffered_safe=int(sum(len(x) for x in self._safe_lists)),
            active_failure_leaves=int(active_leaves),
            compressed_leaves=int(self.n_trunk - active_leaves),
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


class ControllableFeatureDemand:
    """Strictly task-agnostic discovery of controllable, value-aligned feature demands.

    Intended exclusively as a heuristic pruning/ordering signal (separate from
    the calibrated final objective):
      1. Strips pure monotonic drift channels via ``detect_pure_drift_dims``.
      2. Measures multi-step action sensitivity through the learned world model
         and gates by both relative effect and signal-to-noise ratio (vs 1-step
         natural transition delta std) so uncontrollable phase/clock channels
         receive zero weight.
      3. Estimates an occupancy-invariant safe-envelope midpoint
         ``0.5 * (q_lo + q_hi)`` on top-quartile surviving training states so
         behavior-policy occupancy skew inside the viable band does not bias
         the target center.
      4. Weights controllable channels by survival-alignment correlation and
         imminent-failure contrast.
    """

    def __init__(
        self,
        min_rel_effect: float = 0.02,
        min_snr: float = 0.20,
        envelope_low_q: float = 0.05,
        envelope_high_q: float = 0.95,
        survival_top_q: float = 0.75,
        imminent_steps: float = 5.0,
        sample_size: int = 128,
        seed: int = 0,
    ):
        self.min_rel_effect = float(min_rel_effect)
        self.min_snr = float(min_snr)
        self.envelope_low_q = float(envelope_low_q)
        self.envelope_high_q = float(envelope_high_q)
        self.survival_top_q = float(survival_top_q)
        self.imminent_steps = float(imminent_steps)
        self.sample_size = int(sample_size)
        self.seed = int(seed)
        self.fitted = False

    def fit_rows(
        self,
        world,
        agent,
        rows: Sequence[Mapping],
        lag: int = 0,
    ) -> "ControllableFeatureDemand":
        """Fit controllable feature weights & envelope centers from training rows."""
        if not rows:
            raise ValueError("rows must not be empty")
        from substrate_generic import history_features

        obs_all = np.asarray([row["obs"] for row in rows], dtype=float)
        nxt_all = np.asarray([row["next_obs"] for row in rows], dtype=float)
        n, d = obs_all.shape
        obs_std = np.std(obs_all, axis=0)
        obs_std = np.where(obs_std > 1e-8, obs_std, 1.0)
        delta_std = np.std(nxt_all - obs_all, axis=0)
        delta_std = np.where(delta_std > 1e-8, delta_std, 1.0)

        by_ep: Dict[Hashable, List[Mapping]] = {}
        for row in rows:
            by_ep.setdefault(row.get("episode", 0), []).append(row)
        ep_seqs = [
            np.asarray([item["obs"] for item in seq], dtype=float)
            for seq in by_ep.values()
            if len(seq) >= 3
        ]
        drift_dims = set(detect_pure_drift_dims(ep_seqs)) if ep_seqs else set()

        # Multi-step action controllability via the learned world model
        rng = np.random.RandomState(self.seed)
        sample_n = min(self.sample_size, n)
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        s0 = obs_all[sample_idx]
        action_space = tuple(getattr(agent, "action_space", (0, 1)))
        roll_h = max(1, int(lag) + 2)
        action_dim = int(lag) + 1
        prefix = np.zeros(int(lag), dtype=float)

        endpoints = []
        for a_val in action_space:
            st = s0.copy()
            seq = np.full((sample_n, roll_h), float(a_val), dtype=float)
            for step in range(roll_h):
                hist = history_features(
                    seq, step + 1, prefix, action_dim=action_dim,
                )
                st = np.asarray(world.predict(st, hist), dtype=float)
            endpoints.append(st[:, :d])
        ep_stack = np.stack(endpoints, axis=0)
        act_std = np.mean(np.std(ep_stack, axis=0), axis=0)
        rel_effect = act_std / obs_std
        snr = act_std / delta_std
        controllable = np.where(
            (rel_effect >= self.min_rel_effect) & (snr >= self.min_snr),
            rel_effect,
            0.0,
        )
        for dd in drift_dims:
            if 0 <= dd < d:
                controllable[dd] = 0.0

        # Survival-envelope target & alignment gain strictly on terminal episodes
        s_term_list = []
        rem_list = []
        for seq in by_ep.values():
            if not seq or not bool(seq[-1].get("done", False)):
                continue
            arr = np.asarray([item["obs"] for item in seq], dtype=float)
            rem = np.arange(len(arr), 0, -1, dtype=float)
            s_term_list.append(arr)
            rem_list.append(rem)

        if s_term_list:
            s_term = np.vstack(s_term_list)
            rem_all = np.concatenate(rem_list)
            y_val = np.log1p(rem_all)
            q_cut = float(np.quantile(y_val, self.survival_top_q))
            safe_states = s_term[y_val >= q_cut]
            if len(safe_states) < 2:
                safe_states = s_term
            q_lo = np.quantile(safe_states, self.envelope_low_q, axis=0)
            q_hi = np.quantile(safe_states, self.envelope_high_q, axis=0)
            env_center = 0.5 * (q_lo + q_hi)
            env_halfwidth = np.maximum(0.5 * (q_hi - q_lo), 1e-4)

            dead_states = s_term[rem_all <= self.imminent_steps]
            align_gain = np.zeros(d, dtype=float)
            fail_contrast = np.zeros(d, dtype=float)
            for j in range(d):
                if controllable[j] <= 0.0:
                    continue
                closeness = -np.abs(s_term[:, j] - env_center[j])
                if np.std(closeness) > 1e-9 and np.std(y_val) > 1e-9:
                    r = float(np.corrcoef(closeness, y_val)[0, 1])
                    if math.isfinite(r):
                        align_gain[j] = max(0.0, r)
                if len(dead_states) > 0:
                    d_dead = float(np.mean(np.abs(dead_states[:, j] - env_center[j])))
                    d_safe = float(np.mean(np.abs(safe_states[:, j] - env_center[j])))
                    fail_contrast[j] = max(
                        0.0, (d_dead - d_safe) / float(env_halfwidth[j])
                    )
            raw_w = np.where(
                controllable > 0.0,
                align_gain * (align_gain + fail_contrast),
                0.0,
            )
        else:
            env_center = np.median(obs_all, axis=0)
            env_halfwidth = np.maximum(obs_std, 1e-4)
            align_gain = np.zeros(d, dtype=float)
            fail_contrast = np.zeros(d, dtype=float)
            raw_w = np.zeros(d, dtype=float)

        total_w = float(raw_w.sum())
        demand_weights = (
            raw_w / total_w if total_w > 1e-9 else np.zeros(d, dtype=float)
        )

        self.obs_dim_ = int(d)
        self.drift_dims_ = tuple(sorted(int(x) for x in drift_dims))
        self.controllable_ = controllable
        self.env_center_ = np.asarray(env_center, dtype=float)
        self.env_halfwidth_ = np.asarray(env_halfwidth, dtype=float)
        self.align_gain_ = align_gain
        self.fail_contrast_ = fail_contrast
        self.demand_weights_ = np.asarray(demand_weights, dtype=float)
        self._inv_scaled_w = self.demand_weights_ / self.env_halfwidth_
        self.fitted = True
        return self

    def distance_batch(self, states: np.ndarray) -> np.ndarray:
        """Weighted normalized L1 distance to learned controllable envelope center."""
        if not self.fitted:
            raise RuntimeError("fit ControllableFeatureDemand first")
        arr = np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if len(arr) == 0:
            return np.zeros(0, dtype=float)
        view = arr[:, :self.obs_dim_]
        return np.abs(view - self.env_center_[None, :]) @ self._inv_scaled_w

    def snapshot(self) -> dict:
        if not self.fitted:
            return dict(fitted=False)
        return dict(
            fitted=True,
            obs_dim=self.obs_dim_,
            drift_dims=self.drift_dims_,
            controllable_dims=tuple(
                int(j) for j in range(self.obs_dim_) if self.controllable_[j] > 0.0
            ),
            demand_weights=tuple(round(float(w), 4) for w in self.demand_weights_),
            env_center=tuple(round(float(c), 4) for c in self.env_center_),
            align_gain=tuple(round(float(r), 4) for r in self.align_gain_),
        )


class OnlineFeatureDemandUpdater:
    """Strictly task-agnostic online EMA updater for :class:`ControllableFeatureDemand`.

    Keeps the learned structural controllability mask fixed while smoothly
    updating the safe-envelope midpoint, halfwidth, and alignment weights from
    a bounded sliding window of realized online episodes.  Non-terminal capped
    episodes contribute to safe survival states without polluting terminal
    failure states.
    """

    def __init__(
        self,
        base_demand: ControllableFeatureDemand,
        update_every: int = 4,
        ema_alpha: float = 0.30,
        max_online_frames: int = 12000,
        min_online_frames: int = 600,
    ):
        if not base_demand.fitted:
            raise ValueError("base_demand must be fitted")
        if int(update_every) < 1:
            raise ValueError("update_every must be positive")
        if not 0.0 < float(ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if int(max_online_frames) < 1:
            raise ValueError("max_online_frames must be positive")
        if int(min_online_frames) < 1:
            raise ValueError("min_online_frames must be positive")

        self.update_every = int(update_every)
        self.ema_alpha = float(ema_alpha)
        self.max_online_frames = int(max_online_frames)
        self.min_online_frames = int(min_online_frames)

        self.obs_dim_ = int(base_demand.obs_dim_)
        self.controllable_ = np.asarray(base_demand.controllable_, dtype=float).copy()
        self.env_center_ = np.asarray(base_demand.env_center_, dtype=float).copy()
        self.env_halfwidth_ = np.asarray(base_demand.env_halfwidth_, dtype=float).copy()
        self.demand_weights_ = np.asarray(base_demand.demand_weights_, dtype=float).copy()
        self._inv_scaled_w = self.demand_weights_ / self.env_halfwidth_
        self.envelope_low_q = float(base_demand.envelope_low_q)
        self.envelope_high_q = float(base_demand.envelope_high_q)
        self.survival_top_q = float(base_demand.survival_top_q)
        self.imminent_steps = float(base_demand.imminent_steps)

        self._online_chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._online_frames: int = 0
        self.episodes_seen: int = 0
        self.updates: int = 0

    def distance_batch(self, states: np.ndarray) -> np.ndarray:
        """Weighted normalized L1 distance to the latest online safe-envelope center."""
        arr = np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if len(arr) == 0:
            return np.zeros(0, dtype=float)
        view = arr[:, :self.obs_dim_]
        return np.abs(view - self.env_center_[None, :]) @ self._inv_scaled_w

    def record_episode_rows(self, ep_rows: Sequence[Mapping]) -> bool:
        """Ingest one realized episode and EMA-update the safe envelope on cadence."""
        if not ep_rows:
            return False
        arr = np.asarray(
            [row["obs"] for row in ep_rows], dtype=float
        )[:, :self.obs_dim_]
        n = len(arr)
        is_term = bool(ep_rows[-1].get("done", False))
        rem = np.arange(n, 0, -1, dtype=float) + (0.0 if is_term else float(n))
        dead = (rem <= self.imminent_steps) if is_term else np.zeros(n, dtype=bool)

        self._online_chunks.append((arr, rem, dead))
        self._online_frames += n
        while self._online_frames > self.max_online_frames and len(self._online_chunks) > 1:
            dropped = self._online_chunks.pop(0)
            self._online_frames -= len(dropped[0])
        self.episodes_seen += 1

        if (
            self.episodes_seen % self.update_every != 0
            or self._online_frames < self.min_online_frames
        ):
            return False

        s_all = np.vstack([chunk[0] for chunk in self._online_chunks])
        rem_all = np.concatenate([chunk[1] for chunk in self._online_chunks])
        dead_mask = np.concatenate([chunk[2] for chunk in self._online_chunks])

        y_val = np.log1p(rem_all)
        q_cut = float(np.quantile(y_val, self.survival_top_q))
        safe_states = s_all[y_val >= q_cut]
        if len(safe_states) < 16:
            return False
        q_lo = np.quantile(safe_states, self.envelope_low_q, axis=0)
        q_hi = np.quantile(safe_states, self.envelope_high_q, axis=0)
        new_center = 0.5 * (q_lo + q_hi)
        new_hw = np.maximum(0.5 * (q_hi - q_lo), 1e-4)

        dead_states = s_all[dead_mask]
        d = self.obs_dim_
        align_gain = np.zeros(d, dtype=float)
        fail_contrast = np.zeros(d, dtype=float)
        for j in range(d):
            if self.controllable_[j] <= 0.0:
                continue
            closeness = -np.abs(s_all[:, j] - new_center[j])
            if np.std(closeness) > 1e-9 and np.std(y_val) > 1e-9:
                r = float(np.corrcoef(closeness, y_val)[0, 1])
                if math.isfinite(r):
                    align_gain[j] = max(0.0, r)
            if len(dead_states) > 0:
                d_dead = float(np.mean(np.abs(dead_states[:, j] - new_center[j])))
                d_safe = float(np.mean(np.abs(safe_states[:, j] - new_center[j])))
                fail_contrast[j] = max(0.0, (d_dead - d_safe) / float(new_hw[j]))
        raw_w = np.where(
            self.controllable_ > 0.0,
            align_gain * (align_gain + fail_contrast),
            0.0,
        )
        tot_w = float(raw_w.sum())
        new_w = raw_w / tot_w if tot_w > 1e-9 else self.demand_weights_.copy()

        alpha = self.ema_alpha
        self.env_center_ = (1.0 - alpha) * self.env_center_ + alpha * new_center
        self.env_halfwidth_ = np.maximum(
            (1.0 - alpha) * self.env_halfwidth_ + alpha * new_hw, 1e-4,
        )
        w_mix = (1.0 - alpha) * self.demand_weights_ + alpha * new_w
        w_sum = float(w_mix.sum())
        if w_sum > 1e-9:
            self.demand_weights_ = w_mix / w_sum
        self._inv_scaled_w = self.demand_weights_ / self.env_halfwidth_
        self.updates += 1
        return True

    def snapshot(self) -> dict:
        return dict(
            update_every=int(self.update_every),
            ema_alpha=float(self.ema_alpha),
            max_online_frames=int(self.max_online_frames),
            min_online_frames=int(self.min_online_frames),
            episodes_seen=int(self.episodes_seen),
            online_frames=int(self._online_frames),
            updates=int(self.updates),
            env_center=tuple(round(float(c), 4) for c in self.env_center_),
            demand_weights=tuple(round(float(w), 4) for w in self.demand_weights_),
        )


class AdaptiveSearchController:
    """Strictly task-agnostic dual-confidence controller for adaptive segmented search.

    Combines two causal confidence layers:
      1. **Outer evaluator reliability & mode self-audit**: unlocks clean
         multi-step deepening only after the online stage evaluator clears
         ``meta_lcb`` (causal Wilson-LCB from ``SourceAbstainGate``), and
         continuously audits realized episode lengths against the running
         conservative baseline median via a second ``SourceAbstainGate`` so
         any seed/regime where deeper lookahead underperforms automatically
         reverts to the conservative fallback.
      2. **Inner decaying-threshold plan-state confidence**: releases shallow
         plans immediately when their monotonic path confidence clears
         ``tau_k = tau_0 * (conf_decay ** k)`` (optionally self-calibrated from
         a rolling quantile of Tier-0 confidences), while guaranteeing finite
         termination on the final tier.
    """

    def __init__(
        self,
        conf_threshold: float = 0.72,
        conf_decay: float = 0.80,
        conf_quantile: float = 0.0,
        conf_window: int = 256,
        meta_lcb: float = 0.65,
        stage_warmup: int = 5,
        audit_warmup: int = 8,
        audit_target_lcb: float = 0.25,
        audit_ratio: float = 0.55,
        audit_decay: float = 0.90,
        audit_z: float = 1.0,
    ):
        if not 0.0 <= float(conf_threshold) <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if not 0.0 < float(conf_decay) <= 1.0:
            raise ValueError("conf_decay must be in (0, 1]")
        if not 0.0 <= float(conf_quantile) <= 1.0:
            raise ValueError("conf_quantile must be in [0, 1]")
        if int(conf_window) < 8:
            raise ValueError("conf_window must be at least 8")
        if not 0.0 <= float(meta_lcb) <= 1.0:
            raise ValueError("meta_lcb must be in [0, 1]")
        if int(stage_warmup) < 0:
            raise ValueError("stage_warmup must be non-negative")
        if int(audit_warmup) < 1:
            raise ValueError("audit_warmup must be positive")
        if not 0.0 <= float(audit_target_lcb) <= 1.0:
            raise ValueError("audit_target_lcb must be in [0, 1]")
        if float(audit_ratio) < 0.0:
            raise ValueError("audit_ratio must be non-negative")

        from node_variables import SourceAbstainGate

        self.conf_threshold = float(conf_threshold)
        self.conf_decay = float(conf_decay)
        self.conf_quantile = float(conf_quantile)
        self.conf_window = int(conf_window)
        self.meta_lcb = float(meta_lcb)
        self.stage_warmup = int(stage_warmup)
        self.audit_ratio = float(audit_ratio)
        self.mode_audit_gate = SourceAbstainGate(
            warmup=int(audit_warmup),
            decay=float(audit_decay),
            target_lcb=float(audit_target_lcb),
            floor_weight=0.0,
            z=float(audit_z),
        )
        self._base_episode_lengths: List[float] = []
        self._conf_history: deque = deque(maxlen=self.conf_window)
        self.active: bool = False
        self.base_steps: int = 0
        self.adaptive_steps: int = 0
        self.tier_counts: Dict[int, int] = {}
        self.audit_blocked_episodes: int = 0
        self.episodes_seen: int = 0

    def begin_episode(self, stage_gate=None) -> bool:
        """Decide causally whether the upcoming episode uses adaptive deepening."""
        if self.meta_lcb <= 0.0:
            stage_ok = True
        else:
            stage_ok = (
                stage_gate is not None
                and int(getattr(stage_gate, "raw_trials", 0)) >= self.stage_warmup
                and float(stage_gate.lcb()) >= self.meta_lcb
            )
        audit_ok = True
        if (
            self.mode_audit_gate.target_lcb > 0.0
            and self.mode_audit_gate.raw_trials >= self.mode_audit_gate.warmup
            and self.mode_audit_gate.lcb() < self.mode_audit_gate.target_lcb
        ):
            audit_ok = False
        if stage_ok and not audit_ok:
            self.audit_blocked_episodes += 1
        self.active = bool(stage_ok and audit_ok)
        return self.active

    def record_episode(
        self, frames: int, used_adaptive: Optional[bool] = None,
    ) -> None:
        """Record realized episode survival for the causal mode self-audit."""
        ep_len = float(max(int(frames), 0))
        was_adaptive = self.active if used_adaptive is None else bool(used_adaptive)
        self.episodes_seen += 1
        if not was_adaptive:
            self._base_episode_lengths.append(ep_len)
        elif self._base_episode_lengths:
            base_med = float(np.median(self._base_episode_lengths))
            self.mode_audit_gate.observe(ep_len >= self.audit_ratio * base_med)

    def observe_tier0_confidence(self, conf: float) -> Tuple[float, float]:
        """Record Tier-0 plan confidence and return ``(tau_0, decay)``."""
        c = float(conf)
        if math.isfinite(c):
            self._conf_history.append(c)
        if self.conf_quantile > 0.0 and len(self._conf_history) >= 32:
            tau0 = float(np.quantile(self._conf_history, self.conf_quantile))
            return tau0, self.conf_decay
        return self.conf_threshold, self.conf_decay

    @staticmethod
    def threshold_for_tier(
        tier_index: int, tau0: float, decay: float, is_final: bool,
    ) -> float:
        """Return decaying release threshold for ``tier_index`` (final always passes)."""
        if is_final:
            return -1.0
        return float(tau0) * (float(decay) ** max(int(tier_index), 0))

    def record_step(
        self, used_adaptive: bool, tier_index: Optional[int] = None,
    ) -> None:
        if used_adaptive:
            self.adaptive_steps += 1
            if tier_index is not None:
                idx = int(tier_index)
                self.tier_counts[idx] = self.tier_counts.get(idx, 0) + 1
        else:
            self.base_steps += 1

    def snapshot(self) -> dict:
        tot_steps = self.base_steps + self.adaptive_steps
        return dict(
            conf_threshold=self.conf_threshold,
            conf_decay=self.conf_decay,
            conf_quantile=self.conf_quantile,
            meta_lcb=self.meta_lcb,
            active=bool(self.active),
            episodes_seen=int(self.episodes_seen),
            base_episodes=len(self._base_episode_lengths),
            audit_blocked_episodes=int(self.audit_blocked_episodes),
            base_steps=int(self.base_steps),
            adaptive_steps=int(self.adaptive_steps),
            adaptive_step_fraction=(
                round(float(self.adaptive_steps) / float(tot_steps), 4)
                if tot_steps > 0 else 0.0
            ),
            tier_counts={int(k): int(v) for k, v in sorted(self.tier_counts.items())},
            mode_audit_gate=self.mode_audit_gate.snapshot(),
        )
