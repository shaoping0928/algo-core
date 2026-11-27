"""Task-agnostic stage regime summarization, drift stripping, and survival-tree partitioning."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
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
    m_norm = np.einsum("ij,ij->i", matrix, matrix)[:, None]
    for _ in range(int(max_iter)):
        c_norm = np.einsum("ij,ij->i", centers, centers)[None, :]
        d2 = matrix @ (-2.0 * centers.T)
        d2 += m_norm
        d2 += c_norm
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

