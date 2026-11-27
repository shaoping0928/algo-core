"""Streaming self-supervised feature discovery from real transitions.

Candidate predicates are proposed only at episode boundaries from bounded
past experience. Their thresholds are versioned and frozen. Each candidate
then passes a future-only verification block before it can be read as
supported; feedback is delayed until its target horizon is observed.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Mapping, Optional, Sequence

import numpy as np

from node_features import expand_observation
from predictive_feature_bank import EpisodeBalancedFeatureBank


_FAMILIES = (
    "state",
    "relation",
    "temporal",
    "prediction_error",
    "state_temporal_residual",
)
_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _episode_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class _OnlineCandidate:
    feature_id: str
    family: str
    expression: object
    ranges: tuple
    probability_active: float
    probability_inactive: float
    base_rate: float
    created_episode: int
    proposal_round: int
    status: str = "verifying"
    verification_elapsed: int = 0
    verification_gains: list = field(default_factory=list)
    verification_nll_gains: list = field(default_factory=list)
    verification_active_episodes: int = 0
    verification_inactive_episodes: int = 0
    active_rates: deque = field(default_factory=lambda: deque(maxlen=64))
    inactive_rates: deque = field(default_factory=lambda: deque(maxlen=64))
    monitor_gains: deque = field(default_factory=lambda: deque(maxlen=12))
    monitor_nll_gains: deque = field(default_factory=lambda: deque(maxlen=12))
    ood_episode_rates: deque = field(default_factory=lambda: deque(maxlen=12))
    episode_rows_seen: int = 0
    episode_ood_rows: int = 0
    last_reason: Optional[str] = None


class _StreamingLinearWorldModel:
    """Ridge normal-equation updates; predict first, then learn the row."""

    def __init__(self, input_dim, output_dim, ridge=1.0, min_samples=8):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.ridge = float(ridge)
        self.min_samples = int(min_samples)
        width = self.input_dim + 1
        self.gram = np.eye(width, dtype=float) * self.ridge
        self.cross = np.zeros((width, self.output_dim), dtype=float)
        self.samples = 0

    def _vector(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != self.input_dim:
            raise ValueError("world-model input width mismatch")
        return np.concatenate([values, np.ones(1, dtype=float)])

    def predict(self, values):
        if self.samples < self.min_samples:
            return np.zeros(self.output_dim, dtype=float), False
        weights = np.linalg.solve(self.gram, self.cross)
        return self._vector(values) @ weights, True

    def update(self, values, target):
        vector = self._vector(values)
        target = np.asarray(target, dtype=float).reshape(-1)
        if target.size != self.output_dim:
            raise ValueError("world-model target width mismatch")
        self.gram += np.outer(vector, vector)
        self.cross += np.outer(vector, target)
        self.samples += 1


class OnlineFeatureLearner:
    """Bounded-memory streaming feature proposal and future-only validation.

    The stream contract is one real transition at a time:
    ``obs, pending/history, next_obs, signals, done``. No goal, pass flag or
    task geometry is read. A feature's activation is stored when the source
    state is seen; its pain label is finalized only after the future window
    has elapsed or the episode terminates.
    """

    def __init__(
            self, obs_dim: int, lag: int, scales=None,
            label_horizon: int = 4, warmup_episodes: int = 6,
            discovery_interval: int = 4, discovery_window: int = 32,
            max_episode_samples: int = 128, max_features: int = 256,
            max_features_per_family: int = 48,
            min_discovery_support: int = 3,
            prior_episodes: float = 1.0,
            verification_episodes: int = 12,
            verification_timeout: int = 18,
            min_verification_support: int = 3,
            monitor_episodes: int = 12,
            min_model_samples: int = 8,
            seed: int = 0):
        if int(obs_dim) < 1 or int(lag) < 0:
            raise ValueError("observation size must be positive and lag valid")
        if int(label_horizon) < 1:
            raise ValueError("label horizon must be positive")
        if int(warmup_episodes) < 2 or int(discovery_interval) < 1:
            raise ValueError("episode discovery schedule is invalid")
        if int(discovery_window) < 2 or int(max_episode_samples) < 1:
            raise ValueError("discovery memory limits are invalid")
        if int(max_features) < 1 or int(max_features_per_family) < 1:
            raise ValueError("feature budgets must be positive")
        if (int(verification_episodes) < 4
                or int(verification_timeout) < int(verification_episodes)):
            raise ValueError("verification block is too short")
        if int(min_verification_support) < 1:
            raise ValueError("verification support must be positive")
        if float(prior_episodes) < 0.0:
            raise ValueError("prior episode mass must be non-negative")

        self.obs_dim = int(obs_dim)
        self.lag = int(lag)
        self.scales = (
            np.ones(self.obs_dim, dtype=float)
            if scales is None else np.asarray(scales, dtype=float).reshape(-1)
        )
        if self.scales.size < self.obs_dim:
            raise ValueError("scales must include every observation channel")
        self.scales = np.where(
            self.scales[:self.obs_dim] > 1e-8,
            self.scales[:self.obs_dim], 1.0,
        )
        self.label_horizon = int(label_horizon)
        self.warmup_episodes = int(warmup_episodes)
        self.discovery_interval = int(discovery_interval)
        self.discovery_window = int(discovery_window)
        self.max_episode_samples = int(max_episode_samples)
        self.max_features = int(max_features)
        self.max_features_per_family = int(max_features_per_family)
        self.min_discovery_support = int(min_discovery_support)
        self.prior_episodes = float(prior_episodes)
        self.verification_episodes = int(verification_episodes)
        self.verification_timeout = int(verification_timeout)
        self.min_verification_support = int(min_verification_support)
        self.monitor_episodes = int(monitor_episodes)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.world_model = _StreamingLinearWorldModel(
            self.obs_dim + self.lag + 1, self.obs_dim,
            min_samples=min_model_samples,
        )

        self.candidates = {}
        self.activation_signatures = set()
        self.feature_versions = Counter()
        self.proposal_round = 0
        self.discovery = deque(maxlen=self.discovery_window)
        self.base_rates = deque(maxlen=self.discovery_window)
        self.pending_labels = deque()
        self.episode_samples = []
        self.episode_sample_seen = 0
        self.episode_candidate_stats = {}
        self.episode_base_sum = 0.0
        self.episode_base_count = 0
        self.episode = None
        self.episode_count = 0
        self.transition_count = 0
        self.censored_samples = 0
        self.completed_labels = 0
        self.previous_observation = None
        self.previous_delta = None
        self.previous_error = np.zeros(self.obs_dim, dtype=float)
        self.previous_error_valid = False
        self.last_readout = dict(status="unknown", features=(), pain=None)
        self.counts = Counter()

    def _base_rate(self):
        if not self.base_rates:
            return 0.5
        return float(np.mean(self.base_rates))

    def _history(self, transition, pending):
        history = transition.get("history")
        if history is not None:
            values = np.asarray(history, dtype=float).reshape(-1)
            if values.size == self.lag + 1:
                return values
        action = int(transition.get("action", 0))
        return np.asarray(
            [float(action)]
            + [float(value) for value in np.asarray(pending)[::-1]],
            dtype=float,
        )

    def _family_values(self, observation, pending):
        observation = np.asarray(observation, dtype=float).reshape(-1)
        if observation.size != self.obs_dim:
            raise ValueError("observation width mismatch")
        obs = observation / self.scales
        pending = np.asarray(pending, dtype=float).reshape(-1)
        if pending.size != self.lag:
            raise ValueError("pending queue width mismatch")
        state = np.concatenate([obs, pending])
        expanded = expand_observation(obs)
        relation = expanded[self.obs_dim:]

        delta_valid = self.previous_observation is not None
        if delta_valid:
            delta = obs - self.previous_observation
            accel_valid = self.previous_delta is not None
            acceleration = (
                delta - self.previous_delta
                if accel_valid else np.zeros(self.obs_dim, dtype=float)
            )
        else:
            delta = np.zeros(self.obs_dim, dtype=float)
            acceleration = np.zeros(self.obs_dim, dtype=float)
            accel_valid = False
        temporal = np.concatenate([
            delta, acceleration,
            np.asarray([float(delta_valid), float(accel_valid)]),
        ])
        residual = np.concatenate([
            self.previous_error if self.previous_error_valid
            else np.zeros(self.obs_dim, dtype=float),
            np.asarray([float(self.previous_error_valid)]),
        ])
        return dict(
            state=state,
            relation=relation,
            temporal=temporal,
            prediction_error=residual,
            state_temporal_residual=np.concatenate([
                state, temporal, residual,
            ]),
        ), obs, delta, delta_valid

    def _candidate_snapshot(self, family_values, base_rate):
        snapshot = {}
        for feature_id, candidate in self.candidates.items():
            if candidate.status not in ("verifying", "validated"):
                continue
            values = family_values[candidate.family]
            applicable = bool(candidate.expression.applicable(
                values.reshape(1, -1), candidate.ranges,
            )[0])
            if candidate.status == "validated":
                candidate.episode_rows_seen += 1
                candidate.episode_ood_rows += int(not applicable)
            active = bool(candidate.expression.mask(
                values.reshape(1, -1),
            )[0])
            snapshot[feature_id] = dict(
                applicable=applicable,
                active=active,
                probability_active=candidate.probability_active,
                probability_inactive=candidate.probability_inactive,
                base_rate=float(base_rate),
                status=candidate.status,
            )
        return snapshot

    def _readout(self, family_values):
        predictions = []
        active_ids = []
        applicable_ids = []
        for feature_id, candidate in self.candidates.items():
            if candidate.status != "validated":
                continue
            values = family_values[candidate.family]
            row = values.reshape(1, -1)
            if not bool(candidate.expression.applicable(row, candidate.ranges)[0]):
                continue
            active = bool(candidate.expression.mask(row)[0])
            probability = (
                candidate.probability_active
                if active else candidate.probability_inactive
            )
            predictions.append(float(probability))
            applicable_ids.append(feature_id)
            if active:
                active_ids.append(feature_id)
        if not predictions:
            return dict(status="unknown", features=(), pain=None)
        return dict(
            status="supported",
            features=tuple(applicable_ids),
            active_features=tuple(active_ids),
            pain=float(np.mean(predictions)),
        )

    def _store_labeled_sample(self, sample, label):
        record = dict(family_values=sample["family_values"], label=float(label))
        self.episode_sample_seen += 1
        if len(self.episode_samples) < self.max_episode_samples:
            self.episode_samples.append(record)
            return
        index = int(self.rng.randint(0, self.episode_sample_seen))
        if index < self.max_episode_samples:
            self.episode_samples[index] = record

    def _complete_pending_sample(self, sample):
        label = float(sample["pain"])
        self.episode_base_sum += label
        self.episode_base_count += 1
        self._store_labeled_sample(sample, label)
        self.completed_labels += 1
        self.counts["labels_completed"] += 1
        for feature_id, state in sample["candidate_states"].items():
            candidate = self.candidates.get(feature_id)
            if candidate is None or not state["applicable"]:
                continue
            stats = self.episode_candidate_stats.setdefault(
                feature_id,
                dict(
                    active_count=0, active_sum=0.0,
                    inactive_count=0, inactive_sum=0.0,
                    brier_count=0, brier_sum=0.0,
                    nll_sum=0.0,
                    eligible_rows=0,
                ),
            )
            stats["eligible_rows"] += 1
            if state["active"]:
                stats["active_count"] += 1
                stats["active_sum"] += label
            else:
                stats["inactive_count"] += 1
                stats["inactive_sum"] += label
            if state["status"] in ("verifying", "validated"):
                probability = (
                    state["probability_active"]
                    if state["active"] else state["probability_inactive"]
                )
                base = float(state["base_rate"])
                p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
                b = float(np.clip(base, 1e-6, 1.0 - 1e-6))
                stats["brier_sum"] += (
                    (base - label) ** 2 - (p - label) ** 2
                )
                stats["brier_count"] += 1
                stats["nll_sum"] += (
                    -label * np.log(b) - (1.0 - label) * np.log(1.0 - b)
                    + label * np.log(p) + (1.0 - label) * np.log(1.0 - p)
                )

    def _resolve_current_labels(self, pain, terminal=False):
        for sample in self.pending_labels:
            sample["pain"] = max(
                float(sample["pain"]), float(pain),
            )
            sample["age"] += 1
        while (self.pending_labels
               and self.pending_labels[0]["age"] >= self.label_horizon):
            self._complete_pending_sample(self.pending_labels.popleft())
        if terminal:
            while self.pending_labels:
                self._complete_pending_sample(self.pending_labels.popleft())

    def observe_transition(self, transition: Mapping):
        """Consume exactly one newly observed transition and its feedback."""
        try:
            observation = np.asarray(
                transition["obs"], dtype=float,
            ).reshape(-1)
            next_observation = np.asarray(
                transition["next_obs"], dtype=float,
            ).reshape(-1)
            signals = np.asarray(
                transition["signals"], dtype=float,
            ).reshape(-1)
        except (KeyError, TypeError, ValueError):
            self.counts["invalid_rows"] += 1
            return dict(status="unknown", reason="invalid_transition")
        if (observation.size != self.obs_dim
                or next_observation.size != self.obs_dim
                or signals.size < 2
                or not np.isfinite(observation).all()
                or not np.isfinite(next_observation).all()
                or not np.isfinite(signals[:2]).all()):
            self.counts["invalid_rows"] += 1
            return dict(status="unknown", reason="invalid_transition")

        history = transition.get("history")
        pending = transition.get("pending")
        if pending is None:
            if history is None:
                pending = np.zeros(self.lag, dtype=float)
            else:
                history_values = np.asarray(
                    history, dtype=float,
                ).reshape(-1)
                if history_values.size <= self.lag:
                    self.counts["invalid_rows"] += 1
                    return dict(status="unknown", reason="missing_pending")
                pending = history_values[1:self.lag + 1][::-1]
        pending = np.asarray(pending, dtype=float).reshape(-1)
        if pending.size != self.lag or not np.isfinite(pending).all():
            self.counts["invalid_rows"] += 1
            return dict(status="unknown", reason="invalid_pending")
        if history is None:
            action = int(transition.get("action", 0))
            history = np.asarray(
                [float(action)] + list(pending[::-1]), dtype=float,
            )
        else:
            history = np.asarray(history, dtype=float).reshape(-1)
            if history.size != self.lag + 1 or not np.isfinite(history).all():
                self.counts["invalid_rows"] += 1
                return dict(status="unknown", reason="invalid_history")
            expected_pending = history[1:self.lag + 1][::-1]
            if not np.allclose(pending, expected_pending):
                self.counts["invalid_rows"] += 1
                return dict(
                    status="unknown", reason="pending_history_mismatch",
                )

        episode = _episode_value(transition.get("episode"))
        if (self.episode is not None and episode is not None
                and episode != self.episode):
            self._finish_episode(terminal=False, censored=True)
        if self.episode is None:
            self.episode = (
                episode if episode is not None else self.episode_count
            )

        obs_scaled = observation / self.scales
        pending_values = pending.astype(float)
        delta_valid = self.previous_observation is not None
        delta = (
            obs_scaled - self.previous_observation
            if delta_valid else np.zeros(self.obs_dim, dtype=float)
        )
        acceleration_valid = delta_valid and self.previous_delta is not None
        acceleration = (
            delta - self.previous_delta if acceleration_valid
            else np.zeros(self.obs_dim, dtype=float)
        )
        relation = expand_observation(obs_scaled)[self.obs_dim:]
        temporal = np.concatenate([
            delta, acceleration,
            [float(delta_valid), float(acceleration_valid)],
        ])
        residual = np.concatenate([
            self.previous_error if self.previous_error_valid
            else np.zeros(self.obs_dim, dtype=float),
            [float(self.previous_error_valid)],
        ])
        family_values = dict(
            state=np.concatenate([obs_scaled, pending_values]),
            relation=relation,
            temporal=temporal,
            prediction_error=residual,
            state_temporal_residual=np.concatenate([
                obs_scaled, pending_values, temporal, residual,
            ]),
        )

        base_rate = self._base_rate()
        candidate_states = self._candidate_snapshot(
            family_values, base_rate,
        )
        active_ids = []
        supported_ids = []
        predictions = []
        for feature_id, state in candidate_states.items():
            if state["applicable"]:
                supported_ids.append(feature_id)
                if state["active"]:
                    active_ids.append(feature_id)
                prediction = (
                    state["probability_active"] if state["active"]
                    else state["probability_inactive"]
                )
                predictions.append(float(prediction))

        self.last_readout = (
            dict(
                status="supported",
                features=tuple(supported_ids),
                active_features=tuple(active_ids),
                pain=float(np.mean(predictions)),
            ) if predictions else
            dict(status="unknown", features=(), active_features=(),
                 pain=None)
        )
        sample = dict(
            family_values=family_values,
            candidate_states=candidate_states,
            base_rate=base_rate,
            pain=0.0,
            age=0,
        )
        self.pending_labels.append(sample)
        pain = float(signals[1] > 0.5)
        self._resolve_current_labels(
            pain, terminal=bool(transition.get("done", False)),
        )

        model_input = np.concatenate([obs_scaled, history])
        next_scaled = next_observation / self.scales
        prediction, prediction_valid = self.world_model.predict(model_input)
        self.previous_error = (
            next_scaled - prediction
            if prediction_valid else np.zeros(self.obs_dim, dtype=float)
        )
        self.previous_error_valid = bool(prediction_valid)
        if prediction_valid:
            self.counts["residual_available_transitions"] += 1
        self.world_model.update(model_input, next_scaled)
        self.previous_delta = delta.copy() if delta_valid else None
        self.previous_observation = obs_scaled.copy()
        self.transition_count += 1
        self.counts["transitions"] += 1

        if bool(transition.get("done", False)):
            self._finish_episode(terminal=True, censored=False)
        return dict(self.last_readout)

    def _predictive_lower_bound(self, values, candidate):
        values = np.asarray(values, dtype=float)
        if values.size < int(self.verification_episodes):
            return None
        delta = (
            0.05 / float(max(self.max_features, 1))
            / float((int(candidate.proposal_round) + 1) ** 2)
        )
        z = NormalDist().inv_cdf(1.0 - delta)
        return float(
            values.mean() - z * values.std(ddof=1) / np.sqrt(values.size)
        )

    def _estimate_online_probabilities(self, candidate):
        base = self._base_rate()
        active = list(candidate.active_rates)
        inactive = list(candidate.inactive_rates)
        if active:
            candidate.probability_active = (
                sum(active) + self.prior_episodes * base
            ) / (len(active) + self.prior_episodes)
        if inactive:
            candidate.probability_inactive = (
                sum(inactive) + self.prior_episodes * base
            ) / (len(inactive) + self.prior_episodes)

    def _finish_episode(self, terminal=False, censored=False):
        if self.episode is None:
            return
        if terminal:
            self._resolve_current_labels(0.0, terminal=True)
        elif self.pending_labels:
            self.censored_samples += len(self.pending_labels)
            self.counts["censored_labels"] += len(self.pending_labels)
            self.pending_labels.clear()

        if self.episode_base_count:
            self.base_rates.append(
                self.episode_base_sum / float(self.episode_base_count)
            )

        for feature_id, candidate in list(self.candidates.items()):
            stats = self.episode_candidate_stats.get(
                feature_id,
                dict(
                    active_count=0, active_sum=0.0,
                    inactive_count=0, inactive_sum=0.0,
                    brier_count=0, brier_sum=0.0, nll_sum=0.0,
                ),
            )
            active_rate = (
                float(stats["active_sum"]) / float(stats["active_count"])
                if stats["active_count"] else None
            )
            inactive_rate = (
                float(stats["inactive_sum"])
                / float(stats["inactive_count"])
                if stats["inactive_count"] else None
            )
            brier_gain = (
                float(stats["brier_sum"]) / float(stats["brier_count"])
                if stats["brier_count"] else None
            )
            nll_gain = (
                float(stats["nll_sum"]) / float(stats["brier_count"])
                if stats["brier_count"] else None
            )
            if active_rate is not None:
                candidate.active_rates.append(active_rate)
            if inactive_rate is not None:
                candidate.inactive_rates.append(inactive_rate)
            if candidate.status == "verifying":
                candidate.verification_elapsed += 1
                if brier_gain is not None:
                    candidate.verification_gains.append(brier_gain)
                    candidate.verification_nll_gains.append(nll_gain)
                if active_rate is not None:
                    candidate.verification_active_episodes += 1
                if inactive_rate is not None:
                    candidate.verification_inactive_episodes += 1
                if (len(candidate.verification_gains)
                        >= self.verification_episodes):
                    lower = self._predictive_lower_bound(
                        candidate.verification_gains, candidate,
                    )
                    nll_gain = (
                        float(np.mean(candidate.verification_nll_gains))
                        if candidate.verification_nll_gains else -np.inf
                    )
                    if (lower is not None and lower > 0.0
                            and nll_gain > 0.0
                            and candidate.verification_active_episodes
                            >= self.min_verification_support
                            and candidate.verification_inactive_episodes
                            >= self.min_verification_support):
                        candidate.status = "validated"
                        candidate.last_reason = "future_episode_block_passed"
                        self._estimate_online_probabilities(candidate)
                        self.counts["features_validated"] += 1
                    else:
                        candidate.status = "rejected"
                        candidate.last_reason = "future_episode_block_failed"
                        self.counts["features_rejected"] += 1
                elif candidate.verification_elapsed >= self.verification_timeout:
                    candidate.status = "rejected"
                    candidate.last_reason = "verification_support_timeout"
                    self.counts["features_rejected"] += 1
            elif candidate.status == "validated":
                if brier_gain is not None:
                    candidate.monitor_gains.append(brier_gain)
                    candidate.monitor_nll_gains.append(nll_gain)
                self._estimate_online_probabilities(candidate)
                if (len(candidate.monitor_gains) >= self.monitor_episodes
                        and (
                            float(np.mean(candidate.monitor_gains)) <= 0.0
                            or float(np.mean(candidate.monitor_nll_gains)) <= 0.0
                        )):
                    candidate.status = "degraded"
                    candidate.last_reason = "rolling_gain_nonpositive"
                    self.counts["features_degraded"] += 1

        if self.episode_samples:
            family_rows = {
                family: np.asarray([
                    row["family_values"][family]
                    for row in self.episode_samples
                ], dtype=float)
                for family in _FAMILIES
            }
            labels = np.asarray([
                row["label"] for row in self.episode_samples
            ], dtype=float)
            self.discovery.append(dict(
                episode=self.episode_count,
                family_rows=family_rows,
                labels=labels,
            ))
            self.episode_count += 1
            self.counts["episodes_completed"] += 1
        if censored:
            self.counts["episodes_censored"] += 1

        self.pending_labels.clear()
        self.episode_samples = []
        self.episode_sample_seen = 0
        self.episode_candidate_stats = {}
        self.episode_base_sum = 0.0
        self.episode_base_count = 0
        self.previous_observation = None
        self.previous_delta = None
        self.previous_error = np.zeros(self.obs_dim, dtype=float)
        self.previous_error_valid = False
        self.episode = None

        if (self.episode_count >= self.warmup_episodes
                and self.episode_count % self.discovery_interval == 0):
            self._propose_candidates()

    def _propose_candidates(self):
        self.proposal_round += 1
        if not self.discovery:
            return
        if len(self.candidates) >= self.max_features:
            removable = sorted(
                (
                    candidate for candidate in self.candidates.values()
                    if candidate.status in ("rejected", "degraded")
                ),
                key=lambda candidate: candidate.created_episode,
            )
            while (len(self.candidates) >= self.max_features
                   and removable):
                candidate = removable.pop(0)
                self.candidates.pop(candidate.feature_id, None)
                self.counts["features_evicted"] += 1
        if len(self.candidates) >= self.max_features:
            self.counts["feature_budget_full"] += 1
            return
        added = 0
        for family in _FAMILIES:
            values = np.concatenate([
                episode["family_rows"][family]
                for episode in self.discovery
            ], axis=0)
            if values.ndim != 2 or values.shape[1] == 0:
                continue
            labels = np.concatenate([
                episode["labels"] for episode in self.discovery
            ])
            episode_ids = np.concatenate([
                np.full(len(episode["labels"]), episode["episode"],
                        dtype=object)
                for episode in self.discovery
            ])
            bank = EpisodeBalancedFeatureBank(
                family=family,
                min_train_episode_support=self.min_discovery_support,
                min_calibration_episode_support=1,
                min_coverage=0.01,
                prior_episodes=self.prior_episodes,
                max_compose_atoms=8,
                max_frontier=32,
                bootstrap_samples=16,
                seed=self.seed + self.episode_count,
            ).fit(values, labels, episode_ids)
            bands = (
                (0.0, 0.10), (0.10, 0.30), (0.30, 0.70),
                (0.70, 0.90), (0.90, 1.01),
            )
            records_by_band = []
            for lower, upper in bands:
                items = [
                    record for record in bank.records
                    if lower <= record.train_episode_coverage < upper
                ]
                items.sort(
                    key=lambda record: (
                        record.train_gain,
                        record.train_episode_support,
                        -record.expression.complexity,
                    ),
                    reverse=True,
                )
                records_by_band.append(items)
            records = []
            positions = [0] * len(records_by_band)
            while len(records) < self.max_features_per_family:
                added_from_band = False
                for band_index, items in enumerate(records_by_band):
                    position = positions[band_index]
                    if position < len(items):
                        records.append(items[position])
                        positions[band_index] += 1
                        added_from_band = True
                        if len(records) >= self.max_features_per_family:
                            break
                if not added_from_band:
                    break
            family_count = 0
            for record in records:
                if len(self.candidates) >= self.max_features:
                    break
                if family_count >= self.max_features_per_family:
                    break
                if record.feature_id in self.candidates:
                    continue
                self.candidates[record.feature_id] = _OnlineCandidate(
                    feature_id=record.feature_id,
                    family=family,
                    expression=record.expression,
                    ranges=record.ranges,
                    probability_active=record.probability_active,
                    probability_inactive=record.probability_inactive,
                    base_rate=bank.base_rate,
                    created_episode=self.episode_count,
                    proposal_round=self.proposal_round,
                )
                family_count += 1
                added += 1
            if len(self.candidates) >= self.max_features:
                break
        self.counts["features_proposed"] += added

    def snapshot(self):
        by_status = Counter(
            candidate.status for candidate in self.candidates.values()
        )
        return dict(
            episodes=int(self.episode_count),
            transitions=int(self.transition_count),
            world_model_samples=int(self.world_model.samples),
            candidates=int(len(self.candidates)),
            statuses={
                str(key): int(value)
                for key, value in sorted(by_status.items())
            },
            counters={
                str(key): int(value)
                for key, value in sorted(self.counts.items())
            },
            features=[
                dict(
                    feature_id=candidate.feature_id,
                    family=candidate.family,
                    label=candidate.expression.label(),
                    status=candidate.status,
                    created_episode=int(candidate.created_episode),
                    probability_active=float(
                        candidate.probability_active
                    ),
                    probability_inactive=float(
                        candidate.probability_inactive
                    ),
                    active_episode_support=int(
                        len(candidate.active_rates)
                    ),
                    inactive_episode_support=int(
                        len(candidate.inactive_rates)
                    ),
                    verification_episodes=int(
                        len(candidate.verification_gains)
                    ),
                    proposal_round=int(candidate.proposal_round),
                    recent_ood_episode_rate=(
                        float(np.mean(candidate.ood_episode_rates))
                        if candidate.ood_episode_rates else None
                    ),
                    last_reason=candidate.last_reason,
                )
                for candidate in self.candidates.values()
            ],
            label_horizon=int(self.label_horizon),
            discovery_interval=int(self.discovery_interval),
            discovery_window=int(self.discovery_window),
            max_features=int(self.max_features),
        )

    def end_episode(self, terminal=False):
        """Close the current stream, discarding unobserved censored labels."""
        if self.episode is None:
            return
        self._finish_episode(
            terminal=bool(terminal), censored=not bool(terminal),
        )
