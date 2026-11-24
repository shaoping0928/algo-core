"""Predictive-equivalence classes over realized action options.

An option is an ordered action trace executed inside one realized stage.  Two
different traces can be predictively indistinguishable: at the same source
stage they may reach the same sibling set, last for similar durations and
accumulate similar feedback.  This module pools exactly those traces and keeps
the rest separate, so a search no longer has to treat every realized trace as
its own action.

The fit roles follow the stage tree:

* *training* episodes supply one outcome profile per ``(source stage, trace)``;
* a merge is a model-selection decision on training profiles: pooling may cost
  at most ``tolerance`` nats per sample, measured with the shrunk outcome,
  duration and feedback likelihoods;
* *validation* episodes are untouched and decide whether the quotient as a
  whole is accepted over the exact-trace model.

Classes are scoped to the source stage type: the same trace may belong to
different classes at different stages.  An unseen trace, or a class below the
support gates, returns UNKNOWN; no sibling, pooled or counterfactual option is
fabricated.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence, Tuple

import numpy as np

from stage_outcome_memory import (
    ActionTrace,
    StageKey,
    StageOccurrence,
    _OutcomeCounts,
    _native,
)


UNKNOWN = "unknown"


def _log(value):
    return float(np.log(max(float(value), 1e-9)))


def episode_option_samples(episodes):
    """Flatten realized stages into ``(stage, trace, destination)`` samples."""
    samples = []
    for stages in episodes:
        stages = list(stages)
        for index, source in enumerate(stages):
            is_last = index == len(stages) - 1
            terminal = bool(source.terminal)
            censored = bool(source.censored or (is_last and not terminal))
            destination = None
            if not (terminal or censored) and not is_last:
                destination = stages[index + 1]
            samples.append((
                source, tuple(source.action_trace), destination,
                (None if destination is None else destination.key,
                 terminal, censored),
            ))
    return samples


def _merge_counts(left: _OutcomeCounts, right: _OutcomeCounts):
    result = _OutcomeCounts()
    result.count = int(left.count) + int(right.count)
    result.episodes = set(left.episodes) | set(right.episodes)
    result.durations = list(left.durations) + list(right.durations)
    if left.feedback_sum is None:
        result.feedback_sum = (
            None if right.feedback_sum is None else right.feedback_sum.copy()
        )
        result.feedback_sq_sum = (
            None if right.feedback_sq_sum is None
            else right.feedback_sq_sum.copy()
        )
    elif right.feedback_sum is None:
        result.feedback_sum = left.feedback_sum.copy()
        result.feedback_sq_sum = left.feedback_sq_sum.copy()
    else:
        if left.feedback_sum.shape != right.feedback_sum.shape:
            raise ValueError("feedback dimensions must be consistent")
        result.feedback_sum = left.feedback_sum + right.feedback_sum
        result.feedback_sq_sum = left.feedback_sq_sum + right.feedback_sq_sum
    result.destination_terminal_count = (
        int(left.destination_terminal_count)
        + int(right.destination_terminal_count)
    )
    result.destination_censored_count = (
        int(left.destination_censored_count)
        + int(right.destination_censored_count)
    )
    return result


@dataclass
class OptionProfile:
    """Sufficient statistics for one option (or one merged class)."""

    count: int = 0
    episodes: set = field(default_factory=set)
    outcomes: dict = field(default_factory=dict)
    duration_log_sum: float = 0.0
    duration_log_sq_sum: float = 0.0
    feedback_sum: Optional[np.ndarray] = None
    feedback_sq_sum: Optional[np.ndarray] = None

    def add(self, source: StageOccurrence,
            destination: Optional[StageOccurrence],
            outcome) -> None:
        self.count += 1
        self.episodes.add(source.episode)
        counts = self.outcomes.get(outcome)
        if counts is None:
            counts = _OutcomeCounts()
            self.outcomes[outcome] = counts
        counts.add(source, destination)
        stats = source if destination is None else destination
        value = _log(stats.duration)
        self.duration_log_sum += value
        self.duration_log_sq_sum += value * value
        feedback = np.asarray(stats.feedback, dtype=float).reshape(-1)
        if self.feedback_sum is None:
            self.feedback_sum = np.zeros_like(feedback)
            self.feedback_sq_sum = np.zeros_like(feedback)
        if self.feedback_sum.shape != feedback.shape:
            raise ValueError("feedback dimensions must be consistent")
        self.feedback_sum += feedback
        self.feedback_sq_sum += feedback * feedback

    def pooled(self, other: "OptionProfile") -> "OptionProfile":
        result = OptionProfile()
        result.count = int(self.count) + int(other.count)
        result.episodes = set(self.episodes) | set(other.episodes)
        result.outcomes = {
            key: counts for key, counts in self.outcomes.items()
        }
        for outcome, counts in other.outcomes.items():
            if outcome in result.outcomes:
                result.outcomes[outcome] = _merge_counts(
                    result.outcomes[outcome], counts,
                )
            else:
                result.outcomes[outcome] = counts
        result.duration_log_sum = (
            self.duration_log_sum + other.duration_log_sum
        )
        result.duration_log_sq_sum = (
            self.duration_log_sq_sum + other.duration_log_sq_sum
        )
        if self.feedback_sum is None:
            result.feedback_sum = (
                None if other.feedback_sum is None
                else other.feedback_sum.copy()
            )
            result.feedback_sq_sum = (
                None if other.feedback_sq_sum is None
                else other.feedback_sq_sum.copy()
            )
        elif other.feedback_sum is None:
            result.feedback_sum = self.feedback_sum.copy()
            result.feedback_sq_sum = self.feedback_sq_sum.copy()
        else:
            if self.feedback_sum.shape != other.feedback_sum.shape:
                raise ValueError("feedback dimensions must be consistent")
            result.feedback_sum = self.feedback_sum + other.feedback_sum
            result.feedback_sq_sum = (
                self.feedback_sq_sum + other.feedback_sq_sum
            )
        return result

    def duration_moments(self):
        if self.count <= 0:
            return 0.0, 1.0
        mean = self.duration_log_sum / float(self.count)
        variance = max(
            self.duration_log_sq_sum / float(self.count) - mean * mean,
            0.0,
        )
        return float(mean), float(max(variance, 1e-6))

    def feedback_moments(self):
        if self.feedback_sum is None or self.count <= 0:
            return np.zeros(1), np.ones(1)
        mean = self.feedback_sum / float(self.count)
        variance = np.maximum(
            self.feedback_sq_sum / float(self.count) - mean * mean,
            0.0,
        )
        return mean, np.maximum(variance, 1e-6)

    def summary_rows(self):
        total = int(sum(
            counts.count for counts in self.outcomes.values()
        ))
        rows = []
        for outcome, counts in sorted(
                self.outcomes.items(), key=lambda item: repr(item[0])):
            destination, terminal, censored = outcome
            rows.append(counts.summary(
                destination, terminal, censored, total,
            ))
        return tuple(rows)


def _shrunk(moments, base_moments, count: int, prior: float):
    mean, variance = moments
    base_mean, base_variance = base_moments
    if count <= 0:
        return base_mean, base_variance
    shrunk_mean = (count * mean + prior * base_mean) / (count + prior)
    pooled_variance = (
        count * variance + prior * base_variance
        + count * prior / (count + prior) * (mean - base_mean) ** 2
    ) / (count + prior)
    return shrunk_mean, np.maximum(pooled_variance, 1e-6)


def _smoothed_probabilities(profile: OptionProfile, base: OptionProfile,
                            prior: float):
    vocabulary = sorted(
        set(profile.outcomes) | set(base.outcomes), key=repr,
    )
    base_total = int(sum(
        counts.count for counts in base.outcomes.values()
    ))
    result = {}
    for outcome in vocabulary:
        base_probability = (
            float(base.outcomes[outcome].count) / float(base_total)
            if outcome in base.outcomes and base_total
            else 1.0 / max(len(vocabulary), 1)
        )
        count = (
            float(profile.outcomes[outcome].count)
            if outcome in profile.outcomes else 0.0
        )
        result[outcome] = (
            count + prior * base_probability
        ) / (profile.count + prior)
    return result


def _profile_nll(profile: OptionProfile, base: OptionProfile) -> float:
    """MLE negative log-likelihood used only for the merge decision.

    Prediction uses the smoothed/Dirichlet form; the merge comparison must not,
    otherwise two identical empirical distributions would look different just
    because the base profile adds other outcomes.  Variances are floored by the
    base scale so a single-sample group cannot look infinitely likely.
    """
    if profile.count <= 0:
        return 0.0
    total = 0.0
    for counts in profile.outcomes.values():
        probability = float(counts.count) / float(profile.count)
        total -= float(counts.count) * float(np.log(
            max(probability, 1e-12)
        ))
    base_duration_mean, base_duration_var = base.duration_moments()
    mean, variance = profile.duration_moments()
    sigma2 = max(float(variance), 0.1 * float(base_duration_var), 1e-3)
    total += 0.5 * (
        profile.count * np.log(2.0 * np.pi * sigma2)
        + profile.count * float(variance) / sigma2
    )
    if profile.feedback_sum is not None:
        base_feedback_mean, base_feedback_var = base.feedback_moments()
        mean, variance = profile.feedback_moments()
        sigma2 = np.maximum(
            np.asarray(variance, dtype=float),
            np.maximum(0.1 * np.asarray(base_feedback_var, dtype=float), 1e-3),
        )
        total += 0.5 * float(np.sum(
            profile.count * np.log(2.0 * np.pi * sigma2)
            + profile.count * np.asarray(variance, dtype=float) / sigma2
        ))
    return float(total)


@dataclass
class OptionClass:
    class_id: int
    stage_key: StageKey
    members: Tuple[ActionTrace, ...]
    representative: ActionTrace
    support: int
    episode_support: int
    profile: OptionProfile

    @property
    def token(self):
        return ("option_class", int(self.class_id))


class OptionQuotient:
    """Merge realized options that are predictively indistinguishable."""

    def __init__(self, prior: float = 4.0, tolerance: float = 0.05,
                 min_support: int = 2, min_episodes: int = 1,
                 acceptance_tolerance: float = 0.0,
                 unseen_probability: float = 1e-6):
        if float(prior) <= 0.0:
            raise ValueError("prior must be positive")
        if float(tolerance) < 0.0:
            raise ValueError("tolerance must be non-negative")
        if int(min_support) < 1 or int(min_episodes) < 1:
            raise ValueError("support thresholds must be positive")
        if float(acceptance_tolerance) < 0.0:
            raise ValueError("acceptance_tolerance must be non-negative")
        if not 0.0 < float(unseen_probability) < 1.0:
            raise ValueError("unseen_probability must be in (0, 1)")
        self.prior = float(prior)
        self.tolerance = float(tolerance)
        self.min_support = int(min_support)
        self.min_episodes = int(min_episodes)
        self.acceptance_tolerance = float(acceptance_tolerance)
        self.unseen_probability = float(unseen_probability)
        self.profiles = {}
        self.stage_base = {}
        self.global_profile = OptionProfile()
        self.classes = {}
        self.class_of = {}
        self.by_stage = {}
        self.merge_log = []
        self.accepted = False
        self.acceptance_delta = None
        self.acceptance_support = 0

    @classmethod
    def fit(cls, train_episodes, validation_episodes=None, **kwargs):
        quotient = cls(**kwargs)
        quotient._fit_profiles(train_episodes)
        quotient._merge()
        quotient._assign_classes()
        if validation_episodes:
            quotient._accept(validation_episodes)
        return quotient

    def _fit_profiles(self, episodes):
        by_stage = defaultdict(list)
        for source, trace, destination, _outcome in (
                episode_option_samples(episodes)):
            stage_key = tuple(
                _native(token) for token in source.key
            )
            key = (stage_key, trace)
            profile = self.profiles.get(key)
            if profile is None:
                profile = OptionProfile()
                self.profiles[key] = profile
                by_stage[stage_key].append(trace)
            profile.add(source, destination, _outcome)
            base = self.stage_base.get(stage_key)
            if base is None:
                base = OptionProfile()
                self.stage_base[stage_key] = base
            base.add(source, destination, _outcome)
            self.global_profile.add(source, destination, _outcome)
        self.by_stage = {
            stage_key: tuple(traces)
            for stage_key, traces in by_stage.items()
        }

    def _qualifies(self, profile: OptionProfile) -> bool:
        return (
            profile.count >= self.min_support
            and len(profile.episodes) >= self.min_episodes
        )

    def _merge(self):
        groups_by_stage = {}
        for stage_key, traces in self.by_stage.items():
            base = self.stage_base[stage_key]
            groups = {}
            members = {}
            for index, trace in enumerate(traces):
                profile = self.profiles[(stage_key, trace)]
                if not self._qualifies(profile):
                    continue
                groups[index] = profile
                members[index] = {trace}
            next_id = len(traces)
            while len(groups) > 1:
                best = None
                for left, right in combinations(sorted(groups), 2):
                    left_profile = groups[left]
                    right_profile = groups[right]
                    count = left_profile.count + right_profile.count
                    if count <= 0:
                        continue
                    pooled = left_profile.pooled(right_profile)
                    delta = (
                        _profile_nll(pooled, base)
                        - _profile_nll(left_profile, base)
                        - _profile_nll(right_profile, base)
                    ) / float(count)
                    if best is None or delta < best[0]:
                        best = (delta, left, right, pooled)
                if best is None or best[0] > self.tolerance:
                    break
                delta, left, right, pooled = best
                groups[next_id] = pooled
                members[next_id] = set(members[left]) | set(members[right])
                del groups[left]
                del groups[right]
                self.merge_log.append(dict(
                    stage_key=stage_key,
                    left=int(left),
                    right=int(right),
                    support=int(pooled.count),
                    delta=float(delta),
                ))
                next_id += 1
            groups_by_stage[stage_key] = [
                (tuple(sorted(members[group], key=repr)), groups[group])
                for group in sorted(groups)
            ]
        self._groups_by_stage = groups_by_stage

    def _assign_classes(self):
        class_id = 0
        for stage_key in sorted(
                self._groups_by_stage, key=repr):
            assigned = []
            entries = sorted(
                self._groups_by_stage[stage_key],
                key=lambda item: tuple(repr(trace) for trace in item[0]),
            )
            for members, profile in entries:
                option_class = OptionClass(
                    class_id=int(class_id),
                    stage_key=stage_key,
                    members=tuple(members),
                    representative=min(members, key=repr),
                    support=int(profile.count),
                    episode_support=int(len(profile.episodes)),
                    profile=profile,
                )
                self.classes[class_id] = option_class
                assigned.append(int(class_id))
                for trace in members:
                    self.class_of[(stage_key, trace)] = int(class_id)
                class_id += 1
            self.by_stage[stage_key] = tuple(assigned)

    def _accept(self, episodes):
        candidate = 0.0
        exact = 0.0
        support = 0
        for source, trace, destination, outcome in (
                episode_option_samples(episodes)):
            stage_key = tuple(
                _native(token) for token in source.key
            )
            exact_profile = self.profiles.get((stage_key, trace))
            class_id = self.class_of.get((stage_key, trace))
            if exact_profile is None or class_id is None:
                continue
            class_profile = self.classes[class_id].profile
            base = self.stage_base.get(stage_key, self.global_profile)
            candidate += self._sample_loglik(
                class_profile, base, source, destination, outcome,
            )
            exact += self._sample_loglik(
                exact_profile, base, source, destination, outcome,
            )
            support += 1
        if not support:
            self.accepted = False
            self.acceptance_delta = None
            self.acceptance_support = 0
            return
        delta = (candidate - exact) / float(support)
        self.acceptance_delta = float(delta)
        self.acceptance_support = int(support)
        self.accepted = bool(delta >= -self.acceptance_tolerance)

    def _sample_loglik(self, profile, base, source, destination, outcome):
        probabilities = _smoothed_probabilities(profile, base, self.prior)
        probability = probabilities.get(outcome, self.unseen_probability)
        score = float(np.log(max(probability, self.unseen_probability)))
        stats = source if destination is None else destination
        mean, variance = _shrunk(
            profile.duration_moments(), base.duration_moments(),
            profile.count, self.prior,
        )
        score += float(-0.5 * (
            np.log(2.0 * np.pi * variance)
            + (_log(stats.duration) - mean) ** 2 / variance
        ))
        if profile.feedback_sum is not None:
            mean, variance = _shrunk(
                profile.feedback_moments(), base.feedback_moments(),
                profile.count, self.prior,
            )
            feedback = np.asarray(stats.feedback, dtype=float).reshape(-1)
            score += float(np.sum(-0.5 * (
                np.log(2.0 * np.pi * variance)
                + (feedback - mean) ** 2 / variance
            )))
        return float(score)

    def classes_for(self, stage_key):
        stage_key = tuple(_native(token) for token in stage_key)
        result = []
        for class_id in self.by_stage.get(stage_key, ()):
            option_class = self.classes[class_id]
            if not self._qualifies(option_class.profile):
                continue
            result.append(int(class_id))
        return tuple(result)

    def class_for(self, stage_key, trace):
        stage_key = tuple(_native(token) for token in stage_key)
        return self.class_of.get((stage_key, tuple(trace)))

    def predict_class(self, class_id):
        option_class = self.classes.get(int(class_id))
        if option_class is None:
            return dict(
                status=UNKNOWN, outcomes=(), context_depth=None,
                resolution=None, support=0, episode_support=0,
                class_id=None, members=(),
            )
        profile = option_class.profile
        if not self._qualifies(profile):
            return dict(
                status=UNKNOWN, outcomes=(), context_depth=None,
                resolution=None, support=int(profile.count),
                episode_support=int(len(profile.episodes)),
                class_id=int(class_id), members=option_class.members,
            )
        return dict(
            status="supported",
            outcomes=profile.summary_rows(),
            context_depth=1,
            resolution=1,
            support=int(profile.count),
            episode_support=int(len(profile.episodes)),
            class_id=int(class_id),
            members=option_class.members,
            representative=option_class.representative,
        )

    def predict(self, stage_key, trace):
        class_id = self.class_for(stage_key, trace)
        if class_id is None:
            return dict(
                status=UNKNOWN, outcomes=(), context_depth=None,
                resolution=None, support=0, episode_support=0,
                class_id=None, members=(),
            )
        return self.predict_class(class_id)

    def snapshot(self):
        trace_count = len(self.profiles)
        return dict(
            stage_types=int(len(self.by_stage)),
            traces=int(trace_count),
            classes=int(len(self.classes)),
            merges=int(len(self.merge_log)),
            reduction=(
                float(1.0 - len(self.classes) / trace_count)
                if trace_count else 0.0
            ),
            prior=float(self.prior),
            tolerance=float(self.tolerance),
            min_support=int(self.min_support),
            min_episodes=int(self.min_episodes),
            accepted=bool(self.accepted),
            acceptance_delta=(
                None if self.acceptance_delta is None
                else float(self.acceptance_delta)
            ),
            acceptance_support=int(self.acceptance_support),
        )
