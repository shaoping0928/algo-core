"""Predictive-equivalence tree over realized stage types.

Stage identity still starts from an exact key-signal signature.  Two different
signatures, however, can be predictively indistinguishable: they may be
followed by the same sibling set, last for similar durations, and accumulate
similar feedback.  This module merges exactly those types, and only those,
using held-out predictive likelihood as the criterion.

The tree is fit in three separated roles:

* *training* episodes supply the per-stage outcome profiles;
* *validation* episodes decide whether two profiles may be pooled (a merge is
  accepted only when the pooled profile is at least as likely as the separate
  ones within ``tolerance``);
* *test* episodes are untouched and are scored by the probe that consumes this
  object.

Nothing here uses task geometry, stage labels, goals, fixed group counts, or
Euclidean similarity between raw states.  Every parent is defined by a
realized-outcome equivalence decision, and every child keeps its directed
successor, duration and feedback evidence.  An unknown stage key has no leaf;
resolution falls back to the all-stage base profile (or UNKNOWN when even that
has no support), never to a fabricated sibling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from stage_outcome_memory import StageKey, StageOccurrence


UNKNOWN = "unknown"


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_native(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_native(item) for item in value)
    hash(value)
    return value


def stage_outcome(stages: Sequence[StageOccurrence], index: int):
    """The realized categorical outcome of the stage at ``index``."""
    stage = stages[index]
    if stage.terminal or stage.censored:
        return (None, bool(stage.terminal), bool(stage.censored))
    return (stages[index + 1].key, False, False)


_outcome = stage_outcome


@dataclass
class _Profile:
    """Sufficient statistics for successor, duration and feedback."""

    count: int = 0
    episodes: set = field(default_factory=set)
    outcomes: dict = field(default_factory=dict)
    duration_log_sum: float = 0.0
    duration_log_sq_sum: float = 0.0
    feedback_sum: Optional[np.ndarray] = None
    feedback_sq_sum: Optional[np.ndarray] = None

    def add(self, stage: StageOccurrence, outcome):
        self.count += 1
        self.episodes.add(stage.episode)
        self.outcomes[outcome] = int(self.outcomes.get(outcome, 0)) + 1
        value = float(np.log(max(stage.duration, 1e-9)))
        self.duration_log_sum += value
        self.duration_log_sq_sum += value * value
        feedback = np.asarray(stage.feedback, dtype=float).reshape(-1)
        if self.feedback_sum is None:
            self.feedback_sum = np.zeros_like(feedback)
            self.feedback_sq_sum = np.zeros_like(feedback)
        if self.feedback_sum.shape != feedback.shape:
            raise ValueError("feedback dimensions must be consistent")
        self.feedback_sum += feedback
        self.feedback_sq_sum += feedback * feedback

    def pooled(self, other):
        result = _Profile()
        result.count = self.count + other.count
        result.episodes = set(self.episodes) | set(other.episodes)
        result.outcomes = dict(self.outcomes)
        for outcome, count in other.outcomes.items():
            result.outcomes[outcome] = (
                int(result.outcomes.get(outcome, 0)) + int(count)
            )
        result.duration_log_sum = (
            self.duration_log_sum + other.duration_log_sum
        )
        result.duration_log_sq_sum = (
            self.duration_log_sq_sum + other.duration_log_sq_sum
        )
        if self.feedback_sum is None:
            result.feedback_sum = other.feedback_sum
            result.feedback_sq_sum = other.feedback_sq_sum
        elif other.feedback_sum is None:
            result.feedback_sum = self.feedback_sum
            result.feedback_sq_sum = self.feedback_sq_sum
        else:
            if self.feedback_sum.shape != other.feedback_sum.shape:
                raise ValueError("feedback dimensions must be consistent")
            result.feedback_sum = self.feedback_sum + other.feedback_sum
            result.feedback_sq_sum = (
                self.feedback_sq_sum + other.feedback_sq_sum
            )
        return result

    def duration_stats(self, global_profile, prior: float):
        n = max(self.count, 0)
        global_mean, global_var = global_profile.duration_moments()
        if n == 0:
            return global_mean, global_var
        mean = self.duration_log_sum / n
        var = max(self.duration_log_sq_sum / n - mean * mean, 0.0)
        shrunk_mean = (
            n * mean + prior * global_mean
        ) / (n + prior)
        pooled_var = (
            n * var + prior * global_var
            + n * prior / (n + prior) * (mean - global_mean) ** 2
        ) / (n + prior)
        return shrunk_mean, max(pooled_var, 1e-6)

    def duration_moments(self):
        if self.count == 0:
            return 0.0, 1.0
        mean = self.duration_log_sum / self.count
        var = max(self.duration_log_sq_sum / self.count - mean * mean, 0.0)
        return mean, max(var, 1e-6)

    def feedback_stats(self, global_profile, prior: float):
        if self.feedback_sum is None or self.count == 0:
            return global_profile.feedback_moments()
        n = self.count
        mean = self.feedback_sum / n
        var = np.maximum(self.feedback_sq_sum / n - mean * mean, 0.0)
        global_mean, global_var = global_profile.feedback_moments()
        shrunk_mean = (n * mean + prior * global_mean) / (n + prior)
        pooled_var = (
            n * var + prior * global_var
            + n * prior / (n + prior) * (mean - global_mean) ** 2
        ) / (n + prior)
        return shrunk_mean, np.maximum(pooled_var, 1e-6)

    def feedback_moments(self):
        if self.feedback_sum is None or self.count == 0:
            return np.zeros(1), np.ones(1)
        mean = self.feedback_sum / self.count
        var = np.maximum(self.feedback_sq_sum / self.count - mean * mean, 0.0)
        return mean, np.maximum(var, 1e-6)


def _gaussian_logpdf(value, mean, variance):
    value = np.asarray(value, dtype=float)
    mean = np.asarray(mean, dtype=float)
    variance = np.maximum(np.asarray(variance, dtype=float), 1e-9)
    return -0.5 * (
        np.log(2.0 * np.pi * variance)
        + (value - mean) ** 2 / variance
    )


def _categorical_probabilities(profile: _Profile, global_profile: _Profile,
                               prior: float):
    vocabulary = sorted(
        set(global_profile.outcomes) | set(profile.outcomes),
        key=repr,
    )
    total = int(sum(global_profile.outcomes.values()))
    result = {}
    for outcome in vocabulary:
        base = (
            float(global_profile.outcomes.get(outcome, 0)) / total
            if total else 1.0 / max(len(vocabulary), 1)
        )
        result[outcome] = (
            float(profile.outcomes.get(outcome, 0)) + prior * base
        ) / (profile.count + prior)
    return result


class PredictiveStageTree:
    """Agglomerative tree whose merges pass a held-out likelihood test."""

    def __init__(self, prior: float = 4.0, tolerance: float = 0.0,
                 min_validation_support: int = 4,
                 min_support: int = 2, min_episodes: int = 2,
                 unseen_probability: float = 1e-6):
        if float(prior) <= 0.0:
            raise ValueError("prior must be positive")
        if float(tolerance) < 0.0:
            raise ValueError("tolerance must be non-negative")
        if int(min_validation_support) < 1:
            raise ValueError("min_validation_support must be positive")
        if int(min_support) < 1 or int(min_episodes) < 1:
            raise ValueError("support thresholds must be positive")
        if not 0.0 < float(unseen_probability) < 1.0:
            raise ValueError("unseen_probability must be in (0, 1)")
        self.prior = float(prior)
        self.tolerance = float(tolerance)
        self.min_validation_support = int(min_validation_support)
        self.min_support = int(min_support)
        self.min_episodes = int(min_episodes)
        self.unseen_probability = float(unseen_probability)
        self.leaf_of = {}
        self.profiles = {}
        self.parent = {}
        self.level = {}
        self.merge_log = []
        self.global_profile = _Profile()
        self.vocabulary = ()
        self._next_id = 0

    def _new_id(self):
        node_id = int(self._next_id)
        self._next_id += 1
        return node_id

    def _add_stage(self, profile, stage, stages, index):
        outcome = _outcome(stages, index)
        profile.add(stage, outcome)
        self.global_profile.add(stage, outcome)
        return outcome

    @classmethod
    def fit(cls, train_episodes, validation_episodes,
            prior: float = 4.0, tolerance: float = 0.0,
            min_validation_support: int = 4,
            min_support: int = 2, min_episodes: int = 2):
        """Fit leaves on training episodes and merge them on validation."""
        tree = cls(
            prior=prior, tolerance=tolerance,
            min_validation_support=min_validation_support,
            min_support=min_support, min_episodes=min_episodes,
        )
        for stages in train_episodes:
            stages = list(stages)
            if not stages:
                continue
            for index, stage in enumerate(stages):
                key = tuple(_native(token) for token in stage.key)
                if key not in tree.leaf_of:
                    node_id = tree._new_id()
                    tree.leaf_of[key] = node_id
                    tree.profiles[node_id] = _Profile()
                    tree.parent[node_id] = None
                    tree.level[node_id] = 0
                tree._add_stage(
                    tree.profiles[tree.leaf_of[key]], stage, stages, index,
                )
        tree.vocabulary = tuple(sorted(
            tree.global_profile.outcomes, key=repr,
        ))
        if not tree.profiles:
            return tree

        validation = []
        for stages in validation_episodes:
            stages = list(stages)
            if not stages:
                continue
            for index, stage in enumerate(stages):
                key = tuple(_native(token) for token in stage.key)
                leaf = tree.leaf_of.get(key)
                if leaf is None:
                    continue
                validation.append((leaf, stage, _outcome(stages, index)))
        tree._merge(validation)
        return tree

    def _stage_loglik(self, profile: _Profile, stage: StageOccurrence,
                      outcome):
        if profile.count == 0:
            return -np.inf
        probabilities = _categorical_probabilities(
            profile, self.global_profile, self.prior,
        )
        probability = probabilities.get(
            outcome, self.unseen_probability,
        )
        score = float(np.log(probability))
        mean, variance = profile.duration_stats(
            self.global_profile, self.prior,
        )
        score += float(_gaussian_logpdf(
            np.log(max(stage.duration, 1e-9)), mean, variance,
        ))
        feedback_mean, feedback_var = profile.feedback_stats(
            self.global_profile, self.prior,
        )
        feedback = np.asarray(stage.feedback, dtype=float).reshape(-1)
        score += float(np.sum(_gaussian_logpdf(
            feedback, feedback_mean, feedback_var,
        )))
        return float(score)

    def _merge(self, validation):
        groups = {
            node_id: dict(profile=profile, members={node_id})
            for node_id, profile in self.profiles.items()
        }
        group_of = {node_id: node_id for node_id in groups}
        while len(groups) > 1:
            ids = sorted(groups)
            best = None
            for left_index in range(len(ids)):
                for right_index in range(left_index + 1, len(ids)):
                    left, right = ids[left_index], ids[right_index]
                    relevant = [
                        item for item in validation
                        if group_of[item[0]] in (left, right)
                    ]
                    if len(relevant) < self.min_validation_support:
                        continue
                    pooled = groups[left]["profile"].pooled(
                        groups[right]["profile"]
                    )
                    delta = 0.0
                    valid = True
                    for leaf, stage, outcome in relevant:
                        current = groups[group_of[leaf]]["profile"]
                        pooled_ll = self._stage_loglik(
                            pooled, stage, outcome,
                        )
                        current_ll = self._stage_loglik(
                            current, stage, outcome,
                        )
                        if not (np.isfinite(pooled_ll)
                                and np.isfinite(current_ll)):
                            # An outcome outside the model's support makes
                            # both scores -inf; that pair supplies no
                            # evidence, and -inf - -inf must not become nan.
                            valid = False
                            break
                        delta += pooled_ll - current_ll
                    if not valid:
                        continue
                    delta /= float(len(relevant))
                    if best is None or delta > best[0]:
                        best = (delta, left, right, pooled)
            if best is None or best[0] < -self.tolerance:
                break
            delta, left, right, pooled = best
            new_id = self._new_id()
            self.profiles[new_id] = pooled
            self.parent[new_id] = None
            self.level[new_id] = (
                max(self.level[left], self.level[right]) + 1
            )
            self.parent[left] = new_id
            self.parent[right] = new_id
            members = groups[left]["members"] | groups[right]["members"]
            groups[new_id] = dict(profile=pooled, members=members)
            for member in members:
                group_of[member] = new_id
            self.merge_log.append(dict(
                node=int(new_id),
                left=int(left),
                right=int(right),
                level=int(self.level[new_id]),
                validation_delta=float(delta),
                members=int(len(members)),
            ))
            del groups[left]
            del groups[right]

    def chain(self, stage_key):
        """Return the leaf-to-root ancestor ids for a known stage key, or ()."""
        key = tuple(_native(token) for token in stage_key)
        node_id = self.leaf_of.get(key)
        result = []
        while node_id is not None:
            result.append(node_id)
            node_id = self.parent.get(node_id)
        return tuple(result)

    def node_for(self, stage_key, min_support: Optional[int] = None,
                 min_episodes: Optional[int] = None):
        """First ancestor with enough support; None when even the root is thin."""
        support = self.min_support if min_support is None else int(min_support)
        episodes = (
            self.min_episodes if min_episodes is None else int(min_episodes)
        )
        for node_id in self.chain(stage_key):
            profile = self.profiles[node_id]
            if (profile.count >= support
                    and len(profile.episodes) >= episodes):
                return node_id
        return None

    def leaf_node(self, stage_key, min_support: Optional[int] = None,
                  min_episodes: Optional[int] = None):
        """The exact stage leaf when it has enough support, else None."""
        support = self.min_support if min_support is None else int(min_support)
        episodes = (
            self.min_episodes if min_episodes is None else int(min_episodes)
        )
        key = tuple(_native(token) for token in stage_key)
        node_id = self.leaf_of.get(key)
        if node_id is None:
            return None
        profile = self.profiles[node_id]
        if (profile.count < support
                or len(profile.episodes) < episodes):
            return None
        return node_id

    def predict(self, stage_key, min_support: Optional[int] = None,
                min_episodes: Optional[int] = None, ancestors: bool = True):
        """Outcome/duration/feedback readout at the supported node."""
        key = tuple(_native(token) for token in stage_key)
        node_id = (
            self.node_for(key, min_support, min_episodes)
            if ancestors else self.leaf_node(key, min_support, min_episodes)
        )
        if node_id is None:
            profile = self.global_profile
            if (profile.count < self.min_support
                    or len(profile.episodes) < self.min_episodes):
                return dict(
                    status=UNKNOWN, node=None, level=None,
                    support=int(profile.count),
                    episode_support=int(len(profile.episodes)),
                    probabilities={}, duration=None, feedback=None,
                    known_stage=key in self.leaf_of,
                )
            level = "base"
            known = key in self.leaf_of
        else:
            profile = self.profiles[node_id]
            level = int(self.level[node_id])
            known = True
        probabilities = _categorical_probabilities(
            profile, self.global_profile, self.prior,
        )
        duration_mean, duration_var = profile.duration_stats(
            self.global_profile, self.prior,
        )
        feedback_mean, feedback_var = profile.feedback_stats(
            self.global_profile, self.prior,
        )
        return dict(
            status="supported",
            node=node_id,
            level=level,
            support=int(profile.count),
            episode_support=int(len(profile.episodes)),
            probabilities=probabilities,
            duration=dict(
                log_mean=float(duration_mean),
                log_variance=float(duration_var),
            ),
            feedback=dict(
                mean=np.asarray(feedback_mean).tolist(),
                variance=np.asarray(feedback_var).tolist(),
            ),
            known_stage=bool(known),
        )

    def log_likelihood(self, stage: StageOccurrence, outcome,
                       min_support: Optional[int] = None,
                       min_episodes: Optional[int] = None,
                       ancestors: bool = True):
        """Smoothed joint log-likelihood of one held-out stage."""
        components = self.components(
            stage, outcome, min_support=min_support,
            min_episodes=min_episodes, ancestors=ancestors,
        )
        return None if components is None else components["joint"]

    def components(self, stage: StageOccurrence, outcome,
                   min_support: Optional[int] = None,
                   min_episodes: Optional[int] = None,
                   ancestors: bool = True):
        """Per-channel and joint log-likelihood at the selected node."""
        if ancestors:
            node_id = self.node_for(
                stage.key, min_support, min_episodes,
            )
        else:
            node_id = self.leaf_node(
                stage.key, min_support, min_episodes,
            )
        if node_id is None:
            # The all-stage base is a coarse baseline, not an ancestor that
            # the caller's node gate selected; it is available whenever the
            # tree itself has enough base support.
            profile = self.global_profile
            if profile.count < self.min_support:
                return None
        else:
            profile = self.profiles[node_id]
        probabilities = _categorical_probabilities(
            profile, self.global_profile, self.prior,
        )
        probability = probabilities.get(
            outcome, self.unseen_probability,
        )
        categorical = float(np.log(probability))
        mean, variance = profile.duration_stats(
            self.global_profile, self.prior,
        )
        duration = float(_gaussian_logpdf(
            np.log(max(stage.duration, 1e-9)), mean, variance,
        ))
        feedback_mean, feedback_var = profile.feedback_stats(
            self.global_profile, self.prior,
        )
        feedback = np.asarray(stage.feedback, dtype=float).reshape(-1)
        feedback_score = float(np.sum(_gaussian_logpdf(
            feedback, feedback_mean, feedback_var,
        )))
        return dict(
            node=node_id,
            level=(
                int(self.level[node_id]) if node_id is not None else "base"
            ),
            support=int(profile.count),
            episode_support=int(len(profile.episodes)),
            categorical=categorical,
            duration=duration,
            feedback=feedback_score,
            joint=float(categorical + duration + feedback_score),
        )

    def snapshot(self):
        parents = sum(
            1 for node_id in self.profiles if self.parent.get(node_id) is None
        )
        return dict(
            stage_types=int(len(self.leaf_of)),
            nodes=int(len(self.profiles)),
            merges=int(len(self.merge_log)),
            roots=int(parents),
            max_level=int(max(self.level.values(), default=0)),
            validation_support=int(self.min_validation_support),
            prior=float(self.prior),
            tolerance=float(self.tolerance),
            global_support=int(self.global_profile.count),
        )
