"""Similarity-based stage types over integrated states.

Each realized stage contributes one sample: its non-temporal integrated state
``E(s)`` (already key-node weighted), plus the outcome that actually followed.
Stage types are clusters of samples whose weighted profiles are similar.  The
number of types is not fixed: agglomerative merging stops when the best
complete-linkage similarity falls below ``min_similarity``.

A type is only accepted for use when a held-out episode split says it is both
reachable and useful:

* coverage   the held-out profile is within ``min_similarity`` of a train type;
* margin     the best type is separated from the runner-up;
* gain       the type's train outcome distribution predicts the held-out
              outcome better than the global base distribution.

Unknown or low-support types return UNKNOWN.  Nothing here fabricates a type
for a profile that does not match, and no task label enters the similarity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StageSample:
    """One realized stage's integrated profile and its actual outcome."""

    episode: Hashable
    profile: np.ndarray
    outcome: Tuple
    start: int = 0
    end: int = 0

    def __post_init__(self):
        profile = np.asarray(self.profile, dtype=float).reshape(-1)
        if not profile.size:
            raise ValueError("a stage sample needs a non-empty profile")
        if not np.isfinite(profile).all():
            raise ValueError("profile must be finite")
        if (profile < 0.0).any():
            raise ValueError("profile must be non-negative")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "outcome", tuple(self.outcome))


def _normalize(profile):
    profile = np.asarray(profile, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(profile))
    if norm <= 1e-12:
        return np.zeros_like(profile)
    return profile / norm


def _cosine(first, second):
    return float(np.dot(first, second))


@dataclass
class StageType:
    type_id: int
    centroid: np.ndarray
    members: Tuple[int, ...]
    support: int
    outcomes: dict
    purity: float


class StageTypeClusters:
    """Complete-linkage clustering of integrated stage profiles."""

    def __init__(self, min_similarity: float = 0.8,
                 min_coverage: float = 0.5, min_margin: float = 0.0,
                 min_gain: float = 0.0, min_type_support: int = 2,
                 prior: float = 4.0):
        if not 0.0 < float(min_similarity) <= 1.0:
            raise ValueError("min_similarity must be in (0, 1]")
        if not 0.0 <= float(min_coverage) <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")
        if float(min_margin) < 0.0:
            raise ValueError("min_margin must be non-negative")
        if float(min_gain) < 0.0:
            raise ValueError("min_gain must be non-negative")
        if int(min_type_support) < 1:
            raise ValueError("min_type_support must be positive")
        if float(prior) <= 0.0:
            raise ValueError("prior must be positive")
        self.min_similarity = float(min_similarity)
        self.min_coverage = float(min_coverage)
        self.min_margin = float(min_margin)
        self.min_gain = float(min_gain)
        self.min_type_support = int(min_type_support)
        self.prior = float(prior)
        self.types = {}
        self.samples = ()
        self.outcome_vocabulary = ()
        self.global_outcomes = Counter()
        self.accepted = False
        self.coverage = 0.0
        self.mean_margin = 0.0
        self.predictive_gain = 0.0
        self.validation_support = 0

    @classmethod
    def fit(cls, samples: Sequence[StageSample],
            validation: Optional[Sequence[StageSample]] = None, **kwargs):
        model = cls(**kwargs)
        model.samples = tuple(samples)
        if not model.samples:
            return model
        model._cluster()
        if validation:
            model._accept(validation)
        return model

    def _cluster(self):
        profiles = [_normalize(sample.profile) for sample in self.samples]
        clusters = [[index] for index in range(len(profiles))]
        while len(clusters) > 1:
            best = None
            for left, right in combinations(range(len(clusters)), 2):
                similarity = min(
                    _cosine(profiles[first], profiles[second])
                    for first in clusters[left]
                    for second in clusters[right]
                )
                if best is None or similarity > best[0]:
                    best = (similarity, left, right)
            if best is None or best[0] < self.min_similarity:
                break
            _similarity, left, right = best
            clusters[left] = clusters[left] + clusters[right]
            del clusters[right]
        clusters.sort(key=lambda members: tuple(sorted(members)))
        for type_id, members in enumerate(clusters):
            centroid = _normalize(np.mean(
                [profiles[index] for index in members], axis=0,
            ))
            outcomes = Counter(
                self.samples[index].outcome for index in members
            )
            support = len(members)
            self.types[type_id] = StageType(
                type_id=int(type_id),
                centroid=centroid,
                members=tuple(sorted(members)),
                support=int(support),
                outcomes=dict(outcomes),
                purity=(
                    float(max(outcomes.values())) / float(support)
                    if support else 0.0
                ),
            )
        self.outcome_vocabulary = tuple(sorted(
            {sample.outcome for sample in self.samples}, key=repr,
        ))
        self.global_outcomes = Counter(
            sample.outcome for sample in self.samples
        )

    def _base_probability(self, outcome):
        total = float(sum(self.global_outcomes.values()))
        vocabulary = max(len(self.outcome_vocabulary), 1)
        count = float(self.global_outcomes.get(outcome, 0))
        return (count + 1.0) / (total + vocabulary)

    def assign(self, profile):
        """Best train type for a held-out profile, or UNKNOWN."""
        if not self.types:
            return None, 0.0, 0.0
        normalized = _normalize(profile)
        if not np.any(normalized):
            return None, 0.0, 0.0
        scores = sorted(
            (
                (_cosine(normalized, stage_type.centroid), type_id)
                for type_id, stage_type in self.types.items()
            ),
            reverse=True,
        )
        best_similarity, best_id = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        margin = float(best_similarity - runner_up)
        if best_similarity < self.min_similarity:
            return None, float(best_similarity), margin
        if self.types[best_id].support < self.min_type_support:
            return None, float(best_similarity), margin
        return int(best_id), float(best_similarity), margin

    def _accept(self, validation):
        coverage = 0
        margins = []
        gains = []
        for sample in validation:
            type_id, similarity, margin = self.assign(sample.profile)
            if type_id is None:
                continue
            coverage += 1
            margins.append(margin)
            stage_type = self.types[type_id]
            total = float(sum(stage_type.outcomes.values()))
            base = self._base_probability(sample.outcome)
            count = float(stage_type.outcomes.get(sample.outcome, 0))
            probability = (
                count + self.prior * base
            ) / (total + self.prior)
            gains.append(float(np.log(probability / base)))
        self.validation_support = len(validation)
        self.coverage = (
            float(coverage) / float(len(validation))
            if validation else 0.0
        )
        self.mean_margin = float(np.mean(margins)) if margins else 0.0
        self.predictive_gain = float(np.mean(gains)) if gains else 0.0
        self.accepted = bool(
            self.coverage >= self.min_coverage
            and self.mean_margin >= self.min_margin
            and self.predictive_gain >= self.min_gain
        )

    def snapshot(self):
        return dict(
            types=int(len(self.types)),
            samples=int(len(self.samples)),
            min_similarity=float(self.min_similarity),
            min_type_support=int(self.min_type_support),
            type_support=sorted(
                (stage_type.support for stage_type in self.types.values()),
                reverse=True,
            ),
            type_purity=sorted(
                (round(stage_type.purity, 6)
                 for stage_type in self.types.values()),
                reverse=True,
            ),
            validation_support=int(self.validation_support),
            coverage=float(self.coverage),
            mean_margin=float(self.mean_margin),
            predictive_gain=float(self.predictive_gain),
            accepted=bool(self.accepted),
        )
