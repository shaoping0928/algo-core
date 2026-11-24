"""Similarity layers over integrated stage types.

Stage types are not a flat set: profiles that are merely similar at a coarse
scale may separate at a finer one.  This module builds the full complete-linkage
merge tree over integrated stage profiles and chooses a **frontier** of that
tree, so the number of layers is selected by a held-out criterion instead of a
fixed K.

The selection is parsimonious: among frontiers whose held-out coverage and
predictive gain pass the gates, keep the one with the fewest layers whose gain
is within ``tolerance`` of the best.  A coarse layer that already predicts as
well as a finer one wins.  Profiles that do not reach any selected layer return
UNKNOWN.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence, Tuple

import numpy as np

from stage_type_clusters import StageSample, _cosine, _normalize


@dataclass(frozen=True)
class LayerNode:
    node_id: int
    members: Tuple[int, ...]
    centroid: np.ndarray
    outcomes: dict
    support: int
    level: float
    left: Optional[int] = None
    right: Optional[int] = None


class StageTypeLayers:
    """Complete-linkage merge tree with a held-out frontier selection."""

    def __init__(self, min_similarity: float = 0.5,
                 min_coverage: float = 0.5, min_gain: float = 0.01,
                 tolerance: float = 0.01, min_layer_support: int = 2,
                 prior: float = 4.0):
        if not 0.0 < float(min_similarity) <= 1.0:
            raise ValueError("min_similarity must be in (0, 1]")
        if not 0.0 <= float(min_coverage) <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")
        if float(min_gain) < 0.0:
            raise ValueError("min_gain must be non-negative")
        if float(tolerance) < 0.0:
            raise ValueError("tolerance must be non-negative")
        if int(min_layer_support) < 1:
            raise ValueError("min_layer_support must be positive")
        if float(prior) <= 0.0:
            raise ValueError("prior must be positive")
        self.min_similarity = float(min_similarity)
        self.min_coverage = float(min_coverage)
        self.min_gain = float(min_gain)
        self.tolerance = float(tolerance)
        self.min_layer_support = int(min_layer_support)
        self.prior = float(prior)
        self.nodes = {}
        self.partitions = ()
        self.partition = ()
        self.layers = ()
        self.samples = ()
        self.outcome_vocabulary = ()
        self.global_outcomes = Counter()
        self.accepted = False
        self.coverage = 0.0
        self.mean_margin = 0.0
        self.predictive_gain = 0.0
        self.level = 0.0
        self.candidates = 0
        self.validation_support = 0

    @classmethod
    def fit(cls, samples: Sequence[StageSample],
            validation: Optional[Sequence[StageSample]] = None, **kwargs):
        model = cls(**kwargs)
        model.samples = tuple(samples)
        if not model.samples:
            return model
        model._build_tree()
        if validation:
            model._select(validation)
        return model

    def build(self, samples: Sequence[StageSample]):
        """Build the merge tree without selecting a frontier yet."""
        self.samples = tuple(samples)
        self.nodes = {}
        self.partitions = ()
        self.partition = ()
        self.layers = ()
        self.accepted = False
        if self.samples:
            self._build_tree()
        return self

    def _build_tree(self):
        profiles = [_normalize(sample.profile) for sample in self.samples]
        for index, sample in enumerate(self.samples):
            self.nodes[index] = LayerNode(
                node_id=int(index),
                members=(index,),
                centroid=profiles[index],
                outcomes=Counter({sample.outcome: 1}),
                support=1,
                level=1.0,
            )
        active = list(range(len(profiles)))
        similarity = {}
        for left, right in combinations(active, 2):
            similarity[(left, right)] = _cosine(
                profiles[left], profiles[right],
            )
        partitions = [(1.0, tuple(active))]
        next_id = len(profiles)
        while len(active) > 1:
            pair, value = max(
                similarity.items(),
                key=lambda item: (item[1], -item[0][0], -item[0][1]),
            )
            left, right = pair
            members = self.nodes[left].members + self.nodes[right].members
            centroid = _normalize(np.mean(
                [profiles[index] for index in members], axis=0,
            ))
            outcomes = Counter(self.nodes[left].outcomes)
            outcomes.update(self.nodes[right].outcomes)
            self.nodes[next_id] = LayerNode(
                node_id=int(next_id),
                members=tuple(sorted(members)),
                centroid=centroid,
                outcomes=Counter(outcomes),
                support=len(members),
                level=float(value),
                left=int(left),
                right=int(right),
            )
            for other in active:
                if other in (left, right):
                    continue
                other_similarity = min(
                    similarity[(min(left, other), max(left, other))],
                    similarity[(min(right, other), max(right, other))],
                )
                similarity[(min(next_id, other), max(next_id, other))] = (
                    other_similarity
                )
            for other in active:
                if other in (left, right):
                    continue
                similarity.pop((min(left, other), max(left, other)), None)
                similarity.pop((min(right, other), max(right, other)), None)
            similarity.pop(pair, None)
            active = [
                node for node in active if node not in (left, right)
            ]
            active.append(next_id)
            partitions.append((float(value), tuple(sorted(active))))
            next_id += 1
        self.partitions = tuple(partitions)
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

    def _evaluate(self, partition, validation_profiles, validation):
        ids = list(partition)
        centroids = np.vstack([self.nodes[node].centroid for node in ids])
        scores = validation_profiles @ centroids.T
        for column, node_id in enumerate(ids):
            if self.nodes[node_id].support < self.min_layer_support:
                scores[:, column] = -np.inf
        coverage = 0
        margins = []
        gains = []
        for row, sample in enumerate(validation):
            values = scores[row]
            best = int(np.argmax(values))
            best_score = float(values[best])
            if not np.isfinite(best_score):
                continue
            if best_score < self.min_similarity:
                continue
            values = values.copy()
            values[best] = -np.inf
            runner_up = float(np.max(values)) if values.size > 1 else 0.0
            if np.isfinite(runner_up):
                margins.append(best_score - runner_up)
            else:
                margins.append(0.0)
            coverage += 1
            node = self.nodes[ids[best]]
            total = float(sum(node.outcomes.values()))
            base = self._base_probability(sample.outcome)
            count = float(node.outcomes.get(sample.outcome, 0))
            probability = (count + self.prior * base) / (total + self.prior)
            gains.append(float(np.log(probability / base)))
        return dict(
            layers=int(len(ids)),
            coverage=(
                float(coverage) / float(len(validation))
                if validation else 0.0
            ),
            margin=float(np.mean(margins)) if margins else 0.0,
            gain=float(np.mean(gains)) if gains else 0.0,
        )

    def _select(self, validation):
        validation_profiles = np.vstack([
            _normalize(sample.profile) for sample in validation
        ])
        candidates = []
        for level, partition in self.partitions:
            evaluation = self._evaluate(
                partition, validation_profiles, validation,
            )
            evaluation["level"] = float(level)
            evaluation["partition"] = partition
            candidates.append(evaluation)
        self.candidates = len(candidates)
        viable = [
            row for row in candidates
            if row["coverage"] >= self.min_coverage
            and row["gain"] >= self.min_gain
        ]
        self.validation_support = len(validation)
        if not viable:
            self.accepted = False
            return
        best_gain = max(row["gain"] for row in viable)
        near = [
            row for row in viable
            if row["gain"] >= best_gain - self.tolerance
        ]
        chosen = min(
            near,
            key=lambda row: (row["layers"], -row["gain"], -row["level"]),
        )
        self.partition = tuple(chosen["partition"])
        self.layers = tuple(self.nodes[node] for node in self.partition)
        self.level = float(chosen["level"])
        self.coverage = float(chosen["coverage"])
        self.mean_margin = float(chosen["margin"])
        self.predictive_gain = float(chosen["gain"])
        self.accepted = True

    def reselect(self, validation, min_similarity=None, min_coverage=None,
                 min_gain=None, tolerance=None, min_layer_support=None):
        """Re-run frontier selection on an already built tree.

        The tree over training profiles is fixed; only the assignment floor,
        coverage/gain gates, parsimony tolerance or layer-support gate change.
        This lets a stability sweep reuse one tree per fold.
        """
        if not self.nodes:
            raise ValueError("build the tree before reselecting")
        if min_similarity is not None:
            self.min_similarity = float(min_similarity)
        if min_coverage is not None:
            self.min_coverage = float(min_coverage)
        if min_gain is not None:
            self.min_gain = float(min_gain)
        if tolerance is not None:
            self.tolerance = float(tolerance)
        if min_layer_support is not None:
            self.min_layer_support = int(min_layer_support)
        self.accepted = False
        self.coverage = 0.0
        self.mean_margin = 0.0
        self.predictive_gain = 0.0
        self.level = 0.0
        self.partition = ()
        self.layers = ()
        self._select(validation)
        return self

    def assign(self, profile):
        """Best selected layer for a held-out profile, or UNKNOWN."""
        if not self.layers:
            return None, 0.0, 0.0
        normalized = _normalize(profile)
        if not np.any(normalized):
            return None, 0.0, 0.0
        scores = sorted(
            (
                (float(np.dot(normalized, layer.centroid)), layer.node_id)
                for layer in self.layers
                if layer.support >= self.min_layer_support
            ),
            reverse=True,
        )
        if not scores:
            return None, 0.0, 0.0
        best_similarity, best_id = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        if best_similarity < self.min_similarity:
            return None, float(best_similarity), float(
                best_similarity - runner_up
            )
        return int(best_id), float(best_similarity), float(
            best_similarity - runner_up
        )

    def snapshot(self):
        return dict(
            layers=int(len(self.layers)),
            nodes=int(len(self.nodes)),
            candidates=int(self.candidates),
            level=float(self.level),
            min_similarity=float(self.min_similarity),
            min_coverage=float(self.min_coverage),
            min_gain=float(self.min_gain),
            tolerance=float(self.tolerance),
            min_layer_support=int(self.min_layer_support),
            layer_support=sorted(
                (layer.support for layer in self.layers), reverse=True,
            ),
            validation_support=int(self.validation_support),
            coverage=float(self.coverage),
            mean_margin=float(self.mean_margin),
            predictive_gain=float(self.predictive_gain),
            accepted=bool(self.accepted),
        )
