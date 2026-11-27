"""Cross-fold stability selection for stage-type layers.

The held-out frontier gate validates that a layer set is reachable and useful,
but every parsimony tolerance passes it, so the layer count still needs a
stability criterion.  This module fits one merge tree per episode fold, re-runs
frontier selection for each candidate (similarity floor, parsimony tolerance)
on that fold's held-out episodes, and keeps the candidate whose selected layer
count is stable across folds while its held-out gain stays competitive.

Nothing here changes a decision; it only chooses the layer configuration that
the read-only stage layer should report.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from stage_type_layers import StageTypeLayers


@dataclass(frozen=True)
class StabilityCandidate:
    min_similarity: float
    tolerance: float
    layers: Tuple[int, ...]
    coverage: Tuple[float, ...]
    gain: Tuple[float, ...]
    accepted: Tuple[bool, ...]

    @property
    def mode_layers(self):
        if not self.layers:
            return 0
        return int(Counter(self.layers).most_common(1)[0][0])

    @property
    def agreement(self):
        if not self.layers:
            return 0.0
        return float(Counter(self.layers)[self.mode_layers]) / float(
            len(self.layers)
        )

    @property
    def mean_layers(self):
        return float(np.mean(self.layers)) if self.layers else 0.0

    @property
    def mean_coverage(self):
        return float(np.mean(self.coverage)) if self.coverage else 0.0

    @property
    def mean_gain(self):
        return float(np.mean(self.gain)) if self.gain else 0.0

    @property
    def acceptance(self):
        return float(np.mean(self.accepted)) if self.accepted else 0.0

    def snapshot(self):
        return dict(
            min_similarity=float(self.min_similarity),
            tolerance=float(self.tolerance),
            folds=int(len(self.layers)),
            layers=[int(value) for value in self.layers],
            mode_layers=int(self.mode_layers),
            agreement=float(self.agreement),
            mean_layers=float(self.mean_layers),
            mean_coverage=float(self.mean_coverage),
            mean_gain=float(self.mean_gain),
            acceptance=float(self.acceptance),
        )


def episode_folds(samples, folds: int = 5, seed: int = 0):
    """Yield train/validation splits grouped by episode."""
    if int(folds) < 2:
        raise ValueError("folds must be at least two")
    episodes = sorted({sample.episode for sample in samples}, key=repr)
    if len(episodes) < 2:
        return
    rng = np.random.RandomState(int(seed))
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    blocks = np.array_split(np.asarray(shuffled, dtype=object), int(folds))
    for block in blocks:
        held_out = set(block.tolist())
        train = [
            sample for sample in samples if sample.episode not in held_out
        ]
        validation = [
            sample for sample in samples if sample.episode in held_out
        ]
        if train and validation:
            yield train, validation


class LayerStabilitySelector:
    """Choose similarity/tolerance by cross-fold layer-count stability."""

    def __init__(self, min_agreement: float = 0.6,
                 min_coverage: float = 0.5, min_gain: float = 0.01,
                 min_acceptance: float = 0.8,
                 min_type_support: int = 2, prior: float = 4.0):
        if not 0.0 <= float(min_agreement) <= 1.0:
            raise ValueError("min_agreement must be in [0, 1]")
        if not 0.0 <= float(min_coverage) <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")
        if float(min_gain) < 0.0:
            raise ValueError("min_gain must be non-negative")
        if not 0.0 <= float(min_acceptance) <= 1.0:
            raise ValueError("min_acceptance must be in [0, 1]")
        if int(min_type_support) < 1:
            raise ValueError("min_type_support must be positive")
        if float(prior) <= 0.0:
            raise ValueError("prior must be positive")
        self.min_agreement = float(min_agreement)
        self.min_coverage = float(min_coverage)
        self.min_gain = float(min_gain)
        self.min_acceptance = float(min_acceptance)
        self.min_type_support = int(min_type_support)
        self.prior = float(prior)
        self.candidates = ()
        self.chosen: Optional[StabilityCandidate] = None
        self.accepted = False
        self.folds = 0

    @classmethod
    def fit(cls, samples, folds: int = 5, seed: int = 0,
            min_similarities: Sequence[float] = (0.5, 0.7, 0.9),
            tolerances: Sequence[float] = (0.05, 0.1, 0.2), **kwargs):
        selector = cls(**kwargs)
        records = {
            (float(similarity), float(tolerance)): dict(
                layers=[], coverage=[], gain=[], accepted=[],
            )
            for similarity in min_similarities
            for tolerance in tolerances
        }
        for train, validation in episode_folds(samples, folds, seed):
            selector.folds += 1
            model = StageTypeLayers(
                min_coverage=selector.min_coverage,
                min_gain=selector.min_gain,
                min_layer_support=selector.min_type_support,
                prior=selector.prior,
            ).build(train)
            if not model.nodes:
                continue
            for (similarity, tolerance), record in records.items():
                model.reselect(
                    validation, min_similarity=similarity,
                    tolerance=tolerance,
                    min_coverage=selector.min_coverage,
                    min_gain=selector.min_gain,
                    min_layer_support=selector.min_type_support,
                )
                record["layers"].append(len(model.layers))
                record["coverage"].append(float(model.coverage))
                record["gain"].append(float(model.predictive_gain))
                record["accepted"].append(bool(model.accepted))
        candidates = [
            StabilityCandidate(
                min_similarity=similarity,
                tolerance=tolerance,
                layers=tuple(record["layers"]),
                coverage=tuple(record["coverage"]),
                gain=tuple(record["gain"]),
                accepted=tuple(record["accepted"]),
            )
            for (similarity, tolerance), record in sorted(records.items())
            if record["layers"]
        ]
        selector.candidates = tuple(candidates)
        viable = [
            candidate for candidate in candidates
            if candidate.agreement >= selector.min_agreement
            and candidate.mean_coverage >= selector.min_coverage
            and candidate.mean_gain >= selector.min_gain
            and candidate.acceptance >= selector.min_acceptance
        ]
        if viable:
            selector.chosen = max(
                viable,
                key=lambda candidate: (
                    candidate.mean_gain, candidate.agreement,
                    -candidate.mean_layers,
                ),
            )
            selector.accepted = True
        return selector

    def snapshot(self):
        return dict(
            folds=int(self.folds),
            candidates=int(len(self.candidates)),
            chosen=(
                None if self.chosen is None else self.chosen.snapshot()
            ),
            accepted=bool(self.accepted),
            min_agreement=float(self.min_agreement),
            min_coverage=float(self.min_coverage),
            min_gain=float(self.min_gain),
            min_acceptance=float(self.min_acceptance),
        )
