"""Stage-type value readout.

Each stage type aggregates the realized remaining net feedback and terminal rate
of its member stages.  The readout maps a key-weighted integrated profile to a
type value, and maps predictive-tree roots to a normalized penalty.  A planner
can then add a soft endpoint term without recomputing ``E(s)`` for every
candidate.

Like the type clusters, the value model is episode-held-out: types and values
are fit on training episodes, and the cluster gate uses validation episodes.
Unknown profiles have no value and receive a neutral penalty; nothing is
fabricated for them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from stage_type_clusters import StageSample, StageTypeClusters


@dataclass
class StageTypeValue:
    clusters: StageTypeClusters
    values: dict
    normalized: dict
    terminal_rates: dict
    supports: dict
    value_min: float
    value_max: float

    @classmethod
    def fit(cls, samples: Sequence[StageSample],
            validation: Optional[Sequence[StageSample]] = None, **kwargs):
        clusters = StageTypeClusters.fit(samples, validation, **kwargs)
        buckets = defaultdict(list)
        terminals = defaultdict(list)
        for index, sample in enumerate(samples):
            type_id = clusters.sample_types.get(index)
            if type_id is None:
                continue
            buckets[type_id].append(float(sample.value))
            terminals[type_id].append(bool(sample.terminal))
        values = {
            type_id: float(np.mean(bucket))
            for type_id, bucket in buckets.items() if bucket
        }
        value_min = min(values.values(), default=0.0)
        value_max = max(values.values(), default=0.0)
        span = value_max - value_min
        normalized = {
            type_id: (
                (value - value_min) / span if span > 1e-12 else 0.5
            )
            for type_id, value in values.items()
        }
        return cls(
            clusters=clusters,
            values=values,
            normalized=normalized,
            terminal_rates={
                type_id: float(np.mean(bucket))
                for type_id, bucket in terminals.items() if bucket
            },
            supports={
                type_id: int(len(bucket))
                for type_id, bucket in buckets.items()
            },
            value_min=float(value_min),
            value_max=float(value_max),
        )

    def value(self, profile):
        """Return ``(type_id, value, similarity)`` or ``(None, None, sim)``."""
        type_id, similarity, _margin = self.clusters.assign(profile)
        if type_id is None:
            return None, None, float(similarity)
        return int(type_id), float(self.values[type_id]), float(similarity)

    def penalty(self, profile, neutral: float = 0.5):
        """Higher is worse; unknown profiles get a neutral penalty."""
        type_id, _value, _similarity = self.value(profile)
        if type_id is None:
            return float(neutral)
        return float(1.0 - self.normalized[type_id])

    def root_penalties(self, tree, neutral: float = 0.5):
        """Map predictive-tree root tokens to a normalized value penalty."""
        buckets = defaultdict(list)
        for index, sample in enumerate(self.clusters.samples):
            type_id = self.clusters.sample_types.get(index)
            if type_id is None or not sample.key:
                continue
            chain = tree.chain(sample.key)
            token = (
                ("tree", "base") if not chain
                else ("tree", str(chain[-1]))
            )
            buckets[token].append(float(self.normalized[type_id]))
        result = {
            token: float(1.0 - np.mean(values))
            for token, values in buckets.items() if values
        }
        result[("tree", "base")] = float(neutral)
        return result

    def snapshot(self):
        return dict(
            types=int(len(self.values)),
            value_min=float(self.value_min),
            value_max=float(self.value_max),
            support=sorted(
                (int(value) for value in self.supports.values()),
                reverse=True,
            ),
            terminal_rates=sorted(
                (round(float(value), 6)
                 for value in self.terminal_rates.values()),
                reverse=True,
            ),
            clusters=self.clusters.snapshot(),
        )
