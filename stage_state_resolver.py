"""Resolve a continuous current state to a realized stage type, or UNKNOWN.

The exact key-signal token identifies a stage only at the moment a boundary is
observed.  Between boundaries the same token covers very different continuous
states, and a state can enter a stage whose boundary pulse is long past.  This
resolver learns the empirical association between the runtime's own prototype
of a state and the stage keys that actually contained that state in training.

Fit and query are separated: the association table and the state radius come
from training episodes only, and querying a held-out state never updates them.
An unseen prototype, too little episode support, an ambiguous distribution, or
a state farther from its prototype than the runtime's own birth threshold all
return UNKNOWN.  UNKNOWN is not a safe stage and must not be treated as one.

This layer uses no task geometry, stage labels, goals, or hand-written rules.
Predictive-equivalence merging of stage types is a later layer; here a key is
still an exact signal signature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from stage_experience import StageExperienceBatch


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


@dataclass
class _StageCounts:
    count: int = 0
    episodes: set = field(default_factory=set)

    def add(self, episode):
        self.count += 1
        self.episodes.add(episode)


class StageStateResolver:
    """Prototype-conditioned empirical stage resolution with an UNKNOWN gate."""

    def __init__(self, lag: int, radius: float = 0.0, min_support: int = 2,
                 min_episodes: int = 2, min_probability: float = 0.5,
                 min_margin: float = 0.1, radius_quantile: float = 0.99):
        if int(lag) < 0:
            raise ValueError("lag must be non-negative")
        if float(radius) < 0.0 or not np.isfinite(radius):
            raise ValueError("radius must be finite and non-negative")
        if int(min_support) < 1 or int(min_episodes) < 1:
            raise ValueError("support thresholds must be positive")
        if not 0.0 <= float(min_probability) <= 1.0:
            raise ValueError("min_probability must be in [0, 1]")
        if not 0.0 <= float(min_margin) <= 1.0:
            raise ValueError("min_margin must be in [0, 1]")
        if not 0.5 <= float(radius_quantile) <= 1.0:
            raise ValueError("radius_quantile must be in [0.5, 1]")
        self.lag = int(lag)
        self.radius = float(radius)
        self.min_support = int(min_support)
        self.min_episodes = int(min_episodes)
        self.min_probability = float(min_probability)
        self.min_margin = float(min_margin)
        self.radius_quantile = float(radius_quantile)
        self._table = {}
        self.training_rows = 0
        self.stage_types = set()

    @classmethod
    def fit_rows(cls, rows: Sequence[Mapping], agent,
                 experience: Optional[StageExperienceBatch] = None,
                 min_support: int = 2, min_episodes: int = 2,
                 min_probability: float = 0.5, min_margin: float = 0.1,
                 radius_quantile: float = 0.99):
        """Fit the prototype -> stage-type table from realized transitions.

        When ``experience`` is omitted it is fit on the same rows; pass a
        frozen training batch when the rows are held out so the detector and
        the association table come from the training fold only.
        """
        lag = int(agent.lag)
        if experience is None:
            experience = StageExperienceBatch.fit(rows, agent, lag=lag)
        if int(experience.lag) != lag:
            raise ValueError("experience lag does not match the agent")
        resolver = cls(
            lag=lag, min_support=min_support,
            min_episodes=min_episodes,
            min_probability=min_probability,
            min_margin=min_margin,
            radius_quantile=radius_quantile,
        )
        distances = []
        for transition in experience.stage_transitions(rows, agent):
            state = cls._encode(agent, transition.observation,
                                transition.pending, lag)
            prototype, distance = agent._nearest(state)
            if prototype is None or not np.isfinite(distance):
                continue
            key = tuple(_native(token) for token in transition.stage.key)
            counts = resolver._table.setdefault(
                str(prototype), {}
            ).setdefault(key, _StageCounts())
            counts.add(transition.episode)
            resolver.stage_types.add(key)
            resolver.training_rows += 1
            distances.append(float(distance))
        quantile = (
            float(np.quantile(distances, radius_quantile))
            if distances else 0.0
        )
        # The runtime itself would open a new concept beyond this distance,
        # so an old stage must not be claimed for such a state.
        birth = float(getattr(agent, "birth_threshold", 0.0) or 0.0)
        resolver.radius = max(quantile, birth)
        return resolver

    @staticmethod
    def _encode(agent, observation, pending, lag: int) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        queue = np.asarray(pending, dtype=float).reshape(-1)
        if queue.size != int(lag):
            raise ValueError("pending must contain exactly lag actions")
        if not np.isfinite(obs).all() or not np.isfinite(queue).all():
            raise ValueError("state values must be finite")
        state = np.concatenate([obs, queue])
        scales = np.asarray(agent.scales, dtype=float)
        if state.size != scales.size:
            raise ValueError(
                "state dimension does not match the agent's scales"
            )
        return state / scales

    def resolve(self, agent, observation, pending):
        """Resolve one state to a supported stage key or UNKNOWN."""
        state = self._encode(agent, observation, pending, self.lag)
        prototype, distance = agent._nearest(state)
        if prototype is None or not np.isfinite(distance):
            return self._unknown(reason="no_prototype")
        prototype = str(prototype)
        distance = float(distance)
        if distance > self.radius:
            return self._unknown(
                prototype=prototype, distance=distance,
                reason="outside_radius",
            )
        counts = self._table.get(prototype)
        if not counts:
            return self._unknown(
                prototype=prototype, distance=distance,
                reason="unseen_prototype",
            )
        total = sum(item.count for item in counts.values())
        episodes = set()
        for item in counts.values():
            episodes.update(item.episodes)
        if (total < self.min_support
                or len(episodes) < self.min_episodes):
            return self._unknown(
                prototype=prototype, distance=distance,
                support=total, episode_support=len(episodes),
                reason="insufficient_support",
            )
        ranked = sorted(
            counts.items(),
            key=lambda item: (-item[1].count, repr(item[0])),
        )
        key, top = ranked[0]
        probability = float(top.count) / float(total)
        runner_up = (
            float(ranked[1][1].count) / float(total)
            if len(ranked) > 1 else 0.0
        )
        margin = probability - runner_up
        if (probability < self.min_probability
                or margin < self.min_margin):
            return self._unknown(
                prototype=prototype, distance=distance,
                support=total, episode_support=len(episodes),
                probability=probability, margin=margin,
                distribution=tuple(
                    (candidate, float(item.count) / float(total))
                    for candidate, item in ranked
                ),
                reason="ambiguous",
            )
        return dict(
            status="supported",
            key=key,
            probability=probability,
            runner_up_probability=runner_up,
            margin=margin,
            support=int(total),
            episode_support=int(len(episodes)),
            prototype=prototype,
            distance=distance,
            distribution=tuple(
                (candidate, float(item.count) / float(total))
                for candidate, item in ranked
            ),
            reason=None,
            provenance="realized_state_stage_cooccurrence",
            affects_value=False,
        )

    def resolve_experience(self, rows, agent,
                           experience: StageExperienceBatch):
        """Resolve every realized transition in ``rows`` (audit helper)."""
        return tuple(
            self.resolve(agent, transition.observation, transition.pending)
            for transition in experience.stage_transitions(rows, agent)
        )

    @staticmethod
    def _unknown(prototype=None, distance=None, support=0,
                 episode_support=0, probability=None, margin=None,
                 distribution=(), reason=None):
        return dict(
            status=UNKNOWN,
            key=None,
            probability=probability,
            runner_up_probability=None,
            margin=margin,
            support=int(support),
            episode_support=int(episode_support),
            prototype=prototype,
            distance=distance,
            distribution=tuple(distribution),
            reason=reason,
            provenance="realized_state_stage_cooccurrence",
            affects_value=False,
        )

    def snapshot(self):
        return dict(
            prototypes=len(self._table),
            training_rows=int(self.training_rows),
            stage_types=len(self.stage_types),
            radius=float(self.radius),
            radius_quantile=float(self.radius_quantile),
            min_support=int(self.min_support),
            min_episodes=int(self.min_episodes),
            min_probability=float(self.min_probability),
            min_margin=float(self.min_margin),
        )
