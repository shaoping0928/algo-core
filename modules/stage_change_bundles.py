"""Learned co-change bundles as stage-change boundaries.

A stage change is not one detector channel firing.  After a transition, a set of
feature nodes switches on or off together; when the same set recurs across
episodes, that set is a stage-change bundle.  This module learns those bundles
from a static expansion of the runtime's own feature nodes, emits bundle
boundaries, and returns the intervals between them with a key-node-weighted
strength profile.  Stage typing and layering consume those profiles; they are
not decided here.

The static expansion is deliberately task-agnostic: it is the runtime's feature
mask table over one frame, weighted by a caller-supplied (or learned) per-node
weight.  A bundle is supported only by realized co-changes; an unmatched
transition stays unbundled instead of borrowing another bundle's pattern.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


Token = Tuple[str, int]


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class StaticExpansion:
    """Per-frame feature-node strengths for one episode.

    ``strengths`` is ``(frames, nodes)``; ``weights`` is one non-negative key
    weight per node.  Binary feature masks are the common case, but the type
    does not assume it: any non-negative strength works with ``min_change``.
    """

    node_ids: Tuple[str, ...]
    strengths: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        node_ids = tuple(str(node) for node in self.node_ids)
        strengths = np.asarray(self.strengths, dtype=float)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if strengths.ndim != 2:
            raise ValueError("strengths must be a 2D frame-by-node matrix")
        if strengths.shape[1] != len(node_ids):
            raise ValueError("one strength column is required per node")
        if weights.shape != (len(node_ids),):
            raise ValueError("one weight is required per node")
        if not np.isfinite(strengths).all() or not np.isfinite(weights).all():
            raise ValueError("strengths and weights must be finite")
        if (strengths < 0.0).any() or (weights < 0.0).any():
            raise ValueError("strengths and weights must be non-negative")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "strengths", strengths)
        object.__setattr__(self, "weights", weights)

    @property
    def frames(self):
        return int(self.strengths.shape[0])

    def changes(self, min_change: float = 0.5):
        """Change tokens at each frame after the first.

        A token is ``(node, +1)`` for a rise and ``(node, -1)`` for a fall at
        least ``min_change`` in magnitude.  No token is emitted on a sub-
        threshold move.
        """
        min_change = float(min_change)
        if min_change <= 0.0:
            raise ValueError("min_change must be positive")
        if self.frames < 2:
            return tuple(() for _ in range(self.frames))
        delta = np.diff(self.strengths, axis=0)
        tokens = []
        for row in delta:
            current = set()
            for index, value in enumerate(row):
                if value >= min_change:
                    current.add((self.node_ids[index], 1))
                elif value <= -min_change:
                    current.add((self.node_ids[index], -1))
            tokens.append(tuple(sorted(current, key=repr)))
        return tuple(tokens)

    def weighted_profile(self, start: int = 0,
                         end: Optional[int] = None):
        end = self.frames if end is None else int(end)
        block = self.strengths[int(start):end]
        if not len(block):
            return tuple(0.0 for _ in self.node_ids)
        return tuple(float(value) for value in (
            block * self.weights
        ).mean(axis=0))


def static_expansion(agent, observations, weights=None) -> StaticExpansion:
    """Build a static expansion from the runtime's own feature masks.

    ``observations`` must be in the same encoded space the runtime's feature
    rules were fit on (scaled observation columns).  Nothing task-specific is
    read; the nodes are the runtime's feature bank.
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2:
        raise ValueError("observations must be a 2D frame-by-feature matrix")
    node_ids = tuple(str(node) for node in getattr(agent, "_feature_all", ()))
    if not node_ids:
        raise ValueError("the runtime has no feature nodes")
    table = getattr(agent, "_feature_mask_table", None)
    if table is None:
        raise ValueError("the runtime cannot produce a feature mask table")
    masks = [np.asarray(mask, dtype=float).reshape(-1) for mask in table(
        observations,
    )]
    if len(masks) != len(node_ids):
        raise ValueError("feature mask table does not match the node bank")
    if not masks:
        return StaticExpansion(node_ids, np.zeros((len(observations), 0)),
                               np.ones(0))
    lengths = {len(mask) for mask in masks}
    if lengths != {len(observations)}:
        raise ValueError("every feature mask must cover every frame")
    matrix = np.stack(masks, axis=1)
    if weights is None:
        weight_vector = np.ones(len(node_ids), dtype=float)
    elif isinstance(weights, Mapping):
        weight_vector = np.asarray(
            [float(weights.get(node, 1.0)) for node in node_ids],
            dtype=float,
        )
    else:
        weight_vector = np.asarray(weights, dtype=float).reshape(-1)
    return StaticExpansion(node_ids, matrix, weight_vector)


def static_expansion_from_snapshots(node_ids: Sequence[str], snapshots,
                                    weights=None) -> StaticExpansion:
    """Build a binary expansion from per-transition active-feature sets."""
    node_ids = tuple(str(node) for node in node_ids)
    index = {node: position for position, node in enumerate(node_ids)}
    rows = []
    for snapshot in snapshots:
        row = np.zeros(len(node_ids), dtype=float)
        for node in snapshot:
            position = index.get(str(node))
            if position is not None:
                row[position] = 1.0
        rows.append(row)
    matrix = (
        np.vstack(rows) if rows else np.zeros((0, len(node_ids)))
    )
    if weights is None:
        weight_vector = np.ones(len(node_ids), dtype=float)
    elif isinstance(weights, Mapping):
        weight_vector = np.asarray(
            [float(weights.get(node, 1.0)) for node in node_ids],
            dtype=float,
        )
    else:
        weight_vector = np.asarray(weights, dtype=float).reshape(-1)
    return StaticExpansion(node_ids, matrix, weight_vector)


@dataclass(frozen=True)
class ChangeBundle:
    """A learned set of feature-node changes that recur together."""

    tokens: Tuple[Token, ...]
    episode_support: int
    transition_support: int
    lift: float
    key_score: float

    @property
    def size(self):
        return len(self.tokens)


@dataclass(frozen=True)
class StageBoundary:
    frame: int
    bundle: ChangeBundle
    match: float
    score: float


@dataclass(frozen=True)
class StageInterval:
    start: int
    end: int
    entry_bundle: Optional[ChangeBundle]
    profile: Tuple[float, ...]
    changes: Tuple[Tuple[Token, ...], ...]

    @property
    def duration(self):
        return int(self.end) - int(self.start)


class StageChangeBundles:
    """Frequent co-change itemsets over static expansions."""

    def __init__(self, bundles=(), key_weights=None,
                 min_support: int = 2, min_bundle_size: int = 2,
                 min_change: float = 0.5, max_bundle_size: int = 3,
                 min_lift: float = 0.5, max_candidates: int = 200000,
                 allow_truncation: bool = False):
        if int(min_support) < 1:
            raise ValueError("min_support must be positive")
        if int(min_bundle_size) < 2:
            raise ValueError("min_bundle_size must be at least two")
        if float(min_change) <= 0.0:
            raise ValueError("min_change must be positive")
        if int(max_bundle_size) < int(min_bundle_size):
            raise ValueError("max_bundle_size must cover min_bundle_size")
        if not 0.0 <= float(min_lift) <= 1.0:
            raise ValueError("min_lift must be in [0, 1]")
        if int(max_candidates) < 1:
            raise ValueError("max_candidates must be positive")
        self.bundles = tuple(bundles)
        self.key_weights = dict(key_weights or {})
        self.min_support = int(min_support)
        self.min_bundle_size = int(min_bundle_size)
        self.min_change = float(min_change)
        self.max_bundle_size = int(max_bundle_size)
        self.min_lift = float(min_lift)
        self.max_candidates = int(max_candidates)
        self.allow_truncation = bool(allow_truncation)
        self.candidates = 0
        self.truncated = False

    @classmethod
    def fit(cls, expansions: Iterable[StaticExpansion], **kwargs):
        model = cls(**kwargs)
        transactions = [
            [set(tokens) for tokens in expansion.changes(model.min_change)]
            for expansion in expansions
        ]
        transactions = [episode for episode in transactions if episode]
        if not transactions:
            return model
        model._fit(transactions)
        return model

    def _fit(self, transactions):
        episode_count = len(transactions)
        token_episodes = defaultdict(int)
        token_transitions = defaultdict(int)
        node_episodes = defaultdict(set)
        for index, episode in enumerate(transactions):
            seen = set()
            for current in episode:
                for token in current:
                    token_transitions[token] += 1
                    seen.add(token)
                    node_episodes[token[0]].add(index)
            for token in seen:
                token_episodes[token] += 1
        self.key_weights = {
            node: 1.0 / (1.0 + len(episodes) / float(episode_count))
            for node, episodes in node_episodes.items()
        }
        maximum = max(self.key_weights.values(), default=1.0)
        if maximum > 0.0:
            self.key_weights = {
                node: weight / maximum
                for node, weight in self.key_weights.items()
            }

        frequent = {
            frozenset([token]) for token, count in token_episodes.items()
            if count >= self.min_support
        }
        bundles = []
        current_level = frequent
        for size in range(2, self.max_bundle_size + 1):
            candidates = set()
            ordered = sorted(current_level, key=lambda item: repr(sorted(
                item, key=repr,
            )))
            for left, right in combinations(ordered, 2):
                union = left | right
                if len(union) != size:
                    continue
                if not all(
                    frozenset(pair) in current_level
                    for pair in combinations(union, size - 1)
                ):
                    continue
                candidates.add(union)
                self.candidates += 1
                if self.candidates > self.max_candidates:
                    if not self.allow_truncation:
                        raise RuntimeError(
                            "co-change candidate count exceeded "
                            "max_candidates=%d" % self.max_candidates
                        )
                    self.truncated = True
                    break
            if self.truncated:
                break
            next_level = set()
            for candidate in sorted(candidates, key=repr):
                support = self._episode_support(candidate, transactions)
                if support < self.min_support:
                    continue
                next_level.add(candidate)
                token_list = tuple(sorted(candidate, key=repr))
                rarest = min(
                    token_episodes[token] for token in token_list
                )
                lift = (
                    float(support) / float(rarest) if rarest else 0.0
                )
                if lift < self.min_lift:
                    continue
                transition_support = self._transition_support(
                    candidate, transactions,
                )
                key_score = float(np.mean([
                    self.key_weights.get(node, 1.0)
                    for node, _sign in token_list
                ]))
                bundles.append(ChangeBundle(
                    tokens=token_list,
                    episode_support=int(support),
                    transition_support=int(transition_support),
                    lift=float(lift),
                    key_score=float(key_score),
                ))
            current_level = next_level
            if not current_level:
                break
        bundles.sort(key=lambda bundle: (
            -bundle.size, -bundle.key_score, -bundle.episode_support,
            repr(bundle.tokens),
        ))
        self.bundles = tuple(bundles)

    @staticmethod
    def _episode_support(tokens, transactions):
        return sum(
            1 for episode in transactions
            if any(tokens <= current for current in episode)
        )

    @staticmethod
    def _transition_support(tokens, transactions):
        return sum(
            1 for episode in transactions
            for current in episode
            if tokens <= current
        )

    def boundaries(self, expansion: StaticExpansion,
                   min_match: float = 1.0):
        """Frames where a learned bundle matches the realized changes."""
        if not 0.0 < float(min_match) <= 1.0:
            raise ValueError("min_match must be in (0, 1]")
        if not self.bundles:
            return tuple()
        changes = expansion.changes(self.min_change)
        result = []
        for frame, tokens in enumerate(changes):
            current = set(tokens)
            if not current:
                continue
            best = None
            for bundle in self.bundles:
                matched = len(current & set(bundle.tokens))
                match = matched / float(bundle.size)
                if match < float(min_match):
                    continue
                score = bundle.key_score * match
                if best is None or score > best[2]:
                    best = (bundle, match, score)
            if best is None:
                continue
            result.append(StageBoundary(
                frame=int(frame + 1),
                bundle=best[0],
                match=float(best[1]),
                score=float(best[2]),
            ))
        return tuple(result)

    def segments(self, expansion: StaticExpansion,
                 min_match: float = 1.0):
        """Split an episode at bundle boundaries into profiled intervals."""
        boundaries = self.boundaries(expansion, min_match=min_match)
        cuts = [0] + [boundary.frame for boundary in boundaries]
        cuts.append(expansion.frames)
        cuts = sorted(set(cuts))
        intervals = []
        changes = expansion.changes(self.min_change)
        for index in range(len(cuts) - 1):
            start, end = cuts[index], cuts[index + 1]
            if end <= start:
                continue
            entry = None
            if start > 0 and start - 1 < len(boundaries):
                entry = boundaries[start - 1].bundle
            intervals.append(StageInterval(
                start=int(start),
                end=int(end),
                entry_bundle=entry,
                profile=expansion.weighted_profile(start, end),
                changes=tuple(changes[start:end]),
            ))
        return tuple(intervals)

    def snapshot(self):
        return dict(
            bundles=int(len(self.bundles)),
            min_support=int(self.min_support),
            min_bundle_size=int(self.min_bundle_size),
            max_bundle_size=int(self.max_bundle_size),
            min_change=float(self.min_change),
            min_lift=float(self.min_lift),
            candidates=int(self.candidates),
            truncated=bool(self.truncated),
            key_nodes=int(len(self.key_weights)),
            bundle_sizes=sorted(
                (bundle.size for bundle in self.bundles), reverse=True,
            ),
            bundle_support=sorted(
                (bundle.episode_support for bundle in self.bundles),
                reverse=True,
            ),
        )
