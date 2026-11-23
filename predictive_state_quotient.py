"""Data-driven coarse states from action-conditioned successor behavior.

Fine prototype IDs often change from frame to frame, so literal node identity
is too sparse to compose over long views.  This module forms a temporary
quotient of fine nodes using only realized, delayed-action-aligned
successor/feedback statistics.  The number of groups is a model-selection
candidate, not a domain-defined layer count; held-out transition prediction
must choose whether any merge is useful.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass
class PredictiveStateQuotient:
    fine_to_coarse: Dict[str, str]
    fine_profiles: Dict[str, np.ndarray]
    group_labels: Dict[str, int]
    n_groups: int
    actions: tuple
    profile_mean: np.ndarray
    profile_scale: np.ndarray
    outcome_mean: np.ndarray
    outcome_scale: np.ndarray

    @classmethod
    def fit_rows(cls, rows: Sequence[Mapping], agent, n_groups: int,
                 seed: int = 0, iterations: int = 50):
        """Cluster fine states by observed conditional successor behavior.

        A profile has one block per action.  Each block contains an observed
        support flag, log support, standardized next-observation mean and
        spread, mean feedback channels, and terminal rate.  Unsupported
        actions are explicitly masked rather than assigned a fabricated
        outcome.
        """
        lag = int(agent.lag)
        actions = tuple(int(value) for value in agent.action_space)
        scales = np.asarray(agent.scales, dtype=float)
        obs_dim = int(agent.obs_dim)
        usable = []
        next_observations = []

        for row in rows:
            obs = np.asarray(row.get("obs", ()), dtype=float).reshape(-1)
            nxt = np.asarray(row.get("next_obs", ()), dtype=float).reshape(-1)
            history = np.asarray(row.get("history", ()), dtype=float).reshape(-1)
            if (obs.size != obs_dim or nxt.size != obs_dim
                    or history.size <= lag
                    or not np.isfinite(obs).all()
                    or not np.isfinite(nxt).all()
                    or not np.isfinite(history).all()):
                continue
            pending = history[1:lag + 1][::-1]
            if pending.size != lag:
                continue
            source_state = np.concatenate([obs, pending]) / scales
            source, _distance = agent._nearest(source_state)
            if source is None:
                continue
            action = int(round(float(history[lag])))
            if action not in actions:
                continue
            feedback = np.asarray(
                row.get("signals", (0.0, 0.0)), dtype=float
            ).reshape(-1)
            if feedback.size < 2 or not np.isfinite(feedback[:2]).all():
                continue
            usable.append((
                str(source), action, nxt.copy(), feedback[:2].copy(),
                float(bool(row.get("done", False))),
            ))
            next_observations.append(nxt)

        if not usable:
            return cls(
                fine_to_coarse={}, fine_profiles={}, group_labels={},
                n_groups=0, actions=actions,
                profile_mean=np.zeros(0), profile_scale=np.ones(0),
                outcome_mean=np.zeros(obs_dim),
                outcome_scale=np.ones(obs_dim),
            )

        next_observations = np.asarray(next_observations, dtype=float)
        outcome_mean = next_observations.mean(axis=0)
        outcome_scale = np.where(
            next_observations.std(axis=0) > 1e-8,
            next_observations.std(axis=0), 1.0,
        )

        # Sufficient statistics keyed by (fine state, action).
        stats = {}
        fine_states = sorted(set(item[0] for item in usable))
        for source, action, nxt, feedback, terminal in usable:
            entry = stats.setdefault(
                (source, action),
                dict(n=0, next_sum=np.zeros(obs_dim),
                     next_sq=np.zeros(obs_dim),
                     feedback_sum=np.zeros(2), terminal=0.0),
            )
            standardized = (nxt - outcome_mean) / outcome_scale
            entry["n"] += 1
            entry["next_sum"] += standardized
            entry["next_sq"] += standardized * standardized
            entry["feedback_sum"] += feedback
            entry["terminal"] += terminal

        raw_profiles = {}
        for source in fine_states:
            blocks = []
            for action in actions:
                entry = stats.get((source, action))
                if entry is None or int(entry["n"]) == 0:
                    blocks.extend([
                        0.0, 0.0,
                        *([0.0] * obs_dim),
                        *([0.0] * obs_dim),
                        0.0, 0.0, 0.0,
                    ])
                    continue
                count = int(entry["n"])
                mean = entry["next_sum"] / float(count)
                variance = np.maximum(
                    entry["next_sq"] / float(count) - mean * mean, 0.0,
                )
                feedback_mean = entry["feedback_sum"] / float(count)
                blocks.extend([
                    1.0, float(np.log1p(count)),
                    *mean.tolist(),
                    *np.sqrt(variance).tolist(),
                    *feedback_mean.tolist(),
                    float(entry["terminal"]) / float(count),
                ])
            raw_profiles[source] = np.asarray(blocks, dtype=float)

        profile_matrix = np.vstack([
            raw_profiles[source] for source in fine_states
        ])
        profile_mean = profile_matrix.mean(axis=0)
        profile_scale = np.where(
            profile_matrix.std(axis=0) > 1e-8,
            profile_matrix.std(axis=0), 1.0,
        )
        profiles = (profile_matrix - profile_mean) / profile_scale

        count = len(fine_states)
        k = min(max(int(n_groups), 1), count)
        labels = _kmeans(
            profiles, k, seed=int(seed), iterations=int(iterations),
        )
        unique_labels = sorted(set(int(value) for value in labels))
        relabel = {old: new for new, old in enumerate(unique_labels)}
        labels = np.asarray([relabel[int(value)] for value in labels], dtype=int)
        group_labels = {
            source: int(label)
            for source, label in zip(fine_states, labels)
        }
        mapping = {
            source: "q%d" % int(label)
            for source, label in group_labels.items()
        }
        return cls(
            fine_to_coarse=mapping,
            fine_profiles={
                source: profiles[index].copy()
                for index, source in enumerate(fine_states)
            },
            group_labels=group_labels,
            n_groups=len(set(int(value) for value in labels)),
            actions=actions,
            profile_mean=profile_mean,
            profile_scale=profile_scale,
            outcome_mean=outcome_mean,
            outcome_scale=outcome_scale,
        )

    def map_node(self, node):
        if node is None:
            return None
        return self.fine_to_coarse.get(str(node))

    def summary(self):
        sizes = np.bincount(
            np.asarray(list(self.group_labels.values()), dtype=int),
            minlength=int(self.n_groups),
        ) if self.group_labels else np.zeros(0, dtype=int)
        return dict(
            fine_states=len(self.fine_to_coarse),
            coarse_states=int(self.n_groups),
            sizes=sizes.tolist(),
            actions=list(self.actions),
        )


@dataclass
class PredictiveStateHierarchy:
    """Nested predictive partitions, from exact nodes to a pooled root."""

    fine_to_levels: Dict[str, tuple]
    parents: tuple
    level_sizes: tuple
    level_group_counts: tuple

    @classmethod
    def fit_rows(cls, rows: Sequence[Mapping], agent,
                 group_counts=(16, 4, 1), seed: int = 0):
        """Build nested quotients from realized conditional outcome profiles.

        ``group_counts`` is a diagnostic/model-selection sweep. The returned
        hierarchy is nested (each finer group has exactly one parent); it is
        not a prescribed architecture depth or a planner layer quota.
        """
        base = PredictiveStateQuotient.fit_rows(
            rows, agent, n_groups=1, seed=seed,
        )
        fine_states = sorted(base.fine_profiles)
        if not fine_states:
            return cls({}, (), (), ())
        features = np.vstack([
            base.fine_profiles[state] for state in fine_states
        ])
        requested = sorted(
            set(max(1, int(value)) for value in group_counts), reverse=True
        )
        if not requested or requested[-1] != 1:
            requested.append(1)
        requested = [min(value, len(fine_states)) for value in requested]
        requested = sorted(set(requested), reverse=True)
        if requested[-1] != 1:
            requested.append(1)

        maps = [{state: state for state in fine_states}]
        sizes = [len(fine_states)]
        group_counts_out = [len(fine_states)]
        current_labels = None
        current_names = None
        parents = []
        current_features = features

        for level_index, count in enumerate(requested):
            count = min(max(int(count), 1), len(current_features))
            labels = _canonical_labels(_kmeans(
                current_features, count,
                seed=int(seed) + level_index,
            ))
            names = ["q%d_%d" % (count, int(label)) for label in labels]
            if current_labels is None:
                fine_map = {
                    state: names[index]
                    for index, state in enumerate(fine_states)
                }
                maps.append(fine_map)
            else:
                parent_by_child = {}
                for name, label in zip(current_names, labels):
                    parent_by_child[name] = "q%d_%d" % (count, int(label))
                parents.append(parent_by_child)
                fine_map = {
                    state: parent_by_child[maps[-1][state]]
                    for state in fine_states
                }
                maps.append(fine_map)

            unique = sorted(set(names))
            centers = np.vstack([
                current_features[np.asarray(labels) == int(name.rsplit("_", 1)[1])]
                .mean(axis=0)
                for name in unique
            ])
            current_labels = labels
            current_names = unique
            current_features = centers
            sizes.append(len(unique))
            group_counts_out.append(len(unique))

        # Include parent maps for each abstraction edge. The fine layer is
        # identity and the first learned quotient follows it.
        if len(maps) >= 2 and not parents:
            parents.append({
                state: maps[1][state] for state in fine_states
            })
        else:
            parents.insert(0, {
                state: maps[1][state] for state in fine_states
            })
        return cls(
            fine_to_levels={
                state: tuple(mapping[state] for mapping in maps)
                for state in fine_states
            },
            parents=tuple(parents),
            level_sizes=tuple(sizes),
            level_group_counts=tuple(group_counts_out),
        )

    def summary(self):
        return dict(
            fine_states=len(self.fine_to_levels),
            level_sizes=list(self.level_sizes),
            group_counts=list(self.level_group_counts),
            class_sizes=[dict(values) for values in self.class_sizes_by_level],
        )

    def class_size(self, level, node):
        if int(level) < 0 or int(level) >= len(self.class_sizes_by_level):
            return 0
        return int(self.class_sizes_by_level[int(level)].get(str(node), 0))

    @cached_property
    def class_sizes_by_level(self):
        if not self.fine_to_levels:
            return ()
        count = len(next(iter(self.fine_to_levels.values())))
        levels = [dict() for _ in range(count)]
        for labels in self.fine_to_levels.values():
            for level, label in enumerate(labels):
                key = str(label)
                levels[level][key] = levels[level].get(key, 0) + 1
        return tuple(levels)


def _kmeans(features, k, seed=0, iterations=50):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or not len(features):
        raise ValueError("features must be a non-empty matrix")
    k = min(max(int(k), 1), len(features))
    if k == 1:
        return np.zeros(len(features), dtype=int)
    rng = np.random.RandomState(int(seed))
    centers = features[rng.choice(len(features), size=k, replace=False)].copy()
    labels = np.zeros(len(features), dtype=int)
    for _ in range(max(int(iterations), 1)):
        difference = features[:, None, :] - centers[None, :, :]
        distances = np.einsum("ijk,ijk->ij", difference, difference)
        updated_labels = distances.argmin(axis=1)
        updated = centers.copy()
        for index in range(k):
            members = features[updated_labels == index]
            if len(members):
                updated[index] = members.mean(axis=0)
        if np.array_equal(labels, updated_labels):
            centers = updated
            break
        labels = updated_labels
        centers = updated
    return labels


def _canonical_labels(labels):
    labels = np.asarray(labels, dtype=int)
    unique = sorted(set(int(value) for value in labels))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=int)
