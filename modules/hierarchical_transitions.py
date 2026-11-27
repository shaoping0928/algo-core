"""Support-aware transition prediction over nested predictive-state quotients."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from multiview_transitions import MultiViewTransitionModel, UNKNOWN
from predictive_state_quotient import PredictiveStateHierarchy


class HierarchicalTransitionModel:
    """A nested set of action-conditioned kernels with statistical backoff.

    A rollout starts at the finest learned state. For each branch and acting
    action, it uses the finest supported kernel; an unsupported edge backs off
    to that state's parent quotient. If the root has no observed edge, that
    branch becomes UNKNOWN. The branch remains at the abstraction level that
    produced its successor, avoiding a fabricated map back to a fine node.
    """

    def __init__(self, hierarchy, models):
        self.hierarchy = hierarchy
        self.models = list(models)

    @classmethod
    def fit_rows(cls, rows, agent, group_counts=(16, 4, 1), seed=0,
                 lag=None):
        lag = int(agent.lag if lag is None else lag)
        hierarchy = PredictiveStateHierarchy.fit_rows(
            rows, agent, group_counts=group_counts, seed=seed,
        )
        maps = []
        # Level zero is the exact prototype state. Each later level is a
        # nested predictive quotient trained from the same realized rows.
        maps.append(None)
        for level in range(1, len(hierarchy.level_sizes)):
            maps.append({
                fine: labels[level]
                for fine, labels in hierarchy.fine_to_levels.items()
            })
        models = [
            MultiViewTransitionModel.fit_rows(
                rows, agent, lag=lag, node_map=node_map,
            )
            for node_map in maps
        ]
        return cls(hierarchy, models)

    def predict_views(self, agent, observation: Sequence[float],
                      pending: Sequence[int], issued_actions: Sequence[int],
                      windows: Iterable[int], target_nodes=None):
        requested = sorted(set(int(value) for value in windows))
        if any(value < 0 for value in requested):
            raise ValueError("windows must be non-negative")
        actions = [int(value) for value in issued_actions]
        if requested and max(requested) > len(actions):
            raise ValueError("issued_actions must cover the largest window")
        queue = [int(value) for value in pending]
        if len(queue) != self.models[0].lag:
            raise ValueError("pending must contain exactly lag actions")

        obs = np.asarray(observation, dtype=float).reshape(-1)
        state = np.concatenate([
            obs, np.asarray(queue, dtype=float),
        ]) / np.asarray(agent.scales, dtype=float)
        source, distance = agent._nearest(state)
        source = None if source is None else str(source)
        if (source is None
                or source not in self.hierarchy.fine_to_levels
                or float(distance) > self.models[0].state_radius):
            return {
                window: self._unknown_snapshot()
                for window in requested
            }

        target_sets = None
        if target_nodes is not None:
            target_sets = [set() for _ in self.models]
            for fine_node in target_nodes:
                labels = self.hierarchy.fine_to_levels.get(str(fine_node))
                if labels is None:
                    continue
                for level, label in enumerate(labels):
                    target_sets[level].add(str(label))
        source_labels = self.hierarchy.fine_to_levels[source]
        mass = {(0, str(source_labels[0])): 1.0}
        terminal_mass = 0.0
        target_hit = 0.0
        unknown_mass = 0.0
        dopamine = 0.0
        pain = 0.0
        supported_rows = 0.0
        level_usage = np.zeros(len(self.models), dtype=float)
        queue_now = list(queue)
        snapshots = {}

        if 0 in requested:
            if target_sets is not None and source_labels[0] in target_sets[0]:
                target_hit = 1.0
                mass = {}
            snapshots[0] = self._snapshot(
                mass, terminal_mass, target_hit, unknown_mass,
                dopamine, pain, supported_rows, level_usage,
            )

        for step, issued in enumerate(actions, start=1):
            acting = queue_now[0] if self.models[0].lag else int(issued)
            next_mass = defaultdict(float)
            for (level, node), probability in mass.items():
                query_level = int(level)
                query_node = str(node)
                transition = None
                while query_level < len(self.models):
                    transition = self.models[query_level].transition(
                        query_node, acting,
                    )
                    if transition is not None:
                        break
                    if query_level >= len(self.hierarchy.parents):
                        break
                    parent = self.hierarchy.parents[query_level].get(
                        query_node
                    )
                    if parent is None:
                        break
                    query_node = str(parent)
                    query_level += 1
                if transition is None:
                    unknown_mass += probability
                    continue

                distribution, support = transition
                supported_rows += probability * float(support)
                level_usage[query_level] += probability
                edge_map = self.models[query_level].edges.get(
                    (query_node, acting), {}
                )
                for target, edge_probability in distribution.items():
                    flow = probability * float(edge_probability)
                    measure = edge_map[target]
                    dopamine += flow * measure.dopamine_mean
                    pain += flow * measure.pain_mean
                    terminal_flow = flow * measure.terminal_rate
                    terminal_mass += terminal_flow
                    if (target_sets is not None
                            and str(target) in target_sets[query_level]):
                        target_hit += flow
                    live_flow = flow - terminal_flow
                    if live_flow > 0.0:
                        if (target_sets is not None
                                and str(target) in target_sets[query_level]):
                            continue
                        next_mass[(query_level, str(target))] += live_flow
            mass = dict(next_mass)
            if self.models[0].lag:
                queue_now = queue_now[1:] + [int(issued)]
            if step in requested:
                snapshots[step] = self._snapshot(
                    mass, terminal_mass, target_hit, unknown_mass,
                    dopamine, pain, supported_rows, level_usage,
                )
        return snapshots

    def _snapshot(self, mass, terminal, target_hit, unknown, dopamine, pain,
                  supported_rows, level_usage):
        total = float(sum(mass.values()))
        resolution = (
            float(sum(
                float(probability)
                * self.hierarchy.class_size(level, node)
                for (level, node), probability in mass.items()
            ) / total)
            if total > 0.0 else None
        )
        return dict(
            distribution=dict(mass),
            terminal=float(terminal),
            target_hit=float(target_hit),
            unknown=float(unknown),
            dopamine=float(dopamine),
            pain=float(pain),
            supported_rows=float(supported_rows),
            level_usage=np.asarray(level_usage, dtype=float).copy(),
            resolution=resolution,
            status="supported" if unknown <= 1e-12 else UNKNOWN,
        )

    @staticmethod
    def _unknown_snapshot():
        return dict(
            distribution={}, terminal=0.0, target_hit=0.0, unknown=1.0,
            dopamine=0.0, pain=0.0, supported_rows=0.0,
            level_usage=np.zeros(0), resolution=None,
            status=UNKNOWN,
        )

    def summary(self):
        return dict(
            hierarchy=self.hierarchy.summary(),
            levels=[model.summary() for model in self.models],
        )
