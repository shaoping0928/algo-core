"""Action-aligned empirical transitions with arbitrary-window queries.

The model is a measure over transitions the system actually executed:

    (source prototype, action currently acting) -> (next prototype, feedback)

It does not create counterfactual samples.  Actions are aligned to the
transition they caused (the oldest pending action, ``history[lag]``), not the
action just issued.  A prediction composes this same one-step kernel for an
arbitrary set of frame windows; separate span-specific models are not needed.

Unsupported source/action pairs and states outside the observed prototype
support produce UNKNOWN mass.  UNKNOWN is never reinterpreted as a safe
transition.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


UNKNOWN = "unknown"


@dataclass
class EdgeMeasure:
    count: int = 0
    pain_sum: float = 0.0
    dopamine_sum: float = 0.0
    terminal_count: int = 0

    @property
    def pain_mean(self) -> float:
        return self.pain_sum / self.count if self.count else 0.0

    @property
    def dopamine_mean(self) -> float:
        return self.dopamine_sum / self.count if self.count else 0.0

    @property
    def terminal_rate(self) -> float:
        return self.terminal_count / self.count if self.count else 0.0


@dataclass
class MultiViewTransitionModel:
    """Sparse empirical kernel queried at any finite frame window.

    ``edges[(source, acting_action)][target]`` stores realized edge counts
    and feedback sufficient statistics.  ``state_radius`` is the pooled
    high-quantile nearest-prototype distance from the training stream; a
    prediction farther away returns all mass as UNKNOWN instead of aliasing
    an off-support state to a familiar node.
    """

    lag: int
    edges: Dict[Tuple[str, int], Dict[str, EdgeMeasure]]
    state_radius: float
    training_rows: int
    radius_quantile: float = 0.995
    _row_counts: Dict[Tuple[str, int], int] = field(default_factory=dict)
    node_map: Optional[Mapping[str, str]] = None

    @classmethod
    def fit_rows(cls, rows: Sequence[Mapping], agent, lag: Optional[int] = None,
                 radius_quantile: float = 0.995, node_map=None):
        """Fit from realized stream rows and the runtime's learned prototypes.

        The transition's acting action is taken from ``history[lag]``.
        A row lacking that action is excluded rather than silently falling
        back to the newly issued action.
        """
        lag = int(agent.lag if lag is None else lag)
        scales = np.asarray(agent.scales, dtype=float)
        edges = defaultdict(lambda: defaultdict(EdgeMeasure))
        row_counts = defaultdict(int)
        distances = []
        included = 0

        for row in rows:
            obs = np.asarray(row.get("obs", ()), dtype=float).reshape(-1)
            next_obs = np.asarray(row.get("next_obs", ()), dtype=float).reshape(-1)
            history = np.asarray(row.get("history", ()), dtype=float).reshape(-1)
            if (not len(obs) or len(next_obs) != len(obs)
                    or history.size <= lag
                    or not np.isfinite(obs).all()
                    or not np.isfinite(next_obs).all()
                    or not np.isfinite(history).all()):
                continue

            # History is newest-first; the pending action state presented to
            # the runtime is oldest-first.  The action at the oldest pending
            # slot is the action that governs obs -> next_obs.
            pending = history[1:lag + 1][::-1]
            if len(pending) != lag:
                continue
            acting = int(round(float(history[lag])))
            issued = int(round(float(history[0])))
            source_state = np.concatenate([obs, pending]) / scales
            next_pending = (
                np.concatenate([pending[1:], [float(issued)]])
                if lag else np.empty(0, dtype=float)
            )
            target_state = np.concatenate([next_obs, next_pending]) / scales
            source, source_distance = agent._nearest(source_state)
            target, _target_distance = agent._nearest(target_state)
            if source is None or target is None:
                continue
            source = str(source)
            target = str(target)
            if node_map is not None:
                source = node_map.get(source)
                mapped_target = node_map.get(target)
                if source is None:
                    continue
                # A terminal successor is absorbing in prediction, so its
                # post-terminal prototype need not have an outgoing profile.
                if mapped_target is None and bool(row.get("done", False)):
                    target = "__terminal__"
                elif mapped_target is None:
                    continue
                else:
                    target = mapped_target

            source_distance = float(source_distance)
            if np.isfinite(source_distance):
                distances.append(source_distance)

            measure = edges[(str(source), acting)][str(target)]
            signals = np.asarray(
                row.get("signals", (0.0, 0.0)), dtype=float
            ).reshape(-1)
            if signals.size < 2 or not np.isfinite(signals[:2]).all():
                continue
            measure.count += 1
            measure.dopamine_sum += float(signals[0])
            measure.pain_sum += float(signals[1])
            measure.terminal_count += int(bool(row.get("done", False)))
            row_counts[(str(source), acting)] += 1
            included += 1

        radius = (
            float(np.quantile(distances, float(radius_quantile)))
            if distances else 0.0
        )
        return cls(
            lag=lag,
            edges={
                key: dict(targets) for key, targets in edges.items()
            },
            state_radius=radius,
            training_rows=included,
            radius_quantile=float(radius_quantile),
            _row_counts=dict(row_counts),
            node_map=None if node_map is None else dict(node_map),
        )

    def transition(self, source: str, action: int):
        """Return ``(target probabilities, support)`` or ``None`` if unseen."""
        key = (str(source), int(action))
        targets = self.edges.get(key)
        total = int(self._row_counts.get(key, 0))
        if not targets or total <= 0:
            return None
        return (
            {
                target: measure.count / float(total)
                for target, measure in targets.items()
                if measure.count > 0
            },
            total,
        )

    def predict_views(self, agent, observation: Sequence[float],
                      pending: Sequence[int], issued_actions: Sequence[int],
                      windows: Iterable[int], target_nodes=None):
        """Compose the observed kernel and return snapshots at any windows.

        ``pending`` is oldest-first, with its first element acting on the next
        state transition.  ``issued_actions[t]`` is the action newly issued
        on that frame and joins the back of the pending queue.  Each snapshot
        contains a node distribution, cumulative terminal/feedback measures,
        and unsupported mass.  If ``target_nodes`` is supplied, first passage
        into that set is accumulated in ``target_hit`` and becomes absorbing.
        Differences in ``target_hit`` across queried windows are the learned
        stopping-time mass.  A missing edge moves that branch to UNKNOWN.
        """
        requested = sorted(set(int(value) for value in windows))
        if any(value < 0 for value in requested):
            raise ValueError("windows must be non-negative")
        actions = [int(value) for value in issued_actions]
        if requested and max(requested) > len(actions):
            raise ValueError("issued_actions must cover the largest window")
        queue = [int(value) for value in pending]
        if len(queue) != self.lag:
            raise ValueError("pending must contain exactly lag actions")

        obs = np.asarray(observation, dtype=float).reshape(-1)
        scales = np.asarray(agent.scales, dtype=float)
        state = np.concatenate([obs, np.asarray(queue, dtype=float)]) / scales
        source, distance = agent._nearest(state)
        snapshots = {}
        if source is None or float(distance) > self.state_radius:
            for window in requested:
                snapshots[window] = dict(
                    distribution={}, terminal=0.0, target_hit=0.0,
                    unknown=1.0,
                    dopamine=0.0, pain=0.0, supported_rows=0,
                    status=UNKNOWN,
                )
            return snapshots
        if self.node_map is not None:
            source = self.node_map.get(str(source))
            if source is None:
                for window in requested:
                    snapshots[window] = dict(
                        distribution={}, terminal=0.0, target_hit=0.0,
                        unknown=1.0, dopamine=0.0, pain=0.0,
                        supported_rows=0, status=UNKNOWN,
                    )
                return snapshots

        mass = {str(source): 1.0}
        terminal_mass = 0.0
        target_hit = 0.0
        unknown_mass = 0.0
        dopamine = 0.0
        pain = 0.0
        supported_rows = 0
        target_set = (
            None if target_nodes is None
            else {str(value) for value in target_nodes}
        )
        if target_set is not None and str(source) in target_set:
            target_hit = 1.0
            mass = {}
        if 0 in requested:
            snapshots[0] = dict(
                distribution=dict(mass), terminal=terminal_mass,
                target_hit=target_hit,
                unknown=unknown_mass, dopamine=dopamine, pain=pain,
                supported_rows=supported_rows, status="supported",
            )

        for step, issued in enumerate(actions, start=1):
            acting = queue[0] if self.lag else int(issued)
            next_mass = defaultdict(float)
            for node, probability in mass.items():
                transition = self.transition(node, acting)
                if transition is None:
                    unknown_mass += probability
                    continue
                distribution, support = transition
                supported_rows += support
                for target, edge_probability in distribution.items():
                    flow = probability * float(edge_probability)
                    measure = self.edges[(node, acting)][target]
                    dopamine += flow * measure.dopamine_mean
                    pain += flow * measure.pain_mean
                    terminal_flow = flow * measure.terminal_rate
                    terminal_mass += terminal_flow
                    if target_set is not None and target in target_set:
                        target_hit += flow
                    live_flow = flow - terminal_flow
                    if live_flow > 0.0:
                        if target_set is None or target not in target_set:
                            next_mass[target] += live_flow
            mass = dict(next_mass)
            if self.lag:
                queue = queue[1:] + [int(issued)]
            if step in requested:
                snapshots[step] = dict(
                    distribution=dict(mass),
                    terminal=float(terminal_mass),
                    target_hit=float(target_hit),
                    unknown=float(unknown_mass),
                    dopamine=float(dopamine),
                    pain=float(pain),
                    supported_rows=int(supported_rows),
                    status=(
                        "supported" if unknown_mass <= 1e-12 else UNKNOWN
                    ),
                )
        return snapshots

    def summary(self):
        return dict(
            rows=int(self.training_rows),
            source_action_pairs=len(self.edges),
            edges=sum(len(targets) for targets in self.edges.values()),
            state_radius=float(self.state_radius),
            radius_quantile=float(self.radius_quantile),
            lag=int(self.lag),
            coarse_states=(
                len(set(self.node_map.values()))
                if self.node_map is not None else None
            ),
        )
