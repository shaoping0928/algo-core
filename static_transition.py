"""Non-temporal (endogenous) state transitions.

A temporal transition advances the world: ``(state, action) -> next state``.
A static transition stays at the same instant and integrates what is already
known about the state: the active feature nodes, their associations, the
memories they belong to, and the derived nodes whose sources are active.  It is
a closure over the graph's **non-temporal** edges:

* ``association``   symmetric co-occurrence, basis of state integration;
* ``membership``    member -> memory retrieval (never the reverse);
* ``derivation``    source -> derived (never the reverse).

``transition`` edges are deliberately excluded.  Following one would advance
time; that is the extensive, forward transition, and it lives in the stage
outcome memory and the frame model, not here.

The operator is a representation step, not a boundary detector.  It can be
applied at any state, and its output is the integrated state that stage-change
bundles, similarity typing and layering consume.  Unknown nodes stay at zero
strength; no node is fabricated from a counterfactual edge.

Optional read-only ``StaticGraphStrengths`` overlays replace the graph's raw
association and membership weights, so learned continuous strengths can drive
the closure without writing them into the runtime graph.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from node_graph import (
    ASSOCIATION,
    DERIVATION,
    MEMBERSHIP,
    TRANSITION_EDGE,
)


NON_TEMPORAL_EDGES = (ASSOCIATION, MEMBERSHIP, DERIVATION)


@dataclass(frozen=True)
class StaticNode:
    node_id: str
    kind: str
    strength: float
    depth: int


@dataclass(frozen=True)
class StaticState:
    """The integrated state at one instant."""

    nodes: Tuple[StaticNode, ...]
    seeds: Tuple[str, ...]
    max_steps: int
    converged: bool
    edge_counts: Tuple[Tuple[str, int], ...]

    def strengths(self) -> dict:
        return {node.node_id: float(node.strength) for node in self.nodes}

    def vector(self, node_ids: Sequence[str]) -> np.ndarray:
        values = self.strengths()
        return np.asarray(
            [values.get(str(node), 0.0) for node in node_ids],
            dtype=float,
        )

    def change(self, previous: "StaticState", min_change: float = 0.5):
        """Signed node-strength tokens between two integrated states."""
        min_change = float(min_change)
        if min_change <= 0.0:
            raise ValueError("min_change must be positive")
        before = previous.strengths()
        after = self.strengths()
        tokens = []
        for node in sorted(set(before) | set(after)):
            delta = after.get(node, 0.0) - before.get(node, 0.0)
            if delta >= min_change:
                tokens.append((node, 1))
            elif delta <= -min_change:
                tokens.append((node, -1))
        return tuple(tokens)

    def by_kind(self) -> dict:
        result = {}
        for node in self.nodes:
            result.setdefault(node.kind, []).append(node.node_id)
        return {kind: tuple(ids) for kind, ids in result.items()}

    def snapshot(self):
        return dict(
            nodes=int(len(self.nodes)),
            seeds=int(len(self.seeds)),
            max_steps=int(self.max_steps),
            converged=bool(self.converged),
            kinds={
                kind: int(len(ids)) for kind, ids in self.by_kind().items()
            },
            edge_counts={
                str(kind): int(count)
                for kind, count in self.edge_counts
            },
        )


def _graph_of(agent):
    graph = getattr(agent, "graph", agent)
    if not hasattr(graph, "nodes") or not hasattr(graph, "outgoing"):
        raise ValueError("a node graph or an object with .graph is required")
    return graph


def _association_strength(weight: float) -> float:
    weight = float(weight)
    if weight <= 0.0:
        return 0.0
    return weight / (1.0 + weight)


def _binding_strength(weight: float) -> float:
    return min(1.0, max(0.0, float(weight)))


def _neighbours(graph, node_id, strengths=None):
    """Yield ``(edge_kind, neighbour, strength)`` over non-temporal edges.

    When a ``StaticGraphStrengths`` overlay is supplied, its learned
    associations replace the graph's association weights entirely and its
    membership strengths replace the graph's binding weights.
    """
    if strengths is not None:
        for other, value in strengths.associations_of(node_id):
            if other in graph.nodes:
                yield ASSOCIATION, other, float(value)
        for key in graph.incoming.get(node_id, ()):
            kind = key[2]
            edge = graph.edges[key]
            if kind == MEMBERSHIP:
                value = strengths.member_strength(key[0], node_id)
                if value > 0.0:
                    yield MEMBERSHIP, key[0], float(value)
            elif kind == DERIVATION:
                yield DERIVATION, key[0], _binding_strength(edge.weight)
            elif kind == TRANSITION_EDGE:
                continue
        return
    for key in graph.outgoing.get(node_id, ()):
        kind = key[2]
        if kind != ASSOCIATION:
            continue
        edge = graph.edges[key]
        other = key[1] if key[0] == node_id else key[0]
        yield ASSOCIATION, other, _association_strength(edge.weight)
    for key in graph.incoming.get(node_id, ()):
        kind = key[2]
        edge = graph.edges[key]
        if kind == MEMBERSHIP:
            yield MEMBERSHIP, key[0], _binding_strength(edge.weight)
        elif kind == DERIVATION:
            yield DERIVATION, key[0], _binding_strength(edge.weight)
        elif kind == TRANSITION_EDGE:
            # Temporal: never traversed by a static transition.
            continue


def endogenous_transition(agent, state=None, seeds=None,
                          max_steps: int = 2, decay: float = 0.5,
                          min_strength: float = 0.05,
                          strengths=None) -> StaticState:
    """Integrate the current state over non-temporal node edges.

    ``seeds`` are active node ids; when omitted they are read from the runtime
    with ``state`` encoded the same way its feature rules expect.  ``max_steps``
    bounds the closure depth, ``decay`` scales each hop and ``min_strength``
    drops weak activations.  Transition edges are never followed.  An optional
    ``StaticGraphStrengths`` overlay replaces the graph's association and
    membership weights, so learned continuous strengths can be used without
    writing them into the runtime graph.
    """
    if int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative")
    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    if float(min_strength) < 0.0:
        raise ValueError("min_strength must be non-negative")
    graph = _graph_of(agent)
    if seeds is None:
        if state is None:
            raise ValueError("provide seeds or a state for feature seeding")
        seeds = _seed_features(agent, state)
    seeds = tuple(dict.fromkeys(str(node) for node in seeds))
    for node in seeds:
        if node not in graph.nodes:
            raise KeyError(node)
    if not seeds:
        return StaticState((), (), int(max_steps), True, ())

    best = {node: (1.0, 0) for node in seeds}
    queue = deque((node, 1.0, 0) for node in seeds)
    edge_counts = Counter()
    while queue:
        node, strength, depth = queue.popleft()
        if depth >= int(max_steps):
            continue
        for kind, child, edge_strength in _neighbours(
                graph, node, strengths=strengths):
            edge_counts[kind] += 1
            value = float(strength) * float(edge_strength) * float(decay)
            if value < float(min_strength):
                continue
            current = best.get(child)
            if current is not None and value <= current[0] + 1e-12:
                continue
            best[child] = (value, depth + 1)
            queue.append((child, value, depth + 1))

    nodes = tuple(sorted(
        (
            StaticNode(
                node_id=node,
                kind=str(graph.nodes[node].kind),
                strength=float(value),
                depth=int(depth),
            )
            for node, (value, depth) in best.items()
        ),
        key=lambda item: (-item.strength, item.node_id),
    ))
    maximum_depth = max((node.depth for node in nodes), default=0)
    converged = bool(
        int(max_steps) == 0 or maximum_depth < int(max_steps)
    )
    return StaticState(
        nodes=nodes,
        seeds=tuple(seeds),
        max_steps=int(max_steps),
        converged=converged,
        edge_counts=tuple(sorted(edge_counts.items())),
    )


def _seed_features(agent, state):
    obs_dim = int(getattr(agent, "obs_dim", 0))
    if obs_dim <= 0:
        raise ValueError("agent.obs_dim is required for state seeding")
    obs = np.asarray(state, dtype=float).reshape(-1)[:obs_dim]
    scales = np.asarray(agent.scales, dtype=float)[:obs_dim]
    if not np.isfinite(obs).all() or not np.isfinite(scales).all():
        raise ValueError("state and scales must be finite")
    encoded = obs / scales
    active = agent._active_feature_ids(encoded)
    return tuple(str(node) for node in active)


def seed_state(agent, state=None, seeds=None,
               max_steps: int = 0, strengths=None) -> StaticState:
    """The one-step expansion, useful as the ablation baseline."""
    return endogenous_transition(
        agent, state=state, seeds=seeds, max_steps=max_steps,
        strengths=strengths,
    )


def integrate_episode(agent, seeds_per_state: Iterable[Iterable[str]],
                      max_steps: int = 2, decay: float = 0.5,
                      min_strength: float = 0.05,
                      strengths=None) -> Tuple[StaticState, ...]:
    """Apply the static transition to every state of an episode."""
    return tuple(
        endogenous_transition(
            agent, seeds=seeds, max_steps=max_steps, decay=decay,
            min_strength=min_strength, strengths=strengths,
        )
        for seeds in seeds_per_state
    )
