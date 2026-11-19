"""Unified node/edge substrate.

All nodes share activation, a leaky trace, association participation, binding and
derivation. Node kinds add semantics:

    concept       co-occurrence association
    transition    a directed transition, optionally hosted by an edge node
    memory        a hub binding several nodes into one event
    derived       a component calculator derived from another node (bounded depth)

Edges are typed:

    association   symmetric co-occurrence weight
    transition    directed order/causality; may own an edge node
    membership    memory -> member
    derivation    derived -> source

Edge nodes are ordinary nodes hosted by transition edges. They can be associated
with concepts/memories like any other node. Association edges do not get edge nodes.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CONCEPT = "concept"
TRANSITION = "transition"
MEMORY = "memory"
DERIVED = "derived"
EDGE = "edge"

ASSOCIATION = "association"
TRANSITION_EDGE = "transition"
MEMBERSHIP = "membership"
DERIVATION = "derivation"


@dataclass
class Node:
    id: str
    kind: str
    beta: float = 0.2
    activation: float = 0.0
    trace: float = 0.0
    members: List[str] = field(default_factory=list)
    source: Optional[str] = None
    depth: int = 0
    edge_key: Optional[Tuple[str, str]] = None
    value: float = 0.0
    dopamine: float = 0.0
    count: int = 0
    eligibility_pain: float = 0.0
    eligibility_dopamine: float = 0.0
    immediate_pain: float = 0.0
    immediate_dopamine: float = 0.0
    features: Optional[np.ndarray] = None
    role: str = ""
    component_id: Optional[str] = None
    sign: int = 0

    def activate(self, value: float) -> float:
        self.activation = float(value)
        self.trace = (1.0 - self.beta) * self.trace + self.beta * self.activation
        return self.activation

    def decay(self, factor: float) -> None:
        self.trace *= float(factor)
        self.activation *= float(factor)


@dataclass
class Edge:
    source: str
    target: str
    kind: str
    weight: float = 0.0
    eligibility: float = 0.0
    edge_node: Optional[str] = None
    order: int = 0

    def key(self):
        return (self.source, self.target, self.kind)


@dataclass
class ActionEdge:
    """A transition conditioned on the action that caused it.

    Unlike `Edge`, this records how often an action moved one concept node to
    another and the learned expected valence of that transition.  It is the
    substrate for node-level `Q(s, a)` values.
    """

    source: str
    target: str
    action: int
    count: int = 0
    q_pain: float = 0.0
    q_dopamine: float = 0.0
    prediction_error: float = 0.0

    def key(self):
        return (self.source, self.target, int(self.action))


class NodeGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str, str], Edge] = {}
        self.outgoing: Dict[str, List[str]] = {}
        self.incoming: Dict[str, List[str]] = {}
        # Action-conditioned transition edges: (source, target, action) -> edge.
        self.action_edges: Dict[Tuple[str, str, int], ActionEdge] = {}
        self.action_out: Dict[Tuple[str, int], List[Tuple[str, str, int]]] = {}
        self._counter = 0

    # ---- creation -----------------------------------------------------------
    def add_node(self, kind: str, node_id: Optional[str] = None, **kwargs) -> str:
        if node_id is None:
            self._counter += 1
            node_id = "%s_%d" % (kind, self._counter)
        if node_id in self.nodes:
            raise ValueError("duplicate node id %s" % node_id)
        self.nodes[node_id] = Node(node_id, kind, **kwargs)
        self.outgoing[node_id] = []
        self.incoming[node_id] = []
        return node_id

    def add_association(self, first: str, second: str, weight: float = 0.0) -> Edge:
        if first == second:
            raise ValueError("self-association is not allowed")
        a, b = sorted((first, second))
        key = (a, b, ASSOCIATION)
        if key in self.edges:
            return self.edges[key]
        for node in (first, second):
            if node not in self.nodes:
                raise KeyError(node)
        edge = Edge(a, b, ASSOCIATION, weight=float(weight))
        self.edges[key] = edge
        self.outgoing[a].append(key)
        self.incoming[b].append(key)
        self.outgoing[b].append(key)
        self.incoming[a].append(key)
        return edge

    def add_transition(self, source: str, target: str, weight: float = 0.0,
                       create_edge_node: bool = True) -> Edge:
        for node in (source, target):
            if node not in self.nodes:
                raise KeyError(node)
        key = (source, target, TRANSITION_EDGE)
        if key in self.edges:
            return self.edges[key]
        edge = Edge(source, target, TRANSITION_EDGE, weight=float(weight))
        if create_edge_node:
            edge_node = self.add_node(
                EDGE, "edge:%s->%s" % (source, target),
                edge_key=(source, target),
            )
            edge.edge_node = edge_node
        self.edges[key] = edge
        self.outgoing[source].append(key)
        self.incoming[target].append(key)
        return edge

    def add_action_edge(self, source: str, action: int,
                        target: str) -> ActionEdge:
        """Create or return the action-conditioned transition edge."""
        for node in (source, target):
            if node not in self.nodes:
                raise KeyError(node)
        key = (source, target, int(action))
        if key in self.action_edges:
            return self.action_edges[key]
        edge = ActionEdge(source, target, int(action))
        self.action_edges[key] = edge
        self.action_out.setdefault((source, int(action)), []).append(key)
        return edge

    def action_edges_from(self, source: str, action: int) -> List[ActionEdge]:
        return [
            self.action_edges[key]
            for key in self.action_out.get((source, int(action)), [])
        ]

    def action_targets(self, source: str, action: int) -> List[str]:
        return [edge.target for edge in self.action_edges_from(source, action)]

    def add_memory(self, members: Iterable[str], node_id: Optional[str] = None) -> str:
        members = list(dict.fromkeys(members))
        for member in members:
            if member not in self.nodes:
                raise KeyError(member)
        memory = self.add_node(MEMORY, node_id)
        self.nodes[memory].members = members
        for member in members:
            key = (memory, member, MEMBERSHIP)
            edge = Edge(memory, member, MEMBERSHIP)
            self.edges[key] = edge
            self.outgoing[memory].append(key)
            self.incoming[member].append(key)
        return memory

    def add_membership(self, memory_id: str, member_id: str) -> Edge:
        if memory_id not in self.nodes or member_id not in self.nodes:
            raise KeyError(memory_id if memory_id not in self.nodes else member_id)
        memory = self.nodes[memory_id]
        if memory.kind != MEMORY:
            raise ValueError("not a memory node")
        if member_id not in memory.members:
            memory.members.append(member_id)
        key = (memory_id, member_id, MEMBERSHIP)
        if key not in self.edges:
            edge = Edge(memory_id, member_id, MEMBERSHIP)
            self.edges[key] = edge
            self.outgoing[memory_id].append(key)
            self.incoming[member_id].append(key)
            return edge
        return self.edges[key]

    def add_derived(self, source: str, kind: str = "component",
                    max_depth: int = 2, node_id: Optional[str] = None) -> str:
        if source not in self.nodes:
            raise KeyError(source)
        depth = self.nodes[source].depth + 1
        if depth > max_depth:
            raise ValueError("derivation depth exceeds %d" % max_depth)
        derived = self.add_node(DERIVED, node_id, source=source, depth=depth)
        key = (derived, source, DERIVATION)
        edge = Edge(derived, source, DERIVATION)
        self.edges[key] = edge
        self.outgoing[derived].append(key)
        self.incoming[source].append(key)
        return derived

    # ---- observation -------------------------------------------------------
    def activate(self, ids: Iterable[str], values: Optional[Iterable[float]] = None) -> None:
        ids = list(ids)
        if values is None:
            values = [1.0] * len(ids)
        for node_id, value in zip(ids, values):
            self.nodes[node_id].activate(value)

    def observe_association(self, active: Sequence[str], rate: float = 0.05) -> None:
        active = [node for node in dict.fromkeys(active) if node in self.nodes]
        for index, first in enumerate(active):
            for second in active[index + 1:]:
                edge = self.add_association(first, second)
                drive = (
                    self.nodes[first].trace * self.nodes[second].trace
                )
                edge.weight += rate * (drive - edge.weight)
                edge.eligibility = (1.0 - rate) * edge.eligibility + rate * drive

    def observe_transition(self, source: str, target: str,
                           rate: float = 0.1) -> Edge:
        edge = self.add_transition(source, target)
        drive = self.nodes[source].trace * self.nodes[target].trace
        edge.eligibility = (1.0 - rate) * edge.eligibility + rate * drive
        edge.weight += rate * (drive - edge.weight)
        if edge.edge_node is not None:
            self.nodes[edge.edge_node].activate(edge.eligibility)
        return edge

    def observe_memory(self, memory_id: str, rate: float = 0.1) -> None:
        memory = self.nodes[memory_id]
        if memory.kind != MEMORY:
            raise ValueError("not a memory node")
        value = np.mean([self.nodes[member].trace for member in memory.members]) \
            if memory.members else 0.0
        memory.activate(float(value))

    # ---- recall ------------------------------------------------------------
    def propagate(self, steps: int = 1, gain: float = 0.5,
                  decay: float = 0.9) -> Dict[str, float]:
        for _ in range(int(steps)):
            drive = {node_id: 0.0 for node_id in self.nodes}
            for edge in self.edges.values():
                if edge.kind == ASSOCIATION:
                    drive[edge.source] += edge.weight * self.nodes[edge.target].trace
                    drive[edge.target] += edge.weight * self.nodes[edge.source].trace
                elif edge.kind == MEMBERSHIP:
                    member = self.nodes[edge.target]
                    memory = self.nodes[edge.source]
                    drive[edge.source] += gain * member.trace
                    drive[edge.target] += gain * memory.trace
            for node_id, value in drive.items():
                node = self.nodes[node_id]
                node.activate(decay * node.activation + gain * value)
        return {node_id: node.activation for node_id, node in self.nodes.items()}

    def recall(self, cues: Iterable[str], steps: int = 2) -> Dict[str, float]:
        for node in self.nodes.values():
            node.activation = 0.0
        cues = list(cues)
        self.activate(cues, [1.0] * len(cues))
        return self.propagate(steps)

    # ---- forgetting --------------------------------------------------------
    def decay(self, edge_factor: float = 0.999, trace_factor: float = 0.9) -> None:
        for edge in self.edges.values():
            edge.weight *= edge_factor
            edge.eligibility *= trace_factor
        for node in self.nodes.values():
            node.decay(trace_factor)

    def _remove_edge(self, key: Tuple[str, str, str]) -> None:
        edge = self.edges.pop(key)
        for bucket, node_id in (
            (self.outgoing, edge.source),
            (self.incoming, edge.target),
            (self.outgoing, edge.target),
            (self.incoming, edge.source),
        ):
            if key in bucket.get(node_id, []):
                bucket[node_id].remove(key)

    def _remove_action_edge(self, key: Tuple[str, str, int]) -> None:
        edge = self.action_edges.pop(key)
        bucket = self.action_out.get((edge.source, edge.action), [])
        if key in bucket:
            bucket.remove(key)
        if not bucket:
            self.action_out.pop((edge.source, edge.action), None)

    def _remove_node(self, node_id: str) -> None:
        """Remove a node and every edge that still refers to it."""
        for key, edge in list(self.edges.items()):
            if edge.source == node_id or edge.target == node_id:
                self._remove_edge(key)
        for key, edge in list(self.action_edges.items()):
            if edge.source == node_id or edge.target == node_id:
                self._remove_action_edge(key)
        for node in self.nodes.values():
            if node.members:
                node.members = [member for member in node.members
                                if member != node_id]
        self.nodes.pop(node_id, None)
        self.outgoing.pop(node_id, None)
        self.incoming.pop(node_id, None)

    def prune(self, min_association: float = 1e-3,
              min_edge_trace: float = 1e-4) -> List[Tuple[str, str, str]]:
        removed = []
        for key, edge in list(self.edges.items()):
            if key not in self.edges:
                continue
            drop = False
            if edge.kind == ASSOCIATION and abs(edge.weight) < min_association:
                drop = True
            elif edge.kind == TRANSITION_EDGE and edge.edge_node is not None:
                if self.nodes[edge.edge_node].trace < min_edge_trace:
                    drop = True
            if drop:
                self._remove_edge(key)
                removed.append(key)
                if edge.edge_node is not None:
                    self._remove_node(edge.edge_node)
        return removed

    def edge_node(self, source: str, target: str) -> Optional[str]:
        edge = self.edges.get((source, target, TRANSITION_EDGE))
        return edge.edge_node if edge else None

    def snapshot(self):
        return dict(
            nodes=len(self.nodes),
            edges=len(self.edges),
            associations=sum(e.kind == ASSOCIATION for e in self.edges.values()),
            transitions=sum(e.kind == TRANSITION_EDGE for e in self.edges.values()),
            action_edges=len(self.action_edges),
            edge_nodes=sum(n.kind == EDGE for n in self.nodes.values()),
            memories=sum(n.kind == MEMORY for n in self.nodes.values()),
        )

    # ---- persistence -------------------------------------------------------
    def to_state(self) -> dict:
        return dict(
            counter=int(self._counter),
            nodes=[
                dict(
                    id=node.id,
                    kind=node.kind,
                    beta=node.beta,
                    activation=node.activation,
                    trace=node.trace,
                    members=list(node.members),
                    source=node.source,
                    depth=node.depth,
                    edge_key=(
                        None if node.edge_key is None
                        else list(node.edge_key)
                    ),
                    value=node.value,
                    dopamine=node.dopamine,
                    count=node.count,
                    eligibility_pain=node.eligibility_pain,
                    eligibility_dopamine=node.eligibility_dopamine,
                    immediate_pain=node.immediate_pain,
                    immediate_dopamine=node.immediate_dopamine,
                    features=(
                        None if node.features is None
                        else np.asarray(node.features, dtype=float).tolist()
                    ),
                    role=node.role,
                    component_id=node.component_id,
                    sign=node.sign,
                )
                for node in self.nodes.values()
            ],
            edges=[
                dict(
                    source=edge.source,
                    target=edge.target,
                    kind=edge.kind,
                    weight=edge.weight,
                    eligibility=edge.eligibility,
                    edge_node=edge.edge_node,
                    order=edge.order,
                )
                for edge in self.edges.values()
            ],
            action_edges=[
                dict(
                    source=edge.source,
                    target=edge.target,
                    action=edge.action,
                    count=edge.count,
                    q_pain=edge.q_pain,
                    q_dopamine=edge.q_dopamine,
                    prediction_error=edge.prediction_error,
                )
                for edge in self.action_edges.values()
            ],
        )

    @classmethod
    def from_state(cls, state: dict) -> "NodeGraph":
        graph = cls()
        graph.nodes = {}
        graph.edges = {}
        graph.outgoing = {}
        graph.incoming = {}
        graph.action_edges = {}
        graph.action_out = {}
        graph._counter = int(state.get("counter", 0))
        for data in state.get("nodes", []):
            features = data.get("features")
            edge_key = data.get("edge_key")
            node = Node(
                id=data["id"],
                kind=data["kind"],
                beta=float(data.get("beta", 0.2)),
                activation=float(data.get("activation", 0.0)),
                trace=float(data.get("trace", 0.0)),
                members=list(data.get("members", [])),
                source=data.get("source"),
                depth=int(data.get("depth", 0)),
                edge_key=None if edge_key is None else tuple(edge_key),
                value=float(data.get("value", 0.0)),
                dopamine=float(data.get("dopamine", 0.0)),
                count=int(data.get("count", 0)),
                eligibility_pain=float(data.get("eligibility_pain", 0.0)),
                eligibility_dopamine=float(
                    data.get("eligibility_dopamine", 0.0)
                ),
                immediate_pain=float(data.get("immediate_pain", 0.0)),
                immediate_dopamine=float(
                    data.get("immediate_dopamine", 0.0)
                ),
                features=(
                    None if features is None
                    else np.asarray(features, dtype=float)
                ),
                role=data.get("role", ""),
                component_id=data.get("component_id"),
                sign=int(data.get("sign", 0)),
            )
            graph.nodes[node.id] = node
            graph.outgoing[node.id] = []
            graph.incoming[node.id] = []
        for data in state.get("edges", []):
            edge = Edge(
                source=data["source"],
                target=data["target"],
                kind=data["kind"],
                weight=float(data.get("weight", 0.0)),
                eligibility=float(data.get("eligibility", 0.0)),
                edge_node=data.get("edge_node"),
                order=int(data.get("order", 0)),
            )
            key = (edge.source, edge.target, edge.kind)
            graph.edges[key] = edge
            if edge.kind == ASSOCIATION:
                graph.outgoing[edge.source].append(key)
                graph.incoming[edge.target].append(key)
                graph.outgoing[edge.target].append(key)
                graph.incoming[edge.source].append(key)
            else:
                graph.outgoing[edge.source].append(key)
                graph.incoming[edge.target].append(key)
        for data in state.get("action_edges", []):
            edge = ActionEdge(
                source=data["source"],
                target=data["target"],
                action=int(data["action"]),
                count=int(data.get("count", 0)),
                q_pain=float(data.get("q_pain", 0.0)),
                q_dopamine=float(data.get("q_dopamine", 0.0)),
                prediction_error=float(data.get("prediction_error", 0.0)),
            )
            key = (edge.source, edge.target, edge.action)
            graph.action_edges[key] = edge
            graph.action_out.setdefault(
                (edge.source, edge.action), []
            ).append(key)
        return graph
