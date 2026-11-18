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


class NodeGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str, str], Edge] = {}
        self.outgoing: Dict[str, List[str]] = {}
        self.incoming: Dict[str, List[str]] = {}
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

    def prune(self, min_association: float = 1e-3,
              min_edge_trace: float = 1e-4) -> List[Tuple[str, str, str]]:
        removed = []
        for key, edge in list(self.edges.items()):
            drop = False
            if edge.kind == ASSOCIATION and abs(edge.weight) < min_association:
                drop = True
            elif edge.kind == TRANSITION_EDGE and edge.edge_node is not None:
                if self.nodes[edge.edge_node].trace < min_edge_trace:
                    drop = True
            if drop:
                self.edges.pop(key)
                removed.append(key)
                for bucket, node_id in (
                    (self.outgoing, edge.source),
                    (self.incoming, edge.target),
                    (self.outgoing, edge.target),
                    (self.incoming, edge.source),
                ):
                    if key in bucket.get(node_id, []):
                        bucket[node_id].remove(key)
                if edge.edge_node is not None:
                    self.nodes.pop(edge.edge_node, None)
                    self.outgoing.pop(edge.edge_node, None)
                    self.incoming.pop(edge.edge_node, None)
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
            edge_nodes=sum(n.kind == EDGE for n in self.nodes.values()),
            memories=sum(n.kind == MEMORY for n in self.nodes.values()),
        )
