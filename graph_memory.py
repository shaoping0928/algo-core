"""Prototype graph memory over real Flappy observations.

Observation prototypes become concept nodes. Consecutive concepts become directed
transition edges with edge nodes. Each episode becomes a memory node binding its
concepts/edge nodes. Pain only updates values of nodes/edge nodes that were active;
it does not create them.
"""
from collections import deque

import numpy as np

from node_graph import (
    ASSOCIATION,
    EDGE,
    MEMORY,
    TRANSITION_EDGE,
    NodeGraph,
)


def kmeans(observations, clusters, seed=0, iterations=25):
    observations = np.asarray(observations, dtype=float)
    rng = np.random.RandomState(seed)
    if len(observations) <= clusters:
        centers = observations.copy()
    else:
        centers = observations[rng.choice(len(observations), clusters, replace=False)].copy()
    for _ in range(iterations):
        distances = ((observations[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for index in range(clusters):
            members = observations[labels == index]
            if len(members):
                updated[index] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


class PrototypeGraphMemory:
    def __init__(self, clusters=48, seed=0, rate=0.2):
        self.clusters = int(clusters)
        self.seed = int(seed)
        self.rate = float(rate)
        self.centers = None
        self.graph = NodeGraph()
        self.base = 0.0
        self.previous = deque(maxlen=1)
        self.episode = None

    def prototype(self, observations):
        observations = np.asarray(observations, dtype=float)
        distances = ((observations[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def fit(self, rows):
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        pain = np.asarray([row["terminal"] for row in rows], dtype=float)
        signals = np.asarray(
            [row.get("signals", [1.0 - p, p]) for row, p in zip(rows, pain)],
            dtype=float,
        )
        self.base = float(pain.mean()) if len(pain) else 0.0
        self.dopamine_base = (
            float(signals[:, 0].mean()) if len(signals) else 0.0
        )
        self.centers = kmeans(observations, self.clusters, seed=self.seed)
        concepts = [
            self.graph.add_node("concept", "concept_%d" % index, beta=0.2)
            for index in range(len(self.centers))
        ]
        for concept in concepts:
            self.graph.nodes[concept].value = self.base
            self.graph.nodes[concept].dopamine = self.dopamine_base
        previous = None
        memory = None
        current_episode = None
        for index, (row, label) in enumerate(zip(rows, self.prototype(observations))):
            if row["episode"] != current_episode:
                current_episode = row["episode"]
                previous = None
                memory = self.graph.add_memory([], "memory_%d" % current_episode)
            concept = concepts[int(label)]
            self.graph.activate([concept], [1.0])
            self.graph.add_membership(memory, concept)
            if previous is not None:
                self.graph.observe_association([previous, concept], rate=self.rate)
                edge = self.graph.observe_transition(previous, concept, rate=self.rate)
                if edge.edge_node is not None:
                    edge_node = self.graph.nodes[edge.edge_node]
                    edge_node.value = self.base
                    edge_node.dopamine = self.dopamine_base
                    self.graph.add_membership(memory, edge.edge_node)
            pain_signal = float(signals[index, 1])
            dopamine_signal = float(signals[index, 0])
            self._update_value(concept, pain_signal, dopamine_signal)
            if previous is not None:
                edge_node = self.graph.edge_node(previous, concept)
                if edge_node is not None:
                    self._update_value(edge_node, pain_signal, dopamine_signal)
            if memory is not None:
                self.graph.observe_memory(memory, rate=self.rate)
                self._update_value(memory, pain_signal, dopamine_signal)
            previous = concept
        return self

    def _update_value(self, node_id, pain, dopamine):
        node = self.graph.nodes[node_id]
        strength = max(abs(node.trace), 0.05)
        node.value += self.rate * (pain - node.value) * strength
        node.dopamine += self.rate * (dopamine - node.dopamine) * strength

    def risk(self, rows):
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        labels = self.prototype(observations)
        result = []
        previous = None
        current_episode = None
        for row, label in zip(rows, labels):
            if row["episode"] != current_episode:
                current_episode = row["episode"]
                previous = None
            concept = "concept_%d" % int(label)
            value = self.graph.nodes[concept].value
            if previous is not None:
                edge_node = self.graph.edge_node(previous, concept)
                if edge_node is not None:
                    value = 0.5 * (value + self.graph.nodes[edge_node].value)
            result.append(value)
            previous = concept
        return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)

    def risk_labels(self, labels, previous_labels=None):
        labels = np.asarray(labels, dtype=int)
        values = np.asarray(
            [self.graph.nodes["concept_%d" % int(label)].value for label in labels],
            dtype=float,
        )
        if previous_labels is not None:
            previous_labels = np.asarray(previous_labels, dtype=int)
            for index, (previous, current) in enumerate(zip(previous_labels, labels)):
                edge_node = self.graph.edge_node(
                    "concept_%d" % int(previous), "concept_%d" % int(current)
                )
                if edge_node is not None:
                    values[index] = 0.5 * (
                        values[index] + self.graph.nodes[edge_node].value
                    )
        return np.clip(values, 0.0, 1.0)

    def value_labels(self, labels, previous_labels=None):
        labels = np.asarray(labels, dtype=int)
        values = np.asarray(
            [self.graph.nodes["concept_%d" % int(label)].dopamine for label in labels],
            dtype=float,
        )
        if previous_labels is not None:
            previous_labels = np.asarray(previous_labels, dtype=int)
            for index, (previous, current) in enumerate(zip(previous_labels, labels)):
                edge_node = self.graph.edge_node(
                    "concept_%d" % int(previous), "concept_%d" % int(current)
                )
                if edge_node is not None:
                    values[index] = 0.5 * (
                        values[index] + self.graph.nodes[edge_node].dopamine
                    )
        return values

    def risk_states(self, observations, previous_labels=None):
        """Risk for arbitrary predicted observations, without episode boundaries."""
        observations = np.asarray(observations, dtype=float)
        return self.risk_labels(self.prototype(observations), previous_labels)

    def value_states(self, observations, previous_labels=None):
        observations = np.asarray(observations, dtype=float)
        return self.value_labels(self.prototype(observations), previous_labels)

    def snapshot(self):
        return self.graph.snapshot()
