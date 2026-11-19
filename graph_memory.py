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
    MEMBERSHIP,
    MEMORY,
    TRANSITION_EDGE,
    NodeGraph,
)


def kmeans(observations, clusters, seed=0, iterations=25):
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or len(observations) == 0:
        raise ValueError("observations must be a non-empty matrix")
    clusters = int(clusters)
    if clusters < 1:
        raise ValueError("clusters must be positive")
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
    def __init__(self, clusters=48, seed=0, rate=0.2, memory_weight=0.25):
        if int(clusters) < 1:
            raise ValueError("clusters must be positive")
        self.clusters = int(clusters)
        self.seed = int(seed)
        self.rate = float(rate)
        self.memory_weight = float(memory_weight)
        self.centers = None
        self.graph = NodeGraph()
        self.base = 0.0
        self.previous = deque(maxlen=1)
        self.episode = None
        self._concept_values = None
        self._concept_dopamine = None
        self._memory_values = None
        self._memory_dopamine = None

    def prototype(self, observations):
        observations = np.asarray(observations, dtype=float)
        distances = ((observations[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def fit(self, rows):
        if not rows:
            raise ValueError("need at least one transition to fit")
        self.graph = NodeGraph()
        self.centers = None
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        next_observations = np.asarray(
            [row.get("next_obs", row["obs"]) for row in rows], dtype=float
        )
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
        labels = self.prototype(observations)
        next_labels = self.prototype(next_observations)
        previous = None
        memory = None
        current_episode = None
        for index, (row, label, next_label) in enumerate(
            zip(rows, labels, next_labels)
        ):
            if row["episode"] != current_episode:
                current_episode = row["episode"]
                previous = None
                memory = self.graph.add_memory([], "memory_%d" % current_episode)
            concept = concepts[int(label)]
            next_concept = concepts[int(next_label)]
            self.graph.activate([concept], [1.0])
            self.graph.add_membership(memory, concept)
            if previous is not None:
                self.graph.observe_association([previous, concept], rate=self.rate)
            pain_signal = float(signals[index, 1])
            dopamine_signal = float(signals[index, 0])
            # The edge represents the transition that the label describes.
            key = (concept, next_concept, TRANSITION_EDGE)
            is_new = key not in self.graph.edges
            edge = self.graph.observe_transition(
                concept, next_concept, rate=self.rate
            )
            if edge.edge_node is not None:
                edge_node_id = edge.edge_node
                edge_node = self.graph.nodes[edge_node_id]
                if is_new:
                    edge_node.value = self.base
                    edge_node.dopamine = self.dopamine_base
                self.graph.add_membership(memory, edge_node_id)
                self._update_value(edge_node_id, pain_signal, dopamine_signal)
            # The state value belongs to the state that the terminal label names.
            self.graph.activate([next_concept], [1.0])
            self._update_value(next_concept, pain_signal, dopamine_signal)
            if memory is not None:
                self.graph.observe_memory(memory, rate=self.rate)
                self._update_value(memory, pain_signal, dopamine_signal)
            previous = concept
        self._rebuild_tables()
        return self

    def _rebuild_tables(self):
        """Cache per-concept state/memory values for fast batched scoring."""
        count = len(self.centers)
        self._concept_values = np.zeros(count, dtype=float)
        self._concept_dopamine = np.zeros(count, dtype=float)
        self._memory_values = np.full(count, np.nan, dtype=float)
        self._memory_dopamine = np.full(count, np.nan, dtype=float)
        for index in range(count):
            concept = "concept_%d" % index
            node = self.graph.nodes[concept]
            self._concept_values[index] = node.value
            self._concept_dopamine[index] = node.dopamine
            memories = [
                key[0] for key in self.graph.incoming.get(concept, [])
                if key[2] == MEMBERSHIP
            ]
            if memories:
                self._memory_values[index] = float(np.mean([
                    self.graph.nodes[memory].value for memory in memories
                ]))
                self._memory_dopamine[index] = float(np.mean([
                    self.graph.nodes[memory].dopamine for memory in memories
                ]))

    def _update_value(self, node_id, pain, dopamine):
        node = self.graph.nodes[node_id]
        strength = max(abs(node.trace), 0.05)
        node.value += self.rate * (pain - node.value) * strength
        node.dopamine += self.rate * (dopamine - node.dopamine) * strength

    def risk(self, rows):
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        labels = self.prototype(observations)
        previous = np.full(len(labels), -1, dtype=int)
        current_episode = None
        for index, row in enumerate(rows):
            if row["episode"] != current_episode:
                current_episode = row["episode"]
            else:
                previous[index] = labels[index - 1]
        return self.risk_labels(labels, previous)

    def _state_values(self, labels, previous_labels, attribute):
        labels = np.asarray(labels, dtype=int)
        if attribute == "value":
            concept_values = self._concept_values
            memory_values = self._memory_values
        else:
            concept_values = self._concept_dopamine
            memory_values = self._memory_dopamine
        if concept_values is None:
            raise RuntimeError("graph memory is not fitted")
        values = concept_values[labels].copy()
        if previous_labels is not None:
            previous_labels = np.asarray(previous_labels, dtype=int)
            for index, (previous, current) in enumerate(zip(previous_labels, labels)):
                if previous < 0:
                    continue
                edge_node = self.graph.edge_node(
                    "concept_%d" % int(previous), "concept_%d" % int(current)
                )
                if edge_node is not None:
                    values[index] = 0.5 * (
                        values[index] + getattr(self.graph.nodes[edge_node], attribute)
                    )
        memory = memory_values[labels]
        present = ~np.isnan(memory)
        if present.any():
            values[present] = (
                (1.0 - self.memory_weight) * values[present]
                + self.memory_weight * memory[present]
            )
        return np.clip(values, 0.0, 1.0) if attribute == "value" else values

    def risk_labels(self, labels, previous_labels=None):
        return self._state_values(labels, previous_labels, "value")

    def value_labels(self, labels, previous_labels=None):
        return self._state_values(labels, previous_labels, "dopamine")

    def risk_states(self, observations, previous_labels=None):
        """Risk for arbitrary predicted observations, without episode boundaries."""
        observations = np.asarray(observations, dtype=float)
        return self.risk_labels(self.prototype(observations), previous_labels)

    def value_states(self, observations, previous_labels=None):
        observations = np.asarray(observations, dtype=float)
        return self.value_labels(self.prototype(observations), previous_labels)

    def snapshot(self):
        return self.graph.snapshot()
