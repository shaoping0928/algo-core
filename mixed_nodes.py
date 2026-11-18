"""Generic primitive gates and mixed (conjunctive) pain-memory nodes.

A primitive gate is a one-dimensional threshold on an observation channel:

    high: sigmoid((x - threshold) / temperature)
    low : sigmoid((threshold - x) / temperature)

A mixed node is the product of two primitive gates. Candidates are created only when
the conjunction has enough support, raises the observed pain rate over the base rate,
and contains actual pain examples. The node owns a local pain valence; pain updates only
the nodes that are active. No pipe, gap, or collision variable is supplied by the caller.
"""
from itertools import combinations

import numpy as np


class PrimitiveGate:
    def __init__(self, channel, direction, threshold, temperature):
        self.channel = int(channel)
        self.direction = direction
        self.threshold = float(threshold)
        self.temperature = max(float(temperature), 1e-6)

    def activation(self, observations):
        observations = np.asarray(observations, dtype=float)
        x = observations[..., self.channel]
        if self.direction == "high":
            z = (x - self.threshold) / self.temperature
        else:
            z = (self.threshold - x) / self.temperature
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def describe(self, names=None):
        name = names[self.channel] if names and self.channel < len(names) else str(self.channel)
        return "%s %s %.4f" % (name, ">" if self.direction == "high" else "<", self.threshold)


class MixedNode:
    def __init__(self, first, second, valence, support, pain_rate, lift):
        self.first = first
        self.second = second
        self.valence = float(valence)
        self.support = float(support)
        self.pain_rate = float(pain_rate)
        self.lift = float(lift)


class ContinuousMixedMemory:
    """State-driven mixed memory: birth by coverage, pain by valence, forgetting by utility.

    Birth does not use pain. Candidate conjunctions are selected by how much uncovered
    observation mass they explain. Pain only assigns a local valence to already-created
    nodes. A final redundancy pass forgets overlapping low-utility nodes.
    """

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                 min_support=8.0, capacity=64, prior=2.0,
                 forgetting_overlap=0.92):
        self.quantiles = tuple(quantiles)
        self.min_support = float(min_support)
        self.capacity = int(capacity)
        self.prior = float(prior)
        self.forgetting_overlap = float(forgetting_overlap)
        self.gates = []
        self.nodes = []
        self.base_rate = 0.0
        self.created = 0
        self.forgotten = 0

    def _build_gates(self, observations):
        observations = np.asarray(observations, dtype=float)
        for channel in range(observations.shape[1]):
            values = observations[:, channel]
            spread = float(np.std(values))
            if spread <= 1e-9:
                continue
            thresholds = np.unique(np.quantile(values, self.quantiles))
            temperature = max(spread * 0.08, 1e-6)
            for threshold in thresholds:
                self.gates.append(PrimitiveGate(channel, "high", threshold, temperature))
                self.gates.append(PrimitiveGate(channel, "low", threshold, temperature))

    def _gate_matrix(self, observations):
        if not self.gates:
            return np.zeros((len(observations), 0))
        return np.column_stack([gate.activation(observations) for gate in self.gates])

    def fit(self, observations, pain):
        observations = np.asarray(observations, dtype=float)
        pain = np.asarray(pain, dtype=float).reshape(-1)
        if observations.ndim != 2 or len(observations) != len(pain):
            raise ValueError("observations and pain must have matching rows")
        self._build_gates(observations)
        activations = self._gate_matrix(observations)
        self.base_rate = float(pain.mean()) if len(pain) else 0.0
        pair_indices = list(combinations(range(len(self.gates)), 2))
        pair_activations = np.column_stack(
            [activations[:, first] * activations[:, second]
             for first, second in pair_indices]
        ) if pair_indices else np.zeros((len(observations), 0))
        support = pair_activations.sum(axis=0)
        viable = np.flatnonzero(support >= self.min_support)
        if not len(viable):
            return self
        # Birth: greedily explain observation mass not already covered.
        covered = np.zeros(len(observations), dtype=float)
        selected = []
        available = viable.copy()
        for _ in range(min(self.capacity, len(available))):
            residual = pair_activations[:, available] * (1.0 - covered[:, None])
            gains = residual.sum(axis=0)
            best = int(np.argmax(gains))
            if gains[best] <= 0.0:
                break
            index = int(available[best])
            selected.append(index)
            covered = np.maximum(covered, pair_activations[:, index])
            available = np.delete(available, best)
        self.created = len(selected)
        candidates = []
        for index in selected:
            first, second = pair_indices[index]
            conjunction = pair_activations[:, index]
            count = float(conjunction.sum())
            pain_sum = float(conjunction @ pain)
            valence = (
                pain_sum + self.base_rate * self.prior
            ) / (count + self.prior)
            candidates.append(MixedNode(
                first, second, valence, count,
                pain_sum / max(count, 1.0), pain_sum / max(count, 1.0) - self.base_rate,
            ))
        # Forgetting: remove redundant low-utility nodes.
        order = sorted(
            candidates,
            key=lambda node: abs(node.valence - self.base_rate) * node.support,
            reverse=True,
        )
        kept = []
        for node in order:
            a = pair_activations[:, pair_indices.index((node.first, node.second))]
            redundant = False
            for other in kept:
                b = pair_activations[:, pair_indices.index((other.first, other.second))]
                if float(np.minimum(a, b).sum()) > self.forgetting_overlap * min(a.sum(), b.sum()):
                    redundant = True
                    break
            if redundant:
                self.forgotten += 1
            else:
                kept.append(node)
        self.nodes = kept
        return self

    def risk(self, observations):
        observations = np.asarray(observations, dtype=float)
        if not self.nodes:
            return np.full(len(observations), self.base_rate)
        activations = self._gate_matrix(observations)
        risk = np.full(len(observations), self.base_rate, dtype=float)
        for node in self.nodes:
            conjunction = activations[:, node.first] * activations[:, node.second]
            risk += conjunction * (node.valence - self.base_rate)
        return np.clip(risk, 0.0, 1.0)

    def describe(self, names=None, limit=20):
        rows = []
        for node in sorted(self.nodes, key=lambda item: item.valence, reverse=True)[:limit]:
            rows.append(dict(
                first=self.gates[node.first].describe(names),
                second=self.gates[node.second].describe(names),
                valence=node.valence,
                support=node.support,
                pain_rate=node.pain_rate,
                lift=node.lift,
            ))
        return rows


class MixedNodeRiskMemory:
    """Risk readout from locally learned conjunctive pain nodes."""

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                 min_support=12.0, min_lift=0.02, min_pain=2.0,
                 max_nodes=64, prior=2.0, learning_rate=0.05):
        self.quantiles = tuple(quantiles)
        self.min_support = float(min_support)
        self.min_lift = float(min_lift)
        self.min_pain = float(min_pain)
        self.max_nodes = int(max_nodes)
        self.prior = float(prior)
        self.learning_rate = float(learning_rate)
        self.gates = []
        self.nodes = []
        self.base_rate = 0.0

    def _build_gates(self, observations):
        observations = np.asarray(observations, dtype=float)
        for channel in range(observations.shape[1]):
            values = observations[:, channel]
            spread = float(np.std(values))
            if spread <= 1e-9:
                continue
            thresholds = np.unique(np.quantile(values, self.quantiles))
            temperature = max(spread * 0.08, 1e-6)
            for threshold in thresholds:
                self.gates.append(PrimitiveGate(channel, "high", threshold, temperature))
                self.gates.append(PrimitiveGate(channel, "low", threshold, temperature))

    def _gate_matrix(self, observations):
        if not self.gates:
            return np.zeros((len(observations), 0))
        return np.column_stack([gate.activation(observations) for gate in self.gates])

    def fit(self, observations, pain):
        observations = np.asarray(observations, dtype=float)
        pain = np.asarray(pain, dtype=float).reshape(-1)
        if observations.ndim != 2 or len(observations) != len(pain):
            raise ValueError("observations and pain must have matching rows")
        self._build_gates(observations)
        activations = self._gate_matrix(observations)
        self.base_rate = float(pain.mean()) if len(pain) else 0.0
        candidates = []
        for first, second in combinations(range(len(self.gates)), 2):
            conjunction = activations[:, first] * activations[:, second]
            support = float(conjunction.sum())
            if support < self.min_support:
                continue
            pain_sum = float(conjunction @ pain)
            if pain_sum < self.min_pain:
                continue
            pain_rate = pain_sum / support
            lift = pain_rate - self.base_rate
            if lift < self.min_lift:
                continue
            score = lift * np.sqrt(support)
            candidates.append((score, first, second, support, pain_rate, lift))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _score, first, second, support, pain_rate, lift in candidates[:self.max_nodes]:
            valence = (
                pain_rate * support + self.base_rate * self.prior
            ) / (support + self.prior)
            self.nodes.append(MixedNode(
                first, second, valence, support, pain_rate, lift
            ))
        # Local pain memory: update only active mixed nodes.
        if self.nodes:
            for _ in range(3):
                for node in self.nodes:
                    conjunction = activations[:, node.first] * activations[:, node.second]
                    node.valence += (
                        self.learning_rate
                        * (pain - node.valence).dot(conjunction)
                        / max(conjunction.sum(), 1.0)
                    )
        return self

    def risk(self, observations):
        observations = np.asarray(observations, dtype=float)
        if not self.nodes:
            return np.full(len(observations), self.base_rate)
        activations = self._gate_matrix(observations)
        risk = np.full(len(observations), self.base_rate, dtype=float)
        for node in self.nodes:
            conjunction = activations[:, node.first] * activations[:, node.second]
            risk += conjunction * (node.valence - self.base_rate)
        return np.clip(risk, 0.0, 1.0)

    def describe(self, names=None, limit=20):
        rows = []
        for node in sorted(self.nodes, key=lambda item: item.valence, reverse=True)[:limit]:
            rows.append(dict(
                first=self.gates[node.first].describe(names),
                second=self.gates[node.second].describe(names),
                valence=node.valence,
                support=node.support,
                pain_rate=node.pain_rate,
                lift=node.lift,
            ))
        return rows
