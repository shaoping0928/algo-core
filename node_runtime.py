"""Runtime that learns from nodes instead of a hand-labeled risk target.

The only human-supplied inputs are:

    observation, action, valence (dopamine/pain), episode boundary

There is no `terminal` risk label, no gap/collision predicate, and no task
geometry.  The runtime decides which situations precede pain by assigning
observations to concept nodes and learning action-conditioned transition values
with temporal credit assignment.

The graph is used at runtime, not as an offline lookup table:

* observations are assigned to nodes (with online births for novel states);
* every transition creates an action edge ``(node, action) -> node``;
* node values learn discounted future pain/dopamine with TD(lambda) traces;
* action edges learn the one-step expected valence of taking an action;
* decisions compare the learned action-edge values plus an uncertainty bonus.

The implementation deliberately keeps a small, explicit interface.  It is a
prototype, not a replacement for the existing search baselines.
"""
from collections import deque
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from node_components import ComponentBank
from node_features import (
    FeatureRuleBank,
    feature_from_state,
    feature_state,
)
from node_graph import (
    ASSOCIATION,
    CONCEPT,
    MEMBERSHIP,
    MEMORY,
    TRANSITION_EDGE,
    NodeGraph,
)


class NodeRuntime:
    """Online concept graph with action-conditioned valence learning."""

    def __init__(
        self,
        obs_dim,
        action_space: Iterable[int] = (0, 1),
        lag: int = 0,
        gamma: float = 0.99,
        lam: float = 0.9,
        alpha_v: float = 0.05,
        alpha_q: float = 0.1,
        alpha_now: float = 0.2,
        alpha_mc: float = 0.2,
        birth_threshold: float = 0.45,
        max_nodes: int = 5000,
        exploration: float = 0.1,
        uncertainty_weight: float = 0.05,
        unknown_penalty: float = 0.5,
        soft_neighbors: int = 5,
        soft_sigma: float = 0.75,
        component_kinds: Sequence[str] = (),
        component_min_count: int = 8,
        max_components: int = 128,
        component_threshold: float = 0.2,
        component_smoothing: float = 0.5,
        component_weight: float = 0.5,
        component_gating: bool = False,
        graph_rate: float = 0.1,
        memory_decay: float = 0.9,
        memory_min_strength: float = 0.02,
        memory_weight: float = 0.1,
        association_rate: float = 0.0,
        valence_rate: float = 0.2,
        feature_rate: float = 0.05,
        seed: int = 0,
        scales: Optional[Sequence[float]] = None,
    ):
        if int(obs_dim) < 1:
            raise ValueError("obs_dim must be positive")
        if int(lag) < 0:
            raise ValueError("lag must be non-negative")
        if not 0.0 <= float(gamma) < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if not 0.0 <= float(lam) <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        if float(alpha_v) <= 0.0 or float(alpha_q) <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < float(alpha_now) <= 1.0:
            raise ValueError("alpha_now must be in (0, 1]")
        if not 0.0 < float(alpha_mc) <= 1.0:
            raise ValueError("alpha_mc must be in (0, 1]")
        if float(birth_threshold) <= 0.0:
            raise ValueError("birth_threshold must be positive")
        if int(max_nodes) < 1:
            raise ValueError("max_nodes must be positive")
        if not 0.0 <= float(exploration) <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        if float(uncertainty_weight) < 0.0:
            raise ValueError("uncertainty_weight must be non-negative")
        if float(unknown_penalty) < 0.0:
            raise ValueError("unknown_penalty must be non-negative")
        if int(soft_neighbors) < 1:
            raise ValueError("soft_neighbors must be positive")
        if float(soft_sigma) <= 0.0:
            raise ValueError("soft_sigma must be positive")
        if int(component_min_count) < 1:
            raise ValueError("component_min_count must be positive")
        if int(max_components) < 0:
            raise ValueError("max_components must be non-negative")
        if not 0.0 < float(component_smoothing) <= 1.0:
            raise ValueError("component_smoothing must be in (0, 1]")
        if float(component_weight) < 0.0:
            raise ValueError("component_weight must be non-negative")
        if not 0.0 < float(graph_rate) <= 1.0:
            raise ValueError("graph_rate must be in (0, 1]")
        if not 0.0 < float(memory_decay) <= 1.0:
            raise ValueError("memory_decay must be in (0, 1]")
        if float(memory_min_strength) < 0.0:
            raise ValueError("memory_min_strength must be non-negative")
        if not 0.0 <= float(memory_weight) <= 1.0:
            raise ValueError("memory_weight must be in [0, 1]")
        if not 0.0 <= float(association_rate) <= 1.0:
            raise ValueError("association_rate must be in [0, 1]")
        if not 0.0 < float(valence_rate) <= 1.0:
            raise ValueError("valence_rate must be in (0, 1]")
        if not 0.0 < float(feature_rate) <= 1.0:
            raise ValueError("feature_rate must be in (0, 1]")
        actions = tuple(int(action) for action in action_space)
        if not actions:
            raise ValueError("action_space must not be empty")

        self.obs_dim = int(obs_dim)
        self.action_space = actions
        self.lag = int(lag)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.alpha_v = float(alpha_v)
        self.alpha_q = float(alpha_q)
        self.alpha_now = float(alpha_now)
        self.alpha_mc = float(alpha_mc)
        self.birth_threshold = float(birth_threshold)
        self.max_nodes = int(max_nodes)
        self.exploration = float(exploration)
        self.uncertainty_weight = float(uncertainty_weight)
        self.unknown_penalty = float(unknown_penalty)
        self.soft_neighbors = int(soft_neighbors)
        self.soft_sigma = float(soft_sigma)
        self.component_kinds = tuple(str(kind) for kind in component_kinds)
        self.component_min_count = int(component_min_count)
        self.max_components = int(max_components)
        self.component_threshold = float(component_threshold)
        self.component_smoothing = float(component_smoothing)
        self.component_weight = float(component_weight)
        self.component_gating = bool(component_gating)
        self.graph_rate = float(graph_rate)
        self.memory_decay = float(memory_decay)
        self.memory_min_strength = float(memory_min_strength)
        self.memory_weight = float(memory_weight)
        self.association_rate = float(association_rate)
        self.valence_rate = float(valence_rate)
        self.feature_rate = float(feature_rate)
        self.gated_edges: Dict[
            Tuple[str, Tuple[int, ...], int], Dict[str, dict]
        ] = {}
        self.last_signature: Tuple[int, ...] = ()
        self.rng = np.random.RandomState(seed)

        state_dim = self.obs_dim + self.lag
        if scales is None:
            scales = np.ones(state_dim, dtype=float)
        scales = np.asarray(scales, dtype=float).reshape(-1)
        if scales.shape != (state_dim,):
            raise ValueError(
                "scales must have length obs_dim + lag (%d)" % state_dim
            )
        if not np.isfinite(scales).all() or (scales <= 0.0).any():
            raise ValueError("scales must be positive and finite")
        self.scales = scales

        self.graph = NodeGraph()
        self.components = ComponentBank(
            self.graph,
            threshold=self.component_threshold,
            smoothing=self.component_smoothing,
        )
        self.pain_node = self.graph.add_node(
            CONCEPT, "valence:pain", role="valence"
        )
        self.dopamine_node = self.graph.add_node(
            CONCEPT, "valence:dopamine", role="valence"
        )
        self.rules = FeatureRuleBank()
        self.feature_ids: List[str] = []
        self._feature_all: List[str] = []
        self._feature_objects: Dict[str, object] = {}
        self._feature_children: Dict[str, List[str]] = {}
        self._feature_keys: Dict[str, str] = {}
        self._feature_counter = 0
        self._prototype_ids: List[str] = []
        self._prototype_matrix = np.zeros((0, state_dim), dtype=float)
        self._traces: Dict[str, List[float]] = {}
        self.pending = self._zero_pending()
        self.last_node: Optional[str] = None
        self.last_action: Optional[int] = None
        self.last_state: Optional[np.ndarray] = None
        self.last_scores: Dict[int, dict] = {}
        self.last_sources: List[str] = []
        self._component_bound = set()
        self._node_pain_count = {}
        self._node_pain_mean = {}
        self._node_pain_m2 = {}
        self._memory_counter = 0

    # ---- encoding ----------------------------------------------------------
    def _zero_pending(self):
        if self.lag == 0:
            return deque(maxlen=0)
        return deque([0] * self.lag, maxlen=self.lag)

    def encode(self, observation) -> np.ndarray:
        observation = np.asarray(observation, dtype=float).reshape(-1)
        if observation.shape != (self.obs_dim,):
            raise ValueError(
                "expected observation with %d values" % self.obs_dim
            )
        if not np.isfinite(observation).all():
            raise ValueError("observation must be finite")
        if self.lag:
            state = np.concatenate(
                [observation, np.asarray(list(self.pending), dtype=float)]
            )
        else:
            state = observation
        return state / self.scales

    # ---- node assignment ---------------------------------------------------
    def _nearest(self, state: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._prototype_ids:
            return None, float("inf")
        distances = np.linalg.norm(self._prototype_matrix - state, axis=1)
        index = int(np.argmin(distances))
        return self._prototype_ids[index], float(distances[index])

    def _create_node(self, state: np.ndarray) -> str:
        node_id = self.graph.add_node(
            CONCEPT, features=np.asarray(state, dtype=float).copy()
        )
        self._prototype_ids.append(node_id)
        row = np.asarray(state, dtype=float).reshape(1, -1)
        if self._prototype_matrix.size == 0:
            self._prototype_matrix = row.copy()
        else:
            self._prototype_matrix = np.vstack([self._prototype_matrix, row])
        return node_id

    def _assign(self, state: np.ndarray, create: bool = True,
                count: bool = True) -> str:
        node_id, distance = self._nearest(state)
        if node_id is None:
            node_id = self._create_node(state)
        elif (create and len(self._prototype_ids) < self.max_nodes
              and distance > self.birth_threshold):
            node_id = self._create_node(state)
        if count:
            self.graph.nodes[node_id].count += 1
        return node_id

    def _note_pain(self, node_id: str, pain: float) -> None:
        pain = float(pain)
        self._node_pain_count[node_id] = (
            self._node_pain_count.get(node_id, 0) + 1
        )
        count = self._node_pain_count[node_id]
        mean = self._node_pain_mean.get(node_id, 0.0)
        delta = pain - mean
        mean += delta / count
        self._node_pain_mean[node_id] = mean
        self._node_pain_m2[node_id] = (
            self._node_pain_m2.get(node_id, 0.0)
            + delta * (pain - mean)
        )
        self._maybe_bind_components(node_id)

    def _maybe_bind_components(self, node_id: str) -> None:
        if not self.component_kinds or self.max_components == 0:
            return
        if node_id in self._component_bound:
            return
        count = self._node_pain_count.get(node_id, 0)
        if count < self.component_min_count:
            return
        if len(self.components.specs) >= self.max_components:
            return
        variance = self._node_pain_m2.get(node_id, 0.0) / max(
            count - 1, 1
        )
        variances = [
            self._node_pain_m2[other] / max(
                self._node_pain_count[other] - 1, 1
            )
            for other in self._node_pain_count
            if self._node_pain_count[other] >= self.component_min_count
        ]
        if len(variances) > 1:
            threshold = float(np.median(variances) + np.std(variances))
        else:
            threshold = float("inf")
        # The decision to instantiate a component is learned from the
        # node's valence variance relative to the population of visited nodes.
        if variance <= threshold:
            return
        for kind in self.component_kinds:
            if len(self.components.specs) >= self.max_components:
                break
            component_id = "%s:%s" % (node_id, kind)
            if component_id in self.components.specs:
                continue
            self.components.bind(
                node_id, kind=kind, component_id=component_id,
                threshold=self.component_threshold,
                smoothing=self.component_smoothing,
            )
        self._component_bound.add(node_id)

    # ---- runtime components ------------------------------------------------
    def bind_component(self, base_node: str, kind: str = "difference",
                       params: Sequence[float] = (),
                       component_id: Optional[str] = None,
                       threshold: Optional[float] = None,
                       hysteresis: Optional[float] = None,
                       smoothing: Optional[float] = None) -> str:
        """Bind a numeric component; its +/0/- states become concept nodes."""
        return self.components.bind(
            base_node, kind=kind, params=params,
            component_id=component_id, threshold=threshold,
            hysteresis=hysteresis, smoothing=smoothing,
        )

    def _component_step(self, node_id: str,
                        valence: Optional[np.ndarray] = None) -> None:
        if not self.components.specs:
            return
        self.components.step({node_id: 1.0})
        if valence is not None:
            self.components.observe_valence(
                float(valence[1]), float(valence[0])
            )

    def _new_memory(self) -> str:
        self._memory_counter += 1
        return self.graph.add_memory(
            [], "memory_runtime_%d" % self._memory_counter
        )

    def _interact(self, current: str, next_node: str, pain: float = 0.0,
                  dopamine: float = 0.0,
                  current_features: Iterable[str] = (),
                  next_features: Iterable[str] = ()) -> None:
        """Create association, causal transition and signal-driven memory."""
        if current == next_node:
            return
        current_features = list(dict.fromkeys(current_features))
        next_features = list(dict.fromkeys(next_features))
        self.graph.activate([current, next_node], [1.0, 1.0])
        self.graph.observe_association(
            [current, next_node], rate=self.graph_rate
        )
        edge = self.graph.observe_transition(
            current, next_node, rate=self.graph_rate
        )
        pain = float(pain)
        dopamine = float(dopamine)
        memories = self.memories_for(
            [current, next_node] + current_features + next_features
        )
        created_memory = None
        if pain > 0.0 and not memories:
            created_memory = self._new_memory()
            memory = self.graph.nodes[created_memory]
            memory.value = pain
            memory.dopamine = dopamine
            memory.count = 1
            self.graph.add_membership(created_memory, current)
            self.graph.add_membership(created_memory, next_node)
            if edge.edge_node is not None:
                self.graph.add_membership(
                    created_memory, edge.edge_node
                )
            for feature_id in dict.fromkeys(
                current_features + next_features
            ):
                self.graph.add_membership(created_memory, feature_id)
        self._reinforce_memories(
            [current, next_node] + current_features + next_features
        )
        extra = [current, next_node]
        if created_memory is not None:
            extra.append(created_memory)
        self._learn_pain_association(pain, extra)

    def memories_for(self, node_ids: Iterable[str]) -> Dict[str, float]:
        """Find memory nodes through membership edges of active nodes."""
        result: Dict[str, float] = {}
        for node_id in dict.fromkeys(node_ids):
            for key in self.graph.incoming.get(node_id, []):
                if key[2] != MEMBERSHIP:
                    continue
                edge = self.graph.edges[key]
                result[key[0]] = max(
                    result.get(key[0], 0.0), float(edge.weight)
                )
            # Associations are one-hop retrieval paths to memories of
            # related nodes; membership still carries the strength.
            for key in self.graph.outgoing.get(node_id, []):
                if key[2] != ASSOCIATION:
                    continue
                edge = self.graph.edges[key]
                other = (
                    edge.target if edge.source == node_id
                    else edge.source
                )
                for member_key in self.graph.incoming.get(other, []):
                    if member_key[2] != MEMBERSHIP:
                        continue
                    member_edge = self.graph.edges[member_key]
                    result[member_key[0]] = max(
                        result.get(member_key[0], 0.0),
                        float(member_edge.weight),
                    )
        return result

    def _reinforce_memories(self, node_ids: Iterable[str]) -> None:
        active = set(node_ids)
        for memory_id in self.memories_for(node_ids):
            memory = self.graph.nodes.get(memory_id)
            if memory is None or memory.kind != MEMORY:
                continue
            self.graph.activate([memory_id], [1.0])
            for key in self.graph.outgoing.get(memory_id, []):
                if key[2] != MEMBERSHIP:
                    continue
                edge = self.graph.edges[key]
                if key[1] not in active:
                    continue
                edge.weight += self.graph_rate * (1.0 - edge.weight)
                edge.eligibility = (
                    (1.0 - self.graph_rate) * edge.eligibility
                    + self.graph_rate
                )

    def _propagate_association_values(self) -> None:
        """Blend concept values along learned association weights."""
        if self.association_rate <= 0.0:
            return
        concept_ids = {
            node_id for node_id, node in self.graph.nodes.items()
            if node.kind == CONCEPT and node.role != "component_state"
        }
        if not concept_ids:
            return
        values = {
            node_id: self.graph.nodes[node_id].value
            for node_id in concept_ids
        }
        delta = {node_id: 0.0 for node_id in concept_ids}
        for edge in self.graph.edges.values():
            if edge.kind != ASSOCIATION:
                continue
            if edge.source not in values or edge.target not in values:
                continue
            weight = float(edge.weight)
            if weight <= 0.0:
                continue
            delta[edge.source] += weight * (
                values[edge.target] - values[edge.source]
            )
            delta[edge.target] += weight * (
                values[edge.source] - values[edge.target]
            )
        for node_id, change in delta.items():
            self.graph.nodes[node_id].value += (
                self.association_rate * change
            )

    def _forget_memories(self) -> None:
        memories = [
            memory for memory in self.graph.nodes.values()
            if memory.kind == MEMORY
        ]
        if memories:
            median_value = float(np.median([
                memory.value for memory in memories
            ]))
        else:
            median_value = 0.0
        for memory in memories:
            memory.trace *= self.memory_decay
            for key in self.graph.outgoing.get(memory.id, []):
                if key[2] == MEMBERSHIP:
                    self.graph.edges[key].weight *= self.memory_decay
            if (memory.trace < self.memory_min_strength
                    and memory.count <= 1
                    and (memory.value < median_value
                         or memory.value <= 0.0)):
                self.graph._remove_node(memory.id)

    def _prune_components(self) -> None:
        if len(self.components.specs) < 2:
            return
        removed = self.components.prune()
        if removed:
            self.gated_edges.clear()
            for component_id in removed:
                self._component_bound.discard(
                    component_id.rsplit(":", 1)[0]
                )

    def _learn_pain_association(self, pain: float,
                                extra_nodes: Iterable[str] = ()) -> None:
        pain = float(pain)
        if pain <= 0.0:
            return
        self.graph.activate([self.pain_node], [pain])
        nodes = list(dict.fromkeys(list(extra_nodes)))
        for node_id in nodes:
            if node_id == self.pain_node:
                continue
            self.graph.observe_association(
                [node_id, self.pain_node], rate=self.valence_rate
            )

    def _pain_association(self, node_id: str) -> float:
        key = (
            min(node_id, self.pain_node),
            max(node_id, self.pain_node),
            ASSOCIATION,
        )
        edge = self.graph.edges.get(key)
        return float(edge.weight) if edge is not None else 0.0

    def pain_signal_from_weights(self, weights) -> float:
        total = 0.0
        weighted = 0.0
        node_ids = []
        for node_id, weight in weights:
            node_ids.append(node_id)
            total += float(weight)
            weighted += float(weight) * self._pain_association(node_id)
        for memory_id, strength in self.memories_for(node_ids).items():
            total += float(strength)
            weighted += float(strength) * self._pain_association(memory_id)
        if total <= 0.0:
            return 0.0
        return weighted / total

    def pain_signal_state(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        learned = self.pain_signal_from_weights(
            self._neighbor_weights(state)
        )
        row = state.reshape(1, -1)
        if self.feature_ids:
            learned = max(
                learned,
                float(self._feature_risk_batch(row)[0]),
                float(self._feature_memory_risk_batch(row)[0]),
            )
        elif self.rules.rules:
            learned = max(
                learned,
                float(self.rules.activate(row[:, :self.obs_dim])[0]),
            )
        return learned

    def pain_signal_batch(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0)
        result = np.zeros(len(states), dtype=float)
        indices, weights = self._soft_weights_batch(states)
        if indices is not None:
            for row in range(len(states)):
                total = 0.0
                weighted = 0.0
                node_ids = []
                for column, index in enumerate(indices[row]):
                    node_id = self._prototype_ids[int(index)]
                    node_weight = float(weights[row, column])
                    node_ids.append(node_id)
                    total += node_weight
                    weighted += node_weight * self._pain_association(node_id)
                for memory_id, strength in self.memories_for(
                    node_ids
                ).items():
                    total += float(strength)
                    weighted += float(strength) * self._pain_association(
                        memory_id
                    )
                result[row] = weighted / total if total > 0.0 else 0.0
        if self.feature_ids:
            result = np.maximum(result, self._feature_risk_batch(states))
            result = np.maximum(
                result, self._feature_memory_risk_batch(states)
            )
        elif self.rules.rules:
            rule_risk = self.rules.activate(states[:, :self.obs_dim])
            result = np.maximum(result, rule_risk)
        return result

    def _signature(self, node_id: str) -> Tuple[int, ...]:        return tuple(
            self.components.previous_sign[component_id]
            for component_id, spec in self.components.specs.items()
            if spec.base_node == node_id
        )

    def _update_gated(self, node_id: str, signature: Tuple[int, ...],
                      action: int, next_node: str, target_pain: float,
                      target_dopamine: float) -> None:
        if not self.component_gating or not signature:
            return
        targets = self.gated_edges.setdefault(
            (node_id, signature, int(action)), {}
        )
        edge = targets.setdefault(
            next_node, dict(count=0, q_pain=0.0, q_dopamine=0.0)
        )
        edge["count"] += 1
        edge["q_pain"] += self.alpha_q * (
            target_pain - edge["q_pain"]
        )
        edge["q_dopamine"] += self.alpha_q * (
            target_dopamine - edge["q_dopamine"]
        )

    # ---- action values -----------------------------------------------------
    def _neighbor_weights(self, state: np.ndarray):
        """Distance-weighted node mixture used for decisions.

        A hard nearest-node lookup leaves novel states with no outgoing edges.
        Soft neighbours let runtime decisions inherit values from nearby learned
        nodes while keeping the training update tied to the best-matching node.
        """
        if not self._prototype_ids:
            return []
        distances = np.linalg.norm(self._prototype_matrix - state, axis=1)
        count = min(self.soft_neighbors, len(distances))
        order = np.argsort(distances)[:count]
        weights = np.exp(
            -0.5 * (distances[order] / self.soft_sigma) ** 2
        )
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return []
        return [
            (self._prototype_ids[int(index)], float(weight / total))
            for index, weight in zip(order, weights)
        ]

    def _score_actions(self, node_id: str,
                       state: Optional[np.ndarray] = None,
                       sources: Optional[Iterable[str]] = None,
                       signature: Tuple[int, ...] = ()) -> Dict[int, dict]:
        node = self.graph.nodes[node_id]
        if state is None:
            state = (
                node.features if node.features is not None
                else np.zeros(len(self.scales), dtype=float)
            )
        neighbors = self._neighbor_weights(state)
        source_ids = list(dict.fromkeys(sources or ()))
        scores: Dict[int, dict] = {}
        for action in self.action_space:
            gated = (
                self.gated_edges.get((node_id, signature, action))
                if self.component_gating and signature else None
            )
            if gated:
                total = float(sum(item["count"] for item in gated.values()))
                if total > 0.0:
                    pain = sum(
                        item["count"] * item["q_pain"]
                        for item in gated.values()
                    ) / total
                    dopamine = sum(
                        item["count"] * item["q_dopamine"]
                        for item in gated.values()
                    ) / total
                    uncertainty = sum(
                        item["count"] * (1.0 / np.sqrt(item["count"] + 1.0))
                        for item in gated.values()
                    ) / total
                    scores[action] = dict(
                        pain=pain,
                        dopamine=dopamine,
                        uncertainty=uncertainty,
                        count=int(total),
                        cost=pain + self.uncertainty_weight * uncertainty,
                    )
                    continue
            support = 0.0
            pain_sum = 0.0
            dopamine_sum = 0.0
            uncertainty_sum = 0.0
            visits = 0
            for neighbor_id, weight in neighbors:
                for edge in self.graph.action_edges_from(neighbor_id, action):
                    mass = weight * edge.count
                    support += mass
                    pain_sum += mass * edge.q_pain
                    dopamine_sum += mass * edge.q_dopamine
                    uncertainty_sum += mass * (
                        1.0 / np.sqrt(edge.count + 1.0)
                    )
                    visits += edge.count
            for source in source_ids:
                for edge in self.graph.action_edges_from(source, action):
                    mass = self.component_weight * edge.count
                    support += mass
                    pain_sum += mass * edge.q_pain
                    dopamine_sum += mass * edge.q_dopamine
                    uncertainty_sum += mass * (
                        1.0 / np.sqrt(edge.count + 1.0)
                    )
                    visits += edge.count
            if support <= 0.0:
                # Unknown actions are optimistic: preserve the node's current
                # expected pain and add a fixed uncertainty penalty.
                prior_pain = node.value if node.count > 1 else 0.5
                prior_dopamine = node.dopamine if node.count > 1 else 0.0
                scores[action] = dict(
                    pain=prior_pain,
                    dopamine=prior_dopamine,
                    uncertainty=1.0,
                    count=0,
                    cost=prior_pain + self.unknown_penalty,
                )
                continue
            pain = pain_sum / support
            dopamine = dopamine_sum / support
            uncertainty = uncertainty_sum / support
            scores[action] = dict(
                pain=pain,
                dopamine=dopamine,
                uncertainty=uncertainty,
                count=int(visits),
                cost=pain + self.uncertainty_weight * uncertainty,
            )
        return scores

    def decision(self, observation) -> Tuple[str, Dict[int, dict]]:
        """Score actions for an observation without changing the runtime state."""
        state = self.encode(observation)
        node_id = self._assign(state, create=False, count=False)
        sources = self.components.active_sign_nodes(node_id)
        sources = sources + self._active_feature_roots(state)
        signature = self._signature(node_id)
        return node_id, self._score_actions(
            node_id, state, sources=sources, signature=signature
        )

    def act(self, observation, explore: bool = True) -> int:
        state = self.encode(observation)
        node_id = self._assign(state, create=True, count=True)
        self._component_step(node_id)
        sources = self.components.active_sign_nodes(node_id)
        sources = sources + self._active_feature_roots(state)
        signature = self._signature(node_id)
        scores = self._score_actions(
            node_id, state, sources=sources, signature=signature
        )
        if explore and self.rng.rand() < self.exploration:
            action = int(self.rng.choice(self.action_space))
        else:
            action = min(
                scores,
                key=lambda item: (
                    scores[item]["cost"],
                    -scores[item]["dopamine"],
                    item,
                ),
            )
        self.last_node = node_id
        self.last_action = action
        self.last_state = state
        self.last_scores = scores
        self.last_sources = [node_id] + sources
        self.last_signature = signature
        self.pending.append(action)
        return action

    # ---- learning ----------------------------------------------------------
    def _update_transition_value(self, current: str, next_node: str,
                                 target_pain: float,
                                 target_dopamine: float) -> None:
        edge_node = self.graph.edge_node(current, next_node)
        if edge_node is None:
            return
        node = self.graph.nodes[edge_node]
        node.value += self.alpha_q * (target_pain - node.value)
        node.dopamine += self.alpha_q * (
            target_dopamine - node.dopamine
        )

    def _update_action_edges(self, sources: Iterable[str], action: int,
                             next_node: str, target_pain: float,
                             target_dopamine: float) -> None:
        for source in dict.fromkeys(sources):
            edge = self.graph.add_action_edge(source, action, next_node)
            edge.prediction_error = target_pain - edge.q_pain
            edge.count += 1
            edge.q_pain += self.alpha_q * (target_pain - edge.q_pain)
            edge.q_dopamine += (
                self.alpha_q * (target_dopamine - edge.q_dopamine)
            )

    def _update(self, current: str, action: int, next_node: str,
                target_pain: float, target_dopamine: float,
                immediate_pain: float, immediate_dopamine: float,
                sources: Optional[Iterable[str]] = None) -> None:
        self._update_action_edges(
            sources or (current,), action, next_node,
            target_pain, target_dopamine,
        )
        self._update_transition_value(
            current, next_node, target_pain, target_dopamine
        )

        node = self.graph.nodes[current]
        delta_pain = target_pain - node.value
        delta_dopamine = target_dopamine - node.dopamine

        # The world's valence belongs to the state the transition entered.  Keep
        # an immediate component so a terminal endpoint is not scored as V=0.
        entered = self.graph.nodes[next_node]
        entered.immediate_pain += self.alpha_now * (
            immediate_pain - entered.immediate_pain
        )
        entered.immediate_dopamine += self.alpha_now * (
            immediate_dopamine - entered.immediate_dopamine
        )
        self._note_pain(next_node, immediate_pain)

        decay = self.gamma * self.lam
        for node_id, trace in list(self._traces.items()):
            trace[0] *= decay
            trace[1] *= decay
            if abs(trace[0]) < 1e-8 and abs(trace[1]) < 1e-8:
                del self._traces[node_id]
        trace = self._traces.setdefault(current, [0.0, 0.0])
        trace[0] += 1.0
        trace[1] += 1.0

        for node_id, (pain_trace, dopamine_trace) in self._traces.items():
            traced = self.graph.nodes[node_id]
            traced.value += self.alpha_v * delta_pain * pain_trace
            traced.dopamine += (
                self.alpha_v * delta_dopamine * dopamine_trace
            )

    def observe(self, observation, action: int, valence,
                done: bool = False, learn: bool = True) -> None:
        """Consume the real transition caused by the last `act`.

        `valence` is ``[dopamine, pain]`` from the world.  `done` only marks the
        episode boundary for bootstrapping and reset; it is not a risk label.
        """
        if self.last_node is None:
            # No state/action pair has been recorded yet.
            if done:
                self.reset()
            return
        valence = np.asarray(valence, dtype=float).reshape(-1)
        if valence.shape != (2,) or not np.isfinite(valence).all():
            raise ValueError("valence must contain finite dopamine and pain")
        if not learn:
            if done:
                self.reset()
            return

        next_state = self.encode(observation)
        next_node = self._assign(next_state, create=True, count=True)
        current_active = (
            self._active_feature_ids(self.last_state)
            if self.last_state is not None else []
        )
        current_features = [
            node_id for node_id in current_active
            if node_id in self.feature_ids
        ]
        next_active = self._active_feature_ids(next_state)
        next_features = [
            node_id for node_id in next_active
            if node_id in self.feature_ids
        ]
        self._activate_features(next_active)
        self._interact(
            self.last_node, next_node,
            float(valence[1]), float(valence[0]),
            current_features=current_features,
            next_features=next_features,
        )
        self._update_feature_pain(next_active, float(valence[1]))
        pain = float(valence[1])
        dopamine = float(valence[0])
        next_value = self.graph.nodes[next_node]
        target_pain = pain + (0.0 if done else self.gamma * next_value.value)
        target_dopamine = dopamine + (
            0.0 if done else self.gamma * next_value.dopamine
        )
        self._update(
            self.last_node, int(action), next_node,
            target_pain, target_dopamine, pain, dopamine,
            sources=self.last_sources,
        )
        self._update_gated(
            self.last_node, self.last_signature, int(action), next_node,
            target_pain, target_dopamine,
        )
        self._component_step(next_node, valence)
        self.last_node = None
        self.last_action = None
        self.last_state = None
        self.last_sources = []
        self.last_signature = ()
        if done:
            self.reset()

    def reset(self) -> None:
        """Reset episode-local state; node values and edges are retained."""
        self.pending = self._zero_pending()
        self.last_node = None
        self.last_action = None
        self.last_state = None
        self.last_scores = {}
        self.last_sources = []
        self.last_signature = ()
        self._traces.clear()

    def learn_transition(self, observation, action, next_observation,
                         valence, done: bool = False) -> None:
        """Learn one real transition without choosing the action first.

        This is the offline-replay entry point.  `done` only truncates
        bootstrapping and resets the episode; it is not a risk label.
        """
        action = int(action)
        if action not in self.action_space:
            raise ValueError("action is outside the action space")
        valence = np.asarray(valence, dtype=float).reshape(-1)
        if valence.shape != (2,) or not np.isfinite(valence).all():
            raise ValueError("valence must contain finite dopamine and pain")
        state = self.encode(observation)
        current = self._assign(state, create=True, count=True)
        self._component_step(current)
        current_active = self._active_feature_ids(state)
        self._activate_features(current_active)
        current_features = [
            node_id for node_id in current_active
            if node_id in self.feature_ids
        ]
        sources = (
            [current]
            + self.components.active_sign_nodes(current)
            + current_features
        )
        signature = self._signature(current)
        self.pending.append(action)
        next_state = self.encode(next_observation)
        next_node = self._assign(next_state, create=True, count=True)
        next_active = self._active_feature_ids(next_state)
        self._activate_features(next_active)
        next_features = [
            node_id for node_id in next_active
            if node_id in self.feature_ids
        ]
        self._interact(
            current, next_node, float(valence[1]), float(valence[0]),
            current_features=current_features,
            next_features=next_features,
        )
        self._update_feature_pain(next_active, float(valence[1]))
        pain = float(valence[1])
        dopamine = float(valence[0])
        next_value = self.graph.nodes[next_node]
        target_pain = pain + (0.0 if done else self.gamma * next_value.value)
        target_dopamine = dopamine + (
            0.0 if done else self.gamma * next_value.dopamine
        )
        self._update(
            current, action, next_node,
            target_pain, target_dopamine, pain, dopamine,
            sources=sources,
        )
        self._component_step(next_node, valence)
        if done:
            self.reset()

    def learn_episode(self, rows) -> None:
        """Monte-Carlo node/edge values for one real episode.

        `rows` are transition dicts with ``obs``, ``next_obs``, ``history``,
        ``signals`` and ``done`` (an episode boundary, not a risk label).
        Values are accumulated backward from the world's pain signal.
        """
        rows = list(rows)
        if not rows:
            return
        self.reset()
        steps = []
        for row in rows:
            action = int(row["history"][0])
            if action not in self.action_space:
                raise ValueError("action is outside the action space")
            signals = np.asarray(row["signals"], dtype=float).reshape(-1)
            if signals.shape != (2,) or not np.isfinite(signals).all():
                raise ValueError("signals must contain finite valence")
            done = bool(row["done"])
            state = self.encode(row["obs"])
            current = self._assign(state, create=True, count=True)
            self._component_step(current)
            current_active = self._active_feature_ids(state)
            self._activate_features(current_active)
            current_features = [
                node_id for node_id in current_active
                if node_id in self.feature_ids
            ]
            sources = (
                [current]
                + self.components.active_sign_nodes(current)
                + current_features
            )
            signature = self._signature(current)
            self.pending.append(action)
            next_state = self.encode(row["next_obs"])
            next_node = self._assign(next_state, create=True, count=True)
            next_active = self._active_feature_ids(next_state)
            self._activate_features(next_active)
            next_features = [
                node_id for node_id in next_active
                if node_id in self.feature_ids
            ]
            self._interact(
                current, next_node,
                float(signals[1]), float(signals[0]),
                current_features=current_features,
                next_features=next_features,
            )
            self._update_feature_pain(next_active, float(signals[1]))
            entered = self.graph.nodes[next_node]
            entered.immediate_pain += self.alpha_now * (
                signals[1] - entered.immediate_pain
            )
            entered.immediate_dopamine += self.alpha_now * (
                signals[0] - entered.immediate_dopamine
            )
            self._note_pain(next_node, float(signals[1]))
            self._component_step(next_node, signals)
            steps.append((
                current, action, next_node, float(signals[1]),
                float(signals[0]), done, sources, signature,
            ))
            if done:
                self.reset()

        running_pain = 0.0
        running_dopamine = 0.0
        for (current, action, next_node, pain, dopamine,
             terminal, sources, signature) in reversed(steps):
            return_pain = pain + (
                0.0 if terminal else self.gamma * running_pain
            )
            return_dopamine = dopamine + (
                0.0 if terminal else self.gamma * running_dopamine
            )
            node = self.graph.nodes[current]
            node.value += self.alpha_mc * (
                return_pain - node.immediate_pain - node.value
            )
            node.dopamine += self.alpha_mc * (
                return_dopamine - node.immediate_dopamine - node.dopamine
            )
            self._update_action_edges(
                sources, action, next_node, return_pain, return_dopamine
            )
            self._update_transition_value(
                current, next_node, return_pain, return_dopamine
            )
            self._update_gated(
                current, signature, action, next_node,
                return_pain, return_dopamine,
            )
            if terminal:
                entered = self.graph.nodes[next_node]
                entered.value += self.alpha_mc * (
                    pain - entered.immediate_pain - entered.value
                )
            running_pain = return_pain
            running_dopamine = return_dopamine
        self._propagate_association_values()
        self._forget_memories()
        self._prune_components()
        self.reset()

    def induce_rules(self, train_states, train_pain,
                     test_states=None, test_pain=None):
        """Induce compositional pain rules over encoded observation states.

        Only the observation part is used: the rule is a property of the
        entered state, which is what the sandbox predicts.  Lag/action
        coordinates remain available to the node graph.
        """
        train_states = np.asarray(train_states, dtype=float)
        if train_states.ndim != 2 or train_states.shape[1] < self.obs_dim:
            raise ValueError(
                "states must have at least obs_dim columns"
            )
        test_obs = None
        if test_states is not None:
            test_states = np.asarray(test_states, dtype=float)
            if test_states.ndim != 2 or test_states.shape[1] < self.obs_dim:
                raise ValueError(
                    "test states must have at least obs_dim columns"
                )
            test_obs = test_states[:, :self.obs_dim]
        return self.rules.induce(
            train_states[:, :self.obs_dim], train_pain,
            test_obs, test_pain,
        )

    def _materialize_feature(self, feature, role: str) -> str:
        spec = feature_state(feature)
        key = json.dumps(spec, sort_keys=True)
        existing = self._feature_keys.get(key)
        if existing is not None:
            return existing
        self._feature_counter += 1
        node_id = self.graph.add_node(
            CONCEPT, "feature_%d" % self._feature_counter,
            role=role, predicate=spec,
        )
        node = self.graph.nodes[node_id]
        node.beta = self.graph_rate
        self._feature_all.append(node_id)
        self._feature_objects[node_id] = feature
        self._feature_keys[key] = node_id
        children = getattr(feature, "children", None)
        if children:
            child_ids = []
            for child in children:
                child_role = (
                    "composite" if getattr(child, "children", None)
                    else "primitive"
                )
                child_id = self._materialize_feature(child, child_role)
                self.graph.add_derivation(node_id, child_id)
                child_ids.append(child_id)
            self._feature_children[node_id] = child_ids
        return node_id

    def install_feature_rules(self, rules=None) -> List[str]:
        """Materialize induced rules as a shared feature graph.

        Primitive and composite nodes are shared by every rule that uses the
        same sub-feature; derivation edges point from a composite to its
        children.  Root nodes keep the learned P(pain | feature) in ``value``.
        """
        rules = self.rules.rules if rules is None else list(rules)
        for rule in rules:
            node_id = self._materialize_feature(rule.feature, "feature")
            node = self.graph.nodes[node_id]
            node.value = max(float(node.value), float(rule.confidence))
            node.count = max(node.count, int(rule.support))
            if node_id not in self.feature_ids:
                self.feature_ids.append(node_id)
        return list(self.feature_ids)

    def _active_feature_ids(self, state) -> List[str]:
        if not self._feature_all:
            return []
        obs = np.asarray(state, dtype=float).reshape(-1)[:self.obs_dim]
        row = obs.reshape(1, -1)
        return [
            node_id for node_id in self._feature_all
            if node_id in self._feature_objects
            and bool(self._feature_objects[node_id].mask(row)[0])
        ]

    def _active_feature_roots(self, state) -> List[str]:
        active = set(self._active_feature_ids(state))
        return [
            node_id for node_id in self.feature_ids if node_id in active
        ]

    def _update_feature_pain(self, feature_ids, pain: float) -> None:
        pain = float(pain)
        for node_id in feature_ids:
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            node.value += self.feature_rate * (pain - node.value)
            node.count += 1

    def _activate_features(self, feature_ids) -> None:
        active = set(feature_ids)
        for node_id in self._feature_all:
            node = self.graph.nodes.get(node_id)
            if node is not None:
                node.activate(1.0 if node_id in active else 0.0)

    def _feature_risk_batch(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        result = np.zeros(len(states), dtype=float)
        if not self.feature_ids:
            return result
        obs = states[:, :self.obs_dim]
        for node_id in self.feature_ids:
            feature = self._feature_objects.get(node_id)
            node = self.graph.nodes.get(node_id)
            if feature is None or node is None:
                continue
            result = np.maximum(
                result, float(node.value) * feature.mask(obs)
            )
        return result

    def _feature_memory_risk_batch(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        result = np.zeros(len(states), dtype=float)
        if not self.feature_ids:
            return result
        for row, state in enumerate(states):
            feature_ids = self._active_feature_roots(state)
            if not feature_ids:
                continue
            total = 0.0
            weighted = 0.0
            for memory_id, strength in self.memories_for(
                feature_ids
            ).items():
                memory = self.graph.nodes.get(memory_id)
                if memory is None or memory.kind != MEMORY:
                    continue
                total += float(strength)
                weighted += float(strength) * (
                    memory.value + memory.immediate_pain
                )
            if total > 0.0:
                result[row] = weighted / total
        return result

    # ---- analysis ----------------------------------------------------------
    def _soft_weights_batch(self, states: np.ndarray):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.scales):
            raise ValueError("states must be a matrix with the runtime width")
        if not self._prototype_ids:
            return None, None
        # Squared distances via matrix multiplication; this bounds memory to
        # (states x prototypes) instead of materialising states x prototypes x dim.
        prototypes = self._prototype_matrix
        state_norm = np.einsum("ij,ij->i", states, states)[:, None]
        proto_norm = np.einsum("ij,ij->i", prototypes, prototypes)[None, :]
        distances = (
            state_norm + proto_norm - 2.0 * (states @ prototypes.T)
        )
        np.maximum(distances, 0.0, out=distances)
        count = min(self.soft_neighbors, distances.shape[1])
        indices = np.argpartition(distances, count - 1, axis=1)[:, :count]
        nearest = np.take_along_axis(distances, indices, axis=1)
        weights = np.exp(
            -0.5 * nearest / (self.soft_sigma * self.soft_sigma)
        )
        total = weights.sum(axis=1, keepdims=True)
        weights = weights / np.where(total > 0.0, total, 1.0)
        return indices, weights

    def _memory_values_batch(self, indices, weights, attribute: str):
        count = len(indices)
        sums = np.zeros(count, dtype=float)
        mass = np.zeros(count, dtype=float)
        for row in range(count):
            for column, index in enumerate(indices[row]):
                node_id = self._prototype_ids[int(index)]
                for memory_id, strength in self.memories_for(
                    [node_id]
                ).items():
                    memory = self.graph.nodes.get(memory_id)
                    if memory is None or memory.kind != MEMORY:
                        continue
                    weight = float(weights[row, column]) * float(strength)
                    if weight <= 0.0:
                        continue
                    sums[row] += weight * getattr(memory, attribute)
                    mass[row] += weight
        return sums, mass

    def valence_state_batch(self, states: np.ndarray):
        """Soft-neighbour pain/dopamine values for encoded states."""
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0), np.zeros(0)
        indices, weights = self._soft_weights_batch(states)
        if indices is None:
            length = len(np.asarray(states))
            return np.zeros(length), np.zeros(length)
        nodes = [
            self.graph.nodes[self._prototype_ids[int(index)]]
            for index in indices.ravel()
        ]
        pain_values = np.asarray([
            node.value + node.immediate_pain for node in nodes
        ]).reshape(indices.shape)
        dopamine_values = np.asarray([
            node.dopamine + node.immediate_dopamine for node in nodes
        ]).reshape(indices.shape)
        pain = (weights * pain_values).sum(axis=1)
        dopamine = (weights * dopamine_values).sum(axis=1)
        for attribute, values in (
            ("value", pain), ("dopamine", dopamine)
        ):
            memory_sums, memory_mass = self._memory_values_batch(
                indices, weights, attribute
            )
            present = memory_mass > 0.0
            if present.any():
                values[present] = (
                    (1.0 - self.memory_weight) * values[present]
                    + self.memory_weight
                    * memory_sums[present] / memory_mass[present]
                )
        return pain, dopamine

    def _memory_value_from_weights(self, weights, attribute: str = "value"):
        total = 0.0
        weighted = 0.0
        for node_id, weight in weights:
            for memory_id, strength in self.memories_for([node_id]).items():
                mass = float(weight) * float(strength)
                if mass <= 0.0:
                    continue
                memory = self.graph.nodes.get(memory_id)
                if memory is None or memory.kind != MEMORY:
                    continue
                total += mass
                weighted += mass * getattr(memory, attribute)
        if total <= 0.0:
            return None
        return weighted / total

    def immediate_risk_batch(self, states: np.ndarray) -> np.ndarray:
        """Weighted immediate pain of soft-neighbour states."""
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0)
        indices, weights = self._soft_weights_batch(states)
        if indices is None:
            return np.zeros(len(states))
        values = np.asarray([
            self.graph.nodes[self._prototype_ids[int(index)]].immediate_pain
            for index in indices.ravel()
        ]).reshape(indices.shape)
        return (weights * values).sum(axis=1)

    def value_state(self, state: np.ndarray) -> float:
        weights = self._neighbor_weights(np.asarray(state, dtype=float))
        if not weights:
            return 0.0
        value = float(sum(
            weight * (
                self.graph.nodes[node_id].value
                + self.graph.nodes[node_id].immediate_pain
            )
            for node_id, weight in weights
        ))
        memory_value = self._memory_value_from_weights(weights, "value")
        if memory_value is not None:
            value = (
                (1.0 - self.memory_weight) * value
                + self.memory_weight * memory_value
            )
        return value

    def dopamine_state(self, state: np.ndarray) -> float:
        weights = self._neighbor_weights(np.asarray(state, dtype=float))
        return float(sum(
            weight * (
                self.graph.nodes[node_id].dopamine
                + self.graph.nodes[node_id].immediate_dopamine
            )
            for node_id, weight in weights
        )) if weights else 0.0

    def node_value(self, observation) -> float:
        state = self.encode(observation)
        node_id, _distance = self._nearest(state)
        if node_id is None:
            return 0.0
        node = self.graph.nodes[node_id]
        return float(node.value + node.immediate_pain)

    def node_dopamine(self, observation) -> float:
        state = self.encode(observation)
        node_id, _distance = self._nearest(state)
        if node_id is None:
            return 0.0
        node = self.graph.nodes[node_id]
        return float(node.dopamine + node.immediate_dopamine)

    def top_nodes(self, limit: int = 10) -> List[dict]:
        nodes = sorted(
            self.graph.nodes.values(),
            key=lambda node: node.value + node.immediate_pain,
            reverse=True,
        )
        return [
            dict(
                id=node.id,
                count=node.count,
                pain=node.value + node.immediate_pain,
                dopamine=node.dopamine + node.immediate_dopamine,
                value=node.value,
                pain_now=node.immediate_pain,
                dopamine_now=node.immediate_dopamine,
                features=(
                    None if node.features is None
                    else [float(value) for value in node.features]
                ),
            )
            for node in nodes[:limit]
        ]

    def top_edges(self, limit: int = 10) -> List[dict]:
        edges = sorted(
            self.graph.action_edges.values(),
            key=lambda edge: edge.q_pain,
            reverse=True,
        )
        return [
            dict(
                source=edge.source,
                action=edge.action,
                target=edge.target,
                count=edge.count,
                pain=edge.q_pain,
                dopamine=edge.q_dopamine,
            )
            for edge in edges[:limit]
        ]

    def snapshot(self) -> dict:
        return dict(
            nodes=len(self.graph.nodes),
            action_edges=len(self.graph.action_edges),
            associations=sum(
                edge.kind == ASSOCIATION
                for edge in self.graph.edges.values()
            ),
            transitions=sum(
                edge.kind == TRANSITION_EDGE
                for edge in self.graph.edges.values()
            ),
            memories=sum(
                node.kind == MEMORY
                for node in self.graph.nodes.values()
            ),
            prototypes=len(self._prototype_ids),
            rules=len(self.rules.rules),
            features=len(self.feature_ids),
            feature_nodes=len(self._feature_all),
            components=len(self.components.specs),
            component_states=sum(
                node.role == "component_state"
                for node in self.graph.nodes.values()
            ),
        )
