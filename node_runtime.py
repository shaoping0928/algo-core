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


def future_pain_labels(rows, window: int) -> List[float]:
    """Precursor pain labels: ``P_k(t) = max pain over the next k states``.

    `rows` are transition dicts ordered in time (``signals[1]`` is the pain of
    the *entered* state, ``done`` marks an episode boundary).  With
    ``window = k`` the label of row ``t`` is the largest pain observed at
    states ``t+1 .. t+k``, truncated at the first terminal step or episode
    end.  With ``window = 1`` this is the plain entered-state pain.
    """
    rows = list(rows)
    pains = []
    dones = []
    episodes = []
    for row in rows:
        signals = np.asarray(row["signals"], dtype=float).reshape(-1)
        pains.append(float(signals[1]))
        dones.append(bool(row["done"]))
        episodes.append(row.get("episode"))
    span = max(int(window), 1)
    labels = []
    for index in range(len(rows)):
        value = 0.0
        for offset in range(span):
            position = index + offset
            if position >= len(rows) or episodes[position] != episodes[index]:
                break
            value = max(value, pains[position])
            if dones[position]:
                break
        labels.append(value)
    return labels


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
        feature_action_prior: float = 2.0,
        pain_horizon: Optional[int] = None,
        voi_target: int = 3,
        voi_cell_cap: int = 16,
        voi_min_prior: float = 0.5,
        feature_value_prior: float = 4.0,
        feature_value_weight: float = 0.02,
        value_horizon: Optional[int] = None,
        action_temperature: float = 0.1,
        action_uncertainty: float = 0.5,
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
        if float(action_temperature) < 0.0:
            raise ValueError("action_temperature must be non-negative")
        if float(feature_action_prior) < 0.0:
            raise ValueError("feature_action_prior must be non-negative")
        if pain_horizon is not None and int(pain_horizon) < 1:
            raise ValueError("pain_horizon must be positive")
        if int(voi_target) < 1:
            raise ValueError("voi_target must be positive")
        if int(voi_cell_cap) < 1:
            raise ValueError("voi_cell_cap must be positive")
        if not 0.0 <= float(voi_min_prior) <= 1.0:
            raise ValueError("voi_min_prior must be in [0, 1]")
        if float(feature_value_prior) < 0.0:
            raise ValueError("feature_value_prior must be non-negative")
        if float(feature_value_weight) < 0.0:
            raise ValueError("feature_value_weight must be non-negative")
        if value_horizon is not None and int(value_horizon) < 1:
            raise ValueError("value_horizon must be positive")
        if float(action_uncertainty) < 0.0:
            raise ValueError("action_uncertainty must be non-negative")
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
        self.feature_action_prior = float(feature_action_prior)
        self.action_temperature = float(action_temperature)
        self.action_uncertainty = float(action_uncertainty)
        # Per (feature root, acting action) pain statistics.  The acting
        # action of the transition s_m -> s_{m+1} is the one issued `lag`
        # steps earlier (`history[lag]`), which is also the first pending
        # coordinate of the encoded state at decision time.
        self.feature_action_stats = {}
        # Decision-time readout predictions vs actual precursor outcomes,
        # per (feature root, acting action), plus a bounded buffer of the
        # most confident surprises (the counterfactual target list).
        self.feature_calibration = {}
        self.surprise_buffer = []
        self.surprise_capacity = 64
        # Counterfactual guidance: how many samples a (feature, action) cell
        # should have before the guide stops testing it, how many guide
        # tests one cell may receive, and the prior above which a root's
        # consequence counts as large enough to test.
        self.voi_target = int(voi_target)
        self.voi_cell_cap = int(voi_cell_cap)
        self.voi_min_prior = float(voi_min_prior)
        self.voi_tests = {}
        # Sequence-level value: per (feature, acting action) cells learning
        # the discounted dopamine return of the paths that followed them,
        # shrunk toward a running global baseline.  `feature_value_weight`
        # is how much one unit of this value trades against one unit of
        # accumulated hazard in the scalarized search objective.
        self.feature_value_stats = {}
        self.feature_value_prior = float(feature_value_prior)
        self.feature_value_weight = float(feature_value_weight)
        self.value_horizon = (
            None if value_horizon is None else int(value_horizon)
        )
        self.value_baseline = 0.0
        self.value_baseline_rate = 0.02
        # Precursor label window: how many entered states ahead a pain event
        # still labels the current state.  Defaults to lag + 1, the first
        # state an action issued now can physically affect.
        self.pain_horizon = (
            int(pain_horizon) if pain_horizon is not None else int(self.lag) + 1
        )
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
        self.action_nodes = {
            int(action): self.graph.add_node(
                CONCEPT, "action:%d" % int(action), role="action"
            )
            for action in self.action_space
        }
        self.last_action_probs = (1.0, 0.0)
        self.rules = FeatureRuleBank()
        # Optional second bank induced on a longer precursor window; roots
        # installed from it carry their own `pain_horizon` tag.
        self.rules_long = None
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
        if pain > 0.0:
            # Features are installed after the first epoch.  Later pain
            # events reuse the existing memories, so index those memories
            # with the active features here instead of only at creation.
            for memory_id in memories:
                for feature_id in dict.fromkeys(
                    current_features + next_features
                ):
                    self.graph.add_membership(memory_id, feature_id)
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

    def _action_margin(self, scores) -> float:
        actions = list(self.action_space)
        if len(actions) < 2:
            return 0.0
        return float(
            scores[actions[0]]["cost"] - scores[actions[1]]["cost"]
        )

    def _action_probability(self, scores) -> float:
        """Probability of the second action from preference and confidence."""
        actions = list(self.action_space)
        if len(actions) < 2:
            return 1.0
        margin = self._action_margin(scores)
        uncertainty = 0.5 * (
            float(scores[actions[0]]["uncertainty"])
            + float(scores[actions[1]]["uncertainty"])
        )
        temperature = (
            self.action_temperature
            + self.action_uncertainty * max(0.0, uncertainty)
        )
        if temperature <= 0.0:
            return 1.0 if margin > 0.0 else 0.0
        return float(1.0 / (1.0 + np.exp(-margin / temperature)))

    def _set_action_nodes(self, margin: float) -> None:
        actions = list(self.action_space)
        if len(actions) < 2:
            return
        self.graph.activate([self.action_nodes[actions[0]]], [-margin])
        self.graph.activate([self.action_nodes[actions[1]]], [margin])

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
        actions = list(self.action_space)
        margin = self._action_margin(scores)
        probability = self._action_probability(scores)
        self._set_action_nodes(margin)
        self.last_action_probs = (1.0 - probability, probability)
        if explore and len(actions) >= 2:
            action = (
                actions[1] if self.rng.rand() < probability
                else actions[0]
            )
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
        # Precursor labels: a feature on the *current* state is trained to
        # predict pain within its own window (`pain_horizon`, default
        # `lag + 1`), so it can fire while the action can still change the
        # outcome.  Roots installed from a longer-window bank carry their
        # own horizon and get that window's labels.
        horizons = {self._node_horizon(fid) for fid in self.feature_ids}
        horizons.add(self.pain_horizon)
        labels_by_horizon = {
            horizon: future_pain_labels(rows, horizon)
            for horizon in sorted(horizons)
        }
        steps = []
        for index, row in enumerate(rows):
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
            active_by_horizon = {}
            for fid in current_active:
                active_by_horizon.setdefault(
                    self._node_horizon(fid), []
                ).append(fid)
            # Prediction-error loop: record what the readout predicts for
            # this state under the acting action *before* any update, then
            # compare it with the actual precursor label.  Surprises with
            # enough confidence land in the counterfactual target buffer.
            history_row = np.asarray(row["history"], dtype=float).reshape(-1)
            acting = None
            if history_row.size > self.lag:
                acting = int(round(float(history_row[self.lag])))
            if acting is not None and self.feature_ids:
                for horizon, horizon_ids in active_by_horizon.items():
                    self._record_prediction(
                        [fid for fid in horizon_ids
                         if fid in self.feature_ids],
                        acting, labels_by_horizon[horizon][index],
                        state,
                    )
            for horizon, horizon_ids in active_by_horizon.items():
                self._update_feature_pain(
                    horizon_ids, labels_by_horizon[horizon][index]
                )
            # Action-conditioned statistics: the action acting on the
            # transition s_m -> s_{m+1} was issued `lag` steps earlier
            # (`history[lag]`) and is what the search controls per plan.
            if acting is not None:
                for horizon, horizon_ids in active_by_horizon.items():
                    self._update_feature_action(
                        horizon_ids,
                        acting,
                        labels_by_horizon[horizon][index],
                    )
            entered = self.graph.nodes[next_node]
            entered.immediate_pain += self.alpha_now * (
                signals[1] - entered.immediate_pain
            )
            entered.immediate_dopamine += self.alpha_now * (
                signals[0] - entered.immediate_dopamine
            )
            self._note_pain(next_node, float(signals[1]))
            self._component_step(next_node, signals)
            value_roots = [
                fid for fid in current_active if fid in self.feature_ids
            ]
            steps.append((
                current, action, next_node, float(signals[1]),
                float(signals[0]), done, sources, signature,
                acting, value_roots,
            ))
            if done:
                self.reset()

        running_pain = 0.0
        running_dopamine = 0.0
        # Short-horizon value attribution bootstraps from the pre-update node
        # value of the state `value_horizon` steps ahead, so the k-step target
        # keeps the return scale of the MC path while crediting only the
        # actions inside the horizon.
        value_snapshot = (
            {key: node.value for key, node in self.graph.nodes.items()}
            if self.value_horizon else None
        )
        for position in range(len(steps) - 1, -1, -1):
            (current, action, next_node, pain, dopamine,
             terminal, sources, signature, acting,
             value_roots) = steps[position]
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
            if acting is not None and value_roots:
                # Sequence-level value: a (feature, acting action) cell
                # learns the discounted dopamine return of the paths that
                # actually followed it, so "aligned for a long survival"
                # becomes readable where the pain hazard only says "local".
                target = (
                    self._k_step_value_target(
                        steps, position, value_snapshot
                    )
                    if value_snapshot is not None else return_dopamine
                )
                self._update_feature_value(value_roots, acting, target)
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
        """Induce compositional precursor rules over encoded states.

        Only the observation part is used.  With precursor labels the target
        is ``pain within lag + 1 steps`` of the state itself, so a rule
        describes a situation that *precedes* death while an action can
        still change the outcome.  Lag/action coordinates remain available
        to the node graph.
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

    def install_feature_rules(self, rules=None,
                              horizon: Optional[int] = None) -> List[str]:
        """Materialize induced rules as a shared feature graph.

        Primitive and composite nodes are shared by every rule that uses the
        same sub-feature; derivation edges point from a composite to its
        children.  Root nodes keep the learned P(pain | feature) in ``value``
        and, when `horizon` is given, the precursor window their value and
        action cells are trained with.
        """
        rules = self.rules.rules if rules is None else list(rules)
        for rule in rules:
            node_id = self._materialize_feature(rule.feature, "feature")
            node = self.graph.nodes[node_id]
            # Initialize from train counts only: the held-out split that
            # selected the rule must not also seed its predictor value.
            prior = (float(rule.hits) + 1.0) / (float(rule.support) + 2.0)
            node.value = max(float(node.value), prior)
            node.count = max(node.count, int(rule.support))
            node.pain_horizon = (
                int(horizon) if horizon is not None else None
            )
            if node_id not in self.feature_ids:
                self.feature_ids.append(node_id)
        return list(self.feature_ids)

    def _node_horizon(self, node_id: str) -> int:
        node = self.graph.nodes.get(node_id)
        if node is not None and node.pain_horizon is not None:
            return int(node.pain_horizon)
        return self.pain_horizon

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

    def _update_feature_action(self, feature_ids, action: int,
                               pain: float) -> None:
        """Accumulate P(pain | feature active, acting action) cells."""
        pain = float(pain)
        for node_id in feature_ids:
            cells = self.feature_action_stats.setdefault(node_id, {})
            cell = cells.setdefault(int(action), [0, 0.0])
            cell[0] += 1
            cell[1] += pain

    def _feature_cell_read(self, node_id: str, acting: int):
        """One root's readout value under one acting action, with support."""
        node = self.graph.nodes.get(node_id)
        if node is None:
            return 0.0, 0
        base = float(node.value)
        cell = self.feature_action_stats.get(node_id, {}).get(int(acting))
        if cell is not None and cell[0] > 0:
            prior = self.feature_action_prior
            count, hits = int(cell[0]), float(cell[1])
            return (hits + prior * base) / (count + prior), count
        return base, 0

    def _record_prediction(self, root_ids, acting: int, actual: float,
                           state) -> None:
        """Record the pre-update readout against the actual outcome.

        The prediction is what the search would see for this state under the
        acting action.  Confidence comes from the (feature, action) cell
        support; confident surprises enter the counterfactual target buffer.
        """
        actual = float(actual)
        for node_id in root_ids:
            prediction, count = self._feature_cell_read(node_id, acting)
            confidence = count / (count + self.feature_action_prior)
            entry = self.feature_calibration.setdefault(
                (node_id, int(acting)), [0, 0.0, 0.0, 0.0]
            )
            entry[0] += 1
            entry[1] += prediction
            entry[2] += actual
            entry[3] += (actual - prediction) ** 2
            score = confidence * abs(actual - prediction)
            if score <= 0.02:
                continue
            self.surprise_buffer.append(dict(
                state=np.asarray(state, dtype=float).copy(),
                node=node_id,
                acting=int(acting),
                prediction=float(prediction),
                actual=actual,
                confidence=float(confidence),
                score=float(score),
            ))
            if len(self.surprise_buffer) > self.surprise_capacity:
                self.surprise_buffer.sort(key=lambda item: -item["score"])
                del self.surprise_buffer[self.surprise_capacity:]

    def counterfactual_action(self, observation) -> Optional[int]:
        """The action with the most deficient (feature, action) support here.

        Counterfactual guidance for collection: among the feature roots
        active at this state, prefer the one whose minority-action cell is
        emptiest while the learned prior says the action matters.  Returns
        None when every active root already has enough support for both
        actions, when the prior is too low, or when the per-cell test
        budget is spent.  Purely learned signals - no task geometry.
        """
        if not self.feature_ids:
            return None
        obs = np.asarray(observation, dtype=float).reshape(1, -1)
        if obs.shape[1] < self.obs_dim:
            return None
        obs = obs[:, :self.obs_dim]
        best = None
        for node_id in self.feature_ids:
            feature = self._feature_objects.get(node_id)
            node = self.graph.nodes.get(node_id)
            if feature is None or node is None:
                continue
            if not bool(feature.mask(obs)[0]):
                continue
            base = float(node.value)
            if base < self.voi_min_prior:
                continue
            cells = self.feature_action_stats.get(node_id, {})
            counts = [
                int(cells.get(action, [0, 0.0])[0])
                for action in self.action_space
            ]
            minority = int(np.argmin(counts))
            if self.voi_tests.get((node_id, minority), 0) >= self.voi_cell_cap:
                continue
            deficit = self.voi_target - counts[minority]
            if deficit <= 0:
                continue
            score = base * deficit
            if best is None or score > best[0]:
                best = (score, minority, node_id)
        if best is None:
            return None
        score, action, node_id = best
        self.voi_tests[(node_id, action)] = (
            self.voi_tests.get((node_id, action), 0) + 1
        )
        return action

    def _k_step_value_target(self, steps, position, snapshot):
        """k-step dopamine return with a bootstrap from the pre-update node
        value of the state `value_horizon` steps ahead.

        Truncation at the episode end simply sums the rewards that actually
        happened (the same convention as the MC path at a terminal step).
        """
        total = 0.0
        discount = 1.0
        for offset in range(int(self.value_horizon)):
            index = position + offset
            if index >= len(steps):
                return total
            total += discount * float(steps[index][4])
            discount *= self.gamma
        index = position + int(self.value_horizon) - 1
        if index < len(steps):
            total += discount * float(snapshot.get(steps[index][2], 0.0))
        return total

    def _update_feature_value(self, feature_ids, acting: int,
                              return_dopamine: float) -> None:
        """Accumulate E[discounted dopamine return | feature, acting]."""
        value = float(return_dopamine)
        for node_id in feature_ids:
            cells = self.feature_value_stats.setdefault(node_id, {})
            cell = cells.setdefault(int(acting), [0, 0.0])
            cell[0] += 1
            cell[1] += value
        self.value_baseline += self.value_baseline_rate * (
            value - self.value_baseline
        )

    def feature_value_batch(self, states: np.ndarray) -> np.ndarray:
        """Expected discounted dopamine return at each state, in return units.

        The read for a state is the support-shrunk mean over its active
        feature roots of the ``(feature, pending[0])`` value cell; roots and
        cells without data contribute the running global baseline, so the
        readout degrades to a flat prior exactly where nothing was learned.
        """
        states = np.asarray(states, dtype=float)
        baseline = float(self.value_baseline)
        result = np.full(len(states), baseline)
        if not self.feature_ids:
            return result
        obs = states[:, :self.obs_dim]
        acting = None
        if states.shape[1] > self.obs_dim:
            acting = np.rint(states[:, self.obs_dim]).astype(int)
        total = np.zeros(len(states), dtype=float)
        weight = np.zeros(len(states), dtype=float)
        for node_id in self.feature_ids:
            feature = self._feature_objects.get(node_id)
            if feature is None:
                continue
            mask = feature.mask(obs)
            if not mask.any():
                continue
            values = np.full(len(states), baseline)
            if acting is not None:
                for action_key, (count, total_return) in (
                    self.feature_value_stats.get(node_id, {}).items()
                ):
                    if count <= 0:
                        continue
                    selected = mask & (acting == int(action_key))
                    if selected.any():
                        prior = self.feature_value_prior
                        values[selected] = (
                            total_return + prior * baseline
                        ) / (count + prior)
            total += np.where(mask, values, 0.0)
            weight += mask.astype(float)
        averaged = np.where(weight > 0.0, total / np.maximum(weight, 1e-9),
                            baseline)
        return averaged

    def prediction_report(self) -> dict:
        """Calibration of the readout: prediction vs actual per cell."""
        cells = []
        for (node_id, acting), (n, pred_sum, actual_sum, sq) in (
            self.feature_calibration.items()
        ):
            if n <= 0:
                continue
            feature = self._feature_objects.get(node_id)
            pred_mean = pred_sum / n
            actual_mean = actual_sum / n
            cells.append(dict(
                node=node_id,
                feature="" if feature is None else feature.label(),
                action=int(acting),
                n=n,
                pred=pred_mean,
                actual=actual_mean,
                brier=sq / n,
                gap=pred_mean - actual_mean,
                miscalibrated=(abs(pred_mean - actual_mean) > 0.15
                               and n >= 10),
            ))
        cells.sort(
            key=lambda item: -abs(item["gap"]) * min(item["n"], 50)
        )
        return dict(
            cells=cells,
            miscalibrated=sum(1 for item in cells if item["miscalibrated"]),
            buffer_size=len(self.surprise_buffer),
            buffer=[
                dict(score=item["score"], acting=item["acting"],
                     prediction=item["prediction"], actual=item["actual"],
                     confidence=item["confidence"])
                for item in sorted(
                    self.surprise_buffer, key=lambda item: -item["score"]
                )[:8]
            ],
        )

    def _activate_features(self, feature_ids) -> None:
        active = set(feature_ids)
        for node_id in self._feature_all:
            node = self.graph.nodes.get(node_id)
            if node is not None:
                node.activate(1.0 if node_id in active else 0.0)

    def feature_risk_batch(self, states: np.ndarray) -> np.ndarray:
        """Public read of the action-conditioned feature risk cells."""
        return self._feature_risk_batch(states)

    def _feature_risk_batch(self, states: np.ndarray) -> np.ndarray:
        """Action-conditioned feature risk.

        The readout is ``P(pain within lag+1 steps | feature, acting
        action)``.  The acting action is the first pending coordinate of the
        encoded state, so two plans whose pending actions differ receive
        different risk at the same predicted state.  Observed
        ``(feature, action)`` cells shrink toward the feature's scalar
        precursor value with `feature_action_prior` pseudo-counts; cells
        without data fall back to that scalar exactly as before.
        """
        states = np.asarray(states, dtype=float)
        result = np.zeros(len(states), dtype=float)
        if not self.feature_ids:
            return result
        obs = states[:, :self.obs_dim]
        acting = None
        if states.shape[1] > self.obs_dim:
            acting = np.rint(states[:, self.obs_dim]).astype(int)
        for node_id in self.feature_ids:
            feature = self._feature_objects.get(node_id)
            node = self.graph.nodes.get(node_id)
            if feature is None or node is None:
                continue
            mask = feature.mask(obs)
            if not mask.any():
                continue
            base = float(node.value)
            values = np.full(len(states), base)
            if acting is not None:
                for action_key, (count, hits) in (
                    self.feature_action_stats.get(node_id, {}).items()
                ):
                    if count <= 0:
                        continue
                    selected = mask & (acting == int(action_key))
                    if selected.any():
                        values[selected] = (
                            hits + self.feature_action_prior * base
                        ) / (count + self.feature_action_prior)
            result = np.maximum(result, np.where(mask, values, 0.0))
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
