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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from node_graph import CONCEPT, NodeGraph


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
        max_nodes: int = 2000,
        exploration: float = 0.1,
        uncertainty_weight: float = 0.05,
        unknown_penalty: float = 0.5,
        soft_neighbors: int = 5,
        soft_sigma: float = 0.75,
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
        self._prototype_ids: List[str] = []
        self._prototype_matrix = np.zeros((0, state_dim), dtype=float)
        self._traces: Dict[str, List[float]] = {}
        self.pending = self._zero_pending()
        self.last_node: Optional[str] = None
        self.last_action: Optional[int] = None
        self.last_state: Optional[np.ndarray] = None
        self.last_scores: Dict[int, dict] = {}

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
                       state: Optional[np.ndarray] = None) -> Dict[int, dict]:
        node = self.graph.nodes[node_id]
        if state is None:
            state = (
                node.features if node.features is not None
                else np.zeros(len(self.scales), dtype=float)
            )
        neighbors = self._neighbor_weights(state)
        scores: Dict[int, dict] = {}
        for action in self.action_space:
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
        return node_id, self._score_actions(node_id, state)

    def act(self, observation, explore: bool = True) -> int:
        state = self.encode(observation)
        node_id = self._assign(state, create=True, count=True)
        scores = self._score_actions(node_id, state)
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
        self.pending.append(action)
        return action

    # ---- learning ----------------------------------------------------------
    def _update(self, current: str, action: int, next_node: str,
                target_pain: float, target_dopamine: float,
                immediate_pain: float, immediate_dopamine: float) -> None:
        edge = self.graph.add_action_edge(current, action, next_node)
        edge.prediction_error = target_pain - edge.q_pain
        edge.count += 1
        edge.q_pain += self.alpha_q * (target_pain - edge.q_pain)
        edge.q_dopamine += (
            self.alpha_q * (target_dopamine - edge.q_dopamine)
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
        )
        self.last_node = None
        self.last_action = None
        self.last_state = None
        if done:
            self.reset()

    def reset(self) -> None:
        """Reset episode-local state; node values and edges are retained."""
        self.pending = self._zero_pending()
        self.last_node = None
        self.last_action = None
        self.last_state = None
        self.last_scores = {}
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
        self.pending.append(action)
        next_state = self.encode(next_observation)
        next_node = self._assign(next_state, create=True, count=True)
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
        )
        if done:
            self.reset()

    def learn_episode(self, rows) -> None:
        """Monte-Carlo node/edge values for one real episode.

        `rows` are transition dicts with ``obs``, ``next_obs``, ``history``,
        ``signals`` and ``terminal``.  Values are accumulated backward from the
        world's pain signal; no `terminal` risk label is used.
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
            state = self.encode(row["obs"])
            current = self._assign(state, create=True, count=True)
            self.pending.append(action)
            next_state = self.encode(row["next_obs"])
            next_node = self._assign(next_state, create=True, count=True)
            edge = self.graph.add_action_edge(current, action, next_node)
            edge.count += 1
            entered = self.graph.nodes[next_node]
            entered.immediate_pain += self.alpha_now * (
                signals[1] - entered.immediate_pain
            )
            entered.immediate_dopamine += self.alpha_now * (
                signals[0] - entered.immediate_dopamine
            )
            steps.append((
                current, action, next_node, float(signals[1]),
                float(signals[0]), bool(row["terminal"]),
            ))
            if row["terminal"]:
                self.reset()

        running_pain = 0.0
        running_dopamine = 0.0
        for current, action, next_node, pain, dopamine, terminal in reversed(
            steps
        ):
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
            edge = self.graph.action_edges.get(
                (current, next_node, action)
            )
            if edge is not None:
                edge.q_pain += self.alpha_mc * (return_pain - edge.q_pain)
                edge.q_dopamine += self.alpha_mc * (
                    return_dopamine - edge.q_dopamine
                )
            if terminal:
                entered = self.graph.nodes[next_node]
                entered.value += self.alpha_mc * (
                    pain - entered.immediate_pain - entered.value
                )
            running_pain = return_pain
            running_dopamine = return_dopamine
        self.reset()

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
        return (
            (weights * pain_values).sum(axis=1),
            (weights * dopamine_values).sum(axis=1),
        )

    def value_state(self, state: np.ndarray) -> float:
        weights = self._neighbor_weights(np.asarray(state, dtype=float))
        return float(sum(
            weight * (
                self.graph.nodes[node_id].value
                + self.graph.nodes[node_id].immediate_pain
            )
            for node_id, weight in weights
        )) if weights else 0.0

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
            prototypes=len(self._prototype_ids),
        )
