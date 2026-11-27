"""NodeRuntime learning mixin: prototype births/pruning, memory traces, action edges, and rule induction."""
from collections import deque
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from branches import (
    BranchBank,
    CurrentBranch,
    PredictionBranch,
    RecentBranch,
)
from node_components import ComponentBank
from node_features import (
    FeatureRuleBank,
    expand_observation,
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
from online_feature_learner import OnlineFeatureLearner


class NodeLearningMixin:
    """Graph construction, TD(lambda) traces, episode replay, and feature induction methods."""

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
        elif create and distance > self.birth_threshold:
            if len(self._prototype_ids) < self.max_nodes:
                node_id = self._create_node(state)
            elif self.utility_pruning and len(self._prototype_ids) > 1:
                # The budget blocks the birth: consolidate the least
                # useful prototype into its neighbour first (GNG-U), so
                # the cover keeps refining instead of absorbing far
                # states into stale cells forever.  Nodes the running
                # episode still references are not candidates.
                self._prune_prototypes(exclude=self._episode_nodes)
                if len(self._prototype_ids) < self.max_nodes:
                    node_id = self._create_node(state)
        elif create and self.variance_births and self._variance_split(
            node_id, distance
        ):
            # Refinement birth: the nearest cell holds a statistical
            # mixture and the query sits far enough inside its territory
            # to deserve its own statistics.
            if len(self._prototype_ids) < self.max_nodes:
                node_id = self._create_node(state)
            elif self.utility_pruning and len(self._prototype_ids) > 1:
                self._prune_prototypes(exclude=self._episode_nodes)
                if len(self._prototype_ids) < self.max_nodes:
                    node_id = self._create_node(state)
        if count:
            self.graph.nodes[node_id].count += 1
        self._episode_nodes.add(node_id)
        return node_id

    def _refresh_variance_bar(self) -> None:
        """Population-relative valence-variance bar, once per episode."""
        variances = []
        for node_id, count in self._node_pain_count.items():
            if count < self.component_min_count:
                continue
            variances.append(
                self._node_pain_m2.get(node_id, 0.0) / max(count - 1, 1)
            )
        if len(variances) > 1:
            self._variance_bar = float(
                np.median(variances) + np.std(variances)
            )
        else:
            self._variance_bar = None

    def _variance_split(self, node_id: str, distance: float) -> bool:
        """Refinement-birth test for the nearest cell.

        The cell needs enough samples for its variance to mean anything,
        the variance must sit above the same population-relative bar the
        component binder uses, and the query must sit well inside the
        cell's territory - a split at the periphery gives the mixture's
        outer states their own center while the core keeps its history.
        """
        if self._variance_bar is None:
            return False
        count = self._node_pain_count.get(node_id, 0)
        if count < self.component_min_count:
            return False
        variance = self._node_pain_m2.get(node_id, 0.0) / max(count - 1, 1)
        if variance <= self._variance_bar:
            return False
        return distance > 0.5 * self.birth_threshold

    def _prune_prototypes(self, exclude=()) -> Optional[str]:
        """Remove the least useful prototype, merging it into its neighbour.

        Utility follows the growing-neural-gas principle - the error a
        removal would induce: a node's visit count times the squared
        distance to its nearest neighbour plus the squared gap between
        their valence values.  A rare cell far from every other cell with
        a distinct outcome is expensive to delete and survives; a
        near-duplicate of its neighbour is cheap and merges first, its
        statistics carried over as count-weighted means so no evidence is
        lost, only consolidated.  `exclude` names nodes that must not be
        removed - the running episode's own assignments, whose backward
        pass still references them.
        """
        count_all = len(self._prototype_ids)
        if count_all <= 1:
            return None
        matrix = self._prototype_matrix
        state_norm = np.einsum("ij,ij->i", matrix, matrix)[:, None]
        proto_norm = np.einsum("ij,ij->i", matrix, matrix)[None, :]
        distances = state_norm + proto_norm - 2.0 * (matrix @ matrix.T)
        np.fill_diagonal(distances, np.inf)
        nearest_index = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(count_all), nearest_index]
        excluded = set(exclude)
        utilities = np.full(count_all, np.inf)
        for index, node_id in enumerate(self._prototype_ids):
            if node_id in excluded:
                continue
            node = self.graph.nodes[node_id]
            other = self.graph.nodes[self._prototype_ids[nearest_index[index]]]
            gap = node.value - other.value
            utilities[index] = max(float(node.count), 1.0) * (
                nearest_distance[index] ** 2 + gap * gap
            )
        victim_index = int(np.argmin(utilities))
        if not np.isfinite(utilities[victim_index]):
            # Every candidate is protected (a single-node episode at the
            # cap): the birth is absorbed into the nearest cell.
            return None
        victim_id = self._prototype_ids[victim_index]
        host_id = self._prototype_ids[int(nearest_index[victim_index])]
        victim = self.graph.nodes[victim_id]
        host = self.graph.nodes[host_id]
        victim_count = max(float(victim.count), 1.0)
        host_count = max(float(host.count), 1.0)
        total = victim_count + host_count
        blend = victim_count / total
        host.value += blend * (victim.value - host.value)
        host.dopamine += blend * (victim.dopamine - host.dopamine)
        host.immediate_pain += blend * (
            victim.immediate_pain - host.immediate_pain
        )
        host.immediate_dopamine += blend * (
            victim.immediate_dopamine - host.immediate_dopamine
        )
        host.count += victim.count
        victim_stats = self._association_stats.pop(victim_id, None)
        if victim_stats is not None:
            host_stats = self._association_stats.setdefault(
                host_id, [0, 0.0]
            )
            host_stats[0] += victim_stats[0]
            host_stats[1] += victim_stats[1]
        v_count = self._node_pain_count.pop(victim_id, None)
        if v_count:
            h_count = self._node_pain_count.get(host_id, 0)
            v_mean = self._node_pain_mean.get(victim_id, 0.0)
            h_mean = self._node_pain_mean.get(host_id, 0.0)
            v_m2 = self._node_pain_m2.get(victim_id, 0.0)
            h_m2 = self._node_pain_m2.get(host_id, 0.0)
            merged_count = h_count + v_count
            if merged_count > 0:
                delta = v_mean - h_mean
                merged_m2 = (
                    h_m2 + v_m2
                    + delta * delta * h_count * v_count / merged_count
                )
                self._node_pain_count[host_id] = merged_count
                self._node_pain_mean[host_id] = (
                    (h_count * h_mean + v_count * v_mean) / merged_count
                )
                self._node_pain_m2[host_id] = max(merged_m2, 0.0)
        self._component_bound.discard(victim_id)
        self.graph._remove_node(victim_id)
        self._prototype_ids.pop(victim_index)
        self._prototype_matrix = np.delete(
            self._prototype_matrix, victim_index, axis=0
        )
        return victim_id

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
                  next_features: Iterable[str] = (),
                  learn_memory_credit: bool = False) -> None:
        """Create association, causal transition and signal-driven memory."""
        # Every learning path (observe / learn_transition / learn_episode)
        # funnels through here, so one invalidation keeps the cached readout
        # columns and adjacency from going stale after a transition updates
        # edges, prototypes or association rates.
        self._invalidate_readout_cache()
        pain = float(pain)
        dopamine = float(dopamine)
        current_features = list(dict.fromkeys(current_features))
        next_features = list(dict.fromkeys(next_features))
        self.branches.get("recent").step()
        # `recent` holds the state being left (plus older decaying trace);
        # `current` holds the state the transition entered.  transition_pairs
        # therefore reads old -> new when it multiplies recent by current.
        self.branches.activate(
            "recent", [current] + current_features,
            [1.0] * (1 + len(current_features)),
        )
        self.branches.activate(
            "current", [next_node] + next_features,
            [1.0] * (1 + len(next_features)),
        )
        if current == next_node:
            # A self-loop is still a visit: the per-node valence rate must
            # see every transition through the node, even the ones that
            # cannot learn a structural edge.
            self._note_valence([current], pain)
            if learn_memory_credit:
                self._advance_memory_traces(
                    [current] + current_features, (), None, pain
                )
            return
        self.graph.activate([current, next_node], [1.0, 1.0])
        self.graph.observe_association(
            [current, next_node], rate=self.graph_rate
        )
        edge = self.graph.observe_transition(
            current, next_node, rate=self.graph_rate
        )
        memories = self.memories_for(
            [current, next_node] + current_features + next_features
        )
        # Valence rate statistics: every transition through a node - clean
        # or painful - is a sample of its local pain rate.  Pain-only
        # evidence would make the channel a monotone envelope ("has this
        # neighborhood ever been adjacent to pain") instead of a rate.
        self._note_valence(
            [current, next_node] + list(memories), pain
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
            self._note_valence([created_memory], pain)
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
        if learn_memory_credit:
            self._advance_memory_traces(
                [current, next_node] + current_features + next_features,
                list(memories), created_memory, pain,
            )

    def backfill_memory_features(self) -> int:
        """Link event memories to the feature roots active at their members.

        Memories are created before features exist (induction runs after the
        first pass), so the feature-membership read path of the memory risk
        channel is empty unless the links are added afterwards.  This walks
        every memory's member prototypes and links it to the roots that fire
        at those states.  Task-agnostic: only the induced features and the
        prototype states are used.
        """
        if not self.feature_ids:
            return 0
        roots = [
            (node_id, self._feature_objects[node_id])
            for node_id in self.feature_ids
            if node_id in self._feature_objects
        ]
        links = 0
        for memory in list(self.graph.nodes.values()):
            if memory.kind != MEMORY:
                continue
            members = [
                key[1] for key in self.graph.outgoing.get(memory.id, [])
                if key[2] == MEMBERSHIP
            ]
            for member in members:
                node = self.graph.nodes.get(member)
                if node is None or node.features is None:
                    continue
                state = np.asarray(node.features, dtype=float).reshape(
                    1, -1
                )[:, :self.obs_dim]
                for root_id, feature in roots:
                    if bool(feature.mask(state)[0]):
                        edge = self.graph.add_membership(
                            memory.id, root_id
                        )
                        # The link is definitional: the root fires at the
                        # memory's own member state, so the strength starts
                        # at one instead of the zero-weight default.
                        edge.weight = max(float(edge.weight), 1.0)
                        links += 1
        return links

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

    def _note_valence(self, node_ids, pain: float) -> None:
        """Record one transition's outcome as a sample of each node's rate.

        `pain` is the entered state's valence mass.  Clean transitions are
        samples too: without them a node's read would be the frequency of
        pain among pain-adjacent events, not among its transitions.
        """
        pain = float(pain)
        self._association_global[0] += 1
        self._association_global[1] += pain
        for node_id in dict.fromkeys(node_ids):
            stats = self._association_stats.setdefault(node_id, [0, 0.0])
            stats[0] += 1
            stats[1] += pain

    def _note_valence_weighted(self, node_ids, pain: float,
                               weight: float) -> None:
        """Record a fractional outcome sample (retrospective credit).

        An eligibility trace gives the pain event a decayed weight at an
        earlier memory; this adds that fractional observation to the same
        per-node rate statistics a real transition would, so the memory's
        risk read is corrected without ever labelling a flight state as
        painful.
        """
        weight = float(weight)
        if weight <= 0.0:
            return
        pain = float(pain)
        self._association_global[0] += weight
        self._association_global[1] += weight * pain
        for node_id in dict.fromkeys(node_ids):
            stats = self._association_stats.setdefault(node_id, [0.0, 0.0])
            stats[0] += weight
            stats[1] += weight * pain

    def _advance_memory_traces(self, active, memories, created_memory,
                               pain: float) -> None:
        """Advance the episode's memory eligibility and apply the pain event.

        Two corrections happen on a real pain event, both driven only by
        that event and by the episode's own eligibility trace:

        * the values (association rates) of memories that were active
          earlier are corrected by a fractional sample weighted by
          ``(gamma*lambda)^distance``;
        * the event memory is linked back to the preceding context, so a
          similar lead-in retrieves it before the collision state itself.

        No state label is rewritten: the pain stays on the transition that
        produced it.  The context link list is capped by
        ``memory_context_limit`` purely as a resource bound.
        """
        decay = self.gamma * self.lam
        traces = self._memory_traces
        for node_id in list(traces):
            traces[node_id] *= decay
            if traces[node_id] < 1e-4:
                del traces[node_id]
        for node_id in dict.fromkeys(list(active) + list(memories)):
            traces[node_id] = traces.get(node_id, 0.0) + 1.0
        if created_memory is not None:
            traces[created_memory] = traces.get(created_memory, 0.0) + 1.0
        if pain <= 0.0:
            return
        stats = self.memory_credit_stats
        stats["events"] += 1
        current = set(active) | set(memories)
        if created_memory is not None:
            current.add(created_memory)
        for node_id, eligibility in list(traces.items()):
            if node_id in current:
                continue
            node = self.graph.nodes.get(node_id)
            if node is None or node.kind != MEMORY:
                continue
            weight = min(1.0, float(eligibility))
            self._note_valence_weighted([node_id], pain, weight)
            stats["corrected"] += 1
            stats["weight"] += weight
            stats["corrected_ids"].add(node_id)
        target = created_memory
        if target is None and memories:
            target = max(memories, key=lambda mid: traces.get(mid, 0.0))
        if target is None:
            return
        context = [
            (node_id, float(eligibility))
            for node_id, eligibility in traces.items()
            if node_id != target and eligibility >= 0.05
            and self.graph.nodes.get(node_id) is not None
            and self.graph.nodes[node_id].kind != MEMORY
        ]
        context.sort(key=lambda item: -item[1])
        chosen = context[:self.memory_context_limit]
        for member_id, eligibility in chosen:
            edge = self.graph.add_membership(target, member_id)
            edge.weight = max(edge.weight, min(1.0, eligibility))
        stats["links"] += len(chosen)

    def _association_base(self) -> float:
        """Global pain rate per transition, a 1/n cumulative mean."""
        count, hits = self._association_global
        if count <= 0:
            return 0.0
        return hits / count

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
        """Empirical pain rate of a node's transitions, shrunk to the base rate.

        The read is ``(pain mass + prior * base) / (transitions + prior)``
        over every transition that ran through the node, clean ones
        included, so it estimates ``E[pain at the entered state |
        transition through the node]`` - a rate whose uncertainty shrinks
        at 1/count.  A node with no samples reads the global base rate,
        the neutral hazard prior for unseen regions.
        """
        prior = self.association_prior
        base = self._association_base()
        stats = self._association_stats.get(node_id)
        if stats is None or stats[0] <= 0:
            return float(base)
        count, hits = stats
        return (hits + prior * base) / (count + prior)

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

    def _hazard_channels(self, states: np.ndarray,
                         mask_table: Optional[List] = None,
                         include: Optional[Iterable[str]] = None,
                         soft_weights=None
                         ) -> Dict[str, np.ndarray]:
        """The individual hazard readouts the signal composes.

        Each channel reports its own estimate; `pain_signal_batch` either
        max-combines them (the historical hand-written composition) or,
        when the stacked composition is fitted, takes the learned
        non-negative weighted sum.  Keeping the channels in one place
        guarantees the fit and the consumers always see the same columns.

        `include` names the channels a consumer actually reads.  The
        stacked composition zeroes channels that duplicate an already
        covered member, and computing a zero-weight channel would be pure
        cost - it is read out but multiplied by zero.  When omitted every
        channel is computed, which is what the composition fit needs.
        """
        states = np.asarray(states, dtype=float)
        channels: Dict[str, np.ndarray] = {}
        want = None if include is None else set(include)
        if not self.predicate_readouts and (
            want is None or "prototype" in want
        ):
            if soft_weights is None:
                indices, weights, nearest_min = self._soft_weights_batch(states)
            else:
                indices, weights, nearest_min = soft_weights
            prototype = np.zeros(len(states), dtype=float)
            if indices is not None:
                if self.readout_propagation_gain > 0.0:
                    # Propagated readout: the per-prototype hazard column
                    # (each prototype's own rate folded with the rates of the
                    # memories it binds) is cached by `_prototype_columns`, so
                    # the expensive scan runs once per readout phase and the
                    # query itself only relaxes the seed's own support.
                    _pain, _dopamine, combined, associations = (
                        self._prototype_columns()
                    )
                    prototype = self._propagated_readout(
                        indices, weights, combined
                    )
                else:
                    # Vectorised: the row's neighbouring prototypes carry the
                    # association rate, and the memories they bind fold their
                    # own rates in.  One ragged expand replaces the historical
                    # per-row Python walk (and its per-row ``memories_for``).
                    associations = self._prototype_associations()
                    prototype[:] = self._prototype_hazard_from_weights(
                        indices, weights
                    )
                if self.ood_trust_scale > 0.0:
                    # The shrink target is only read when trust scaling is
                    # on; with the default zero scale the baseline sweep
                    # over every prototype was pure cost.
                    prototype = self._shrink_to_baseline(
                        prototype, nearest_min,
                        baseline=float(associations.mean())
                        if len(associations) else 0.0,
                    )
                prototype = prototype * self.obs_trust
            channels["prototype"] = prototype
        if self.feature_ids:
            if want is None or "feature" in want or "memory" in want:
                if mask_table is None:
                    mask_table = self._feature_mask_table(
                        states[:, :self.obs_dim]
                    )
            if want is None or "feature" in want:
                channels["feature"] = self._feature_risk_batch(
                    states, mask_table
                )
            if self.event_memory_risk and (
                want is None or "memory" in want
            ):
                channels["memory"] = self._feature_memory_risk_batch(
                    states, mask_table
                )
        elif self.rules.rules and (want is None or "rules" in want):
            channels["rules"] = self.rules.activate(
                states[:, :self.obs_dim]
            )
        return channels

    def pain_signal_batch(self, states: np.ndarray,
                          mask_table: Optional[List] = None,
                          soft_weights=None) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0)
        composition = self.hazard_composition
        if composition:
            # Only the channels with positive learned weight are read; a
            # zeroed channel contributes exactly nothing to the stacked
            # composition, so computing it would be pure cost.
            needed = [
                name for name, weight in composition.items()
                if float(weight) > 0.0
            ]
            if soft_weights is None:
                channels = self._hazard_channels(
                    states, mask_table, include=needed
                )
            else:
                channels = self._hazard_channels(
                    states, mask_table, include=needed,
                    soft_weights=soft_weights,
                )
        else:
            if soft_weights is None:
                channels = self._hazard_channels(states, mask_table)
            else:
                channels = self._hazard_channels(
                    states, mask_table, soft_weights=soft_weights
                )
        if composition:
            # P2 stacked composition: the non-negative least-squares
            # weights were fitted to the precursor labels, so the read is
            # the channels' own regression estimate instead of the
            # hand-written max (whose fixed point is E[max of correlated
            # estimates], structurally above max of the truths).  The
            # monotone map then holds the composed level to the empirical
            # rate.
            read = np.zeros(len(states), dtype=float)
            for name, weight in composition.items():
                column = channels.get(name)
                if column is not None and weight > 0.0:
                    read = read + float(weight) * column
            calibration = self.hazard_calibration
            readout = getattr(self, "hazard_readout", "calibrated")
            if calibration is not None and readout != "raw":
                mids, rates = calibration
                read = np.interp(read, mids, rates)
            return read
        result = np.zeros(len(states), dtype=float)
        if "prototype" in channels:
            result = np.maximum(result, channels["prototype"])
        for name in ("feature", "memory", "rules"):
            if name in channels:
                result = np.maximum(result, channels[name])
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

    def action_edge_q_table(self):
        """Mean q_pain per (nearest prototype, acting action).

        Action edges are keyed by their target too; the planner only needs a
        state-action value, so targets are averaged with count weights.
        """
        actions = [int(action) for action in self.action_space]
        table = np.zeros((len(self._prototype_ids), len(actions)), dtype=float)
        counts = np.zeros_like(table)
        if not self._prototype_ids or not actions:
            return table, counts
        index = {
            node_id: row for row, node_id in enumerate(self._prototype_ids)
        }
        action_index = {
            action: column for column, action in enumerate(actions)
        }
        for (source, _target, action), edge in self.graph.action_edges.items():
            row = index.get(source)
            column = action_index.get(int(action))
            if row is None or column is None:
                continue
            weight = float(edge.count)
            if weight <= 0.0:
                continue
            table[row, column] += weight * float(edge.q_pain)
            counts[row, column] += weight
        table = np.where(
            counts > 0.0, table / np.maximum(counts, 1e-9), 0.0
        )
        return table, counts

    def action_edge_q_batch(self, states: np.ndarray, table=None):
        """Q read for encoded states; the acting action is the oldest pending."""
        states = np.asarray(states, dtype=float)
        result = np.zeros(len(states), dtype=float)
        counts = np.zeros(len(states), dtype=float)
        if (
            len(states) == 0 or states.shape[1] <= self.obs_dim
            or not self._prototype_ids
        ):
            return result, counts
        if table is None:
            table = self.action_edge_q_table()
        q_table, q_counts = table
        indices, _weights, _nearest = self._soft_weights_batch(states)
        if indices is None:
            return result, counts
        nearest = indices[:, 0].astype(int)
        acting = np.rint(states[:, self.obs_dim]).astype(int)
        columns = {
            int(action): column
            for column, action in enumerate(self.action_space)
        }
        for row in range(len(states)):
            column = columns.get(int(acting[row]))
            if column is None:
                continue
            result[row] = q_table[nearest[row], column]
            counts[row] = q_counts[nearest[row], column]
        return result, counts

    def action_edge_transition_table(self):
        """Exact (previous prototype, next prototype, action) Q table."""
        index = {
            node_id: row for row, node_id in enumerate(self._prototype_ids)
        }
        table = {}
        for (source, target, action), edge in self.graph.action_edges.items():
            previous = index.get(source)
            following = index.get(target)
            if previous is None or following is None or edge.count <= 0.0:
                continue
            table[(previous, following, int(action))] = (
                float(edge.q_pain), float(edge.count)
            )
        return table

    def action_edge_transition_q_batch(self, previous_states,
                                       next_states, table):
        """Target-conditioned Q read for encoded consecutive states."""
        previous_states = np.asarray(previous_states, dtype=float)
        next_states = np.asarray(next_states, dtype=float)
        result = np.zeros(len(next_states), dtype=float)
        counts = np.zeros(len(next_states), dtype=float)
        if (
            table is None or len(table) == 0 or len(next_states) == 0
            or next_states.shape[1] <= self.obs_dim
            or not self._prototype_ids
        ):
            return result, counts
        previous_indices, _pw, _pm = self._soft_weights_batch(
            previous_states
        )
        next_indices, _nw, _nm = self._soft_weights_batch(next_states)
        if previous_indices is None or next_indices is None:
            return result, counts
        previous_nearest = previous_indices[:, 0].astype(int)
        next_nearest = next_indices[:, 0].astype(int)
        acting = np.rint(next_states[:, self.obs_dim]).astype(int)
        for row in range(len(next_states)):
            entry = table.get((
                int(previous_nearest[row]),
                int(next_nearest[row]),
                int(acting[row]),
            ))
            if entry is None:
                continue
            result[row], counts[row] = entry
        return result, counts

    def _update_action_edges(self, sources: Iterable[str], action: int,
                             next_node: str, target_pain: float,
                             target_dopamine: float,
                             weights: Optional[Dict[str, float]] = None,
                             scale: float = 1.0) -> None:
        for source in dict.fromkeys(sources):
            weight = float(scale)
            if weights is not None:
                weight *= float(weights.get(source, 1.0))
            if weight <= 0.0:
                continue
            edge = self.graph.add_action_edge(source, action, next_node)
            edge.prediction_error = target_pain - edge.q_pain
            edge.count += weight
            edge.q_pain += (
                self.alpha_q * weight * (target_pain - edge.q_pain)
            )
            edge.q_dopamine += (
                self.alpha_q * weight
                * (target_dopamine - edge.q_dopamine)
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

        if self.online_feature_learner is not None:
            source = np.asarray(self.last_state, dtype=float).reshape(-1)
            source_observation = (
                source[:self.obs_dim] * self.scales[:self.obs_dim]
            )
            pending = (
                source[self.obs_dim:] * self.scales[self.obs_dim:]
            )
            issued = (
                int(self.last_action)
                if self.last_action is not None else int(action)
            )
            history = np.concatenate([
                np.asarray([float(issued)]),
                np.asarray(pending, dtype=float)[::-1],
            ])
            self.observe_online_transition(dict(
                obs=source_observation,
                pending=pending,
                history=history,
                next_obs=np.asarray(observation, dtype=float),
                signals=valence.copy(),
                done=bool(done),
            ))

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
        target_pain = pain + (0.0 if done else self.pain_gamma * next_value.value)
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
        if self.online_feature_learner is not None:
            self.online_feature_learner.end_episode(terminal=False)
        self._invalidate_readout_cache()
        self.pending = self._zero_pending()
        self.last_node = None
        self.last_action = None
        self.last_state = None
        self.last_scores = {}
        self.last_sources = []
        self.last_signature = ()
        self._traces.clear()
        self._memory_traces.clear()
        self._episode_nodes = set()

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
        if self.online_feature_learner is not None:
            pending = np.asarray(
                list(self.pending), dtype=float,
            ).reshape(-1)
            history = np.concatenate([
                np.asarray([float(action)]), pending[::-1],
            ])
            self.observe_online_transition(dict(
                obs=np.asarray(observation, dtype=float),
                pending=pending,
                history=history,
                next_obs=np.asarray(next_observation, dtype=float),
                signals=valence.copy(),
                done=bool(done),
            ))
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
        target_pain = pain + (0.0 if done else self.pain_gamma * next_value.value)
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

    def learn_episode(self, rows, learn_online=True) -> None:
        """Monte-Carlo node/edge values for one real episode.

        `rows` are transition dicts with ``obs``, ``next_obs``, ``history``,
        ``signals`` and ``done`` (an episode boundary, not a risk label).
        Values are accumulated backward from the world's pain signal.
        `learn_online=False` lets batch replay train the runtime without
        counting the same real episode again in online feature statistics.
        """
        rows = list(rows)
        if not rows:
            return
        self.reset()
        if self.online_feature_learner is not None and learn_online:
            for row in rows:
                self.observe_online_transition(row)
        # Every node's activation over this episode is computed once: the
        # per-row sweeps used to re-evaluate all masks for both the current
        # and the next state of every row.
        scale = self.scales[:self.obs_dim]
        obs_table = self._feature_mask_table(
            np.asarray(
                [np.asarray(row["obs"], dtype=float) / scale for row in rows],
                dtype=float,
            )
        )
        next_table = self._feature_mask_table(
            np.asarray(
                [
                    np.asarray(row["next_obs"], dtype=float) / scale
                    for row in rows
                ],
                dtype=float,
            )
        )
        # Label regime: point-event pain (default) trains every root on the
        # entered state's own feedback; the precursor-window path (opt-in)
        # trains each root on P(pain within its configured window).
        horizons = {self._node_horizon(fid) for fid in self.feature_ids}
        horizons.add(self.pain_horizon)
        if self.point_labels:
            # The world's pain is the feedback of the entered state; with
            # no window that single value is every root's label.  Nothing
            # is spread back onto states where the world reported normal
            # flight.
            point = [
                float(np.asarray(row["signals"], dtype=float)[1] > 0.5)
                for row in rows
            ]
            labels_by_horizon = {
                horizon: point for horizon in sorted(horizons)
            }
        else:
            from node_runtime import future_pain_labels
            labels_by_horizon = {
                horizon: future_pain_labels(rows, horizon)
                for horizon in sorted(horizons)
            }
        # The refinement-birth bar is population-relative; one O(nodes)
        # pass per episode keeps it fresh without dominating the loop.
        self._refresh_variance_bar()
        value_roots_all = self._all_value_roots()
        steps = []
        # Per-row propensities within the episode: the window's clipped
        # importance weight needs the propensities of the governing
        # actions, which are the current row's action and the ones issued
        # up to `lag` rows earlier.  Cleared at episode boundaries.
        recent_propensities: List[float] = []
        for index, row in enumerate(rows):
            action = int(row["history"][0])
            if action not in self.action_space:
                raise ValueError("action is outside the action space")
            signals = np.asarray(row["signals"], dtype=float).reshape(-1)
            if signals.shape != (2,) or not np.isfinite(signals).all():
                raise ValueError("signals must contain finite valence")
            done = bool(row["done"])
            # Per-decision clipped importance weight toward the uniform
            # target over the window's governing actions (the current
            # row's propensity plus up to `lag` earlier ones).  Rows
            # without a stored propensity read one - unweighted.
            row_propensity = float(row.get("propensity", 1.0))
            row_weight = 1.0
            if self.importance_weighting:
                row_weight = 1.0
                for propensity in (
                    recent_propensities[-self.lag:] + [row_propensity]
                ):
                    behavior = max(propensity, 1e-9)
                    row_weight *= min(
                        1.0, self.importance_target / behavior
                    )
            recent_propensities.append(row_propensity)
            state = self.encode(row["obs"])
            current = self._assign(state, create=True, count=True)
            self._component_step(current)
            current_active = [
                node_id
                for node_id, mask in zip(self._feature_all, obs_table)
                if mask is not None and bool(mask[index])
            ]
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
            next_active = [
                node_id
                for node_id, mask in zip(self._feature_all, next_table)
                if mask is not None and bool(mask[index])
            ]
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
                learn_memory_credit=self.memory_credit,
            )
            active_by_horizon = {}
            active_set = set(current_active)
            if self.subgraph_cells:
                # D2: each active root trains its own window's labels, and
                # every active descendant joins its root's group - a branch
                # inherits the window of the rule it belongs to.  Shared
                # sub-features are deduplicated so one sample updates a
                # node once.
                for root_id in current_features:
                    group = active_by_horizon.setdefault(
                        self._node_horizon(root_id), []
                    )
                    group.append(root_id)
                    group.extend(
                        node_id
                        for node_id in self._feature_descendants.get(
                            root_id, ()
                        )
                        if node_id in active_set
                    )
                for horizon, group in active_by_horizon.items():
                    active_by_horizon[horizon] = list(dict.fromkeys(group))
            else:
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
                    horizon_ids, labels_by_horizon[horizon][index],
                    weight=row_weight,
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
                        weight=row_weight,
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
                fid for fid in current_active if fid in value_roots_all
            ]
            if self.subgraph_cells:
                for root_id in value_roots:
                    value_roots.extend(
                        node_id
                        for node_id in self._feature_descendants.get(
                            root_id, ()
                        )
                        if node_id in active_set
                    )
                value_roots = list(dict.fromkeys(value_roots))
            branch_action = acting if acting is not None else action
            branch_pairs = (
                self.branches.transition_pairs(
                    action=branch_action, top=16
                ) if self.branch_transitions else ()
            )
            steps.append((
                current, action, next_node, float(signals[1]),
                float(signals[0]), done, sources, signature,
                acting, value_roots, row_weight, branch_pairs,
            ))
            if done:
                self.reset()
                # Rows from distinct real episodes can arrive concatenated
                # in one call.  The backward pass below still walks every
                # recorded step, so restore prune protection for every
                # node those steps reference - the reset above dropped the
                # running episode's own set, and a later birth in this call
                # could otherwise delete a node the backward pass needs.
                for step in steps:
                    self._episode_nodes.add(step[0])
                    self._episode_nodes.update(step[6])
                    if step[2] is not None:
                        self._episode_nodes.add(step[2])
                recent_propensities.clear()

        running_pain = 0.0
        running_dopamine = 0.0
        # Short-horizon value attribution bootstraps from the pre-update
        # dopamine value of the state `value_horizon` steps ahead, so the
        # k-step target keeps the return scale of the MC path while
        # crediting only the actions inside the horizon.  The snapshot
        # reads the dopamine channel - the same quantity the cells
        # estimate - not the pain channel.
        dopamine_snapshot = (
            {key: node.dopamine + node.immediate_dopamine
             for key, node in self.graph.nodes.items()}
            if self.value_horizon else None
        )
        for position in range(len(steps) - 1, -1, -1):
            (current, action, next_node, pain, dopamine,
             terminal, sources, signature, acting,
             value_roots, row_weight, branch_pairs) = steps[position]
            return_pain = pain + (
                0.0 if terminal else self.pain_gamma * running_pain
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
            # The transition's outcome is governed by the action issued
            # `lag` steps earlier (`acting`), not by the one just issued -
            # keying the Q table by the just-issued action would attribute
            # the outcome of one action to another.
            self._update_action_edges(
                sources,
                acting if acting is not None else action,
                next_node, return_pain, return_dopamine,
            )
            if self.branch_transitions and branch_pairs:
                effective = acting if acting is not None else action
                weights = {}
                extra = []
                for old, new, act, weight in branch_pairs:
                    if new == next_node and act == effective:
                        weights[old] = weight
                        if old not in sources:
                            extra.append(old)
                if extra:
                    # Decayed recent sources add transition evidence with a
                    # smaller weight; the current sources are already
                    # updated by the unweighted call above.
                    self._update_action_edges(
                        extra, effective, next_node,
                        return_pain, return_dopamine,
                        weights=weights, scale=0.25,
                    )
            self._update_transition_value(
                current, next_node, return_pain, return_dopamine
            )
            self._update_gated(
                current, signature,
                acting if acting is not None else action,
                next_node, return_pain, return_dopamine,
            )
            if acting is not None and value_roots:
                # Sequence-level value: a (feature, acting action) cell
                # learns the discounted dopamine return of the paths that
                # actually followed it, so "aligned for a long survival"
                # becomes readable where the pain hazard only says "local".
                target = (
                    self._k_step_value_target(
                        steps, position, dopamine_snapshot
                    )
                    if dopamine_snapshot is not None else return_dopamine
                )
                self._update_feature_value(
                    value_roots, acting, target, rows[0].get("episode"),
                    weight=row_weight,
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

    def replay_features(self, rows) -> None:
        """Feature-only backfit pass.

        Replays real transitions to update the feature action/pain cells,
        the sequence-value cells and the prediction ledger.  Prototypes,
        action edges, memories and their counts are not touched: the
        prototype layer keeps its single-pass statistics, which isolates
        the "reconnect the feature channels" effect from the doubled
        prototype MC learning of a full second epoch.
        """
        rows = list(rows)
        if not rows or not self.feature_ids:
            return
        scale = self.scales[:self.obs_dim]
        observations = np.asarray([
            np.asarray(row["obs"], dtype=float) / scale for row in rows
        ])
        next_observations = np.asarray([
            np.asarray(row["next_obs"], dtype=float) / scale for row in rows
        ])
        obs_table = self._feature_mask_table(observations)
        horizons = {self._node_horizon(fid) for fid in self.feature_ids}
        horizons.add(self.pain_horizon)
        if self.point_labels:
            point = [
                float(np.asarray(row["signals"], dtype=float)[1] > 0.5)
                for row in rows
            ]
            labels_by_horizon = {
                horizon: point for horizon in sorted(horizons)
            }
        else:
            from node_runtime import future_pain_labels
            labels_by_horizon = {
                horizon: future_pain_labels(rows, horizon)
                for horizon in sorted(horizons)
            }
        value_roots_all = set(self._all_value_roots())
        active_per_row = []
        for index, row in enumerate(rows):
            current_active = [
                node_id
                for node_id, mask in zip(self._feature_all, obs_table)
                if mask is not None and bool(mask[index])
            ]
            active_per_row.append(current_active)
            self._activate_features(current_active)
            history = np.asarray(row["history"], dtype=float).reshape(-1)
            acting = int(round(float(history[self.lag]))) \
                if history.size > self.lag \
                else int(round(float(history[0])))
            state = np.concatenate([
                observations[index], np.zeros(self.lag, dtype=float)
            ])
            by_horizon = {}
            for node_id in current_active:
                by_horizon.setdefault(
                    self._node_horizon(node_id), []
                ).append(node_id)
            for horizon, node_ids in by_horizon.items():
                root_ids = [
                    node_id for node_id in node_ids
                    if node_id in self.feature_ids
                ]
                if root_ids:
                    self._record_prediction(
                        root_ids, acting,
                        labels_by_horizon[horizon][index], state,
                    )
                self._update_feature_pain(
                    node_ids, labels_by_horizon[horizon][index]
                )
                self._update_feature_action(
                    node_ids, acting, labels_by_horizon[horizon][index]
                )
        running = 0.0
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            signals = np.asarray(row["signals"], dtype=float)
            dopamine = float(signals[0])
            running = dopamine + (
                0.0 if bool(row["done"]) else self.gamma * running
            )
            history = np.asarray(row["history"], dtype=float).reshape(-1)
            acting = int(round(float(history[self.lag]))) \
                if history.size > self.lag \
                else int(round(float(history[0])))
            roots = [
                node_id for node_id in active_per_row[index]
                if node_id in value_roots_all
            ]
            if roots:
                self._update_feature_value(
                    roots, acting, running,
                    episode_id=int(row["episode"]),
                )
        self.reset()

    def induce_rules(self, train_states, train_pain,
                     test_states=None, test_pain=None, coverage_states=None):
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
        coverage_obs = None
        if coverage_states is not None:
            coverage_states = np.asarray(coverage_states, dtype=float)
            if (coverage_states.ndim != 2
                    or coverage_states.shape[1] < self.obs_dim):
                raise ValueError(
                    "coverage states must have at least obs_dim columns"
                )
            coverage_obs = coverage_states[:, :self.obs_dim]
        return self.rules.induce(
            train_states[:, :self.obs_dim], train_pain,
            test_obs, test_pain, coverage_x=coverage_obs,
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
                              horizon: Optional[int] = None,
                              channel: str = "pain") -> List[str]:
        """Materialize induced rules as a shared feature graph.

        Primitive and composite nodes are shared by every rule that uses the
        same sub-feature; derivation edges point from a composite to its
        children.  Root nodes keep the learned P(pain | feature) in ``value``
        and, when `horizon` is given, the precursor window their value and
        action cells are trained with.

        `channel` selects which readout trains on the installed roots:
        "pain" roots carry the precursor statistics, "value" roots join the
        sequence-level value readout.  Sub-feature nodes are shared between
        channels - only root bookkeeping differs.
        """
        if channel not in ("pain", "value"):
            raise ValueError("unknown feature channel %r" % channel)
        rules = self.rules.rules if rules is None else list(rules)
        for rule in rules:
            node_id = self._materialize_feature(rule.feature, "feature")
            self._feature_descendants[node_id] = self._subtree_ids(node_id)
            node = self.graph.nodes[node_id]
            if channel == "value":
                if node_id not in self.value_feature_ids:
                    self.value_feature_ids.append(node_id)
                continue
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
        if channel == "value":
            return list(self.value_feature_ids)
        return list(self.feature_ids)

    def uninstall_feature_rules(self, node_ids) -> None:
        """Remove root registrations added by a rejected structure stage.

        The materialized predicate nodes and any shared sub-features stay
        in place (another live root may use them); only the root's
        participation in the readouts and training groups is withdrawn.
        The materialization key is kept, so re-installing the same rule
        later returns the existing node instead of duplicating it.
        """
        for node_id in list(node_ids):
            if node_id in self.feature_ids:
                self.feature_ids.remove(node_id)
            if node_id in self._feature_all:
                self._feature_all.remove(node_id)
            self._feature_descendants.pop(node_id, None)
            self._feature_objects.pop(node_id, None)
            self._feature_children.pop(node_id, None)

    def _node_horizon(self, node_id: str) -> int:
        if self.point_labels:
            # Point-event mode has no window: every root predicts the pain
            # of the state the transition entered.
            return 1
        node = self.graph.nodes.get(node_id)
        if node is not None and node.pain_horizon is not None:
            return int(node.pain_horizon)
        return self.pain_horizon

    def _active_feature_ids(self, state) -> List[str]:
        if not self._feature_all:
            return []
        obs = np.asarray(state, dtype=float).reshape(-1)[:self.obs_dim]
        row = obs.reshape(1, -1)
        if self.pairwise_features:
            row = expand_observation(row)
        return [
            node_id for node_id in self._feature_all
            if node_id in self._feature_objects
            and bool(self._feature_objects[node_id].mask(row)[0])
        ]

    def _table_entry(self, mask_table, node_id, feature, obs):
        """A node's mask from a shared table, or computed on the spot."""
        if mask_table is None:
            return feature.mask(obs)
        return mask_table[self._feature_all.index(node_id)]

    # ---- prediction branch -------------------------------------------------
    def _branch_active_features(self, observation) -> List[str]:
        """Feature roots active at an observation, for branch comparisons."""
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.shape[0] < self.obs_dim:
            return []
        encoded = obs[:self.obs_dim] / self.scales[:self.obs_dim]
        active = self._active_feature_ids(encoded)
        return [node_id for node_id in active if node_id in self.feature_ids]

    def set_prediction(self, observation) -> List[str]:
        """Record the roots the planner expects at the next state."""
        node_ids = self.prediction_nodes(observation)
        self.prediction_branch.activate(
            node_ids, [1.0] * len(node_ids)
        )
        self._last_prediction = set(node_ids)
        return node_ids

    def active_feature_roots(self, observation) -> List[str]:
        """Public read of the roots active at an observation."""
        return self._branch_active_features(observation)

    def nearest_prototype(self, observation):
        """Nearest prototype to an observation with zero pending actions."""
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.shape[0] < self.obs_dim or not self._prototype_ids:
            return None, float("inf")
        state = np.concatenate([
            obs[:self.obs_dim], np.zeros(self.lag, dtype=float)
        ]) / self.scales
        return self._nearest(state)

    def prediction_nodes(self, observation) -> List[str]:
        """Nodes whose prediction ability is tracked at this observation.

        Feature roots are the preferred granularity.  When a world's roots
        do not fire (a known held-out failure mode), fall back to the
        nearest prototype so the prediction channel still has an anchor.
        """
        roots = self.active_feature_roots(observation)
        if roots:
            return roots
        node_id, _distance = self.nearest_prototype(observation)
        return [node_id] if node_id is not None else []

    def reset_prediction_stats(self) -> None:
        self.prediction_stats.clear()
        self._last_prediction = set()
        self.prediction_branch.reset()

    def observe_prediction(self, observation) -> dict:
        """Compare the recorded prediction with the actually entered state."""
        actual = set(self.prediction_nodes(observation))
        predicted = set(self._last_prediction)
        for node_id in predicted:
            stats = self.prediction_stats.setdefault(
                node_id, [0.0, 0.0]
            )
            stats[1] += 1.0
            if node_id in actual:
                stats[0] += 1.0
        hits = len(predicted & actual)
        self.prediction_branch.reset()
        self._last_prediction = set()
        return dict(
            predicted=len(predicted), actual=len(actual), hits=hits,
        )

    def prediction_credibility(self, node_id: str,
                               prior: float = 2.0,
                               base: float = 0.5) -> float:
        hits, count = self.prediction_stats.get(
            node_id, [0.0, 0.0]
        )
        return float((hits + prior * base) / (count + prior))

    def prediction_learning_rate(self, node_id: str,
                                 prior: float = 2.0) -> float:
        """How fast this node's prediction estimate should still move."""
        _hits, count = self.prediction_stats.get(
            node_id, [0.0, 0.0]
        )
        return float(1.0 / (count + prior))

    def prediction_summary(self) -> dict:
        if not self.prediction_stats:
            return dict(
                nodes=0, mean_credibility=0.0, mean_learning_rate=0.0,
                hits=0.0, count=0.0,
            )
        hits = sum(stat[0] for stat in self.prediction_stats.values())
        count = sum(stat[1] for stat in self.prediction_stats.values())
        credibility = [
            self.prediction_credibility(node_id)
            for node_id in self.prediction_stats
        ]
        learning = [
            self.prediction_learning_rate(node_id)
            for node_id in self.prediction_stats
        ]
        return dict(
            nodes=len(self.prediction_stats),
            mean_credibility=float(np.mean(credibility)),
            mean_learning_rate=float(np.mean(learning)),
            hits=float(hits), count=float(count),
        )

