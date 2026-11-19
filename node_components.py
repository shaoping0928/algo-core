"""Runtime components with tri-state concept nodes.

Numeric operators are computations bound to a base node, not graph nodes:

    difference   x(t) - x(t-1)
    capacitor    (1 - beta) u(t-1) + beta x(t)
    resistor     x(t) - u(t)   (deviation from a leaky baseline)

Each component emits a sign state (+1 activate, 0 hold, -1 inhibit).  The sign
is represented by three ordinary concept nodes in the graph.  Those nodes carry
the concept/state identity: they can be associated, form transitions, join
memories, receive valence and be persisted.  The numeric component itself is
runtime-only and never enters `NodeGraph.nodes`.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from node_graph import CONCEPT, NodeGraph


SIGN_SUFFIX = {1: "+", 0: "0", -1: "-"}


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    base_node: str
    kind: str
    params: Tuple[float, ...] = ()
    threshold: float = 0.2
    hysteresis: float = 0.0
    smoothing: float = 0.5


class RuntimeComponent:
    """A small numeric filter; it has no graph identity."""

    def __init__(self, kind: str, params: Sequence[float] = ()):
        kind = str(kind)
        if kind not in ("difference", "capacitor", "resistor"):
            raise ValueError("unknown component kind %s" % kind)
        self.kind = kind
        self.params = tuple(float(value) for value in params)
        if kind in ("capacitor", "resistor"):
            if not self.params:
                self.params = (0.2,)
            if not 0.0 < self.params[0] <= 1.0:
                raise ValueError("leak beta must be in (0, 1]")
        self.previous_input = 0.0
        self.state = 0.0
        self.value = 0.0

    def update(self, x: float) -> float:
        x = float(x)
        if self.kind == "difference":
            value = x - self.previous_input
        elif self.kind == "capacitor":
            beta = self.params[0]
            self.state = (1.0 - beta) * self.state + beta * x
            value = self.state
        else:  # resistor: deviation from a leaky baseline
            beta = self.params[0]
            self.state = (1.0 - beta) * self.state + beta * x
            value = x - self.state
        self.previous_input = x
        self.value = float(value)
        return self.value


def classify(value: float, previous: int, threshold: float,
             hysteresis: float) -> int:
    """Clamp to [-1, 1] and round to +1 / 0 / -1 with a hold band.

    With the default threshold 0.2 this is exactly:

        [-1, -0.2] -> -1     (-0.2, 0.2) -> 0     [0.2, 1] -> +1

    `hysteresis` widens the hold band on the way back, so a signal that has
    already crossed a trigger does not chatter around the boundary.
    """
    value = max(-1.0, min(1.0, float(value)))
    threshold = max(float(threshold), 0.0)
    hysteresis = max(float(hysteresis), 0.0)
    if value > threshold + hysteresis:
        return 1
    if value < -threshold - hysteresis:
        return -1
    if abs(value) <= threshold:
        return 0
    return int(previous)


class ComponentBank:
    """Bind runtime components to graph nodes and expose their sign nodes."""

    def __init__(self, graph: NodeGraph, rate: float = 0.2,
                 threshold: float = 0.2, hysteresis: float = 0.0,
                 smoothing: float = 0.5):
        if rate <= 0.0:
            raise ValueError("rate must be positive")
        if not 0.0 < float(smoothing) <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.graph = graph
        self.rate = float(rate)
        self.threshold = float(threshold)
        self.hysteresis = float(hysteresis)
        self.smoothing = float(smoothing)
        self.specs: Dict[str, ComponentSpec] = {}
        self.components: Dict[str, RuntimeComponent] = {}
        self.sign_nodes: Dict[str, Dict[int, str]] = {}
        self.previous_sign: Dict[str, int] = {}
        self.previous_sign_node: Dict[str, Optional[str]] = {}
        self.smoothed: Dict[str, float] = {}
        self.sign_counts: Dict[str, Dict[int, int]] = {}
        self.sign_pain_sum: Dict[str, Dict[int, float]] = {}
        self.sign_pain_count: Dict[str, Dict[int, int]] = {}
        self.last_active: List[str] = []

    def bind(self, base_node: str, kind: str = "difference",
             params: Sequence[float] = (), component_id: Optional[str] = None,
             threshold: Optional[float] = None,
             hysteresis: Optional[float] = None,
             smoothing: Optional[float] = None) -> str:
        if base_node not in self.graph.nodes:
            raise KeyError(base_node)
        component_id = (
            str(component_id) if component_id is not None
            else "%s:%s" % (base_node, kind)
        )
        if component_id in self.specs:
            raise ValueError("duplicate component id %s" % component_id)
        component = RuntimeComponent(kind, params)
        smoothing = (
            self.smoothing if smoothing is None else float(smoothing)
        )
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        spec = ComponentSpec(
            id=component_id,
            base_node=base_node,
            kind=component.kind,
            params=component.params,
            threshold=(
                self.threshold if threshold is None else float(threshold)
            ),
            hysteresis=(
                self.hysteresis if hysteresis is None else float(hysteresis)
            ),
            smoothing=smoothing,
        )
        nodes = {}
        for sign, suffix in SIGN_SUFFIX.items():
            node_id = "component:%s:%s" % (component_id, suffix)
            self.graph.add_node(
                CONCEPT,
                node_id=node_id,
                role="component_state",
                component_id=component_id,
                sign=sign,
            )
            nodes[sign] = node_id
        self.specs[component_id] = spec
        self.components[component_id] = component
        self.sign_nodes[component_id] = nodes
        self.previous_sign[component_id] = 0
        self.previous_sign_node[component_id] = None
        self.smoothed[component_id] = 0.0
        self.sign_counts[component_id] = {1: 0, 0: 0, -1: 0}
        self.sign_pain_sum[component_id] = {1: 0.0, 0: 0.0, -1: 0.0}
        self.sign_pain_count[component_id] = {1: 0, 0: 0, -1: 0}
        return component_id

    def step(self, activations: Dict[str, float]) -> Dict[str, int]:
        """Update bound components and activate their sign nodes."""
        result: Dict[str, int] = {}
        for component_id, spec in self.specs.items():
            if spec.base_node not in activations:
                continue
            x = float(activations[spec.base_node])
            self.graph.activate([spec.base_node], [x])
            raw = self.components[component_id].update(x)
            previous_smoothed = self.smoothed.get(component_id, 0.0)
            value = (
                (1.0 - spec.smoothing) * previous_smoothed
                + spec.smoothing * raw
            )
            value = max(-1.0, min(1.0, value))
            self.smoothed[component_id] = value
            previous = self.previous_sign[component_id]
            sign = classify(
                value, previous, spec.threshold, spec.hysteresis
            )
            self.previous_sign[component_id] = sign
            node_id = self.sign_nodes[component_id][sign]
            self.sign_counts[component_id][sign] += 1
            self.graph.activate([node_id], [1.0])
            for other_sign, other in self.sign_nodes[component_id].items():
                if other_sign != sign:
                    self.graph.activate([other], [0.0])
            self.graph.observe_association(
                [spec.base_node, node_id], rate=self.rate
            )
            previous_node = self.previous_sign_node[component_id]
            if previous_node is not None and previous_node != node_id:
                self.graph.observe_transition(
                    previous_node, node_id, rate=self.rate
                )
            self.previous_sign_node[component_id] = node_id
            result[component_id] = sign
        if result:
            self.last_active = [
                self.sign_nodes[component_id][sign]
                for component_id, sign in result.items()
            ]
        return result

    def active_sign_nodes(self, base_node: Optional[str] = None) -> List[str]:
        if base_node is None:
            return list(self.last_active)
        return [
            self.sign_nodes[component_id][self.previous_sign[component_id]]
            for component_id, spec in self.specs.items()
            if spec.base_node == base_node
        ]

    def add_memory(self, component_ids: Optional[Iterable[str]] = None) -> str:
        ids = (
            list(component_ids) if component_ids is not None
            else list(self.specs)
        )
        members = [
            self.sign_nodes[component_id][self.previous_sign[component_id]]
            for component_id in ids
        ]
        return self.graph.add_memory(members)

    def observe_valence(self, pain: float, dopamine: float = 0.0,
                        rate: Optional[float] = None,
                        active: Optional[Iterable[str]] = None) -> None:
        """Update the active sign nodes from world valence signals."""
        rate = self.rate if rate is None else float(rate)
        ids = list(active) if active is not None else self.active_sign_nodes()
        pain = float(pain)
        dopamine = float(dopamine)
        for node_id in ids:
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            node.value += rate * (pain - node.value)
            node.dopamine += rate * (dopamine - node.dopamine)
            node.count += 1
            component_id = node.component_id
            if component_id in self.sign_pain_sum:
                sign = int(node.sign)
                self.sign_pain_sum[component_id][sign] += pain
                self.sign_pain_count[component_id][sign] += 1

    # ---- persistence -------------------------------------------------------
    def utility(self, component_id: str) -> float:
        """How much the sign state explains pain (weighted variance of means)."""
        means = []
        weights = []
        for sign in (1, 0, -1):
            count = self.sign_pain_count.get(component_id, {}).get(sign, 0)
            if count <= 0:
                continue
            means.append(
                self.sign_pain_sum[component_id][sign] / count
            )
            weights.append(count)
        if len(means) < 2:
            return 0.0
        total = float(sum(weights))
        if total <= 0.0:
            return 0.0
        mean = sum(
            weight * value for weight, value in zip(weights, means)
        ) / total
        return sum(
            weight * (value - mean) ** 2
            for weight, value in zip(weights, means)
        ) / total

    def observations(self, component_id: str) -> int:
        return int(sum(self.sign_counts.get(component_id, {}).values()))

    def remove(self, component_id: str) -> bool:
        if component_id not in self.specs:
            return False
        nodes = list(self.sign_nodes.get(component_id, {}).values())
        for node_id in nodes:
            if node_id in self.graph.nodes:
                self.graph._remove_node(node_id)
        self.specs.pop(component_id, None)
        self.components.pop(component_id, None)
        self.sign_nodes.pop(component_id, None)
        self.previous_sign.pop(component_id, None)
        self.previous_sign_node.pop(component_id, None)
        self.smoothed.pop(component_id, None)
        self.sign_counts.pop(component_id, None)
        self.sign_pain_sum.pop(component_id, None)
        self.sign_pain_count.pop(component_id, None)
        self.last_active = [
            node_id for node_id in self.last_active
            if node_id in self.graph.nodes
        ]
        return True

    def prune(self, min_observations: int = 20,
              quantile: float = 0.25) -> List[str]:
        """Forget components whose sign states do not separate pain."""
        candidates = [
            component_id for component_id in self.specs
            if self.observations(component_id) >= min_observations
        ]
        if len(candidates) < 2:
            return []
        utilities = np.asarray([
            self.utility(component_id) for component_id in candidates
        ])
        threshold = float(np.quantile(utilities, quantile))
        removed = []
        for component_id in candidates:
            if self.utility(component_id) <= threshold:
                if self.remove(component_id):
                    removed.append(component_id)
        return removed

    def state_dict(self) -> dict:
        return dict(
            rate=self.rate,
            threshold=self.threshold,
            hysteresis=self.hysteresis,
            specs=[
                dict(
                    id=spec.id,
                    base_node=spec.base_node,
                    kind=spec.kind,
                    params=list(spec.params),
                    threshold=spec.threshold,
                    hysteresis=spec.hysteresis,
                    smoothing=spec.smoothing,
                )
                for spec in self.specs.values()
            ],
            signs=dict(self.previous_sign),
            smoothed=dict(self.smoothed),
            sign_counts={
                component_id: {
                    str(sign): count
                    for sign, count in counts.items()
                }
                for component_id, counts in self.sign_counts.items()
            },
            sign_pain_sum={
                component_id: {
                    str(sign): value
                    for sign, value in values.items()
                }
                for component_id, values in self.sign_pain_sum.items()
            },
            sign_pain_count={
                component_id: {
                    str(sign): count
                    for sign, count in counts.items()
                }
                for component_id, counts in self.sign_pain_count.items()
            },
            sign_nodes={
                component_id: {
                    str(sign): node_id for sign, node_id in nodes.items()
                }
                for component_id, nodes in self.sign_nodes.items()
            },
            values={
                node_id: dict(
                    value=self.graph.nodes[node_id].value,
                    dopamine=self.graph.nodes[node_id].dopamine,
                    count=self.graph.nodes[node_id].count,
                )
                for nodes in self.sign_nodes.values()
                for node_id in nodes.values()
                if node_id in self.graph.nodes
            },
            graph=self.graph.to_state(),
        )

    def load_state_dict(self, state: dict) -> "ComponentBank":
        self.rate = float(state.get("rate", self.rate))
        self.threshold = float(state.get("threshold", self.threshold))
        self.hysteresis = float(state.get("hysteresis", self.hysteresis))
        self.smoothing = float(state.get("smoothing", self.smoothing))
        self.graph = NodeGraph.from_state(state["graph"])
        self.specs = {}
        self.components = {}
        self.sign_nodes = {}
        self.previous_sign = {}
        self.previous_sign_node = {}
        self.smoothed = {}
        self.sign_counts = {}
        self.sign_pain_sum = {}
        self.sign_pain_count = {}
        sign_nodes = state.get("sign_nodes", {})
        signs = state.get("signs", {})
        smoothed = state.get("smoothed", {})
        for data in state.get("specs", []):
            component_id = data["id"]
            spec = ComponentSpec(
                id=component_id,
                base_node=data["base_node"],
                kind=data["kind"],
                params=tuple(float(v) for v in data.get("params", [])),
                threshold=float(data.get("threshold", self.threshold)),
                hysteresis=float(data.get("hysteresis", self.hysteresis)),
                smoothing=float(data.get("smoothing", self.smoothing)),
            )
            self.specs[component_id] = spec
            self.components[component_id] = RuntimeComponent(
                spec.kind, spec.params
            )
            self.smoothed[component_id] = float(
                smoothed.get(component_id, 0.0)
            )
            self.sign_nodes[component_id] = {
                int(sign): node_id
                for sign, node_id in sign_nodes.get(
                    component_id, {}
                ).items()
            }
            sign = int(signs.get(component_id, 0))
            self.previous_sign[component_id] = sign
            self.previous_sign_node[component_id] = self.sign_nodes[
                component_id
            ].get(sign)
            self.sign_counts[component_id] = {
                int(key): int(value)
                for key, value in state.get("sign_counts", {}).get(
                    component_id, {}
                ).items()
            }
            self.sign_pain_sum[component_id] = {
                int(key): float(value)
                for key, value in state.get("sign_pain_sum", {}).get(
                    component_id, {}
                ).items()
            }
            self.sign_pain_count[component_id] = {
                int(key): int(value)
                for key, value in state.get("sign_pain_count", {}).get(
                    component_id, {}
                ).items()
            }
        for node_id, values in state.get("values", {}).items():
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            node.value = float(values.get("value", node.value))
            node.dopamine = float(values.get("dopamine", node.dopamine))
            node.count = int(values.get("count", node.count))
        self.last_active = [
            self.sign_nodes[component_id][sign]
            for component_id, sign in self.previous_sign.items()
            if sign in self.sign_nodes[component_id]
        ]
        return self
