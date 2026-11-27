"""Role-specific strength states over one shared node graph.

A branch owns only a sparse ``node_id -> strength`` map; nodes and edges
remain shared.  This is the layer the flat runtime was missing: the current
observation, the recent trace, a prediction and a slow value can coexist on
the same node without overwriting one another's strength.

The base class defines the lifecycle (activate, decay, read, compare,
snapshot).  Concrete branches differ only in how strength is updated and how
fast it fades, which is exactly the role the caller instantiates:

    CurrentBranch      instantaneous state (overwrite each step)
    RecentBranch       exponentially decaying trace
    PredictionBranch   expected activation, compared against current
    ValueBranch        slow pooled value (1/n or fixed learning rate)

Branch strengths are sparse dictionaries, so adding branches does not copy
the graph and costs only the active nodes.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class NodeBranch:
    def __init__(self, name: str, graph=None, decay: float = 0.0,
                 learn_rate: float = 1.0, timescale: str = "fast"):
        if timescale not in ("fast", "slow"):
            raise ValueError("timescale must be 'fast' or 'slow'")
        self.name = str(name)
        self.graph = graph
        self.decay_rate = float(decay)
        self.learn_rate = float(learn_rate)
        self.timescale = timescale
        self.strength: Dict[str, float] = {}

    # ---- lifecycle --------------------------------------------------------
    def activate(self, node_ids: Iterable[str],
                 values: Optional[Sequence[float]] = None,
                 weight: float = 1.0) -> None:
        node_ids = list(node_ids)
        if values is None:
            values = [1.0] * len(node_ids)
        for node_id, value in zip(node_ids, values):
            self.strength[str(node_id)] = float(value) * float(weight)

    def step(self) -> None:
        if self.decay_rate <= 0.0:
            return
        for node_id in list(self.strength):
            value = self.strength[node_id] * self.decay_rate
            if value < 1e-6:
                del self.strength[node_id]
            else:
                self.strength[node_id] = value

    def read(self, node_id: str) -> float:
        return float(self.strength.get(str(node_id), 0.0))

    def top(self, count: int = 5) -> List[Tuple[str, float]]:
        return sorted(
            self.strength.items(), key=lambda item: -item[1]
        )[:max(int(count), 0)]

    def reset(self) -> None:
        self.strength.clear()

    # ---- comparisons ------------------------------------------------------
    def compare(self, other: "NodeBranch") -> dict:
        keys = set(self.strength) | set(other.strength)
        errors = {
            key: abs(self.read(key) - other.read(key)) for key in keys
        }
        overlap = set(self.strength) & set(other.strength)
        norm_self = np.sqrt(sum(v * v for v in self.strength.values()))
        norm_other = np.sqrt(sum(v * v for v in other.strength.values()))
        dot = sum(
            self.strength[key] * other.strength[key] for key in overlap
        )
        cosine = (
            float(dot / (norm_self * norm_other))
            if norm_self > 0.0 and norm_other > 0.0 else 0.0
        )
        return dict(
            mean_error=(
                float(np.mean(list(errors.values()))) if errors else 0.0
            ),
            max_error=float(max(errors.values())) if errors else 0.0,
            overlap=len(overlap),
            cosine=cosine,
            errors=errors,
        )

    def snapshot(self) -> dict:
        return dict(
            name=self.name, timescale=self.timescale,
            decay=self.decay_rate, learn_rate=self.learn_rate,
            active=len(self.strength),
            top=[(node_id, float(value)) for node_id, value in self.top(5)],
        )


class CurrentBranch(NodeBranch):
    """Instantaneous state: each activation replaces the previous one."""

    def __init__(self, name: str = "current", graph=None):
        super().__init__(name, graph=graph, decay=0.0, timescale="fast")

    def activate(self, node_ids, values=None, weight=1.0) -> None:
        self.strength.clear()
        super().activate(node_ids, values=values, weight=weight)


class RecentBranch(NodeBranch):
    """Exponentially decaying trace of recently active nodes."""

    def __init__(self, name: str = "recent", graph=None, decay: float = 0.8):
        super().__init__(
            name, graph=graph, decay=decay, learn_rate=1.0,
            timescale="fast",
        )

    def activate(self, node_ids, values=None, weight=1.0) -> None:
        node_ids = list(node_ids)
        if values is None:
            values = [1.0] * len(node_ids)
        for node_id, value in zip(node_ids, values):
            current = self.strength.get(str(node_id), 0.0)
            self.strength[str(node_id)] = min(
                1.0, current + float(value) * float(weight)
            )


class PredictionBranch(NodeBranch):
    """Expected activation from a rollout; compared against current."""

    def __init__(self, name: str = "prediction", graph=None):
        super().__init__(name, graph=graph, decay=0.0, timescale="fast")

    def activate(self, node_ids, values=None, weight=1.0) -> None:
        self.strength.clear()
        super().activate(node_ids, values=values, weight=weight)


class ValueBranch(NodeBranch):
    """Slow pooled value with a fixed or 1/n learning rate."""

    def __init__(self, name: str = "value", graph=None,
                 learn_rate: float = 0.05):
        super().__init__(
            name, graph=graph, decay=0.0, learn_rate=learn_rate,
            timescale="slow",
        )
        self.support: Dict[str, int] = {}

    def activate(self, node_ids, values=None, weight=1.0) -> None:
        node_ids = list(node_ids)
        if values is None:
            values = [1.0] * len(node_ids)
        for node_id, value in zip(node_ids, values):
            node_id = str(node_id)
            self.support[node_id] = self.support.get(node_id, 0) + 1
            rate = (
                self.learn_rate if self.learn_rate > 0.0
                else 1.0 / self.support[node_id]
            )
            current = self.strength.get(node_id, 0.0)
            self.strength[node_id] = (
                current + rate * (float(value) - current)
            )


class BranchBank:
    """A named set of branches sharing one graph."""

    def __init__(self, graph=None):
        self.graph = graph
        self.branches: Dict[str, NodeBranch] = {}

    def add(self, branch: NodeBranch) -> NodeBranch:
        if branch.name in self.branches:
            raise ValueError("duplicate branch name: %s" % branch.name)
        self.branches[branch.name] = branch
        return branch

    def get(self, name: str) -> NodeBranch:
        return self.branches[name]

    def activate(self, name: str, node_ids, values=None,
                 weight: float = 1.0) -> None:
        self.branches[name].activate(node_ids, values=values, weight=weight)

    def step(self) -> None:
        for branch in self.branches.values():
            branch.step()

    def reset(self) -> None:
        for branch in self.branches.values():
            branch.reset()

    # ---- cross-branch operations -----------------------------------------
    def transition_pairs(self, action: Optional[int] = None,
                         top: int = 32, threshold: float = 1e-6
                         ) -> List[Tuple[str, str, int, float]]:
        """Pair recent(old) and current(new) strengths for edge learning.

        The product is the weight of the evidence: a pair matters only when
        the old node was recently active and the new node is active now.
        """
        recent = self.branches.get("recent")
        current = self.branches.get("current")
        if recent is None or current is None:
            return []
        pairs = []
        for old_id, old_strength in recent.strength.items():
            for new_id, new_strength in current.strength.items():
                weight = float(old_strength) * float(new_strength)
                if weight >= threshold:
                    pairs.append((old_id, new_id, action, weight))
        pairs.sort(key=lambda item: -item[3])
        return pairs[:max(int(top), 0)]

    def prediction_error(self) -> Optional[dict]:
        prediction = self.branches.get("prediction")
        current = self.branches.get("current")
        if prediction is None or current is None:
            return None
        return prediction.compare(current)

    def snapshot(self) -> dict:
        return {
            name: branch.snapshot()
            for name, branch in self.branches.items()
        }
