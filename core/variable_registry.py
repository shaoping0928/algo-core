"""Typed variables and their relations.

Every learned quantity is a variable with an estimate, effective support,
variance, a timescale and free metadata.  Relations say how variables depend
on each other:

    part_of   a finer memory is part of a coarser one
    tracks    a fast variable tracks the same quantity as a slow one
    competes  siblings partition the same region at one granularity

Credibility and the effective learning rate are derived from those
relations rather than stored as independent constants:

    credibility(v)  = min(1, own_credibility * (1 + parent_gain * parent_cred))
    own_credibility = support(v) / (support(v) + base_prior)
    learning_rate(v)= 1 / (support(v) + base_prior)                 (slow)
                    = fast_base * (1 - persistence/2)
                      * (0.5 + 0.5 * parent_credibility)          (fast)

A variable with credible parents is credible sooner; a fast variable inside
a credible context may move quickly, while a persistent one is deliberately
slowed down.  This is the bookkeeping the flat per-cell counters lacked:
the numbers now have a neighbourhood.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Variable:
    variable_id: str
    kind: str
    estimate: np.ndarray
    support: float = 0.0
    variance: float = 0.0
    timescale: str = "slow"
    metadata: dict = field(default_factory=dict)


class VariableRegistry:
    def __init__(self, base_prior: float = 4.0,
                 parent_gain: float = 1.0,
                 fast_base: float = 0.5,
                 slow_scale: float = 1.0):
        self.base_prior = float(base_prior)
        self.parent_gain = float(parent_gain)
        self.fast_base = float(fast_base)
        self.slow_scale = float(slow_scale)
        self.variables: Dict[str, Variable] = {}
        self.relations: List[Tuple[str, str, str, float]] = []

    # ---- construction -----------------------------------------------------
    def add(self, variable_id: str, kind: str, estimate,
            support: float = 0.0, variance: float = 0.0,
            timescale: str = "slow",
            metadata: Optional[dict] = None) -> Variable:
        if timescale not in ("slow", "fast"):
            raise ValueError("timescale must be 'slow' or 'fast'")
        estimate = np.asarray(estimate, dtype=float).reshape(-1)
        variable = Variable(
            variable_id=variable_id, kind=kind, estimate=estimate,
            support=float(support), variance=float(variance),
            timescale=timescale, metadata=dict(metadata or {}),
        )
        self.variables[variable_id] = variable
        return variable

    def relate(self, source: str, target: str, kind: str,
               weight: float = 1.0) -> None:
        if source == target:
            raise ValueError("a variable cannot relate to itself")
        if kind not in ("part_of", "tracks", "competes", "supports",
                        "summarizes"):
            raise ValueError("unknown relation kind")
        if source not in self.variables or target not in self.variables:
            raise KeyError("both variables must exist before relating them")
        relation = (source, target, kind, float(weight))
        if relation not in self.relations:
            self.relations.append(relation)

    # ---- relation queries -------------------------------------------------
    def _incoming(self, variable_id: str, kinds) -> List[Tuple[str, float]]:
        return [
            (source, weight)
            for source, target, kind, weight in self.relations
            if target == variable_id and kind in kinds
        ]

    def _outgoing(self, variable_id: str, kinds) -> List[Tuple[str, float]]:
        return [
            (target, weight)
            for source, target, kind, weight in self.relations
            if source == variable_id and kind in kinds
        ]

    def parents(self, variable_id: str) -> List[Tuple[str, float]]:
        # A `part_of`/`tracks` relation is stored child -> parent.
        return self._outgoing(variable_id, ("part_of", "tracks"))

    def children(self, variable_id: str) -> List[Tuple[str, float]]:
        return self._incoming(variable_id, ("part_of", "tracks"))

    def siblings(self, variable_id: str) -> List[str]:
        result = []
        for parent, _weight in self.parents(variable_id):
            for child, _weight_out in self.children(parent):
                if child != variable_id:
                    result.append(child)
        return result

    # ---- derived attributes ----------------------------------------------
    def _parent_credibility(self, variable_id: str) -> float:
        parents = self.parents(variable_id)
        if not parents:
            return 0.0
        total = sum(weight for _parent, weight in parents)
        if total <= 0.0:
            return 0.0
        return sum(
            self.credibility(parent) * weight
            for parent, weight in parents
        ) / total

    def prior(self, variable_id: str) -> float:
        # Pseudo-count for a variable's own estimate.  The neighbourhood
        # enters through credibility and the fast learning rate, not by
        # inflating this prior.
        return self.base_prior

    def credibility(self, variable_id: str) -> float:
        variable = self.variables[variable_id]
        denominator = variable.support + self.base_prior
        own = (
            float(variable.support / denominator) if denominator > 0.0 else 0.0
        )
        parent_credibility = self._parent_credibility(variable_id)
        return float(min(
            1.0, own * (1.0 + self.parent_gain * parent_credibility)
        ))

    def persistence(self, variable_id: str) -> float:
        return float(
            self.variables[variable_id].metadata.get("persistence", 0.0)
        )

    def _parent_support(self, variable_id: str) -> float:
        parents = self.parents(variable_id)
        if not parents:
            return 0.0
        total = sum(weight for _parent, weight in parents)
        if total <= 0.0:
            return 0.0
        return sum(
            self.variables[parent].support * self.credibility(parent) * weight
            for parent, weight in parents
        ) / total

    def learning_rate(self, variable_id: str) -> float:
        variable = self.variables[variable_id]
        if variable.timescale == "slow":
            # A well-supported parent contributes prior evidence, so the
            # child's estimate is deliberately stabilized (smaller step).
            effective = variable.support + self.parent_gain * (
                self._parent_support(variable_id)
            )
            denominator = effective + self.base_prior
            return self.slow_scale / denominator if denominator > 0.0 else 1.0
        parent_credibility = self._parent_credibility(variable_id)
        persistence = self.persistence(variable_id)
        return self.fast_base * (
            1.0 - 0.5 * persistence
        ) * (0.5 + 0.5 * parent_credibility)

    # ---- updates ----------------------------------------------------------
    def update(self, variable_id: str, value, weight: float = 1.0,
               persistence: Optional[float] = None) -> Variable:
        variable = self.variables[variable_id]
        observation = np.asarray(value, dtype=float).reshape(-1)
        if observation.shape != variable.estimate.shape:
            raise ValueError("estimate dimension mismatch")
        weight = float(weight)
        if weight <= 0.0:
            return variable
        old_support = variable.support
        new_support = old_support + weight
        delta = observation - variable.estimate
        variable.estimate = variable.estimate + delta * (weight / new_support)
        variable.variance = (
            variable.variance * old_support
            + weight * float(np.mean((observation - variable.estimate) ** 2))
        ) / new_support
        variable.support = new_support
        if persistence is not None:
            variable.metadata["persistence"] = float(persistence)
        return variable

    # ---- tree / direction registration -----------------------------------
    def register_tree(self, tree) -> None:
        """Mirror a MemoryTree into variables and part_of relations."""
        for node_id, node in tree.nodes.items():
            variable_id = "mem:%d" % node_id
            level = int(node.level)
            if variable_id in self.variables:
                # The tree is the source of truth; mirror its statistics
                # instead of adding them a second time.
                variable = self.variables[variable_id]
                variable.estimate = node.mean.copy()
                variable.support = float(node.count)
                variable.variance = node.variance()
                variable.metadata["level"] = level
                variable.metadata["leaf"] = bool(node.leaf)
            else:
                self.add(
                    variable_id, "context", node.mean,
                    support=node.count, variance=node.variance(),
                    timescale="fast" if level == 0 else "slow",
                    metadata={"level": level, "leaf": bool(node.leaf)},
                )
        for child_id, parent_id in tree.parent.items():
            child = "mem:%d" % child_id
            parent = "mem:%d" % parent_id
            if child in self.variables and parent in self.variables:
                self.relate(child, parent, "part_of")
        for node_id, node in tree.nodes.items():
            children = [
                "mem:%d" % child for child in node.children
                if "mem:%d" % child in self.variables
            ]
            for index, first in enumerate(children):
                for second in children[index + 1:]:
                    self.relate(first, second, "competes")

    def register_direction(self, index: int, estimate,
                           support: float, variance: float,
                           persistence: float) -> str:
        variable_id = "dir:%d" % int(index)
        if variable_id in self.variables:
            self.update(
                variable_id, estimate, weight=max(support, 1e-9),
                persistence=persistence,
            )
            self.variables[variable_id].metadata["persistence"] = float(
                persistence
            )
        else:
            self.add(
                variable_id, "direction", estimate, support=support,
                variance=variance, timescale="slow",
                metadata={"persistence": float(persistence), "index": index},
            )
        return variable_id

    def register_summaries(self, tree) -> None:
        """Register each parent's keywords and its summarizes relations."""
        for node_id, node in tree.nodes.items():
            if node.leaf or node.keywords is None:
                continue
            parent_variable = "mem:%d" % node_id
            if parent_variable not in self.variables:
                continue
            persistence = tree.keyword_persistence(node_id)
            for item in persistence:
                keyword_id = "kw:%d:%d" % (node_id, item["index"])
                direction = node.keywords[item["index"]]
                if keyword_id in self.variables:
                    variable = self.variables[keyword_id]
                    variable.estimate = direction.copy()
                    variable.support = float(
                        node.count * max(item["weight"], 1e-9)
                    )
                    variable.metadata["persistence"] = float(
                        item["similarity"]
                    )
                else:
                    self.add(
                        keyword_id, "keyword", direction,
                        support=node.count * max(item["weight"], 1e-9),
                        variance=float(node.residual),
                        timescale="slow",
                        metadata={
                            "level": int(node.level),
                            "persistence": float(item["similarity"]),
                            "persistent": bool(item["persistent"]),
                        },
                    )
                self.relate(keyword_id, parent_variable, "part_of")
            for child_id in node.children:
                child_variable = "mem:%d" % child_id
                if child_variable in self.variables:
                    self.relate(
                        parent_variable, child_variable, "summarizes"
                    )

    def snapshot(self) -> dict:
        return {
            variable_id: dict(
                kind=variable.kind,
                support=round(float(variable.support), 4),
                variance=round(float(variable.variance), 6),
                credibility=round(self.credibility(variable_id), 4),
                learning_rate=round(self.learning_rate(variable_id), 6),
                persistence=round(self.persistence(variable_id), 4),
                timescale=variable.timescale,
                metadata=dict(variable.metadata),
            )
            for variable_id, variable in self.variables.items()
        }
