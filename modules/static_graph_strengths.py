"""Learned continuous strengths for the non-temporal node graph.

The runtime's association edges are only updated when ``association_rate`` is
positive, and membership edges are created with zero weight, so a static
transition over the default graph can only follow derivation edges.  This module
learns the missing strengths from realized co-activation and exposes them as a
read-only overlay:

* association  Jaccard co-activation of two nodes across states, gated by a
  minimum co-occurrence count;
* membership   a structural binding strength, optionally supplied per edge.

Nothing here writes to the runtime graph.  ``endogenous_transition`` accepts the
overlay and uses it instead of the graph's raw edge weights, so the integration
can reach multiple hops without changing any decision path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Tuple


def _pair(first, second):
    return (first, second) if first <= second else (second, first)


@dataclass
class StaticGraphStrengths:
    """Read-only association and membership strengths."""

    associations: dict = field(default_factory=dict)
    membership: dict = field(default_factory=dict)
    support: dict = field(default_factory=dict)
    min_support: int = 2
    normalized: bool = False

    def association(self, first, second) -> float:
        return float(self.associations.get(_pair(first, second), 0.0))

    def associations_of(self, node):
        result = []
        for (first, second), strength in self.associations.items():
            if first == node:
                result.append((second, float(strength)))
            elif second == node:
                result.append((first, float(strength)))
        if self.normalized and result:
            total = sum(strength for _other, strength in result)
            if total > 0.0:
                result = [
                    (other, strength / total)
                    for other, strength in result
                ]
        result.sort(key=lambda item: (-item[1], item[0]))
        return tuple(result)

    def member_strength(self, memory, member) -> float:
        return float(self.membership.get((memory, member), 0.0))

    def snapshot(self):
        return dict(
            associations=int(len(self.associations)),
            membership=int(len(self.membership)),
            min_support=int(self.min_support),
            max_association=(
                float(max(self.associations.values()))
                if self.associations else 0.0
            ),
            mean_association=(
                float(sum(self.associations.values())
                      / len(self.associations))
                if self.associations else 0.0
            ),
        )


def learn_static_strengths(active_sets: Iterable[Iterable],
                           graph=None, min_support: int = 2,
                           membership: Optional[Mapping] = None,
                           membership_strength: float = 1.0,
                           min_association: float = 0.0,
                           normalized: bool = False):
    """Learn association Jaccard strengths from realized co-activation.

    ``active_sets`` is one iterable of active node ids per state.  A pair is
    kept only when it co-occurs in at least ``min_support`` states.  Membership
    is structural: when a graph is supplied every existing memory binding gets
    ``membership_strength`` unless an explicit mapping overrides it.
    """
    if int(min_support) < 1:
        raise ValueError("min_support must be positive")
    marginal = {}
    cooccurrence = {}
    states = 0
    for active in active_sets:
        nodes = tuple(dict.fromkeys(str(node) for node in active))
        if not nodes:
            continue
        states += 1
        for node in nodes:
            marginal[node] = int(marginal.get(node, 0)) + 1
        for index, first in enumerate(nodes):
            for second in nodes[index + 1:]:
                key = _pair(first, second)
                cooccurrence[key] = int(cooccurrence.get(key, 0)) + 1
    associations = {}
    support = {}
    for key, count in cooccurrence.items():
        if count < int(min_support):
            continue
        first, second = key
        union = marginal[first] + marginal[second] - count
        if union <= 0:
            continue
        strength = float(count) / float(union)
        if strength < float(min_association):
            continue
        associations[key] = strength
        support[key] = int(count)
    bindings = {}
    if membership is not None:
        for key, value in dict(membership).items():
            bindings[(str(key[0]), str(key[1]))] = float(value)
    elif graph is not None:
        for key, edge in graph.edges.items():
            if key[2] != "membership":
                continue
            pair = (str(key[0]), str(key[1]))
            if float(edge.weight) > 0.0:
                bindings[pair] = float(edge.weight)
            else:
                bindings[pair] = float(membership_strength)
    return StaticGraphStrengths(
        associations=associations,
        membership=bindings,
        support=support,
        min_support=int(min_support),
        normalized=bool(normalized),
    )
