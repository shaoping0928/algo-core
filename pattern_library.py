"""A cross-world library of rule schemas.

The library stores which relations and modes a learned death rule uses,
not the numeric thresholds.  A new world retrieves the schema (the
coordinate/relation dimensions), then re-instantiates the thresholds from
a few labelled transitions.  This is the generic analogue of recognising
"alignment plus leaving an interval" without hard-coding the geometry.

The library is deliberately task-neutral: a pattern is a set of feature
dimensions plus its mode/label and support.  Thresholds remain local.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from death_rules import DeathRuleSet


@dataclass(frozen=True)
class Pattern:
    dims: Tuple[int, ...]
    label: int
    support: int
    text: str
    world: Optional[int] = None


class PatternLibrary:
    """Collect rule schemas and return a reusable dimension set."""

    def __init__(self, min_support: int = 1):
        self.min_support = int(min_support)
        self.patterns: List[Pattern] = []

    def add_rule(self, rule, world: Optional[int] = None) -> None:
        dims = tuple(int(dim) for dim in rule.signature())
        if not dims:
            return
        self.patterns.append(Pattern(
            dims=dims,
            label=int(rule.label),
            support=int(rule.support),
            text=rule.text(),
            world=world,
        ))

    def add_rules(self, rule_set: DeathRuleSet,
                  world: Optional[int] = None) -> None:
        for rule in rule_set.rules_:
            self.add_rule(rule, world=world)

    def add_rule_sets(self, rule_sets: Iterable[DeathRuleSet]) -> None:
        for index, rule_set in enumerate(rule_sets):
            self.add_rules(rule_set, world=index)

    def schema_dims(self, label: int = 1,
                    min_fraction: float = 0.25) -> Tuple[int, ...]:
        """Dimensions used by a stable fraction of the class's patterns.

        Dimensions are counted by pattern presence rather than by raw
        support, so a frequent mode cannot erase a rarer mode.
        """
        selected = [
            pattern for pattern in self.patterns
            if pattern.label == int(label)
            and pattern.support >= self.min_support
        ]
        if not selected:
            return ()
        counts: dict = {}
        for pattern in selected:
            for dim in pattern.dims:
                counts[dim] = counts.get(dim, 0) + 1
        threshold = max(
            1, int(np.ceil(float(min_fraction) * len(selected)))
        )
        return tuple(sorted(
            dim for dim, count in counts.items()
            if count >= threshold
        ))

    def instantiate(self, states: np.ndarray, labels: np.ndarray,
                    label: int = 1, **rule_kwargs) -> DeathRuleSet:
        """Fit a rule set restricted to the library's schema dimensions."""
        dims = self.schema_dims(label)
        if not dims:
            dims = None
        return DeathRuleSet(feature_dims=dims, **rule_kwargs).fit(
            states, labels
        )

    def describe(self, limit: int = 12) -> str:
        lines = []
        for pattern in self.patterns[:limit]:
            lines.append("dims=%s label=%d n=%d %s" % (
                pattern.dims, pattern.label, pattern.support,
                pattern.text,
            ))
        return "\n".join(lines)
