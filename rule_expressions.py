"""Composable rule expressions for 0/1 predicates.

The death-condition layer must not assume a flat disjunction of
conjunctions.  A rule is an expression tree over:

- coordinate literals (raw or magnitude thresholds),
- unary ``NOT``,
- n-ary ``AND``, ``OR`` and ``XOR``,
- numeric terms (sum, difference, product, min, max, mean, ratio, abs)
  compared against a threshold.

The tree is evaluated directly (no probability), and every node reports
its coordinate signature and complexity so a learner can search over
compositions and a reader can see which coordinates a rule uses.  The
module is task-neutral: coordinates are only ``f0 .. f_{d-1}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from node_features import Primitive


class Expr:
    """Base class for a boolean expression over a state matrix."""

    def mask(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def text(self) -> str:
        raise NotImplementedError

    def signature(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def complexity(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Literal(Expr):
    primitive: Primitive

    def mask(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        return self.primitive.mask(states)

    def text(self) -> str:
        return self.primitive.label()

    def signature(self) -> Tuple[int, ...]:
        return (self.primitive.dim,)

    def complexity(self) -> int:
        return 1


@dataclass(frozen=True)
class Not(Expr):
    child: Expr

    def mask(self, states: np.ndarray) -> np.ndarray:
        return ~self.child.mask(states)

    def text(self) -> str:
        return "NOT (%s)" % self.child.text()

    def signature(self) -> Tuple[int, ...]:
        return self.child.signature()

    def complexity(self) -> int:
        return 1 + self.child.complexity()


@dataclass(frozen=True)
class Nary(Expr):
    op: str
    children: Tuple[Expr, ...]

    def __post_init__(self):
        if self.op not in ("and", "or", "xor"):
            raise ValueError("unknown boolean op %s" % self.op)
        if len(self.children) < 2:
            raise ValueError("n-ary expression needs at least two children")

    def mask(self, states: np.ndarray) -> np.ndarray:
        masks = [child.mask(states) for child in self.children]
        if self.op == "and":
            result = masks[0].copy()
            for item in masks[1:]:
                result &= item
            return result
        if self.op == "or":
            result = masks[0].copy()
            for item in masks[1:]:
                result |= item
            return result
        result = masks[0].astype(int)
        for item in masks[1:]:
            result = result + item.astype(int)
        return (result % 2) == 1

    def text(self) -> str:
        joiner = {"and": " AND ", "or": " OR ", "xor": " XOR "}[self.op]
        return "(" + joiner.join(child.text() for child in self.children) + ")"

    def signature(self) -> Tuple[int, ...]:
        dims = set()
        for child in self.children:
            dims.update(child.signature())
        return tuple(sorted(dims))

    def complexity(self) -> int:
        return 1 + sum(child.complexity() for child in self.children)


# ---- numeric terms --------------------------------------------------------
class Term:
    def values(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def text(self) -> str:
        raise NotImplementedError

    def signature(self) -> Tuple[int, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class FeatureTerm(Term):
    dim: int
    magnitude: bool = False

    def values(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        values = states[:, self.dim]
        return np.abs(values) if self.magnitude else values

    def text(self) -> str:
        return ("|f%d|" if self.magnitude else "f%d") % self.dim

    def signature(self) -> Tuple[int, ...]:
        return (self.dim,)


@dataclass(frozen=True)
class NumericOp(Term):
    op: str
    children: Tuple[Term, ...]

    def __post_init__(self):
        if self.op not in (
            "sum", "diff", "product", "min", "max", "mean", "ratio", "abs"
        ):
            raise ValueError("unknown numeric op %s" % self.op)
        if self.op == "abs" and len(self.children) != 1:
            raise ValueError("abs takes one child")
        if self.op == "ratio" and len(self.children) != 2:
            raise ValueError("ratio takes two children")
        if len(self.children) < 1:
            raise ValueError("numeric op needs at least one child")

    def values(self, states: np.ndarray) -> np.ndarray:
        values = [child.values(states) for child in self.children]
        if self.op == "abs":
            return np.abs(values[0])
        if self.op == "sum":
            result = values[0].copy()
            for item in values[1:]:
                result += item
            return result
        if self.op == "diff":
            result = values[0].copy()
            for item in values[1:]:
                result -= item
            return result
        if self.op == "product":
            result = values[0].copy()
            for item in values[1:]:
                result *= item
            return result
        if self.op == "min":
            result = values[0].copy()
            for item in values[1:]:
                result = np.minimum(result, item)
            return result
        if self.op == "max":
            result = values[0].copy()
            for item in values[1:]:
                result = np.maximum(result, item)
            return result
        if self.op == "mean":
            result = values[0].copy()
            for item in values[1:]:
                result += item
            return result / float(len(values))
        denominator = values[1].copy()
        denominator = np.where(
            np.abs(denominator) < 1e-12,
            np.where(denominator < 0.0, -1e-12, 1e-12),
            denominator,
        )
        return values[0] / denominator

    def text(self) -> str:
        if self.op == "abs":
            return "abs(%s)" % self.children[0].text()
        joiner = {
            "sum": " + ", "diff": " - ", "product": " * ",
            "min": " min ", "max": " max ", "mean": " mean ",
            "ratio": " / ",
        }[self.op]
        return "(" + joiner.join(child.text() for child in self.children) + ")"

    def signature(self) -> Tuple[int, ...]:
        dims = set()
        for child in self.children:
            dims.update(child.signature())
        return tuple(sorted(dims))


@dataclass(frozen=True)
class Compare(Expr):
    term: Term
    op: str
    threshold: float

    def __post_init__(self):
        if self.op not in ("<", ">", "<=", ">="):
            raise ValueError("unknown comparison op %s" % self.op)

    def mask(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        values = self.term.values(states)
        with np.errstate(invalid="ignore"):
            if self.op == "<":
                result = values < self.threshold
            elif self.op == ">":
                result = values > self.threshold
            elif self.op == "<=":
                result = values <= self.threshold
            else:
                result = values >= self.threshold
        return np.where(np.isfinite(values), result, False)

    def text(self) -> str:
        return "%s %s %.4f" % (
            self.term.text(), self.op, self.threshold
        )

    def signature(self) -> Tuple[int, ...]:
        return self.term.signature()

    def complexity(self) -> int:
        return 1 + len(self.term.signature())


# ---- rules ----------------------------------------------------------------
@dataclass(frozen=True)
class ExpressionRule:
    expression: Expr
    label: int
    support: int
    positives: int
    negatives: int

    def mask(self, states: np.ndarray) -> np.ndarray:
        return self.expression.mask(states)

    def text(self) -> str:
        return "%s -> %s" % (
            self.expression.text(),
            "death" if self.label == 1 else "safe",
        )

    def signature(self) -> Tuple[int, ...]:
        return self.expression.signature()

    def complexity(self) -> int:
        return self.expression.complexity()

    @property
    def literals(self) -> Tuple[Primitive, ...]:
        return tuple(_collect_literals(self.expression))


def _collect_literals(expression: Expr) -> List[Primitive]:
    if isinstance(expression, Literal):
        return [expression.primitive]
    if isinstance(expression, Not):
        return _collect_literals(expression.child)
    if isinstance(expression, Nary):
        result: List[Primitive] = []
        for child in expression.children:
            result.extend(_collect_literals(child))
        return result
    return []


# ---- composition search ---------------------------------------------------
def _rank_literals(literals: Sequence[Primitive], states: np.ndarray,
                   target: np.ndarray, negatives: np.ndarray,
                   limit: int) -> List[Primitive]:
    scored = []
    for literal in literals:
        mask = literal.mask(states)
        positives = int((mask & target).sum())
        covered_negatives = int((mask & negatives).sum())
        if positives == 0:
            continue
        scored.append((positives, -covered_negatives, literal))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [item[2] for item in scored[:limit]]


def _expression_stats(expression: Expr, states: np.ndarray,
                      target: np.ndarray, negatives: np.ndarray) -> dict:
    mask = expression.mask(states)
    return dict(
        mask=mask,
        support=int(mask.sum()),
        positives=int((mask & target).sum()),
        negatives=int((mask & negatives).sum()),
    )


def search_compositions(states: np.ndarray, target: np.ndarray,
                        literals: Sequence[Primitive], label: int,
                        negatives: Optional[np.ndarray] = None,
                        max_base: int = 24, max_pairs: int = 256,
                        max_rules: int = 8, min_support: int = 2,
                        max_complexity: int = 8) -> List[ExpressionRule]:
    """Search shallow nested combinations of the strongest literals.

    The search is deliberately bounded: rank single literals, try
    ``AND/OR/XOR`` and ``AND(a, NOT b)`` on pairs, then combine the best
    pair expressions with one more literal.  It is a generic beam that
    can express XOR and negation; deeper recursive search can plug into
    the same expression classes later.

    ``target`` is the set of positives still to cover; ``negatives`` is
    the true negative set (defaults to the complement of ``target``).
    """
    states = np.asarray(states, dtype=float)
    target = np.asarray(target, dtype=bool)
    if negatives is None:
        negatives = ~target
    else:
        negatives = np.asarray(negatives, dtype=bool)
    base_literals = _rank_literals(
        literals, states, target, negatives, max_base
    )
    # A zero cut is a universal reference and is often the only split that
    # separates a conditional XOR; keep it available even when the
    # positive-coverage ranking is crowded by broad near-duplicates.
    zero_literals = [
        literal for literal in literals
        if abs(float(literal.threshold)) <= 1e-12
    ]
    base_literals = list(dict.fromkeys(zero_literals + base_literals))
    if not base_literals:
        return []
    base = [Literal(literal) for literal in base_literals]
    candidates: List[Expr] = []
    for expression in base:
        candidates.append(expression)
        candidates.append(Not(expression))
    for first in range(len(base)):
        for second in range(first + 1, len(base)):
            a, b = base[first], base[second]
            candidates.extend([
                Nary("and", (a, b)),
                Nary("or", (a, b)),
                Nary("xor", (a, b)),
                Nary("and", (a, Not(b))),
                Nary("and", (Not(a), b)),
                Nary("xor", (Not(a), Not(b))),
            ])
    pair_candidates = list(candidates)
    # A second level lets a nested pair (for example A AND B) combine
    # with another literal through XOR/AND/OR.
    scored_pairs = []
    for expression in pair_candidates:
        stats = _expression_stats(expression, states, target, negatives)
        if stats["positives"] <= 0:
            continue
        scored_pairs.append((
            stats["positives"] - stats["negatives"],
            stats["positives"],
            expression,
        ))
    scored_pairs.sort(key=lambda item: (-item[0], -item[1]))
    for _score, _positives, expression in scored_pairs[:max_pairs]:
        if expression.complexity() + 1 > max_complexity:
            continue
        for literal in base[:max_base]:
            if literal.signature() == expression.signature() and \
                    isinstance(expression, Literal):
                continue
            candidates.extend([
                Nary("and", (expression, literal)),
                Nary("or", (expression, literal)),
                Nary("xor", (expression, literal)),
            ])
    seen = set()
    rules: List[ExpressionRule] = []
    remaining = target.copy()
    ranked = []
    for expression in candidates:
        if expression.complexity() > max_complexity:
            continue
        stats = _expression_stats(expression, states, target, negatives)
        if stats["negatives"] != 0:
            continue
        if stats["positives"] < min_support:
            continue
        key = stats["mask"].tobytes()
        if key in seen:
            continue
        seen.add(key)
        ranked.append((
            stats["positives"], -expression.complexity(), expression, stats,
        ))
    ranked.sort(key=lambda item: (-item[0], -item[1]))
    for _positives, _neg, expression, stats in ranked:
        if len(rules) >= max_rules:
            break
        covered = stats["mask"] & remaining
        if not covered.any():
            continue
        rules.append(ExpressionRule(
            expression=expression,
            label=int(label),
            support=stats["support"],
            positives=stats["positives"],
            negatives=stats["negatives"],
        ))
        remaining &= ~stats["mask"]
        if not remaining.any():
            break
    return rules
