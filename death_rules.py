"""Explicit 0/1 death-condition rules.

The learner asks a different question from the probabilistic hazard layer:
*is this state a death state?*  It receives boolean labels (1 = the
transition entered a terminal state, 0 = it did not) and induces a DNF of
conjunctions of coordinate thresholds.  A rule is accepted only when it
covers training positives and **no** training negatives; states covered
by neither a death rule nor a safe rule are reported as ``UNKNOWN``
instead of being assigned a rate.  This keeps the model falsifiable: an
unknown state is a request for a counterexample, not a probability.

The induction is discrete and non-gradient.  Candidate literals are
single-coordinate thresholds (raw or magnitude); each rule is grown from
a positive seed by adding the literal that removes the most currently
covered negatives while retaining positive support (the classic
separate-and-conquer / AQ step).  A pure rule is removed from the
positive set and the next seed grows the next rule.  The same procedure
is run on the safe class, so the two-sided predicate can answer 0, 1, or
UNKNOWN.

No task vocabulary enters the module: coordinates are only
``f0 .. f_{d-1}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from node_features import Primitive
from rule_expressions import ExpressionRule, search_compositions

UNKNOWN = -1


def negate(primitive: Primitive) -> Primitive:
    """The exact complement of a midpoint-threshold primitive."""
    return Primitive(
        primitive.dim,
        primitive.magnitude,
        ">" if primitive.op == "<" else "<",
        primitive.threshold,
    )


@dataclass(frozen=True)
class DeathRule:
    """A conjunction that implies a 0/1 label."""

    literals: Tuple[Primitive, ...]
    label: int
    support: int
    positives: int
    negatives: int

    def mask(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        result = np.ones(len(states), dtype=bool)
        for literal in self.literals:
            result &= literal.mask(states)
        return result

    def mask_margin(self, states: np.ndarray,
                    margins: np.ndarray) -> np.ndarray:
        """Expand every threshold outward by the per-coordinate margin."""
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        margins = np.asarray(margins, dtype=float)
        result = np.ones(len(states), dtype=bool)
        for literal in self.literals:
            values = (
                np.abs(states[:, literal.dim])
                if literal.magnitude else states[:, literal.dim]
            )
            margin = float(margins[literal.dim])
            if literal.op == "<":
                result &= values < literal.threshold + margin
            else:
                result &= values > literal.threshold - margin
        return result

    def text(self) -> str:
        if not self.literals:
            body = "TRUE"
        else:
            body = " AND ".join(
                literal.label() for literal in self.literals
            )
        return "%s -> %s" % (
            body, "death" if self.label == 1 else "safe"
        )

    def signature(self) -> Tuple[int, ...]:
        return tuple(sorted({
            literal.dim for literal in self.literals
        }))

    def complexity(self) -> int:
        return 1 + len(self.literals)


class DeathRuleSet:
    """A pure two-sided rule set for a 0/1 death predicate.

    ``fit`` stores the samples, builds the candidate literals, and grows
    death rules and safe rules separately.  ``classify`` returns 1, 0, or
    ``UNKNOWN``.  ``refine`` appends a counterexample and re-induces,
    which is the verification/refinement loop the rule layer is meant to
    support.
    """

    def __init__(self, max_rules: int = 24, max_literals: int = 4,
                 min_support: int = 2, max_thresholds: int = 48,
                 max_attempts: int = 96, compose: bool = False,
                 coverage_first: bool = False,
                 boundary_margin: float = 0.0,
                 feature_dims: Optional[Sequence[int]] = None,
                 pure_support_fraction: float = 0.05,
                 compose_max_base: int = 24, compose_max_pairs: int = 256,
                 compose_rules: int = 8, compose_complexity: int = 8):
        if max_rules < 1:
            raise ValueError("max_rules must be positive")
        if max_literals < 1:
            raise ValueError("max_literals must be positive")
        if min_support < 1:
            raise ValueError("min_support must be positive")
        if max_thresholds < 1:
            raise ValueError("max_thresholds must be positive")
        self.max_rules = int(max_rules)
        self.max_literals = int(max_literals)
        self.min_support = int(min_support)
        self.max_thresholds = int(max_thresholds)
        self.max_attempts = int(max_attempts)
        self.compose = bool(compose)
        self.coverage_first = bool(coverage_first)
        self.boundary_margin = max(float(boundary_margin), 0.0)
        self.margins_: Optional[np.ndarray] = None
        self.feature_dims = (
            None if feature_dims is None
            else tuple(sorted({int(dim) for dim in feature_dims}))
        )
        self.pure_support_fraction = max(
            float(pure_support_fraction), 0.0
        )
        self.compose_max_base = int(compose_max_base)
        self.compose_max_pairs = int(compose_max_pairs)
        self.compose_rules = int(compose_rules)
        self.compose_complexity = int(compose_complexity)
        self.states_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.literals_: List[Primitive] = []
        self._masks: Optional[np.ndarray] = None
        self.rules_: list = []
        self.coverage_: Optional[dict] = None
        self.refinement_log: List[dict] = []

    # ---- thresholds -------------------------------------------------------
    def _thresholds(self, values: np.ndarray, labels: np.ndarray) -> List[float]:
        """Midpoint cuts, prioritising observed class changes.

        Thresholds are never placed on a sample value: every candidate is
        a midpoint between adjacent unique values, so ``<`` and ``>`` are
        exact complements on the observed data.
        """
        values = np.asarray(values, dtype=float)
        unique = np.unique(values)
        if len(unique) < 2:
            return []
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_labels = np.asarray(labels, dtype=float)[order]
        unique_values, inverse = np.unique(
            sorted_values, return_inverse=True
        )
        counts = np.bincount(inverse, minlength=len(unique_values))
        sums = np.bincount(
            inverse, weights=sorted_labels, minlength=len(unique_values)
        )
        rates = sums / np.maximum(counts, 1)
        midpoints = (
            unique_values[:-1] + unique_values[1:]
        ) / 2.0
        if len(midpoints) == 0:
            return []
        # Marginal rate changes are the sharpest candidates and are always
        # retained.  The rest is a rank-spread sample of every adjacent
        # midpoint, so a conditional cut in a wide empty band is not
        # crowded out by many high-overlap neighbours in one corner.
        changes = np.flatnonzero(np.abs(np.diff(rates)) > 1e-12)
        rate_values = [float(midpoints[index]) for index in changes]
        if len(midpoints) <= self.max_thresholds:
            spread = [float(value) for value in midpoints]
        else:
            positions = np.linspace(
                0, len(midpoints) - 1, self.max_thresholds
            ).astype(int)
            spread = [float(midpoints[index]) for index in positions]
        candidates = list(dict.fromkeys(rate_values + spread))
        if len(candidates) > self.max_thresholds:
            # Keep every rate-change cut, then fill the remaining budget
            # with the spread sample.
            keep = list(dict.fromkeys(rate_values))
            for value in spread:
                if len(keep) >= self.max_thresholds:
                    break
                if value not in keep:
                    keep.append(value)
            candidates = keep
        # Zero is a universal reference cut (not a task constant).  Without
        # it a conditional boundary that sits at zero can be skipped by
        # the rank-spread sample.  Append after the budget cut so it is
        # never crowded out.
        if unique_values[0] < 0.0 < unique_values[-1]:
            candidates.append(0.0)
        quantiles = np.unique(np.quantile(
            unique_values, np.linspace(0.02, 0.98, 32)
        ))
        if len(quantiles) > 1:
            for index in range(len(quantiles) - 1):
                value = float(
                    (quantiles[index] + quantiles[index + 1]) / 2.0
                )
                if value not in candidates:
                    candidates.append(value)
        result: List[float] = []
        for value in candidates:
            if not np.isfinite(value):
                continue
            if any(abs(value - seen) <= 1e-12 for seen in result):
                continue
            result.append(value)
        return result

    def _candidate_literals(self, states: np.ndarray,
                            labels: np.ndarray) -> List[Primitive]:
        literals: List[Primitive] = []
        dims = (
            range(states.shape[1])
            if self.feature_dims is None else self.feature_dims
        )
        for dim in dims:
            if dim < 0 or dim >= states.shape[1]:
                raise ValueError("feature dim out of range: %d" % dim)
            for magnitude in (False, True):
                values = (
                    np.abs(states[:, dim]) if magnitude else states[:, dim]
                )
                for threshold in self._thresholds(values, labels):
                    literals.append(Primitive(
                        dim, magnitude, "<", float(threshold)
                    ))
                    literals.append(Primitive(
                        dim, magnitude, ">", float(threshold)
                    ))
        # Dataclass primitives are hashable; keep deterministic order.
        return list(dict.fromkeys(literals))

    # ---- AQ rule growth ---------------------------------------------------
    def _grow_rule(self, target: np.ndarray, seed: int,
                   remaining: Optional[np.ndarray] = None,
                   max_literals: Optional[int] = None):
        """Grow one pure conjunction from a positive seed.

        ``remaining`` marks the positives not yet covered by earlier
        rules.  Separate-and-conquer must grow the next rule against the
        uncovered positives only; otherwise a later rule is pulled toward
        the broad region the first rule already handles.

        Returns ``(literals, mask)`` or ``None`` when no conjunction can
        separate the seed from every negative within ``max_literals``.
        """
        total = len(target)
        positives = target if remaining is None else remaining
        current = np.ones(total, dtype=bool)
        current_negative = int((current & ~target).sum())
        if current_negative == 0:
            return (), current
        selected: List[Primitive] = []
        min_support = max(1, self.min_support)
        limit = self.max_literals if max_literals is None else int(max_literals)
        for _ in range(limit):
            candidates = np.flatnonzero(self._masks[:, seed])
            if len(candidates) == 0:
                return None
            new_masks = self._masks[candidates] & current[None, :]
            new_positives = (new_masks & positives[None, :]).sum(axis=1)
            new_negatives = (new_masks & ~target[None, :]).sum(axis=1)
            valid = (
                (new_positives >= min_support)
                & (new_negatives < current_negative)
            )
            if not valid.any():
                return None
            valid_indices = np.flatnonzero(valid)
            # Prefer a pure literal when it retains at least half of the
            # best positive support; otherwise keep the widest positive
            # support and let the next literal remove negatives.  Purity
            # alone picks tiny slices; width alone can pick a literal that
            # never reaches purity within the budget.
            max_positives = int(new_positives[valid_indices].max())
            support_floor = max(
                min_support,
                int(np.ceil(
                    self.pure_support_fraction * float(max_positives)
                )),
            )
            pure = (
                valid
                & (new_negatives == 0)
                & (new_positives >= support_floor)
            )
            if pure.any():
                pure_indices = np.flatnonzero(pure)
                order = np.lexsort((
                    candidates[pure_indices],
                    -new_positives[pure_indices],
                ))
                best = int(pure_indices[order[0]])
            else:
                order = np.lexsort((
                    candidates[valid_indices],
                    new_negatives[valid_indices],
                    -new_positives[valid_indices],
                ))
                best = int(valid_indices[order[0]])
            index = int(candidates[best])
            selected.append(self.literals_[index])
            current = new_masks[best]
            current_negative = int((current & ~target).sum())
            if current_negative == 0:
                return tuple(selected), current
        return None

    def _conjunction_mask(self, literals: Sequence[Primitive]) -> np.ndarray:
        mask = np.ones(len(self.states_), dtype=bool)
        for literal in literals:
            mask &= literal.mask(self.states_)
        return mask

    def _simplify(self, literals: Sequence[Primitive],
                  target: np.ndarray, mask: np.ndarray):
        """Remove literals that do not change purity or positive coverage."""
        selected = list(literals)
        best_positive = int((mask & target).sum())
        changed = True
        while changed and len(selected) > 1:
            changed = False
            for index in range(len(selected)):
                candidate = selected[:index] + selected[index + 1:]
                candidate_mask = self._conjunction_mask(candidate)
                if int((candidate_mask & ~target).sum()) != 0:
                    continue
                positive = int((candidate_mask & target).sum())
                if positive < best_positive:
                    continue
                selected = candidate
                mask = candidate_mask
                best_positive = positive
                changed = True
                break
        return selected, mask

    def _induce_class(self, target: np.ndarray, label: int,
                      remaining_mask: Optional[np.ndarray] = None,
                      max_literals: Optional[int] = None) -> List[DeathRule]:
        positives = np.flatnonzero(target)
        remaining = set(int(index) for index in (
            positives if remaining_mask is None
            else np.flatnonzero(remaining_mask)
        ))
        remaining_mask = (
            target.copy() if remaining_mask is None
            else np.asarray(remaining_mask, dtype=bool).copy()
        )
        rules: List[DeathRule] = []
        attempts = 0
        while remaining and len(rules) < self.max_rules:
            if attempts >= self.max_attempts:
                break
            attempts += 1
            seed = min(remaining)
            grown = self._grow_rule(
                target, seed, remaining_mask, max_literals=max_literals
            )
            if grown is None:
                remaining.discard(seed)
                continue
            literals, mask = grown
            literals, mask = self._simplify(literals, target, mask)
            covered = [
                index for index in positives
                if index in remaining and bool(mask[index])
            ]
            if not covered:
                remaining.discard(seed)
                continue
            if np.any(mask & ~target):
                remaining.discard(seed)
                continue
            rules.append(DeathRule(
                literals=tuple(literals),
                label=int(label),
                support=int(mask.sum()),
                positives=int((mask & target).sum()),
                negatives=int((mask & ~target).sum()),
            ))
            remaining.difference_update(covered)
            remaining_mask[covered] = False
        return rules

    # ---- fitting / reading ------------------------------------------------
    def fit(self, states: np.ndarray, labels: np.ndarray) -> "DeathRuleSet":
        states = np.asarray(states, dtype=float)
        labels = np.asarray(labels, dtype=float).reshape(-1)
        if states.ndim != 2 or len(states) != len(labels):
            raise ValueError("states/labels shape mismatch")
        if len(states) == 0:
            raise ValueError("need at least one state")
        self.states_ = states.copy()
        self.labels_ = (labels > 0.5).astype(int)
        span = (
            np.max(self.states_, axis=0) - np.min(self.states_, axis=0)
        )
        span = np.where(np.isfinite(span) & (span > 0.0), span, 0.0)
        self.margins_ = self.boundary_margin * span
        self.literals_ = self._candidate_literals(
            self.states_, self.labels_
        )
        if self.literals_:
            self._masks = np.asarray(
                [literal.mask(self.states_) for literal in self.literals_],
                dtype=bool,
            )
        else:
            self._masks = np.zeros((0, len(states)), dtype=bool)
        death_target = self.labels_ == 1
        safe_target = self.labels_ == 0
        death_rules = self._induce_class(death_target, 1)
        safe_rules = self._induce_class(safe_target, 0)

        def covered(rules):
            result = np.zeros(len(states), dtype=bool)
            for rule in rules:
                result |= rule.mask(self.states_)
            return result

        if self.coverage_first:
            # A pure rule that does not cover every observed death is a
            # coverage gap, not a safe state.  Spend a larger conjunction
            # budget on the uncovered positives before giving them up.
            remaining_death = death_target & ~covered(death_rules)
            if remaining_death.any():
                death_rules.extend(self._induce_class(
                    death_target, 1,
                    remaining_mask=remaining_death,
                    max_literals=max(self.max_literals, 8),
                ))
        if self.compose or self.coverage_first:
            remaining_death = death_target & ~covered(death_rules)
            if remaining_death.any():
                death_rules.extend(search_compositions(
                    self.states_, remaining_death, self.literals_, 1,
                    negatives=safe_target,
                    max_base=self.compose_max_base,
                    max_pairs=self.compose_max_pairs,
                    max_rules=self.compose_rules,
                    min_support=self.min_support,
                    max_complexity=self.compose_complexity,
                ))
        if self.compose:
            remaining_safe = safe_target & ~covered(safe_rules)
            if remaining_safe.any():
                safe_rules.extend(search_compositions(
                    self.states_, remaining_safe, self.literals_, 0,
                    negatives=death_target,
                    max_base=self.compose_max_base,
                    max_pairs=self.compose_max_pairs,
                    max_rules=self.compose_rules,
                    min_support=self.min_support,
                    max_complexity=self.compose_complexity,
                ))
        self.rules_ = death_rules + safe_rules
        self.coverage_ = self.coverage_report()
        return self

    def coverage_report(self, states: Optional[np.ndarray] = None,
                        labels: Optional[np.ndarray] = None) -> dict:
        """Coverage of the death side, with the safe side reported apart."""
        if states is None:
            states = self.states_
            truth = self.labels_
        else:
            if labels is None:
                raise ValueError("labels are required with external states")
            states = np.asarray(states, dtype=float)
            truth = (np.asarray(labels, dtype=float).reshape(-1) > 0.5).astype(int)
        death, near = self._death_side(states)
        safe = np.zeros(len(states), dtype=bool)
        for rule in self.rules_:
            if rule.label == 0:
                safe |= rule.mask(states)
        positives = truth == 1
        negatives = truth == 0
        covered_positive = int(np.sum(death & positives))
        safe_answered = safe & ~near
        return dict(
            total=int(len(truth)),
            positive=int(positives.sum()),
            negative=int(negatives.sum()),
            death_covered_positive=covered_positive,
            death_uncovered_positive=int(positives.sum()) - covered_positive,
            death_coverage=(
                float(covered_positive) / float(positives.sum())
                if positives.any() else 0.0
            ),
            death_near_positive=int(np.sum(near & positives)),
            death_covered_negative=int(np.sum(death & negatives)),
            death_near_negative=int(np.sum(near & negatives)),
            safe_covered_negative=int(np.sum(safe_answered & negatives)),
            safe_blocked_near_death=int(np.sum(
                safe & near & ~death
            )),
            unknown=int(np.sum(~(death | safe_answered))),
            unknown_positive=int(np.sum(
                ~(death | safe_answered) & positives
            )),
        )

    def _death_side(self, states: np.ndarray):
        exact = np.zeros(len(states), dtype=bool)
        near = np.zeros(len(states), dtype=bool)
        for rule in self.rules_:
            if rule.label != 1:
                continue
            exact |= rule.mask(states)
            if self.margins_ is not None and hasattr(rule, "mask_margin"):
                near |= rule.mask_margin(states, self.margins_)
            else:
                near |= rule.mask(states)
        return exact, near

    def classify(self, states: np.ndarray) -> np.ndarray:
        if self.states_ is None:
            raise RuntimeError("fit must be called before classify")
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        death, near = self._death_side(states)
        safe = np.zeros(len(states), dtype=bool)
        for rule in self.rules_:
            if rule.label == 0:
                safe |= rule.mask(states)
        result = np.full(len(states), UNKNOWN, dtype=int)
        result[safe & ~near] = 0
        result[death] = 1
        return result

    def classify_death(self, states: np.ndarray,
                       include_margin: bool = False) -> np.ndarray:
        """Only the death side: 1 when a death rule fires, else UNKNOWN.

        The planner's hard constraint only needs this side.  Not claiming
        safety from a finite rule set keeps a missed death visible as
        uncertainty instead of silently becoming a false negative.

        With ``include_margin=True``, states inside the fitted boundary
        margin are also reported as death.  This is the model-error-aware
        envelope: a predicted state that lands just outside a death rule is
        treated as risky instead of silently unknown.  The default keeps
        the exact death side, so callers that do not opt in are unchanged.
        """
        if self.states_ is None:
            raise RuntimeError("fit must be called before classify_death")
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        death, near = self._death_side(states)
        selected = near if include_margin else death
        return np.where(selected, 1, UNKNOWN)

    def verify(self, states: np.ndarray, labels: np.ndarray) -> dict:
        prediction = self.classify(states)
        truth = (np.asarray(labels, dtype=float).reshape(-1) > 0.5).astype(int)
        answered = prediction != UNKNOWN
        positives = truth == 1
        negatives = truth == 0
        true_positive = int(np.sum(answered & (prediction == 1) & positives))
        false_positive = int(np.sum(answered & (prediction == 1) & negatives))
        true_negative = int(np.sum(answered & (prediction == 0) & negatives))
        false_negative = int(np.sum(answered & (prediction == 0) & positives))
        unknown = int(np.sum(~answered))
        total = len(truth)
        correct = true_positive + true_negative
        death_hits = np.zeros(len(states), dtype=bool)
        safe_hits = np.zeros(len(states), dtype=bool)
        for rule in self.rules_:
            hit = rule.mask(states)
            if rule.label == 1:
                death_hits |= hit
            else:
                safe_hits |= hit
        return dict(
            total=total,
            answered=int(answered.sum()),
            unknown=unknown,
            coverage=(float(answered.mean()) if total else 0.0),
            accuracy=(
                float(correct) / float(answered.sum())
                if answered.any() else 0.0
            ),
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
            positive=int(positives.sum()),
            negative=int(negatives.sum()),
            conflict=int(np.sum(death_hits & safe_hits)),
        )

    def rules_for(self, label: int) -> List[DeathRule]:
        return [rule for rule in self.rules_ if rule.label == int(label)]

    def modes(self, label: int = 1) -> List[dict]:
        """Group rules by the coordinate set they use.

        Two rules that use different coordinates are two discovered modes
        of the same 0/1 predicate.  The grouping is purely algebraic; no
        mode is named or privileged.
        """
        groups: dict = {}
        for rule in self.rules_for(label):
            signature = tuple(rule.signature())
            groups.setdefault(signature, []).append(rule)
        return [
            dict(signature=signature, rules=rules)
            for signature, rules in sorted(
                groups.items(), key=lambda item: (-len(item[1]), item[0])
            )
        ]

    def describe_modes(self, label: int = 1, limit: int = 8) -> str:
        lines = []
        for index, mode in enumerate(self.modes(label)[:limit]):
            lines.append("mode %d: dims=%s rules=%d" % (
                index, mode["signature"], len(mode["rules"]),
            ))
            for rule in mode["rules"][:3]:
                lines.append("  " + rule.text())
        return "\n".join(lines)

    def describe(self, limit: int = 16) -> str:
        lines = []
        for rule in self.rules_[:limit]:
            lines.append(
                "%s  n=%d (+%d/-%d)" % (
                    rule.text(), rule.support, rule.positives,
                    rule.negatives,
                )
            )
        return "\n".join(lines)

    # ---- refinement -------------------------------------------------------
    def add_examples(self, states: np.ndarray, labels: np.ndarray) -> None:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        labels = np.asarray(labels, dtype=float).reshape(-1)
        if len(states) != len(labels):
            raise ValueError("states/labels shape mismatch")
        if self.states_ is None:
            self.fit(states, labels)
            return
        self.fit(
            np.vstack([self.states_, states]),
            np.concatenate([self.labels_, (labels > 0.5).astype(int)]),
        )

    def refine(self, state: np.ndarray, label: int) -> int:
        state = np.asarray(state, dtype=float).reshape(1, -1)
        before = int(self.classify(state)[0])
        self.add_examples(state, [label])
        after = int(self.classify(state)[0])
        self.refinement_log.append(dict(
            label=int(label), before=before, after=after
        ))
        return after
