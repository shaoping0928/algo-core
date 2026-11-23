"""Learned derivation rules for component signs.

The code-level difference/capacitor/resistor filters are training-time
labelers.  At prediction time the sign becomes an ordinary graph node filled
by a local rule.  This module covers the first, hidden-state-free case:
``sign(dy)`` from a raw value or from ``vy``.

The rules are ordinary ``query_graph.Rule`` objects, so a query asks for the
sign slot and the rule is admitted only when its input slots are available.
A missing input yields UNKNOWN; it never falls back to the numeric filter.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from query_graph import (
    UNKNOWN,
    CallablePredictor,
    DtSpec,
    Pattern,
    Rule,
    Slot,
)


def sign_of(value: float, threshold: float = 0.0) -> int:
    value = float(value)
    threshold = max(float(threshold), 0.0)
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


@dataclass
class ThresholdSignRule:
    """``score > t -> +``, ``score < -t -> -``, otherwise hold."""

    name: str
    score_slot: Slot
    threshold: float = 0.0
    support: int = 0
    confidence: float = 0.0

    def predict(self, state: Mapping[Slot, Any]) -> Any:
        value = state.get(self.score_slot, UNKNOWN)
        if value is UNKNOWN:
            return UNKNOWN
        return sign_of(float(value), self.threshold)

    def predict_value(self, score: float) -> int:
        return sign_of(float(score), self.threshold)


@dataclass
class DifferenceSignRule:
    """``sign(current - previous)`` with an optional hold band."""

    name: str
    current_slot: Slot
    previous_slot: Slot
    threshold: float = 0.0
    support: int = 0
    confidence: float = 0.0

    def predict(self, state: Mapping[Slot, Any]) -> Any:
        current = state.get(self.current_slot, UNKNOWN)
        previous = state.get(self.previous_slot, UNKNOWN)
        if current is UNKNOWN or previous is UNKNOWN:
            return UNKNOWN
        return sign_of(float(current) - float(previous), self.threshold)

    def predict_value(self, current: float, previous: float) -> int:
        return sign_of(float(current) - float(previous), self.threshold)


def _fit_threshold(scores: Sequence[float], labels: Sequence[int]) -> Tuple[
        float, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    candidates = np.unique(np.concatenate([
        np.asarray([0.0]),
        np.quantile(np.abs(scores), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]),
    ]))
    best_threshold = 0.0
    best_accuracy = -1.0
    for threshold in candidates:
        prediction = np.asarray([
            sign_of(value, float(threshold)) for value in scores
        ])
        accuracy = float(np.mean(prediction == labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def fit_threshold_rule(scores: Sequence[float], labels: Sequence[int],
                       name: str = "sign") -> ThresholdSignRule:
    threshold, accuracy = _fit_threshold(scores, labels)
    return ThresholdSignRule(
        name=name,
        score_slot=Slot("score", 0),
        threshold=threshold,
        support=int(len(scores)),
        confidence=accuracy,
    )


def fit_difference_rule(current: Sequence[float], previous: Sequence[float],
                        labels: Sequence[int],
                        name: str = "difference") -> DifferenceSignRule:
    deltas = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
    threshold, accuracy = _fit_threshold(deltas, labels)
    return DifferenceSignRule(
        name=name,
        current_slot=Slot("value", 0),
        previous_slot=Slot("value", -1),
        threshold=threshold,
        support=int(len(deltas)),
        confidence=accuracy,
    )


def make_threshold_rule(rule: ThresholdSignRule, score_slot: Slot,
                        target_slot: Slot) -> Rule:
    def predict(state: Mapping[Slot, Any]) -> Any:
        value = state.get(score_slot, UNKNOWN)
        if value is UNKNOWN:
            return UNKNOWN
        return sign_of(float(value), rule.threshold)

    return Rule(
        name=rule.name,
        pattern_in=Pattern((score_slot,)),
        pattern_out=Pattern((target_slot,)),
        dt=DtSpec.steps(1),
        predictor=CallablePredictor(predict),
        support=rule.support,
        confidence=rule.confidence,
    )


def make_difference_rule(rule: DifferenceSignRule, current_slot: Slot,
                         previous_slot: Slot, target_slot: Slot) -> Rule:
    def predict(state: Mapping[Slot, Any]) -> Any:
        current = state.get(current_slot, UNKNOWN)
        previous = state.get(previous_slot, UNKNOWN)
        if current is UNKNOWN or previous is UNKNOWN:
            return UNKNOWN
        return sign_of(float(current) - float(previous), rule.threshold)

    return Rule(
        name=rule.name,
        pattern_in=Pattern((current_slot, previous_slot)),
        pattern_out=Pattern((target_slot,)),
        dt=DtSpec.steps(1),
        predictor=CallablePredictor(predict),
        support=rule.support,
        confidence=rule.confidence,
    )


def evaluate_rule(rule, states: Sequence[Mapping[Slot, Any]],
                  labels: Sequence[int]) -> Tuple[float, float]:
    """Return (accuracy over known predictions, UNKNOWN rate)."""
    correct = 0
    known = 0
    unknown = 0
    for state, label in zip(states, labels):
        prediction = rule.predict(state)
        if prediction is UNKNOWN:
            unknown += 1
            continue
        known += 1
        correct += int(prediction == int(label))
    accuracy = correct / float(known) if known else float("nan")
    return accuracy, unknown / float(max(len(states), 1))


def threshold_states(scores: Sequence[float],
                     slot: Slot) -> list:
    return [{slot: float(value)} for value in scores]


def difference_states(current: Sequence[float], previous: Sequence[float],
                      slot: Slot) -> list:
    return [
        {slot: float(now), Slot(slot.name, -1): float(before)}
        for now, before in zip(current, previous)
    ]


class ComponentDerivationBank:
    """Runtime side of the label/free-run split.

    A component id maps to an ordinary rule.  During prediction the bank is
    asked for the sign; a missing rule or a missing input returns UNKNOWN and
    never falls back to the numeric filter.  Wiring this bank into
    ``ComponentBank.step`` is deliberately left opt-in/for the next slice.
    """

    def __init__(self, rules: Optional[Mapping[str, Rule]] = None) -> None:
        self.rules: Dict[str, Rule] = dict(rules or {})

    def bind(self, component_id: str, rule: Rule) -> None:
        self.rules[str(component_id)] = rule

    def predict(self, component_id: str, state: Mapping[Slot, Any]) -> Any:
        rule = self.rules.get(str(component_id))
        if rule is None:
            return UNKNOWN
        return rule.predict(state)

    def predictions(self, state: Mapping[Slot, Any]) -> Dict[str, Any]:
        return {
            component_id: rule.predict(state)
            for component_id, rule in self.rules.items()
        }
