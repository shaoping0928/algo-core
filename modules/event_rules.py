"""Event predicates as ordinary query-graph rules.

The existing ``DeathRuleSet`` is the production 0/1 rule layer; this module
wraps it instead of inventing a second death semantics.  The wrapper reads the
anonymous state slots, classifies the composed state, and writes a discrete
event flag.  A missing slot or a ``DeathRuleSet`` UNKNOWN becomes the graph's
UNKNOWN, never a safe value.
"""

from typing import Iterable, Optional

import numpy as np

from death_rules import UNKNOWN as DEATH_UNKNOWN
from death_rules import DeathRuleSet
from query_graph import (
    UNKNOWN,
    CallablePredictor,
    DtSpec,
    Pattern,
    Rule,
    Slot,
)


def make_death_event_rule(
    rule_set: DeathRuleSet,
    state_slots: Iterable[Slot],
    target_slot: Slot = Slot("event", 1),
    name: str = "death-rules",
    support: int = 1,
    confidence: float = 1.0,
) -> Rule:
    """Wrap a fitted DeathRuleSet as a one-step event rule."""
    slots = tuple(state_slots)

    def predict(state) -> object:
        values = []
        for slot in slots:
            value = state.get(slot, UNKNOWN)
            if value is UNKNOWN:
                return UNKNOWN
            values.append(float(value))
        prediction = int(rule_set.classify(
            np.asarray(values, dtype=float).reshape(1, -1)
        )[0])
        if prediction == int(DEATH_UNKNOWN):
            return UNKNOWN
        return int(prediction)

    return Rule(
        name=name,
        pattern_in=Pattern(slots),
        pattern_out=Pattern((target_slot,)),
        dt=DtSpec.steps(1),
        predictor=CallablePredictor(predict),
        support=int(support),
        confidence=float(confidence),
    )
