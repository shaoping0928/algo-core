"""Query-driven transition rewriting over an anonymous state graph.

This module is the structural substrate for the Batch 6 plan.  It does not
name a task, a frame, or a stage: a node is a slot name plus a time index, a
rule is a pair of slot patterns plus a time specification, and a query asks
for a set of target slots from a source graph.

The design follows the query process:

    source graph -> target slots -> backward retrieval -> forward application
    -> readout

A rule is admitted only when its output pattern can unify with the requested
slot and its time specification is compatible.  There is no hard-coded
``FRAME``/``STAGE`` enum: a frame rule is simply one whose output slots sit at
``+1`` and whose ``dt`` is exactly one; a stage rule is one whose output sits
at a variable offset and whose ``dt`` is an interval.  The runtime matches
patterns, not labels.

Rules carry their own applicability (entry condition), support and confidence.
A bank may also carry one shared applicability, which is checked once per
query and then selects a rule group; this is what keeps a shared-regime query
linear in the number of regimes instead of the product over targets.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


class _Unknown:
    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN = _Unknown()

ACTIVE = "active"
INACTIVE = "inactive"
UNKNOWN_STATUS = "unknown"


@dataclass(frozen=True)
class Slot:
    """A node identity plus an integer time index."""

    name: str
    time: int = 0

    def at(self, time: int) -> "Slot":
        return Slot(self.name, int(time))


@dataclass(frozen=True)
class Pattern:
    """A set of slots, with times relative to the rule's application."""

    slots: Tuple[Slot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))

    def names(self) -> Tuple[str, ...]:
        return tuple(slot.name for slot in self.slots)

    def contains(self, slot: Slot) -> bool:
        return slot in self.slots


@dataclass(frozen=True)
class DtSpec:
    """Exact one-step time or a bounded variable-duration interval."""

    exact: Optional[int] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None

    @classmethod
    def steps(cls, count: int = 1) -> "DtSpec":
        return cls(exact=int(count))

    @classmethod
    def interval(cls, minimum: int, maximum: int) -> "DtSpec":
        if int(minimum) > int(maximum):
            raise ValueError("minimum must not exceed maximum")
        return cls(exact=None, minimum=int(minimum), maximum=int(maximum))

    @property
    def is_exact(self) -> bool:
        return self.exact is not None

    def allows(self, count: int) -> bool:
        count = int(count)
        if self.exact is not None:
            return count == self.exact
        if self.minimum is not None and count < self.minimum:
            return False
        if self.maximum is not None and count > self.maximum:
            return False
        return True


@dataclass(frozen=True)
class TimeWindow:
    """A coarse target: any outcome time in the inclusive interval."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if int(self.minimum) > int(self.maximum):
            raise ValueError("minimum must not exceed maximum")

    def contains(self, time: int) -> bool:
        return int(self.minimum) <= int(time) <= int(self.maximum)

    def overlaps(self, minimum: int, maximum: int) -> bool:
        return int(self.maximum) >= int(minimum) and (
            int(self.minimum) <= int(maximum)
        )


@dataclass
class Applicability:
    """A learned entry condition over source slots.

    ``center``/``radius`` define an optional region in the selected slots;
    support and confidence are hard gates.  Missing slots return UNKNOWN,
    never ACTIVE.
    """

    slots: Tuple[Slot, ...] = ()
    center: Optional[np.ndarray] = None
    radius: Optional[float] = None
    support: int = 0
    confidence: float = 1.0
    min_support: int = 0
    min_confidence: float = 0.0

    def check(self, state: Mapping[Slot, Any]) -> str:
        if int(self.support) < int(self.min_support):
            return UNKNOWN_STATUS
        if float(self.confidence) < float(self.min_confidence):
            return UNKNOWN_STATUS
        if self.center is None or self.radius is None:
            return ACTIVE
        values = []
        for slot in self.slots:
            value = state.get(slot, UNKNOWN)
            if value is UNKNOWN:
                return UNKNOWN_STATUS
            values.append(float(value))
        distance = float(np.linalg.norm(
            np.asarray(values, dtype=float) - np.asarray(self.center, dtype=float)
        ))
        return ACTIVE if distance <= float(self.radius) else INACTIVE


class Predictor:
    def predict(self, state: Mapping[Slot, Any]) -> Any:
        raise NotImplementedError


@dataclass
class LinearPredictor(Predictor):
    """Intercept plus a linear combination of the rule's input slots."""

    input_slots: Tuple[Slot, ...]
    weights: np.ndarray
    output_mean: float = 0.0
    output_scale: float = 1.0

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)

    def predict(self, state: Mapping[Slot, Any]) -> Any:
        values = [1.0]
        for slot in self.input_slots:
            value = state.get(slot, UNKNOWN)
            if value is UNKNOWN:
                return UNKNOWN
            values.append(float(value))
        if len(values) != len(self.weights):
            return UNKNOWN
        return float(
            self.output_mean
            + self.output_scale * float(np.dot(self.weights, values))
        )


@dataclass
class ThresholdPredictor(Predictor):
    """A discrete predicate over one input slot."""

    slot: Slot
    threshold: float
    above: Any = 1
    below: Any = 0
    direction: int = 1

    def predict(self, state: Mapping[Slot, Any]) -> Any:
        value = state.get(self.slot, UNKNOWN)
        if value is UNKNOWN:
            return UNKNOWN
        value = float(value)
        if int(self.direction) >= 0:
            return self.above if value > float(self.threshold) else self.below
        return self.above if value < float(self.threshold) else self.below


@dataclass
class CallablePredictor(Predictor):
    function: Callable[[Mapping[Slot, Any]], Any]

    def predict(self, state: Mapping[Slot, Any]) -> Any:
        return self.function(state)


@dataclass
class Rule:
    """A local transition: input pattern -> output pattern at a time offset."""

    name: str
    pattern_in: Pattern
    pattern_out: Pattern
    dt: DtSpec
    predictor: Predictor
    applicability: Applicability = field(default_factory=Applicability)
    support: int = 0
    confidence: float = 1.0
    group: Optional[int] = None

    def predict(self, state: Mapping[Slot, Any]) -> Any:
        return self.predictor.predict(state)


@dataclass
class Readout:
    """A consumer that declares the slots it needs and returns 0/1/UNKNOWN."""

    name: str
    required_slots: Tuple[Slot, ...]
    function: Callable[[Mapping[Slot, Any]], Any]
    support: int = 0
    confidence: float = 1.0

    def run(self, state: Mapping[Slot, Any]) -> Any:
        for slot in self.required_slots:
            if state.get(slot, UNKNOWN) is UNKNOWN:
                return UNKNOWN
        return self.function(state)


@dataclass
class Query:
    target_slots: Tuple[Slot, ...]
    source: Mapping[Slot, Any]
    readout: Optional[Readout] = None
    budget: int = 1000
    group: Optional[int] = None
    target_windows: Optional[Mapping[Slot, TimeWindow]] = None


@dataclass
class QueryResult:
    values: Dict[Slot, Any]
    visited: int
    pruned: int
    applied: int
    unknown: int
    conflicts: int
    shared_status: str
    readout: Any
    readout_unknown: bool
    times: Dict[Slot, Optional[int]] = field(default_factory=dict)
    coarse: Tuple[Slot, ...] = ()


class RuleBank:
    """Rules indexed by output name; application is demand-driven."""

    def __init__(
        self,
        rules: Iterable[Rule],
        shared_applicability: Optional[Applicability] = None,
    ) -> None:
        self.rules: List[Rule] = list(rules)
        self.shared_applicability = shared_applicability
        self._by_output: Dict[str, List[Rule]] = {}
        for rule in self.rules:
            for slot in rule.pattern_out.slots:
                self._by_output.setdefault(slot.name, []).append(rule)
        self._checked: Dict[str, str] = {}

    def candidates(self, slot: Slot) -> List[Rule]:
        """Rules whose output can unify with ``slot`` under exact time."""
        found = []
        for rule in self._by_output.get(slot.name, ()):
            if not rule.dt.is_exact:
                # An exact target cannot be filled by a variable-duration
                # rule; this is the structural coarse/fine separation.
                continue
            application_time = int(slot.time) - int(rule.dt.exact)
            if application_time < 0:
                continue
            if any(
                int(output.time) != int(rule.dt.exact)
                for output in rule.pattern_out.slots
            ):
                continue
            found.append(rule)
        return found

    def query(self, query: Query) -> QueryResult:
        return _Engine(self, query).run()

    def candidates_for_window(self, slot: Slot,
                              window: TimeWindow) -> List[tuple]:
        """Candidates for a coarse target window.

        Exact-dt rules fill the nominal ``slot.time`` if it lies in the
        window.  A variable-dt rule is admitted only when its output slot is
        relative time 0 and its duration interval overlaps the window; its
        result is marked coarse and its actual time is unknown.
        """
        found = []
        if not window.contains(slot.time):
            return found
        for rule in self._by_output.get(slot.name, ()):
            if rule.dt.is_exact:
                application_time = int(slot.time) - int(rule.dt.exact)
                if application_time < 0:
                    continue
                if any(
                    int(output.time) != int(rule.dt.exact)
                    for output in rule.pattern_out.slots
                ):
                    continue
                found.append((rule, int(slot.time), False, application_time))
            else:
                if rule.dt.minimum is None or rule.dt.maximum is None:
                    continue
                if any(int(output.time) != 0 for output in rule.pattern_out.slots):
                    continue
                if not window.overlaps(rule.dt.minimum, rule.dt.maximum):
                    continue
                found.append((rule, None, True, 0))
        return found


class _Engine:
    def __init__(self, bank: RuleBank, query: Query) -> None:
        self.bank = bank
        self.query = query
        self.values: Dict[Slot, Any] = dict(query.source)
        self.visited = 0
        self.pruned = 0
        self.applied = 0
        self.conflicts = 0
        self.shared_status = "none"
        self._stack: set = set()
        self._resolved: Dict[Slot, Any] = {}
        self._checks: Dict[str, str] = {}
        self.times: Dict[Slot, Optional[int]] = {
            slot: slot.time for slot in query.source
        }
        self.coarse_slots: set = set()
        self._resolved_times: Dict[Slot, Optional[int]] = {}
        self._resolved_coarse: Dict[Slot, bool] = {}

    def run(self) -> QueryResult:
        if self.bank.shared_applicability is not None:
            status = self.bank.shared_applicability.check(self.values)
            self.shared_status = status
            if status != ACTIVE:
                for slot in self.query.target_slots:
                    self.values[slot] = UNKNOWN
                    self.times[slot] = None
                return self._result()
        for slot in self.query.target_slots:
            self._resolve(slot)
        readout_value = UNKNOWN
        readout_unknown = False
        if self.query.readout is not None:
            readout_value = self.query.readout.run(self.values)
            readout_unknown = readout_value is UNKNOWN
        return self._result(readout_value, readout_unknown)

    def _result(self, readout_value: Any = UNKNOWN,
                readout_unknown: bool = False) -> QueryResult:
        unknown = sum(
            1 for slot in self.query.target_slots
            if self.values.get(slot, UNKNOWN) is UNKNOWN
        )
        coarse = tuple(
            slot for slot in self.query.target_slots
            if slot in self.coarse_slots
        )
        return QueryResult(
            values=dict(self.values),
            visited=self.visited,
            pruned=self.pruned,
            applied=self.applied,
            unknown=int(unknown),
            conflicts=int(self.conflicts),
            shared_status=self.shared_status,
            readout=readout_value,
            readout_unknown=bool(readout_unknown),
            times=dict(self.times),
            coarse=coarse,
        )

    def _window_for(self, slot: Slot) -> Optional[TimeWindow]:
        if self.query.target_windows is None:
            return None
        return self.query.target_windows.get(slot)

    def _resolve(self, slot: Slot) -> Any:
        if slot in self.values and self.values[slot] is not UNKNOWN:
            self.times.setdefault(slot, slot.time)
            return self.values[slot]
        if slot in self._resolved and self._resolved[slot] is not UNKNOWN:
            value = self._resolved[slot]
            self.values[slot] = value
            self.times[slot] = self._resolved_times.get(slot)
            if self._resolved_coarse.get(slot, False):
                self.coarse_slots.add(slot)
            return value
        if self.visited >= int(self.query.budget):
            self.values[slot] = UNKNOWN
            self.times[slot] = None
            return UNKNOWN
        if slot in self._stack:
            self.values[slot] = UNKNOWN
            self.times[slot] = None
            return UNKNOWN
        self._stack.add(slot)
        window = self._window_for(slot)
        if window is None:
            candidates = [
                (
                    rule, int(slot.time), False,
                    int(slot.time) - int(rule.dt.exact),
                )
                for rule in self.bank.candidates(slot)
            ]
        else:
            candidates = self.bank.candidates_for_window(slot, window)
        if self.query.group is not None:
            candidates = [
                item for item in candidates
                if item[0].group is None
                or int(item[0].group) == int(self.query.group)
            ]
        outcomes: List[Tuple[Rule, Any, bool, Optional[int]]] = []
        for rule, output_time, is_coarse, application_time in candidates:
            status = self._applicability(rule)
            self.visited += 1
            if status != ACTIVE:
                self.pruned += 1
                continue
            local_state = self._resolve_inputs(rule, int(application_time))
            if local_state is None:
                continue
            prediction = rule.predict(local_state)
            if prediction is UNKNOWN:
                continue
            outcomes.append((rule, prediction, bool(is_coarse), output_time))
        self._stack.discard(slot)
        if not outcomes:
            self.values[slot] = UNKNOWN
            self.times[slot] = None
            return UNKNOWN
        chosen, conflict, is_coarse, output_time = self._choose(outcomes)
        if conflict:
            self.values[slot] = UNKNOWN
            self.times[slot] = None
            self.conflicts += 1
            return UNKNOWN
        self.values[slot] = chosen
        self.times[slot] = output_time
        if is_coarse:
            self.coarse_slots.add(slot)
        self._resolved[slot] = chosen
        self._resolved_times[slot] = output_time
        self._resolved_coarse[slot] = bool(is_coarse)
        self.applied += 1
        return chosen

    def _applicability(self, rule: Rule) -> str:
        cached = self._checks.get(rule.name)
        if cached is not None:
            return cached
        status = rule.applicability.check(self.values)
        self._checks[rule.name] = status
        return status

    def _resolve_inputs(self, rule: Rule,
                        application_time: int) -> Optional[Dict[Slot, Any]]:
        local: Dict[Slot, Any] = {}
        for slot in rule.pattern_in.slots:
            absolute = Slot(slot.name, int(application_time) + int(slot.time))
            value = self._resolve(absolute)
            if value is UNKNOWN:
                return None
            local[slot] = value
        return local

    def _choose(self, outcomes: List[
            Tuple[Rule, Any, bool, Optional[int]]]) -> Tuple[
                Any, bool, bool, Optional[int]]:
        outcomes = sorted(
            outcomes,
            key=lambda item: (
                bool(item[2]),
                -float(item[0].confidence),
                -int(item[0].support),
                item[0].name,
            ),
        )
        best_rule, best_value, best_coarse, best_time = outcomes[0]
        same_resolution = [
            item for item in outcomes
            if bool(item[2]) == bool(best_coarse)
        ]
        conflicting = [
            item[1] for item in same_resolution[1:]
            if not _values_equal(item[1], best_value)
        ]
        if not conflicting:
            return best_value, False, bool(best_coarse), best_time
        if len(same_resolution) > 1:
            gap = float(same_resolution[0][0].confidence) - float(
                same_resolution[1][0].confidence
            )
            if gap > 1e-9:
                return (
                    best_value, False, bool(best_coarse), best_time,
                )
        return UNKNOWN, True, False, None


def _values_equal(left: Any, right: Any) -> bool:
    if left is UNKNOWN or right is UNKNOWN:
        return left is right
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right, (int, float, np.integer, np.floating)
    ):
        return abs(float(left) - float(right)) <= 1e-6
    return left == right


def make_linear_rule(
    name: str,
    target_slot: Slot,
    input_slots: Tuple[Slot, ...],
    weights: np.ndarray,
    confidence: float,
    support: int,
    output_mean: float = 0.0,
    output_scale: float = 1.0,
    applicability: Optional[Applicability] = None,
    group: Optional[int] = None,
) -> Rule:
    return Rule(
        name=name,
        pattern_in=Pattern(tuple(input_slots)),
        pattern_out=Pattern((target_slot,)),
        dt=DtSpec.steps(1),
        predictor=LinearPredictor(
            input_slots=tuple(input_slots),
            weights=weights,
            output_mean=output_mean,
            output_scale=output_scale,
        ),
        applicability=applicability or Applicability(),
        support=int(support),
        confidence=float(confidence),
        group=group,
    )


def make_event_rule(
    name: str,
    target_slot: Slot,
    input_slot: Slot,
    threshold: float,
    confidence: float,
    support: int,
    above: Any = 1,
    below: Any = 0,
    direction: int = 1,
    applicability: Optional[Applicability] = None,
    group: Optional[int] = None,
) -> Rule:
    """A one-step event predicate; its output is a discrete flag."""
    return Rule(
        name=name,
        pattern_in=Pattern((input_slot,)),
        pattern_out=Pattern((target_slot,)),
        dt=DtSpec.steps(1),
        predictor=ThresholdPredictor(
            slot=input_slot, threshold=threshold, above=above, below=below,
            direction=direction,
        ),
        applicability=applicability or Applicability(),
        support=int(support),
        confidence=float(confidence),
        group=group,
    )
