"""Failure-driven refinement of 0/1 death rules.

The loop is task-neutral: it only needs labelled states.  A terminal state
the death side leaves ``UNKNOWN`` is a missed positive; a non-terminal state
the death side calls death is a false alarm.  ``refine_once`` adds those
examples and refits, and ``run_loop`` repeats with a caller-supplied
collector so new examples can come from planner rollouts, world interaction,
or a held-out stream.

The collector must not simply return the data the rules were fitted on; in
that case the failures are already in the training set and refinement has
nothing new to learn.  The intended use is:

    rules.fit(source_states, source_labels)
    history = run_loop(
        rules,
        collect=lambda round: held_out_stream(round),
        rounds=3,
    )

The returned reports keep the before/after coverage and the number of
examples added, so a caller can stop when coverage stops improving or when
false alarms start to grow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from death_rules import DeathRuleSet


@dataclass(frozen=True)
class RefineRound:
    round: int
    added_missed: int
    added_false_alarm: int
    before: dict
    after: dict

    @property
    def coverage_gain(self) -> float:
        return float(
            self.after.get("death_coverage", 0.0)
            - self.before.get("death_coverage", 0.0)
        )

    @property
    def false_positive_gain(self) -> int:
        return int(
            self.after.get("death_covered_negative", 0)
            - self.before.get("death_covered_negative", 0)
        )


def death_failures(rules: DeathRuleSet, states: np.ndarray,
                   labels: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split failures into missed terminals and non-terminal false alarms.

    Only the death side is inspected.  A non-terminal state left UNKNOWN is
    not a failure: the rule set is allowed to abstain.  A terminal state
    left UNKNOWN is a missed positive, and a non-terminal state claimed as
    death is a false alarm.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    labels = (np.asarray(labels, dtype=float).reshape(-1) > 0.5).astype(int)
    if len(states) != len(labels):
        raise ValueError("states/labels shape mismatch")
    prediction = rules.classify_death(states)
    terminal = labels == 1
    missed = terminal & (prediction != 1)
    false_alarm = (~terminal) & (prediction == 1)
    return (
        states[missed], labels[missed],
        states[false_alarm], labels[false_alarm],
    )


def refine_once(rules: DeathRuleSet, states: np.ndarray, labels: np.ndarray,
                include_false_alarms: bool = True) -> RefineRound:
    """Add one stream's failures and refit; return before/after reports."""
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    labels = (np.asarray(labels, dtype=float).reshape(-1) > 0.5).astype(int)
    before = rules.coverage_report(states, labels)
    missed_states, missed_labels, false_states, false_labels = (
        death_failures(rules, states, labels)
    )
    pieces = []
    targets = []
    if len(missed_states):
        pieces.append(missed_states)
        targets.append(missed_labels)
    if include_false_alarms and len(false_states):
        pieces.append(false_states)
        targets.append(false_labels)
    if pieces:
        rules.add_examples(
            np.vstack(pieces), np.concatenate(targets)
        )
    after = rules.coverage_report(states, labels)
    return RefineRound(
        round=-1,
        added_missed=int(len(missed_states)),
        added_false_alarm=int(len(false_states)),
        before=before,
        after=after,
    )


def run_loop(rules: DeathRuleSet,
             collect: Callable[[int], Tuple[np.ndarray, np.ndarray]],
             rounds: int = 3,
             include_false_alarms: bool = True,
             min_coverage_gain: float = 0.0) -> List[RefineRound]:
    """Refine repeatedly from a caller-supplied stream.

    Stops when the requested rounds are exhausted, when no failures are
    added, or when coverage gain falls to ``min_coverage_gain`` after the
    first round.  False alarms are never added silently: the report keeps
    their count and the resulting covered-negative change.
    """
    rounds = max(int(rounds), 0)
    history: List[RefineRound] = []
    for index in range(rounds):
        states, labels = collect(index)
        report = refine_once(
            rules, states, labels,
            include_false_alarms=include_false_alarms,
        )
        report = RefineRound(
            round=index,
            added_missed=report.added_missed,
            added_false_alarm=report.added_false_alarm,
            before=report.before,
            after=report.after,
        )
        history.append(report)
        if report.added_missed == 0 and report.added_false_alarm == 0:
            break
        if index > 0 and report.coverage_gain <= float(min_coverage_gain):
            break
    return history
