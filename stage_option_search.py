"""Exhaustive option-level search over learned stage transitions.

This module is the decision-side counterpart of ``StageOutcomeMemory``.  The
memory predicts, for one realized action option,

    (stage context, option) ->
        (destination stage, duration, feedback, terminal/censored)

The search composes those non-frame transitions into complete paths.  A path
advances one *stage* at a time, so its depth counts stages, not frames; elapsed
time is the sum of the realized stage durations along the path.

Completeness and accuracy are the two hard requirements here:

* no beam, no top-k and no score-based pruning: every option the fitted memory
  supports at a node is expanded and every outcome of a supported edge is
  followed;
* an edge's outcome probabilities are used exactly as reported, and a search
  raises if they do not sum to one;
* an unsupported option is never given a destination, a duration or a
  probability.  A node with no supported option is reported as unresolved mass;
* every supported edge's outcome mass is checked against one before the edge is
  expanded, so a malformed or truncated outcome table raises instead of
  silently shrinking the tree.

Option probabilities are conditional, not marginal: the memory conditions on
an option that the environment actually executed, so it does not model
``P(option)``.  Options are decisions, not random events.  A leaf's
probability is therefore the exact product of
``P(outcome | context, option)`` along the chosen option sequence, and the
enumeration is a *family* of conditional outcome distributions - one per option
sequence.  No cross-option normalization is performed or implied.  The caller
supplies any option preference or value; this module never invents one.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from stage_outcome_memory import (
    ActionTrace,
    StageKey,
    _native,
)


TERMINAL = "terminal"
CENSORED = "censored"
OPEN = "open"
DEPTH_LIMIT = "max_transitions"
NO_SUPPORTED_OPTION = "no_supported_option"


class StageOptionSearchCapacityError(RuntimeError):
    """Raised when an exhaustive search would exceed ``max_leaves``."""


@dataclass(frozen=True)
class OptionStep:
    """One realized stage transition taken under a chosen option."""

    context: Tuple[StageKey, ...]
    option: ActionTrace
    destination: Optional[StageKey]
    terminal: bool
    censored: bool
    probability: float
    support: int
    episode_support: int
    duration_mean: float
    feedback_mean: Tuple[float, ...]
    context_depth: int
    resolution: Optional[int]


@dataclass(frozen=True)
class OptionLeaf:
    """One fully expanded path from the root context.

    ``outcome`` is ``terminal``, ``censored`` or ``open``.  ``open`` means the
    path reached ``max_transitions`` while still supported; it is a depth cut,
    not an unknown transition.
    """

    options: Tuple[ActionTrace, ...]
    steps: Tuple[OptionStep, ...]
    probability: float
    elapsed_mean: float
    feedback_mean: Tuple[float, ...]
    outcome: str
    final_context: Tuple[StageKey, ...]
    reason: Optional[str] = None


@dataclass(frozen=True)
class UnresolvedNode:
    """Mass that reached a context with no supported option.

    ``probability`` is the probability of reaching the context, not the
    probability of an outcome.  The future is unknown, so the mass is reported
    separately and is never renormalized into the resolved leaves.
    """

    options: Tuple[ActionTrace, ...]
    context: Tuple[StageKey, ...]
    probability: float
    reason: str


@dataclass(frozen=True)
class PlanSummary:
    """The exact outcome distribution of one option sequence."""

    options: Tuple[ActionTrace, ...]
    outcome_mass: Tuple[Tuple[str, float], ...]
    known_mass: float
    expected_elapsed: float
    expected_feedback: Tuple[float, ...]
    leaves: int


@dataclass(frozen=True)
class OptionSearchResult:
    """A complete, unpruned enumeration of stage-level paths.

    ``leaves`` and ``unresolved`` are exhaustive over the fitted option
    vocabulary.  Their probabilities are conditional path likelihoods, so they
    are not summed to one across different option sequences; see the module
    docstring.
    """

    root_context: Tuple[StageKey, ...]
    max_transitions: int
    min_context_depth: int
    leaves: Tuple[OptionLeaf, ...]
    unresolved: Tuple[UnresolvedNode, ...]
    counts: dict
    complete: bool

    @property
    def known_mass(self) -> float:
        """Sum of the enumerated conditional path likelihoods."""
        return float(sum(leaf.probability for leaf in self.leaves))

    @property
    def unresolved_mass(self) -> float:
        """Mass whose next stage has no supported option."""
        return float(sum(node.probability for node in self.unresolved))

    @property
    def enumerated_mass(self) -> float:
        return self.known_mass + self.unresolved_mass

    def plans(self) -> Tuple[PlanSummary, ...]:
        """Group leaves by the exact option sequence that produced them.

        ``known_mass`` is the probability of reaching that plan's depth under
        its prefix; it is not renormalized to one, because a prefix outcome may
        have ended earlier (terminal/censored) or the same option sequence may
        not be supported after every branch.
        """
        groups = {}
        for leaf in self.leaves:
            group = groups.setdefault(leaf.options, {
                "leaves": [],
                "outcomes": Counter(),
                "elapsed": 0.0,
                "feedback": None,
            })
            group["leaves"].append(leaf)
            group["outcomes"][leaf.outcome] += float(leaf.probability)
            group["elapsed"] += float(leaf.probability) * float(
                leaf.elapsed_mean
            )
            if leaf.feedback_mean:
                vector = np.asarray(leaf.feedback_mean, dtype=float)
                if group["feedback"] is None:
                    group["feedback"] = np.zeros_like(vector)
                elif group["feedback"].shape != vector.shape:
                    raise ValueError(
                        "feedback dimensions differ inside one plan"
                    )
                group["feedback"] += float(leaf.probability) * vector
        summaries = []
        for options, group in groups.items():
            known = float(sum(group["outcomes"].values()))
            if known > 0.0:
                elapsed = group["elapsed"] / known
                feedback = (
                    ()
                    if group["feedback"] is None
                    else tuple(
                        float(value)
                        for value in group["feedback"] / known
                    )
                )
            else:
                elapsed = 0.0
                feedback = ()
            summaries.append(PlanSummary(
                options=options,
                outcome_mass=tuple(sorted(group["outcomes"].items())),
                known_mass=known,
                expected_elapsed=float(elapsed),
                expected_feedback=feedback,
                leaves=len(group["leaves"]),
            ))
        summaries.sort(key=lambda plan: (
            tuple(repr(option) for option in plan.options),
        ))
        return tuple(summaries)

    def snapshot(self):
        plans = self.plans()
        return dict(
            root_depth=len(self.root_context),
            max_transitions=int(self.max_transitions),
            min_context_depth=int(self.min_context_depth),
            leaves=len(self.leaves),
            plans=len(plans),
            unresolved=int(len(self.unresolved)),
            known_mass=float(self.known_mass),
            unresolved_mass=float(self.unresolved_mass),
            enumerated_mass=float(self.enumerated_mass),
            complete=bool(self.complete),
            counts=dict(self.counts),
        )


def _normalize_history(stage_history: Sequence[StageKey]):
    history = tuple(
        tuple(_native(token) for token in key) for key in stage_history
    )
    if not history:
        raise ValueError(
            "stage_history must end with the current stage key"
        )
    return history


def _add_feedback(current, row):
    """Accumulate the *destination* stage's feedback; end outcomes add none."""
    values = tuple(float(value) for value in (row.get("feedback_mean") or ()))
    if not values:
        return current
    vector = np.asarray(values, dtype=float)
    if current is None:
        return vector
    if current.shape != vector.shape:
        raise ValueError("feedback dimensions differ across an option edge")
    return current + vector


class StageOptionSearch:
    """Enumerate every stage-level path the fitted memory supports.

    The option vocabulary is the set of realized action traces stored in the
    memory.  Options are exact sequences on purpose: an unseen trace is not a
    supported edge and is never replaced by a pooled or counterfactual one.

    ``min_context_depth`` (default 1) rejects a prediction whose chosen
    backoff context no longer contains the current stage.  The memory's root
    context pools every source stage, which is a useful marginal for a target
    readout but is not a transition from *this* stage, so the search treats it
    as unsupported instead of expanding a fabricated edge.
    """

    def __init__(self, memory, max_transitions: int = 2,
                 max_leaves: int = 200000, tolerance: float = 1e-6,
                 min_context_depth: int = 1,
                 allow_truncation: bool = False):
        if memory is None:
            raise ValueError("a fitted StageOutcomeMemory is required")
        if int(max_transitions) < 1:
            raise ValueError("max_transitions must be positive")
        if int(max_leaves) < 1:
            raise ValueError("max_leaves must be positive")
        if float(tolerance) <= 0.0:
            raise ValueError("tolerance must be positive")
        if int(min_context_depth) < 0:
            raise ValueError("min_context_depth must be non-negative")
        self.memory = memory
        self.max_transitions = int(max_transitions)
        self.max_leaves = int(max_leaves)
        self.tolerance = float(tolerance)
        self.min_context_depth = int(min_context_depth)
        self.allow_truncation = bool(allow_truncation)

    def search(self, stage_history: Sequence[StageKey],
               root_elapsed: float = 0.0,
               root_feedback=None) -> OptionSearchResult:
        """Return every supported path from ``stage_history``.

        ``root_elapsed`` is the time already spent in the current stage and
        ``root_feedback`` its realized feedback so far.  Both are constants
        added to every leaf; they are not predicted by the memory.
        """
        context = _normalize_history(stage_history)
        root_elapsed = float(root_elapsed)
        if not np.isfinite(root_elapsed) or root_elapsed < 0.0:
            raise ValueError("root_elapsed must be finite and non-negative")
        root_feedback = (
            None if root_feedback is None
            else np.asarray(root_feedback, dtype=float).reshape(-1)
        )
        if root_feedback is not None and not np.isfinite(root_feedback).all():
            raise ValueError("root_feedback must be finite")

        leaves = []
        unresolved = []
        counts = Counter(
            leaf_terminal=0, leaf_censored=0, leaf_open=0,
            backoff_refused=0, unknown_options=0,
        )
        cache = {}
        state = dict(
            stopped=False, complete=True, max_depth=0,
            max_edge_deviation=0.0,
        )
        options = tuple(sorted(self.memory.action_options, key=repr))
        counts["option_vocabulary"] = len(options)
        counts["queries"] = 0

        def supported_options(node):
            cached = cache.get(node)
            if cached is not None:
                return cached
            supported = []
            for option in options:
                prediction = self.memory.predict(node, option)
                counts["queries"] += 1
                if prediction.get("status") != "supported":
                    counts["unknown_options"] += 1
                    continue
                context_depth = int(prediction.get("context_depth") or 0)
                if context_depth < self.min_context_depth:
                    # A root-only prediction pools unrelated source stages,
                    # so it is not a transition from this node.  Refusing it
                    # keeps the search from fabricating an edge.
                    counts["backoff_refused"] += 1
                    counts["unknown_options"] += 1
                    continue
                mass = float(sum(
                    float(row["probability"])
                    for row in prediction.get("outcomes", ())
                ))
                deviation = abs(mass - 1.0)
                counts["edge_mass_checks"] += 1
                if deviation > state["max_edge_deviation"]:
                    state["max_edge_deviation"] = deviation
                if deviation > self.tolerance:
                    raise ValueError(
                        "option outcomes must sum to one; got %.12f at %r"
                        % (mass, node)
                    )
                if not prediction.get("outcomes"):
                    raise ValueError(
                        "supported edge must report at least one outcome"
                    )
                supported.append((option, prediction))
            cache[node] = tuple(supported)
            counts["supported_options"] += len(supported)
            return cache[node]

        def emit(leaf):
            leaves.append(leaf)
            counts["leaves"] += 1
            counts["leaf_%s" % leaf.outcome] += 1
            if len(leaves) >= self.max_leaves:
                if not self.allow_truncation:
                    raise StageOptionSearchCapacityError(
                        "exhaustive search reached max_leaves=%d; raise the "
                        "cap or reduce max_transitions"
                        % self.max_leaves
                    )
                state["complete"] = False
                state["stopped"] = True

        def visit(node, chosen, steps, probability, elapsed, feedback,
                  remaining):
            if state["stopped"]:
                return
            depth = len(steps)
            state["max_depth"] = max(state["max_depth"], depth)
            counts["nodes"] += 1
            if remaining <= 0:
                emit(OptionLeaf(
                    options=chosen,
                    steps=steps,
                    probability=float(probability),
                    elapsed_mean=float(elapsed),
                    feedback_mean=(
                        () if feedback is None
                        else tuple(float(value) for value in feedback)
                    ),
                    outcome=OPEN,
                    final_context=node,
                    reason=DEPTH_LIMIT,
                ))
                return
            supported = supported_options(node)
            if not supported:
                counts["unresolved"] += 1
                unresolved.append(UnresolvedNode(
                    options=chosen,
                    context=node,
                    probability=float(probability),
                    reason=NO_SUPPORTED_OPTION,
                ))
                return
            for option, prediction in supported:
                counts["edges"] += 1
                for row in prediction.get("outcomes", ()):
                    edge_probability = float(row["probability"])
                    if edge_probability <= 0.0:
                        raise ValueError(
                            "a supported outcome must have positive mass"
                        )
                    destination = row.get("destination")
                    step = OptionStep(
                        context=node,
                        option=option,
                        destination=(
                            None if destination is None
                            else tuple(
                                _native(token) for token in destination
                            )
                        ),
                        terminal=bool(row.get("terminal", False)),
                        censored=bool(row.get("censored", False)),
                        probability=edge_probability,
                        support=int(row.get("count", 0)),
                        episode_support=int(row.get("episode_support", 0)),
                        duration_mean=float(
                            (row.get("duration") or {}).get("mean", 0.0)
                        ),
                        feedback_mean=tuple(
                            float(value)
                            for value in (row.get("feedback_mean") or ())
                        ),
                        context_depth=int(prediction.get("context_depth") or 0),
                        resolution=prediction.get("resolution"),
                    )
                    probability_next = float(probability) * edge_probability
                    if step.terminal or step.censored:
                        emit(OptionLeaf(
                            options=chosen + (option,),
                            steps=steps + (step,),
                            probability=probability_next,
                            elapsed_mean=float(elapsed),
                            feedback_mean=(
                                () if feedback is None
                                else tuple(float(value) for value in feedback)
                            ),
                            outcome=(
                                TERMINAL if step.terminal else CENSORED
                            ),
                            final_context=node,
                            reason=None,
                        ))
                        continue
                    if step.destination is None:
                        raise ValueError(
                            "a non-terminal outcome must report a destination"
                        )
                    # Only a stage we actually enter contributes duration and
                    # feedback; a terminal/censored row describes the stage we
                    # are already in and would otherwise double count it.
                    visit(
                        node + (step.destination,),
                        chosen + (option,),
                        steps + (step,),
                        probability_next,
                        float(elapsed) + step.duration_mean,
                        _add_feedback(feedback, row),
                        remaining - 1,
                    )

        visit(
            context, (), (), 1.0, root_elapsed,
            None if root_feedback is None else root_feedback.copy(),
            self.max_transitions,
        )

        counts["max_depth"] = int(state["max_depth"])
        counts["unique_nodes"] = len(cache)
        result_counts = {
            str(key): int(value)
            for key, value in sorted(counts.items())
        }
        result_counts["enumerated_mass"] = (
            float(sum(leaf.probability for leaf in leaves))
            + float(sum(node.probability for node in unresolved))
        )
        result_counts["max_edge_deviation"] = float(
            state["max_edge_deviation"]
        )
        return OptionSearchResult(
            root_context=context,
            max_transitions=int(self.max_transitions),
            min_context_depth=int(self.min_context_depth),
            leaves=tuple(leaves),
            unresolved=tuple(unresolved),
            counts=result_counts,
            complete=bool(state["complete"]),
        )


def exhaustive_option_search(memory, stage_history: Sequence[StageKey],
                             max_transitions: int = 2,
                             **kwargs) -> OptionSearchResult:
    """Convenience wrapper for a one-shot exhaustive stage-level search."""
    return StageOptionSearch(
        memory, max_transitions=max_transitions, **kwargs,
    ).search(stage_history)
