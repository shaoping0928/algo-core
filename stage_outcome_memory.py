"""Task-agnostic memory of signal-delimited stage outcomes.

This first slice deliberately separates:

* stage identity, supplied as generic key-signal tokens;
* temporal context, represented by a variable-order suffix tree;
* directed outcomes, learned only from adjacent realized stages.

The memory does not name task events or infer that an outcome is good/safe.
It returns empirical successor, duration, feedback, terminal/censoring,
support, and resolution statistics.  Outcome duration and feedback describe
the destination stage that actually followed; a terminal/censored outcome has
no destination, so it reports the source stage's own duration/feedback at the
episode's end.  Unsupported queries return UNKNOWN.  Duration summaries are
descriptive; censoring is flagged separately and is not yet a calibrated
survival estimator. Stage keys are exact signal tokens; continuous-state
predictive-equivalence merging is a later layer.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


UNKNOWN = "unknown"
StageKey = Tuple[Hashable, ...]
ActionTrace = Tuple[Tuple[Hashable, int], ...]
OutcomeKey = Tuple[Optional[StageKey], bool, bool]


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_native(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_native(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_native(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted(
            ((_native(key), _native(item)) for key, item in value.items()),
            key=repr,
        ))
    hash(value)
    return value


def encode_action_trace(actions: Iterable[Hashable]) -> ActionTrace:
    """Run-length encode an ordered action trace without losing order."""
    result = []
    for action in actions:
        action = _native(action)
        if result and result[-1][0] == action:
            previous, count = result[-1]
            result[-1] = (previous, count + 1)
        else:
            result.append((action, 1))
    return tuple(result)


def stage_key_from_signals(signals: Iterable["KeySignal"]) -> StageKey:
    """Return a stable, task-agnostic key from simultaneous signal events."""
    return tuple(sorted({signal.token for signal in signals}, key=repr))


def stage_context_path(stage_history: Sequence[StageKey],
                       max_depth: Optional[int] = None):
    """Return the current-to-root suffix path for chronological stage keys."""
    history = tuple(
        tuple(_native(token) for token in key) for key in stage_history
    )
    if max_depth is not None:
        max_depth = int(max_depth)
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        history = history[-max_depth:] if max_depth else ()
    return tuple(
        tuple(history[-depth:]) if depth else ()
        for depth in range(len(history), -1, -1)
    )


@dataclass(frozen=True)
class KeySignal:
    """A generic key-signal pulse emitted by a detector.

    ``channel`` is an opaque detector/channel identifier; the memory assigns
    no task meaning to it. Sustained levels should be converted to rising-edge
    pulses by the detector or by :func:`segment_episode`.
    """

    step: int
    channel: Hashable
    polarity: int = 0
    magnitude: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "channel", _native(self.channel))
        object.__setattr__(self, "polarity", int(self.polarity))
        object.__setattr__(self, "magnitude", float(self.magnitude))
        if self.step < 0:
            raise ValueError("signal step must be non-negative")
        if self.polarity not in (-1, 0, 1):
            raise ValueError("polarity must be -1, 0, or 1")
        if not np.isfinite(self.magnitude) or self.magnitude < 0.0:
            raise ValueError("magnitude must be finite and non-negative")

    @property
    def token(self):
        return self.channel, self.polarity


class GenericKeySignalDetector:
    """Fold-fitted detector for generic transition and feedback changes.

    The detector learns one upper-tail threshold per opaque input channel.
    It can consume observation changes, feedback changes, and optional model
    prediction errors. Thresholds are fit only on the episodes passed to
    :meth:`fit`; applying them to held-out episodes does not update the fit.
    """

    def __init__(self, quantile: float = 0.98):
        self.quantile = float(quantile)
        if not 0.5 < self.quantile < 1.0:
            raise ValueError("quantile must be between 0.5 and 1")
        self.thresholds = {}
        self.fitted = False

    @staticmethod
    def _matrix(values, name):
        result = np.asarray(values, dtype=float)
        if result.ndim == 1:
            result = result[:, None]
        if result.ndim != 2 or not np.isfinite(result).all():
            raise ValueError("%s must be a finite vector or matrix" % name)
        return result

    def fit(self, episodes: Sequence[Mapping]):
        """Fit channel thresholds from training episodes only."""
        magnitudes = defaultdict(list)
        dimensions = {}
        presence = {}
        for episode in episodes:
            observations = self._matrix(
                episode["observations"], "observations",
            )
            state_delta = np.diff(observations, axis=0)
            self._check_dimension(
                dimensions, "observation_change", state_delta.shape[1],
            )
            self._append_magnitudes(
                magnitudes, "observation_change", state_delta,
            )

            feedback = episode.get("feedback")
            self._check_presence(presence, "feedback_change", feedback)
            if feedback is not None:
                feedback = self._matrix(feedback, "feedback")
                if len(feedback) != len(state_delta):
                    raise ValueError(
                        "feedback must have one row per transition"
                    )
                self._check_dimension(
                    dimensions, "feedback_change", feedback.shape[1],
                )
                if len(feedback) > 1:
                    self._append_magnitudes(
                        magnitudes, "feedback_change",
                        np.diff(feedback, axis=0),
                    )

            errors = episode.get("prediction_error")
            self._check_presence(presence, "prediction_error", errors)
            if errors is not None:
                errors = self._matrix(errors, "prediction_error")
                if len(errors) != len(state_delta):
                    raise ValueError(
                        "prediction_error must have one row per transition"
                    )
                self._check_dimension(
                    dimensions, "prediction_error", errors.shape[1],
                )
                self._append_magnitudes(
                    magnitudes, "prediction_error", errors,
                )

        if "observation_change" not in dimensions:
            raise ValueError("at least one training episode is required")
        self.thresholds = {
            channel: float(np.quantile(values, self.quantile))
            for channel, values in magnitudes.items()
            if values and float(np.max(values)) > 0.0
        }
        self.dimensions = dimensions
        self.fitted = True
        return self

    @staticmethod
    def _append_magnitudes(target, kind, values):
        for dimension in range(values.shape[1]):
            column = np.abs(values[:, dimension])
            target[(kind, int(dimension))].extend(
                float(value) for value in column if value > 0.0
            )

    @staticmethod
    def _check_dimension(dimensions, kind, size):
        previous = dimensions.get(kind)
        if previous is not None and previous != int(size):
            raise ValueError("%s dimensions must be consistent" % kind)
        dimensions[kind] = int(size)

    @staticmethod
    def _check_presence(presence, kind, value):
        present = value is not None
        previous = presence.get(kind)
        if previous is not None and previous != present:
            raise ValueError("%s must be present in every training episode"
                             % kind)
        presence[kind] = present

    def transform_episode(self, observations, feedback=None,
                          prediction_error=None):
        """Emit rising-edge signals aligned to state boundaries."""
        if not self.fitted:
            raise RuntimeError("fit the key-signal detector before transform")
        observations = self._matrix(observations, "observations")
        if observations.shape[1] != self.dimensions["observation_change"]:
            raise ValueError("observation dimensions differ from training")
        state_delta = np.diff(observations, axis=0)
        transition_count = len(state_delta)
        signals = [[] for _ in range(len(observations))]
        self._emit(
            signals, "observation_change", state_delta,
            step_offset=1, directional=True,
        )

        if feedback is not None:
            feedback = self._matrix(feedback, "feedback")
            expected = self.dimensions.get("feedback_change")
            if expected is not None and feedback.shape[1] != expected:
                raise ValueError("feedback dimensions differ from training")
            if len(feedback) != transition_count:
                raise ValueError(
                    "feedback must have one row per transition"
                )
            feedback_delta = np.zeros_like(feedback)
            if len(feedback) > 1:
                feedback_delta[1:] = np.diff(feedback, axis=0)
            self._emit(
                signals, "feedback_change", feedback_delta,
                step_offset=1, directional=True,
            )
        if prediction_error is not None:
            errors = self._matrix(prediction_error, "prediction_error")
            expected = self.dimensions.get("prediction_error")
            if expected is not None and errors.shape[1] != expected:
                raise ValueError(
                    "prediction_error dimensions differ from training"
                )
            if len(errors) != transition_count:
                raise ValueError(
                    "prediction_error must have one row per transition"
                )
            self._emit(
                signals, "prediction_error", errors,
                step_offset=1, directional=False,
            )
        return tuple(tuple(sorted(
            row, key=lambda signal: repr(signal.token)
        )) for row in signals)

    def _emit(self, output, kind, values, step_offset, directional):
        for dimension in range(values.shape[1]):
            channel = (kind, int(dimension))
            threshold = self.thresholds.get(channel, 0.0)
            if threshold <= 0.0:
                continue
            for index, value in enumerate(values[:, dimension]):
                magnitude = abs(float(value))
                if magnitude < threshold:
                    continue
                polarity = (
                    (1 if value > 0.0 else -1)
                    if directional else 0
                )
                step = index + int(step_offset)
                output[step].append(KeySignal(
                    step=step,
                    channel=channel,
                    polarity=polarity,
                    magnitude=magnitude / threshold,
                ))


@dataclass(frozen=True)
class StageOccurrence:
    """One realized, variable-duration stage in an episode.

    ``active_features`` are the features at stage entry; ``feature_snapshots``
    retain the per-transition activations used for demand association.
    """

    episode: Hashable
    key: StageKey
    entry_state: np.ndarray
    exit_state: np.ndarray
    action_trace: ActionTrace
    duration: float
    feedback: np.ndarray
    terminal: bool = False
    censored: bool = False
    signal_strengths: Tuple[float, ...] = ()
    active_features: Tuple[Hashable, ...] = ()
    feature_snapshots: Tuple[Tuple[Hashable, ...], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "episode", _native(self.episode))
        object.__setattr__(
            self, "key", tuple(_native(token) for token in self.key)
        )
        entry = np.asarray(self.entry_state, dtype=float).reshape(-1).copy()
        exit_state = np.asarray(self.exit_state, dtype=float).reshape(-1).copy()
        feedback = np.asarray(self.feedback, dtype=float).reshape(-1).copy()
        if entry.shape != exit_state.shape:
            raise ValueError("entry_state and exit_state dimensions differ")
        if (not np.isfinite(entry).all()
                or not np.isfinite(exit_state).all()
                or not np.isfinite(feedback).all()):
            raise ValueError("stage vectors must be finite")
        trace = tuple(
            (_native(action), int(count))
            for action, count in self.action_trace
        )
        if any(count <= 0 for _action, count in trace):
            raise ValueError("action trace run lengths must be positive")
        duration = float(self.duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        if bool(self.terminal) and bool(self.censored):
            raise ValueError("a stage cannot be terminal and censored")
        strengths = tuple(float(value) for value in self.signal_strengths)
        if (not np.isfinite(strengths).all()
                or any(value < 0.0 for value in strengths)):
            raise ValueError("signal strengths must be finite and non-negative")
        if strengths and len(strengths) != len(self.key):
            raise ValueError("one signal strength is required per key token")
        features = tuple(dict.fromkeys(
            _native(feature) for feature in self.active_features
        ))
        snapshots = tuple(
            tuple(dict.fromkeys(_native(feature) for feature in row))
            for row in self.feature_snapshots
        )
        if not features and snapshots:
            features = snapshots[0]
        object.__setattr__(self, "entry_state", entry)
        object.__setattr__(self, "exit_state", exit_state)
        object.__setattr__(self, "feedback", feedback)
        object.__setattr__(self, "action_trace", trace)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "terminal", bool(self.terminal))
        object.__setattr__(self, "censored", bool(self.censored))
        object.__setattr__(self, "signal_strengths", strengths)
        object.__setattr__(self, "active_features", features)
        object.__setattr__(self, "feature_snapshots", snapshots)


def segment_episode(episode: Hashable, observations, actions, feedback,
                    key_signals: Sequence[Sequence[KeySignal]],
                    terminal: bool = False, censored: bool = False,
                    active_features: Optional[Sequence[Iterable[Hashable]]] = None):
    """Build variable-duration stages from generic signal onsets.

    ``observations`` has one more row than ``actions``. ``actions`` must be
    the actions that actually acted on those transitions. ``key_signals`` is
    indexed at state boundaries and contains event pulses. Repeated sustained
    signals only create a boundary on their rising edge. A stage starts at an
    event boundary and ends at the next one; no fixed frame horizon is used.
    Optional ``active_features`` are per-transition read-only feature IDs.
    """
    observations = np.asarray(observations, dtype=float)
    actions = [_native(action) for action in actions]
    feedback = np.asarray(feedback, dtype=float)
    if feedback.ndim == 1:
        feedback = feedback[:, None]
    if observations.ndim != 2:
        raise ValueError("observations must be a 2D array")
    if len(observations) != len(actions) + 1:
        raise ValueError("observations must have one more row than actions")
    if feedback.ndim != 2 or len(feedback) != len(actions):
        raise ValueError("feedback must have one row per action")
    if len(key_signals) != len(observations):
        raise ValueError("key_signals must align with observation boundaries")
    if (active_features is not None
            and len(active_features) != len(actions)):
        raise ValueError("active_features must have one row per action")
    if (not np.isfinite(observations).all()
            or not np.isfinite(feedback).all()):
        raise ValueError("observations and feedback must be finite")
    if terminal and censored:
        raise ValueError("episode cannot be terminal and censored")
    if not actions:
        return []

    boundary_signals = {}
    previous = set()
    for step, signals in enumerate(key_signals[:-1]):
        if any(signal.step != step for signal in signals):
            raise ValueError("key signal step does not match its boundary")
        current = {signal.token: signal for signal in signals}
        rising = set(current) - previous
        if rising:
            boundary_signals[step] = tuple(
                current[token] for token in sorted(rising, key=repr)
            )
        previous = set(current)

    starts = [0] + sorted(
        step for step in boundary_signals if 0 < step < len(actions)
    )
    occurrences = []
    for position, start in enumerate(starts):
        end = (
            starts[position + 1]
            if position + 1 < len(starts) else len(actions)
        )
        stage_terminal = bool(terminal and end == len(actions))
        stage_censored = bool(censored and end == len(actions))
        event_signals = boundary_signals.get(start, ())
        key = stage_key_from_signals(event_signals)
        feature_snapshots = (
            tuple(tuple(dict.fromkeys(
                _native(feature) for feature in row
            )) for row in active_features[start:end])
            if active_features is not None else ()
        )
        occurrences.append(StageOccurrence(
            episode=episode,
            key=key,
            entry_state=observations[start],
            exit_state=observations[end],
            action_trace=encode_action_trace(actions[start:end]),
            duration=float(end - start),
            feedback=feedback[start:end].sum(axis=0),
            terminal=stage_terminal,
            censored=stage_censored,
            signal_strengths=tuple(
                signal.magnitude for signal in event_signals
            ),
            active_features=(
                feature_snapshots[0] if feature_snapshots else ()
            ),
            feature_snapshots=feature_snapshots,
        ))
    return occurrences


@dataclass
class _OutcomeCounts:
    count: int = 0
    episodes: set = field(default_factory=set)
    durations: list = field(default_factory=list)
    feedback_sum: Optional[np.ndarray] = None
    feedback_sq_sum: Optional[np.ndarray] = None
    destination_terminal_count: int = 0
    destination_censored_count: int = 0

    def add(self, occurrence: StageOccurrence,
            destination_stage: Optional[StageOccurrence] = None):
        """Accumulate the realized outcome's statistics.

        For a successor outcome the duration and feedback come from the
        destination stage - the part of the episode the transition leads to.
        A terminal/censored outcome has no destination, so the source stage's
        own duration/feedback describe the episode's end.
        """
        self.count += 1
        self.episodes.add(occurrence.episode)
        stats_stage = (
            occurrence if destination_stage is None else destination_stage
        )
        self.durations.append(float(stats_stage.duration))
        feedback = np.asarray(stats_stage.feedback, dtype=float).reshape(-1)
        if self.feedback_sum is None:
            self.feedback_sum = np.zeros_like(feedback)
            self.feedback_sq_sum = np.zeros_like(feedback)
        if self.feedback_sum.shape != feedback.shape:
            raise ValueError("feedback dimensions must be consistent")
        self.feedback_sum += feedback
        self.feedback_sq_sum += feedback ** 2
        if destination_stage is not None:
            self.destination_terminal_count += int(
                bool(destination_stage.terminal)
            )
            self.destination_censored_count += int(
                bool(destination_stage.censored)
            )

    def summary(self, destination, terminal, censored, total):
        durations = np.asarray(self.durations, dtype=float)
        feedback_mean = self.feedback_sum / float(self.count)
        feedback_variance = np.maximum(
            self.feedback_sq_sum / float(self.count)
            - feedback_mean ** 2,
            0.0,
        )
        return dict(
            destination=destination,
            terminal=bool(terminal),
            censored=bool(censored),
            destination_terminal_rate=(
                float(self.destination_terminal_count) / float(self.count)
            ),
            destination_censored_rate=(
                float(self.destination_censored_count) / float(self.count)
            ),
            count=int(self.count),
            episode_support=int(len(self.episodes)),
            probability=float(self.count) / float(total),
            duration=dict(
                mean=float(durations.mean()),
                median=float(np.quantile(durations, 0.5)),
                q90=float(np.quantile(durations, 0.9)),
                minimum=float(durations.min()),
                maximum=float(durations.max()),
            ),
            feedback_mean=feedback_mean.tolist(),
            feedback_variance=feedback_variance.tolist(),
        )


class StageOutcomeMemory:
    """Variable-order stage context tree with empirical sibling outcomes.

    Each context is the current stage plus zero or more preceding stage keys.
    Moving toward the root drops the oldest context key. Sibling outcomes are
    the distinct next-stage keys observed under the same context and exact
    ordered action option. Context depth is selected by episode support, not
    a fixed number of semantic levels. Stage keys are currently exact key
    signal signatures; this class does not merge different signatures by
    continuous-state predictive equivalence.
    """

    def __init__(self, min_support: int = 2, min_episodes: int = 2,
                 max_context_depth: Optional[int] = None):
        if int(min_support) < 1:
            raise ValueError("min_support must be positive")
        if int(min_episodes) < 1:
            raise ValueError("min_episodes must be positive")
        if (max_context_depth is not None
                and int(max_context_depth) < 1):
            raise ValueError("max_context_depth must be positive or None")
        self.min_support = int(min_support)
        self.min_episodes = int(min_episodes)
        self.max_context_depth = (
            None if max_context_depth is None
            else int(max_context_depth)
        )
        self._counts = {}
        self._pooled = {}
        self.stage_types = set()
        self.contexts = set()
        self.action_options = set()
        self.realized_outcomes = 0

    def fit(self, episodes: Sequence[Sequence[StageOccurrence]]):
        """Fit only from realized, ordered stage occurrences."""
        self._counts = {}
        self._pooled = {}
        self.stage_types = set()
        self.contexts = {()}
        self.action_options = set()
        self.realized_outcomes = 0
        state_dim = None
        feedback_dim = None

        for episode in episodes:
            stages = list(episode)
            if not stages:
                continue
            episode_ids = {stage.episode for stage in stages}
            if len(episode_ids) != 1:
                raise ValueError("each stage sequence must contain one episode")
            if any(stage.terminal for stage in stages[:-1]):
                raise ValueError("no stage may follow a terminal stage")
            if any(stage.censored for stage in stages[:-1]):
                raise ValueError("no stage may follow a censored stage")
            self.stage_types.update(stage.key for stage in stages)
            for stage in stages:
                if state_dim is None:
                    state_dim = int(stage.entry_state.size)
                    feedback_dim = int(stage.feedback.size)
                elif (stage.entry_state.size != state_dim
                      or stage.feedback.size != feedback_dim):
                    raise ValueError(
                        "stage state and feedback dimensions must be consistent"
                    )

            keys = [stage.key for stage in stages]
            for index, source in enumerate(stages):
                is_last = index == len(stages) - 1
                terminal = bool(source.terminal)
                censored = bool(
                    source.censored or (is_last and not terminal)
                )
                if terminal or censored:
                    destination = None
                elif is_last:
                    destination = None
                    censored = True
                else:
                    destination = stages[index + 1].key

                history = keys[:index + 1]
                max_depth = len(history)
                if self.max_context_depth is not None:
                    max_depth = min(max_depth, self.max_context_depth)
                action_trace = source.action_trace
                self.action_options.add(action_trace)
                destination_stage = (
                    stages[index + 1] if destination is not None else None
                )
                outcome_key = (destination, terminal, censored)
                for depth in range(max_depth, -1, -1):
                    context = (
                        tuple(history[-depth:]) if depth else ()
                    )
                    self.contexts.add(context)
                    edge_key = (context, action_trace)
                    outcomes = self._counts.setdefault(edge_key, {})
                    counts = outcomes.setdefault(
                        outcome_key, _OutcomeCounts(),
                    )
                    counts.add(source, destination_stage)
                    pooled_counts = self._pooled.setdefault(
                        context, {}
                    ).setdefault(outcome_key, _OutcomeCounts())
                    pooled_counts.add(source, destination_stage)
                self.realized_outcomes += 1
        return self

    def predict(self, stage_history: Sequence[StageKey],
                action_trace: ActionTrace):
        """Predict observed sibling outcomes with support-aware backoff.

        ``stage_history`` is chronological and must end with the current stage
        key. ``action_trace`` is the exact run-length encoding of a proposed
        option; unsupported traces do not inherit another option's outcomes.
        """
        history = tuple(tuple(_native(token) for token in key)
                        for key in stage_history)
        trace = tuple(
            (_native(action), int(count))
            for action, count in action_trace
        )
        if any(count <= 0 for _action, count in trace):
            raise ValueError("action trace run lengths must be positive")
        path = stage_context_path(history, self.max_context_depth)
        for context in path:
            depth = len(context)
            outcomes = self._counts.get((context, trace))
            if not outcomes:
                continue
            total = sum(item.count for item in outcomes.values())
            episode_support = set()
            for item in outcomes.values():
                episode_support.update(item.episodes)
            if (total < self.min_support
                    or len(episode_support) < self.min_episodes):
                continue
            summaries = [
                item.summary(destination, terminal, censored, total)
                for (destination, terminal, censored), item
                in outcomes.items()
            ]
            summaries.sort(key=lambda row: repr((
                row["destination"], row["terminal"], row["censored"],
            )))
            return dict(
                status="supported",
                context=context,
                context_depth=int(depth),
                resolution=int(depth),
                support=int(total),
                episode_support=int(len(episode_support)),
                outcome_diversity=int(len(summaries)),
                terminal_rate=float(sum(
                    row["probability"] for row in summaries
                    if row["terminal"]
                )),
                censored_rate=float(sum(
                    row["probability"] for row in summaries
                    if row["censored"]
                )),
                siblings=tuple(
                    row["destination"] for row in summaries
                    if row["destination"] is not None
                ),
                outcomes=tuple(summaries),
                option_pooled=False,
            )
        return dict(
            status=UNKNOWN,
            context=(),
            context_depth=None,
            resolution=None,
            support=0,
            episode_support=0,
            outcome_diversity=0,
            terminal_rate=None,
            censored_rate=None,
            siblings=(),
            outcomes=(),
            option_pooled=False,
        )

    def predict_pooled(self, stage_history: Sequence[StageKey]):
        """Predict the next stage over all realized action options.

        At a decision where the future option is not known yet, the exact
        trace cannot be queried.  This aggregates only transitions that
        actually occurred under the same stage context, and applies the same
        support and episode gates as :meth:`predict`.  It is still an
        observational measure: no counterfactual option is added.
        """
        history = tuple(tuple(_native(token) for token in key)
                        for key in stage_history)
        path = stage_context_path(history, self.max_context_depth)
        for context in path:
            outcomes = self._pooled.get(context)
            if not outcomes:
                continue
            total = sum(item.count for item in outcomes.values())
            episode_support = set()
            for item in outcomes.values():
                episode_support.update(item.episodes)
            if (total < self.min_support
                    or len(episode_support) < self.min_episodes):
                continue
            summaries = [
                item.summary(destination, terminal, censored, total)
                for (destination, terminal, censored), item
                in outcomes.items()
            ]
            summaries.sort(key=lambda row: repr((
                row["destination"], row["terminal"], row["censored"],
            )))
            return dict(
                status="supported",
                context=context,
                context_depth=int(len(context)),
                resolution=int(len(context)),
                support=int(total),
                episode_support=int(len(episode_support)),
                outcome_diversity=int(len(summaries)),
                terminal_rate=float(sum(
                    row["probability"] for row in summaries
                    if row["terminal"]
                )),
                censored_rate=float(sum(
                    row["probability"] for row in summaries
                    if row["censored"]
                )),
                siblings=tuple(
                    row["destination"] for row in summaries
                    if row["destination"] is not None
                ),
                outcomes=tuple(summaries),
                option_pooled=True,
            )
        return dict(
            status=UNKNOWN,
            context=(),
            context_depth=None,
            resolution=None,
            support=0,
            episode_support=0,
            outcome_diversity=0,
            terminal_rate=None,
            censored_rate=None,
            siblings=(),
            outcomes=(),
            option_pooled=True,
        )

    def supported_options(self, stage_history: Sequence[StageKey],
                          min_context_depth: int = 1):
        """Return every realized option this history can condition on.

        The order is deterministic.  An option is included only when
        :meth:`predict` returns a supported edge for it whose chosen backoff
        context still contains the current stage (``min_context_depth``).
        The root context pools every source stage, so it is excluded by
        default: it is a marginal, not a transition from this stage.
        """
        if int(min_context_depth) < 0:
            raise ValueError("min_context_depth must be non-negative")
        options = sorted(self.action_options, key=repr)
        result = []
        for option in options:
            prediction = self.predict(stage_history, option)
            if prediction["status"] != "supported":
                continue
            if int(prediction.get("context_depth") or 0) < int(
                    min_context_depth):
                continue
            result.append(option)
        return tuple(result)

    def snapshot(self):
        return dict(
            stage_types=len(self.stage_types),
            contexts=len(self.contexts),
            action_options=len(self.action_options),
            transition_rows=int(self.realized_outcomes),
            context_depth=max(
                (len(context) for context in self.contexts), default=0
            ),
            supported_edges=len(self._counts),
            pooled_contexts=len(self._pooled),
        )
