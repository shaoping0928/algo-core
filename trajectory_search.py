"""Task-agnostic exhaustive receding-horizon search over action sequences.

The search core knows only:
  * an initial numeric state,
  * a set of candidate actions,
  * a batched one-step transition predictor, and
  * a batched evaluator that returns heuristic scores and feasibility flags.

Task-specific state encodings, learned models, goals, and constraints are supplied by adapters.
"""
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    actions: np.ndarray
    score: Union[float, Tuple[float, ...]]
    feasible: bool
    feasible_count: int
    candidate_count: int
    first_state: np.ndarray
    final_state: np.ndarray


def _sequence_batches(
    action_space: Sequence,
    horizon: int,
    batch_size: int,
) -> Iterator[np.ndarray]:
    batch = []
    for sequence in product(action_space, repeat=horizon):
        batch.append(sequence)
        if len(batch) == batch_size:
            yield np.asarray(batch, dtype=float)
            batch = []
    if batch:
        yield np.asarray(batch, dtype=float)


def _score_matrix(values, count, source):
    scores = np.asarray(values, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or scores.shape[0] != count:
        raise ValueError("%s must return one score row per candidate" % source)
    return scores


def _lexicographic_argmin(scores, mask):
    remaining = np.flatnonzero(mask)
    if not len(remaining):
        raise ValueError("cannot select from an empty candidate set")
    for column in range(scores.shape[1]):
        best = np.min(scores[remaining, column])
        remaining = remaining[scores[remaining, column] == best]
        if len(remaining) == 1:
            break
    return int(remaining[0])


def _lexicographic_order(scores, indices):
    """Order indices by column 0 first, then column 1, ...; lower is better."""
    subset = scores[indices]
    keys = tuple(subset[:, column] for column in reversed(range(subset.shape[1])))
    return np.lexsort(keys)


def _combine_modes(combine, columns):
    if isinstance(combine, str):
        modes = (combine,) * columns
    else:
        modes = tuple(combine)
        if len(modes) != columns:
            raise ValueError("score_combine must have one mode per objective")
    if any(mode not in ("sum", "max") for mode in modes):
        raise ValueError("score_combine modes must be 'sum' or 'max'")
    return modes


def _combine_scores(current, increment, modes):
    if current is None:
        return increment.copy()
    if current.shape != increment.shape:
        raise ValueError("score objective count changed during the rollout")
    for column, mode in enumerate(modes):
        if mode == "max":
            current[:, column] = np.maximum(current[:, column], increment[:, column])
        else:
            current[:, column] += increment[:, column]
    return current


def exhaustive_search(
    initial_state: np.ndarray,
    action_space: Iterable,
    horizon: int,
    transition_batch: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    evaluate_step: Callable[
        [np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]
    ],
    batch_size: int = 4096,
    evaluate_terminal: Optional[Callable[
        [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ]] = None,
    score_combine: Union[str, Sequence[str]] = "sum",
) -> SearchResult:
    """Enumerate action sequences and select plans using supplied model/evaluator callbacks.

    `transition_batch(states, sequences, step)` predicts one step from batched states. The
    complete candidate sequence is available so models with delayed actions can read its prefix.
    `evaluate_step(states, sequences, step)` returns (score_increment, feasible). Scores may be
    scalar or multi-objective; vector scores are minimized lexicographically. An optional
    `evaluate_terminal(final_states, sequences)` adds a terminal score and feasibility check.
    Lower scores are better. Only current, first, and final states are retained; the core does
    not store every candidate's full trajectory.

    If no candidate is feasible, the lowest-score candidate is returned with `feasible=False`,
    allowing the task adapter to detect and handle that case.
    """
    state0 = np.asarray(initial_state, dtype=float)
    if state0.ndim == 0:
        state0 = state0.reshape(1)
    actions = tuple(action_space)
    if not actions:
        raise ValueError("action_space must not be empty")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    best_feasible = None
    best_any = None
    feasible_count = 0
    candidate_count = 0

    for sequences in _sequence_batches(actions, horizon, batch_size):
        count = len(sequences)
        states = np.broadcast_to(state0, (count,) + state0.shape).copy()
        scores = None
        feasible = np.ones(count, dtype=bool)
        first_states = None

        for step in range(horizon):
            states = np.asarray(
                transition_batch(states, sequences, step), dtype=float
            )
            if states.shape != (count,) + state0.shape:
                raise ValueError(
                    "transition_batch must preserve the batched state shape; "
                    "got %r, expected %r" % (states.shape, (count,) + state0.shape)
                )
            if step == 0:
                first_states = states.copy()
            increments, step_feasible = evaluate_step(
                states, sequences, step
            )
            increments = _score_matrix(increments, count, "evaluate_step")
            step_feasible = np.asarray(step_feasible, dtype=bool).reshape(-1)
            if step_feasible.shape != (count,):
                raise ValueError(
                    "evaluate_step must return one feasibility flag per plan"
                )
            if scores is None:
                modes = _combine_modes(score_combine, increments.shape[1])
                scores = increments.copy()
            else:
                scores = _combine_scores(scores, increments, modes)
            feasible &= step_feasible

        if evaluate_terminal is not None:
            increments, terminal_feasible = evaluate_terminal(states, sequences)
            increments = _score_matrix(increments, count, "evaluate_terminal")
            terminal_feasible = np.asarray(terminal_feasible, dtype=bool).reshape(-1)
            if terminal_feasible.shape != (count,):
                raise ValueError(
                    "evaluate_terminal must return one feasibility flag per plan"
                )
            if scores is None:
                modes = _combine_modes(score_combine, increments.shape[1])
                scores = increments.copy()
            else:
                scores = _combine_scores(scores, increments, modes)
            feasible &= terminal_feasible

        if scores is None:
            raise ValueError("at least one evaluator must provide a score")
        finite = np.isfinite(scores).all(axis=1)
        if not finite.any():
            candidate_count += count
            continue

        feasible &= finite
        feasible_count += int(feasible.sum())
        candidate_count += count

        any_index = _lexicographic_argmin(scores, finite)
        any_candidate = (
            tuple(float(value) for value in scores[any_index]),
            sequences[any_index].copy(),
            first_states[any_index].copy(),
            states[any_index].copy(),
        )
        if best_any is None or any_candidate[0] < best_any[0]:
            best_any = any_candidate

        if feasible.any():
            feasible_index = _lexicographic_argmin(scores, feasible)
            feasible_candidate = (
                tuple(float(value) for value in scores[feasible_index]),
                sequences[feasible_index].copy(),
                first_states[feasible_index].copy(),
                states[feasible_index].copy(),
            )
            if best_feasible is None or feasible_candidate[0] < best_feasible[0]:
                best_feasible = feasible_candidate

    selected = best_feasible if best_feasible is not None else best_any
    if selected is None:
        raise ValueError("evaluators returned no finite score for any candidate")
    score, sequence, first_state, final_state = selected
    score = score[0] if len(score) == 1 else score
    return SearchResult(
        actions=sequence,
        score=score,
        feasible=best_feasible is not None,
        feasible_count=feasible_count,
        candidate_count=candidate_count,
        first_state=first_state,
        final_state=final_state,
    )


def beam_search(
    initial_state: np.ndarray,
    action_space: Iterable,
    horizon: int,
    transition_batch: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    evaluate_step: Callable[
        [np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]
    ],
    evaluate_terminal: Optional[Callable[
        [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ]] = None,
    beam_width: int = 256,
    score_combine: Union[str, Sequence[str]] = "sum",
) -> SearchResult:
    """Beam-limited version of exhaustive_search for long horizons.

    At each depth it expands every live beam with every action, keeps the best
    `beam_width` candidates by the same lexicographic objective, and then applies the
    optional terminal evaluator. It is a search approximation, not a task-specific policy.
    """
    state0 = np.asarray(initial_state, dtype=float)
    if state0.ndim == 0:
        state0 = state0.reshape(1)
    actions = tuple(action_space)
    if not actions:
        raise ValueError("action_space must not be empty")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if beam_width < 1:
        raise ValueError("beam_width must be a positive integer")

    states = state0.reshape(1, -1)
    sequences = np.zeros((1, horizon), dtype=float)
    scores = None
    feasible = np.ones(1, dtype=bool)
    first_states = None
    candidate_count = 0
    feasible_checks = 0
    width = len(actions)

    for step in range(horizon):
        count = len(states)
        next_states = np.repeat(states, width, axis=0)
        next_sequences = np.repeat(sequences, width, axis=0)
        next_sequences[:, step] = np.tile(
            np.asarray(actions, dtype=float), count
        )
        next_states = np.asarray(
            transition_batch(next_states, next_sequences, step), dtype=float
        )
        if next_states.shape != (count * width,) + state0.shape:
            raise ValueError(
                "transition_batch must preserve the batched state shape; got %r"
                % (next_states.shape,)
            )
        increments, step_feasible = evaluate_step(
            next_states, next_sequences, step
        )
        increments = _score_matrix(increments, count * width, "evaluate_step")
        step_feasible = np.asarray(step_feasible, dtype=bool).reshape(-1)
        if step_feasible.shape != (count * width,):
            raise ValueError(
                "evaluate_step must return one feasibility flag per candidate"
            )

        candidate_first_states = (
            next_states.copy() if first_states is None
            else np.repeat(first_states, width, axis=0)
        )
        if scores is None:
            modes = _combine_modes(score_combine, increments.shape[1])
            candidate_scores = increments.copy()
        else:
            candidate_scores = _combine_scores(
                np.repeat(scores, width, axis=0), increments, modes
            )
        candidate_feasible = np.repeat(feasible, width) & step_feasible
        feasible_checks += int(candidate_feasible.sum())
        finite = np.isfinite(candidate_scores).all(axis=1)
        selectable = np.flatnonzero(finite & candidate_feasible)
        if not len(selectable):
            selectable = np.flatnonzero(finite)
        if not len(selectable):
            raise ValueError("evaluators returned no finite score for any candidate")

        order = _lexicographic_order(candidate_scores, selectable)
        keep = selectable[order[:beam_width]]
        states = next_states[keep]
        sequences = next_sequences[keep]
        scores = candidate_scores[keep]
        feasible = candidate_feasible[keep]
        first_states = candidate_first_states[keep]
        candidate_count += count * width

    if evaluate_terminal is not None:
        increments, terminal_feasible = evaluate_terminal(states, sequences)
        increments = _score_matrix(increments, len(states), "evaluate_terminal")
        terminal_feasible = np.asarray(terminal_feasible, dtype=bool).reshape(-1)
        scores = _combine_scores(scores, increments, modes)
        feasible = feasible & terminal_feasible

    finite = np.isfinite(scores).all(axis=1)
    if not finite.any():
        raise ValueError("evaluators returned no finite score for any candidate")
    feasible_indices = np.flatnonzero(feasible & finite)
    selectable = feasible_indices if len(feasible_indices) else np.flatnonzero(finite)
    order = _lexicographic_order(scores, selectable)
    index = int(selectable[order[0]])
    score = tuple(float(value) for value in scores[index])
    score = score[0] if len(score) == 1 else score
    return SearchResult(
        actions=sequences[index].copy(),
        score=score,
        feasible=bool(len(feasible_indices)),
        feasible_count=feasible_checks,
        candidate_count=candidate_count,
        first_state=first_states[index].copy(),
        final_state=states[index].copy(),
    )
