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

# Status codes for `evaluate_death_status`.  Death is a hard veto, near is
# a soft risk that may be penalised, unknown is not safe but not a veto.
STATUS_DEATH = 1
STATUS_SAFE = 0
STATUS_UNKNOWN = -1
STATUS_NEAR = 2


@dataclass(frozen=True)
class SearchResult:
    actions: np.ndarray
    score: Union[float, Tuple[float, ...]]
    feasible: bool
    feasible_count: int
    candidate_count: int
    first_state: np.ndarray
    final_state: np.ndarray
    first_death: int = -1
    # Tri-state bookkeeping for the selected plan.  These stay at their
    # defaults when the caller only supplies the legacy death callback:
    # there is no way to distinguish "safe" from "unknown" without a
    # status callback, and inventing one would silently change meaning.
    first_unknown: int = -1
    unknown_count: int = 0
    safe_count: int = 0
    # Near-death is a soft status (2): inside the rule's boundary margin
    # but not an exact death.  It can be penalised without vetoing.
    first_near: int = -1
    near_count: int = 0


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
    evaluate_death: Optional[Callable[
        [np.ndarray, np.ndarray, int], np.ndarray
    ]] = None,
    evaluate_death_status: Optional[Callable[
        [np.ndarray, np.ndarray, int], np.ndarray
    ]] = None,
    unknown_penalty: float = 0.0,
    near_penalty: float = 0.0,
    candidate_callback: Optional[Callable[[dict], None]] = None,
    score_combine: Union[str, Sequence[str]] = "sum",
) -> SearchResult:
    """Enumerate action sequences and select plans using supplied model/evaluator callbacks.

    `transition_batch(states, sequences, step)` predicts one step from batched states. The
    complete candidate sequence is available so models with delayed actions can read its prefix.
    `evaluate_step(states, sequences, step)` returns (score_increment, feasible). Scores may be
    scalar or multi-objective; vector scores are minimized lexicographically. An optional
    `evaluate_terminal(final_states, sequences)` adds a terminal score and feasibility check.
    An optional `evaluate_death(states, sequences, step)` returns the death step for each
    candidate (-1 = alive). A death marks the plan infeasible; if no candidate survives the
    whole horizon, the candidate with the latest first death wins, then the score.
    An optional `evaluate_death_status(states, sequences, step)` returns one status per
    candidate and step: ``1`` = death, ``2`` = near death, ``0`` = safe, ``-1`` = unknown.
    It is mutually exclusive with `evaluate_death`; the richer callback supplies the death
    step as well as the first-unknown/first-near step and the number of unknown/safe/near
    steps before the first death.
    Unknown is never treated as safe: it does not veto a plan by itself, but
    `unknown_penalty` can add a cost per unknown step to the primary objective for
    selection (the returned score stays the unpenalised objective).  The default
    `unknown_penalty=0` keeps the old selection path.  A positive penalty is
    only meaningful when the status callback can return ``0`` for a known-safe
    step; with a death-only callback (``1``/``-1``) every non-death step is
    unknown, so penalising unknowns would prefer an early death.
    `near_penalty` is the soft-risk counterpart: it adds a cost per ``2``
    status without changing feasibility, so a near-death state can be avoided
    without being vetoed.
    An optional `candidate_callback(batch)` receives the per-batch sequences, scores,
    feasibility, death/unknown bookkeeping and predicted states; it is for probes and
    diagnostics and is never required by the search itself.
    Lower scores are better. Only current, first, and final states are retained; the core does
    not store every candidate's full trajectory.

    If no candidate is feasible and no death callback is supplied, the lowest-score candidate is
    returned with `feasible=False`, allowing the task adapter to detect and handle that case.
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
    if evaluate_death is not None and evaluate_death_status is not None:
        raise ValueError(
            "supply evaluate_death or evaluate_death_status, not both"
        )
    if float(unknown_penalty) < 0.0:
        raise ValueError("unknown_penalty must be non-negative")
    if float(unknown_penalty) > 0.0 and evaluate_death_status is None:
        raise ValueError(
            "unknown_penalty needs evaluate_death_status to count unknowns"
        )
    if float(near_penalty) < 0.0:
        raise ValueError("near_penalty must be non-negative")
    if float(near_penalty) > 0.0 and evaluate_death_status is None:
        raise ValueError(
            "near_penalty needs evaluate_death_status to count near states"
        )

    best_feasible = None
    best_any = None
    best_late = None
    feasible_count = 0
    candidate_count = 0

    for sequences in _sequence_batches(actions, horizon, batch_size):
        count = len(sequences)
        states = np.broadcast_to(state0, (count,) + state0.shape).copy()
        scores = None
        feasible = np.ones(count, dtype=bool)
        model_feasible = np.ones(count, dtype=bool)
        first_death = np.full(count, -1, dtype=int)
        first_unknown = np.full(count, -1, dtype=int)
        unknown_count = np.zeros(count, dtype=int)
        safe_count = np.zeros(count, dtype=int)
        first_near = np.full(count, -1, dtype=int)
        near_count = np.zeros(count, dtype=int)
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
            model_feasible &= step_feasible
            if evaluate_death_status is not None:
                status = np.asarray(
                    evaluate_death_status(states, sequences, step)
                ).reshape(-1)
                if status.shape != (count,):
                    raise ValueError(
                        "evaluate_death_status must return one status per plan"
                    )
                if not np.isin(status, (-1, 0, 1, 2)).all():
                    raise ValueError(
                        "evaluate_death_status values must be -1, 0, 1 or 2"
                    )
                status = status.astype(int)
                # Only the prefix before the first death is meaningful:
                # once a plan is dead, later states cannot make it safe.
                active = first_death < 0
                death_now = active & (status == 1)
                unknown_now = active & (status == -1)
                safe_now = active & (status == 0)
                near_now = active & (status == 2)
                first_death = np.where(death_now, step, first_death)
                first_unknown = np.where(
                    unknown_now & (first_unknown < 0), step, first_unknown
                )
                first_near = np.where(
                    near_now & (first_near < 0), step, first_near
                )
                unknown_count += unknown_now.astype(int)
                safe_count += safe_now.astype(int)
                near_count += near_now.astype(int)
                feasible &= ~death_now
            elif evaluate_death is not None:
                death = np.asarray(
                    evaluate_death(states, sequences, step)
                ).reshape(-1)
                if death.shape != (count,):
                    raise ValueError(
                        "evaluate_death must return one step per plan"
                    )
                death = death.astype(int)
                first_death = np.where(
                    (first_death < 0) & (death >= 0), death, first_death
                )
                feasible &= death < 0

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
            model_feasible &= terminal_feasible

        if scores is None:
            raise ValueError("at least one evaluator must provide a score")
        finite = np.isfinite(scores).all(axis=1)
        ranking_scores = scores
        if unknown_penalty > 0.0 or near_penalty > 0.0:
            ranking_scores = scores.copy()
            if unknown_penalty > 0.0:
                ranking_scores[:, 0] += (
                    float(unknown_penalty) * unknown_count
                )
            if near_penalty > 0.0:
                ranking_scores[:, 0] += (
                    float(near_penalty) * near_count
                )

        if candidate_callback is not None:
            candidate_callback(dict(
                sequences=sequences.copy(),
                scores=scores.copy(),
                ranking_scores=np.asarray(ranking_scores).copy(),
                feasible=feasible.copy(),
                model_feasible=model_feasible.copy(),
                finite=finite.copy(),
                first_death=first_death.copy(),
                first_unknown=first_unknown.copy(),
                unknown_count=unknown_count.copy(),
                safe_count=safe_count.copy(),
                first_near=first_near.copy(),
                near_count=near_count.copy(),
                first_states=first_states.copy(),
                final_states=states.copy(),
            ))

        if not finite.any():
            candidate_count += count
            continue

        feasible &= finite
        feasible_count += int(feasible.sum())
        candidate_count += count

        def candidate(index, death_value):
            # The layout keeps the legacy comparison keys at the front:
            # [0] first death, [4] selection score.  The original score at
            # [5] is what the caller sees; the ranking score may carry the
            # optional unknown or near penalty.
            return (
                int(death_value),
                int(first_unknown[index]),
                int(unknown_count[index]),
                int(safe_count[index]),
                tuple(float(value) for value in ranking_scores[index]),
                tuple(float(value) for value in scores[index]),
                sequences[index].copy(),
                first_states[index].copy(),
                states[index].copy(),
                int(first_near[index]),
                int(near_count[index]),
            )

        any_index = _lexicographic_argmin(ranking_scores, finite)
        any_candidate = candidate(any_index, first_death[any_index])
        if best_any is None or any_candidate[4] < best_any[4]:
            best_any = any_candidate

        if feasible.any():
            feasible_index = _lexicographic_argmin(ranking_scores, feasible)
            feasible_candidate = candidate(feasible_index, -1)
            if best_feasible is None or (
                feasible_candidate[4] < best_feasible[4]
            ):
                best_feasible = feasible_candidate

        if evaluate_death is not None or evaluate_death_status is not None:
            # When every candidate dies, keep the one whose first death is
            # latest; the ordinary objective is only the tie-break.
            late = finite & (first_death >= 0)
            if late.any():
                latest = int(first_death[late].max())
                late_indices = late & (first_death == latest)
                late_index = _lexicographic_argmin(
                    ranking_scores, late_indices
                )
                late_candidate = candidate(late_index, latest)
                if best_late is None or (
                    late_candidate[0] > best_late[0]
                    or (
                        late_candidate[0] == best_late[0]
                        and late_candidate[4] < best_late[4]
                    )
                ):
                    best_late = late_candidate

    selected = best_feasible
    if selected is None and best_late is not None:
        selected = best_late
    if selected is None:
        selected = best_any
    if selected is None:
        raise ValueError("evaluators returned no finite score for any candidate")
    (
        death,
        selected_first_unknown,
        selected_unknown_count,
        selected_safe_count,
        _rank,
        score,
        sequence,
        first_state,
        final_state,
        selected_first_near,
        selected_near_count,
    ) = selected
    score = score[0] if len(score) == 1 else score
    return SearchResult(
        actions=sequence,
        score=score,
        feasible=best_feasible is not None,
        feasible_count=feasible_count,
        candidate_count=candidate_count,
        first_state=first_state,
        final_state=final_state,
        first_death=int(death),
        first_unknown=int(selected_first_unknown),
        unknown_count=int(selected_unknown_count),
        safe_count=int(selected_safe_count),
        first_near=int(selected_first_near),
        near_count=int(selected_near_count),
    )


def best_candidate_index(scores, feasible) -> int:
    """Lexicographic best index among feasible candidates, else finite ones.

    This reproduces the search's baseline selection offline from a
    candidate batch: ``feasible`` is the model-feasibility mask *before*
    any death veto, so a caller can compare a death-aware search with the
    no-death baseline without running the search twice.
    """
    scores = np.asarray(scores, dtype=float)
    feasible = np.asarray(feasible, dtype=bool).reshape(-1)
    finite = np.isfinite(scores).all(axis=1)
    usable = feasible & finite
    if usable.any():
        return _lexicographic_argmin(scores, usable)
    return _lexicographic_argmin(scores, finite)


def merge_candidate_batches(batches) -> dict:
    """Concatenate the per-batch dictionaries from `candidate_callback`."""
    batches = list(batches)
    if not batches:
        raise ValueError("at least one candidate batch is required")
    keys = batches[0].keys()
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in keys
    }


def override_first_action(result: SearchResult, batches,
                          action: int) -> SearchResult:
    """Replace a search result with the best feasible plan starting with `action`.

    This is the inquiry hook: when the death side leaves every candidate
    unknown, the caller may prefer an action that changes the rule's
    relation dimensions.  The override only chooses among candidates the
    search already marked feasible; it never turns a predicted death into
    an accepted plan.  If the action has no feasible candidate, the
    original result is returned unchanged.
    """
    merged = merge_candidate_batches(batches)
    sequences = np.asarray(merged["sequences"], dtype=float)
    first = np.asarray(sequences[:, 0], dtype=int)
    mask = first == int(action)
    if not mask.any():
        return result
    feasible = np.asarray(merged["feasible"], dtype=bool).reshape(-1)
    finite = np.asarray(
        merged.get("finite", np.ones(len(mask), dtype=bool)), dtype=bool
    ).reshape(-1)
    usable = mask & feasible & finite
    if not usable.any():
        return result
    ranking = np.asarray(merged["ranking_scores"], dtype=float)
    index = _lexicographic_argmin(ranking, usable)
    scores = np.asarray(merged["scores"], dtype=float)
    score = tuple(float(value) for value in scores[index])
    return SearchResult(
        actions=sequences[index].copy(),
        score=score[0] if len(score) == 1 else score,
        feasible=True,
        feasible_count=result.feasible_count,
        candidate_count=result.candidate_count,
        first_state=np.asarray(merged["first_states"][index]).copy(),
        final_state=np.asarray(merged["final_states"][index]).copy(),
        first_death=int(merged["first_death"][index]),
        first_unknown=int(merged["first_unknown"][index]),
        unknown_count=int(merged["unknown_count"][index]),
        safe_count=int(merged["safe_count"][index]),
        first_near=int(merged["first_near"][index]),
        near_count=int(merged["near_count"][index]),
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
    rank_step: Optional[Callable[
        [np.ndarray, np.ndarray, int], np.ndarray
    ]] = None,
    rank_mode: str = "order",
    rank_margin: float = 0.0,
) -> SearchResult:
    """Beam-limited version of exhaustive_search for long horizons.

    At each depth it expands every live beam with every action, keeps the best
    `beam_width` candidates by the same lexicographic objective, and then applies the
    optional terminal evaluator. ``rank_step`` is an optional, higher-is-better
    ordering hint used only when pruning intermediate beams. Its increments do
    not change feasibility, accumulated objective scores, or final selection
    among the surviving candidates.  ``rank_mode="order"`` applies the hint
    before the objective score.  ``rank_mode="tie"`` applies it only inside the
    near-tie band ``score[0] <= best + rank_margin``; paths with a clearly
    better objective score keep their order, so the hint can break close calls
    without discarding a stronger plan.
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
    if rank_mode not in ("order", "tie"):
        raise ValueError("rank_mode must be order or tie")
    if float(rank_margin) < 0.0:
        raise ValueError("rank_margin must be non-negative")

    states = state0.reshape(1, -1)
    sequences = np.zeros((1, horizon), dtype=float)
    scores = None
    priorities = None
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

        if rank_step is None:
            # Keep the old path exactly when the opt-in hint is disabled.
            order = _lexicographic_order(candidate_scores, selectable)
            keep = selectable[order[:beam_width]]
            next_priorities = None
        else:
            increments_priority = np.asarray(
                rank_step(next_states, next_sequences, step), dtype=float,
            )
            if increments_priority.ndim == 2 and increments_priority.shape[1] == 1:
                increments_priority = increments_priority[:, 0]
            if increments_priority.shape != (count * width,):
                raise ValueError(
                    "rank_step must return one scalar per candidate"
                )
            if not np.isfinite(increments_priority).all():
                raise ValueError("rank_step priorities must be finite")
            next_priorities = increments_priority.copy()
            if priorities is not None:
                next_priorities += np.repeat(priorities, width)
            if rank_mode == "tie":
                # Priority only reorders candidates whose primary objective is
                # within a near-tie band of the best candidate.  A clearly
                # better-scoring path is never displaced by the hint.
                primary = np.asarray(candidate_scores[:, 0], dtype=float)
                best = float(primary[selectable].min())
                near = [
                    index for index in selectable
                    if float(primary[index]) <= best + float(rank_margin)
                ]
                near_set = set(near)
                far = [
                    index for index in selectable if index not in near_set
                ]
                ordered = sorted(
                    near,
                    key=lambda index: (
                        -float(next_priorities[index]),
                        tuple(float(value) for value in candidate_scores[index]),
                        int(index),
                    ),
                ) + sorted(
                    far,
                    key=lambda index: (
                        tuple(float(value) for value in candidate_scores[index]),
                        int(index),
                    ),
                )
                keep = np.asarray(ordered[:beam_width], dtype=int)
            else:
                keep = np.asarray(sorted(
                    selectable.tolist(),
                    key=lambda index: (
                        -float(next_priorities[index]),
                        tuple(float(value) for value in candidate_scores[index]),
                        int(index),
                    ),
                )[:beam_width], dtype=int)
        states = next_states[keep]
        sequences = next_sequences[keep]
        scores = candidate_scores[keep]
        feasible = candidate_feasible[keep]
        first_states = candidate_first_states[keep]
        priorities = (
            None if next_priorities is None else next_priorities[keep]
        )
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
