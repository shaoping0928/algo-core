"""Fit one sparse linear transition rule per target dimension.

The learner sees anonymous observation and action-history slots; it never
sees a coordinate name.  Forward selection with a BIC gate chooses the input
scope, then the final map is fit on the raw values so a rule can be applied
directly to a source graph.

The important default is one rule per target, not one rule per target per
context.  The scale probe shows that context splits add no held-out accuracy
on this stream while multiplying the per-target fan-out by the number of
contexts.  A caller that has evidence for a split can fit the extra rules and
give them their own applicability/group.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from query_graph import Rule, Slot, make_event_rule, make_linear_rule


def standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (matrix - mean) / scale


def design(matrix: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    if not columns:
        return np.ones((len(matrix), 1), dtype=float)
    return np.column_stack([
        np.ones(len(matrix), dtype=float),
        np.asarray(matrix, dtype=float)[:, list(columns)],
    ])


def ols(matrix: np.ndarray, target: np.ndarray,
        ridge: float = 1e-8) -> np.ndarray:
    gram = matrix.T @ matrix + float(ridge) * np.eye(matrix.shape[1])
    return np.linalg.solve(gram, matrix.T @ target)


def subset_fit(matrix: np.ndarray, target: np.ndarray,
               columns: Sequence[int]) -> Tuple[np.ndarray, float]:
    weights = ols(design(matrix, columns), target)
    prediction = design(matrix, columns) @ weights
    residual = target - prediction
    return weights, float(residual @ residual)


def forward_select(matrix: np.ndarray, target: np.ndarray,
                   max_terms: int = 6,
                   bic_gain: float = 1.0) -> Tuple[Tuple[int, ...], np.ndarray, float]:
    """Greedy BIC forward selection; returns columns, weights, R2."""
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    n_rows, n_columns = matrix.shape
    selected: List[int] = []
    remaining = list(range(n_columns))
    weights, sse = subset_fit(matrix, target, selected)
    bic = n_rows * np.log(sse / n_rows + 1e-12)
    best = (bic, tuple(selected), weights, sse)
    while remaining and len(selected) < int(max_terms):
        candidate = None
        for column in remaining:
            columns = selected + [column]
            weights, sse = subset_fit(matrix, target, columns)
            score = (
                n_rows * np.log(sse / n_rows + 1e-12)
                + len(columns) * np.log(n_rows)
            )
            if candidate is None or score < candidate[0]:
                candidate = (score, tuple(columns), weights, sse)
        if candidate is None or candidate[0] >= best[0] - float(bic_gain):
            break
        best = candidate
        selected = list(candidate[1])
        remaining.remove(selected[-1])
    bic, selected, weights, sse = best
    variance = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - sse / (variance + 1e-12) if variance > 0.0 else 0.0
    return selected, weights, float(r2)


def heldout_fit(matrix: np.ndarray, target: np.ndarray,
                max_terms: int = 6, seed: int = 0,
                fraction: float = 0.8) -> Tuple[
                    Tuple[int, ...], np.ndarray, float, float]:
    """Select and fit on a train split; report raw held-out R2."""
    n_rows = len(matrix)
    if n_rows < 8:
        selected, weights, r2 = forward_select(matrix, target, max_terms)
        return selected, weights, r2, float("nan")
    order = np.random.RandomState(seed).permutation(n_rows)
    cut = max(4, int(round(float(fraction) * n_rows)))
    train, test = order[:cut], order[cut:]
    selected, _, r2_train = forward_select(
        standardize(matrix[train]), target[train], max_terms=max_terms
    )
    weights, _ = subset_fit(matrix[train], target[train], selected)
    if len(test) < 3:
        return selected, weights, float(r2_train), float("nan")
    prediction = design(matrix[test], selected) @ weights
    residual = target[test] - prediction
    base = target[test] - target[train].mean()
    r2 = 1.0 - float(residual @ residual) / (
        float(base @ base) + 1e-12
    )
    return selected, weights, float(r2_train), float(r2)


@dataclass
class SparseScope:
    target: int
    inputs: Tuple[int, ...]
    weights: np.ndarray
    r2_train: float
    r2_heldout: float
    support: int


def fit_sparse_scopes(
    observations: np.ndarray,
    histories: np.ndarray,
    next_observations: np.ndarray,
    max_terms: int = 6,
) -> List[SparseScope]:
    """One sparse rule per target dimension, fit on raw values."""
    observations = np.asarray(observations, dtype=float)
    histories = np.asarray(histories, dtype=float)
    next_observations = np.asarray(next_observations, dtype=float)
    inputs = np.concatenate([observations, histories], axis=1)
    scopes = []
    for target in range(next_observations.shape[1]):
        selected, weights, r2_train, r2_heldout = heldout_fit(
            inputs, next_observations[:, target],
            max_terms=max_terms, seed=target,
        )
        scopes.append(SparseScope(
            target=target,
            inputs=tuple(int(value) for value in selected),
            weights=np.asarray(weights, dtype=float),
            r2_train=float(r2_train),
            r2_heldout=float(r2_heldout),
            support=int(len(inputs)),
        ))
    return scopes


def slot_names(n_state: int, n_action: int) -> Tuple[str, ...]:
    return tuple(
        ["x%d" % index for index in range(n_state)]
        + ["a%d" % index for index in range(n_action)]
    )


def make_rules(scopes: Iterable[SparseScope], n_state: int,
               n_action: int, prefix: str = "rule") -> List[Rule]:
    """Convert fitted scopes into query-graph rules."""
    names = slot_names(n_state, n_action)
    rules = []
    for scope in scopes:
        inputs = tuple(Slot(names[index], 0) for index in scope.inputs)
        target = Slot("y%d" % scope.target, 1)
        rules.append(make_linear_rule(
            name="%s-y%d" % (prefix, scope.target),
            target_slot=target,
            input_slots=inputs,
            weights=scope.weights,
            confidence=float(scope.r2_heldout)
            if np.isfinite(scope.r2_heldout) else float(scope.r2_train),
            support=int(scope.support),
        ))
    return rules


def fit_event_rule(
    observations: np.ndarray,
    labels: Sequence[int],
    target_slot: Slot,
    thresholds: int = 16,
    min_support: int = 10,
) -> Optional[Tuple[Rule, dict]]:
    """Fit a one-slot threshold predicate with balanced accuracy.

    The event is a 0/1 world signal; the rule reads one anonymous state slot
    and writes a discrete flag.  Both directions are searched, so a low-side
    event (e.g. leaving a safe band) is not excluded.  Returns ``None`` when
    no threshold beats chance or support is too small.
    """
    observations = np.asarray(observations, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(observations) != len(labels) or len(labels) < int(min_support):
        return None
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    quantiles = np.linspace(0.05, 0.95, max(int(thresholds), 2))
    best = None
    for column in range(observations.shape[1]):
        values = observations[:, column]
        for threshold in np.unique(np.quantile(values, quantiles)):
            for direction in (1, -1):
                if direction >= 0:
                    prediction = values > threshold
                else:
                    prediction = values < threshold
                true_positive = int(np.sum(prediction & (labels == 1)))
                false_negative = positives - true_positive
                true_negative = int(np.sum((~prediction) & (labels == 0)))
                false_positive = negatives - true_negative
                tpr = true_positive / float(
                    true_positive + false_negative + 1e-12
                )
                tnr = true_negative / float(
                    true_negative + false_positive + 1e-12
                )
                score = 0.5 * (tpr + tnr)
                if best is None or score > best[0]:
                    best = (
                        score, column, float(threshold), int(direction),
                        true_positive, false_positive,
                        false_negative, true_negative,
                    )
    if best is None or best[0] <= 0.5:
        return None
    score, column, threshold, direction, tp, fp, fn, tn = best
    rule = make_event_rule(
        name="event-x%d" % column,
        target_slot=target_slot,
        input_slot=Slot("x%d" % column, 0),
        threshold=threshold,
        confidence=float(score),
        support=int(len(labels)),
        direction=int(direction),
    )
    return rule, dict(
        column=int(column), threshold=float(threshold),
        direction=int(direction), balanced_accuracy=float(score),
        true_positive=int(tp), false_positive=int(fp),
        false_negative=int(fn), true_negative=int(tn),
    )
