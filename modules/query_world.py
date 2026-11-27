"""A world-model adapter backed by the query-graph rule bank.

This is an opt-in prediction path, not a planner change.  It exposes the same
``predict(observations, histories)`` signature as the linear observation
model, so a probe can compare the two on identical rows and rollouts.  A
missing target becomes NaN: the adapter never silently substitutes a source
value or a coarser rule.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from query_graph import UNKNOWN, Query, RuleBank, Slot
from sparse_rules import fit_sparse_scopes, make_rules


@dataclass
class QueryWorldModel:
    bank: RuleBank
    n_state: int
    n_action: int
    n_targets: int = 0
    target_names: Tuple[str, ...] = ()
    budget: int = 200

    def __post_init__(self) -> None:
        if not self.target_names:
            count = int(self.n_targets) if int(self.n_targets) > 0 else int(
                self.n_state
            )
            self.target_names = tuple(
                "y%d" % index for index in range(count)
            )

    def source(self, observation: Sequence[float],
               history: Sequence[float]):
        state = {
            Slot("x%d" % index, 0): float(value)
            for index, value in enumerate(observation)
        }
        state.update({
            Slot("a%d" % index, 0): float(value)
            for index, value in enumerate(history)
        })
        return state

    def predict_one(self, observation: Sequence[float],
                    history: Sequence[float]) -> np.ndarray:
        targets = tuple(
            Slot(name, 1) for name in self.target_names
        )
        result = self.bank.query(Query(
            target_slots=targets,
            source=self.source(observation, history),
            budget=int(self.budget),
        ))
        values = np.full(len(targets), np.nan, dtype=float)
        for index, target in enumerate(targets):
            value = result.values.get(target)
            if value is not UNKNOWN:
                values[index] = float(value)
        return values

    def predict(self, observations, histories) -> np.ndarray:
        observations = np.asarray(observations, dtype=float)
        histories = np.asarray(histories, dtype=float)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if histories.ndim == 1:
            histories = histories.reshape(1, -1)
        if len(observations) != len(histories):
            raise ValueError("observations and histories must align")
        predictions = np.empty(
            (len(observations), len(self.target_names)), dtype=float
        )
        for index in range(len(observations)):
            predictions[index] = self.predict_one(
                observations[index], histories[index]
            )
        return predictions


def fit_query_world(
    observations: np.ndarray,
    histories: np.ndarray,
    next_observations: np.ndarray,
    max_terms: int = 6,
    budget: int = 200,
) -> QueryWorldModel:
    observations = np.asarray(observations, dtype=float)
    histories = np.asarray(histories, dtype=float)
    next_observations = np.asarray(next_observations, dtype=float)
    scopes = fit_sparse_scopes(
        observations, histories, next_observations, max_terms=max_terms,
    )
    rules = make_rules(
        scopes, observations.shape[1], histories.shape[1],
    )
    return QueryWorldModel(
        bank=RuleBank(rules),
        n_state=observations.shape[1],
        n_action=histories.shape[1],
        n_targets=next_observations.shape[1],
        budget=int(budget),
    )
