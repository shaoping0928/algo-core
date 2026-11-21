"""Task-neutral action-effect statistics over feature dimensions.

A death rule says which relations matter, but not which action changes them.
This library answers the second question from observed transitions:

    action -> mean signed / absolute change in each feature dimension

It is deliberately local and empirical: no task vocabulary, no fixed action
semantics.  The inquiry loop can use ``preferred`` to pick an action that
changes the dimensions a rule inspects when every candidate plan is unknown.
The library does not claim that an action's effect transfers across worlds;
that is a separate, data-dependent question.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ActionEffect:
    action: int
    count: int
    mean_delta: np.ndarray
    mean_abs_delta: np.ndarray
    scale: np.ndarray

    def change(self, dims: Optional[Sequence[int]] = None) -> float:
        """Mean absolute change, optionally restricted to dimensions."""
        values = self.mean_abs_delta
        if dims is not None:
            values = values[np.asarray(list(dims), dtype=int)]
        return float(np.mean(values)) if len(values) else 0.0

    def normalized_change(
        self, dims: Optional[Sequence[int]] = None
    ) -> float:
        """Mean absolute change relative to the observed dimension scale."""
        values = self.mean_abs_delta
        scale = self.scale
        if dims is not None:
            index = np.asarray(list(dims), dtype=int)
            values = values[index]
            scale = scale[index]
        safe = np.where(scale > 1e-12, scale, 1.0)
        return float(np.mean(values / safe)) if len(values) else 0.0


class ActionEffectLibrary:
    """Empirical action effects from (state, action, next_state) rows."""

    def __init__(self):
        self.effects: Dict[int, ActionEffect] = {}

    def fit(self, states: np.ndarray, actions: np.ndarray,
            next_states: np.ndarray,
            scales: Optional[np.ndarray] = None) -> "ActionEffectLibrary":
        states = np.asarray(states, dtype=float)
        next_states = np.asarray(next_states, dtype=float)
        actions = np.asarray(actions).reshape(-1)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if next_states.ndim == 1:
            next_states = next_states.reshape(1, -1)
        if states.shape != next_states.shape:
            raise ValueError("states/next_states shape mismatch")
        if len(states) != len(actions):
            raise ValueError("states/actions shape mismatch")
        if len(states) == 0:
            raise ValueError("at least one transition is required")
        if scales is None:
            scale = np.std(states, axis=0)
        else:
            scale = np.asarray(scales, dtype=float).reshape(-1)
            if scale.shape != (states.shape[1],):
                raise ValueError("scales must have one value per dimension")
        scale = np.where(scale > 1e-12, scale, 1.0)
        deltas = next_states - states
        self.effects = {}
        for action in sorted(set(int(value) for value in actions)):
            mask = actions.astype(int) == action
            selected = deltas[mask]
            self.effects[action] = ActionEffect(
                action=action,
                count=int(mask.sum()),
                mean_delta=np.mean(selected, axis=0),
                mean_abs_delta=np.mean(np.abs(selected), axis=0),
                scale=scale.copy(),
            )
        return self

    def __contains__(self, action: int) -> bool:
        return int(action) in self.effects

    def effect(self, action: int) -> ActionEffect:
        action = int(action)
        if action not in self.effects:
            raise KeyError("no effect statistics for action %d" % action)
        return self.effects[action]

    def change(self, action: int,
               dims: Optional[Sequence[int]] = None) -> float:
        return self.effect(action).change(dims)

    def normalized_change(
        self, action: int, dims: Optional[Sequence[int]] = None
    ) -> float:
        return self.effect(action).normalized_change(dims)

    def _selected(
        self, actions: Optional[Iterable[int]]
    ) -> Tuple[int, ...]:
        if actions is None:
            return tuple(sorted(self.effects))
        selected = tuple(
            int(action) for action in actions
            if int(action) in self.effects
        )
        if not selected:
            raise KeyError("none of the requested actions has statistics")
        return selected

    def preferred(self, dims: Optional[Sequence[int]] = None,
                  normalized: bool = True,
                  actions: Optional[Iterable[int]] = None) -> int:
        """Action with the largest effect on the requested dimensions."""
        if not self.effects:
            raise RuntimeError("fit must be called before preferred")
        scores = {}
        for action in self._selected(actions):
            effect = self.effects[action]
            scores[action] = (
                effect.normalized_change(dims) if normalized
                else effect.change(dims)
            )
        return int(min(scores, key=lambda key: (-scores[key], key)))

    def ranked(self, dims: Optional[Sequence[int]] = None,
               normalized: bool = True,
               actions: Optional[Iterable[int]] = None) -> Tuple[int, ...]:
        """Actions ordered by descending effect, ties by action id."""
        if not self.effects:
            raise RuntimeError("fit must be called before ranked")
        scored = [
            (
                -(
                    self.effects[action].normalized_change(dims)
                    if normalized
                    else self.effects[action].change(dims)
                ),
                action,
            )
            for action in self._selected(actions)
        ]
        return tuple(action for _score, action in sorted(scored))

    def describe(self) -> str:
        lines = []
        for action in sorted(self.effects):
            effect = self.effects[action]
            lines.append(
                "action %d n=%d abs=%s delta=%s" % (
                    action, effect.count,
                    np.round(effect.mean_abs_delta, 4).tolist(),
                    np.round(effect.mean_delta, 4).tolist(),
                )
            )
        return "\n".join(lines)
