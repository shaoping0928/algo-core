"""Time-in-state features for the P0 representation experiment.

The state currently has no explicit age: the action history is a short window,
so the model cannot tell whether the last flap was two frames or forty frames
ago, and the periodic phase of the world is implicit in the coordinates.  This
module appends generic, task-agnostic age features:

* one age per action value: frames since that action was last issued, capped;
* a periodic phase (sin/cos) and its rate ``1 / period``, where the period is
  estimated from the observation stream by a per-episode spectrum.

Nothing here names a task or a coordinate.  The wrapper is opt-in: the default
environment is unchanged unless an ``env_factory`` is supplied.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def estimate_period(observations: np.ndarray, episodes: np.ndarray,
                    low: int = 8, high: int = 120) -> int:
    """Dominant per-episode period, median across episodes.

    Copied from the phase-stage probe's estimator: a periodogram per episode
    restricted to a plausible band, never a named clock.
    """
    observations = np.asarray(observations, dtype=float)
    episodes = np.asarray(episodes)
    periods = []
    for episode in np.unique(episodes):
        block = observations[episodes == episode]
        count = len(block)
        if count < 3 * int(low):
            continue
        z = (block - block.mean(axis=0)) / (block.std(axis=0) + 1e-9)
        spectrum = np.abs(np.fft.rfft(z, axis=0)) ** 2
        power = spectrum.mean(axis=1)
        frequencies = np.fft.rfftfreq(count)
        top = min(int(high), count // 3)
        mask = (
            (frequencies >= 1.0 / max(top, 1))
            & (frequencies <= 1.0 / max(int(low), 1))
        )
        if not mask.any():
            continue
        indices = np.flatnonzero(mask)
        best = indices[int(np.argmax(power[indices]))]
        periods.append(1.0 / float(frequencies[best]))
    if not periods:
        return 1
    return max(1, int(round(float(np.median(periods)))))


@dataclass
class TimeInStateEnv:
    """Wrap a task world and append action ages plus a phase clock."""

    base: object
    period: float = 1.0
    max_age: int = 64
    actions: Tuple[int, ...] = field(default_factory=tuple)
    action_age: Dict[int, int] = field(default_factory=dict)
    include_actions: bool = True
    include_phase: bool = True

    def __post_init__(self) -> None:
        self.period = max(float(self.period), 1.0)
        self.max_age = max(int(self.max_age), 1)
        if not self.actions:
            self.actions = tuple(int(value) for value in getattr(
                self.base, "action_space", (0, 1)
            ))
        if not self.actions:
            self.actions = (0, 1)
        self.action_age = {
            int(action): int(self.max_age) for action in self.actions
        }

    @property
    def feature_dim(self) -> int:
        return (
            (len(self.actions) if self.include_actions else 0)
            + (3 if self.include_phase else 0)
        )

    def features(self) -> np.ndarray:
        parts = []
        if self.include_actions:
            parts.append(np.asarray([
                min(int(self.action_age[action]), self.max_age)
                / float(self.max_age)
                for action in self.actions
            ], dtype=float))
        if self.include_phase:
            phase = (
                2.0 * np.pi * float(getattr(self.base, "t", 0)) / self.period
            )
            parts.append(np.asarray([
                float(np.sin(phase)),
                float(np.cos(phase)),
                1.0 / float(self.period),
            ]))
        if not parts:
            return np.zeros(0, dtype=float)
        return np.concatenate(parts)

    def obs(self) -> np.ndarray:
        return np.concatenate([
            np.asarray(self.base.obs(), dtype=float),
            self.features(),
        ])

    def step(self, action: int):
        for value in self.actions:
            self.action_age[value] = min(
                int(self.action_age[value]) + 1, self.max_age
            )
        action = int(action)
        if action in self.action_age:
            self.action_age[action] = 0
        _obs, feedback, dead = self.base.step(action)
        return self.obs(), feedback, dead

    def reset(self):
        self.base.reset()
        self.action_age = {
            int(action): int(self.max_age) for action in self.actions
        }
        return self.obs()

    def __getattr__(self, name):
        return getattr(self.base, name)


def make_env_factory(period: int, max_age: int = 64, lag: int = 3,
                     include_actions: bool = True,
                     include_phase: bool = True,
                     age_actions: Optional[Iterable[int]] = None):
    """An env factory for ``collect``/``evaluate`` with time features."""
    from latent_world import latent

    def factory(seed: int, **kwargs):
        return TimeInStateEnv(
            latent[int(lag)](seed, **kwargs),
            period=period,
            max_age=max_age,
            actions=tuple(age_actions) if age_actions is not None else (),
            include_actions=bool(include_actions),
            include_phase=bool(include_phase),
        )

    return factory


def augment_observations(observations: np.ndarray,
                         actions: Optional[np.ndarray],
                         episodes: np.ndarray,
                         period: int,
                         max_age: int = 64,
                         action_space: Sequence[int] = (0, 1),
                         include_actions: bool = True,
                         include_phase: bool = True):
    """Offline version: append the same features to a stream.

    ``actions[t]`` is the action issued at state ``t``; the next state's ages
    are updated the same way as ``TimeInStateEnv.step``.  This is for probes
    that already have rows and do not need an online wrapper.
    """
    observations = np.asarray(observations, dtype=float)
    episodes = np.asarray(episodes)
    if actions is None:
        actions = np.zeros(len(observations), dtype=int)
    actions = np.asarray(actions, dtype=int)
    actions_space = tuple(int(value) for value in action_space) or (0, 1)
    width = (
        (len(actions_space) if include_actions else 0)
        + (3 if include_phase else 0)
    )
    features = np.zeros((len(observations), width), dtype=float)
    ages = {int(action): int(max_age) for action in actions_space}
    period = max(int(period), 1)
    last_episode = None
    step = 0
    for index in range(len(observations)):
        if episodes[index] != last_episode:
            ages = {int(action): int(max_age) for action in actions_space}
            step = 0
            last_episode = episodes[index]
        parts = []
        if include_actions:
            parts.append(np.asarray([
                min(int(ages[action]), max_age) / float(max_age)
                for action in actions_space
            ], dtype=float))
        if include_phase:
            phase = 2.0 * np.pi * float(step) / float(period)
            parts.append(np.asarray([
                float(np.sin(phase)), float(np.cos(phase)),
                1.0 / float(period),
            ]))
        if parts:
            features[index] = np.concatenate(parts)
        action = int(actions[index])
        for value in actions_space:
            ages[value] = min(int(ages[value]) + 1, max_age)
        if action in ages:
            ages[action] = 0
        step += 1
    return np.concatenate([observations, features], axis=1)
