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


@dataclass
class ControlFeatureEnv:
    """A non-constant, task-free extra dimension for dimension controls."""

    base: object
    frequency: float = 0.7

    @property
    def feature_dim(self) -> int:
        return 1

    def features(self) -> np.ndarray:
        return np.asarray([
            float(np.sin(self.frequency * float(getattr(self.base, "t", 0))))
        ])

    def obs(self) -> np.ndarray:
        return np.concatenate([
            np.asarray(self.base.obs(), dtype=float), self.features(),
        ])

    def step(self, action: int):
        _obs, feedback, dead = self.base.step(int(action))
        return self.obs(), feedback, dead

    def reset(self):
        self.base.reset()
        return self.obs()

    def __getattr__(self, name):
        return getattr(self.base, name)


def make_control_env_factory(lag: int = 3, frequency: float = 0.7):
    """An env factory with one non-constant control dimension."""
    from latent_world import latent

    def factory(seed: int, **kwargs):
        return ControlFeatureEnv(
            latent[int(lag)](seed, **kwargs), frequency=frequency,
        )

    return factory


class ConditionedTimeWorld:
    """Keep the raw world as the predictor and add a time-feature residual.

    The full state is raw observation plus auxiliary time features.  The raw
    world predicts the next raw observation exactly as the baseline; a small
    residual model may correct it from the time features, and the auxiliary
    features are advanced by their known update rule.  The agent never sees
    the auxiliary dimensions, so this is the world-model-only test.
    """

    def __init__(self, base, residual, raw_dim: int, feature_dim: int,
                 max_age: int, age_actions: Tuple[int, ...],
                 residual_scale: float = 1.0,
                 validation_gain: float = 0.0):
        self.base = base
        self.residual = residual
        self.raw_dim = int(raw_dim)
        self.feature_dim = int(feature_dim)
        self.max_age = max(int(max_age), 1)
        self.age_actions = tuple(int(value) for value in age_actions)
        self.residual_scale = float(residual_scale)
        self.validation_gain = float(validation_gain)

    @property
    def obs_dim(self) -> int:
        return self.raw_dim + self.feature_dim

    @classmethod
    def from_base(cls, base, observations, histories, next_observations,
                  max_age: int = 16,
                  age_actions: Sequence[int] = (1,),
                  ridge: float = 1e-3,
                  residual_scale: float = 1.0,
                  episodes: Optional[Sequence[int]] = None,
                  min_gain: float = 1e-6):
        from substrate_generic import LinearObservationModel

        observations = np.asarray(observations, dtype=float)
        histories = np.asarray(histories, dtype=float)
        next_observations = np.asarray(next_observations, dtype=float)
        raw_dim = int(base.obs_dim)
        raw = observations[:, :raw_dim]
        features = observations[:, raw_dim:]
        target = next_observations[:, :raw_dim]
        residual = target - base.predict(raw, histories)
        gate_gain = 0.0
        if episodes is not None and len(episodes) == len(observations):
            episodes = np.asarray(episodes)
            train = episodes % 5 != 4
            test = ~train
            if int(train.sum()) >= 10 and int(test.sum()) >= 5:
                model = LinearObservationModel(
                    features.shape[1], histories.shape[1],
                    ridge=float(ridge),
                ).fit(
                    features[train], histories[train], residual[train],
                )
                prediction = model.predict(
                    features[test], histories[test]
                )
                mse_zero = float(np.mean(residual[test] ** 2))
                mse_model = float(np.mean((residual[test] - prediction) ** 2))
                gate_gain = (mse_zero - mse_model) / (mse_zero + 1e-12)
                if gate_gain <= float(min_gain):
                    residual_scale = 0.0
                return cls(
                    base=base,
                    residual=model,
                    raw_dim=raw_dim,
                    feature_dim=features.shape[1],
                    max_age=max_age,
                    age_actions=age_actions,
                    residual_scale=residual_scale,
                    validation_gain=gate_gain,
                )
        model = LinearObservationModel(
            features.shape[1], histories.shape[1], ridge=float(ridge)
        ).fit(features, histories, residual)
        return cls(
            base=base,
            residual=model,
            raw_dim=raw_dim,
            feature_dim=features.shape[1],
            max_age=max_age,
            age_actions=age_actions,
            residual_scale=residual_scale,
            validation_gain=gate_gain,
        )

    def _advance(self, features: np.ndarray,
                 histories: np.ndarray) -> np.ndarray:
        ages = np.rint(
            np.asarray(features, dtype=float) * float(self.max_age)
        ).astype(int)
        actions = np.asarray(histories, dtype=float)[:, 0].astype(int)
        ages = np.minimum(ages + 1, self.max_age)
        for column, action in enumerate(self.age_actions):
            if column >= ages.shape[1]:
                break
            ages[actions == int(action), column] = 0
        return ages / float(self.max_age)

    def predict(self, states, histories) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        histories = np.asarray(histories, dtype=float)
        raw = states[:, :self.raw_dim]
        features = states[:, self.raw_dim:]
        raw_next = (
            self.base.predict(raw, histories)
            + self.residual_scale * self.residual.predict(features, histories)
        )
        next_features = self._advance(features, histories)
        return np.concatenate([raw_next, next_features], axis=1)


class PerTargetAgeWorld:
    """Raw base world plus per-target, age-only residual corrections.

    Only targets whose held-out residual gain passes the gate are corrected;
    horizontal/time targets are excluded by the caller.  The auxiliary age is
    advanced by the known rule and the agent still sees the raw prefix.
    """

    def __init__(self, base, raw_dim: int, feature_dim: int, max_age: int,
                 age_actions: Tuple[int, ...],
                 corrections: Optional[dict] = None,
                 gains: Optional[dict] = None):
        self.base = base
        self.raw_dim = int(raw_dim)
        self.feature_dim = int(feature_dim)
        self.max_age = max(int(max_age), 1)
        self.age_actions = tuple(int(value) for value in age_actions)
        self.corrections = dict(corrections or {})
        self.gains = dict(gains or {})

    @property
    def obs_dim(self) -> int:
        return self.raw_dim + self.feature_dim

    @classmethod
    def from_base(cls, base, observations, histories, next_observations,
                  episodes, max_age: int = 16,
                  age_actions: Sequence[int] = (1,),
                  targets: Sequence[int] = (0, 1, 3, 5),
                  min_gain: float = 1e-3,
                  ridge: float = 1e-6):
        observations = np.asarray(observations, dtype=float)
        histories = np.asarray(histories, dtype=float)
        next_observations = np.asarray(next_observations, dtype=float)
        episodes = np.asarray(episodes)
        raw_dim = int(base.obs_dim)
        raw = observations[:, :raw_dim]
        features = observations[:, raw_dim:]
        target = next_observations[:, :raw_dim]
        residual = target - base.predict(raw, histories)
        train = episodes % 5 != 4
        test = ~train
        corrections = {}
        gains = {}
        for column in targets:
            column = int(column)
            x_train = np.column_stack([
                np.ones(int(train.sum())), features[train]
            ])
            x_test = np.column_stack([
                np.ones(int(test.sum())), features[test]
            ])
            y_train = residual[train, column]
            gram = x_train.T @ x_train + float(ridge) * np.eye(
                x_train.shape[1]
            )
            weights = np.linalg.solve(gram, x_train.T @ y_train)
            prediction = x_test @ weights
            mse_zero = float(np.mean(residual[test, column] ** 2))
            mse_model = float(np.mean(
                (residual[test, column] - prediction) ** 2
            ))
            gain = (mse_zero - mse_model) / (mse_zero + 1e-12)
            gains[column] = float(gain)
            if gain > float(min_gain):
                corrections[column] = weights
        return cls(
            base=base,
            raw_dim=raw_dim,
            feature_dim=features.shape[1],
            max_age=max_age,
            age_actions=age_actions,
            corrections=corrections,
            gains=gains,
        )

    def _advance(self, features: np.ndarray,
                 histories: np.ndarray) -> np.ndarray:
        ages = np.rint(
            np.asarray(features, dtype=float) * float(self.max_age)
        ).astype(int)
        actions = np.asarray(histories, dtype=float)[:, 0].astype(int)
        ages = np.minimum(ages + 1, self.max_age)
        for column, action in enumerate(self.age_actions):
            if column >= ages.shape[1]:
                break
            ages[actions == int(action), column] = 0
        return ages / float(self.max_age)

    def predict(self, states, histories) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        histories = np.asarray(histories, dtype=float)
        raw = states[:, :self.raw_dim]
        features = states[:, self.raw_dim:]
        raw_next = self.base.predict(raw, histories)
        for column, weights in self.corrections.items():
            design = np.column_stack([
                np.ones(len(features)), features
            ])
            raw_next[:, column] += design @ weights
        next_features = self._advance(features, histories)
        return np.concatenate([raw_next, next_features], axis=1)


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
