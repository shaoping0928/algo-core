"""Self-supervised pipeline: realized rows -> signals -> stages -> samples.

This is the automatic experience path behind the stage-memory base.  The only
inputs are the records the world interaction already produced:

* observations and next observations,
* the action that actually acted on each transition (``history[lag]``),
* the world feedback channels,
* the episode's terminal / censoring flag,
* and the runtime's own learned feature masks, which are read-only.

Nothing here names a task event, reads task geometry, supplies goals, or
creates counterfactual rows.  The detector is fit on the rows handed to
:meth:`StageExperienceBatch.fit`; transforming other episodes never updates
its thresholds.  Continuous-state identity, predictive-equivalence merging,
and demand selection are separate layers that consume this batch.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from stage_demand_heuristic import DemandAssociationSample
from stage_outcome_memory import (
    GenericKeySignalDetector,
    StageOccurrence,
    segment_episode,
)


@dataclass(frozen=True)
class _EpisodeFrame:
    """One contiguous episode in the form the stage segmenter expects."""

    episode: Hashable
    observations: np.ndarray
    actions: Tuple[int, ...]
    feedback: np.ndarray
    active_features: Tuple[Tuple[Hashable, ...], ...]
    terminal: bool
    censored: bool


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_native(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_native(item) for item in value)
    hash(value)
    return value


def _row_features(agent, obs) -> Tuple[Hashable, ...]:
    """Internal feature roots active at one observation (read-only)."""
    branch = getattr(agent, "_branch_active_features", None)
    features = branch(obs) if branch is not None else ()
    return tuple(dict.fromkeys(_native(feature) for feature in features))


def _episode_frames(rows: Sequence[Mapping], agent, lag: int):
    """Group rows by episode and reject anything that cannot be attributed.

    Rows are consumed in the order given; the acting action is taken from
    ``history[lag]``.  An episode missing that slot is dropped rather than
    silently falling back to the newly issued action, and its row count is
    reported as skipped.
    """
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(_native(row.get("episode", 0)), []).append(row)

    frames = []
    skipped_episodes = 0
    skipped_rows = 0
    for episode, episode_rows in grouped.items():
        if not episode_rows:
            skipped_episodes += 1
            continue
        observations = []
        actions = []
        feedback = []
        features = []
        usable = True
        for row in episode_rows:
            obs = np.asarray(row.get("obs", ()), dtype=float).reshape(-1)
            next_obs = np.asarray(
                row.get("next_obs", ()), dtype=float
            ).reshape(-1)
            history = np.asarray(
                row.get("history", ()), dtype=float
            ).reshape(-1)
            signals = np.asarray(
                row.get("signals", ()), dtype=float
            ).reshape(-1)
            if (not obs.size or next_obs.size != obs.size
                    or history.size <= int(lag)
                    or signals.size < 2
                    or not np.isfinite(obs).all()
                    or not np.isfinite(next_obs).all()
                    or not np.isfinite(history).all()
                    or not np.isfinite(signals[:2]).all()):
                usable = False
                break
            observations.append(obs)
            # history is newest-first; history[lag] acted on this transition.
            actions.append(int(round(float(history[int(lag)]))))
            feedback.append(signals[:2])
            features.append(_row_features(agent, obs))
        if not usable:
            skipped_episodes += 1
            skipped_rows += len(episode_rows)
            continue
        observations.append(np.asarray(
            episode_rows[-1]["next_obs"], dtype=float
        ).reshape(-1))
        terminal = bool(
            episode_rows[-1].get("terminal", episode_rows[-1].get("done", False))
        )
        frames.append(_EpisodeFrame(
            episode=episode,
            observations=np.asarray(observations, dtype=float),
            actions=tuple(actions),
            feedback=np.asarray(feedback, dtype=float),
            active_features=tuple(features),
            terminal=terminal,
            censored=not terminal,
        ))
    return frames, skipped_episodes, skipped_rows


def _census(episodes, samples, skipped_episodes, skipped_rows) -> dict:
    stages = [stage for episode in episodes for stage in episode]
    durations = np.asarray(
        [stage.duration for stage in stages], dtype=float
    )
    snapshots = [
        row for stage in stages for row in stage.feature_snapshots
    ]
    keyed = [stage for stage in stages if stage.key]
    feature_ids = set()
    feature_rows = 0
    for row in snapshots:
        if row:
            feature_rows += 1
            feature_ids.update(row)
    transitions = sum(max(len(episode) - 1, 0) for episode in episodes)
    return dict(
        episodes=int(len(episodes)),
        episodes_skipped=int(skipped_episodes),
        rows=int(sum(stage.duration for stage in stages)),
        rows_skipped=int(skipped_rows),
        stages=int(len(stages)),
        transitions=int(transitions),
        stage_types=int(len({stage.key for stage in stages})),
        keyed_stages=int(len(keyed)),
        empty_key_stages=int(len(stages) - len(keyed)),
        demand_samples=int(len(samples)),
        terminated_episodes=int(sum(
            1 for episode in episodes
            if episode and episode[-1].terminal
        )),
        censored_episodes=int(sum(
            1 for episode in episodes
            if episode and episode[-1].censored
        )),
        mean_stages_per_episode=(
            float(np.mean([len(episode) for episode in episodes]))
            if episodes else 0.0
        ),
        duration_mean=(
            float(durations.mean()) if durations.size else 0.0
        ),
        duration_median=(
            float(np.quantile(durations, 0.5)) if durations.size else 0.0
        ),
        duration_q90=(
            float(np.quantile(durations, 0.9)) if durations.size else 0.0
        ),
        feature_rows=int(feature_rows),
        feature_coverage=(
            float(feature_rows) / float(len(snapshots))
            if snapshots else 0.0
        ),
        feature_ids=int(len(feature_ids)),
    )


@dataclass(frozen=True)
class StageExperienceBatch:
    """A frozen detector plus the stages and demand samples it produced.

    ``census["rows"]`` counts the realized transitions that entered stages.
    ``episodes`` is chronological per episode, and ``demand_samples`` pairs
    each per-transition feature snapshot with the stage that actually
    followed, never with a counterfactual successor.
    """

    detector: GenericKeySignalDetector
    lag: int
    episodes: Tuple[Tuple[StageOccurrence, ...], ...]
    demand_samples: Tuple[DemandAssociationSample, ...]
    census: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: Sequence[Mapping], agent, lag: Optional[int] = None,
            quantile: float = 0.98, build_demand: bool = True):
        """Fit the key-signal detector on these rows and segment them.

        Held-out episodes should go through :meth:`transform`, which reuses
        this frozen detector and cannot move its thresholds.
        """
        lag = int(agent.lag if lag is None else lag)
        if lag < 0:
            raise ValueError("lag must be non-negative")
        frames, skipped_episodes, skipped_rows = _episode_frames(
            rows, agent, lag,
        )
        if not frames:
            raise ValueError("no usable episodes in the training rows")
        detector = GenericKeySignalDetector(quantile=quantile).fit([
            dict(observations=frame.observations, feedback=frame.feedback)
            for frame in frames
        ])
        episodes, samples = _segment_frames(
            frames, detector, build_demand=build_demand,
        )
        census = _census(
            episodes, samples, skipped_episodes, skipped_rows,
        )
        census["signal_channels"] = int(len(detector.thresholds))
        return cls(
            detector=detector,
            lag=lag,
            episodes=episodes,
            demand_samples=samples,
            census=census,
        )

    def transform(self, rows: Sequence[Mapping], agent,
                  build_demand: bool = True) -> "StageExperienceBatch":
        """Segment other episodes with the already-fitted detector."""
        frames, skipped_episodes, skipped_rows = _episode_frames(
            rows, agent, self.lag,
        )
        episodes, samples = _segment_frames(
            frames, self.detector, build_demand=build_demand,
        )
        census = _census(
            episodes, samples, skipped_episodes, skipped_rows,
        )
        census["signal_channels"] = int(len(self.detector.thresholds))
        return StageExperienceBatch(
            detector=self.detector,
            lag=self.lag,
            episodes=episodes,
            demand_samples=samples,
            census=census,
        )

    def outcome_episodes(self):
        """Stage sequences in the form ``StageOutcomeMemory.fit`` expects."""
        return [list(episode) for episode in self.episodes]

    def snapshot(self):
        return dict(self.census)


def _segment_frames(frames, detector: GenericKeySignalDetector,
                    build_demand: bool = True):
    episodes = []
    for frame in frames:
        signals = detector.transform_episode(
            frame.observations, feedback=frame.feedback,
        )
        stages = segment_episode(
            episode=frame.episode,
            observations=frame.observations,
            actions=frame.actions,
            feedback=frame.feedback,
            key_signals=signals,
            terminal=frame.terminal,
            censored=frame.censored,
            active_features=frame.active_features,
        )
        if stages:
            episodes.append(tuple(stages))
    samples = (
        DemandAssociationSample.from_stage_episodes(episodes)
        if build_demand else ()
    )
    return tuple(episodes), tuple(samples)
