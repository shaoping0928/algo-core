"""Select a StageDemand from supported siblings with a learned outcome readout.

The selector is the consumer the high-level stage layer was missing.  It does
not read a task goal, a CLI field, or the modal successor.  It enumerates the
siblings that the stage outcome memory actually supports for the current
context and exact ordered action option, then scores each one with a readout
fit from realized world feedback.

Training is episode-held-out: the memory used to produce a candidate's
features is fit on the other episodes of the fold, so the readout never sees
the realized outcome it is asked to score as its own training input.  The
target is the realized remaining net feedback after the source stage
(``dopamine - pain`` summed over every stage that actually followed).  Remaining
duration and terminal/censored flags are reported alongside, but the demand is
chosen by the learned net-feedback value; this is observational, not causal,
and remains a rank-only hint until a quality gate says otherwise.

Terminal and censored siblings are never demanded, unsupported siblings never
inherit another option's outcomes, and an empty candidate set returns None
(UNKNOWN) rather than a fabricated target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from stage_demand_heuristic import StageDemand
from stage_outcome_memory import (
    ActionTrace,
    StageOccurrence,
    StageOutcomeMemory,
)


def outcome_features(outcome, feedback_dim: int = 2) -> np.ndarray:
    """Readout features available at selection time, derived from one outcome."""
    mean = list(outcome.get("feedback_mean") or ())
    variance = list(outcome.get("feedback_variance") or ())
    mean = (mean + [0.0] * feedback_dim)[:feedback_dim]
    variance = (variance + [0.0] * feedback_dim)[:feedback_dim]
    duration = outcome.get("duration") or {}
    return np.asarray(
        [
            float(outcome.get("probability") or 0.0),
            float(np.log1p(max(duration.get("mean", 0.0), 0.0))),
            float(np.log1p(max(duration.get("q90", 0.0), 0.0))),
        ]
        + [float(value) for value in mean]
        + [float(value) for value in variance]
        + [
            float(bool(outcome.get("terminal", False))),
            float(bool(outcome.get("censored", False))),
            float(outcome.get("destination_terminal_rate", 0.0) or 0.0),
            float(outcome.get("destination_censored_rate", 0.0) or 0.0),
            float(np.log1p(max(outcome.get("support", 0), 0))),
            float(np.log1p(max(outcome.get("episode_support", 0), 0))),
        ],
        dtype=float,
    )


def remaining_net_feedback(stages: Sequence[StageOccurrence], index: int):
    """Realized net feedback after the source stage, never a counterfactual."""
    total = 0.0
    for stage in stages[index + 1:]:
        feedback = np.asarray(stage.feedback, dtype=float).reshape(-1)
        if feedback.size >= 2:
            total += float(feedback[0] - feedback[1])
        elif feedback.size:
            total += float(feedback[0])
    return float(total)


class StageOutcomeReadout:
    """Ridge readout of the realized remaining net feedback."""

    def __init__(self, mean, scale, weights, samples: int,
                 target_mean: float, target_std: float,
                 feedback_dim: int = 2):
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.samples = int(samples)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.feedback_dim = int(feedback_dim)

    @classmethod
    def fit(cls, episodes: Sequence[Sequence[StageOccurrence]],
            folds: int = 5, ridge: float = 1.0,
            memory_support: int = 2, memory_episodes: int = 2):
        """Fit on fold-out candidates so features are never the label's source."""
        episodes = [list(stages) for stages in episodes if stages]
        if len(episodes) < 2:
            return cls([], [], [], 0, 0.0, 0.0)
        feedback_dim = max(
            (
                int(np.asarray(stage.feedback).reshape(-1).size)
                for stages in episodes for stage in stages
            ),
            default=2,
        )
        feedback_dim = max(1, feedback_dim)
        folds = max(2, min(int(folds), len(episodes)))
        features = []
        targets = []
        for fold in range(folds):
            train = [
                stages for index, stages in enumerate(episodes)
                if index % folds != fold
            ]
            heldout = [
                stages for index, stages in enumerate(episodes)
                if index % folds == fold
            ]
            if not train or not heldout:
                continue
            memory = StageOutcomeMemory(
                min_support=memory_support,
                min_episodes=memory_episodes,
            ).fit(train)
            for stages in heldout:
                for index, source in enumerate(stages[:-1]):
                    if source.terminal or source.censored:
                        continue
                    history = [stage.key for stage in stages[:index + 1]]
                    prediction = memory.predict(
                        history, source.action_trace,
                    )
                    if prediction.get("status") != "supported":
                        continue
                    realized = stages[index + 1].key
                    outcome = next(
                        (
                            row for row in prediction["outcomes"]
                            if row.get("destination") == realized
                        ),
                        None,
                    )
                    if outcome is None:
                        continue
                    features.append(
                        outcome_features(outcome, feedback_dim)
                    )
                    targets.append(remaining_net_feedback(stages, index))
        if not features:
            return cls([], [], [], 0, 0.0, 0.0, feedback_dim=feedback_dim)
        matrix = np.asarray(features, dtype=float)
        target = np.asarray(targets, dtype=float)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        standardized = (matrix - mean) / scale
        design = np.column_stack([
            np.ones(len(standardized)), standardized,
        ])
        penalty = float(ridge) * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(
            design.T @ design + penalty, design.T @ target,
        )
        return cls(
            mean=mean, scale=scale, weights=weights,
            samples=len(target),
            target_mean=float(target.mean()),
            target_std=float(target.std()),
            feedback_dim=feedback_dim,
        )

    def score(self, outcome) -> float:
        if self.samples == 0:
            return float("nan")
        features = (
            outcome_features(outcome, self.feedback_dim) - self.mean
        ) / self.scale
        design = np.concatenate([[1.0], features])
        return float(design @ self.weights)

    def snapshot(self):
        return dict(
            samples=int(self.samples),
            features=int(len(self.mean)),
            target_mean=float(self.target_mean),
            target_std=float(self.target_std),
            weights=np.asarray(self.weights).tolist(),
        )


class StageDemandSelector:
    """Choose one supported non-terminal sibling using the learned readout."""

    def __init__(self, readout: StageOutcomeReadout,
                 min_outcome_episodes: int = 2,
                 min_samples: int = 4):
        if int(min_outcome_episodes) < 1:
            raise ValueError("min_outcome_episodes must be positive")
        if int(min_samples) < 1:
            raise ValueError("min_samples must be positive")
        self.readout = readout
        self.min_outcome_episodes = int(min_outcome_episodes)
        self.min_samples = int(min_samples)

    def select(self, prediction, stage_history,
               action_trace: Optional[ActionTrace] = None,
               association_action_trace: Optional[ActionTrace] = None):
        """Return ``(StageDemand, report)`` or ``(None, report)``."""
        report = dict(
            status="unknown",
            candidates=(),
            reason=None,
            readout_samples=int(self.readout.samples),
        )
        if prediction.get("status") != "supported":
            report["reason"] = "no_supported_context"
            return None, report
        if self.readout.samples < self.min_samples:
            report["reason"] = "readout_unsupported"
            return None, report
        candidates = []
        for outcome in prediction.get("outcomes", ()):
            if outcome.get("terminal") or outcome.get("censored"):
                continue
            if (outcome.get("destination_terminal_rate", 0.0)
                    or outcome.get("destination_censored_rate", 0.0)):
                # A destination type that ever ended the episode is not a
                # safe high-level demand, regardless of its learned value.
                continue
            destination = outcome.get("destination")
            if destination is None:
                continue
            if int(outcome.get("episode_support", 0)) < (
                    self.min_outcome_episodes):
                continue
            value = self.readout.score(outcome)
            if not np.isfinite(value):
                continue
            candidates.append(dict(
                destination=tuple(destination),
                value=float(value),
                probability=float(outcome.get("probability") or 0.0),
                support=int(outcome.get("support", 0)),
                episode_support=int(outcome.get("episode_support", 0)),
                duration_mean=float(
                    (outcome.get("duration") or {}).get("mean", 0.0)
                ),
                feedback_mean=list(outcome.get("feedback_mean") or ()),
            ))
        candidates.sort(key=lambda row: (-row["value"], repr(row["destination"])))
        report["candidates"] = tuple(candidates)
        if not candidates:
            report["reason"] = "no_supported_nonterminal_sibling"
            return None, report
        best = candidates[0]
        # Build from the exact row that was scored.  Re-matching by target
        # could pick another outcome row with the same destination but a
        # different probability when the memory reports duplicates.
        demand = StageDemand(
            target_stage=best["destination"],
            stage_history=tuple(stage_history),
            action_trace=action_trace,
            association_action_trace=association_action_trace,
            transition_probability=best["probability"],
            transition_support=int(prediction.get("support", 0)),
            transition_resolution=prediction.get("resolution"),
        )
        report.update(
            status="supported",
            target=best["destination"],
            value=best["value"],
            runner_up_value=(
                candidates[1]["value"] if len(candidates) > 1 else None
            ),
            margin=(
                best["value"] - candidates[1]["value"]
                if len(candidates) > 1 else None
            ),
            provenance="world_feedback_outcome_readout",
            affects_value=False,
        )
        return demand, report

    def snapshot(self):
        return dict(
            readout=self.readout.snapshot(),
            min_outcome_episodes=int(self.min_outcome_episodes),
            min_samples=int(self.min_samples),
            affects_value=False,
        )
