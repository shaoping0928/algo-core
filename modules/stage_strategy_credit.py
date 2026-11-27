"""Episode-level validation credit for learned stage strategy proposals."""
from __future__ import annotations

from math import comb


def _freeze(value):
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    hash(value)
    return value


class StageStrategyCredit:
    """Track whether a proposed stage successor matches a realized stage.

    Each strategy receives at most one outcome per episode. Unobserved or
    censored outcomes are omitted, and credit is suitable only for a separate
    search-priority channel, not for final objective scoring.
    """

    def __init__(self, min_episodes=2, confidence_threshold=0.8,
                 max_priority_adjustment=0.25):
        if int(min_episodes) < 1:
            raise ValueError("min_episodes must be positive")
        if not 0.5 < float(confidence_threshold) < 1.0:
            raise ValueError("confidence_threshold must be in (0.5, 1)")
        if not 0.0 <= float(max_priority_adjustment) < 1.0:
            raise ValueError(
                "max_priority_adjustment must be in [0, 1)"
            )
        self.min_episodes = int(min_episodes)
        self.confidence_threshold = float(confidence_threshold)
        self.max_priority_adjustment = float(max_priority_adjustment)
        self._outcomes = {}

    @staticmethod
    def _key(source_history, target_stage):
        return (
            _freeze(tuple(source_history)),
            _freeze(tuple(target_stage)),
        )

    def record(self, source_history, target_stage, episode, similarity):
        """Record one realized outcome, ignoring duplicate episode evidence."""
        score = float(similarity)
        if not 0.0 <= score <= 1.0:
            raise ValueError("similarity must be in [0, 1]")
        key = self._key(source_history, target_stage)
        outcomes = self._outcomes.setdefault(key, {})
        if episode in outcomes:
            return False
        outcomes[episode] = score
        return True

    @staticmethod
    def _probability_better_than_chance(successes, failures):
        # Exact Beta(successes + 1, failures + 1) posterior mass above 0.5.
        trials = int(successes) + int(failures) + 1
        return sum(
            comb(trials, index)
            for index in range(int(successes) + 1)
        ) / float(2 ** trials)

    def estimate(self, source_history, target_stage):
        outcomes = self._outcomes.get(
            self._key(source_history, target_stage), {},
        )
        scores = tuple(outcomes.values())
        successes = sum(score >= 0.5 for score in scores)
        failures = len(scores) - successes
        probability = (
            self._probability_better_than_chance(successes, failures)
            if scores else 0.5
        )
        multiplier = 1.0
        if len(scores) >= self.min_episodes:
            if probability >= self.confidence_threshold:
                multiplier += self.max_priority_adjustment * (
                    2.0 * probability - 1.0
                )
            elif probability <= 1.0 - self.confidence_threshold:
                multiplier -= self.max_priority_adjustment * (
                    1.0 - 2.0 * probability
                )
        return dict(
            status=(
                "supported" if len(scores) >= self.min_episodes
                else "unknown"
            ),
            episode_support=len(scores),
            mean_similarity=(
                sum(scores) / len(scores) if scores else None
            ),
            successes=int(successes),
            failures=int(failures),
            probability_better_than_chance=float(probability),
            priority_multiplier=float(multiplier),
        )

    def snapshot(self):
        summaries = []
        for (source_history, target_stage), outcomes in self._outcomes.items():
            estimate = self.estimate(source_history, target_stage)
            summaries.append(dict(
                source_history=source_history,
                target_stage=target_stage,
                **estimate,
            ))
        return dict(
            strategies=len(self._outcomes),
            observations=sum(len(rows) for rows in self._outcomes.values()),
            estimates=tuple(summaries),
            min_episodes=self.min_episodes,
            confidence_threshold=self.confidence_threshold,
            max_priority_adjustment=self.max_priority_adjustment,
        )
