"""Demand-conditioned feature ranking, separate from final value.

The high-level planner selects a supported stage outcome and emits a
``StageDemand``. This module learns directional associations from realized
source-stage features to realized successor stages. At search time it returns
an ordering priority only; it never changes value estimates, feedback, or
feasibility.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log
from typing import Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from stage_outcome_memory import (
    ActionTrace,
    StageKey,
    encode_action_trace,
    stage_context_path,
)


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_native(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_native(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_native(item) for item in value)
    hash(value)
    return value


def _features_with_activation(features):
    if isinstance(features, Mapping):
        result = {
            _native(feature): float(activation)
            for feature, activation in features.items()
        }
    else:
        result = {_native(feature): 1.0 for feature in features}
    if any(
            not np.isfinite(activation) or activation < 0.0
            for activation in result.values()
    ):
        raise ValueError("feature activations must be finite and non-negative")
    return {
        feature: activation
        for feature, activation in result.items()
        if activation > 0.0
    }


@dataclass(frozen=True)
class StageDemand:
    """A requested successor stage selected by a higher-level planner.

    ``action_trace`` identifies the high-level transition option. The feature
    heuristic may use a separately trained action-conditioned association
    trace; when absent it deliberately uses its pooled observational readout.
    ``demand_strength`` scales only the temporary search priority, never the
    feature's learned value or the planner's final objective.
    """

    target_stage: StageKey
    stage_history: Tuple[StageKey, ...]
    action_trace: Optional[ActionTrace] = None
    association_action_trace: Optional[ActionTrace] = None
    transition_probability: Optional[float] = None
    transition_support: int = 0
    transition_resolution: Optional[int] = None
    transition_episode_support: int = 0
    demand_strength: float = 1.0

    def __post_init__(self):
        target = tuple(_native(token) for token in self.target_stage)
        history = tuple(
            tuple(_native(token) for token in key)
            for key in self.stage_history
        )
        trace = (
            None if self.action_trace is None else tuple(
                (_native(action), int(count))
                for action, count in self.action_trace
            )
        )
        association_trace = (
            None if self.association_action_trace is None else tuple(
                (_native(action), int(count))
                for action, count in self.association_action_trace
            )
        )
        if trace is not None and any(count <= 0 for _action, count in trace):
            raise ValueError("action trace run lengths must be positive")
        if (association_trace is not None
                and any(count <= 0 for _action, count in association_trace)):
            raise ValueError(
                "association action trace run lengths must be positive"
            )
        probability = self.transition_probability
        if probability is not None:
            probability = float(probability)
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("transition_probability must be in [0, 1]")
        if int(self.transition_support) < 0:
            raise ValueError("transition_support must be non-negative")
        if int(self.transition_episode_support) < 0:
            raise ValueError(
                "transition_episode_support must be non-negative"
            )
        strength = float(self.demand_strength)
        if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("demand_strength must be in [0, 1]")
        object.__setattr__(self, "target_stage", target)
        object.__setattr__(self, "stage_history", history)
        object.__setattr__(self, "action_trace", trace)
        object.__setattr__(
            self, "association_action_trace", association_trace
        )
        object.__setattr__(self, "transition_probability", probability)
        object.__setattr__(
            self, "transition_support", int(self.transition_support)
        )
        object.__setattr__(
            self, "transition_episode_support",
            int(self.transition_episode_support),
        )
        object.__setattr__(self, "demand_strength", strength)

    @classmethod
    def from_transition_prediction(cls, prediction, target_stage,
                                   stage_history, action_trace,
                                   association_action_trace=None):
        """Build a demand from a supported, non-terminal sibling outcome."""
        if prediction.get("status") != "supported":
            raise ValueError("cannot demand a target from UNKNOWN")
        target = tuple(_native(token) for token in target_stage)
        matching = [
            outcome for outcome in prediction.get("outcomes", ())
            if outcome.get("destination") == target
            and not outcome.get("terminal", False)
            and not outcome.get("censored", False)
            and not outcome.get("destination_terminal_rate", 0.0)
            and not outcome.get("destination_censored_rate", 0.0)
        ]
        if not matching:
            raise ValueError("target is not a supported sibling outcome")
        outcome = matching[0]
        episode_support = int(outcome.get("episode_support", 0))
        strength = episode_support / float(episode_support + 2)
        return cls(
            target_stage=target,
            stage_history=tuple(stage_history),
            action_trace=action_trace,
            association_action_trace=association_action_trace,
            transition_probability=float(outcome["probability"]),
            transition_support=int(prediction["support"]),
            transition_resolution=prediction.get("resolution"),
            transition_episode_support=episode_support,
            demand_strength=strength,
        )


@dataclass(frozen=True)
class StageDemandAlternatives:
    """A bounded slate of learned successor strategies for one shared search."""

    demands: Tuple[StageDemand, ...]

    def __post_init__(self):
        demands = tuple(self.demands)
        if not demands or any(
                not isinstance(demand, StageDemand) for demand in demands):
            raise ValueError(
                "demands must contain at least one StageDemand"
            )
        object.__setattr__(self, "demands", demands)

    @property
    def target_stage(self):
        return self.demands[0].target_stage

    @property
    def target_stages(self):
        return tuple(demand.target_stage for demand in self.demands)

    @property
    def stage_history(self):
        return self.demands[0].stage_history

    @property
    def association_action_trace(self):
        return self.demands[0].association_action_trace


@dataclass(frozen=True)
class DemandAssociationSample:
    """One realized stage transition and its source-stage active features."""

    episode: Hashable
    stage_history: Tuple[StageKey, ...]
    active_features: Tuple[Hashable, ...]
    destination_stage: StageKey
    action_trace: Optional[ActionTrace] = None

    def __post_init__(self):
        episode = _native(self.episode)
        history = tuple(
            tuple(_native(token) for token in key)
            for key in self.stage_history
        )
        features = tuple(dict.fromkeys(
            _native(feature) for feature in self.active_features
        ))
        destination = tuple(
            _native(token) for token in self.destination_stage
        )
        trace = (
            None if self.action_trace is None else tuple(
                (_native(action), int(count))
                for action, count in self.action_trace
            )
        )
        if trace is not None and any(count <= 0 for _action, count in trace):
            raise ValueError("action trace run lengths must be positive")
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "stage_history", history)
        object.__setattr__(self, "active_features", features)
        object.__setattr__(self, "destination_stage", destination)
        object.__setattr__(self, "action_trace", trace)

    @classmethod
    def from_stage_episodes(cls, episodes):
        """Build pointwise samples from observed adjacent stage outcomes.

        Each source-stage feature snapshot is paired with the actually
        observed next stage. The option is left pooled: its complete realized
        action trace would otherwise reveal future actions to an earlier
        decision-state heuristic.
        """
        samples = []
        for episode in episodes:
            stages = list(episode)
            if any(
                    stage.terminal or stage.censored
                    for stage in stages[:-1]
            ):
                raise ValueError("no stage may follow a terminal/censored one")
            for index, source in enumerate(stages[:-1]):
                if source.terminal or source.censored:
                    continue
                snapshots = (
                    source.feature_snapshots
                    if source.feature_snapshots
                    else (source.active_features,)
                )
                for active_features in snapshots:
                    samples.append(cls(
                        episode=source.episode,
                        stage_history=tuple(
                            stage.key for stage in stages[:index + 1]
                        ),
                        active_features=active_features,
                        destination_stage=stages[index + 1].key,
                        action_trace=None,
                    ))
        return tuple(samples)

    @classmethod
    def from_stage_roots(cls, episodes, collapse, pooled=False):
        """Build samples whose feature is the source stage's collapsed root.

        ``collapse(stage_key)`` returns the predictive-stage-tree key for a
        stage.  The feature is available for any candidate state through the
        resolver, so this association is denser than a per-context prototype
        token.  With ``pooled=False`` the option is the causal action prefix
        before each transition; with ``pooled=True`` one sample per source
        stage is emitted with no option, giving the root transition table a
        dense option-free fallback.  No future action enters either mode.
        """
        samples = []
        for episode in episodes:
            stages = list(episode)
            if any(
                    stage.terminal or stage.censored
                    for stage in stages[:-1]
            ):
                raise ValueError("no stage may follow a terminal/censored one")
            collapsed_history = []
            for index, source in enumerate(stages[:-1]):
                source_root = collapse(source.key)
                target_root = collapse(stages[index + 1].key)
                collapsed_history.append(source_root)
                if source.terminal or source.censored:
                    continue
                actions = []
                for action, count in source.action_trace:
                    actions.extend([action] * int(count))
                snapshots = (
                    source.feature_snapshots
                    if source.feature_snapshots
                    else (source.active_features,)
                )
                if pooled:
                    samples.append(cls(
                        episode=source.episode,
                        stage_history=tuple(collapsed_history),
                        active_features=(source_root[0],),
                        destination_stage=target_root,
                        action_trace=None,
                    ))
                    continue
                for offset, _snapshot in enumerate(snapshots):
                    if offset >= len(actions):
                        break
                    prefix = actions[:offset]
                    samples.append(cls(
                        episode=source.episode,
                        stage_history=tuple(collapsed_history),
                        active_features=(source_root[0],),
                        destination_stage=target_root,
                        action_trace=(
                            encode_action_trace(prefix) if prefix else None
                        ),
                    ))
        return tuple(samples)

    @classmethod
    def from_stage_prefixes(cls, episodes):
        """Build samples with the causal action prefix known before each step.

        For every transition inside a source stage, the option is the
        run-length encoding of the actions executed **before** that
        transition.  The action that will act on the scored transition, and
        every later action, are excluded, so a decision-time caller can use
        the same option.  Offset zero has no prior action and is pooled.
        """
        samples = []
        for episode in episodes:
            stages = list(episode)
            if any(
                    stage.terminal or stage.censored
                    for stage in stages[:-1]
            ):
                raise ValueError("no stage may follow a terminal/censored one")
            for index, source in enumerate(stages[:-1]):
                if source.terminal or source.censored:
                    continue
                actions = []
                for action, count in source.action_trace:
                    actions.extend([action] * int(count))
                snapshots = (
                    source.feature_snapshots
                    if source.feature_snapshots
                    else (source.active_features,)
                )
                history = tuple(stage.key for stage in stages[:index + 1])
                for offset, active_features in enumerate(snapshots):
                    if offset >= len(actions):
                        break
                    prefix = actions[:offset]
                    samples.append(cls(
                        episode=source.episode,
                        stage_history=history,
                        active_features=active_features,
                        destination_stage=stages[index + 1].key,
                        action_trace=(
                            encode_action_trace(prefix) if prefix else None
                        ),
                    ))
        return tuple(samples)


def _option_backoff(trace):
    """Causal option backoff: the exact prefix, then its recent suffixes.

    A query with an option never falls back to the pooled table; that would
    silently mix different action options.  Only shorter suffixes of the same
    executed prefix are tried, and every one of them is known at decision time.
    """
    if trace is None:
        return (None,)
    actions = []
    for action, count in trace:
        actions.extend([action] * int(count))
    result = []
    for length in range(len(actions), 0, -1):
        candidate = encode_action_trace(actions[-length:])
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


class StageDemandHeuristic:
    """Episode-supported feature associations for search ordering only.

    For a demand ``g``, each candidate feature receives a smoothed log-lift
    ``log(P(g | feature) / P(g))`` within the matching stage context. Counts
    and distinct-episode support shrink the contribution. This is associative
    evidence, not a causal value function; callers must keep it separate from
    final objective scores.

    An option-conditioned query tries the exact causal prefix and then shorter
    suffixes of that same prefix; it never falls back to the pooled table.
    """

    def __init__(self, min_support: int = 2, min_episodes: int = 2,
                 prior: float = 4.0):
        if int(min_support) < 1 or int(min_episodes) < 1:
            raise ValueError("support thresholds must be positive")
        if float(prior) <= 0.0:
            raise ValueError("prior must be positive")
        self.min_support = int(min_support)
        self.min_episodes = int(min_episodes)
        self.prior = float(prior)
        self._base = {}
        self._base_episodes = {}
        self._feature_counts = {}
        self._feature_episodes = {}
        self._feature_target_counts = {}
        self._feature_target_episodes = {}
        self.sample_count = 0

    def fit(self, samples: Sequence[DemandAssociationSample]):
        """Fit from realized source-feature to next-stage observations."""
        base = defaultdict(lambda: defaultdict(int))
        base_episodes = defaultdict(lambda: defaultdict(set))
        feature_counts = defaultdict(int)
        feature_episodes = defaultdict(set)
        feature_target_counts = defaultdict(int)
        feature_target_episodes = defaultdict(set)

        for sample in samples:
            # Keep an exact option-conditioned table and a pooled table.
            # Queries with an option never silently back off across options.
            action_keys = (
                (sample.action_trace, None)
                if sample.action_trace is not None else (None,)
            )
            contexts = stage_context_path(sample.stage_history)
            for context in contexts:
                for action_trace in action_keys:
                    context_key = (context, action_trace)
                    target = sample.destination_stage
                    base[context_key][target] += 1
                    base_episodes[context_key][target].add(sample.episode)
                    for feature in sample.active_features:
                        feature_key = (context_key, feature)
                        joint_key = (context_key, feature, target)
                        feature_counts[feature_key] += 1
                        feature_episodes[feature_key].add(sample.episode)
                        feature_target_counts[joint_key] += 1
                        feature_target_episodes[joint_key].add(
                            sample.episode
                        )

        self._base = {
            key: dict(counts) for key, counts in base.items()
        }
        self._base_episodes = {
            key: {
                target: set(episodes)
                for target, episodes in outcomes.items()
            }
            for key, outcomes in base_episodes.items()
        }
        self._feature_counts = dict(feature_counts)
        self._feature_episodes = {
            key: set(episodes) for key, episodes in feature_episodes.items()
        }
        self._feature_target_counts = dict(feature_target_counts)
        self._feature_target_episodes = {
            key: set(episodes)
            for key, episodes in feature_target_episodes.items()
        }
        self.sample_count = len(samples)
        return self

    def score(self, active_features, demand: StageDemand):
        """Return a separate heuristic priority; higher means more relevant."""
        features = _features_with_activation(active_features)
        if isinstance(demand, StageDemandAlternatives):
            results = [
                (proposal, self.score(features, proposal))
                for proposal in demand.demands
            ]
            supported = [
                (proposal, result) for proposal, result in results
                if result["status"] == "supported"
            ]
            if not supported:
                return self._unknown()
            best_proposal, best = max(
                supported,
                key=lambda item: float(item[1]["priority"]),
            )
            return dict(
                best,
                proposal_count=len(demand.demands),
                proposal_target=best_proposal.target_stage,
                provenance="bounded_realized_strategy_slate",
            )
        trace = demand.association_action_trace
        traces = _option_backoff(trace)
        fallback = None
        for context in stage_context_path(demand.stage_history):
            for candidate_trace in traces:
                context_key = (context, candidate_trace)
                base = self._base.get(context_key)
                if not base:
                    continue
                total = int(sum(base.values()))
                all_episodes = set()
                for episode_ids in self._base_episodes[context_key].values():
                    all_episodes.update(episode_ids)
                target_count = int(base.get(demand.target_stage, 0))
                target_episodes = self._base_episodes[context_key].get(
                    demand.target_stage, set(),
                )
                if (total < self.min_support
                        or len(all_episodes) < self.min_episodes
                        or len(target_episodes) < self.min_episodes):
                    continue
                fallback = (
                    context, len(context), total, len(all_episodes),
                )
                outcome_count = len(base)
                base_rate = (
                    (target_count + 1.0)
                    / (total + max(outcome_count, 1))
                )
                evidence = []
                weighted_priority = 0.0
                activation_total = 0.0
                for feature, activation in features.items():
                    feature_key = (context_key, feature)
                    joint_key = (context_key, feature, demand.target_stage)
                    feature_count = int(
                        self._feature_counts.get(feature_key, 0)
                    )
                    feature_episodes = self._feature_episodes.get(
                        feature_key, set(),
                    )
                    target_feature_episodes = (
                        self._feature_target_episodes.get(joint_key, set())
                    )
                    if (feature_count < self.min_support
                            or len(feature_episodes) < self.min_episodes
                            or len(target_feature_episodes)
                            < self.min_episodes):
                        continue
                    joint_count = int(
                        self._feature_target_counts.get(joint_key, 0)
                    )
                    conditional = (
                        joint_count + self.prior * base_rate
                    ) / (feature_count + self.prior)
                    lift = float(log(
                        max(conditional, 1e-12) / max(base_rate, 1e-12)
                    ))
                    confidence = feature_count / (
                        feature_count + self.prior
                    )
                    contribution = confidence * lift
                    weighted_priority += activation * contribution
                    activation_total += activation
                    evidence.append(dict(
                        feature=feature,
                        support=feature_count,
                        episode_support=len(feature_episodes),
                        target_support=joint_count,
                        conditional_probability=float(conditional),
                        base_rate=float(base_rate),
                        lift=lift,
                        contribution=float(contribution),
                        activation=float(activation),
                    ))
                if activation_total <= 0.0:
                    # The context has base support but no feature evidence.
                    # A parent context may pool enough evidence; do not stop.
                    continue
                return dict(
                    status="supported",
                    priority=float(
                        demand.demand_strength
                        * weighted_priority / activation_total
                    ),
                    context=context,
                    context_depth=int(len(context)),
                    resolution=int(len(context)),
                    support=int(total),
                    episode_support=int(len(all_episodes)),
                    outcome_diversity=int(outcome_count),
                    target_base_rate=float(base_rate),
                    action_conditioned=trace is not None,
                    evidence=tuple(evidence),
                    demand_strength=float(demand.demand_strength),
                    provenance="observed_cooccurrence",
                    affects_value=False,
                )
        if fallback is not None:
            context, depth, support, episode_support = fallback
            return self._unknown(
                context=context, context_depth=depth,
                support=support, episode_support=episode_support,
            )
        return self._unknown()

    @staticmethod
    def _unknown(context=(), context_depth=None, support=0,
                 episode_support=0):
        return dict(
            status="unknown",
            priority=None,
            context=tuple(context),
            context_depth=context_depth,
            resolution=context_depth,
            support=int(support),
            episode_support=int(episode_support),
            outcome_diversity=0,
            target_base_rate=None,
            action_conditioned=None,
            evidence=(),
            provenance="observed_cooccurrence",
            affects_value=False,
        )

    def snapshot(self):
        return dict(
            samples=int(self.sample_count),
            context_feature_pairs=len(self._feature_counts),
            context_outcomes=len(self._base),
            min_support=int(self.min_support),
            min_episodes=int(self.min_episodes),
            prior=float(self.prior),
        )


def make_demand_rank_step(heuristic: StageDemandHeuristic,
                          demand: StageDemand,
                          feature_extractor, stats=None):
    """Adapt feature associations to ``beam_search(rank_step=...)``.

    ``feature_extractor(states, sequences, step)`` is supplied by the runtime
    that owns the learned concept/node representation. UNKNOWN contributes a
    neutral ordering hint, never a safety assertion or value score.  ``stats``
    is an optional diagnostic counter; it never changes the returned hints.
    """
    def rank_step(states, sequences, step):
        feature_sets = feature_extractor(states, sequences, step)
        if len(feature_sets) != len(states):
            raise ValueError(
                "feature_extractor must return one feature set per state"
            )
        priorities = []
        groups = []
        for features in feature_sets:
            result = heuristic.score(features, demand)
            supported = result["status"] == "supported"
            priority = (
                float(result["priority"]) if supported else 0.0
            )
            if stats is not None:
                stats["candidates"] = int(stats.get("candidates", 0)) + 1
                if isinstance(demand, StageDemandAlternatives):
                    stats["proposal_scores"] = (
                        int(stats.get("proposal_scores", 0))
                        + len(demand.demands)
                    )
                if supported:
                    stats["supported"] = (
                        int(stats.get("supported", 0)) + 1
                    )
                    if priority != 0.0:
                        stats["nonzero"] = (
                            int(stats.get("nonzero", 0)) + 1
                        )
                    proposal_target = result.get("proposal_target")
                    if proposal_target is not None:
                        key = "proposal_wins:%r" % (proposal_target,)
                        stats[key] = int(stats.get(key, 0)) + 1
                else:
                    stats["unknown"] = (
                        int(stats.get("unknown", 0)) + 1
                    )
            priorities.append(priority)
            groups.append(
                result.get("proposal_target")
                if supported and priority > 0.0 else None
            )
        priority_array = np.asarray(priorities, dtype=float)
        if isinstance(demand, StageDemandAlternatives):
            return priority_array, tuple(groups)
        return priority_array

    return rank_step
