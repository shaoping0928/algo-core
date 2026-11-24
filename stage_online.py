"""Online stage context and a demand provider for the rank-only search channel.

The tracker consumes only transitions the environment already executed.  It
re-runs the frozen key-signal detector and segmenter on the episode prefix, so
the current stage history, its partial action trace and its boundary are all
causal: no future action or observation enters the context.

The provider turns that context into a rank hint.  Because the future action
option is not known at decision time, it asks the stage outcome memory for an
option-pooled successor distribution, uses the learned outcome readout to pick
a supported non-terminal target, then maps the exact target to the predictive
stage tree's root and conditions the association heuristic on the causal
action prefix.  The demand is scoped to the current partial stage; a new stage
boundary recomputes it, and an unsupported context returns None (UNKNOWN).

The planner still owns feasibility, objective values and the final selection.
This module only produces the intermediate-frontier ordering hint.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from stage_demand_heuristic import StageDemand
from stage_outcome_memory import segment_episode


class OnlineStageTracker:
    """Causal stage context over the transitions executed so far."""

    def __init__(self, detector, lag: int, episode: Optional[int] = None):
        self.detector = detector
        self.lag = int(lag)
        self.reset(episode)

    def reset(self, episode: Optional[int] = None):
        self.episode = episode
        self.observations = []
        self.actions = []
        self.feedback = []
        return self

    def observe_transition(self, transition) -> bool:
        """Append one real transition; return False when it cannot be used."""
        try:
            obs = np.asarray(transition["obs"], dtype=float).reshape(-1)
            next_obs = np.asarray(
                transition["next_obs"], dtype=float
            ).reshape(-1)
            history = np.asarray(
                transition["history"], dtype=float
            ).reshape(-1)
            signals = np.asarray(
                transition["signals"], dtype=float
            ).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return False
        if (not obs.size or next_obs.size != obs.size
                or history.size <= self.lag or signals.size < 2
                or not np.isfinite(obs).all()
                or not np.isfinite(next_obs).all()
                or not np.isfinite(history).all()
                or not np.isfinite(signals[:2]).all()):
            return False
        if not self.observations:
            if self.episode is None:
                self.episode = int(transition.get("episode", 0))
            self.observations.append(obs)
        self.actions.append(int(round(float(history[self.lag]))))
        self.feedback.append(signals[:2])
        self.observations.append(next_obs)
        return True

    def context(self):
        """Current stages and the partial current stage, or None."""
        if not self.actions:
            return None
        observations = np.asarray(self.observations, dtype=float)
        feedback = np.asarray(self.feedback, dtype=float)
        try:
            signals = self.detector.transform_episode(
                observations, feedback=feedback,
            )
            stages = segment_episode(
                episode=0 if self.episode is None else self.episode,
                observations=observations,
                actions=self.actions,
                feedback=feedback,
                key_signals=signals,
                terminal=False,
                censored=True,
                active_features=tuple(() for _ in self.actions),
            )
        except ValueError:
            return None
        if not stages:
            return None
        return dict(
            stages=tuple(stages),
            history=tuple(stage.key for stage in stages),
            current=stages[-1],
            prefix=stages[-1].action_trace,
        )


class OnlineStageDemandProvider:
    """Produce a tree-root StageDemand from the current online context."""

    def __init__(self, agent, feature_source: Optional[str] = None,
                 rank_feature_source: Optional[str] = None):
        experience = getattr(agent, "stage_experience", None)
        if experience is None:
            raise ValueError("agent has no fitted stage experience")
        self.tracker = OnlineStageTracker(experience.detector, agent.lag)
        self.memory = getattr(agent, "stage_outcome_memory", None)
        self.selector = getattr(agent, "stage_demand_selector", None)
        self.tree = getattr(agent, "stage_tree", None)
        self.rank_feature_source = (
            rank_feature_source or "prototype"
        )
        if self.rank_feature_source == "root":
            self.heuristic = getattr(agent, "stage_root_heuristic", None)
        else:
            self.heuristic = getattr(agent, "stage_tree_heuristic", None)
        self.feature_source = (
            getattr(agent, "stage_demand_feature_source", None)
            if feature_source is None else feature_source
        )
        if (self.memory is None or self.selector is None
                or self.tree is None or self.heuristic is None):
            raise ValueError(
                "the agent is missing the fitted stage demand components"
            )
        self.counts = Counter()
        self.last = None

    def reset_episode(self, *_args, **_kwargs):
        self.tracker.reset()
        return self

    def observe_transition(self, transition):
        self.tracker.observe_transition(transition)

    @staticmethod
    def _collapse_key(tree, key):
        chain = tree.chain(key)
        if not chain:
            return (("tree", "base"),)
        return (("tree", str(chain[-1])),)

    def __call__(self, agent, observation, prefix, actions):
        self.counts["requests"] += 1
        context = self.tracker.context()
        if context is None:
            self.counts["no_context"] += 1
            return None
        prediction = self.memory.predict_pooled(context["history"])
        if prediction.get("status") != "supported":
            self.counts["no_prediction"] += 1
            return None
        association_trace = (
            None if self.rank_feature_source == "root"
            else context["prefix"]
        )
        demand, report = self.selector.select(
            prediction,
            context["history"],
            action_trace=None,
            association_action_trace=association_trace,
        )
        if demand is None:
            self.counts["no_demand"] += 1
            return None
        exact_target = demand.target_stage
        chain = self.tree.chain(exact_target)
        if not chain:
            self.counts["no_tree_target"] += 1
            return None
        root_token = ("tree", str(chain[-1]))
        collapsed_history = tuple(
            self._collapse_key(self.tree, key)
            for key in context["history"]
        )
        node_demand = StageDemand(
            target_stage=(root_token,),
            stage_history=collapsed_history,
            action_trace=None,
            association_action_trace=association_trace,
            transition_probability=demand.transition_probability,
            transition_support=demand.transition_support,
            transition_resolution=demand.transition_resolution,
        )
        self.counts["supported"] += 1
        self.last = dict(
            target=(root_token,),
            history=collapsed_history,
            prefix=association_trace,
            prediction_support=int(prediction.get("support", 0)),
        )
        return node_demand, self.heuristic

    def snapshot(self):
        return dict(
            counts={
                str(key): int(value)
                for key, value in sorted(self.counts.items())
            },
            feature_source=self.feature_source,
            rank_feature_source=self.rank_feature_source,
        )
