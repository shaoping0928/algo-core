"""Minimal stage-level value and option executor.

This is an experimental decision loop, not the default planner.  At a new stage
boundary it enumerates the stage-level options supported by the fitted outcome
memory, scores each first option by its realized expected net feedback minus a
terminal penalty and a duration cost, and then executes that option's action
trace frame by frame.  When no option is selected, or the stage changes before
the trace ends, the loop falls back to the ordinary frame planner.

The value is observational: it uses the memory's outcome rows, never a task goal
or a counterfactual.  Options are only started at a stage entry; the boundary
option is recorded in acting-action coordinates. The executor only follows a
trace when its already-acted prefix and pending delayed-action queue match;
otherwise it falls back rather than executing a different option than the one
the memory evaluated.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from stage_online import OnlineStageTracker
from stage_option_search import (
    StageOptionSearch,
    StageOptionSearchCapacityError,
)
from substrate_nodes import LAG, decision_action, latent


def expand_trace(trace):
    actions = []
    for action, count in trace:
        actions.extend([int(action)] * int(count))
    return actions


def executable_option_suffix(trace, acted_prefix, issued_prefix, lag: int):
    """Return the unissued tail if a learned trace matches committed actions.

    Stage memory records actions that acted on transitions. ``issued_prefix``
    is newest-first; its reverse is the already-committed order that will act
    next. Those actions cannot be replaced by the executor.
    """
    lag = int(lag)
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if issued_prefix is None:
        if lag:
            return None
        pending = []
    else:
        values = np.asarray(issued_prefix, dtype=float).reshape(-1)
        if values.size != lag or not np.isfinite(values).all():
            return None
        pending = [
            int(round(float(value))) for value in values[::-1]
        ]
    committed = [
        int(action) for action in acted_prefix
    ] + pending
    actions = expand_trace(trace)
    if (len(actions) < len(committed)
            or actions[:len(committed)] != committed):
        return None
    return actions[len(committed):]


def option_values(result, terminal_weight: float = 1.0,
                  duration_weight: float = 0.0):
    """Expected single-stage option value from a complete enumeration.

    Multi-stage leaves condition each future transition on a later option.
    Combining them by their first option would require an explicit
    continuation policy, so this scorer refuses that result. The final tuple
    field is the combined terminal/censored risk mass.
    """
    if int(getattr(result, "max_transitions", 1)) != 1:
        raise ValueError(
            "first-option scoring requires max_transitions=1"
        )
    if (not bool(getattr(result, "complete", True))
            or getattr(result, "unresolved", ())):
        return []
    groups = {}
    for leaf in result.leaves:
        if not leaf.options:
            continue
        option = leaf.options[0]
        group = groups.setdefault(option, dict(
            mass=0.0, feedback=0.0, risk=0.0, duration=0.0,
        ))
        probability = float(leaf.probability)
        group["mass"] += probability
        feedback = leaf.feedback_mean
        if len(feedback) >= 2:
            net = float(feedback[0]) - float(feedback[1])
        elif feedback:
            net = float(feedback[0])
        else:
            net = 0.0
        group["feedback"] += probability * net
        group["risk"] += probability * (
            1.0 if leaf.outcome in ("terminal", "censored") else 0.0
        )
        group["duration"] += probability * float(leaf.elapsed_mean)
    rows = []
    for option, group in groups.items():
        mass = float(group["mass"])
        if mass <= 0.0:
            continue
        value = (
            group["feedback"] / mass
            - float(terminal_weight) * group["risk"] / mass
            - float(duration_weight) * group["duration"] / mass
        )
        rows.append((
            float(value), mass, option,
            float(group["risk"] / mass),
        ))
    rows.sort(key=lambda row: (-row[0], -row[1], repr(row[2])))
    return rows


class StageExecutor:
    """Select a stage option at boundaries and execute it frame by frame.

    ``max_terminal_mass`` is retained as the public setting name, but gates
    both terminal and censored mass. Any unresolved or incomplete search is
    rejected outright and falls back to the frame planner.
    """

    def __init__(self, agent, horizon: int = 8, lag: int = LAG,
                 max_transitions: int = 1, terminal_weight: float = 1.0,
                 duration_weight: float = 0.0, max_terminal_mass: float = 0.0,
                 max_option_actions: Optional[int] = None,
                 min_context_depth: int = 0, min_support: int = 0,
                 max_hazard: float = 1.0,
                 option_quotient=None):
        experience = getattr(agent, "stage_experience", None)
        memory = getattr(agent, "stage_outcome_memory", None)
        if experience is None or memory is None:
            raise ValueError("the agent has no fitted stage experience/memory")
        if int(max_transitions) != 1:
            raise ValueError(
                "StageExecutor requires max_transitions=1 until a "
                "continuation policy is defined"
            )
        self.agent = agent
        self.horizon = int(horizon)
        self.lag = int(lag)
        self.terminal_weight = float(terminal_weight)
        self.duration_weight = float(duration_weight)
        if not 0.0 <= float(max_terminal_mass) <= 1.0:
            raise ValueError("max_terminal_mass must be in [0, 1]")
        self.max_terminal_mass = float(max_terminal_mass)
        if max_option_actions is not None and int(max_option_actions) < 1:
            raise ValueError("max_option_actions must be positive")
        self.max_option_actions = (
            None if max_option_actions is None else int(max_option_actions)
        )
        if int(min_context_depth) < 0 or int(min_support) < 0:
            raise ValueError("reliability gates must be non-negative")
        if not 0.0 <= float(max_hazard) <= 1.0:
            raise ValueError("max_hazard must be in [0, 1]")
        self.min_context_depth = int(min_context_depth)
        self.min_support = int(min_support)
        self.max_hazard = float(max_hazard)
        self.hazard_aborts = 0
        self.reliability_rejects = 0
        self.search_rejects = 0
        self.alignment_rejects = 0
        self.exhausted_options = 0
        self.search = StageOptionSearch(
            memory, max_transitions=1,
            option_quotient=option_quotient, allow_truncation=False,
        )
        self.option_quotient = option_quotient
        self.tracker = OnlineStageTracker(experience.detector, lag)
        self.active = []
        self.stage_count = 0
        self.selected = 0
        self.fallback = 0
        self.aborted = 0
        self.option_lengths = []
        self.selected_values = []

    def reset(self, episode: Optional[int] = None):
        self.tracker.reset(episode)
        self.active = []
        self.stage_count = 0
        self.selected = 0
        self.fallback = 0
        self.aborted = 0
        self.hazard_aborts = 0
        self.reliability_rejects = 0
        self.search_rejects = 0
        self.alignment_rejects = 0
        self.exhausted_options = 0
        self.option_lengths = []
        self.selected_values = []
        return self

    def observe(self, transition):
        self.tracker.observe_transition(transition)

    def _hazard(self, env, prefix):
        observation = np.asarray(env.obs(), dtype=float)[
            :int(self.agent.obs_dim)
        ]
        pending = np.asarray(prefix, dtype=float).reshape(-1)[::-1]
        encoded = np.concatenate([
            observation, pending,
        ]) / np.asarray(self.agent.scales, dtype=float)
        return float(
            self.agent.pain_signal_batch(encoded.reshape(1, -1))[0]
        )

    def maybe_plan(self, env=None, prefix=None):
        context = self.tracker.context()
        if context is None:
            return
        count = len(context["stages"])
        if count > self.stage_count:
            # A new stage started: any running trace belongs to the old one.
            if self.active:
                self.aborted += 1
                self.active = []
            self.stage_count = count
            if self.active:
                return
            if env is not None and prefix is not None:
                if (self.max_hazard < 1.0
                        and self._hazard(env, prefix) > self.max_hazard):
                    self.hazard_aborts += 1
                    return
            try:
                result = self.search.search(context["history"])
            except StageOptionSearchCapacityError:
                self.search_rejects += 1
                return
            if not result.complete or result.unresolved:
                self.search_rejects += 1
                return
            rows = option_values(
                result, self.terminal_weight, self.duration_weight,
            )
            if not rows:
                return
            current = context.get("current")
            if current is None or not hasattr(current, "action_trace"):
                self.alignment_rejects += 1
                return
            acted_prefix = expand_trace(current.action_trace)
            chosen = None
            for row in rows:
                unsafe_mass = row[3]
                if unsafe_mass > self.max_terminal_mass:
                    continue
                option = row[2]
                leaf = next(
                    (item for item in result.leaves
                     if item.options and item.options[0] == option),
                    None,
                )
                if leaf is None or not leaf.steps:
                    continue
                step = leaf.steps[0]
                if int(step.context_depth) < self.min_context_depth:
                    self.reliability_rejects += 1
                    continue
                if int(step.support) < self.min_support:
                    self.reliability_rejects += 1
                    continue
                if option and option[0] == "option_class":
                    class_id = int(option[1])
                    if self.option_quotient is None:
                        self.alignment_rejects += 1
                        continue
                    option_class = self.option_quotient.classes[class_id]
                    trace = option_class.representative
                    members = option_class.members
                else:
                    trace = option
                    members = (trace,)
                suffix = executable_option_suffix(
                    trace, acted_prefix, prefix, self.lag,
                )
                if suffix is None or not suffix:
                    if suffix is None:
                        self.alignment_rejects += 1
                    else:
                        self.exhausted_options += 1
                    continue
                if any(
                        executable_option_suffix(
                            member, acted_prefix, prefix, self.lag,
                        ) is None
                        for member in members):
                    # A pooled class cannot represent members that disagree
                    # with actions already committed in this episode.
                    self.alignment_rejects += 1
                    continue
                chosen = (row, suffix, len(expand_trace(trace)))
                break
            if chosen is None:
                return
            row, suffix, option_length = chosen
            value = row[0]
            self.active = suffix
            if self.max_option_actions is not None:
                self.active = self.active[:self.max_option_actions]
            self.option_lengths.append(option_length)
            self.selected_values.append(float(value))
            self.selected += 1

    def act(self, env, world, agent, prefix, horizon, **kwargs):
        self.maybe_plan(env, prefix)
        if self.active:
            if (self.max_hazard < 1.0
                    and self._hazard(env, prefix) > self.max_hazard):
                self.hazard_aborts += 1
                self.aborted += 1
                self.active = []
            else:
                return int(self.active.pop(0)), "option"
        self.fallback += 1
        action = decision_action(
            env, world, agent, prefix, horizon, **kwargs,
        )
        return int(action), "fallback"

    def snapshot(self):
        return dict(
            selected=int(self.selected),
            fallback=int(self.fallback),
            aborted=int(self.aborted),
            hazard_aborts=int(self.hazard_aborts),
            reliability_rejects=int(self.reliability_rejects),
            search_rejects=int(self.search_rejects),
            alignment_rejects=int(self.alignment_rejects),
            exhausted_options=int(self.exhausted_options),
            option_length_mean=(
                float(np.mean(self.option_lengths))
                if self.option_lengths else 0.0
            ),
            selected_value_mean=(
                float(np.mean(self.selected_values))
                if self.selected_values else 0.0
            ),
            stage_count=int(self.stage_count),
        )


def run_stage_executor(seed, world, agent, cap: int = 400,
                       horizon: int = 8, lag: int = LAG,
                       env_seed=None, spacing=None,
                       terminal_weight: float = 1.0,
                       duration_weight: float = 0.0,
                       max_transitions: int = 1,
                       max_option_actions=None,
                       min_context_depth: int = 0, min_support: int = 0,
                       max_hazard: float = 1.0,
                       option_quotient=None, metrics_out=None,
                       trace_out=None):
    """Run one episode with the stage-level option executor."""
    if int(cap) < 1:
        raise ValueError("cap must be positive")
    env = latent[int(lag)](
        seed if env_seed is None else int(env_seed),
        spacing=None if spacing is None else float(spacing),
    )
    executor = StageExecutor(
        agent, horizon=int(horizon), lag=int(lag),
        max_transitions=int(max_transitions),
        terminal_weight=float(terminal_weight),
        duration_weight=float(duration_weight),
        max_option_actions=max_option_actions,
        min_context_depth=int(min_context_depth),
        min_support=int(min_support),
        max_hazard=float(max_hazard),
        option_quotient=option_quotient,
    ).reset(0)
    prefix = np.zeros(int(lag), dtype=float)
    recent = [0.0] * int(lag)
    last_action = None
    decisions = []
    while not env.dead and env.t < cap:
        obs = np.asarray(env.obs(), dtype=float)
        action, source = executor.act(
            env, world, agent, prefix, int(horizon),
            lag=int(lag), step_stride=2,
            previous_action=last_action,
        )
        if trace_out is not None:
            pending = np.asarray(prefix, dtype=float).reshape(-1)[::-1]
            encoded = np.concatenate([
                obs[:int(agent.obs_dim)], pending,
            ]) / np.asarray(agent.scales, dtype=float)
            prototype, distance = agent._nearest(encoded)
            resolved = agent.stage_resolver.resolve(
                agent, obs[:int(agent.obs_dim)], pending,
            ) if getattr(agent, "stage_resolver", None) is not None else {}
            trace_out.append(dict(
                source=source,
                distance=float(distance) if prototype is not None else None,
                resolved=str(resolved.get("status")),
                action=int(action),
            ))
        history = np.asarray([float(action)] + recent[:int(lag)], dtype=float)
        env.step(action)
        if trace_out is not None:
            trace_out[-1]["death"] = bool(env.dead)
        feedback = env.feedback()
        next_obs = np.asarray(env.obs(), dtype=float)
        executor.observe(dict(
            obs=obs,
            next_obs=next_obs,
            history=history,
            signals=np.asarray(
                [feedback["dopamine"], feedback["pain"]], dtype=float,
            ),
            terminal=bool(env.dead),
            done=bool(env.dead),
            episode=0,
        ))
        decisions.append(source)
        last_action = int(action)
        prefix = np.concatenate(
            [[float(action)], prefix[:-1]]
        ) if int(lag) else prefix
        recent = [float(action)] + recent[:-1]
    stats = executor.snapshot()
    stats["option_share"] = (
        float(stats["selected"]) / float(len(decisions))
        if decisions else 0.0
    )
    stats["decisions"] = int(len(decisions))
    if metrics_out is not None:
        metrics_out["passes"] = int(getattr(env, "passed", 0))
        metrics_out["stage_executor"] = stats
    return int(env.t), stats
