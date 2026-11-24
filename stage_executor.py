"""Minimal stage-level value and option executor.

This is an experimental decision loop, not the default planner.  At a new stage
boundary it enumerates the stage-level options supported by the fitted outcome
memory, scores each first option by its realized expected net feedback minus a
terminal penalty and a duration cost, and then executes that option's action
trace frame by frame.  When no option is selected, or the stage changes before
the trace ends, the loop falls back to the ordinary frame planner.

The value is observational: it uses the memory's outcome rows, never a task goal
or a counterfactual.  Options are only started at a stage entry; the boundary
transition's own action is treated as the first action of the stage, so the
executed trace drops that one action.  An unknown or unsupported stage falls
back instead of fabricating a plan.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from stage_online import OnlineStageTracker
from stage_option_search import StageOptionSearch
from substrate_nodes import LAG, decision_action, latent


def expand_trace(trace):
    actions = []
    for action, count in trace:
        actions.extend([int(action)] * int(count))
    return actions


def option_values(result, terminal_weight: float = 1.0,
                  duration_weight: float = 0.0):
    """Expected first-option value from complete option-search leaves."""
    groups = {}
    for leaf in result.leaves:
        if not leaf.options:
            continue
        option = leaf.options[0]
        group = groups.setdefault(option, dict(
            mass=0.0, feedback=0.0, terminal=0.0, duration=0.0,
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
        group["terminal"] += probability * (
            1.0 if leaf.outcome == "terminal" else 0.0
        )
        group["duration"] += probability * float(leaf.elapsed_mean)
    rows = []
    for option, group in groups.items():
        mass = float(group["mass"])
        if mass <= 0.0:
            continue
        value = (
            group["feedback"] / mass
            - float(terminal_weight) * group["terminal"] / mass
            - float(duration_weight) * group["duration"] / mass
        )
        rows.append((
            float(value), mass, option,
            float(group["terminal"] / mass),
        ))
    rows.sort(key=lambda row: (-row[0], -row[1], repr(row[2])))
    return rows


class StageExecutor:
    """Select a stage option at boundaries and execute it frame by frame."""

    def __init__(self, agent, horizon: int = 8, lag: int = LAG,
                 max_transitions: int = 1, terminal_weight: float = 1.0,
                 duration_weight: float = 0.0, max_terminal_mass: float = 0.0,
                 max_option_actions: Optional[int] = None,
                 option_quotient=None):
        experience = getattr(agent, "stage_experience", None)
        memory = getattr(agent, "stage_outcome_memory", None)
        if experience is None or memory is None:
            raise ValueError("the agent has no fitted stage experience/memory")
        if int(max_transitions) < 1:
            raise ValueError("max_transitions must be positive")
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
        self.search = StageOptionSearch(
            memory, max_transitions=int(max_transitions),
            option_quotient=option_quotient, allow_truncation=True,
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
        self.option_lengths = []
        self.selected_values = []
        return self

    def observe(self, transition):
        self.tracker.observe_transition(transition)

    def maybe_plan(self):
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
            result = self.search.search(context["history"])
            rows = option_values(
                result, self.terminal_weight, self.duration_weight,
            )
            if not rows:
                return
            chosen = None
            for row in rows:
                if row[3] <= self.max_terminal_mass:
                    chosen = row
                    break
            if chosen is None:
                return
            value, _mass, option, _terminal = chosen
            if option and option[0] == "option_class":
                class_id = int(option[1])
                if self.option_quotient is None:
                    return
                trace = self.option_quotient.classes[class_id].representative
            else:
                trace = option
            actions = expand_trace(trace)
            if len(actions) <= 1:
                return
            # The boundary transition's own action already started this stage.
            self.active = actions[1:]
            if self.max_option_actions is not None:
                self.active = self.active[:self.max_option_actions]
            self.option_lengths.append(len(actions))
            self.selected_values.append(float(value))
            self.selected += 1

    def act(self, env, world, agent, prefix, horizon, **kwargs):
        self.maybe_plan()
        if self.active:
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
                       option_quotient=None, metrics_out=None):
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
        history = np.asarray([float(action)] + recent[:int(lag)], dtype=float)
        env.step(action)
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
