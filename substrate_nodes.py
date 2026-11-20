"""Node-centric planning on Flappy using only signals, not task rules.

The runtime receives:

    observation, action, valence (dopamine/pain), episode boundary

It never reads task geometry, a collision window, or a `terminal` risk label.
A learned observation model predicts candidate trajectories; the node runtime
supplies the learned valence of their endpoints.
The node graph itself is trained from real transitions with TD(lambda) credit
assignment and action-conditioned edges.
"""
import argparse
import json
import os
import sys
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)

import numpy as np

from eval_protocol import (
    cluster_summary,
    group_summaries,
    parse_seed_spec,
)
from latent_world import SEEDS, latent
from node_features import FeatureRuleBank
from node_runtime import NodeRuntime, future_pain_labels, future_return_labels
from substrate_generic import (
    LinearObservationModel,
    collect,
    history_features,
)
from trajectory_search import exhaustive_search

LAG = 3
DEFAULT_JOBS = min(8, os.cpu_count() or 1)
FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()


def _train(transitions, seed, epochs, lag, max_nodes, birth_threshold,
           component_kinds, component_min_count, max_components,
           component_weight, component_gating, memory_weight,
           association_rate, induce_features, precursor_labels,
           pain_horizon, voi_target, feature_value_weight,
           value_horizon=None, ood_trust_scale=0.0, obs_trust=1.0,
           feature_action_prior=2.0, feature_value_prior=4.0,
           value_features=True, value_label_quantile=0.8,
           subgraph_cells=False, predicate_readouts=False):
    """Fit the world model and the node runtime on a transition stream."""
    obs_dim = len(transitions[0]["obs"])
    action_dim = lag + 1
    world = LinearObservationModel(obs_dim, action_dim).fit(
        [row["obs"] for row in transitions],
        [row["history"] for row in transitions],
        [row["next_obs"] for row in transitions],
    )
    observations = np.asarray(
        [row["obs"] for row in transitions], dtype=float
    )
    # Scale only from observed state spread; no task-specific constants.
    observation_scale = observations.std(axis=0)
    observation_scale = np.where(
        observation_scale > 1e-8, observation_scale, 1.0
    )
    scales = np.concatenate([observation_scale, np.ones(lag, dtype=float)])
    agent = NodeRuntime(
        obs_dim=obs_dim, action_space=(0, 1), lag=lag,
        birth_threshold=birth_threshold, max_nodes=max_nodes,
        exploration=0.0, seed=seed, scales=scales,
        component_kinds=component_kinds,
        component_min_count=component_min_count,
        max_components=max_components,
        component_weight=component_weight,
        component_gating=component_gating,
        memory_weight=memory_weight,
        association_rate=association_rate,
        pain_horizon=pain_horizon,
        voi_target=voi_target,
        feature_value_weight=feature_value_weight,
        value_horizon=(value_horizon or None),
        ood_trust_scale=ood_trust_scale,
        obs_trust=obs_trust,
        feature_action_prior=feature_action_prior,
        feature_value_prior=feature_value_prior,
        subgraph_cells=subgraph_cells,
        predicate_readouts=predicate_readouts,
    )
    episodes = {}
    for row in transitions:
        episodes.setdefault(row["episode"], []).append(row)
    ordered = [episodes[key] for key in sorted(episodes)]

    # Rules are induced after the first pass, then materialized as feature
    # nodes.  Precursor labels let a rule describe the *current* state and
    # predict pain within its own window, so it can fire while the action
    # can still change the outcome.  A short window (default `lag + 1`)
    # detects imminent death; a longer optional window gives the approach
    # phase positive coverage.  Both banks are installed and every root
    # trains with its own window's labels.  Later epochs train transitions,
    # action edges and memories with those feature nodes active, so features
    # take part in learning rather than only in the final readout.
    long_window = None
    if precursor_labels:
        window = int(pain_horizon) if pain_horizon else lag + 1
        long_window = window if window != lag + 1 else None
        states = np.asarray(
            [row["obs"] for row in transitions], dtype=float
        ) / scales[:obs_dim]
        pain = np.asarray(
            future_pain_labels(transitions, window), dtype=float
        )
        if long_window is not None:
            pain_short = np.asarray(
                future_pain_labels(transitions, lag + 1), dtype=float
            )
        else:
            pain_short = pain
    else:
        # Pre-precursor behavior: rules on the entered state with its own
        # pain label.
        long_window = None
        pain_short = None
        states = np.asarray(
            [row["next_obs"] for row in transitions], dtype=float
        ) / scales[:obs_dim]
        pain = np.asarray(
            [row["signals"][1] > 0.5 for row in transitions], dtype=float
        )
    groups = np.asarray([row["episode"] for row in transitions])
    order = np.unique(groups)
    # Interleaved episode split: every fifth episode is held out.  A contiguous
    # tail split can leave rare causes entirely in train, so their rules can
    # never be verified.
    episode_index = np.searchsorted(order, groups)
    test = (episode_index % 5 == 4)
    train = ~test

    for epoch in range(epochs):
        for episode in ordered:
            agent.learn_episode(episode)
        if induce_features and epoch == 0:
            primary = pain_short if long_window is not None else pain
            agent.induce_rules(
                states[train], primary[train], states[test], primary[test]
            )
            agent.install_feature_rules()
            if long_window is not None:
                bank = FeatureRuleBank()
                bank.induce(
                    states[train], pain[train], states[test], pain[test]
                )
                agent.rules_long = bank
                agent.install_feature_rules(bank.rules, horizon=long_window)
            # Value-channel bank: the same grammar, but the labels are the
            # top of the empirical discounted-return distribution instead
            # of pain.  The selected predicates tile the states whose value
            # cells carry data - regions the pain labels never cover - so
            # the sequence-level value readout gains its own resolution.
            # Same transitions, no extra collection.
            if value_features:
                returns = np.asarray(
                    future_return_labels(transitions, agent.gamma),
                    dtype=float,
                )
                cut = float(np.quantile(returns, value_label_quantile))
                high = (returns > cut).astype(float)
                if high.sum() >= 2:
                    if not precursor_labels:
                        value_states = np.asarray(
                            [row["obs"] for row in transitions], dtype=float
                        ) / scales[:obs_dim]
                    else:
                        value_states = states
                    value_bank = FeatureRuleBank()
                    value_bank.induce(
                        value_states[train], high[train],
                        value_states[test], high[test],
                    )
                    if value_bank.rules:
                        agent.rules_value = value_bank
                        agent.install_feature_rules(
                            value_bank.rules, channel="value"
                        )
    return world, agent


def collect_planner(seed, world, agent, episodes=2, cap=1200, horizon=10,
                    lag=LAG, step_stride=2, stride_includes_lag=False,
                    offset=50000):
    """Record the trajectories the planner actually executes.

    These are real transitions with real world feedback, in the same schema
    as `collect`; no counterfactual prediction enters the stream.  The world
    seeds are offset from the behavior stream so episode ids cannot collide.
    """
    transitions = []
    frames = []
    for episode in range(int(episodes)):
        env_seed = int(seed) * 10000 + int(offset) + episode
        length, _actions, rows = evaluate(
            seed, world, agent, cap=cap, horizon=horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
            env_seed=env_seed, record=True,
            episode=int(offset) + episode,
        )
        transitions.extend(rows)
        frames.append(int(length))
    return transitions, frames


def fit(seed, steps=5000, coverage=0.1, max_nodes=5000,
        birth_threshold=0.45, epochs=5, lag=LAG,
        component_kinds=(), component_min_count=8,
        max_components=128, component_weight=0.5,
        component_gating=False, memory_weight=0.1,
        association_rate=0.0, induce_features=True,
        precursor_labels=True, pain_horizon=None,
        voi_fraction=0.0, voi_target=3, feature_value_weight=0.02,
        dagger_rounds=0, dagger_episodes=2, dagger_cap=1200,
        dagger_offset=50000, dagger_horizon=10, step_stride=2,
        stride_includes_lag=False, value_horizon=0,
        ood_trust_scale=0.0, obs_trust=1.0,
        feature_action_prior=2.0, feature_value_prior=4.0,
        value_features=True, value_label_quantile=0.8,
        subgraph_cells=False, predicate_readouts=False):
    """Collect transitions and train the node runtime on them.

    With `voi_fraction > 0` collection runs in two stages: the first stage
    bootstraps the runtime with the plain behavior policy, the second stage
    asks that bootstrap agent where its (feature, action) support is
    deficient and overrides at most `voi_fraction * steps` actions with the
    less-supported one.  The override is additive - the behavior sampler
    still runs the rest of the time - and everything is retrained on the
    combined stream.

    With `dagger_rounds > 0` the planner's own executed trajectories are
    appended after each round and the runtime is retrained from scratch on
    the combined real-transition stream.  Only executed actions and their
    observed feedback are recorded.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if int(lag) < 0:
        raise ValueError("lag must be non-negative")
    if int(component_min_count) < 1:
        raise ValueError("component_min_count must be positive")
    if int(max_components) < 0:
        raise ValueError("max_components must be non-negative")
    if not 0.0 <= float(voi_fraction) < 1.0:
        raise ValueError("voi_fraction must be in [0, 1)")
    if int(voi_target) < 1:
        raise ValueError("voi_target must be positive")
    if float(feature_value_weight) < 0.0:
        raise ValueError("feature_value_weight must be non-negative")
    if int(dagger_rounds) < 0:
        raise ValueError("dagger_rounds must be non-negative")
    if int(dagger_episodes) < 0:
        raise ValueError("dagger_episodes must be non-negative")
    if dagger_rounds and int(dagger_episodes) < 1:
        raise ValueError("dagger_episodes must be positive")
    if int(dagger_cap) < 1:
        raise ValueError("dagger_cap must be positive")
    if int(dagger_offset) < 0:
        raise ValueError("dagger_offset must be non-negative")
    if int(dagger_horizon) < 1:
        raise ValueError("dagger_horizon must be positive")
    if int(step_stride) < 1:
        raise ValueError("step_stride must be positive")
    if int(value_horizon) < 0:
        raise ValueError("value_horizon must be non-negative")
    if float(ood_trust_scale) < 0.0:
        raise ValueError("ood_trust_scale must be non-negative")
    if not 0.0 <= float(obs_trust) <= 1.0:
        raise ValueError("obs_trust must be in [0, 1]")
    if float(feature_action_prior) < 0.0:
        raise ValueError("feature_action_prior must be non-negative")
    if float(feature_value_prior) < 0.0:
        raise ValueError("feature_value_prior must be non-negative")
    if not 0.0 < float(value_label_quantile) < 1.0:
        raise ValueError("value_label_quantile must be in (0, 1)")
    lag = int(lag)
    voi_fraction = float(voi_fraction)
    if voi_fraction > 0.0:
        if steps < 2:
            raise ValueError("two-stage collection needs at least 2 steps")
        steps1 = max(1, int(round(steps * (1.0 - voi_fraction))))
        stage1, _counts = collect(seed, steps1, coverage=coverage, lag=lag)
        if not stage1:
            raise ValueError("collection produced no transitions")
        _bootstrap_world, guide_agent = _train(
            stage1, seed=seed, epochs=epochs, lag=lag,
            max_nodes=max_nodes, birth_threshold=birth_threshold,
            component_kinds=component_kinds,
            component_min_count=component_min_count,
            max_components=max_components,
            component_weight=component_weight,
            component_gating=component_gating,
            memory_weight=memory_weight,
            association_rate=association_rate,
            induce_features=induce_features,
            precursor_labels=precursor_labels,
            pain_horizon=pain_horizon,
            voi_target=voi_target,
            feature_value_weight=feature_value_weight,
        )
        propensities = []
        stage2, _counts2 = collect(
            seed, steps - steps1, coverage=coverage, lag=lag,
            guide=guide_agent.counterfactual_action,
            budget=int(round(voi_fraction * steps)),
            episode_start=max(int(row["episode"]) for row in stage1) + 1,
            propensities=propensities,
        )
        transitions = list(stage1) + list(stage2)
    else:
        transitions, _counts = collect(seed, steps, coverage=coverage, lag=lag)
    if not transitions:
        raise ValueError("collection produced no transitions")
    train_kwargs = dict(
        seed=seed, epochs=epochs, lag=lag,
        max_nodes=max_nodes, birth_threshold=birth_threshold,
        component_kinds=component_kinds,
        component_min_count=component_min_count,
        max_components=max_components,
        component_weight=component_weight,
        component_gating=component_gating,
        memory_weight=memory_weight,
        association_rate=association_rate,
        induce_features=induce_features,
        precursor_labels=precursor_labels,
        pain_horizon=pain_horizon,
        voi_target=voi_target,
        feature_value_weight=feature_value_weight,
        value_horizon=value_horizon,
        ood_trust_scale=ood_trust_scale,
        obs_trust=obs_trust,
        feature_action_prior=feature_action_prior,
        feature_value_prior=feature_value_prior,
        value_features=value_features,
        value_label_quantile=value_label_quantile,
        subgraph_cells=subgraph_cells,
        predicate_readouts=predicate_readouts,
    )
    world, agent = _train(transitions, **train_kwargs)
    dagger_log = []
    for _round in range(int(dagger_rounds)):
        planner, _frames = collect_planner(
            seed, world, agent, episodes=dagger_episodes,
            cap=dagger_cap, horizon=dagger_horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
            # Each round must get fresh episode ids: merging two rollouts
            # under one id would make the backward pass treat the second as
            # a continuation of the first.
            offset=dagger_offset + _round * int(dagger_episodes),
        )
        if not planner:
            break
        transitions = list(transitions) + planner
        dagger_log.append(dict(
            round=_round + 1,
            planner_steps=len(planner),
            planner_frames=list(_frames),
            total_steps=len(transitions),
        ))
        world, agent = _train(transitions, **train_kwargs)
    agent.dagger_log = dagger_log
    return world, agent


def policy_result(env, world, agent, prefix, horizon, lag=LAG,
                  step_stride=2, stride_includes_lag=False):
    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)
    lag = int(lag)

    def predict_batch(states, sequences, step):
        history = history_features(
            sequences, step + 1, pending_prefix, action_dim=action_dim
        )
        return world.predict(states, history)

    def evaluate_step(states, sequences, step):
        # `lag` is the first predicted state whose acting action comes from
        # the plan rather than the pending prefix, so a plain stride can step
        # over the first point the plan can actually influence.
        if step % step_stride != 0 and not (
            stride_includes_lag and step == lag
        ):
            return (
                np.zeros((len(states), 2)),
                np.isfinite(states).all(axis=1),
            )
        history = history_features(
            sequences, step + 1, pending_prefix, action_dim=action_dim
        )
        pending = history[:, :agent.lag][:, ::-1]
        encoded = np.concatenate([states, pending], axis=1) / agent.scales
        risk = agent.pain_signal_batch(encoded)
        return (
            np.column_stack([risk, np.zeros(len(states))]),
            np.isfinite(encoded).all(axis=1),
        )

    def evaluate_terminal(states, sequences):
        history = history_features(
            sequences, horizon, pending_prefix, action_dim=action_dim
        )
        # `history` is newest-first; the node state stores pending actions
        # oldest-first and only the last `lag` actions remain pending.
        pending = history[:, :agent.lag][:, ::-1]
        encoded = np.concatenate([states, pending], axis=1) / agent.scales
        pain, dopamine = agent.valence_state_batch(encoded)
        mask_table = agent._feature_mask_table(encoded[:, :agent.obs_dim])
        if getattr(agent, "predicate_readouts", False):
            # D3: the terminal hazard comes from the predicates only; the
            # prototype valence no longer bounds it.
            risk = agent.pain_signal_batch(encoded, mask_table)
        else:
            risk = np.maximum(
                pain, agent.pain_signal_batch(encoded, mask_table)
            )
        # Scalarized objective: accumulated hazard minus the sequence-level
        # value of the endpoint, so a plan that accepts local risk in
        # exchange for a long-survival alignment can win.
        value = agent.feature_value_batch(encoded, mask_table)
        cost = risk - agent.feature_value_weight * value
        return np.column_stack([cost, -dopamine]), np.isfinite(encoded).all(
            axis=1
        )

    result = exhaustive_search(
        initial_state=initial,
        action_space=(0, 1),
        horizon=horizon,
        transition_batch=predict_batch,
        evaluate_step=evaluate_step,
        evaluate_terminal=evaluate_terminal,
        batch_size=2 ** horizon,
        score_combine=("sum", "sum"),
    )
    return result


def policy_action(env, world, agent, prefix, horizon, lag=LAG,
                  step_stride=2, stride_includes_lag=False):
    return int(policy_result(
        env, world, agent, prefix, horizon, lag=lag,
        step_stride=step_stride,
        stride_includes_lag=stride_includes_lag,
    ).actions[0])


def cell_q_action(env, world, agent, prefix, lag=LAG):
    """Decide without search, from the learned (feature, action) cells.

    The pending actions are already issued and fixed, so the world model is
    rolled forward `lag` steps on them alone.  The candidate action issued
    now acts `lag` steps from now, on the rolled state; both candidates are
    scored there with the same cells the search reads:

        score(a) = feature hazard(s, a) - w * sequence value(s, a)

    and the lower score wins.  This is the cell-Q-only ablation: it measures
    what the MPC shell adds beyond reading the learned values directly.
    """
    if int(lag) < 0:
        raise ValueError("lag must be non-negative")
    lag = int(lag)
    state = np.asarray(env.obs(), dtype=float).reshape(1, -1)
    pending = list(env.pending()[:lag])
    if len(pending) < lag:
        pending = [0.0] * (lag - len(pending)) + pending
    prefix = np.asarray(prefix, dtype=float).reshape(-1)
    for step in range(lag):
        applied = float(pending[step])
        columns = [applied]
        for offset in range(1, lag + 1):
            index = step - offset
            columns.append(
                float(pending[index]) if index >= 0
                else float(prefix[-index - 1])
            )
        history = np.asarray([columns], dtype=float)
        state = np.asarray(world.predict(state, history), dtype=float)
    scores = {}
    for action in (0, 1):
        columns = [float(action)] + [float(item) for item in pending[1:]]
        encoded = np.concatenate(
            [state, np.asarray([columns], dtype=float)], axis=1
        ) / agent.scales
        risk = float(agent.feature_risk_batch(encoded)[0])
        value = float(agent.feature_value_batch(encoded)[0])
        scores[action] = risk - agent.feature_value_weight * value
    return int(min(scores, key=lambda key: (scores[key], key)))


def decision_action(env, world, agent, prefix, horizon, lag=LAG,
                    step_stride=2, stride_includes_lag=False,
                    decision="search"):
    """Dispatch one decision to the searched or the cell-Q-only policy."""
    if decision == "search":
        return policy_action(
            env, world, agent, prefix, horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
        )
    if decision == "cellq":
        return cell_q_action(env, world, agent, prefix, lag=lag)
    raise ValueError("unknown decision mode %r" % decision)


def plan_score(initial_state, world, agent, prefix, plan, horizon, lag=LAG,
               step_stride=2, stride_includes_lag=False):
    """Score one fixed plan with the same objective as `policy_result`.

    Reproduces the `exhaustive_search` scoring under
    ``score_combine=("sum", "sum")``: column 0 is the accumulated
    `pain_signal` hazard over the strided predicted states plus the
    terminal risk, minus the sequence-level value of the endpoint; column 1
    is the negative terminal dopamine.  Used by diagnostics that compare
    fixed plans against the searched plan, and later by plan-level
    posteriors.
    """
    initial = np.asarray(initial_state, dtype=float).reshape(1, -1)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)
    lag = int(lag)
    plan = np.asarray(plan, dtype=float).reshape(1, -1)
    states = initial.copy()
    total = 0.0
    for step in range(horizon):
        history = history_features(
            plan, step + 1, pending_prefix, action_dim=action_dim
        )
        states = np.asarray(world.predict(states, history), dtype=float)
        if step % step_stride == 0 or (
            stride_includes_lag and step == lag
        ):
            pending = history[:, :agent.lag][:, ::-1]
            encoded = np.concatenate([states, pending], axis=1) / agent.scales
            if not np.isfinite(encoded).all():
                return (float("inf"), float("inf"))
            total += float(agent.pain_signal_batch(encoded)[0])
    history = history_features(
        plan, horizon, pending_prefix, action_dim=action_dim
    )
    pending = history[:, :agent.lag][:, ::-1]
    encoded = np.concatenate([states, pending], axis=1) / agent.scales
    if not np.isfinite(encoded).all():
        return (float("inf"), float("inf"))
    pain, dopamine = agent.valence_state_batch(encoded)
    mask_table = agent._feature_mask_table(encoded[:, :agent.obs_dim])
    if getattr(agent, "predicate_readouts", False):
        terminal = float(agent.pain_signal_batch(encoded, mask_table)[0])
    else:
        terminal = max(
            float(pain[0]),
            float(agent.pain_signal_batch(encoded, mask_table)[0]),
        )
    value = float(agent.feature_value_batch(encoded, mask_table)[0])
    total += terminal - agent.feature_value_weight * value
    return (total, -float(dopamine[0]))


def evaluate(seed, world, agent, cap=6000, horizon=10, lag=LAG,
             step_stride=2, stride_includes_lag=False, env_seed=None,
             record=False, episode=0, decision="search"):
    """Run one deterministic evaluation episode.

    ``env_seed`` selects the world instance; when omitted the historical
    ``seed`` world is used.  With ``record=True`` the real transitions the
    planner actually executed are returned in the same schema as ``collect``
    (``obs``/``next_obs``/``history``/``signals``/``done``/``episode``), which
    is what the closed-loop (DAgger) path consumes.  No counterfactual
    prediction is recorded.
    """
    if cap < 1:
        raise ValueError("cap must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if step_stride < 1:
        raise ValueError("step_stride must be positive")
    lag = int(lag)
    env = latent[lag](seed if env_seed is None else int(env_seed))
    prefix = np.zeros(lag, dtype=float)
    recent = [0.0] * lag
    actions = []
    transitions = []
    while not env.dead and env.t < cap:
        obs = np.asarray(env.obs(), dtype=float)
        action = decision_action(
            env, world, agent, prefix, horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
            decision=decision,
        )
        history = np.asarray([float(action)] + recent[:lag], dtype=float)
        actions.append(action)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(action)
        if record:
            feedback = env.feedback()
            transitions.append(dict(
                obs=obs,
                next_obs=np.asarray(env.obs(), dtype=float),
                history=history,
                signals=np.asarray(
                    [feedback["dopamine"], feedback["pain"]], dtype=float
                ),
                terminal=bool(env.dead),
                done=bool(env.dead),
                episode=int(episode),
            ))
        recent = [float(action)] + recent[:-1]
    return env.t, actions, transitions


def run_seed(task):
    """Train one model and score it on the requested evaluation worlds.

    ``task`` is a plain dict so the worker payload stays picklable and new
    protocol fields do not require positional shuffling.  ``eval_seeds`` is
    explicit: when it is empty the historical self-evaluation is kept, with a
    warning printed by ``main``.  Each trained model is scored on every
    evaluation seed and every requested episode of that seed.
    """
    seed = int(task["seed"])
    steps = int(task.get("steps", 5000))
    coverage = float(task.get("coverage", 0.1))
    cap = int(task.get("cap", 6000))
    max_nodes = int(task.get("max_nodes", 5000))
    birth_threshold = float(task.get("birth_threshold", 0.45))
    horizon = int(task.get("horizon", 10))
    epochs = int(task.get("epochs", 5))
    lag = int(task.get("lag", LAG))
    component_kinds = tuple(task.get("component_kinds", ()))
    component_min_count = int(task.get("component_min_count", 8))
    max_components = int(task.get("max_components", 128))
    component_weight = float(task.get("component_weight", 0.5))
    component_gating = bool(task.get("component_gating", False))
    memory_weight = float(task.get("memory_weight", 0.1))
    association_rate = float(task.get("association_rate", 0.0))
    induce_features = bool(task.get("induce_features", True))
    precursor_labels = bool(task.get("precursor_labels", True))
    pain_horizon = int(task.get("pain_horizon", 0))
    voi_fraction = float(task.get("voi_fraction", 0.0))
    voi_target = int(task.get("voi_target", 3))
    feature_value_weight = float(task.get("feature_value_weight", 0.02))
    step_stride = int(task.get("step_stride", 2))
    stride_includes_lag = bool(task.get("stride_includes_lag", False))
    decision = str(task.get("decision", "search"))
    ood_trust_scale = float(task.get("ood_trust_scale", 0.0))
    obs_trust = float(task.get("obs_trust", 1.0))
    feature_action_prior = float(task.get("feature_action_prior", 2.0))
    feature_value_prior = float(task.get("feature_value_prior", 4.0))
    value_features = bool(task.get("value_features", True))
    value_label_quantile = float(task.get("value_label_quantile", 0.8))
    subgraph_cells = bool(task.get("subgraph_cells", False))
    predicate_readouts = bool(task.get("predicate_readouts", False))
    eval_episodes = int(task.get("eval_episodes", 1))
    eval_seeds = list(task.get("eval_seeds", ()))
    dagger_rounds = int(task.get("dagger_rounds", 0))
    dagger_episodes = int(task.get("dagger_episodes", 2))
    dagger_cap = int(task.get("dagger_cap", 1200))
    dagger_offset = int(task.get("dagger_offset", 50000))
    dagger_horizon = int(task.get("dagger_horizon", horizon))

    value_horizon = int(task.get("value_horizon", 0))

    world, agent = fit(
        seed, steps=steps, coverage=coverage,
        max_nodes=max_nodes, birth_threshold=birth_threshold,
        epochs=epochs, lag=lag,
        component_kinds=component_kinds,
        component_min_count=component_min_count,
        max_components=max_components,
        component_weight=component_weight,
        component_gating=component_gating,
        memory_weight=memory_weight,
        association_rate=association_rate,
        induce_features=induce_features,
        precursor_labels=precursor_labels,
        pain_horizon=(pain_horizon or None),
        voi_fraction=voi_fraction,
        voi_target=voi_target,
        feature_value_weight=feature_value_weight,
        dagger_rounds=dagger_rounds,
        dagger_episodes=dagger_episodes,
        dagger_cap=dagger_cap,
        dagger_offset=dagger_offset,
        dagger_horizon=dagger_horizon,
        step_stride=step_stride,
        stride_includes_lag=stride_includes_lag,
        value_horizon=value_horizon,
        ood_trust_scale=ood_trust_scale,
        obs_trust=obs_trust,
        feature_action_prior=feature_action_prior,
        feature_value_prior=feature_value_prior,
        value_features=value_features,
        value_label_quantile=value_label_quantile,
        subgraph_cells=subgraph_cells,
        predicate_readouts=predicate_readouts,
    )
    explicit_eval = bool(eval_seeds)
    if not explicit_eval:
        eval_seeds = [seed]
    rows = []
    for eval_seed in eval_seeds:
        for episode in range(eval_episodes):
            if explicit_eval:
                # Match `collect`'s world-naming scheme so evaluation worlds
                # are reproducible and disjoint from the training stream.
                env_seed = int(eval_seed) * 10000 + episode
            elif episode == 0:
                # Historical self-evaluation world.
                env_seed = int(eval_seed)
            else:
                env_seed = int(eval_seed) * 10000 + episode
            frames, actions, _ = evaluate(
                seed, world, agent, cap=cap, horizon=horizon, lag=lag,
                step_stride=step_stride,
                stride_includes_lag=stride_includes_lag,
                env_seed=env_seed,
                decision=decision,
            )
            rows.append(dict(
                train_seed=seed,
                eval_seed=int(eval_seed),
                episode=episode,
                env_seed=int(env_seed),
                frames=int(frames),
                action_rate=(
                    float(np.mean(actions)) if actions else 0.0
                ),
            ))
    return dict(
        seed=seed,
        frames=rows[0]["frames"],
        action_rate=rows[0]["action_rate"],
        evals=rows,
        graph=agent.snapshot(),
        dagger=getattr(agent, "dagger_log", []),
        top_nodes=agent.top_nodes(5),
        top_edges=agent.top_edges(5),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=6000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--birth-threshold", type=float, default=0.45)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lag", type=int, default=LAG)
    parser.add_argument("--components", default="",
                        help="comma-separated runtime component kinds")
    parser.add_argument("--component-min-count", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--component-weight", type=float, default=0.5)
    parser.add_argument("--component-gating", action="store_true")
    parser.add_argument("--memory-weight", type=float, default=0.1)
    parser.add_argument("--association-rate", type=float, default=0.0)
    parser.add_argument("--no-features", action="store_true",
                        help="disable the compositional feature layer")
    parser.add_argument("--instant-labels", action="store_true",
                        help="induce rules on the entered state with its own "
                             "pain label instead of precursor labels")
    parser.add_argument("--pain-horizon", type=int, default=0,
                        help="precursor label window in entered states; "
                             "0 uses the default lag + 1")
    parser.add_argument("--voi-fraction", type=float, default=0.0,
                        help="fraction of collection steps guided by the "
                             "counterfactual budget (two-stage collect)")
    parser.add_argument("--voi-target", type=int, default=3,
                        help="minimum samples per (feature, action) cell")
    parser.add_argument("--feature-value-weight", type=float, default=0.02,
                        help="how much sequence-level value trades against "
                             "accumulated hazard in the search objective")
    parser.add_argument("--eval-seeds", default="",
                        help="held-out seed spec, e.g. 120-139; empty keeps "
                             "the legacy self-evaluation")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="episodes per evaluation seed")
    parser.add_argument("--step-stride", type=int, default=2,
                        help="query every Nth predicted state in search")
    parser.add_argument("--stride-includes-lag", action="store_true",
                        help="also query step == lag, the first predicted "
                             "state the plan action can affect")
    parser.add_argument("--decision", choices=("search", "cellq"),
                        default="search",
                        help="search = full MPC planning; cellq = read the "
                             "learned (feature, action) cells directly")
    parser.add_argument("--value-horizon", type=int, default=0,
                        help="k-step bootstrap target for the sequence-value "
                             "cells; 0 keeps the full-episode MC return")
    parser.add_argument("--ood-trust-scale", type=float, default=0.0,
                        help="shrink prototype readouts toward the population "
                             "baseline beyond this distance (0 = off; the "
                             "birth threshold is a natural setting)")
    parser.add_argument("--obs-trust", type=float, default=1.0,
                        help="weight of the observation-anchored prototype/"
                             "memory channel against induced feature channels")
    parser.add_argument("--feature-action-prior", type=float, default=2.0,
                        help="pseudo-counts shrinking pain cells toward the "
                             "feature scalar; lower = more trust in verified "
                             "action-conditioned cells")
    parser.add_argument("--feature-value-prior", type=float, default=4.0,
                        help="pseudo-counts shrinking value cells toward the "
                             "baseline; lower = more trust in verified "
                             "success cells")
    parser.add_argument("--no-value-features", action="store_true",
                        help="keep the value readout on the pain-coverage "
                             "roots instead of adding a dedicated "
                             "high-return bank")
    parser.add_argument("--value-quantile", type=float, default=0.8,
                        help="return-distribution quantile that defines the "
                             "high-return labels for value-channel "
                             "induction")
    parser.add_argument("--subgraph-cells", action="store_true",
                        help="train and read feature statistics on rule "
                             "branches (most specific active member wins); "
                             "measured inert on survival, off by default")
    parser.add_argument("--predicate-readouts", action="store_true",
                        help="drop the prototype channel from the risk "
                             "readout; prototypes stay as event memory")
    parser.add_argument("--report", default="",
                        help="write per-episode rows and summaries as JSON")
    parser.add_argument("--dagger-rounds", type=int, default=0,
                        help="closed-loop rounds: record the planner's own "
                             "executed trajectories and retrain")
    parser.add_argument("--dagger-episodes", type=int, default=2,
                        help="planner episodes recorded per DAgger round")
    parser.add_argument("--dagger-cap", type=int, default=1200,
                        help="frame cap per recorded planner episode")
    parser.add_argument("--dagger-offset", type=int, default=50000,
                        help="episode-id offset for recorded planner data")
    parser.add_argument("--dagger-horizon", type=int, default=0,
                        help="search horizon while recording planner data; "
                             "0 uses --horizon")
    parser.add_argument("--fast", action="store_true",
                        help="small development preset: 1500 steps, 3 epochs, "
                             "1000 nodes, 400-frame evaluation")
    args = parser.parse_args()
    if args.fast:
        args.steps = 1500
        args.epochs = 3
        args.max_nodes = 1000
        args.cap = 400
        args.horizon = 8
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.coverage <= 1.0:
        raise ValueError("--coverage must be in [0, 1]")
    if args.cap < 1:
        raise ValueError("--cap must be positive")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.max_nodes < 1:
        raise ValueError("--max-nodes must be positive")
    if args.birth_threshold <= 0.0:
        raise ValueError("--birth-threshold must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.lag < 0:
        raise ValueError("--lag must be non-negative")
    if args.pain_horizon < 0:
        raise ValueError("--pain-horizon must be non-negative")
    if not 0.0 <= args.voi_fraction < 1.0:
        raise ValueError("--voi-fraction must be in [0, 1)")
    if args.voi_target < 1:
        raise ValueError("--voi-target must be positive")
    if args.feature_value_weight < 0.0:
        raise ValueError("--feature-value-weight must be non-negative")
    if not 0.0 < args.value_quantile < 1.0:
        raise ValueError("--value-quantile must be in (0, 1)")
    if args.component_min_count < 1:
        raise ValueError("--component-min-count must be positive")
    if args.max_components < 0:
        raise ValueError("--max-components must be non-negative")
    if args.component_weight < 0.0:
        raise ValueError("--component-weight must be non-negative")
    if not 0.0 <= args.memory_weight <= 1.0:
        raise ValueError("--memory-weight must be in [0, 1]")
    if not 0.0 <= args.association_rate <= 1.0:
        raise ValueError("--association-rate must be in [0, 1]")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be positive")
    if args.step_stride < 1:
        raise ValueError("--step-stride must be positive")
    if args.dagger_rounds < 0:
        raise ValueError("--dagger-rounds must be non-negative")
    if args.dagger_rounds and args.dagger_episodes < 1:
        raise ValueError("--dagger-episodes must be positive")
    if args.dagger_cap < 1:
        raise ValueError("--dagger-cap must be positive")
    if args.dagger_offset < 0:
        raise ValueError("--dagger-offset must be non-negative")
    if args.dagger_horizon < 0:
        raise ValueError("--dagger-horizon must be non-negative")
    if args.dagger_horizon == 0:
        args.dagger_horizon = args.horizon
    if args.value_horizon < 0:
        raise ValueError("--value-horizon must be non-negative")
    if args.ood_trust_scale < 0.0:
        raise ValueError("--ood-trust-scale must be non-negative")
    if not 0.0 <= args.obs_trust <= 1.0:
        raise ValueError("--obs-trust must be in [0, 1]")
    if args.feature_action_prior < 0.0:
        raise ValueError("--feature-action-prior must be non-negative")
    if args.feature_value_prior < 0.0:
        raise ValueError("--feature-value-prior must be non-negative")
    component_kinds = tuple(
        kind.strip() for kind in args.components.split(",") if kind.strip()
    )
    eval_seeds = parse_seed_spec(args.eval_seeds)

    seeds = [seed for seed in SEEDS if seed >= args.seed][:args.seeds]
    if len(seeds) < args.seeds:
        seeds = list(range(args.seed, args.seed + args.seeds))
    if eval_seeds:
        overlap = sorted(set(seeds) & set(eval_seeds))
        if overlap:
            print(
                "warning: evaluation seeds overlap training seeds: %s"
                % ",".join(str(seed) for seed in overlap)
            )
    else:
        print(
            "warning: --eval-seeds is empty; using legacy self-evaluation "
            "(training seed == evaluation seed)"
        )
    tasks = [
        dict(
            seed=seed,
            steps=args.steps,
            coverage=args.coverage,
            cap=args.cap,
            max_nodes=args.max_nodes,
            birth_threshold=args.birth_threshold,
            horizon=args.horizon,
            epochs=args.epochs,
            lag=args.lag,
            component_kinds=component_kinds,
            component_min_count=args.component_min_count,
            max_components=args.max_components,
            component_weight=args.component_weight,
            component_gating=args.component_gating,
            memory_weight=args.memory_weight,
            association_rate=args.association_rate,
            induce_features=not args.no_features,
            precursor_labels=not args.instant_labels,
            pain_horizon=args.pain_horizon,
            voi_fraction=args.voi_fraction,
            voi_target=args.voi_target,
            feature_value_weight=args.feature_value_weight,
            step_stride=args.step_stride,
            stride_includes_lag=args.stride_includes_lag,
            decision=args.decision,
            value_horizon=args.value_horizon,
            ood_trust_scale=args.ood_trust_scale,
            obs_trust=args.obs_trust,
            feature_action_prior=args.feature_action_prior,
            feature_value_prior=args.feature_value_prior,
            value_features=not args.no_value_features,
            value_label_quantile=args.value_quantile,
            subgraph_cells=args.subgraph_cells,
            predicate_readouts=args.predicate_readouts,
            eval_seeds=eval_seeds,
            eval_episodes=args.eval_episodes,
            dagger_rounds=args.dagger_rounds,
            dagger_episodes=args.dagger_episodes,
            dagger_cap=args.dagger_cap,
            dagger_offset=args.dagger_offset,
            dagger_horizon=args.dagger_horizon,
        )
        for seed in seeds
    ]
    def _drain(futures):
        """Collect worker results, streaming one line per finished model."""
        rows = []
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            frames = [item["frames"] for item in row["evals"]]
            print(
                " model %3d done: held-out mean %.1f over %d worlds"
                % (row["seed"], float(np.mean(frames)), len(frames)),
                flush=True,
            )
        return rows

    if args.jobs == 1:
        rows = []
        for task in tasks:
            row = run_seed(task)
            frames = [item["frames"] for item in row["evals"]]
            print(
                " model %3d done: held-out mean %.1f over %d worlds"
                % (row["seed"], float(np.mean(frames)), len(frames)),
                flush=True,
            )
            rows.append(row)
    elif FREE_THREADED:
        # The free-threaded build runs seed-level Python and numpy work in
        # parallel without pickling graphs between workers.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows = _drain([
                pool.submit(run_seed, task) for task in tasks
            ])
    else:
        # A GIL build still gets true parallelism from independent seeds; the
        # payload per seed is small (plain dicts and lists).
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = _drain([
                pool.submit(run_seed, task) for task in tasks
            ])
    rows.sort(key=lambda row: row["seed"])

    print("train  concepts  graph_nodes  action_edges  components")
    for row in rows:
        graph = row["graph"]
        print(" %4d  %8d  %11d  %12d  %10d"
              % (row["seed"], graph["prototypes"], graph["nodes"],
                 graph["action_edges"], graph["components"]))

    for row in rows:
        if row["dagger"]:
            parts = [
                "r%d +%d steps (%s frames)"
                % (item["round"], item["planner_steps"],
                   "/".join(str(value) for value in item["planner_frames"]))
                for item in row["dagger"]
            ]
            print(" dagger seed %d: %s" % (row["seed"], ", ".join(parts)))

    eval_rows = [row for result in rows for row in result["evals"]]
    print("\ntrain  eval  ep  env_seed  frames  action_rate")
    for row in eval_rows:
        print(" %5d  %4d  %2d  %8d  %6d  %10.3f"
              % (row["train_seed"], row["eval_seed"], row["episode"],
                 row["env_seed"], row["frames"], row["action_rate"]))

    per_train = group_summaries(eval_rows, group="train_seed")
    per_eval = group_summaries(eval_rows, group="eval_seed")
    overall = cluster_summary(eval_rows, cluster="train_seed")
    print("\nframes per training seed (mean over evaluation worlds)")
    for key, summary in per_train.items():
        print(" train %4s  n=%2d  mean %6.1f  median %6.1f  range %d-%d"
              % (key, summary["n"], summary["mean"], summary["median"],
                 summary["min"], summary["max"]))
    print("frames per evaluation seed (mean over training seeds)")
    for key, summary in per_eval.items():
        print(" eval  %4s  n=%2d  mean %6.1f  median %6.1f"
              % (key, summary["n"], summary["mean"], summary["median"]))
    print("\nframes overall: n=%d mean %.1f median %.1f IQR %.1f [%.1f, %.1f] "
          "range %d-%d"
          % (overall["n"], overall["mean"], overall["median"],
             overall["iqr"], overall["q1"], overall["q3"],
             overall["min"], overall["max"]))
    print("bootstrap 95%% CI over %d training seeds: [%.1f, %.1f]"
          % (overall["clusters"], overall["cluster_ci_low"],
             overall["cluster_ci_high"]))
    if overall["clusters"] < 5:
        print("warning: fewer than 5 trained models; the interval is "
              "descriptive, not a significance test")

    if args.report:
        report = dict(
            config=dict(
                training_seeds=seeds,
                eval_seeds=eval_seeds,
                eval_episodes=args.eval_episodes,
                steps=args.steps,
                coverage=args.coverage,
                cap=args.cap,
                horizon=args.horizon,
                epochs=args.epochs,
                max_nodes=args.max_nodes,
                lag=args.lag,
                step_stride=args.step_stride,
                stride_includes_lag=args.stride_includes_lag,
                decision=args.decision,
                value_horizon=args.value_horizon,
                ood_trust_scale=args.ood_trust_scale,
                obs_trust=args.obs_trust,
                feature_action_prior=args.feature_action_prior,
                feature_value_prior=args.feature_value_prior,
                feature_value_weight=args.feature_value_weight,
                features=not args.no_features,
                precursor_labels=not args.instant_labels,
                value_features=not args.no_value_features,
                value_label_quantile=args.value_quantile,
                subgraph_cells=args.subgraph_cells,
                predicate_readouts=args.predicate_readouts,
                self_evaluation=not bool(eval_seeds),
                dagger_rounds=args.dagger_rounds,
                dagger_episodes=args.dagger_episodes,
                dagger_cap=args.dagger_cap,
                dagger_offset=args.dagger_offset,
                dagger_horizon=args.dagger_horizon,
            ),
            rows=eval_rows,
            dagger=[
                dict(seed=row["seed"], rounds=row["dagger"])
                for row in rows
            ],
            per_train_seed=per_train,
            per_eval_seed=per_eval,
            overall=overall,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("wrote %s" % args.report)

    print("\nlearned high-pain nodes (seed %d):" % rows[0]["seed"])
    for node in rows[0]["top_nodes"]:
        features = node["features"]
        preview = (
            "n/a" if features is None
            else "[" + ", ".join("%+.2f" % value for value in features[:6]) + "]"
        )
        print("  %-8s count=%4d pain=%.3f dopamine=%.2f state=%s"
              % (node["id"], node["count"], node["pain"],
                 node["dopamine"], preview))

    print("\nlearned high-pain action edges (seed %d):" % rows[0]["seed"])
    for edge in rows[0]["top_edges"]:
        print("  %-8s -a%d-> %-8s count=%4d pain=%.3f dopamine=%.2f"
              % (edge["source"], edge["action"], edge["target"],
                 edge["count"], edge["pain"], edge["dopamine"]))


if __name__ == "__main__":
    main()
