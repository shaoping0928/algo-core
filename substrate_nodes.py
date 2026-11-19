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
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from latent_world import SEEDS, latent
from node_features import FeatureRuleBank
from node_runtime import NodeRuntime, future_pain_labels
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
           pain_horizon, voi_target):
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
    return world, agent


def fit(seed, steps=5000, coverage=0.1, max_nodes=5000,
        birth_threshold=0.45, epochs=5, lag=LAG,
        component_kinds=(), component_min_count=8,
        max_components=128, component_weight=0.5,
        component_gating=False, memory_weight=0.1,
        association_rate=0.0, induce_features=True,
        precursor_labels=True, pain_horizon=None,
        voi_fraction=0.0, voi_target=3):
    """Collect transitions and train the node runtime on them.

    With `voi_fraction > 0` collection runs in two stages: the first stage
    bootstraps the runtime with the plain behavior policy, the second stage
    asks that bootstrap agent where its (feature, action) support is
    deficient and overrides at most `voi_fraction * steps` actions with the
    less-supported one.  The override is additive - the behavior sampler
    still runs the rest of the time - and everything is retrained on the
    combined stream.
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
    return _train(
        transitions, seed=seed, epochs=epochs, lag=lag,
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
    )


def policy_result(env, world, agent, prefix, horizon, lag=LAG,
                  step_stride=2):
    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)

    def predict_batch(states, sequences, step):
        history = history_features(
            sequences, step + 1, pending_prefix, action_dim=action_dim
        )
        return world.predict(states, history)

    def evaluate_step(states, sequences, step):
        if step % step_stride != 0:
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
        risk = np.maximum(pain, agent.pain_signal_batch(encoded))
        return np.column_stack([risk, -dopamine]), np.isfinite(encoded).all(
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
        score_combine=("max", "sum"),
    )
    return result


def policy_action(env, world, agent, prefix, horizon, lag=LAG,
                  step_stride=2):
    return int(policy_result(
        env, world, agent, prefix, horizon, lag=lag,
        step_stride=step_stride,
    ).actions[0])


def plan_score(initial_state, world, agent, prefix, plan, horizon, lag=LAG,
               step_stride=2):
    """Score one fixed plan with the same objective as `policy_result`.

    Reproduces the `exhaustive_search` scoring under
    ``score_combine=("max", "sum")``: column 0 is the worst `pain_signal`
    risk over the strided predicted states and the terminal state, column 1
    is the negative terminal dopamine.  Used by diagnostics that compare
    fixed plans against the searched plan, and later by plan-level posteriors.
    """
    initial = np.asarray(initial_state, dtype=float).reshape(1, -1)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)
    plan = np.asarray(plan, dtype=float).reshape(1, -1)
    states = initial.copy()
    worst = 0.0
    for step in range(horizon):
        history = history_features(
            plan, step + 1, pending_prefix, action_dim=action_dim
        )
        states = np.asarray(world.predict(states, history), dtype=float)
        if step % step_stride == 0:
            pending = history[:, :agent.lag][:, ::-1]
            encoded = np.concatenate([states, pending], axis=1) / agent.scales
            if not np.isfinite(encoded).all():
                return (float("inf"), float("inf"))
            worst = max(worst, float(agent.pain_signal_batch(encoded)[0]))
    history = history_features(
        plan, horizon, pending_prefix, action_dim=action_dim
    )
    pending = history[:, :agent.lag][:, ::-1]
    encoded = np.concatenate([states, pending], axis=1) / agent.scales
    if not np.isfinite(encoded).all():
        return (float("inf"), float("inf"))
    pain, dopamine = agent.valence_state_batch(encoded)
    terminal = max(
        float(pain[0]), float(agent.pain_signal_batch(encoded)[0])
    )
    return (max(worst, terminal), -float(dopamine[0]))


def evaluate(seed, world, agent, cap=6000, horizon=10, lag=LAG,
             step_stride=2):
    if cap < 1:
        raise ValueError("cap must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if step_stride < 1:
        raise ValueError("step_stride must be positive")
    lag = int(lag)
    env = latent[lag](seed)
    prefix = np.zeros(lag, dtype=float)
    actions = []
    while not env.dead and env.t < cap:
        action = policy_action(
            env, world, agent, prefix, horizon, lag=lag,
            step_stride=step_stride,
        )
        actions.append(action)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(action)
    return env.t, actions


def run_seed(task):
    (seed, steps, coverage, cap, max_nodes, birth_threshold, horizon,
     epochs, lag, component_kinds, component_min_count, max_components,
     component_weight, component_gating, memory_weight,
     association_rate, induce_features, precursor_labels,
     pain_horizon, voi_fraction, voi_target) = task
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
    )
    frames, actions = evaluate(
        seed, world, agent, cap=cap, horizon=horizon, lag=lag
    )
    return dict(
        seed=seed,
        frames=frames,
        flap_rate=float(np.mean(actions)) if actions else 0.0,
        graph=agent.snapshot(),
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
    component_kinds = tuple(
        kind.strip() for kind in args.components.split(",") if kind.strip()
    )

    seeds = [seed for seed in SEEDS if seed >= args.seed][:args.seeds]
    if len(seeds) < args.seeds:
        seeds = list(range(args.seed, args.seed + args.seeds))
    tasks = [
        (seed, args.steps, args.coverage, args.cap,
         args.max_nodes, args.birth_threshold, args.horizon, args.epochs,
         args.lag, component_kinds, args.component_min_count,
         args.max_components, args.component_weight,
         args.component_gating, args.memory_weight,
         args.association_rate, not args.no_features,
         not args.instant_labels, args.pain_horizon,
         args.voi_fraction, args.voi_target)
        for seed in seeds
    ]
    if args.jobs == 1:
        rows = [run_seed(task) for task in tasks]
    elif FREE_THREADED:
        # The free-threaded build runs seed-level Python and numpy work in
        # parallel without pickling graphs between workers.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(run_seed, tasks))
    else:
        # A GIL build still gets true parallelism from independent seeds; the
        # payload per seed is small (plain dicts and lists).
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(run_seed, tasks))

    print(" seed  frames  flap_rate  concepts  graph_nodes  action_edges  components")
    for row in rows:
        graph = row["graph"]
        print(" %4d  %6d  %9.3f  %8d  %11d  %12d  %10d"
              % (row["seed"], row["frames"], row["flap_rate"],
                 graph["prototypes"], graph["nodes"],
                 graph["action_edges"], graph["components"]))
    print("frames mean %.1f median %.1f"
          % (np.mean([row["frames"] for row in rows]),
             np.median([row["frames"] for row in rows])))

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
