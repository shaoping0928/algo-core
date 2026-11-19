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
from node_runtime import NodeRuntime
from substrate_generic import (
    LinearObservationModel,
    collect,
    history_features,
)
from trajectory_search import exhaustive_search

LAG = 3
DEFAULT_JOBS = min(8, os.cpu_count() or 1)
FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()


def fit(seed, steps=5000, coverage=0.1, max_nodes=2000,
        birth_threshold=0.45, epochs=5, lag=LAG,
        component_kinds=(), component_min_count=8,
        max_components=128, component_weight=0.5,
        component_gating=False):
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
    lag = int(lag)
    transitions, _counts = collect(
        seed, steps, coverage=coverage, lag=lag
    )
    if not transitions:
        raise ValueError("collection produced no transitions")
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
    )
    episodes = {}
    for row in transitions:
        episodes.setdefault(row["episode"], []).append(row)
    ordered = [episodes[key] for key in sorted(episodes)]
    for _epoch in range(epochs):
        for episode in ordered:
            agent.learn_episode(episode)
    return world, agent


def policy_action(env, world, agent, prefix, horizon, lag=LAG):
    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_dim = int(lag) + 1

    def predict_batch(states, sequences, step):
        history = history_features(
            sequences, step + 1, pending_prefix, action_dim=action_dim
        )
        return world.predict(states, history)

    def evaluate_step(states, sequences, step):
        return np.zeros((len(states), 2)), np.isfinite(states).all(axis=1)

    def evaluate_terminal(states, sequences):
        history = history_features(
            sequences, horizon, pending_prefix, action_dim=action_dim
        )
        # `history` is newest-first; the node state stores pending actions
        # oldest-first and only the last `lag` actions remain pending.
        pending = history[:, :agent.lag][:, ::-1]
        encoded = np.concatenate([states, pending], axis=1) / agent.scales
        pain, dopamine = agent.valence_state_batch(encoded)
        return np.column_stack([pain, -dopamine]), np.isfinite(encoded).all(
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
        score_combine="sum",
    )
    return int(result.actions[0])


def evaluate(seed, world, agent, cap=6000, horizon=10, lag=LAG):
    if cap < 1:
        raise ValueError("cap must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    lag = int(lag)
    env = latent[lag](seed)
    prefix = np.zeros(lag, dtype=float)
    actions = []
    while not env.dead and env.t < cap:
        action = policy_action(
            env, world, agent, prefix, horizon, lag=lag
        )
        actions.append(action)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(action)
    return env.t, actions


def run_seed(task):
    (seed, steps, coverage, cap, max_nodes, birth_threshold, horizon,
     epochs, lag, component_kinds, component_min_count, max_components,
     component_weight, component_gating) = task
    world, agent = fit(
        seed, steps=steps, coverage=coverage,
        max_nodes=max_nodes, birth_threshold=birth_threshold,
        epochs=epochs, lag=lag,
        component_kinds=component_kinds,
        component_min_count=component_min_count,
        max_components=max_components,
        component_weight=component_weight,
        component_gating=component_gating,
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
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--birth-threshold", type=float, default=0.45)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lag", type=int, default=LAG)
    parser.add_argument("--components", default="",
                        help="comma-separated runtime component kinds")
    parser.add_argument("--component-min-count", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--component-weight", type=float, default=0.5)
    parser.add_argument("--component-gating", action="store_true")
    args = parser.parse_args()
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
    if args.component_min_count < 1:
        raise ValueError("--component-min-count must be positive")
    if args.max_components < 0:
        raise ValueError("--max-components must be non-negative")
    if args.component_weight < 0.0:
        raise ValueError("--component-weight must be non-negative")
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
         args.component_gating)
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

    print(" seed  frames  flap_rate  nodes  components  action_edges")
    for row in rows:
        graph = row["graph"]
        print(" %4d  %6d  %9.3f  %5d  %10d  %12d"
              % (row["seed"], row["frames"], row["flap_rate"],
                 graph["nodes"], graph["components"],
                 graph["action_edges"]))
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
