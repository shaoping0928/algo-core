"""Search using graph-memory risk on top of a learned observation model.

The planner reads raw observations, predicts next observations with the generic linear
world model, and scores candidate trajectories with concept/transition-edge values from
the prototype graph memory. It does not read pipes, gap centres, or collision windows.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from graph_memory import PrototypeGraphMemory
from latent_world import SEEDS, latent
from substrate_generic import (
    ACTION_DIM,
    H,
    LAG,
    LinearObservationModel,
    collect,
    history_features,
)
from substrate_value import (
    TerminalRiskHead,
    TransitionReplay,
    fit_terminal_risk,
)
from trajectory_search import exhaustive_search

DEFAULT_JOBS = min(4, os.cpu_count() or 1)


def fit(seed, steps, coverage, clusters):
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    if clusters < 1:
        raise ValueError("clusters must be positive")
    transitions, _counts = collect(seed, steps, coverage=coverage)
    if not transitions:
        raise ValueError("collection produced no transitions")
    world = LinearObservationModel(
        len(transitions[0]["obs"]), ACTION_DIM
    ).fit(
        [row["obs"] for row in transitions],
        [row["history"] for row in transitions],
        [row["next_obs"] for row in transitions],
    )
    graph = PrototypeGraphMemory(clusters=clusters, seed=seed).fit(transitions)
    risk = TerminalRiskHead(feature_dim=len(transitions[0]["obs"]))
    replay = TransitionReplay.for_stream(len(transitions), seed=seed)
    for row in transitions:
        state = np.asarray(row["next_obs"], dtype=float)
        pain = float(row["terminal"])
        replay.add(state, [1.0 - pain, pain], state, bool(pain))
    fit_terminal_risk(
        risk, replay, 4000, batch_size=128, terminal_fraction=0.15
    )
    return world, graph, risk


def policy_action(env, world, graph, risk_head, prefix):
    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    label_cache = {}

    def predict_batch(states, sequences, step):
        history = history_features(sequences, step + 1, pending_prefix)
        return world.predict(states, history)

    def previous_labels(sequences, step):
        if step == 0:
            return None
        return np.asarray([
            label_cache.get(seq[:step].tobytes(), -1) for seq in sequences
        ], dtype=int)

    def remember(sequences, labels, step):
        for seq, label in zip(sequences, labels):
            label_cache[seq[:step + 1].tobytes()] = int(label)

    def combined_risk(states, labels, previous):
        graph_risk = graph.risk_labels(labels, previous)
        global_risk = risk_head.probability(states)
        return 0.5 * (graph_risk + global_risk)

    def evaluate_step(states, sequences, step):
        labels = graph.prototype(states)
        previous = previous_labels(sequences, step)
        risk = combined_risk(states, labels, previous)
        remember(sequences, labels, step)
        return np.column_stack([risk, np.zeros(len(states))]), np.ones(
            len(states), dtype=bool
        )

    def evaluate_terminal(states, sequences):
        labels = graph.prototype(states)
        previous = previous_labels(sequences, sequences.shape[1] - 1)
        risk = combined_risk(states, labels, previous)
        value = graph.value_labels(labels, previous)
        return np.column_stack([risk, -value]), np.ones(
            len(states), dtype=bool
        )

    result = exhaustive_search(
        initial_state=initial,
        action_space=(0, 1),
        horizon=H,
        transition_batch=predict_batch,
        evaluate_step=evaluate_step,
        evaluate_terminal=evaluate_terminal,
        batch_size=2 ** H,
        score_combine=("max", "sum"),
    )
    return int(result.actions[0])


def evaluate(seed, world, graph, risk_head, cap=6000):
    if cap < 1:
        raise ValueError("cap must be positive")
    env = latent[LAG](seed)
    prefix = np.zeros(LAG, dtype=float)
    actions = []
    while not env.dead and env.t < cap:
        action = policy_action(env, world, graph, risk_head, prefix)
        actions.append(action)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(action)
    return env.t, actions


def run_seed(task):
    seed, steps, coverage, clusters, cap = task
    world, graph, risk_head = fit(seed, steps, coverage, clusters)
    frames, actions = evaluate(seed, world, graph, risk_head, cap=cap)
    return dict(
        seed=seed, frames=frames,
        flap_rate=float(np.mean(actions)) if actions else 0.0,
        graph=graph.snapshot(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--clusters", type=int, default=48)
    parser.add_argument("--cap", type=int, default=6000)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.coverage <= 1.0:
        raise ValueError("--coverage must be in [0, 1]")
    if args.clusters < 1:
        raise ValueError("--clusters must be positive")
    if args.cap < 1:
        raise ValueError("--cap must be positive")
    seeds = [
        seed for seed in SEEDS if seed >= args.seed
    ][:args.seeds]
    if len(seeds) < args.seeds:
        seeds = list(range(args.seed, args.seed + args.seeds))
    tasks = [
        (seed, args.steps, args.coverage, args.clusters, args.cap)
        for seed in seeds
    ]
    if args.jobs == 1:
        rows = [run_seed(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(run_seed, tasks))
    print(" seed  frames  flap_rate  nodes  edges  edge_nodes  memories")
    for row in rows:
        graph = row["graph"]
        print(" %4d  %6d  %9.3f  %5d  %5d  %10d  %8d"
              % (row["seed"], row["frames"], row["flap_rate"],
                 graph["nodes"], graph["edges"], graph["edge_nodes"],
                 graph["memories"]))
    print("frames mean %.1f median %.1f"
          % (np.mean([row["frames"] for row in rows]),
             np.median([row["frames"] for row in rows])))


if __name__ == "__main__":
    main()
