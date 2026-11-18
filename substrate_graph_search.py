"""Search using graph-memory risk on top of a learned observation model.

The planner reads raw observations, predicts next observations with the generic linear
world model, and scores candidate trajectories with concept/transition-edge values from
the prototype graph memory. It does not read pipes, gap centres, or collision windows.
"""
import argparse

import numpy as np

from graph_memory import PrototypeGraphMemory
from latent_world import latent
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


def fit(seed, steps, coverage, clusters):
    transitions, _counts = collect(seed, steps, coverage=coverage)
    world = LinearObservationModel(
        len(transitions[0]["obs"]), ACTION_DIM
    ).fit(
        [row["obs"] for row in transitions],
        [row["history"] for row in transitions],
        [row["next_obs"] for row in transitions],
    )
    graph = PrototypeGraphMemory(clusters=clusters, seed=seed).fit(transitions)
    risk = TerminalRiskHead(feature_dim=len(transitions[0]["obs"]))
    replay = TransitionReplay(capacity=len(transitions) + 1, seed=seed)
    for row in transitions:
        obs = np.asarray(row["obs"], dtype=float)
        pain = float(row["terminal"])
        replay.add(obs, [1.0 - pain, pain], obs, bool(pain))
    fit_terminal_risk(
        risk, replay, 4000, batch_size=128, terminal_fraction=0.15
    )
    return world, graph, risk


def policy_action(env, world, graph, risk_head, prefix):
    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    last_labels = None

    def predict_batch(states, sequences, step):
        history = history_features(sequences, step + 1, pending_prefix)
        return world.predict(states, history)

    def combined_risk(states, labels, previous):
        graph_risk = graph.risk_labels(labels, previous)
        global_risk = risk_head.probability(states)
        return 0.5 * (graph_risk + global_risk)

    def evaluate_step(states, sequences, step):
        nonlocal last_labels
        labels = graph.prototype(states)
        previous = None if step == 0 else last_labels
        risk = combined_risk(states, labels, previous)
        last_labels = labels
        return np.column_stack([risk, np.zeros(len(states))]), np.ones(
            len(states), dtype=bool
        )

    def evaluate_terminal(states, sequences):
        labels = graph.prototype(states)
        risk = combined_risk(states, labels, last_labels)
        value = graph.value_labels(labels, last_labels)
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
    env = latent[LAG](seed)
    prefix = np.zeros(LAG, dtype=float)
    actions = []
    while not env.dead and env.t < cap:
        action = policy_action(env, world, graph, risk_head, prefix)
        actions.append(action)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(action)
    return env.t, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--clusters", type=int, default=48)
    args = parser.parse_args()
    world, graph, risk_head = fit(
        args.seed, args.steps, args.coverage, args.clusters
    )
    frames, actions = evaluate(args.seed, world, graph, risk_head)
    print("seed", args.seed, "frames", frames, "graph", graph.snapshot())
    print("flap rate %.3f" % (np.mean(actions) if actions else 0.0))


if __name__ == "__main__":
    main()
