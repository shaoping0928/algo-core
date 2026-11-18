"""Permutation sensitivity of the generic world/value/risk models.

For each input dimension, shuffle that column on held-out real transitions and measure
how much the prediction degrades. A near-zero delta means the model is not using that
signal; a large delta means it is.

This is a diagnostic only; it does not alter the controller.
"""
import argparse

import numpy as np

from substrate_generic import ACTION_DIM, LinearObservationModel, collect
from substrate_value import (
    WorldFeedbackValueHead,
    TerminalRiskHead,
    binary_auc,
    fit_terminal_risk,
    fit_value_head_returns,
)
from trace_generic_training import (
    OBS_NAMES,
    build_training_replays,
    held_records,
    split_episodes,
)

ACTION_NAMES = ("a0", "a1", "a2", "a3")


def risk_arrays(held):
    return (
        np.asarray([row["next_obs"] for row in held], dtype=float),
        np.asarray([row["history"] for row in held], dtype=float),
        np.asarray([row["terminal"] for row in held], dtype=bool),
    )


def value_arrays(held):
    return (
        np.asarray([row["obs"] for row in held], dtype=float),
        np.asarray([row["history"] for row in held], dtype=float),
    )


def report(name, deltas, names):
    order = np.argsort(deltas)[::-1]
    print("\n%s permutation importance (largest first):" % name)
    for index in order:
        print("  %-4s %+.4e" % (names[index], deltas[index]))


def run(seed, steps, coverage, value_updates=3000, risk_updates=6000):
    rows, _counts = collect(seed, steps, coverage=coverage)
    train, held = split_episodes(rows)
    world = LinearObservationModel(
        len(train[0]["obs"]), ACTION_DIM
    ).fit(
        [row["obs"] for row in train],
        [row["history"] for row in train],
        [row["next_obs"] for row in train],
    )
    train_obs = np.asarray([row["obs"] for row in train], dtype=float)
    train_hist = np.asarray([row["history"] for row in train], dtype=float)
    train_next = np.asarray([row["next_obs"] for row in train], dtype=float)
    errors = np.abs(
        world.predict(train_obs, train_hist) - train_next
    ).mean(axis=1)
    scale = float(np.median(errors))
    priorities = 1.0 + np.minimum(errors / max(scale, 1e-12), 10.0)
    value_replay, risk_replay, obs_dim = build_training_replays(
        train, priorities
    )
    value_head = WorldFeedbackValueHead(
        feature_dim=obs_dim + ACTION_DIM, discount=0.99
    )
    risk_head = TerminalRiskHead(feature_dim=obs_dim + ACTION_DIM)
    fit_value_head_returns(
        value_head, value_replay, value_updates, batch_size=128,
        priority_alpha=1.0,
    )
    fit_terminal_risk(
        risk_head, risk_replay, risk_updates, batch_size=128,
        terminal_fraction=0.15, priority_alpha=1.0,
    )
    print("seed=%d held=%d" % (seed, len(held)))

    obs, hist, next_obs = (
        np.asarray([row["obs"] for row in held], dtype=float),
        np.asarray([row["history"] for row in held], dtype=float),
        np.asarray([row["next_obs"] for row in held], dtype=float),
    )
    world_base = np.abs(world.predict(obs, hist) - next_obs).mean()
    print("world one-step baseline %.3e" % world_base)

    rng = np.random.RandomState(seed + 700)
    world_deltas = []
    for index in range(obs_dim):
        shuffled = obs.copy()
        shuffled[:, index] = shuffled[rng.permutation(len(shuffled)), index]
        error = np.abs(world.predict(shuffled, hist) - next_obs).mean()
        world_deltas.append(error - world_base)
    for index in range(ACTION_DIM):
        shuffled = hist.copy()
        shuffled[:, index] = shuffled[rng.permutation(len(shuffled)), index]
        error = np.abs(world.predict(obs, shuffled) - next_obs).mean()
        world_deltas.append(error - world_base)
    report("world", np.asarray(world_deltas), OBS_NAMES[:obs_dim] + ACTION_NAMES)

    risk_obs, risk_hist, terminal = risk_arrays(held)
    risk_input = np.concatenate([risk_obs, risk_hist], axis=1)
    risk_base = risk_head.probability(risk_input)
    risk_auc = binary_auc(risk_base, terminal)
    print("\nrisk baseline AUC %.3f" % risk_auc)
    risk_deltas = []
    for index in range(risk_input.shape[1]):
        shuffled = risk_input.copy()
        shuffled[:, index] = shuffled[rng.permutation(len(shuffled)), index]
        risk_auc_perm = binary_auc(risk_head.probability(shuffled), terminal)
        risk_deltas.append(risk_auc - risk_auc_perm)
    report("risk", np.asarray(risk_deltas),
           OBS_NAMES[:obs_dim] + ACTION_NAMES)

    value_obs, value_hist = value_arrays(held)
    value_input = np.concatenate([value_obs, value_hist], axis=1)
    value_pain = value_head.value(value_input)[:, 1]
    value_auc = binary_auc(value_pain, terminal)
    print("\nvalue pain baseline AUC %.3f" % value_auc)
    value_deltas = []
    for index in range(value_input.shape[1]):
        shuffled = value_input.copy()
        shuffled[:, index] = shuffled[rng.permutation(len(shuffled)), index]
        value_auc_perm = binary_auc(
            value_head.value(shuffled)[:, 1], terminal
        )
        value_deltas.append(value_auc - value_auc_perm)
    report("value", np.asarray(value_deltas),
           OBS_NAMES[:obs_dim] + ACTION_NAMES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    args = parser.parse_args()
    run(args.seed, args.steps, args.coverage)


if __name__ == "__main__":
    main()
