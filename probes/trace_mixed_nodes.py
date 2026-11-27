"""Compare a global risk head with generic mixed conjunctive pain nodes.

The mixed-node builder sees only observation channels and the world's pain label. It
creates a conjunction only when two primitive gates co-activate often enough and raise
the pain rate. No pipe, gap, or collision rule is supplied.
"""
import argparse

import numpy as np

from latent_world import GAP
from mixed_nodes import (
    ContinuousMixedMemory,
    LayeredMixedMemory,
    MixedNodeRiskMemory,
)
from substrate_generic import collect
from substrate_value import (
    TerminalRiskHead,
    TransitionReplay,
    binary_auc,
    fit_terminal_risk,
)
from trace_generic_training import OBS_NAMES, split_episodes


def near_masks(rows):
    observations = np.asarray([row["next_obs"] for row in rows], dtype=float)
    near = (np.abs(observations[:, 2]) < 0.20) & (observations[:, 2] > -0.10)
    inside = np.abs(observations[:, 3]) <= GAP / 2.0
    return observations, near, inside


def report_metrics(name, risk, pain, near, inside):
    print(
        "%-8s AUC=%.3f near_auc=%.3f Brier=%.3f "
        "near_in=%.3f near_out=%.3f"
        % (
            name, binary_auc(risk, pain),
            binary_auc(risk[near], pain[near]) if near.any() else float("nan"),
            np.mean((risk - pain) ** 2),
            risk[near & inside].mean() if (near & inside).any() else float("nan"),
            risk[near & ~inside].mean() if (near & ~inside).any() else float("nan"),
        )
    )


def permutation_importance(name, score_fn, observations, pain, rng):
    base = binary_auc(score_fn(observations), pain)
    deltas = []
    for channel in range(observations.shape[1]):
        shuffled = observations.copy()
        shuffled[:, channel] = shuffled[rng.permutation(len(shuffled)), channel]
        deltas.append(base - binary_auc(score_fn(shuffled), pain))
    order = np.argsort(deltas)[::-1]
    print("\n%s sensitivity (AUC drop):" % name)
    for index in order:
        print("  %-4s %+.4f" % (OBS_NAMES[index], deltas[index]))


def run(seed, steps, coverage):
    rows, _counts = collect(seed, steps, coverage=coverage)
    train, held = split_episodes(rows)
    train_obs, train_near, train_inside = near_masks(train)
    held_obs, held_near, held_inside = near_masks(held)
    train_pain = np.asarray([row["terminal"] for row in train], dtype=float)
    held_pain = np.asarray([row["terminal"] for row in held], dtype=bool)

    memory = MixedNodeRiskMemory().fit(train_obs, train_pain)
    mixed_risk = memory.risk(held_obs)
    continuous = ContinuousMixedMemory(capacity=64).fit(train_obs, train_pain)
    continuous_risk = continuous.risk(held_obs)
    layered = LayeredMixedMemory(
        common_capacity=48, rare_capacity=16
    ).fit(train_obs, train_pain)
    layered_risk = layered.risk(held_obs)

    global_head = TerminalRiskHead(feature_dim=train_obs.shape[1])
    replay = TransitionReplay.for_stream(len(train), seed=seed)
    for row, pain in zip(train, train_pain):
        state = np.asarray(row["next_obs"], dtype=float)
        replay.add(state, [1.0 - pain, pain], state, bool(pain))
    fit_terminal_risk(
        global_head, replay, 4000, batch_size=128,
        terminal_fraction=0.15,
    )
    global_risk = global_head.probability(held_obs)
    residual = train_pain - global_head.probability(train_obs)
    residual_memory = MixedNodeRiskMemory(
        min_support=8.0, min_lift=0.0, min_pain=0.0, max_nodes=32
    ).fit(train_obs, residual)
    residual_risk = np.clip(
        global_risk + residual_memory.risk(held_obs), 0.0, 1.0
    )

    print(
        "seed=%d train=%d held=%d nodes=%d gates=%d residual_nodes=%d "
        "continuous=%d created=%d forgotten=%d layered=%d rare=%d"
        % (seed, len(train), len(held), len(memory.nodes), len(memory.gates),
           len(residual_memory.nodes), len(continuous.nodes),
           continuous.created, continuous.forgotten, len(layered.nodes),
           layered.rare_nodes)
    )
    report_metrics("global", global_risk, held_pain, held_near, held_inside)
    report_metrics("mixed", mixed_risk, held_pain, held_near, held_inside)
    report_metrics("residual", residual_risk, held_pain, held_near, held_inside)
    report_metrics("continuous", continuous_risk, held_pain, held_near, held_inside)
    report_metrics("layered", layered_risk, held_pain, held_near, held_inside)

    rng = np.random.RandomState(seed + 900)
    permutation_importance(
        "global", lambda x: global_head.probability(x),
        held_obs, held_pain, rng,
    )
    permutation_importance(
        "mixed", memory.risk, held_obs, held_pain,
        np.random.RandomState(seed + 901),
    )
    permutation_importance(
        "residual", residual_memory.risk, held_obs, held_pain,
        np.random.RandomState(seed + 902),
    )
    permutation_importance(
        "continuous", continuous.risk, held_obs, held_pain,
        np.random.RandomState(seed + 903),
    )
    permutation_importance(
        "layered", layered.risk, held_obs, held_pain,
        np.random.RandomState(seed + 904),
    )

    print("\nlayered nodes (highest valence):")
    for row in layered.describe(OBS_NAMES, limit=16):
        print("  %-18s x %-18s v=%.3f support=%.0f pain=%.3f lift=%.3f"
              % (row["first"], row["second"], row["valence"], row["support"],
                 row["pain_rate"], row["lift"]))
    print("\ncontinuous nodes (highest valence):")
    for row in continuous.describe(OBS_NAMES, limit=12):
        print("  %-18s x %-18s v=%.3f support=%.0f pain=%.3f lift=%.3f"
              % (row["first"], row["second"], row["valence"], row["support"],
                 row["pain_rate"], row["lift"]))
    print("\nmixed nodes (highest valence):")
    for row in memory.describe(OBS_NAMES, limit=12):
        print("  %-18s x %-18s v=%.3f support=%.0f pain=%.3f lift=%.3f"
              % (row["first"], row["second"], row["valence"], row["support"],
                 row["pain_rate"], row["lift"]))
    print("\nresidual nodes (highest valence):")
    for row in residual_memory.describe(OBS_NAMES, limit=12):
        print("  %-18s x %-18s v=%.3f support=%.0f pain=%.3f lift=%.3f"
              % (row["first"], row["second"], row["valence"], row["support"],
                 row["pain_rate"], row["lift"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.coverage <= 1.0:
        raise ValueError("--coverage must be in [0, 1]")
    run(args.seed, args.steps, args.coverage)


if __name__ == "__main__":
    main()
