"""Evaluate the prototype graph memory on held-out real Flappy transitions."""
import argparse

import numpy as np

from graph_memory import PrototypeGraphMemory
from latent_world import GAP
from substrate_generic import collect
from substrate_value import (
    TerminalRiskHead,
    TransitionReplay,
    binary_auc,
    fit_terminal_risk,
)
from trace_generic_training import OBS_NAMES, split_episodes


def metrics(name, risk, held):
    pain = np.asarray([row["terminal"] for row in held], dtype=bool)
    obs = np.asarray([row["obs"] for row in held], dtype=float)
    near = (np.abs(obs[:, 2]) < 0.20) & (obs[:, 2] > -0.10)
    inside = np.abs(obs[:, 3]) <= GAP / 2.0
    print(
        "%-8s AUC=%.3f near_auc=%.3f Brier=%.3f near_in=%.3f near_out=%.3f"
        % (
            name, binary_auc(risk, pain),
            binary_auc(risk[near], pain[near]) if near.any() else float("nan"),
            np.mean((risk - pain) ** 2),
            risk[near & inside].mean() if (near & inside).any() else float("nan"),
            risk[near & ~inside].mean() if (near & ~inside).any() else float("nan"),
        )
    )


def sensitivity(name, score, held, rng):
    obs = np.asarray([row["obs"] for row in held], dtype=float)
    pain = np.asarray([row["terminal"] for row in held], dtype=bool)
    base = binary_auc(score(obs, held), pain)
    deltas = []
    for channel in range(obs.shape[1]):
        shuffled = obs.copy()
        shuffled[:, channel] = shuffled[rng.permutation(len(shuffled)), channel]
        rows = [dict(row, obs=shuffled[index])
                for index, row in enumerate(held)]
        deltas.append(base - binary_auc(score(shuffled, rows), pain))
    print("\n%s sensitivity:" % name)
    for index in np.argsort(deltas)[::-1]:
        print("  %-4s %+.4f" % (OBS_NAMES[index], deltas[index]))


def run(seed, steps, coverage, clusters):
    rows, _counts = collect(seed, steps, coverage=coverage)
    train, held = split_episodes(rows)
    memory = PrototypeGraphMemory(clusters=clusters, seed=seed).fit(train)
    graph_risk = memory.risk(held)
    print("seed=%d train=%d held=%d graph=%s"
          % (seed, len(train), len(held), memory.snapshot()))

    obs_dim = len(rows[0]["obs"])
    global_head = TerminalRiskHead(feature_dim=obs_dim)
    replay = TransitionReplay(capacity=len(train) + 1, seed=seed)
    for row in train:
        obs = np.asarray(row["obs"], dtype=float)
        pain = float(row["terminal"])
        replay.add(obs, [1.0 - pain, pain], obs, bool(pain))
    fit_terminal_risk(global_head, replay, 4000, batch_size=128,
                      terminal_fraction=0.15)
    global_risk = global_head.probability(
        np.asarray([row["obs"] for row in held], dtype=float)
    )
    metrics("global", global_risk, held)
    metrics("graph", graph_risk, held)

    sensitivity(
        "graph", lambda _obs, rows_: memory.risk(rows_), held,
        np.random.RandomState(seed + 1000),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--clusters", type=int, default=48)
    args = parser.parse_args()
    run(args.seed, args.steps, args.coverage, args.clusters)


if __name__ == "__main__":
    main()
