"""Trace training waves for the task-agnostic prototype.

This is a diagnostic, not a controller. It reports:

1. data-collection waves: action/terminal/pipe-proximity coverage over time;
2. world-model one-step errors per observation dimension on held-out episodes;
3. value/risk training checkpoints: held-out pain/risk AUC, Brier, and value MSE.

Pipe/gap columns are used only to label diagnostics; the controller in
substrate_generic.py does not read them.
"""
import argparse

import numpy as np

from latent_world import GAP
from substrate_generic import (
    ACTION_DIM,
    LinearObservationModel,
    collect,
    history_features,
)
from substrate_value import (
    ReturnReplay,
    TerminalRiskHead,
    TransitionReplay,
    WorldFeedbackValueHead,
    discounted_returns,
    value_probe_metrics,
)

LAG = 3
OBS_NAMES = ("y", "vy", "dx1", "gap1", "dx2", "gap2")


def split_episodes(rows):
    train = [row for row in rows if row["episode"] % 5 != 4]
    held = [row for row in rows if row["episode"] % 5 == 4]
    return train, held


def report_data_waves(rows, width=500):
    print("\nData waves:")
    print("  steps  episodes  flap_rate  terminals  near_pipe  near_flap")
    for start in range(0, len(rows), width):
        chunk = rows[start:start + width]
        if not chunk:
            continue
        near = [
            row for row in chunk
            if abs(row["obs"][2]) < 0.2 and row["obs"][2] > -0.1
        ]
        episodes = len({row["episode"] for row in chunk})
        print("  %5d  %8d  %9.3f  %9d  %9d  %9d"
              % (start + len(chunk), episodes,
                 np.mean([row["history"][0] for row in chunk]),
                 sum(row["terminal"] for row in chunk), len(near),
                 sum(int(row["history"][0]) for row in near)))


def report_world(world, rows):
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    histories = np.asarray([row["history"] for row in rows], dtype=float)
    next_observations = np.asarray(
        [row["next_obs"] for row in rows], dtype=float
    )
    features = world._features(observations, histories)
    errors = np.abs(features @ world.weights - next_observations)
    print("\nWorld model one-step held-out error:")
    print("  dim   mean       max")
    for index, name in enumerate(OBS_NAMES[:errors.shape[1]]):
        print("  %-4s  %.2e  %.2e"
              % (name, errors[:, index].mean(), errors[:, index].max()))


def build_training_replays(train, priorities=None):
    obs_dim = len(train[0]["obs"])
    value_replay = ReturnReplay(capacity=len(train) + 1, seed=11)
    risk_replay = TransitionReplay(capacity=len(train) + 1, seed=12)
    by_episode = {}
    for index, row in enumerate(train):
        priority = 1.0 if priorities is None else float(priorities[index])
        by_episode.setdefault(row["episode"], []).append((index, row))
        risk = np.concatenate([row["next_obs"], row["history"]])
        risk_replay.add(
            risk, row["signals"], risk, row["terminal"], priority=priority
        )
    for episode in by_episode.values():
        rows = [row for _index, row in episode]
        targets = discounted_returns(rows, 0.99)
        for (index, row), target in zip(episode, targets):
            value = np.concatenate([row["obs"], row["history"]])
            priority = 1.0 if priorities is None else float(priorities[index])
            value_replay.add(value, target, priority=priority)
    return value_replay, risk_replay, obs_dim


def held_records(held):
    records = []
    targets = []
    by_episode = {}
    for row in held:
        by_episode.setdefault(row["episode"], []).append(row)
    for episode in by_episode.values():
        values = discounted_returns(episode, 0.99)
        for row, target in zip(episode, values):
            feature = np.concatenate([row["obs"], row["history"]])
            records.append(dict(
                features=feature,
                risk_features=np.concatenate([row["next_obs"], row["history"]]),
                terminal=row["terminal"],
                y=float(row["obs"][0] + 0.5),
                pipe_dx=float(row["obs"][2]),
                gap_dy=float(row["obs"][3]),
            ))
            targets.append(target)
    return records, targets


def report_checkpoint(wave, updates, value_head, risk_head, records, targets):
    metrics = value_probe_metrics(records, value_head, risk_head, gap=GAP)
    value_mse = float(np.mean([
        np.mean((value_head.value(row["features"]) - target) ** 2)
        for row, target in zip(records, targets)
    ])) if targets else float("nan")
    print("  %5d  %9.2e  %8.3f  %7.3f  %7.3f  %7.3f  %7.3f"
          % (updates, value_mse, metrics["pain_auc"], metrics["risk_auc"],
             metrics["near_pipe_risk_auc"], metrics["risk_brier"],
             metrics["risk_terminal"] - metrics["risk_safe"]))


def trace(seed, steps, coverage, checkpoints, batch_size=128):
    rows, counts = collect(seed, steps, coverage=coverage)
    train, held = split_episodes(rows)
    print("seed=%d transitions=%d train=%d held=%d coverage=%.2f"
          % (seed, len(rows), len(train), len(held), coverage))
    report_data_waves(rows)

    obs_dim = len(rows[0]["obs"])
    world = LinearObservationModel(obs_dim, ACTION_DIM).fit(
        [row["obs"] for row in train],
        [row["history"] for row in train],
        [row["next_obs"] for row in train],
    )
    report_world(world, held)

    value_replay, risk_replay, obs_dim = build_training_replays(train)
    value_head = WorldFeedbackValueHead(
        feature_dim=obs_dim + ACTION_DIM, discount=0.99
    )
    risk_head = TerminalRiskHead(feature_dim=obs_dim + ACTION_DIM)
    records, targets = held_records(held)

    print("\nTraining checkpoints:")
    print("  update  value_mse  pain_auc  risk_auc  near_auc  brier  separation")
    report_checkpoint(0, 0, value_head, risk_head, records, targets)
    step = 0
    per_checkpoint = max(1, checkpoints // 12)
    for wave in range(1, 13):
        for _ in range(per_checkpoint):
            value_batch = value_replay.sample(batch_size)
            if value_batch is not None:
                features, returns = value_batch
                for index in range(len(features)):
                    value_head.update_return(features[index], returns[index])
            risk_batch = risk_replay.sample(batch_size, terminal_fraction=0.15)
            if risk_batch is not None:
                risk_head.update_batch(risk_batch[0], risk_batch[3])
            step += 1
        report_checkpoint(wave, step * batch_size, value_head, risk_head,
                          records, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--checkpoints", type=int, default=6000)
    args = parser.parse_args()
    trace(args.seed, args.steps, args.coverage, args.checkpoints)


if __name__ == "__main__":
    main()
