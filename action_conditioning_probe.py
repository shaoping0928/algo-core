"""Does the acting action add predictive power?  LOEO + permutation test.

The stage-graph probe showed that segment/cluster/edge counts do not
separate the real stream from an action-shuffled control.  This probe runs
the test the roundtable agreed on instead: a k-step transition model with
leave-one-episode-out prediction, comparing

    A: state only            X = obs(t)
    B: state + acting actions X = obs(t) + one-hot(acting(t) .. acting(t+k-1))

and asking whether B's out-of-fold R^2 beats A by more than an
action-shuffled null.  Episodes are the independent unit, so the minimum
detectable delta-R^2 at 80% power is ~4 / N_episodes.

Usage::

    python action_conditioning_probe.py --streams stage_streams.npz \
        --k 2 --permutations 200 --report action_conditioning.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from stage_support_probe import load_streams
from substrate_nodes import LAG


def ridge_fit(features, target, alpha=1e-3):
    design = np.concatenate(
        [features, np.ones((len(features), 1))], axis=1
    )
    gram = design.T @ design + alpha * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def loeo_r2(features, target, groups):
    predictions = np.zeros_like(target)
    for group in np.unique(groups):
        train = groups != group
        test = groups == group
        weights = ridge_fit(features[train], target[train])
        design = np.concatenate(
            [features[test], np.ones((int(test.sum()), 1))], axis=1
        )
        predictions[test] = design @ weights
    sse = ((target - predictions) ** 2).sum(axis=0)
    sst = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(sst > 0.0, 1.0 - sse / np.maximum(sst, 1e-12), 0.0)
    return float(np.mean(r2))


def build_dataset(rows, k, lag=LAG):
    scale = np.asarray(
        [np.std([row["obs"][dim] for row in rows]) for dim in range(
            len(rows[0]["obs"])
        )],
        dtype=float,
    )
    scale = np.where(scale > 1e-8, scale, 1.0)
    episodes = defaultdict(list)
    for row in rows:
        episodes[int(row["episode"])].append(row)
    states = []
    actions = []
    targets = []
    groups = []
    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        observations = np.asarray(
            [row["obs"] for row in episode], dtype=float
        ) / scale
        acting = np.asarray([
            int(row["history"][lag])
            if len(row["history"]) > lag else int(row["history"][0])
            for row in episode
        ], dtype=int)
        for start in range(0, len(episode) - k, k):
            window = acting[start:start + k]
            one_hot = np.zeros(2 ** k, dtype=float)
            index = 0
            for value in window:
                index = index * 2 + int(value)
            one_hot[index] = 1.0
            states.append(observations[start])
            actions.append(one_hot)
            targets.append(observations[start + k] - observations[start])
            groups.append(episode_id)
    return (
        np.asarray(states, dtype=float),
        np.asarray(actions, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(groups, dtype=int),
        acting_episodes(episodes, k, lag),
    )


def acting_episodes(episodes, k, lag):
    """Per-episode acting sequences, for the within-episode shuffle."""
    result = {}
    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        result[episode_id] = np.asarray([
            int(row["history"][lag])
            if len(row["history"]) > lag else int(row["history"][0])
            for row in episode
        ], dtype=int)
    return result


def action_matrix(sequences, k):
    rows = []
    for episode_id in sorted(sequences):
        acting = sequences[episode_id]
        for start in range(0, len(acting) - k, k):
            one_hot = np.zeros(2 ** k, dtype=float)
            index = 0
            for value in acting[start:start + k]:
                index = index * 2 + int(value)
            one_hot[index] = 1.0
            rows.append(one_hot)
    return np.asarray(rows, dtype=float)


def run(args):
    streams = load_streams(args.streams)
    report = dict(k=int(args.k), permutations=int(args.permutations),
                  streams=[])
    for index, rows in enumerate(streams):
        states, actions, targets, groups, sequences = build_dataset(
            rows, args.k
        )
        r2_state = loeo_r2(states, targets, groups)
        r2_with_actions = loeo_r2(
            np.concatenate([states, actions], axis=1), targets, groups
        )
        real_delta = r2_with_actions - r2_state
        rng = np.random.RandomState(20261121)
        null = []
        for _ in range(args.permutations):
            shuffled = {
                episode_id: rng.permutation(values)
                for episode_id, values in sequences.items()
            }
            permuted = action_matrix(shuffled, args.k)
            if len(permuted) != len(actions):
                continue
            null.append(
                loeo_r2(
                    np.concatenate([states, permuted], axis=1),
                    targets, groups,
                ) - r2_state
            )
        null = np.asarray(null, dtype=float)
        p_value = float(
            (1.0 + float((null >= real_delta).sum())) / (1.0 + len(null))
        ) if len(null) else 1.0
        report["streams"].append(dict(
            phase=index + 1,
            frames=len(rows),
            episodes=int(len(np.unique(groups))),
            windows=int(len(targets)),
            r2_state=r2_state,
            r2_state_action=r2_with_actions,
            delta_r2=real_delta,
            null_mean=float(null.mean()) if len(null) else 0.0,
            null_sd=float(null.std()) if len(null) else 0.0,
            p_value=p_value,
            detectable_delta=(
                4.0 / float(len(np.unique(groups)))
                if len(np.unique(groups)) else 1.0
            ),
        ))
    return report


def print_summary(report):
    print()
    print("== k=%d ==" % report["k"])
    for stream in report["streams"]:
        print(
            "  phase %d: %d frames / %d episodes / %d windows"
            % (stream["phase"], stream["frames"], stream["episodes"],
               stream["windows"])
        )
        print(
            "    R2 state %.4f -> state+action %.4f  delta %.4f"
            % (stream["r2_state"], stream["r2_state_action"],
               stream["delta_r2"])
        )
        print(
            "    permutation null: mean %.4f sd %.4f  p %.3f  "
            "(detectable delta ~ %.3f)"
            % (stream["null_mean"], stream["null_sd"],
               stream["p_value"], stream["detectable_delta"])
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", required=True,
                        help="npz from stage_support_probe --dump-streams")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = run(args)
    print_summary(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
