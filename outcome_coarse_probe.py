"""Outcome-based coarse layer probe (read-only).

The stage-graph route stalled because action-relevant stage boundaries are
not detectable at the current scale.  This probe tests the alternative the
roundtable suggested: use *outcomes* (time-to-terminal) as the coarse
state, and ask whether such a layer is estimable and whether it can
discriminate actions.

Coarse state = time-to-terminal bucket (0 = terminal, 1 = 1-5 frames,
2 = 6-15, 3 = 16+).  Only the episode boundary (done) is used - no task
geometry.  Controls: action shuffle and random bucket labels (the bucket
sequence is nearly deterministic, so this control is essential).

Reported:
- cells / transitions / cross-episode informative transitions for the real
  and both controls;
- the action discrimination inside each bucket: mean time-to-terminal for
  a=0 vs a=1 (bounded by the bucket width, so a near-zero difference means
  the coarse layer cannot tell actions apart).

Usage::

    python outcome_coarse_probe.py --streams stage_streams.npz
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from stage_support_probe import load_streams
from substrate_nodes import LAG


def bucket_of(remaining):
    if remaining <= 0:
        return 0
    if remaining <= 5:
        return 1
    if remaining <= 15:
        return 2
    return 3


def buckets_for_episode(length):
    return [bucket_of(length - 1 - index) for index in range(length)]


def analyse(rows, control="none", seed=12345):
    rng = np.random.RandomState(seed)
    episodes = defaultdict(list)
    for row in rows:
        episodes[int(row["episode"])].append(row)
    cells = Counter()
    edge_frames = Counter()
    edge_episodes = defaultdict(set)
    discrimination = defaultdict(list)
    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        buckets = buckets_for_episode(len(episode))
        if control == "random_buckets":
            buckets = [int(value) for value in rng.permutation(buckets)]
        actions = [
            int(row["history"][LAG])
            if len(row["history"]) > LAG else 0
            for row in episode
        ]
        if control == "shuffle_actions":
            actions = [int(value) for value in rng.permutation(actions)]
        for index, (row, bucket, action) in enumerate(
            zip(episode, buckets, actions)
        ):
            cells[(bucket, action)] += 1
            discrimination[(bucket, action)].append(len(episode) - 1 - index)
            next_bucket = (
                buckets[index + 1] if index + 1 < len(episode) else None
            )
            edge_frames[(bucket, action, next_bucket)] += 1
            edge_episodes[(bucket, action, next_bucket)].add(episode_id)
    informative = [
        (key, count) for key, count in edge_frames.items()
        if key[2] is not None and key[2] != key[0]
    ]
    cross3 = sum(
        1 for key, count in informative
        if count >= 5 and len(edge_episodes[key]) >= 3
    )
    per_bucket = []
    for bucket in range(4):
        entry = dict(bucket=bucket)
        for action in (0, 1):
            values = discrimination.get((bucket, action), [])
            entry["a%d_frames" % action] = len(values)
            entry["a%d_mean_remaining" % action] = (
                float(np.mean(values)) if values else 0.0
            )
        entry["remaining_gap"] = (
            entry["a1_mean_remaining"] - entry["a0_mean_remaining"]
        )
        per_bucket.append(entry)
    return dict(
        control=control,
        cells=len(cells),
        transitions=len(edge_frames),
        informative=len(informative),
        cross3=cross3,
        cells_ge2=sum(1 for value in cells.values() if value >= 2),
        cells_ge5=sum(1 for value in cells.values() if value >= 5),
        per_bucket=per_bucket,
    )


def run(args):
    streams = load_streams(args.streams)
    report = dict(streams=[])
    for index, rows in enumerate(streams):
        report["streams"].append(dict(
            phase=index + 1,
            frames=len(rows),
            real=analyse(rows),
            shuffle_actions=analyse(rows, control="shuffle_actions"),
            random_buckets=analyse(rows, control="random_buckets"),
        ))
    return report


def print_summary(report):
    print()
    for stream in report["streams"]:
        print(
            "== phase %d: %d frames =="
            % (stream["phase"], stream["frames"])
        )
        for label in ("real", "shuffle_actions", "random_buckets"):
            entry = stream[label]
            print(
                "  %-16s cells %d (>=5: %d)  transitions %d  "
                "informative %d  cross3 %d"
                % (label, entry["cells"], entry["cells_ge5"],
                   entry["transitions"], entry["informative"],
                   entry["cross3"])
            )
        print("  action discrimination inside buckets:")
        for entry in stream["real"]["per_bucket"]:
            print(
                "    bucket %d: a0 %d frames (mean remaining %.1f)  "
                "a1 %d (%.1f)  gap %+.2f"
                % (entry["bucket"], entry["a0_frames"],
                   entry["a0_mean_remaining"], entry["a1_frames"],
                   entry["a1_mean_remaining"], entry["remaining_gap"])
            )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = run(args)
    print_summary(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
