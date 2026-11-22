"""Stages as state-space classes, not time segments.

The temporal segmenter latches onto the one discontinuity with the largest
residual gain, and that discontinuity is the world's periodic reset: it
sits at the same frame in every episode, so it carries no decision
information.  The fix is to invert the object: a stage is a *class of
states*, and a transition is the state crossing from one class to another.

This probe partitions the encoded state space directly and checks the
three properties a usable class-defined stage must have:

* **dwell** - how long a visit lasts (decision scale means ~8-24 frames);
* **state timing** - the frame at which a class is entered should vary
  across episodes; classes entered at the same frame every time are the
  clock again;
* **outcome variance** - does the class explain the eight-step pain label
  (eta squared), i.e. is the partition decision-relevant at all.

Read-only: k-means over one recorded evaluation stream.
"""

import argparse

import numpy as np

from stage_graph import _kmeans
from substrate_nodes import LAG, _curiosity_off, _encoded_pending, evaluate, fit


def labels_for(rows, agent, count):
    obs = np.asarray([row["obs"] for row in rows], dtype=float)
    pending = _encoded_pending(rows, LAG)
    encoded = np.concatenate([obs, pending], axis=1) / agent.scales
    return np.asarray(_kmeans(encoded, count, seed=0), dtype=int), encoded


def runs(labels):
    """(label, start, length) for every maximal run of one label."""
    out = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            out.append((int(labels[start]), start, index - start))
            start = index
    return out


def horizon_labels(rows, horizon=8):
    episodes = {}
    for index, row in enumerate(rows):
        episodes.setdefault(int(row["episode"]), []).append(index)
    y = np.zeros(len(rows), dtype=float)
    for indices in episodes.values():
        for position, index in enumerate(indices):
            window = indices[position:position + horizon]
            if any(float(rows[other]["signals"][1]) > 0.5 for other in window):
                y[index] = 1.0
    return y


def eta_squared(labels, y):
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        members = y[labels == label]
        between += len(members) * (members.mean() - y.mean()) ** 2
    return float(between / total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--counts", default="4,8,16,32")
    parser.add_argument("--cap", type=int, default=5000)
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    eval_seeds = parse_seed_spec(args.eval_seeds)
    counts = [int(value) for value in args.counts.split(",")]

    world, agent = fit(
        seed=args.seed, steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=8, step_stride=2,
    )
    rows = []
    with _curiosity_off(agent):
        for eval_seed in eval_seeds:
            _frames, _actions, transitions = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=8,
                lag=LAG, step_stride=2, env_seed=eval_seed * 10000,
                record=True,
            )
            for row in transitions:
                row = dict(row)
                row["episode"] = int(eval_seed)
                rows.append(row)
    y = horizon_labels(rows)
    base = float(y.mean())
    print("== state partition: seed %d, %d frames, %d episodes, "
          "y8 base %.3f ==" % (args.seed, len(rows), len(eval_seeds), base))
    print("  k   dwell-p50  in-window  cells  shared  clock-like  eta2(y8)")

    for count in counts:
        labels, _encoded = labels_for(rows, agent, count)
        episode_labels = {}
        offset = 0
        for eval_seed in eval_seeds:
            size = sum(
                1 for row in rows if int(row["episode"]) == int(eval_seed)
            )
            episode_labels[int(eval_seed)] = labels[offset:offset + size]
            offset += size
        dwells = []
        entries = {}
        for episode, block in episode_labels.items():
            for label, start, length in runs(block):
                dwells.append(length)
                entries.setdefault(label, []).append(start)
        shared = sum(
            1 for label, starts in entries.items()
            if len(starts) >= 2 and label in set(labels)
        )
        clock_like = sum(
            1 for label, starts in entries.items()
            if len(starts) >= 2 and float(np.std(starts)) < 5.0
        )
        print("  %2d  %10.1f  %9.0f%%  %5d  %6d  %9d  %8.3f" % (
            count, float(np.median(dwells)), 100.0 * float(np.mean([
                8 <= length <= 24 for length in dwells
            ])), int(len(np.unique(labels))), shared, clock_like,
            eta_squared(labels, y),
        ))


if __name__ == "__main__":
    main()
