"""Feasibility probe for the stage graph (read-only).

The stage-graph direction needs three things to be estimable from real data:

1. how many stages a real episode splits into (BIC-optimal, action-conditioned
   local transition model), and how long they are;
2. whether ``(stage, action)`` cells get enough support for a readout;
3. whether ``(stage, action) -> next stage`` edges get enough support, and
   how often the pair is unseen (the abstain rate).

The probe runs the real benchmark fit path (1500 offline frames + one 5000
frame DAgger round), captures the exact transition streams passed to each
``_train`` call, and segments them offline.  No planner, no counterfactual
rows: only the recorded real transitions are used.

Stage model: per segment, one mean delta per action; a segment is split when
the BIC of one model minus the BIC of two models is positive (MDL penalty
included).  Stage identity is then the single-link cluster of segment mean
deltas at a data-driven radius (median pairwise distance by default).

Usage::

    python stage_support_probe.py --seed 100 --report stage_support.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

import substrate_nodes as sn
from eval_protocol import parse_seed_spec
from substrate_nodes import LAG, collect, fit


def segment_cost(deltas, actions, start, end):
    """BIC of one action-conditioned mean-delta model on deltas[start:end]."""
    length = end - start
    if length <= 0:
        return 0.0
    rss = 0.0
    for action in (0, 1):
        indices = [
            index for index in range(start, end)
            if actions[index] == action
        ]
        if not indices:
            continue
        mean = deltas[indices].mean(axis=0)
        rss += float(((deltas[indices] - mean) ** 2).sum())
    parameters = 2 * deltas.shape[1]
    return (
        length * np.log(rss / max(length, 1) + 1e-12)
        + parameters * np.log(max(length, 1))
    )


def segment_episode(deltas, actions, min_length=5, gains=None):
    """Greedy BIC splitting; returns [(start, end), ...].

    When `gains` is a list, every accepted split appends ``(cut, gain)`` so
    the caller can inspect how decisive each boundary was.
    """
    length = len(deltas)
    if length < 2 * min_length:
        return [(0, length)]
    boundaries = [0, length]
    stack = [(0, length)]
    while stack:
        start, end = stack.pop()
        if end - start < 2 * min_length:
            continue
        base = segment_cost(deltas, actions, start, end)
        best_gain = 0.0
        best_cut = None
        for cut in range(start + min_length, end - min_length + 1):
            gain = base - (
                segment_cost(deltas, actions, start, cut)
                + segment_cost(deltas, actions, cut, end)
            )
            if gain > best_gain:
                best_gain = gain
                best_cut = cut
        if best_cut is not None:
            boundaries.append(best_cut)
            if gains is not None:
                gains.append((best_cut, best_gain))
            stack.append((start, best_cut))
            stack.append((best_cut, end))
    boundaries = sorted(set(boundaries))
    return [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]


def single_link_clusters(vectors, radius):
    """Union-find single-link clustering of segment mean vectors."""
    count = len(vectors)
    if count == 0:
        return []
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(count):
        for second in range(first + 1, count):
            if np.linalg.norm(vectors[first] - vectors[second]) <= radius:
                first_root = find(first)
                second_root = find(second)
                if first_root != second_root:
                    parent[second_root] = first_root
    labels = [find(index) for index in range(count)]
    mapping = {root: index for index, root in enumerate(sorted(set(labels)))}
    return [mapping[label] for label in labels]


def analyse_stream(rows, radius=None, min_length=5):
    """Everything the stage graph would need from one real stream."""
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

    segments = []          # (episode, start, end, mean delta vector)
    episode_report = []
    split_gains = []
    short_episodes = 0
    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        deltas = np.asarray([
            (np.asarray(row["next_obs"], dtype=float)
             - np.asarray(row["obs"], dtype=float)) / scale
            for row in episode
        ])
        actions = [int(row["history"][0]) for row in episode]
        if len(episode) < 2 * min_length:
            short_episodes += 1
        found = segment_episode(
            deltas, actions, min_length=min_length, gains=split_gains
        )
        episode_report.append(dict(
            episode=episode_id,
            frames=len(episode),
            stages=len(found),
            lengths=[end - start for start, end in found],
        ))
        for start, end in found:
            segments.append(dict(
                episode=episode_id,
                start=start,
                end=end,
                mean=deltas[start:end].mean(axis=0),
                actions=[actions[index] for index in range(start, end)],
            ))

    means = np.asarray([segment["mean"] for segment in segments])
    pairwise = np.linalg.norm(
        means[:, None, :] - means[None, :, :], axis=2
    )
    upper = pairwise[np.triu_indices(len(means), 1)] if len(means) > 1 \
        else np.asarray([1.0])
    if radius is None:
        radius = float(np.median(upper))
    labels = single_link_clusters(means, radius)
    clusters = len(set(labels))

    # (stage, action) support and (stage, action, next stage) edges.
    cell_frames = Counter()
    cell_episodes = defaultdict(set)
    edge_frames = Counter()
    edge_episodes = defaultdict(set)
    for index, segment in enumerate(segments):
        stage = labels[index]
        for offset, action in enumerate(segment["actions"]):
            cell_frames[(stage, action)] += 1
            cell_episodes[(stage, action)].add(segment["episode"])
        # One edge per (stage, action, next stage); the next stage is the
        # following segment in the same episode, or END at the boundary.
        following = segments[index + 1] \
            if index + 1 < len(segments) \
            and segments[index + 1]["episode"] == segment["episode"] \
            else None
        next_stage = labels[index + 1] if following is not None else -1
        for action in sorted(set(segment["actions"])):
            edge_frames[(stage, action, next_stage)] += 1
            edge_episodes[(stage, action, next_stage)].add(
                segment["episode"]
            )

    total_frames = sum(cell_frames.values())
    cell_supports = np.asarray(sorted(cell_frames.values()), dtype=float)
    edge_supports = np.asarray(sorted(edge_frames.values()), dtype=float)
    uncovered_frames = sum(
        count for key, count in cell_frames.items()
        if cell_frames[key] < 2
    )
    uncovered_edges = sum(
        count for key, count in edge_frames.items()
        if edge_frames[key] < 2
    )
    edge_total = sum(edge_frames.values())
    top_edges = sorted(
        edge_frames.items(), key=lambda item: -item[1]
    )[:6]
    gains = np.asarray([gain for _cut, gain in split_gains], dtype=float)
    lengths = np.asarray(
        [end - start for start, end in
         [(segment["start"], segment["end"]) for segment in segments]],
        dtype=float,
    )

    return dict(
        frames=len(rows),
        episodes=len(episodes),
        short_episodes=short_episodes,
        segments=len(segments),
        clusters=clusters,
        cluster_radius=radius,
        episode_frames=[item["frames"] for item in episode_report],
        stages_per_episode=[item["stages"] for item in episode_report],
        stage_lengths=lengths.tolist(),
        stage_length_p50=float(np.percentile(lengths, 50)) if len(lengths) else 0.0,
        stage_length_p90=float(np.percentile(lengths, 90)) if len(lengths) else 0.0,
        stage_length_max=float(lengths.max()) if len(lengths) else 0.0,
        split_gains=[float(gain) for gain in gains],
        split_gain_p50=float(np.percentile(gains, 50)) if len(gains) else 0.0,
        split_gain_p90=float(np.percentile(gains, 90)) if len(gains) else 0.0,
        split_gain_min=float(gains.min()) if len(gains) else 0.0,
        multi_stage_episodes=sum(
            1 for item in episode_report if item["stages"] > 1
        ),
        cell_pairs=len(cell_frames),
        cell_support_median=float(np.median(cell_supports)),
        cell_support_p90=float(np.percentile(cell_supports, 90)),
        cell_support_ge2=sum(1 for value in cell_frames.values() if value >= 2),
        cell_support_ge5=sum(1 for value in cell_frames.values() if value >= 5),
        cell_top1_share=(
            float(max(cell_frames.values())) / float(total_frames)
            if total_frames else 0.0
        ),
        cell_action_coverage=float(np.mean([
            sum(1 for action in (0, 1) if (stage, action) in cell_frames)
            / 2.0
            for stage in range(clusters)
        ])) if clusters else 0.0,
        frame_abstain_rate=(
            float(uncovered_frames) / float(total_frames)
            if total_frames else 0.0
        ),
        edges=len(edge_frames),
        edge_support_median=float(np.median(edge_supports)),
        edge_support_ge2=sum(1 for value in edge_frames.values() if value >= 2),
        edge_support_ge5=sum(1 for value in edge_frames.values() if value >= 5),
        edge_top4_share=(
            float(sum(count for _key, count in top_edges[:4]))
            / float(edge_total)
            if edge_total else 0.0
        ),
        top_edges=[
            dict(stage=int(stage), action=int(action),
                 next_stage=int(next_stage), frames=int(count))
            for (stage, action, next_stage), count in top_edges
        ],
        edge_frame_abstain_rate=(
            float(uncovered_edges) / float(total_frames)
            if total_frames else 0.0
        ),
        cluster_sizes=[
            int(sum(1 for label in labels if label == index))
            for index in range(clusters)
        ],
    )


def run(args):
    streams = []
    original_train = sn._train

    def traced_train(transitions, **kwargs):
        world, agent = original_train(transitions, **kwargs)
        streams.append(list(transitions))
        return world, agent

    sn._train = traced_train
    try:
        fit(
            seed=args.seed, steps=args.steps, coverage=0.1,
            epochs=1, max_nodes=args.max_nodes, birth_threshold=0.45,
            curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=args.rounds, dagger_episodes=2,
            dagger_cap=args.dagger_cap, dagger_frames=args.dagger_frames,
            dagger_horizon=8, step_stride=2,
        )
    finally:
        sn._train = original_train
    return dict(
        seed=int(args.seed),
        streams=[
            dict(phase=index + 1, rows=len(stream),
                 **analyse_stream(stream, radius=args.stage_radius))
            for index, stream in enumerate(streams)
        ],
    )


def print_summary(report):
    print()
    for stream in report["streams"]:
        print(
            "== phase %d: %d frames / %d episodes =="
            % (stream["phase"], stream["frames"], stream["episodes"])
        )
        print(
            "  stages: %d segments, %d clusters (radius %.3f), "
            "multi-stage episodes %d/%d, short %d"
            % (stream["segments"], stream["clusters"],
               stream["cluster_radius"], stream["multi_stage_episodes"],
               stream["episodes"], stream["short_episodes"])
        )
        print(
            "  split gains: n=%d  p50 %.2f  p90 %.2f  min %.3f"
            % (len(stream["split_gains"]), stream["split_gain_p50"],
               stream["split_gain_p90"], stream["split_gain_min"])
        )
        print(
            "  per-episode stages: mean %.2f  max %d  "
            "stage length p50 %.0f p90 %.0f max %.0f"
            % (np.mean(stream["stages_per_episode"]),
               max(stream["stages_per_episode"]),
               stream["stage_length_p50"], stream["stage_length_p90"],
               stream["stage_length_max"])
        )
        print(
            "  (stage, action) cells: %d  support median %.0f p90 %.0f  "
            ">=2: %d  >=5: %d  both-action coverage %.0f%%  "
            "frame abstain %.0f%%"
            % (stream["cell_pairs"], stream["cell_support_median"],
               stream["cell_support_p90"], stream["cell_support_ge2"],
               stream["cell_support_ge5"],
               100.0 * stream["cell_action_coverage"],
               100.0 * stream["frame_abstain_rate"])
        )
        print(
            "  (stage, action, next) edges: %d  support median %.0f  "
            ">=2: %d  >=5: %d  top4 share %.0f%%  frame abstain %.0f%%"
            % (stream["edges"], stream["edge_support_median"],
               stream["edge_support_ge2"], stream["edge_support_ge5"],
               100.0 * stream["edge_top4_share"],
               100.0 * stream["edge_frame_abstain_rate"])
        )
        for edge in stream["top_edges"][:4]:
            print(
                "    top edge s%d a%d -> s%d  frames %d"
                % (edge["stage"], edge["action"], edge["next_stage"],
                   edge["frames"])
            )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--dagger-frames", type=int, default=5000)
    parser.add_argument("--dagger-cap", type=int, default=5000)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--stage-radius", type=float, default=None,
                        help="single-link radius for stage clusters; "
                             "default = median pairwise segment distance")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = run(args)
    print_summary(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
