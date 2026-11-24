"""Look-ahead diagnosis: does the model see beyond the current pipe?

Flappy exposes the current pipe (dx, dy) and the next pipe (next_dx, next_dy).
The world model is trained one frame at a time; this probe asks whether its
rollout stays accurate far enough to carry the next-pipe information, whether
the runtime hazard anticipates a death before that pipe, and whether the stage
type retains the next-pipe coordinates at all.

Probe-only: the environment's pipe labels are used for measurement, never as a
learner input.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from flappy import GAP, SPEED
from stage_type_cluster_probe import (
    integrated,
    key_weights,
    split,
    stage_seeds,
)
from stage_type_clusters import StageSample, StageTypeClusters
from static_graph_strengths import learn_static_strengths
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


HORIZONS = (1, 2, 4, 8, 16, 32, 64)
PIPE_BINS = ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10_000))
PIPE_DIMS = (2, 3, 4, 5)


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    sorted_values = combined[order]
    ranks = np.empty(len(combined), dtype=float)
    start = 0
    while start < len(combined):
        end = start
        while (end + 1 < len(combined)
               and sorted_values[end + 1] == sorted_values[start]):
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return float(
        (ranks[:len(positive)].sum()
         - len(positive) * (len(positive) + 1) / 2.0)
        / (len(positive) * len(negative))
    )


def ridge_r2(train_x, train_y, test_x, test_y, ridge=1.0):
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0) + 1e-9
    train = np.column_stack([np.ones(len(train_x)),
                             (train_x - mean) / scale])
    test = np.column_stack([np.ones(len(test_x)),
                            (test_x - mean) / scale])
    penalty = ridge * np.eye(train.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        train.T @ train + penalty, train.T @ np.asarray(train_y, float),
    )
    predicted = test @ weights
    residual = float(((test_y - predicted) ** 2).sum())
    total = float(((test_y - np.mean(test_y)) ** 2).sum())
    return 1.0 - residual / total if total > 0.0 else 0.0


def eta_squared(types, values):
    groups = defaultdict(list)
    for type_id, value in zip(types, values):
        if type_id is not None:
            groups[type_id].append(float(value))
    if not groups:
        return None
    all_values = np.concatenate([np.asarray(v) for v in groups.values()])
    total = float(((all_values - all_values.mean()) ** 2).sum())
    if total <= 0.0:
        return 0.0
    within = 0.0
    for group in groups.values():
        group = np.asarray(group)
        within += float(((group - group.mean()) ** 2).sum())
    return 1.0 - within / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int,
                        nargs="+", default=[120, 121, 122, 123, 124, 125])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--strength-min-support", type=int, default=2)
    parser.add_argument("--min-similarity", type=float, default=0.7)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
    )
    scales = np.asarray(agent.scales, dtype=float)
    obs_dim = int(agent.obs_dim)
    horizons = HORIZONS
    errors = {k: [] for k in horizons}
    all_errors = {k: [] for k in horizons}
    hazard_rows = []
    baseline_rows = []
    for eval_seed in args.eval_seeds:
        with _curiosity_off(agent):
            _length, _actions, rows = evaluate(
                args.seed, world, agent, cap=int(args.cap),
                horizon=int(args.horizon), lag=LAG, step_stride=2,
                env_seed=int(eval_seed) * 10000, decision="search",
                beam_width=0, record=True,
            )
        baseline_rows.append(rows)
        observations = np.asarray(
            [rows[0]["obs"]] + [row["next_obs"] for row in rows],
            dtype=float,
        )
        for start in range(len(rows)):
            predicted = observations[start].copy()
            for step in range(max(horizons)):
                index = start + step
                if index >= len(rows):
                    break
                history = np.asarray(
                    rows[index]["history"], dtype=float,
                ).reshape(1, -1)
                predicted = world.predict(
                    predicted.reshape(1, -1), history,
                )[0]
                horizon = step + 1
                if horizon not in errors or start + horizon >= len(observations):
                    continue
                truth = observations[start + horizon]
                errors[horizon].append(
                    np.abs(predicted[list(PIPE_DIMS)]
                           - truth[list(PIPE_DIMS)])
                )
                all_errors[horizon].append(np.abs(predicted - truth))
            dx = float(observations[start][2])
            time_to_pipe = max(dx, 0.0) / SPEED
            pending = np.asarray(
                rows[start]["history"], dtype=float,
            )[1:LAG + 1][::-1]
            state = np.concatenate([
                observations[start][:obs_dim], pending,
            ]) / scales
            hazard = float(agent.pain_signal_batch(state.reshape(1, -1))[0])
            span = min(len(rows), start + int(time_to_pipe) + 2)
            death_index = next(
                (index for index in range(start, span)
                 if bool(rows[index].get("done"))),
                None,
            )
            collision = False
            if death_index is not None:
                death_obs = observations[death_index]
                collision = (
                    abs(float(death_obs[2])) < 0.06
                    and abs(float(death_obs[3])) > GAP / 2.0
                )
            hazard_rows.append((
                time_to_pipe, hazard, bool(collision),
            ))

    rollouts = {}
    for horizon in horizons:
        if errors[horizon]:
            matrix = np.asarray(errors[horizon], dtype=float)
            per_dim = matrix.mean(axis=0)
            rollouts[int(horizon)] = dict(
                pipe_mae=float(per_dim.mean()),
                all_mae=float(np.mean(all_errors[horizon])),
                dx=float(per_dim[0]),
                dy=float(per_dim[1]),
                next_dx=float(per_dim[2]),
                next_dy=float(per_dim[3]),
            )
    bins = []
    for low, high in PIPE_BINS:
        subset = [
            (hazard, died) for time_to_pipe, hazard, died in hazard_rows
            if low <= time_to_pipe < high
        ]
        if not subset:
            continue
        scores = [hazard for hazard, _died in subset]
        labels = [died for _hazard, died in subset]
        bins.append(dict(
            low=float(low), high=float(high), samples=int(len(subset)),
            positives=int(sum(labels)),
            hazard_mean=float(np.mean(scores)),
            auc=auc(scores, labels),
        ))

    # Does the stage type retain the next-pipe coordinates?
    episodes = list(agent.stage_experience.episodes)
    seeds = stage_seeds(agent, episodes)
    flat = [state for episode in seeds for state in episode]
    strengths = learn_static_strengths(
        flat, graph=getattr(agent, "graph", None),
        min_support=int(args.strength_min_support),
        membership_strength=1.0, normalized=True,
    )
    states = integrated(agent, seeds, strengths, args.depth)
    node_ids = []
    seen = set()
    for episode_states in states:
        for state in episode_states:
            for node in state.nodes:
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    node_ids.append(node.node_id)
    weights = key_weights(states)
    weight_vector = np.asarray(
        [weights.get(node, 0.0) for node in node_ids], dtype=float,
    )
    samples = []
    raw_entries = []
    next_pipes = []
    for episode_index, (episode_states, episode) in enumerate(
            zip(states, episodes)):
        for index, state in enumerate(episode_states):
            profile = state.vector(node_ids) * weight_vector
            if not np.any(profile):
                continue
            entry = np.asarray(
                episode[index].entry_state, dtype=float,
            ).reshape(-1)
            samples.append(StageSample(
                episode=episode_index, profile=profile,
                outcome=(), key=tuple(episode[index].key),
            ))
            raw_entries.append(entry[:obs_dim])
            next_pipes.append(entry[4:6])
    train, validation = split(samples)
    train_entries = [entry for sample, entry in zip(samples, raw_entries)
                     if sample.episode % 5 != 4]
    test_entries = [entry for sample, entry in zip(samples, raw_entries)
                    if sample.episode % 5 == 4]
    train_pipes = [pipe for sample, pipe in zip(samples, next_pipes)
                   if sample.episode % 5 != 4]
    test_pipes = [pipe for sample, pipe in zip(samples, next_pipes)
                  if sample.episode % 5 == 4]
    clusters = StageTypeClusters.fit(
        train, validation, min_similarity=float(args.min_similarity),
        min_type_support=2,
    )
    type_ids = [
        clusters.assign(sample.profile)[0] for sample in validation
    ]
    validation_pipes = np.asarray([
        pipe for sample, pipe in zip(samples, next_pipes)
        if sample.episode % 5 == 4
    ])
    raw_r2 = [
        ridge_r2(train_entries, [p[dim] for p in train_pipes],
                 test_entries, [p[dim] for p in test_pipes])
        for dim in range(2)
    ]
    type_eta = [
        eta_squared(type_ids, validation_pipes[:, dim])
        for dim in range(2)
    ]
    report = dict(
        seed=int(args.seed),
        config=dict(
            eval_seeds=[int(v) for v in args.eval_seeds],
            cap=int(args.cap), horizon=int(args.horizon),
            steps=int(args.steps), depth=int(args.depth),
        ),
        rollouts=rollouts,
        hazard_bins=bins,
        next_pipe=dict(
            raw_r2=raw_r2,
            type_eta_squared=type_eta,
            types=int(len(clusters.types)),
            validation=int(len(validation)),
        ),
    )
    print("look-ahead diagnosis: seed %d | %d worlds"
          % (args.seed, len(args.eval_seeds)))
    print("  rollout MAE (pipe dims / all): %s" % (rollouts,))
    print("  hazard by frames-to-pipe: %s" % (bins,))
    print("  next pipe readout: %s" % (report["next_pipe"],))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
