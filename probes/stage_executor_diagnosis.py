"""Diagnose the stage executor: policy shift and within-stage hazard.

Two questions:

* are option-executed states farther from the behaviour prototypes than the
  fallback planner's states (off-policy execution)?
* is a frame-level hazard learnable from the same state features, so a safety
  envelope could exist?

Nothing here changes a decision.
"""
import argparse
import json

import numpy as np

from stage_executor import run_stage_executor
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    return float(
        (ranks[:len(positive)].sum()
         - len(positive) * (len(positive) + 1) / 2.0)
        / (len(positive) * len(negative))
    )


def encode_rows(agent, rows, lag):
    encoded = []
    distances = []
    signals = []
    scales = np.asarray(agent.scales, dtype=float)
    obs_dim = int(agent.obs_dim)
    for row in rows:
        obs = np.asarray(row["obs"], dtype=float)[:obs_dim]
        history = np.asarray(row["history"], dtype=float)
        pending = history[1:int(lag) + 1][::-1]
        state = np.concatenate([obs, pending]) / scales
        prototype, distance = agent._nearest(state)
        encoded.append(state)
        distances.append(float(distance) if prototype is not None else np.nan)
        signals.append(float(
            agent.pain_signal_batch(state.reshape(1, -1))[0]
        ))
    return (
        np.asarray(encoded, dtype=float),
        np.asarray(distances, dtype=float),
        np.asarray(signals, dtype=float),
    )


def within_labels(rows, horizon):
    done = np.asarray([bool(row.get("done", False)) for row in rows])
    labels = np.zeros(len(done), dtype=bool)
    for index in range(len(done)):
        labels[index] = bool(done[index:index + int(horizon)].any())
    return labels


def ridge_auc(features, labels, episodes, folds=5, ridge=1.0):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    episodes = np.asarray(episodes)
    unique = sorted(set(episodes.tolist()))
    if len(unique) < folds:
        return None
    scores = []
    for fold in range(folds):
        held = {ep for index, ep in enumerate(unique) if index % folds == fold}
        train = np.asarray([ep not in held for ep in episodes])
        test = ~train
        if labels[train].sum() == 0 or labels[test].sum() == 0:
            continue
        design = np.column_stack([
            np.ones(train.sum()), features[train],
        ])
        penalty = ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(
            design.T @ design + penalty, design.T @ labels[train],
        )
        test_design = np.column_stack([
            np.ones(test.sum()), features[test],
        ])
        value = auc(test_design @ weights, labels[test])
        if value is not None:
            scores.append(value)
    return float(np.mean(scores)) if scores else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[120, 121, 122])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--hazard-horizons", type=int, nargs="+",
                        default=[1, 2, 4])
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
    )
    threshold = float(getattr(agent, "birth_threshold", 0.45))
    executor_traces = []
    baseline_rows = []
    for eval_seed in args.eval_seeds:
        env_seed = int(eval_seed) * 10000
        trace = []
        with _curiosity_off(agent):
            run_stage_executor(
                args.seed, world, agent, cap=int(args.cap),
                horizon=int(args.horizon), lag=LAG, env_seed=env_seed,
                trace_out=trace,
            )
        executor_traces.append(trace)
        with _curiosity_off(agent):
            _length, _actions, rows = evaluate(
                args.seed, world, agent, cap=int(args.cap),
                horizon=int(args.horizon), lag=LAG, step_stride=2,
                env_seed=env_seed, decision="search", beam_width=0,
                record=True,
            )
        baseline_rows.append(rows)

    def summarize(traces, key):
        values = [
            item["distance"] for trace in traces for item in trace
            if item["source"] == key and item.get("distance") is not None
        ]
        deaths = [
            item for trace in traces for item in trace
            if item["source"] == key
        ]
        if not values:
            return dict(count=0)
        values = np.asarray(values, dtype=float)
        return dict(
            count=int(len(values)),
            distance_mean=float(values.mean()),
            distance_q90=float(np.quantile(values, 0.9)),
            ood_rate=float((values > threshold).mean()),
            death_rate=(
                float(np.mean([bool(item.get("death")) for item in deaths]))
                if deaths else 0.0
            ),
        )

    option = summarize(executor_traces, "option")
    fallback = summarize(executor_traces, "fallback")
    baseline_distances = []
    baseline_features = []
    baseline_episodes = []
    baseline_signals = []
    baseline_labels = {horizon: [] for horizon in args.hazard_horizons}
    for episode_index, rows in enumerate(baseline_rows):
        features, distances, signals = encode_rows(agent, rows, LAG)
        baseline_features.append(features)
        baseline_distances.extend(distances.tolist())
        baseline_signals.append(signals)
        baseline_episodes.extend([episode_index] * len(rows))
        for horizon in args.hazard_horizons:
            baseline_labels[horizon].extend(
                within_labels(rows, horizon).tolist()
            )
    baseline_distance_values = np.asarray(baseline_distances, dtype=float)
    baseline_summary = dict(
        count=int(len(baseline_distance_values)),
        distance_mean=float(np.nanmean(baseline_distance_values)),
        distance_q90=float(np.nanquantile(baseline_distance_values, 0.9)),
        ood_rate=float(np.nanmean(
            baseline_distance_values > threshold
        )),
    )
    features = np.vstack(baseline_features)
    episodes = np.asarray(baseline_episodes)
    signals = np.concatenate(baseline_signals)
    hazards = {}
    for horizon in args.hazard_horizons:
        labels = np.asarray(baseline_labels[horizon], dtype=bool)
        hazards[int(horizon)] = dict(
            positives=int(labels.sum()),
            base_rate=float(labels.mean()) if len(labels) else 0.0,
            signal_auc=auc(signals, labels),
            readout_auc=ridge_auc(features, labels, episodes),
        )
    report = dict(
        seed=int(args.seed),
        config=dict(
            eval_seeds=[int(v) for v in args.eval_seeds],
            cap=int(args.cap), horizon=int(args.horizon),
            steps=int(args.steps), epochs=int(args.epochs),
            max_nodes=int(args.max_nodes),
            ood_threshold=float(threshold),
        ),
        option=option,
        fallback=fallback,
        baseline=baseline_summary,
        hazard=hazards,
    )
    print("stage executor diagnosis: seed %d | threshold %.3f"
          % (args.seed, threshold))
    print("  option:   %s" % (option,))
    print("  fallback: %s" % (fallback,))
    print("  baseline: %s" % (baseline_summary,))
    print("  hazard:   %s" % (hazards,))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
