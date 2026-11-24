"""Accuracy-coverage curve for the stage transition memory.

A stage transition is only trustworthy on the subset of data it has support
for.  This probe fits the memory on training episodes, evaluates every
validation transition, and reports top-1 accuracy, realized-outcome likelihood
and base rate **conditional on support** (edge count, context depth, episode
support).  Coverage is the fraction of validation transitions that reach each
support level.

It also reports the frame hazard's held-out AUC by distance-to-prototype, so
the reliable safety subset is visible next to the transition subset.

Probe-only; no learner path changes.
"""
import argparse
import json

import numpy as np

from stage_outcome_memory import StageOutcomeMemory
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def split_episodes(episodes, modulus=5, held_out=4):
    train = [
        episode for index, episode in enumerate(episodes)
        if index % modulus != held_out
    ]
    validation = [
        episode for index, episode in enumerate(episodes)
        if index % modulus == held_out
    ]
    return train, validation


def realized_outcome(stages, index):
    stage = stages[index]
    if stage.terminal:
        return (None, True, False)
    if stage.censored:
        return (None, False, True)
    if index + 1 < len(stages):
        return (stages[index + 1].key, False, False)
    return (None, False, True)


def outcome_key(row):
    return (row.get("destination"), bool(row.get("terminal", False)),
            bool(row.get("censored", False)))


def transition_rows(memory, episodes):
    rows = []
    for stages in episodes:
        for index, source in enumerate(stages):
            history = tuple(stage.key for stage in stages[:index + 1])
            prediction = memory.predict(history, source.action_trace)
            realized = realized_outcome(stages, index)
            row = dict(
                supported=prediction.get("status") == "supported",
                support=int(prediction.get("support", 0)),
                depth=int(prediction.get("context_depth") or 0),
                episode_support=int(prediction.get("episode_support", 0)),
                realized=realized,
                correct=False,
                probability=0.0,
                top1=None,
            )
            if row["supported"]:
                outcomes = prediction["outcomes"]
                best = max(
                    outcomes,
                    key=lambda item: (
                        float(item["probability"]), repr(outcome_key(item)),
                    ),
                )
                row["top1"] = outcome_key(best)
                row["correct"] = bool(row["top1"] == realized)
                matched = [
                    float(item["probability"]) for item in outcomes
                    if outcome_key(item) == realized
                ]
                row["probability"] = matched[0] if matched else 0.0
            rows.append(row)
    return rows


def curve(rows, key, thresholds, total):
    output = []
    realized_counts = {}
    for row in rows:
        realized_counts[row["realized"]] = (
            realized_counts.get(row["realized"], 0) + 1
        )
    base = (
        max(realized_counts.values()) / float(total)
        if realized_counts else 0.0
    )
    for threshold in thresholds:
        subset = [row for row in rows if row[key] >= threshold]
        supported = [row for row in subset if row["supported"]]
        if not supported:
            output.append(dict(
                threshold=float(threshold), count=0,
                coverage=0.0, accuracy=None, nll=None, base=base,
            ))
            continue
        accuracy = float(np.mean([row["correct"] for row in supported]))
        probabilities = np.maximum(
            np.asarray([row["probability"] for row in supported], float),
            1e-12,
        )
        nll = float(-np.log(probabilities).mean())
        output.append(dict(
            threshold=float(threshold),
            count=int(len(supported)),
            coverage=float(len(supported)) / float(total) if total else 0.0,
            accuracy=accuracy,
            nll=nll,
            base=base,
            lift=accuracy - base,
        ))
    return output


def hazard_curve(seed, world, agent, eval_seeds, cap, horizon):
    rows = []
    for eval_seed in eval_seeds:
        with _curiosity_off(agent):
            _length, _actions, transitions = evaluate(
                seed, world, agent, cap=int(cap), horizon=int(horizon),
                lag=LAG, step_stride=2, env_seed=int(eval_seed) * 10000,
                decision="search", beam_width=0, record=True,
            )
        observations = np.asarray(
            [transitions[0]["obs"]]
            + [row["next_obs"] for row in transitions],
            dtype=float,
        )
        done = np.asarray([bool(row.get("done")) for row in transitions])
        for index, row in enumerate(transitions):
            history = np.asarray(row["history"], dtype=float)
            pending = history[1:LAG + 1][::-1]
            obs_dim = int(agent.obs_dim)
            state = np.concatenate([
                observations[index][:obs_dim], pending,
            ]) / np.asarray(agent.scales, dtype=float)
            prototype, distance = agent._nearest(state)
            hazard = float(
                agent.pain_signal_batch(state.reshape(1, -1))[0]
            )
            label = bool(done[index:index + 2].any())
            rows.append((
                float(distance) if prototype is not None else np.nan,
                hazard, label,
            ))
    values = np.asarray([row[0] for row in rows], dtype=float)
    hazards = np.asarray([row[1] for row in rows], dtype=float)
    labels = np.asarray([row[2] for row in rows], dtype=bool)
    finite = np.isfinite(values)
    if not finite.any():
        return []
    quantiles = np.quantile(values[finite], [0.25, 0.5, 0.75, 0.9])
    edges = [0.0] + list(quantiles) + [float(values[finite].max()) + 1e-6]
    output = []
    for index in range(len(edges) - 1):
        mask = (
            finite & (values >= edges[index]) & (values < edges[index + 1])
        )
        if mask.sum() < 10:
            continue
        output.append(dict(
            low=float(edges[index]), high=float(edges[index + 1]),
            samples=int(mask.sum()),
            coverage=float(mask.sum()) / float(len(rows)),
            positives=int(labels[mask].sum()),
            hazard_mean=float(hazards[mask].mean()),
            auc=_auc(hazards[mask], labels[mask]),
        ))
    return output


def _auc(scores, labels):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+",
                        default=[120, 121, 122, 123, 124, 125])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--supports", type=int, nargs="+",
                        default=[1, 2, 3, 5, 10])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episode-supports", type=int, nargs="+",
                        default=[1, 2, 3])
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
    )
    episodes = list(agent.stage_experience.episodes)
    train, validation = split_episodes(episodes)
    memory = StageOutcomeMemory().fit(train)
    rows = transition_rows(memory, validation)
    total = len(rows)
    supported = [row for row in rows if row["supported"]]
    report = dict(
        seed=int(args.seed),
        config=dict(
            steps=int(args.steps), epochs=int(args.epochs),
            max_nodes=int(args.max_nodes), cap=int(args.cap),
            horizon=int(args.horizon),
            eval_seeds=[int(v) for v in args.eval_seeds],
        ),
        episodes=dict(train=int(len(train)), validation=int(len(validation))),
        transitions=int(total),
        supported=int(len(supported)),
        supported_coverage=float(len(supported)) / float(total)
        if total else 0.0,
        supported_accuracy=(
            float(np.mean([row["correct"] for row in supported]))
            if supported else None
        ),
        support_curve=curve(rows, "support", args.supports, total),
        depth_curve=curve(rows, "depth", args.depths, total),
        episode_curve=curve(
            rows, "episode_support", args.episode_supports, total,
        ),
        hazard=hazard_curve(
            args.seed, world, agent, args.eval_seeds,
            args.cap, args.horizon,
        ),
    )
    print(
        "accuracy-coverage: seed %d | train %d / validation %d episodes | "
        "%d transitions, %d supported (%.2f), top1 %.3f"
        % (
            args.seed, len(train), len(validation), total,
            len(supported), report["supported_coverage"],
            report["supported_accuracy"] or 0.0,
        )
    )
    print("  support curve: %s" % (report["support_curve"],))
    print("  depth curve:   %s" % (report["depth_curve"],))
    print("  episode curve: %s" % (report["episode_curve"],))
    print("  hazard by distance: %s" % (report["hazard"],))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
