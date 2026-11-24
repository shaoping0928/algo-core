"""Decision-relevant look-ahead: clearance at the next pipe.

Full-state rollout error is not the right target.  What matters is whether the
current state carries enough information to predict the clearance margin at the
next pipe, tens of frames ahead.  This probe measures exactly that: the true gap
is already observable, so the uncertain part is the bird's height when the pipe
arrives.  A ridge readout on the current observation is compared with the
myopic margin (current dy) at several look-ahead bins.

Probe-only: pipe labels are used for measurement, never as a learner input.
"""
import argparse
import json

import numpy as np

from flappy import GAP, SPEED
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


def ridge_fit(train_x, train_y, ridge=1.0):
    train_x = np.asarray(train_x, dtype=float)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0) + 1e-9
    design = np.column_stack([
        np.ones(len(train_x)), (train_x - mean) / scale,
    ])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty, design.T @ np.asarray(train_y, float),
    )
    return mean, scale, weights


def ridge_predict(model, test_x):
    mean, scale, weights = model
    test_x = np.asarray(test_x, dtype=float)
    design = np.column_stack([
        np.ones(len(test_x)), (test_x - mean) / scale,
    ])
    return design @ weights


def episode_frame(rows):
    observations = np.asarray(
        [rows[0]["obs"]] + [row["next_obs"] for row in rows], dtype=float,
    )
    done = np.asarray([bool(row.get("done")) for row in rows])
    return observations, done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+",
                        default=[120, 121, 122, 123, 124, 125, 126, 127])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--bins", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
    )
    episodes = []
    for eval_seed in args.eval_seeds:
        with _curiosity_off(agent):
            _length, _actions, rows = evaluate(
                args.seed, world, agent, cap=int(args.cap),
                horizon=int(args.horizon), lag=LAG, step_stride=2,
                env_seed=int(eval_seed) * 10000, decision="search",
                beam_width=0, record=True,
            )
        observations, done = episode_frame(rows)
        episodes.append((observations, done))
    edges = [0] + [int(value) for value in args.bins]
    rows_by_bin = {index: [] for index in range(len(edges) - 1)}
    censored = 0
    for episode_index, (observations, done) in enumerate(episodes):
        for start in range(len(observations) - 1):
            dx = float(observations[start][2])
            if dx <= 0.0:
                continue
            time_to_pipe = dx / SPEED
            future = start + int(round(time_to_pipe))
            if future >= len(observations):
                censored += 1
                continue
            y = float(observations[start][0]) + 0.5
            gap = y + float(observations[start][3])
            if bool(done[start:future].any()):
                clear = False
            else:
                y_future = float(observations[future][0]) + 0.5
                clear = abs(gap - y_future) <= GAP / 2.0
            myopic = -abs(float(observations[start][3]))
            for index in range(len(edges) - 1):
                low, high = edges[index], edges[index + 1]
                if low <= time_to_pipe < high:
                    rows_by_bin[index].append((
                        episode_index,
                        observations[start][:6].copy(),
                        float(myopic),
                        bool(clear),
                    ))
                    break
    output = []
    for index in range(len(edges) - 1):
        rows = rows_by_bin[index]
        if not rows:
            continue
        episodes_in = sorted({row[0] for row in rows})
        fold_of = {episode: position % 5
                   for position, episode in enumerate(episodes_in)}
        train = [row for row in rows if fold_of[row[0]] != 0]
        test = [row for row in rows if fold_of[row[0]] == 0]
        if not train or not test:
            continue
        labels_train = np.asarray([row[3] for row in train], dtype=float)
        labels_test = np.asarray([row[3] for row in test], dtype=bool)
        model = ridge_fit([row[1] for row in train], labels_train)
        predicted = ridge_predict(model, [row[1] for row in test])
        myopic = np.asarray([row[2] for row in test], dtype=float)
        output.append(dict(
            low=float(edges[index]), high=float(edges[index + 1]),
            samples=int(len(rows)), test=int(len(test)),
            clear=int(labels_test.sum()),
            base_rate=float(labels_test.mean()) if len(labels_test) else 0.0,
            myopic_auc=auc(myopic, labels_test),
            readout_auc=auc(predicted, labels_test),
        ))
    report = dict(
        seed=int(args.seed),
        config=dict(
            eval_seeds=[int(v) for v in args.eval_seeds],
            cap=int(args.cap), horizon=int(args.horizon),
            steps=int(args.steps), epochs=int(args.epochs),
            max_nodes=int(args.max_nodes),
            gap=float(GAP), speed=float(SPEED),
        ),
        censored=int(censored),
        bins=output,
    )
    print("clearance look-ahead: seed %d | gap %.3f | speed %.4f"
          % (args.seed, GAP, SPEED))
    print("  censored %d" % censored)
    for row in output:
        print(
            "  %d-%d frames: n %d (test %d, clear %d, base %.2f) | "
            "myopic AUC %s | readout AUC %s"
            % (
                row["low"], row["high"], row["samples"], row["test"],
                row["clear"], row["base_rate"], row["myopic_auc"],
                row["readout_auc"],
            )
        )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
