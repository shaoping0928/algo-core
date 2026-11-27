"""Compare teacher-forced and free-running derived-sign prediction.

The code-level difference sign is the labeler.  At prediction time the sign is
a normal target slot: a learned threshold rule on ``vy`` (the user's
``vy > 0 -> dy > 0`` case) or a difference rule on ``y``.  The probe measures
both under observed states (teacher forced) and under the learned world
model's predicted states (free running).  Missing inputs are UNKNOWN and are
counted separately from errors.
"""

import argparse

import numpy as np

from component_derivation import (
    difference_states,
    evaluate_rule,
    fit_difference_rule,
    fit_threshold_rule,
    sign_of,
    threshold_states,
)
from query_graph import UNKNOWN, Slot
from substrate_generic import LinearObservationModel, collect
from substrate_nodes import LAG


def episode_rows(rows):
    episodes = {}
    for row in rows:
        episodes.setdefault(int(row["episode"]), []).append(row)
    return episodes


def fit_and_evaluate(rows, test_modulus=5):
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    histories = np.asarray([row["history"] for row in rows], dtype=float)
    next_observations = np.asarray(
        [row["next_obs"] for row in rows], dtype=float
    )
    world = LinearObservationModel(
        observations.shape[1], histories.shape[1]
    ).fit(observations, histories, next_observations)
    episodes = episode_rows(rows)
    train_scores, train_threshold_labels = [], []
    train_current, train_previous, train_difference_labels = [], [], []
    test_scores, test_threshold_labels = [], []
    test_current, test_previous, test_difference_labels = [], [], []
    for episode, sequence in episodes.items():
        if len(sequence) < 3:
            continue
        test = int(episode) % int(test_modulus) == int(test_modulus) - 1
        for index in range(1, len(sequence)):
            row = sequence[index]
            previous = sequence[index - 1]
            score = float(row["obs"][1])
            label = sign_of(
                float(row["next_obs"][0]) - float(row["obs"][0])
            )
            difference_label = sign_of(
                float(row["obs"][0]) - float(previous["obs"][0])
            )
            if test:
                test_scores.append(score)
                test_threshold_labels.append(label)
                test_current.append(float(row["obs"][0]))
                test_previous.append(float(previous["obs"][0]))
                test_difference_labels.append(difference_label)
            else:
                train_scores.append(score)
                train_threshold_labels.append(label)
                train_current.append(float(row["obs"][0]))
                train_previous.append(float(previous["obs"][0]))
                train_difference_labels.append(difference_label)
    threshold = fit_threshold_rule(train_scores, train_threshold_labels)
    difference = fit_difference_rule(
        train_current, train_previous, train_difference_labels
    )
    threshold_teacher = evaluate_rule(
        threshold,
        threshold_states(test_scores, Slot("score", 0)),
        test_threshold_labels,
    )
    difference_teacher = evaluate_rule(
        difference,
        difference_states(
            test_current, test_previous, Slot("value", 0)
        ),
        test_difference_labels,
    )
    threshold_free = free_run_threshold(
        world, episodes, threshold.threshold, test_modulus,
    )
    difference_free = free_run_difference(
        world, episodes, difference.threshold, test_modulus,
    )
    return dict(
        threshold=threshold,
        difference=difference,
        threshold_teacher=threshold_teacher,
        difference_teacher=difference_teacher,
        threshold_free=threshold_free,
        difference_free=difference_free,
        test_rows=len(test_scores),
    )


def free_run_threshold(world, episodes, threshold, test_modulus,
                       horizon=8):
    correct = 0
    known = 0
    unknown = 0
    for episode, sequence in episodes.items():
        if int(episode) % int(test_modulus) != int(test_modulus) - 1:
            continue
        for start in range(1, len(sequence) - horizon):
            state = np.asarray(sequence[start]["obs"], dtype=float)
            for step in range(horizon):
                row = sequence[start + step]
                predicted = world.predict(
                    state.reshape(1, -1),
                    np.asarray(row["history"], dtype=float).reshape(1, -1),
                )[0]
                label = sign_of(
                    float(row["next_obs"][0]) - float(row["obs"][0])
                )
                score = float(predicted[1])
                prediction = sign_of(score, threshold)
                known += 1
                correct += int(prediction == label)
                state = predicted
    accuracy = correct / float(known) if known else float("nan")
    return accuracy, unknown / float(max(known, 1))


def free_run_difference(world, episodes, threshold, test_modulus,
                        horizon=8):
    correct = 0
    known = 0
    unknown = 0
    for episode, sequence in episodes.items():
        if int(episode) % int(test_modulus) != int(test_modulus) - 1:
            continue
        for start in range(1, len(sequence) - horizon):
            state = np.asarray(sequence[start]["obs"], dtype=float)
            previous_y = float(sequence[start - 1]["obs"][0])
            for step in range(horizon):
                row = sequence[start + step]
                predicted = world.predict(
                    state.reshape(1, -1),
                    np.asarray(row["history"], dtype=float).reshape(1, -1),
                )[0]
                label = sign_of(
                    float(row["next_obs"][0]) - float(row["obs"][0])
                )
                prediction = sign_of(
                    float(state[0]) - float(previous_y), threshold
                )
                known += 1
                correct += int(prediction == label)
                previous_y = float(state[0])
                state = predicted
    accuracy = correct / float(known) if known else float("nan")
    return accuracy, unknown / float(max(known, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--test-modulus", type=int, default=5)
    args = parser.parse_args()
    rows, _ = collect(seed=args.seed, steps=args.steps, lag=LAG)
    result = fit_and_evaluate(rows, test_modulus=args.test_modulus)
    threshold = result["threshold"]
    difference = result["difference"]
    print("== seed %d steps %d held-out rows %d ==" % (
        args.seed, args.steps, result["test_rows"],
    ))
    print(
        "vy threshold: t=%.4f support=%d fit_acc=%.4f" % (
            threshold.threshold, threshold.support, threshold.confidence,
        )
    )
    print(
        "difference:   t=%.4f support=%d fit_acc=%.4f" % (
            difference.threshold, difference.support,
            difference.confidence,
        )
    )
    print()
    print("rule          teacher acc  teacher UNKNOWN  free acc  free UNKNOWN")
    for name, teacher, free in (
        ("vy threshold", result["threshold_teacher"],
         result["threshold_free"]),
        ("difference", result["difference_teacher"],
         result["difference_free"]),
    ):
        print(
            "%-12s %11.4f %16.4f %9.4f %12.4f" % (
                name, teacher[0], teacher[1], free[0], free[1],
            )
        )


if __name__ == "__main__":
    main()
