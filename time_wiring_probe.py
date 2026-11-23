"""Diagnose where the time-in-state wiring loses quality.

This is a read-only probe.  It does not change the learner or the planner.
It checks four things separately:

1. feature wiring: the online TimeInStateEnv and the offline augmenter must
   produce the same action-age column for the same stream;
2. raw next-state gain: does the age column improve each target dimension
   under an episode-held-out ridge, or is the gain diluted across targets?
3. hazard signal: does age predict point pain or pain within eight steps
   beyond the raw observation?
4. residual structure: where in the raw next state does the age residual
   actually carry signal?

The point is to separate "the feature has no signal" from "the feature has
signal but the world-model consumer/gate is the wrong place".
"""

import argparse

import numpy as np

from substrate_generic import collect
from substrate_nodes import LAG
from time_in_state import augment_observations, make_env_factory


class StandardizedRidge:
    """Ridge regression with per-column standardisation; multi-output."""

    def __init__(self, ridge: float = 1e-3):
        self.ridge = float(ridge)
        self.mean = None
        self.scale = None
        self.weights = None

    def fit(self, matrix, target):
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        normalized = (matrix - self.mean) / self.scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        self.weights = np.linalg.solve(gram, design.T @ target)
        return self

    def predict(self, matrix):
        normalized = (
            np.asarray(matrix, dtype=float) - self.mean
        ) / self.scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        return design @ self.weights


def episode_parts(rows):
    episodes = {}
    for index, row in enumerate(rows):
        episodes.setdefault(int(row["episode"]), []).append(index)
    return episodes


def pain_labels(rows):
    labels = np.asarray([
        1 if float(row["signals"][1]) > 0.5 else 0 for row in rows
    ], dtype=float)
    pain8 = np.zeros(len(rows), dtype=float)
    for indices in episode_parts(rows).values():
        for position, index in enumerate(indices):
            pain8[index] = labels[indices[position:position + 8]].max()
    return labels, pain8


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1) > 0.5
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).mean()
    equal = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * equal)


def r2(prediction, actual):
    residual = float(((prediction - actual) ** 2).sum())
    total = float(((actual - actual.mean(axis=0)) ** 2).sum())
    return 1.0 - residual / (total + 1e-12)


def per_target_gain(base, augmented, actual):
    base_mse = ((base - actual) ** 2).mean(axis=0)
    augmented_mse = ((augmented - actual) ** 2).mean(axis=0)
    return 1.0 - augmented_mse / (base_mse + 1e-12), base_mse, augmented_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--max-age", type=int, default=16)
    args = parser.parse_args()

    rows, _ = collect(seed=args.seed, steps=args.steps, lag=LAG)
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    histories = np.asarray([row["history"] for row in rows], dtype=float)
    next_observations = np.asarray(
        [row["next_obs"] for row in rows], dtype=float
    )
    actions = np.asarray([row["history"][0] for row in rows], dtype=int)
    episodes = np.asarray([row["episode"] for row in rows])
    alive = observations[:, 6]
    step = np.zeros(len(rows), dtype=float)
    for indices in episode_parts(rows).values():
        step[indices] = np.arange(len(indices), dtype=float)
    offline = augment_observations(
        observations, actions, episodes, period=1,
        max_age=args.max_age, action_space=(1,), include_phase=False,
    )
    age = offline[:, observations.shape[1]]

    print("== time-in-state wiring seed %d steps %d ==" % (
        args.seed, args.steps,
    ))
    print("age stats: mean %.4f std %.4f unique %d min %.3f max %.3f" % (
        float(age.mean()), float(age.std()), len(np.unique(age)),
        float(age.min()), float(age.max()),
    ))
    print("corr(age, alive) %.4f  corr(age, step) %.4f  "
          "corr(alive, step) %.4f" % (
              float(np.corrcoef(age, alive)[0, 1]),
              float(np.corrcoef(age, step)[0, 1]),
              float(np.corrcoef(alive, step)[0, 1]),
          ))

    # 1. online/offline wiring check.
    factory = make_env_factory(
        period=1, lag=LAG, max_age=args.max_age,
        include_actions=True, include_phase=False, age_actions=(1,),
    )
    online, _ = collect(
        seed=args.seed, steps=args.steps, lag=LAG, env_factory=factory,
    )
    if len(online) == len(rows):
        raw_match = np.allclose(
            np.asarray([row["obs"] for row in online], dtype=float)[:, :7],
            observations, atol=1e-9,
        )
        feature_match = np.allclose(
            np.asarray([row["obs"] for row in online], dtype=float)[:, 7],
            age, atol=1e-9,
        )
        action_match = np.allclose(
            np.asarray([row["history"][0] for row in online], dtype=float),
            actions, atol=1e-9,
        )
        print("wiring: raw %s  age %s  actions %s" % (
            raw_match, feature_match, action_match,
        ))
    else:
        print("wiring: stream length mismatch %d vs %d" % (
            len(online), len(rows),
        ))

    # 2. episode-held-out raw next-state gain.
    train = episodes % 5 != 4
    test = ~train
    feature_sets = {
        "raw": observations,
        "raw+age": np.column_stack([observations, age]),
        "raw+clock": np.column_stack([observations, step]),
        "raw+alive-copy": np.column_stack([observations, alive]),
    }
    predictions = {}
    raw_train_prediction = None
    raw_test_prediction = None
    print()
    print("raw next-state, episode-held-out (standardized ridge):")
    print("feature set      heldout R2  target gains (1 - mse_aug/mse_base)")
    for name, features in feature_sets.items():
        model = StandardizedRidge().fit(
            features[train], next_observations[train]
        )
        prediction = model.predict(features[test])
        predictions[name] = prediction
        if name == "raw":
            raw_train_prediction = model.predict(features[train])
            raw_test_prediction = prediction
            gains = np.zeros(next_observations.shape[1])
        else:
            gains, _, _ = per_target_gain(
                raw_test_prediction, prediction, next_observations[test]
            )
        print(
            "%-15s %10.4f  %s" % (
                name, r2(prediction, next_observations[test]),
                " ".join("%+.4f" % value for value in gains),
            )
        )
    base_mse = ((raw_test_prediction - next_observations[test]) ** 2).mean(
        axis=0
    )
    print(
        "base target MSE " + " ".join("%.3e" % value for value in base_mse)
    )

    # 3. hazard signal: point pain and pain within eight steps.
    point_pain, pain8 = pain_labels(rows)
    print()
    print("hazard AUC, episode-held-out (ridge score):")
    print("feature set      point-pain  pain8")
    for name, features in {
        "raw": observations,
        "raw+age": np.column_stack([observations, age]),
        "age only": age.reshape(-1, 1),
        "raw+clock": np.column_stack([observations, step]),
        "clock only": step.reshape(-1, 1),
        "raw+alive-copy": np.column_stack([observations, alive]),
    }.items():
        model = StandardizedRidge().fit(features[train], point_pain[train])
        point_score = model.predict(features[test])
        model8 = StandardizedRidge().fit(features[train], pain8[train])
        pain8_score = model8.predict(features[test])
        print(
            "%-15s %10.4f %8.4f" % (
                name,
                auc(point_score, point_pain[test]),
                auc(pain8_score, pain8[test]),
            )
        )

    # 4. per-target residual gain from age alone.
    print()
    print("age residual per target (validation MSE gain):")
    residuals_train = next_observations[train] - raw_train_prediction
    residuals_test = next_observations[test] - raw_test_prediction
    residual_model = StandardizedRidge().fit(
        age[train].reshape(-1, 1), residuals_train
    )
    residual_prediction = residual_model.predict(age[test].reshape(-1, 1))
    corrected = raw_test_prediction + residual_prediction
    gains, zero_mse, corrected_mse = per_target_gain(
        raw_test_prediction, corrected, next_observations[test]
    )
    print(
        "gain " + " ".join("%+.4f" % value for value in gains)
    )
    print(
        "zero-mse " + " ".join("%.3e" % value for value in zero_mse)
    )
    print(
        "corrected-mse " + " ".join("%.3e" % value for value in corrected_mse)
    )
    correlation = []
    for target in range(residuals_test.shape[1]):
        correlation.append(float(np.corrcoef(
            residuals_test[:, target], age[test]
        )[0, 1]))
    print(
        "corr(age, residual) " + " ".join(
            "%+.4f" % value for value in correlation
        )
    )


if __name__ == "__main__":
    main()
