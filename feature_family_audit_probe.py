"""Episode-held-out audit of generic feature families.

This probe compares observation, pending-action, relational, temporal and
world-model-residual features against real future pain feedback. Feature
rules are induced on training episodes, selected on a separate calibration
episode split, and scored only on outer held-out episodes. No task labels or
geometry enter the feature construction.
"""
import argparse
import json
from collections import OrderedDict

import numpy as np

from node_features import expand_observation
from node_runtime import future_pain_labels
from predictive_feature_bank import EpisodeBalancedFeatureBank
from substrate_generic import LinearObservationModel, collect


def group_episodes(rows):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row.get("episode", 0), []).append(row)
    return grouped


def split_episode_rows(rows, outer_folds=5, outer_fold=4,
                       inner_folds=4, calibration_fold=3):
    """Split whole episodes into induction, calibration and outer test sets."""
    outer_folds = int(outer_folds)
    outer_fold = int(outer_fold)
    inner_folds = int(inner_folds)
    calibration_fold = int(calibration_fold)
    if outer_folds < 2 or not 0 <= outer_fold < outer_folds:
        raise ValueError("outer fold must be in the configured fold range")
    if inner_folds < 2 or not 0 <= calibration_fold < inner_folds:
        raise ValueError("calibration fold must be in the configured range")
    episodes = list(group_episodes(rows).items())
    induction = []
    calibration = []
    test = []
    for index, (_episode, episode_rows) in enumerate(episodes):
        if index % outer_folds == outer_fold:
            test.extend(episode_rows)
        elif index % inner_folds == calibration_fold:
            calibration.extend(episode_rows)
        else:
            induction.extend(episode_rows)
    if not induction or not calibration or not test:
        raise ValueError("episode split must populate all three partitions")
    return induction, calibration, test


def _pending(history, lag):
    values = np.asarray(history, dtype=float).reshape(-1)
    if values.size <= int(lag):
        raise ValueError("history does not contain the pending action queue")
    return values[1:int(lag) + 1][::-1]


def _stack_rows(rows, key):
    return np.asarray([row[key] for row in rows], dtype=float)


def _temporal_features(rows):
    """Episode-local first and second differences with validity flags."""
    grouped = group_episodes(rows)
    by_episode = {}
    for episode, episode_rows in grouped.items():
        observations = _stack_rows(episode_rows, "obs")
        width = observations.shape[1]
        features = np.zeros((len(episode_rows), width * 2 + 2), dtype=float)
        previous_delta = np.zeros(width, dtype=float)
        for index in range(len(episode_rows)):
            if index == 0:
                continue
            delta = observations[index] - observations[index - 1]
            features[index, :width] = delta
            features[index, width * 2] = 1.0
            if index >= 2:
                features[index, width:width * 2] = delta - previous_delta
                features[index, width * 2 + 1] = 1.0
            previous_delta = delta
        by_episode[episode] = features
    return by_episode


def _relation_features(rows):
    grouped = group_episodes(rows)
    by_episode = {}
    for episode, episode_rows in grouped.items():
        observations = _stack_rows(episode_rows, "obs")
        expanded = expand_observation(observations)
        by_episode[episode] = expanded[:, observations.shape[1]:]
    return by_episode


def _fit_world_model(rows, lag):
    if len(rows) < 2:
        raise ValueError("need at least two training transitions")
    observations = _stack_rows(rows, "obs")
    histories = _stack_rows(rows, "history")
    next_observations = _stack_rows(rows, "next_obs")
    return LinearObservationModel(
        observations.shape[1], int(lag) + 1,
    ).fit(observations, histories, next_observations, holdout=0.0)


def _residual_features(rows, model):
    """Predict each current observation from the preceding real transition."""
    grouped = group_episodes(rows)
    by_episode = {}
    for episode, episode_rows in grouped.items():
        observations = _stack_rows(episode_rows, "obs")
        width = observations.shape[1]
        features = np.zeros((len(episode_rows), width + 1), dtype=float)
        for index in range(1, len(episode_rows)):
            previous = episode_rows[index - 1]
            prediction = model.predict(
                np.asarray(previous["obs"], dtype=float).reshape(1, -1),
                np.asarray(previous["history"], dtype=float).reshape(1, -1),
            )[0]
            features[index, :width] = observations[index] - prediction
            features[index, width] = 1.0
        by_episode[episode] = features
    return by_episode


def _oof_residual_features(rows, lag, folds=4):
    """Build train features with episode-cross-fitted one-step residuals."""
    episodes = list(group_episodes(rows).items())
    output = {}
    for fold in range(int(folds)):
        held = [
            episode_rows for index, (_episode, episode_rows)
            in enumerate(episodes) if index % int(folds) == fold
        ]
        train = [
            episode_rows for index, (_episode, episode_rows)
            in enumerate(episodes) if index % int(folds) != fold
        ]
        train_rows = [row for episode_rows in train for row in episode_rows]
        if not held or len(train_rows) < 2:
            continue
        model = _fit_world_model(train_rows, lag)
        held_rows = [row for episode_rows in held for row in episode_rows]
        output.update(_residual_features(held_rows, model))
    if len(output) != len(episodes):
        raise ValueError("could not build residuals for every training episode")
    return output


def _family_features(rows, lag, residuals):
    grouped = group_episodes(rows)
    temporal = _temporal_features(rows)
    relations = _relation_features(rows)
    result = {}
    for episode, episode_rows in grouped.items():
        observations = _stack_rows(episode_rows, "obs")
        pending = np.asarray([
            _pending(row["history"], lag) for row in episode_rows
        ], dtype=float)
        state_pending = np.concatenate([observations, pending], axis=1)
        temporal_values = temporal[episode]
        residual_values = residuals[episode]
        result[episode] = dict(
            observation=observations,
            state_pending=state_pending,
            relation=relations[episode],
            temporal=temporal_values,
            prediction_error=residual_values,
            state_temporal_residual=np.concatenate([
                state_pending, temporal_values, residual_values,
            ], axis=1),
        )
    return result


def _flatten_features(episodes, features, labels, family):
    rows = []
    targets = []
    episode_ids = []
    for episode_rows in episodes:
        episode = episode_rows[0].get("episode", 0)
        rows.append(features[episode][family])
        targets.extend(labels[episode])
        episode_ids.extend([episode] * len(episode_rows))
    return (
        np.concatenate(rows, axis=0),
        np.asarray(targets, dtype=float),
        np.asarray(episode_ids),
    )


def _standardize(train_x, other_x):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train_x - mean) / scale, (other_x - mean) / scale


def _episode_labels(rows, window):
    labels = np.asarray(future_pain_labels(rows, int(window)), dtype=float)
    labels = (labels > 0.5).astype(float)
    grouped = group_episodes(rows)
    output = {}
    start = 0
    for episode, episode_rows in grouped.items():
        output[episode] = labels[start:start + len(episode_rows)]
        start += len(episode_rows)
    return output


def _feature_equivalence_signature(record, train_x, calibration_x):
    """Compare activation and empirical readout without using outer-test data."""
    train_mask = record.expression.mask(train_x)
    calibration_mask = record.expression.mask(calibration_x)
    return (
        np.packbits(train_mask).tobytes(),
        np.packbits(calibration_mask).tobytes(),
        round(float(record.probability_active), 8),
        round(float(record.probability_inactive), 8),
    )


def run_seed(seed, steps=6500, coverage=0.1, lag=3, pain_window=4,
             outer_folds=5, outer_fold=4, inner_folds=4,
             calibration_fold=3):
    rows, _action_counts = collect(
        int(seed), steps=int(steps), coverage=float(coverage), lag=int(lag),
    )
    induction_rows, calibration_rows, test_rows = split_episode_rows(
        rows, outer_folds=outer_folds, outer_fold=outer_fold,
        inner_folds=inner_folds, calibration_fold=calibration_fold,
    )
    induction_episodes = list(group_episodes(induction_rows).values())
    calibration_episodes = list(group_episodes(calibration_rows).values())
    test_episodes = list(group_episodes(test_rows).values())
    labels = _episode_labels(rows, pain_window)

    train_residuals = _oof_residual_features(
        induction_rows, lag=lag, folds=inner_folds,
    )
    train_features = _family_features(
        induction_rows, lag=lag, residuals=train_residuals,
    )
    world_model = _fit_world_model(induction_rows, lag)
    calibration_residuals = _residual_features(
        calibration_rows, world_model,
    )
    test_residuals = _residual_features(test_rows, world_model)
    calibration_features = _family_features(
        calibration_rows, lag=lag, residuals=calibration_residuals,
    )
    test_features = _family_features(
        test_rows, lag=lag, residuals=test_residuals,
    )

    family_names = (
        "observation", "state_pending", "relation", "temporal",
        "prediction_error", "state_temporal_residual",
    )
    reports = {}
    equivalent_features = {}
    for family in family_names:
        train_x, train_y, train_ids = _flatten_features(
            induction_episodes, train_features, labels, family,
        )
        calibration_x, calibration_y, calibration_ids = _flatten_features(
            calibration_episodes, calibration_features, labels, family,
        )
        test_x, test_y, test_ids = _flatten_features(
            test_episodes, test_features, labels, family,
        )
        raw_train_x = train_x.copy()
        train_x, calibration_x = _standardize(train_x, calibration_x)
        _, test_x = _standardize(raw_train_x, test_x)
        bank = EpisodeBalancedFeatureBank(
            family=family,
            min_train_episode_support=3,
            min_calibration_episode_support=3,
            min_coverage=0.05,
            prior_episodes=1.0,
            max_compose_atoms=12,
            max_frontier=64,
            bootstrap_samples=300,
            seed=int(seed),
        )
        bank.fit(
            train_x, train_y, train_ids,
            feature_names=[
                "%s_%d" % (family, index)
                for index in range(train_x.shape[1])
            ],
        )
        calibration_report = bank.calibrate(
            calibration_x, calibration_y, calibration_ids,
        )
        records_by_id = {
            record.feature_id: record for record in bank.records
        }
        for feature_id in calibration_report["frontier_ids"]:
            record = records_by_id[feature_id]
            signature = _feature_equivalence_signature(
                record, train_x, calibration_x,
            )
            equivalent_features.setdefault(signature, []).append(dict(
                family=family,
                feature_id=feature_id,
                label=record.expression.label(bank.feature_names),
                status=record.status,
            ))
        outer_test = bank.evaluate(
            test_x, test_y, test_ids,
            feature_ids=calibration_report["frontier_ids"],
        )
        selected_outer = [
            item for item in outer_test
            if item["calibration_status"] == "calibration_supported"
        ]

        def selected_mean(metric):
            values = [
                item["outer_test"].get(metric)
                for item in selected_outer
                if item["outer_test"].get(metric) is not None
            ]
            return float(np.mean(values)) if values else None

        reports[family] = dict(
            dimensions=int(train_x.shape[1]),
            bank=bank.snapshot(),
            calibration=calibration_report,
            outer_test=outer_test,
            calibrated_outer_count=int(len(selected_outer)),
            outer_test_supported=sum(
                int(item["outer_test_supported"]) for item in selected_outer
            ),
            selected_outer_mean_brier_gain=selected_mean(
                "mean_episode_brier_gain"
            ),
            selected_outer_mean_nll_gain=selected_mean(
                "mean_episode_nll_gain"
            ),
            selected_outer_mean_episode_coverage=selected_mean(
                "episode_coverage"
            ),
        )

    observation = reports["observation"]
    for metrics in reports.values():
        baseline_brier = observation["selected_outer_mean_brier_gain"]
        baseline_nll = observation["selected_outer_mean_nll_gain"]
        metrics["incremental_brier_vs_observation"] = (
            None
            if (metrics["selected_outer_mean_brier_gain"] is None
                or baseline_brier is None)
            else (
                metrics["selected_outer_mean_brier_gain"] - baseline_brier
            )
        )
        metrics["incremental_nll_vs_observation"] = (
            None
            if (metrics["selected_outer_mean_nll_gain"] is None
                or baseline_nll is None)
            else metrics["selected_outer_mean_nll_gain"] - baseline_nll
        )
    equivalence_groups = []
    for members in equivalent_features.values():
        if len(members) > 1:
            equivalence_groups.append(dict(
                representative=members[0],
                aliases=members[1:],
            ))

    return dict(
        seed=int(seed),
        config=dict(
            steps=int(steps), coverage=float(coverage), lag=int(lag),
            pain_window=int(pain_window), outer_folds=int(outer_folds),
            outer_fold=int(outer_fold), inner_folds=int(inner_folds),
            calibration_fold=int(calibration_fold),
        ),
        split=dict(
            induction_episodes=int(len(induction_episodes)),
            calibration_episodes=int(len(calibration_episodes)),
            test_episodes=int(len(test_episodes)),
            induction_rows=int(len(induction_rows)),
            calibration_rows=int(len(calibration_rows)),
            test_rows=int(len(test_rows)),
        ),
        families=reports,
        cross_family_equivalence_groups=equivalence_groups,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--steps", type=int, default=6500)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--pain-window", type=int, default=4)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--calibration-fold", type=int, default=3)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    reports = [
        run_seed(
            seed, steps=args.steps, coverage=args.coverage, lag=args.lag,
            pain_window=args.pain_window,
            outer_folds=args.outer_folds, outer_fold=args.outer_fold,
            inner_folds=args.inner_folds,
            calibration_fold=args.calibration_fold,
        )
        for seed in args.seeds
    ]
    for report in reports:
        print(
            "feature audit seed %d | split %s"
            % (report["seed"], report["split"])
        )
        for family, metrics in report["families"].items():
            print(
                "  %-24s dims=%3d candidates=%4d frontier=%3d "
                "cal=%3d cov=%s test ΔBrier=%s ΔNLL=%s "
                "vs-obs=(%s,%s) pass=%3d"
                % (
                    family, metrics["dimensions"],
                    metrics["bank"]["candidates"],
                    metrics["bank"]["frontier"],
                    metrics["bank"]["selected"],
                    "None" if metrics["selected_outer_mean_episode_coverage"]
                    is None else "%.3f"
                    % metrics["selected_outer_mean_episode_coverage"],
                    "None" if metrics["selected_outer_mean_brier_gain"]
                    is None else "%.4f"
                    % metrics["selected_outer_mean_brier_gain"],
                    "None" if metrics["selected_outer_mean_nll_gain"]
                    is None else "%.4f"
                    % metrics["selected_outer_mean_nll_gain"],
                    "None" if metrics["incremental_brier_vs_observation"]
                    is None else "%+.4f"
                    % metrics["incremental_brier_vs_observation"],
                    "None" if metrics["incremental_nll_vs_observation"]
                    is None else "%+.4f"
                    % metrics["incremental_nll_vs_observation"],
                    metrics["outer_test_supported"],
                )
            )
        print(
            "  cross-family equivalent frontier groups: %d"
            % len(report["cross_family_equivalence_groups"])
        )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
