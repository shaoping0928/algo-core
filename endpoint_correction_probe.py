"""B1-plan: can a predicted-endpoint readout correct plan/suffix ranking?

B0.6 showed that the risk column is effectively an endpoint terminal hazard
and that it is inverted because the world rollout's endpoint drifts: the
selected suffix looks safer at its predicted endpoint, yet it truly dies while
the same-action-best suffix survives.  The next question is whether a small
readout, trained on the planner's *predicted* endpoints and the true terminal
outcome, can learn that correction.

This probe is deliberately offline and read-only:

* it replays the B0 decision protocol (seed 101, eval 120-124) and keeps the
  candidate traces;
* for every feasible candidate it builds features from the predicted endpoint,
  the endpoint's OOD distance, and optionally the fine score vector;
* the label is ``true terminal within the planner horizon`` from an oracle
  replay of that candidate;
* every decision is held out by evaluation world (leave-one-world-out), so a
  decision's own world never contributes to its model or scaler;
* it compares fine-only, endpoint-only, endpoint+OOD, endpoint+fine and a
  label-shuffled placebo, and reports same-action suffix ranking, regret,
  calibration and OOD rates.

No model enters the default path.  The output is a gate: if the endpoint
readout cannot beat the fine score out of world, or cannot beat the shuffled
placebo, plan-level endpoint re-ranking is not justified yet.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from decision_b0_probe import (
    _nearest_distances, ood_reference, run as b0_run, training_index,
)
from substrate_nodes import LAG


FEATURE_ARMS = ("endpoint", "endpoint_ood", "endpoint_fine", "fine")
MODEL_ARMS = (
    "endpoint", "endpoint_ood", "endpoint_fine", "fine", "shuffle",
    "feature_shuffle",
)
ARM_NAMES = {
    "baseline": "A0 fine",
    "endpoint": "A1 endpoint",
    "endpoint_ood": "A2 endpoint+ood",
    "endpoint_fine": "A3 endpoint+fine",
    "fine": "A5 fine-only",
    "shuffle": "A4 shuffled-label",
    "feature_shuffle": "A4b shuffled-feature",
}


def sigmoid(value):
    value = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(features, labels, weights=None, l2=0.01, iterations=60):
    """Weighted ridge logistic regression with Newton steps."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if weights is None:
        weights = np.ones(len(features), dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / max(float(weights.sum()), 1e-12)
    design = np.concatenate(
        [features, np.ones((len(features), 1))], axis=1
    )
    coefficients = np.zeros(design.shape[1], dtype=float)
    ridge = np.eye(design.shape[1], dtype=float) * float(l2)
    ridge[-1, -1] = 0.0
    for _ in range(int(iterations)):
        probability = sigmoid(design @ coefficients)
        residual = (probability - labels) * weights
        gradient = design.T @ residual + ridge @ coefficients
        curvature = design.T @ (
            design * (probability * (1.0 - probability) * weights)[:, None]
        ) + ridge
        try:
            step = np.linalg.solve(
                curvature + 1e-9 * np.eye(curvature.shape[0]), gradient
            )
        except np.linalg.LinAlgError:
            step = gradient
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return coefficients


def predict_logistic(coefficients, features):
    features = np.asarray(features, dtype=float)
    design = np.concatenate(
        [features, np.ones((len(features), 1))], axis=1
    )
    return sigmoid(design @ np.asarray(coefficients, dtype=float))


def standardize(train, test):
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    positive = int(np.sum(labels > 0.5))
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average
        start = end
    positive_rank = float(ranks[labels > 0.5].sum())
    return (
        positive_rank - positive * (positive + 1) / 2.0
    ) / float(positive * negative)


def calibration_error(probabilities, labels, bins=10):
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (
            probabilities < high if high < 1.0 else probabilities <= high
        )
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(
            float(probabilities[mask].mean() - labels[mask].mean())
        )
    return float(error)


def calibration_slope(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    logit = np.log(np.clip(probabilities, 1e-6, 1.0 - 1e-6) / np.clip(
        1.0 - probabilities, 1e-6, 1.0
    ))
    design = np.concatenate(
        [logit.reshape(-1, 1), np.ones((len(logit), 1))], axis=1
    )
    try:
        coefficients = np.linalg.lstsq(design, labels, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return float(coefficients[0])


def calibration_metrics(probabilities, labels, distances=None, threshold=None):
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    base = float(labels.mean()) if len(labels) else 0.0
    brier = float(np.mean((probabilities - labels) ** 2)) if len(labels) else None
    base_brier = (
        float(np.mean((base - labels) ** 2)) if len(labels) else None
    )
    log_loss = (
        float(np.mean(-labels * np.log(np.clip(probabilities, 1e-12, 1.0))
                       - (1.0 - labels)
                       * np.log(np.clip(1.0 - probabilities, 1e-12, 1.0))))
        if len(labels) else None
    )
    ood_rate = None
    brier_in = None
    brier_out = None
    if distances is not None and threshold is not None:
        distances = np.asarray(distances, dtype=float)
        ood = distances > float(threshold)
        ood_rate = float(ood.mean()) if len(ood) else None
        if (~ood).any():
            brier_in = float(np.mean(
                (probabilities[~ood] - labels[~ood]) ** 2
            ))
        if ood.any():
            brier_out = float(np.mean(
                (probabilities[ood] - labels[ood]) ** 2
            ))
    return dict(
        candidates=int(len(labels)),
        base_rate=base,
        brier=brier,
        brier_skill=(
            1.0 - brier / base_brier
            if brier is not None and base_brier and base_brier > 0.0 else None
        ),
        log_loss=log_loss,
        auc=rank_auc(probabilities, labels),
        ece=calibration_error(probabilities, labels),
        slope=calibration_slope(probabilities, labels),
        ood_rate=ood_rate,
        brier_in_distribution=brier_in,
        brier_ood=brier_out,
    )


def within_decision_metrics(predictions, horizon):
    """Ranking and calibration *inside* each decision, not pooled across it.

    A pooled AUC is inflated when late states have a higher death rate: a model
    can learn the decision-level base rate and still rank candidates at chance.
    The same-action task is within-decision, so this is the gate that matters.
    """
    decision_auc = []
    same_auc = []
    brier_skill = []
    prediction_std = []
    top1 = 0
    counted = 0
    for entry in predictions.values():
        item = entry["decision"]
        probability = np.asarray(entry["prediction"], dtype=float)
        labels = (
            np.asarray(item["true_value"], dtype=int) <= int(horizon)
        ).astype(float)
        local = {
            int(index): position
            for position, index in enumerate(item["indices"])
        }
        same_positions = np.asarray([
            local[int(index)] for index in np.flatnonzero(item["same_mask"])
        ], dtype=int)
        same_mask_local = np.isin(
            np.arange(len(probability)), same_positions
        )
        counted += 1
        if np.unique(labels).size == 2:
            auc = rank_auc(probability, labels)
            if auc is not None:
                decision_auc.append(auc)
        same_labels = labels[same_mask_local]
        if np.unique(same_labels).size == 2:
            auc = rank_auc(probability[same_mask_local], same_labels)
            if auc is not None:
                same_auc.append(auc)
        base = float(labels.mean())
        brier = float(np.mean((probability - labels) ** 2))
        base_brier = float(np.mean((base - labels) ** 2))
        if base_brier > 0.0:
            brier_skill.append(1.0 - brier / base_brier)
        prediction_std.append(float(probability.std()))
        if same_positions.size:
            best_position = choose(
                probability, same_mask_local, item["scores"]
            )
            chosen_value = int(item["true_value"][best_position])
            if chosen_value == item["same_best_value"]:
                top1 += 1
    return dict(
        decisions=int(counted),
        decision_auc_mean=(
            float(np.mean(decision_auc)) if decision_auc else None
        ),
        decision_auc_count=int(len(decision_auc)),
        same_action_auc_mean=(
            float(np.mean(same_auc)) if same_auc else None
        ),
        same_action_auc_count=int(len(same_auc)),
        within_brier_skill_mean=(
            float(np.mean(brier_skill)) if brier_skill else None
        ),
        prediction_std_mean=(
            float(np.mean(prediction_std)) if prediction_std else None
        ),
        same_action_top1=int(top1),
    )


def choose(prediction, mask, scores):
    """Lowest predicted terminal risk; ties break on the fine cost."""
    indices = np.flatnonzero(mask)
    order = np.lexsort((scores[indices, 0], prediction[indices]))
    return int(indices[order[0]])


def build_dataset(record, row, index, reference, obs_dim):
    """Every feasible candidate of one decision, with endpoint features."""
    trace = record["trace"]
    sequences = np.asarray(record["sequences"], dtype=int)
    scores = np.asarray(record["scores"], dtype=float)
    feasible = np.asarray(record["feasible"], dtype=bool)
    finite = np.asarray(trace["finite"], dtype=bool)
    usable = feasible & finite
    if not usable.any():
        return None
    selected_match = np.flatnonzero(np.all(
        sequences == np.asarray(record["result"].actions, dtype=int), axis=1
    ))
    if not len(selected_match):
        return None
    selected_index = int(selected_match[0])
    selected_action = int(sequences[selected_index, 0])
    same_mask = usable & (sequences[:, 0] == selected_action)
    if not same_mask.any():
        return None
    true_value = np.asarray(record["true_value"], dtype=int)
    endpoints = (
        np.asarray(trace["final_states"], dtype=float)[:, :int(obs_dim)]
        / index["scale"]
    )
    distances = _nearest_distances(index["states"], endpoints)
    global_best = int(true_value[usable].max())
    same_best = int(true_value[same_mask].max())
    predicted_death = np.asarray(trace["first_death"], dtype=int)
    indices = np.flatnonzero(usable)
    return dict(
        eval_seed=int(record["eval_seed"]),
        point=int(record["point"]),
        selected_index=selected_index,
        selected_action=selected_action,
        selected_value=int(true_value[selected_index]),
        same_best_value=same_best,
        global_best_value=global_best,
        decision_relevant=bool(row["decision_relevant"]),
        class_name=row["class"],
        indices=indices,
        same_mask=same_mask,
        endpoint=endpoints[indices],
        distance=distances[indices],
        scores=scores[indices],
        true_value=true_value[indices],
        predicted_death=predicted_death[indices],
    )


def feature_matrix(decision, arm):
    endpoint = decision["endpoint"]
    distance = decision["distance"].reshape(-1, 1)
    scores = decision["scores"]
    if arm == "endpoint":
        return endpoint
    if arm == "endpoint_ood":
        return np.concatenate([endpoint, distance], axis=1)
    if arm == "endpoint_fine":
        return np.concatenate([endpoint, distance, scores], axis=1)
    if arm == "fine":
        return scores
    raise ValueError("unknown feature arm %s" % arm)


def decision_metrics(decision, chosen_index, chosen_value):
    true_value = decision["true_value"]
    selected_value = decision["selected_value"]
    predicted_death = decision["predicted_death"]
    local_index = int(np.flatnonzero(decision["indices"] == chosen_index)[0])
    return dict(
        chosen_index=int(chosen_index),
        chosen_value=int(chosen_value),
        same_action=bool(decision["same_mask"][chosen_index]),
        same_action_best=bool(
            decision["same_mask"][chosen_index]
            and chosen_value == decision["same_best_value"]
        ),
        global_best=bool(chosen_value == decision["global_best_value"]),
        same_action_regret=int(decision["same_best_value"] - chosen_value),
        global_regret=int(decision["global_best_value"] - chosen_value),
        relevant_miss=bool(
            chosen_value < decision["global_best_value"]
            and predicted_death[local_index] < 0
        ),
        changed=bool(chosen_value != selected_value),
        improved=bool(chosen_value > selected_value),
        worsened=bool(chosen_value < selected_value),
    )


def run_arm(arm, decisions, folds, feature_arm, horizon,
            shuffle_labels=False, shuffle_features=False, seed=20261123):
    rng = np.random.RandomState(int(seed))
    predictions = {}
    for fold in folds:
        train = [item for item in decisions if item["eval_seed"] != fold]
        test = [item for item in decisions if item["eval_seed"] == fold]
        if not train or not test:
            continue
        train_features = np.concatenate([
            feature_matrix(item, feature_arm) for item in train
        ], axis=0)
        train_labels = np.concatenate([
            (item["true_value"] <= int(horizon)).astype(float)
            for item in train
        ])
        weights = np.concatenate([
            np.full(len(item["indices"]), 1.0 / len(item["indices"]))
            for item in train
        ])
        if shuffle_labels:
            offset = 0
            shuffled = []
            for item in train:
                size = len(item["indices"])
                block = train_labels[offset:offset + size].copy()
                shuffled.append(rng.permutation(block))
                offset += size
            train_labels = np.concatenate(shuffled)
        if shuffle_features:
            offset = 0
            shuffled_features = []
            for item in train:
                size = len(item["indices"])
                block = train_features[offset:offset + size].copy()
                shuffled_features.append(block[rng.permutation(size)])
                offset += size
            train_features = np.concatenate(shuffled_features)
        test_features = np.concatenate([
            feature_matrix(item, feature_arm) for item in test
        ], axis=0)
        train_scaled, test_scaled = standardize(
            train_features, test_features
        )
        coefficients = fit_logistic(
            train_scaled, train_labels, weights=weights,
        )
        test_predictions = predict_logistic(coefficients, test_scaled)
        offset = 0
        for item in test:
            size = len(item["indices"])
            predictions[(item["eval_seed"], item["point"])] = {
                "prediction": test_predictions[offset:offset + size],
                "decision": item,
            }
            offset += size
    return predictions


def evaluate_predictions(predictions, decisions):
    rows = []
    for item in decisions:
        key = (item["eval_seed"], item["point"])
        if key not in predictions:
            continue
        entry = predictions[key]
        prediction = entry["prediction"]
        scores = item["scores"]
        # item["indices"] are global trace indices; map local positions.
        local = {
            int(index): position
            for position, index in enumerate(item["indices"])
        }
        same_local = np.asarray([
            local[int(index)] for index in np.flatnonzero(item["same_mask"])
        ], dtype=int)
        same_choice = choose(
            prediction, np.isin(np.arange(len(item["indices"])), same_local),
            scores,
        )
        global_choice = choose(
            prediction, np.ones(len(item["indices"]), dtype=bool), scores,
        )
        chosen_index = int(item["indices"][same_choice])
        row = dict(
            eval_seed=item["eval_seed"], point=item["point"],
            class_name=item["class_name"],
            decision_relevant=item["decision_relevant"],
            baseline_value=item["selected_value"],
            same_action_choice_value=int(item["true_value"][same_choice]),
            global_choice_value=int(item["true_value"][global_choice]),
            same_action_best_value=item["same_best_value"],
            global_best_value=item["global_best_value"],
            same_action_changed=bool(
                chosen_index != item["selected_index"]
            ),
            same_action_improved=bool(
                item["true_value"][same_choice] > item["selected_value"]
            ),
            same_action_worsened=bool(
                item["true_value"][same_choice] < item["selected_value"]
            ),
            same_action_oracle=bool(
                item["true_value"][same_choice] == item["same_best_value"]
            ),
            global_oracle=bool(
                item["true_value"][global_choice] == item["global_best_value"]
            ),
            same_action_regret=int(
                item["same_best_value"] - item["true_value"][same_choice]
            ),
            global_regret=int(
                item["global_best_value"] - item["true_value"][global_choice]
            ),
            relevant_miss=bool(
                item["true_value"][same_choice] < item["global_best_value"]
                and item["predicted_death"][same_choice] < 0
            ),
        )
        rows.append(row)
    return rows


def summarize_arm(rows):
    improved = sum(row["same_action_improved"] for row in rows)
    worsened = sum(row["same_action_worsened"] for row in rows)
    return dict(
        decisions=int(len(rows)),
        same_action_oracle=int(sum(
            row["same_action_oracle"] for row in rows
        )),
        same_action_regret_mean=(
            float(np.mean([row["same_action_regret"] for row in rows]))
            if rows else None
        ),
        same_action_regret_max=(
            int(max(row["same_action_regret"] for row in rows))
            if rows else None
        ),
        global_oracle=int(sum(row["global_oracle"] for row in rows)),
        global_regret_mean=(
            float(np.mean([row["global_regret"] for row in rows]))
            if rows else None
        ),
        relevant_misses=int(sum(
            row["relevant_miss"] for row in rows
        )),
        changed=int(sum(row["same_action_changed"] for row in rows)),
        improved=int(improved),
        worsened=int(worsened),
        flip_precision=(
            float(improved) / float(improved + worsened)
            if improved + worsened else None
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--cache", default="")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    b0_args = argparse.Namespace(
        seed=args.seed, eval_seeds=args.eval_seeds,
        branches=args.branches, tail=args.tail,
        horizon=args.horizon, margin=args.margin,
        neighbours=64, min_support=4, min_episodes=2,
        ood_quantile=0.95, ood_high_quantile=0.99,
        cache=args.cache, refit=bool(args.refit), report="",
    )
    records = []
    context = {}
    b0_summary = b0_run(
        b0_args, trace_out=records, context_out=context,
    )
    agent = context["agent"]
    obs_dim = int(agent.obs_dim)
    index = training_index(
        agent.milestone_training_rows, obs_dim, LAG, agent.scales,
    )
    reference = ood_reference(index)
    row_map = {
        (row["eval_seed"], row["point"]): row
        for row in b0_summary["rows"]
    }
    decisions = []
    for record in records:
        key = (record["eval_seed"], record["point"])
        if key not in row_map:
            continue
        built = build_dataset(
            record, row_map[key], index, reference, obs_dim,
        )
        if built is not None:
            decisions.append(built)
    folds = sorted(set(item["eval_seed"] for item in decisions))
    skipped = int(len(records) - len(decisions))
    print(
        "== B1-plan endpoint correction: %d decisions, %d candidates, "
        "%d folds, %d B0 decisions skipped ==" % (
            len(decisions),
            sum(len(item["indices"]) for item in decisions),
            len(folds), skipped,
        )
    )

    summaries = {}
    calibration = {}
    rows_by_arm = {}
    within = {}
    predictions_by_arm = {}
    for arm in MODEL_ARMS:
        feature_arm = (
            "endpoint" if arm in ("shuffle", "feature_shuffle") else arm
        )
        predictions = run_arm(
            arm, decisions, folds, feature_arm, args.horizon,
            shuffle_labels=(arm == "shuffle"),
            shuffle_features=(arm == "feature_shuffle"),
        )
        predictions_by_arm[arm] = predictions
        rows_by_arm[arm] = evaluate_predictions(predictions, decisions)
        summaries[arm] = summarize_arm(rows_by_arm[arm])
        within[arm] = within_decision_metrics(predictions, args.horizon)
        probabilities = []
        labels = []
        distances = []
        for key, entry in predictions.items():
            item = entry["decision"]
            probabilities.append(entry["prediction"])
            labels.append((item["true_value"] <= args.horizon).astype(float))
            distances.append(item["distance"])
        if probabilities:
            calibration[arm] = calibration_metrics(
                np.concatenate(probabilities), np.concatenate(labels),
                np.concatenate(distances), reference["threshold"],
            )

    # A0 is the fine score itself: choose the best same-action plan by score.
    baseline_rows = []
    for item in decisions:
        prediction = item["scores"][:, 0]
        same_choice = choose(
            prediction,
            np.asarray([
                item["same_mask"][int(index)] for index in item["indices"]
            ], dtype=bool),
            item["scores"],
        )
        chosen_index = int(item["indices"][same_choice])
        baseline_rows.append(dict(
            eval_seed=item["eval_seed"], point=item["point"],
            class_name=item["class_name"],
            decision_relevant=item["decision_relevant"],
            baseline_value=item["selected_value"],
            same_action_choice_value=int(item["true_value"][same_choice]),
            global_choice_value=item["selected_value"],
            same_action_best_value=item["same_best_value"],
            global_best_value=item["global_best_value"],
            same_action_changed=bool(
                chosen_index != item["selected_index"]
            ),
            same_action_improved=False,
            same_action_worsened=False,
            same_action_oracle=bool(
                item["true_value"][same_choice] == item["same_best_value"]
            ),
            global_oracle=bool(
                item["selected_value"] == item["global_best_value"]
            ),
            same_action_regret=int(
                item["same_best_value"] - item["true_value"][same_choice]
            ),
            global_regret=int(
                item["global_best_value"] - item["selected_value"]
            ),
            relevant_miss=bool(
                item["true_value"][same_choice] < item["global_best_value"]
                and item["predicted_death"][same_choice] < 0
            ),
        ))
    summaries["baseline"] = summarize_arm(baseline_rows)
    rows_by_arm["baseline"] = baseline_rows

    print(
        "   candidate death base rate %.3f, endpoint OOD rate %.3f" % (
            float(np.mean(np.concatenate([
                (item["true_value"] <= args.horizon).astype(float)
                for item in decisions
            ]))),
            calibration["endpoint"]["ood_rate"]
            if "endpoint" in calibration else float("nan"),
        )
    )
    print(
        "   arm                  same-oracle  same-regret  global-oracle  "
        "rel-miss  changed  improved  worsened  flip-prec"
    )
    for arm in ("baseline",) + MODEL_ARMS:
        row = summaries[arm]
        print(
            "   %-20s %2d/%-2d        %6.3f       %2d/%-2d         "
            "%2d        %2d        %2d         %2d        %s" % (
                ARM_NAMES.get(arm, arm),
                row["same_action_oracle"], row["decisions"],
                row["same_action_regret_mean"] or 0.0,
                row["global_oracle"], row["decisions"],
                row["relevant_misses"], row["changed"],
                row["improved"], row["worsened"],
                "None" if row["flip_precision"] is None
                else "%.2f" % row["flip_precision"],
            )
        )
    print(
        "   calibration (held-out candidates):"
    )
    for arm in MODEL_ARMS:
        if arm not in calibration:
            continue
        row = calibration[arm]
        print(
            "   %-20s base %.3f  brier %.4f  skill %s  auc %s  "
            "ece %.3f  slope %s  ood %.3f" % (
                ARM_NAMES[arm], row["base_rate"], row["brier"],
                "None" if row["brier_skill"] is None
                else "%+.3f" % row["brier_skill"],
                "None" if row["auc"] is None else "%.3f" % row["auc"],
                row["ece"],
                "None" if row["slope"] is None else "%.3f" % row["slope"],
                row["ood_rate"] if row["ood_rate"] is not None else 0.0,
            )
        )

    print(
        "   within-decision ranking (the gate that matters):"
    )
    for arm in MODEL_ARMS:
        row = within[arm]
        print(
            "   %-20s dec-AUC %s (%d)  same-AUC %s (%d)  "
            "within-Brier-skill %s  p-std %.4f  top1 %d/%d" % (
                ARM_NAMES[arm],
                "None" if row["decision_auc_mean"] is None
                else "%.3f" % row["decision_auc_mean"],
                row["decision_auc_count"],
                "None" if row["same_action_auc_mean"] is None
                else "%.3f" % row["same_action_auc_mean"],
                row["same_action_auc_count"],
                "None" if row["within_brier_skill_mean"] is None
                else "%+.3f" % row["within_brier_skill_mean"],
                row["prediction_std_mean"] or 0.0,
                row["same_action_top1"], row["decisions"],
            )
        )
    miss_rows = [
        row for row in rows_by_arm["baseline"]
        if row["decision_relevant"]
    ]
    if miss_rows:
        print("   B0 misses: baseline same-oracle %d/%d" % (
            sum(row["same_action_oracle"] for row in miss_rows),
            len(miss_rows),
        ))
        for arm in MODEL_ARMS:
            subset = [
                row for row in rows_by_arm[arm]
                if row["decision_relevant"]
            ]
            print(
                "     %-20s same-oracle %d/%d  same-regret %.3f  "
                "changed %d  improved %d  worsened %d" % (
                    ARM_NAMES[arm],
                    sum(row["same_action_oracle"] for row in subset),
                    len(subset),
                    float(np.mean([
                        row["same_action_regret"] for row in subset
                    ])) if subset else 0.0,
                    sum(row["same_action_changed"] for row in subset),
                    sum(row["same_action_improved"] for row in subset),
                    sum(row["same_action_worsened"] for row in subset),
                )
            )

    report = dict(
        decisions=len(decisions),
        skipped=skipped,
        candidates=int(sum(len(item["indices"]) for item in decisions)),
        folds=folds,
        arms=summaries,
        calibration=calibration,
        within=within,
        rows=rows_by_arm,
        baseline_rows=baseline_rows,
        b0=dict(
            decisions=b0_summary["decisions"],
            misses=b0_summary["misses"],
            plan_level=b0_summary["plan_level_misses"],
        ),
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return report


if __name__ == "__main__":
    main()
