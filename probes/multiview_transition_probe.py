"""Held-out validation for the action-aligned multi-view transition kernel.

The probe withholds whole realized episodes from both the state vocabulary
and the edge measure by default.  Each fold rebuilds the runtime prototypes
from training episodes only, then maps held-out states through that frozen
representation.  ``--representation-mode fixed`` is retained only to
reproduce the earlier, representation-leaky diagnostic.

No counterfactual trajectory is generated or used as a training label.  The
held-out action schedule is the sequence actually executed in that episode.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eval_protocol import parse_seed_spec
from executor_horizon_probe import horizon_config
from hierarchical_transitions import HierarchicalTransitionModel
from multiview_transitions import MultiViewTransitionModel
from probe_cache import load_or_fit
from predictive_state_quotient import PredictiveStateQuotient
from substrate_nodes import _train


def _pending(history, lag):
    history = np.asarray(history, dtype=float).reshape(-1)
    if history.size <= int(lag):
        return None
    return [int(round(float(value)))
            for value in history[1:int(lag) + 1][::-1]]


def _node(agent, observation, pending):
    encoded = np.concatenate([
        np.asarray(observation, dtype=float).reshape(-1),
        np.asarray(pending, dtype=float).reshape(-1),
    ]) / np.asarray(agent.scales, dtype=float)
    node, _distance = agent._nearest(encoded)
    return None if node is None else str(node)


def _endpoint_node(agent, rows, start, window, lag):
    pending = _pending(rows[int(start)]["history"], lag)
    if pending is None:
        return None
    for offset in range(int(window)):
        issued = int(round(float(
            np.asarray(rows[int(start) + offset]["history"])[0]
        )))
        if lag:
            pending = pending[1:] + [issued]
    final_row = rows[int(start) + int(window) - 1]
    return _node(agent, final_row["next_obs"], pending)


def _observed_outcome(rows, start, horizon):
    """Return observed duration/event, or None for a right-censored start."""
    duration = min(int(horizon), len(rows) - int(start))
    for offset in range(duration):
        if bool(rows[int(start) + offset].get("done", False)):
            return offset + 1, True
    if duration == int(horizon):
        return duration, False
    return None


def _empty_level_audit():
    return dict(
        eligible=0, mapped_targets=0, unmapped_targets=0,
        unsupported=0, no_live_prediction=0,
        all_correct=[], all_log_loss=[],
        supported_correct=[], supported_log_loss=[],
        supported_targets=[],
        fine_resolution=[],
    )


def _summarize_level_audit(item, train_counts):
    targets = item.pop("_targets", [])
    eligible = int(item["eligible"])
    mapped_targets = int(item["mapped_targets"])
    keys = sorted(train_counts)
    train_total = int(sum(train_counts.values()))
    class_count = len(keys)
    denominator = float(train_total + class_count)
    priors = {
        key: (float(train_counts[key]) + 1.0) / denominator
        for key in keys
    } if denominator > 0.0 else {}

    base_top1 = None
    base_log_loss = None
    if targets and priors:
        best = min(
            keys, key=lambda key: (-train_counts[key], str(key))
        )
        base_top1 = float(np.mean([
            str(target) == str(best) for target in targets
        ]))
        base_log_loss = float(np.mean([
            -np.log(max(priors.get(str(target), 1e-12), 1e-12))
            for target in targets
        ]))

    all_top1 = (
        float(np.mean(item["all_correct"]))
        if item["all_correct"] else None
    )
    all_log_loss = (
        float(np.mean(item["all_log_loss"]))
        if item["all_log_loss"] else None
    )
    supported_top1 = (
        float(np.mean(item["supported_correct"]))
        if item["supported_correct"] else None
    )
    supported_log_loss = (
        float(np.mean(item["supported_log_loss"]))
        if item["supported_log_loss"] else None
    )
    supported_base_rate_log_loss = (
        float(np.mean([
            -np.log(max(priors.get(str(target), 1e-12), 1e-12))
            for target in item["supported_targets"]
        ]))
        if item["supported_targets"] and priors else None
    )
    return dict(
        eligible=eligible,
        mapped_targets=mapped_targets,
        unmapped_targets=int(item["unmapped_targets"]),
        base_rate_train_n=train_total,
        base_rate_classes=class_count,
        unsupported=int(item["unsupported"]),
        no_live_prediction=int(item["no_live_prediction"]),
        coverage=(
            float(
                mapped_targets - item["unsupported"]
                - item["no_live_prediction"]
            ) / mapped_targets
            if mapped_targets else 0.0
        ),
        unknown_rate=(
            float(item["unsupported"] + item["unmapped_targets"])
            / eligible if eligible else 0.0
        ),
        top1=all_top1,
        log_loss=all_log_loss,
        supported_top1=supported_top1,
        supported_log_loss=supported_log_loss,
        supported_base_rate_log_loss=supported_base_rate_log_loss,
        supported_log_loss_gain=(
            supported_base_rate_log_loss - supported_log_loss
            if (supported_base_rate_log_loss is not None
                and supported_log_loss is not None)
            else None
        ),
        base_rate_top1=base_top1,
        base_rate_log_loss=base_log_loss,
        log_loss_gain=(
            base_log_loss - all_log_loss
            if base_log_loss is not None and all_log_loss is not None
            else None
        ),
        supported_n=int(len(item["supported_correct"])),
        fine_node_resolution=(
            float(np.mean(item["fine_resolution"]))
            if item["fine_resolution"] else None
        ),
    )


def evaluate_fold(agent, train_rows, test_rows, lag, windows, max_starts,
                  seed, n_groups=None, hierarchy_groups=None):
    quotient = None
    hierarchy_model = hierarchy_groups is not None
    if hierarchy_groups is not None:
        model = HierarchicalTransitionModel.fit_rows(
            train_rows, agent, group_counts=hierarchy_groups, seed=seed,
            lag=lag,
        )
        quotient = model.hierarchy
    else:
        node_map = None
        if n_groups is not None:
            quotient = PredictiveStateQuotient.fit_rows(
                train_rows, agent, n_groups=int(n_groups), seed=int(seed),
            )
            node_map = quotient.fine_to_coarse
        model = MultiViewTransitionModel.fit_rows(
            train_rows, agent, lag=lag, node_map=node_map,
        )
    by_episode = {}
    for row in test_rows:
        by_episode.setdefault(int(row.get("episode", 0)), []).append(row)

    metrics = {
        int(window): dict(
            terminal_nll=[], terminal_event=[], terminal_event_all=[],
            node_top1=[], node_log_loss=[], node_count=0,
            unknown=[], support=[], level_usage=[], resolution=[],
            hierarchy_level_top1={}, hierarchy_level_log_loss={},
            hierarchy_level_audit={},
        )
        for window in windows
    }
    train_by_episode = {}
    for row in train_rows:
        train_by_episode.setdefault(
            int(row.get("episode", 0)), []
        ).append(row)
    level_base_counts = {}
    if hierarchy_model:
        for window in windows:
            counts = [dict() for _ in model.models]
            for rows in train_by_episode.values():
                for start in range(max(0, len(rows) - int(window) + 1)):
                    outcome = _observed_outcome(rows, start, window)
                    if outcome != (int(window), False):
                        continue
                    target_node = _endpoint_node(
                        agent, rows, start, window, lag,
                    )
                    labels = (
                        quotient.fine_to_levels.get(target_node)
                        if target_node is not None else None
                    )
                    if labels is None:
                        continue
                    for level, label in enumerate(labels):
                        key = str(label)
                        counts[level][key] = counts[level].get(key, 0) + 1
            level_base_counts[int(window)] = counts

    rng = np.random.RandomState(int(seed))
    for episode, rows in sorted(by_episode.items()):
        if not rows:
            continue
        for window in windows:
            # Each temporal view gets its own eligible starts. Requiring a
            # complete max-window trajectory for every shorter view would
            # exclude starts that terminate inside the shorter window.
            candidates = [
                start for start in range(len(rows))
                if _pending(rows[start]["history"], lag) is not None
                and _observed_outcome(rows, start, window) is not None
            ]
            if len(candidates) > int(max_starts):
                candidates = sorted(rng.choice(
                    candidates, size=int(max_starts), replace=False
                ).tolist())
            for start in candidates:
                prefix = _pending(rows[start]["history"], lag)
                if prefix is None:
                    continue
                duration, terminal = _observed_outcome(rows, start, window)
                actions = [
                    int(round(float(
                        np.asarray(rows[start + offset]["history"])[0]
                    )))
                    for offset in range(duration)
                ]
                prediction = model.predict_views(
                    agent, rows[start]["obs"], prefix, actions,
                    range(1, duration + 1),
                )
                result = prediction[duration]
                item = metrics[int(window)]
                item["terminal_event_all"].append(int(terminal))
                item["unknown"].append(float(result["unknown"]))
                item["support"].append(int(result["supported_rows"]))
                if "level_usage" in result:
                    item["level_usage"].append(result["level_usage"])
                if result.get("resolution") is not None:
                    item["resolution"].append(float(result["resolution"]))
                if result["status"] == "supported":
                    if terminal:
                        prior = (
                            0.0 if duration == 1
                            else float(prediction[duration - 1]["terminal"])
                        )
                        event_mass = max(
                            float(result["terminal"]) - prior, 1e-12,
                        )
                        item["terminal_nll"].append(
                            -float(np.log(event_mass))
                        )
                        item["terminal_event"].append(1)
                    else:
                        survival_mass = max(
                            1.0 - float(result["terminal"]), 1e-12,
                        )
                        item["terminal_nll"].append(
                            -float(np.log(survival_mass))
                        )
                        item["terminal_event"].append(0)
                if not terminal and result["status"] == "supported":
                    target_node = _endpoint_node(
                        agent, rows, start, window, lag,
                    )
                    hierarchy = getattr(model, "hierarchy", None)
                    if hierarchy is not None:
                        actual_levels = (
                            hierarchy.fine_to_levels.get(target_node)
                            if target_node is not None else None
                        )
                        distribution = result["distribution"]
                        if actual_levels is None:
                            probability = 0.0
                            predicted_correct = False
                        else:
                            actual_keys = {
                                (level, str(label))
                                for level, label in enumerate(actual_levels)
                            }
                            probability = float(sum(
                                value for key, value in distribution.items()
                                if key in actual_keys
                            ))
                            best_key = (
                                max(distribution, key=distribution.get)
                                if distribution else None
                            )
                            predicted_correct = best_key in actual_keys
                            for level, label in enumerate(actual_levels):
                                level_distribution = {
                                    str(node): float(probability)
                                    for (pred_level, node), probability
                                    in distribution.items()
                                    if int(pred_level) == int(level)
                                }
                                level_total = sum(
                                    level_distribution.values()
                                )
                                if level_total <= 0.0:
                                    continue
                                level_probability = (
                                    level_distribution.get(str(label), 0.0)
                                    / level_total
                                )
                                item["hierarchy_level_top1"].setdefault(
                                    int(level), []
                                ).append(float(
                                    str(label) == max(
                                        level_distribution,
                                        key=level_distribution.get,
                                    )
                                ))
                                item["hierarchy_level_log_loss"].setdefault(
                                    int(level), []
                                ).append(
                                    -np.log(max(level_probability, 1e-12))
                                )
                    else:
                        if model.node_map is not None:
                            target_node = model.node_map.get(target_node)
                        distribution = result["distribution"]
                        probability = (
                            float(distribution.get(target_node, 0.0))
                            if target_node is not None else 0.0
                        )
                        predicted_correct = bool(
                            distribution
                            and target_node == max(
                                distribution, key=distribution.get
                            )
                        )
                    if probability > 0.0:
                        item["node_top1"].append(float(predicted_correct))
                        item["node_log_loss"].append(
                            -np.log(max(probability, 1e-12))
                        )
                    else:
                        item["node_top1"].append(0.0)
                        item["node_log_loss"].append(-np.log(1e-12))
                    item["node_count"] += 1

                hierarchy = getattr(model, "hierarchy", None)
                if hierarchy is not None and not terminal:
                    target_node = _endpoint_node(
                        agent, rows, start, window, lag,
                    )
                    actual_levels = (
                        hierarchy.fine_to_levels.get(target_node)
                        if target_node is not None else None
                    )
                    for level, level_model in enumerate(model.models):
                        audit = item["hierarchy_level_audit"].setdefault(
                            int(level), _empty_level_audit(),
                        )
                        audit["eligible"] += 1
                        if actual_levels is None:
                            audit["unmapped_targets"] += 1
                            continue

                        target_label = str(actual_levels[level])
                        audit["mapped_targets"] += 1
                        audit.setdefault("_targets", []).append(target_label)
                        level_result = level_model.predict_views(
                            agent, rows[start]["obs"], prefix, actions,
                            (int(window),),
                        )[int(window)]
                        distribution = level_result["distribution"]
                        is_supported = (
                            level_result["status"] == "supported"
                            and float(level_result["unknown"]) <= 1e-12
                        )
                        live_mass = float(sum(distribution.values()))
                        if not is_supported:
                            audit["unsupported"] += 1
                            audit["all_correct"].append(0.0)
                            audit["all_log_loss"].append(-np.log(1e-12))
                            continue
                        if live_mass <= 1e-12:
                            audit["no_live_prediction"] += 1
                            audit["all_correct"].append(0.0)
                            audit["all_log_loss"].append(-np.log(1e-12))
                            continue

                        probabilities = {
                            str(node): float(value) / live_mass
                            for node, value in distribution.items()
                        }
                        predicted = min(
                            probabilities,
                            key=lambda node: (-probabilities[node], node),
                        )
                        probability = probabilities.get(target_label, 0.0)
                        correct = float(predicted == target_label)
                        loss = -np.log(max(probability, 1e-12))
                        audit["all_correct"].append(correct)
                        audit["all_log_loss"].append(float(loss))
                        audit["supported_correct"].append(correct)
                        audit["supported_log_loss"].append(float(loss))
                        audit["supported_targets"].append(target_label)
                        audit["fine_resolution"].append(float(sum(
                            probabilities[node]
                            * hierarchy.class_size(level, node)
                            for node in probabilities
                        )))

    report = dict(
        model=model.summary(),
        representation_nodes=int(len(getattr(agent, "_prototype_ids", ()))),
        hierarchy_level_audit_scope=(
            "independent per-level kernels; UNKNOWN/no-live predictions "
            "count as misses; base rates use training-fold endpoints"
            if hierarchy_model else None
        ),
        quotient=(
            quotient.summary() if quotient is not None else None
        ),
        windows={},
    )
    for window, item in metrics.items():
        usage_vectors = [
            np.asarray(values, dtype=float)
            for values in item["level_usage"]
            if np.asarray(values).size
        ]
        if item["unknown"]:
            report["windows"][str(window)] = dict(
                n=int(len(item["unknown"])),
                event_rate_all=(
                    float(np.mean(item["terminal_event_all"]))
                    if item["terminal_event_all"] else None
                ),
                event_rate_supported=(
                    float(np.mean(item["terminal_event"]))
                    if item["terminal_event"] else None
                ),
                terminal_nll=(
                    float(np.mean(item["terminal_nll"]))
                    if item["terminal_nll"] else None
                ),
                terminal_nll_n=int(len(item["terminal_nll"])),
                unknown_mean=float(np.mean(item["unknown"])),
                fully_supported=float(np.mean(
                    np.asarray(item["unknown"]) <= 1e-12
                )),
                level_usage=(
                    np.mean(np.vstack(usage_vectors), axis=0).tolist()
                    if usage_vectors else None
                ),
                resolution=(
                    float(np.mean(item["resolution"]))
                    if item["resolution"] else None
                ),
                hierarchy_level_top1={
                    str(level): float(np.mean(values))
                    for level, values
                    in item["hierarchy_level_top1"].items()
                },
                hierarchy_level_n={
                    str(level): int(len(values))
                    for level, values
                    in item["hierarchy_level_top1"].items()
                },
                hierarchy_level_log_loss={
                    str(level): float(np.mean(values))
                    for level, values
                    in item["hierarchy_level_log_loss"].items()
                },
                hierarchy_level_audit={
                    str(level): _summarize_level_audit(
                        values,
                        level_base_counts.get(int(window), [{}] * (
                            max(item["hierarchy_level_audit"], default=-1) + 1
                        ))[int(level)],
                    )
                    for level, values
                    in item["hierarchy_level_audit"].items()
                },
                supported_node_n=int(item["node_count"]),
                node_top1=(
                    float(np.mean(item["node_top1"]))
                    if item["node_top1"] else None
                ),
                node_log_loss=(
                    float(np.mean(item["node_log_loss"]))
                    if item["node_log_loss"] else None
                ),
            )
        else:
            report["windows"][str(window)] = dict(n=0)
    return report


def fit_fold_representation(rows, seed, lag, max_nodes):
    """Fit the node vocabulary without any held-out episode rows."""
    _world, agent = _train(
        rows,
        seed=int(seed),
        epochs=1,
        lag=int(lag),
        max_nodes=int(max_nodes),
        birth_threshold=0.45,
        component_kinds=(),
        component_min_count=8,
        max_components=128,
        component_weight=0.5,
        component_gating=False,
        memory_weight=0.1,
        association_rate=0.0,
        induce_features=True,
        precursor_labels=False,
        pain_horizon=None,
        voi_target=3,
        feature_value_weight=0.02,
        curiosity_weight=0.5,
    )
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100")
    parser.add_argument("--windows", default="1,4,8,16")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-starts", type=int, default=2000)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument(
        "--quotient-groups", default="1,4,16",
        help=("diagnostic capacity sweep; each quotient is evaluated on "
              "held-out episodes, never installed as a fixed layer count"),
    )
    parser.add_argument(
        "--hierarchy-groups", default="16,4,1",
        help=("nested predictive-state backoff sweep, ordered fine-to-coarse; "
              "an empty value disables it"),
    )
    parser.add_argument(
        "--representation-mode", choices=("fold-refit", "fixed"),
        default="fold-refit",
        help=("fold-refit prevents held-out episodes from defining the "
              "state vocabulary; fixed reproduces the earlier diagnostic"),
    )
    parser.add_argument("--cache-prefix", default="multiview")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    windows = sorted(set(int(value) for value in args.windows.split(",")))
    if not windows or windows[0] < 1:
        raise SystemExit("windows must be positive integers")

    reports = {}
    for seed in parse_seed_spec(args.seeds):
        config = horizon_config(seed, 8)
        world, agent, cached = load_or_fit(
            config, "%s-%d-8.pkl" % (args.cache_prefix, seed),
            refit=bool(args.refit),
        )
        del world
        rows = list(getattr(agent, "milestone_training_rows", ()))
        by_episode = {}
        for row in rows:
            by_episode.setdefault(int(row.get("episode", 0)), []).append(row)
        episodes = sorted(by_episode)
        if len(episodes) < 2:
            raise SystemExit("need at least two episodes for held-out folds")
        fold_ids = np.arange(len(episodes)) % max(2, int(args.folds))
        seed_report = dict(
            model_cache="hit" if cached else "fit",
            representation_mode=args.representation_mode,
            folds={},
        )
        for fold in np.unique(fold_ids):
            held = {
                episodes[index] for index in range(len(episodes))
                if fold_ids[index] == fold
            }
            train_rows = [
                row for episode in episodes if episode not in held
                for row in by_episode[episode]
            ]
            test_rows = [
                row for episode in sorted(held)
                for row in by_episode[episode]
            ]
            fold_agent = (
                fit_fold_representation(
                    train_rows, int(seed) * 100 + int(fold),
                    int(agent.lag), args.max_nodes,
                )
                if args.representation_mode == "fold-refit"
                else agent
            )
            models = {}
            models["fine"] = evaluate_fold(
                fold_agent, train_rows, test_rows, int(agent.lag),
                windows, args.max_starts, int(seed) + int(fold),
            )
            for n_groups in _int_values(args.quotient_groups):
                name = "quotient_%d" % n_groups
                models[name] = evaluate_fold(
                    fold_agent, train_rows, test_rows, int(agent.lag),
                    windows, args.max_starts,
                    int(seed) + int(fold) + n_groups,
                    n_groups=n_groups,
                )
            hierarchy_groups = [
                int(value) for value in args.hierarchy_groups.split(",")
                if value.strip()
            ]
            if hierarchy_groups:
                name = "hierarchy_" + "_".join(
                    str(value) for value in hierarchy_groups
                )
                models[name] = evaluate_fold(
                    fold_agent, train_rows, test_rows, int(agent.lag),
                    windows, args.max_starts,
                    int(seed) + int(fold) + sum(hierarchy_groups),
                    hierarchy_groups=hierarchy_groups,
                )
            seed_report["folds"][str(int(fold))] = dict(models=models)
        reports[str(seed)] = seed_report
        print("== seed %d (%s, %d episodes) ==" % (
            seed, seed_report["model_cache"], len(episodes),
        ))
        model_names = ["fine"] + [
            "quotient_%d" % value
            for value in _int_values(args.quotient_groups)
        ]
        hierarchy_groups = [
            int(value) for value in args.hierarchy_groups.split(",")
            if value.strip()
        ]
        if hierarchy_groups:
            model_names.append(
                "hierarchy_" + "_".join(
                    str(value) for value in hierarchy_groups
                )
            )
        for model_name in model_names:
            for window in windows:
                rows_at_window = [
                    fold["models"][model_name]["windows"][str(window)]
                    for fold in seed_report["folds"].values()
                    if fold["models"][model_name]["windows"][str(window)]
                    .get("n", 0)
                ]
                if not rows_at_window:
                    print("  %-12s h%-2d no valid starts" % (
                        model_name, window,
                    ))
                    continue
                event_rows = [
                    item for item in rows_at_window
                    if item["event_rate_all"] is not None
                ]
                event_rate = (
                    float(np.average(
                        [item["event_rate_all"] for item in event_rows],
                        weights=[item["n"] for item in event_rows],
                    )) if event_rows else None
                )
                usage_rows = [
                    np.asarray(item["level_usage"], dtype=float)
                    for item in rows_at_window
                    if item.get("level_usage") is not None
                    and len(item["level_usage"]) > 1
                ]
                backoff_text = ""
                if usage_rows:
                    usage = np.mean(np.vstack(usage_rows), axis=0)
                    total = float(usage.sum())
                    if total > 0.0:
                        backoff_text = " backoff %.2f/root %.2f" % (
                            float(usage[1:].sum()) / total,
                            float(usage[-1]) / total,
                        )
                resolutions = [
                    item["resolution"] for item in rows_at_window
                    if item.get("resolution") is not None
                ]
                resolution_text = (
                    " resolution %.1f" % float(np.mean(resolutions))
                    if resolutions else ""
                )
                level_ids = sorted({
                    int(level)
                    for item in rows_at_window
                    for level in item.get("hierarchy_level_top1", {})
                })
                level_values = []
                for level in level_ids:
                    pairs = [
                        (
                            item["hierarchy_level_top1"][str(level)],
                            item["hierarchy_level_n"][str(level)],
                        )
                        for item in rows_at_window
                        if str(level) in item.get(
                            "hierarchy_level_top1", {}
                        )
                    ]
                    total_n = sum(count for _value, count in pairs)
                    if total_n:
                        accuracy = sum(
                            value * count for value, count in pairs
                        ) / float(total_n)
                        level_values.append("%d:%.2f" % (level, accuracy))
                level_text = (
                    " mixed-level-top1[" + ",".join(level_values) + "]"
                    if level_values else ""
                )
                audit_levels = sorted({
                    int(level)
                    for item in rows_at_window
                    for level in item.get("hierarchy_level_audit", {})
                })
                audit_values = []
                for level in audit_levels:
                    level_rows = [
                        item["hierarchy_level_audit"][str(level)]
                        for item in rows_at_window
                        if str(level) in item.get(
                            "hierarchy_level_audit", {}
                        )
                    ]

                    def weighted(field, weight_field):
                        pairs = [
                            (row.get(field), row.get(weight_field, 0))
                            for row in level_rows
                            if row.get(field) is not None
                            and row.get(weight_field, 0) > 0
                        ]
                        total_weight = sum(
                            count for _value, count in pairs
                        )
                        return (
                            sum(
                                value * count
                                for value, count in pairs
                            ) / float(total_weight)
                            if total_weight else None
                        )

                    eligible = sum(row["eligible"] for row in level_rows)
                    audit_values.append(
                        "L%d n=%d cov=%.2f top1=%s/base=%s "
                        "ll=%s/base=%s unk=%.2f res=%s" % (
                            level,
                            sum(row["mapped_targets"] for row in level_rows),
                            weighted("coverage", "mapped_targets") or 0.0,
                            _fmt(weighted("top1", "mapped_targets")),
                            _fmt(weighted(
                                "base_rate_top1", "mapped_targets",
                            )),
                            _fmt(weighted("log_loss", "mapped_targets")),
                            _fmt(weighted(
                                "base_rate_log_loss", "mapped_targets",
                            )),
                            (
                                sum(
                                    row["unknown_rate"] * row["eligible"]
                                    for row in level_rows
                                ) / float(eligible)
                                if eligible else 0.0
                            ),
                            _fmt(weighted(
                                "fine_node_resolution", "supported_n",
                            )),
                        )
                    )
                audit_text = (
                    " per-level{" + "; ".join(audit_values) + "}"
                    if audit_values else ""
                )
                print(
                    "  %-18s h%-2d n=%d event %.3f NLL %s "
                    "node-top1 %s logloss %s unknown %.3f support %.3f"
                    "%s%s%s%s" % (
                        model_name, window,
                        sum(item["n"] for item in rows_at_window),
                        event_rate if event_rate is not None else 0.0,
                        _fmt(_mean([
                            item["terminal_nll"] for item in rows_at_window
                            if item["terminal_nll"] is not None
                        ])),
                        _fmt(_mean([
                            item["node_top1"] for item in rows_at_window
                            if item["node_top1"] is not None
                        ])),
                        _fmt(_mean([
                            item["node_log_loss"] for item in rows_at_window
                            if item["node_log_loss"] is not None
                        ])),
                        np.mean([
                            item["unknown_mean"] for item in rows_at_window
                        ]),
                        np.mean([
                            item["fully_supported"]
                            for item in rows_at_window
                        ]),
                        backoff_text,
                        resolution_text,
                        level_text,
                        audit_text,
                    )
                )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return reports


def _fmt(value):
    return (
        "n/a" if value is None or not np.isfinite(value)
        else "%.3f" % value
    )


def _int_values(spec):
    spec = str(spec).strip().lower()
    if not spec or spec in ("none", "off"):
        return []
    return [int(value) for value in spec.split(",") if value.strip()]


def _mean(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


if __name__ == "__main__":
    main()
