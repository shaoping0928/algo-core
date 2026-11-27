"""E2: calibrate the span readout on the h6 consumption distribution.

E1 showed the two-level span table holds inside the planner stream, but its
passive-to-planner transfer is coverage-brittle and the tabular history
features fragment the support.  P1-D.0 showed the event-instance warning
does not transfer to the h6 executor's own evaluation stream at all
(precision 0.50 -> 0.12-0.49).  This probe closes that loop: the readout
is fitted on the distribution where it is consumed - the h6 DAgger stream -
and the threshold is calibrated on held-out episodes of that same stream,
then read on the h6 evaluation episodes (held-out worlds 120-124).

Arms (span 8/12/16, primary 12):

- ``table_deploy``: state-only table fitted on a fresh 5000-frame stream
  collected under the deployment configuration (curiosity off, same
  horizon and decision rule) - the E2.1 test of the collection-policy
  shift;
- ``table_dagger``: the same table fitted on the training-time DAgger
  stream (curiosity on);
- ``table_passive``: the same table fitted on the passive stream;
- ``logistic_deploy`` / ``logistic_dagger`` / ``logistic_passive``: the
  parametric state + lags + event-history hazard on the same three
  streams;
- ``fuse_deploy``: E2.2, the within-stream rank average of the two deploy
  readouts (``0.5 * rank(table) + 0.5 * rank(logistic)``), with tau still
  calibrated on the deploy stream's out-of-fold predictions.  Per-entry
  evaluation predictions are stored in the report so a later fusion does
  not need another collection run.

Also measured on the eval stream: reliability bins of the table readout,
and two online-usable uncertainty proxies (fine-node support and distance
to the nearest node) against the realised error - can the table know when
it is wrong?  Read-only; no score column, planner path or default changes.

Usage::

    python span_consume_probe.py --seed 100 --report consume-100.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from endpoint_correction_probe import rank_auc
from event_span_probe import _fit_blocks, _stream, _survival_blocks
from event_transition_probe import (
    EVENT_LEVELS, _oof_curve, history_features,
)
from executor_horizon_probe import horizon_config
from latent_world import latent
from outcome_level_probe import (
    clock_report, entry_labels, fold_splits,
)
from probe_cache import load_or_fit
from substrate_generic import collect
from substrate_nodes import (
    LAG, _curiosity_off, collect_planner, evaluate,
)
from tree_transition_probe import (
    HORIZON, _assign, _valid, calibration_transfer, evaluate_table,
    fit_tree, predict_tree,
)

WINDOWS = (8, 12, 16)


def load_h6(seed, args):
    config = horizon_config(seed, 6)
    world, agent, cached = load_or_fit(
        config, "%s-%d-6.pkl" % (args.cache_prefix, seed),
        refit=bool(args.refit),
    )
    print("h6 model: %s" % ("cache" if cached else "fit"))
    rows = [
        row for row in agent.milestone_training_rows
        if int(row.get("episode", 0)) >= 50000
    ]
    return world, agent, rows


def collect_deploy(seed, world, agent, args):
    """The readout's training stream under the deployment configuration.

    ``collect_planner`` is the training-time recorder; wrapping it in
    ``_curiosity_off`` reproduces the evaluation policy exactly (curiosity
    off, same horizon and decision rule, no death rules).  Episode ids use
    a fresh offset so they cannot collide with the behaviour or DAgger
    streams.
    """
    with _curiosity_off(agent):
        rows, frames = collect_planner(
            int(seed), world, agent,
            episodes=int(args.deploy_frames),
            cap=5000, horizon=6, lag=LAG, step_stride=2,
            offset=int(args.deploy_offset),
            max_frames=int(args.deploy_frames),
        )
    print(
        "deploy stream: %d rows, %d episodes, frames %s" % (
            len(rows), len(frames), "/".join(str(value) for value in frames[:6]),
        )
    )
    return rows


def eval_tables(seed, world, agent, eval_seeds):
    """Replay the h6 evaluation episodes and rebuild their frame tables."""
    tables = []
    for eval_seed in eval_seeds:
        env_seed = int(eval_seed) * 10000
        with _curiosity_off(agent):
            frames, actions, _rows = evaluate(
                int(seed), world, agent, cap=5000, horizon=6, lag=LAG,
                step_stride=2, env_seed=env_seed, decision="search",
                record=False, episode=0,
            )
        actions = np.asarray(actions, dtype=int)
        if not len(actions):
            continue
        env = latent[LAG](env_seed)
        observations = []
        done = np.zeros(len(actions), dtype=bool)
        for position, action in enumerate(actions):
            observations.append(np.asarray(env.obs(), dtype=float))
            env.step(int(action))
            done[position] = bool(env.dead)
        tables.append(dict(
            episode=int(eval_seed), obs=np.asarray(observations),
            actions=actions, done=done,
        ))
    return tables


def restandardize(entries, train_entries):
    raw = entries["phi"] * entries["scale"] + entries["mean"]
    return (raw - train_entries["mean"]) / train_entries["scale"]


def table_oof_windows(features, step, remaining, episode, windows, args,
                      folds=5):
    fold = fold_splits(episode, folds)
    output = {
        int(window): np.full(len(step), np.nan) for window in windows
    }
    for value in np.unique(fold):
        train = fold != value
        test = fold == value
        if not train.any() or not test.any():
            continue
        model = fit_tree(
            features[train], step[train], remaining[train],
            horizon=HORIZON, k_fine=args.k_fine, k_coarse=args.k_coarse,
            seed=args.seed + int(value),
        )
        for window in windows:
            output[int(window)][test] = predict_tree(
                model, features[test], window,
            )
    return output


def table_arm(train_stream, eval_stream, args):
    entries = train_stream["entries"]
    step, _duration, _ = entry_labels(
        train_stream["terminal"], entries, HORIZON,
    )
    remaining = entries["remaining"]
    features = np.asarray(entries["phi"], dtype=float)
    oof = table_oof_windows(
        features, step, remaining, entries["episode"], WINDOWS, args,
    )
    model = fit_tree(
        features, step, remaining, horizon=HORIZON,
        k_fine=args.k_fine, k_coarse=args.k_coarse, seed=args.seed,
    )
    eval_entries = eval_stream["entries"]
    eval_step, _duration, _ = entry_labels(
        eval_stream["terminal"], eval_entries, HORIZON,
    )
    eval_remaining = eval_entries["remaining"]
    eval_features = restandardize(eval_entries, entries)
    report = {}
    eval_prediction = {}
    for window in WINDOWS:
        prediction = predict_tree(model, eval_features, window)
        eval_prediction[int(window)] = prediction
        metrics = evaluate_table(
            prediction, eval_step, eval_remaining, window,
        )
        metrics.update(calibration_transfer(
            oof[window], step, remaining, window,
            prediction, eval_step, eval_remaining,
            precision_target=args.precision_target,
        ))
        report[str(window)] = metrics
    return dict(
        report=report, model=model, eval_features=eval_features,
        train_prediction={int(key): value for key, value in oof.items()},
        eval_prediction=eval_prediction,
        step=step, eval_step=eval_step,
        remaining=remaining, eval_remaining=eval_remaining,
    )


def logistic_arm(train_stream, eval_stream, args):
    entries = train_stream["entries"]
    step, duration, _ = entry_labels(
        train_stream["terminal"], entries, HORIZON,
    )
    hist_train = history_features(
        train_stream["masks"], train_stream["names"], entries,
    )
    blocks_train = [entries["phi"], entries["lags"], hist_train]
    curve = _oof_curve(
        blocks_train, step, duration, entries["episode"], HORIZON,
        args.folds, l2=args.l2, iterations=args.iterations,
    )
    coefficients, mean, scale = _fit_blocks(
        blocks_train, step, duration, HORIZON, args.l2, args.iterations,
    )
    eval_entries = eval_stream["entries"]
    eval_step, _duration, _ = entry_labels(
        eval_stream["terminal"], eval_entries, HORIZON,
    )
    eval_hist = history_features(
        eval_stream["masks"], eval_stream["names"], eval_entries,
    )
    eval_blocks = [
        restandardize(eval_entries, entries),
        eval_entries["lags"],
        eval_hist,
    ]
    eval_curve = _survival_blocks(
        coefficients, mean, scale, eval_blocks, HORIZON,
    )
    eval_remaining = eval_entries["remaining"]
    report = {}
    for window in WINDOWS:
        metrics = evaluate_table(
            eval_curve[:, int(window) - 1], eval_step, eval_remaining,
            window,
        )
        metrics.update(calibration_transfer(
            curve[:, int(window) - 1], step, entries["remaining"],
            window, eval_curve[:, int(window) - 1], eval_step,
            eval_remaining, precision_target=args.precision_target,
        ))
        report[str(window)] = metrics
    return dict(
        report=report, train_prediction=curve, eval_prediction=eval_curve,
        step=step, eval_step=eval_step,
        remaining=entries["remaining"], eval_remaining=eval_remaining,
    )


def percentile_rank(values):
    """Average-rank percentile in [0, 1]; ties share a rank, NaN stays NaN.

    Ranks are taken inside one stream, so the fusion arm compares a stream's
    own entries with each other.  That is the pre-registered
    ``rank(table_deploy)``/``rank(logistic_deploy)`` form: a scale-free
    combination that needs no shared units between an empirical table and a
    parametric curve.
    """
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan)
    keep = np.isfinite(values)
    count = int(keep.sum())
    if count == 0:
        return output
    order = np.argsort(values[keep], kind="mergesort")
    ordered = values[keep][order]
    ranks = np.empty(count, dtype=float)
    start = 0
    while start < count:
        end = start + 1
        while end < count and ordered[end] == ordered[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    ranked = np.empty(count, dtype=float)
    ranked[order] = ranks
    output[keep] = ranked / max(1.0, count - 1.0)
    return output


def blend_ranks(left, right, weight=0.5):
    """Weighted rank average; a NaN side falls back to the other side."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    both = np.isfinite(left) & np.isfinite(right)
    return np.where(
        both, float(weight) * left + (1.0 - float(weight)) * right,
        np.where(np.isfinite(left), left, right),
    )


def fusion_arm(table_output, logistic_output, args, weight=0.5):
    """E2.2: fuse the two deploy-stream readouts by within-stream rank.

    The threshold calibration (training stream) and the read (evaluation
    stream) use the same fusion, so tau still comes from the deployment
    stream's out-of-fold predictions and no evaluation label enters a fit.
    """
    report = {}
    entries = {}
    for window in WINDOWS:
        if int(window) not in table_output["train_prediction"]:
            continue
        if int(window) not in table_output["eval_prediction"]:
            continue
        train_fused = blend_ranks(
            percentile_rank(table_output["train_prediction"][int(window)]),
            percentile_rank(
                logistic_output["train_prediction"][:, int(window) - 1]
            ),
            weight,
        )
        eval_fused = blend_ranks(
            percentile_rank(table_output["eval_prediction"][int(window)]),
            percentile_rank(
                logistic_output["eval_prediction"][:, int(window) - 1]
            ),
            weight,
        )
        step = np.asarray(table_output["eval_step"])
        metrics = evaluate_table(
            eval_fused, step, table_output["eval_remaining"], window,
        )
        metrics.update(calibration_transfer(
            train_fused, table_output["step"],
            table_output["remaining"], window,
            eval_fused, table_output["eval_step"],
            table_output["eval_remaining"],
            precision_target=args.precision_target,
        ))
        report[str(window)] = metrics
        entries[str(window)] = dict(
            prediction=eval_fused.tolist(),
            table=table_output["eval_prediction"][int(window)].tolist(),
            logistic=logistic_output["eval_prediction"][
                :, int(window) - 1
            ].tolist(),
            step=step.tolist(),
            remaining=np.asarray(
                table_output["eval_remaining"]
            ).tolist(),
            label=(
                (step > 0) & (step <= int(window))
            ).astype(int).tolist(),
        )
    return report, entries


def probability_fusion(table_output, logistic_output, args, weight=0.5):
    """E2.3: average the two deploy readouts as calibrated probabilities.

    Rank fusion is a good ranker but its score is a percentile, so a
    threshold calibrated on the deploy stream does not transfer to a stream
    with a different share of dangerous entries (the six-seed failure was
    mostly this).  Both source arms are already probability-like (table:
    node stop-time frequency; logistic: cumulative hazard), so a weighted
    average keeps the ranking and stays calibratable.  tau still comes from
    the deploy stream's out-of-fold predictions.
    """
    report = {}
    for window in WINDOWS:
        if int(window) not in table_output["train_prediction"]:
            continue
        train = (
            float(weight) * table_output["train_prediction"][int(window)]
            + (1.0 - float(weight))
            * logistic_output["train_prediction"][:, int(window) - 1]
        )
        evaluation = (
            float(weight) * table_output["eval_prediction"][int(window)]
            + (1.0 - float(weight))
            * logistic_output["eval_prediction"][:, int(window) - 1]
        )
        metrics = evaluate_table(
            evaluation, table_output["eval_step"],
            table_output["eval_remaining"], window,
        )
        metrics.update(calibration_transfer(
            train, table_output["step"], table_output["remaining"], window,
            evaluation, table_output["eval_step"],
            table_output["eval_remaining"],
            precision_target=args.precision_target,
        ))
        report[str(window)] = metrics
    return report


def reliability_bins(prediction, step, remaining, window, bins=4):
    keep = _valid(step, remaining) & np.isfinite(prediction)
    truth = ((np.asarray(step) > 0) & (np.asarray(step) <= window)).astype(
        float
    )
    values = np.asarray(prediction)[keep]
    labels = truth[keep]
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    report = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high)
        if not mask.any():
            continue
        report.append(dict(
            low=float(low), high=float(high), n=int(mask.sum()),
            mean_prediction=float(values[mask].mean()),
            mean_truth=float(labels[mask].mean()),
        ))
    return report


def uncertainty_report(model, features, step, remaining, window):
    prediction = predict_tree(model, features, window)
    keep = _valid(step, remaining) & np.isfinite(prediction)
    truth = ((np.asarray(step) > 0) & (np.asarray(step) <= window)).astype(
        float
    )
    error = np.abs(prediction - truth)
    if not keep.any():
        return None
    stand = (features - model["mean"]) / model["scale"]
    difference = stand[:, None, :] - model["centers"][None, :, :]
    distance = np.sqrt(
        np.einsum("ijk,ijk->ij", difference, difference)
    )
    fine = distance.argmin(axis=1)
    support = model["fine_total"][fine]
    nearest = distance.min(axis=1)
    cutoff = float(np.median(error[keep]))
    correct = (error <= cutoff).astype(float)
    return dict(
        n=int(keep.sum()),
        mean_error=float(np.mean(error[keep])),
        auc_support=rank_auc(support[keep], correct[keep]),
        auc_distance=rank_auc(nearest[keep], correct[keep]),
        support_low=float(np.mean(support[keep] <= 5.0)),
        distance_median=float(np.median(nearest[keep])),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n-lags", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k-fine", type=int, default=32)
    parser.add_argument("--k-coarse", type=int, default=4)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--precision-target", type=float, default=0.5)
    parser.add_argument("--deploy-frames", type=int, default=5000)
    parser.add_argument("--deploy-offset", type=int, default=70000)
    parser.add_argument("--cache-prefix", default="consume")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    from death_transition_quality import seed_spec
    eval_seeds = seed_spec(args.eval_seeds)
    world, agent, dagger_rows = load_h6(args.seed, args)
    passive_rows, _counts = collect(
        seed=args.seed, steps=args.steps, coverage=args.coverage, lag=LAG,
    )
    deploy_rows = collect_deploy(args.seed, world, agent, args)
    from outcome_level_probe import episode_tables
    dagger_tables = episode_tables(dagger_rows)
    passive_tables = episode_tables(passive_rows)
    deploy_tables = episode_tables(deploy_rows)
    evaluation_tables = eval_tables(args.seed, world, agent, eval_seeds)
    if not dagger_tables or not evaluation_tables:
        raise SystemExit("missing streams")
    dims = [
        dim for dim, info in clock_report(
            dagger_tables, dagger_tables[0]["obs"].shape[1],
        ).items() if not info["excluded"]
    ]
    dagger = _stream(
        dagger_tables, dims, EVENT_LEVELS, HORIZON, args.n_lags, args.stride,
    )
    passive = _stream(
        passive_tables, dims, EVENT_LEVELS, HORIZON, args.n_lags,
        args.stride,
    )
    deploy = _stream(
        deploy_tables, dims, EVENT_LEVELS, HORIZON, args.n_lags,
        args.stride,
    )
    print(
        "span consume probe: seed %d  h6 dagger %d ep / %d deaths  "
        "deploy %d ep / %d deaths  passive %d ep  eval %d ep  dims %s" % (
            args.seed, len(dagger_tables), dagger["n_deaths"],
            len(deploy_tables), deploy["n_deaths"],
            len(passive_tables), len(evaluation_tables), dims,
        )
    )
    print("  eval frames: %s" % "/".join(
        str(len(table["actions"])) for table in evaluation_tables
    ))

    arms = {}
    diagnostics = {}
    outputs = {}
    for name, train_stream in (
        ("deploy", deploy), ("dagger", dagger), ("passive", passive),
    ):
        eval_stream = _stream(
            evaluation_tables, dims, EVENT_LEVELS, HORIZON, args.n_lags, 1,
            predicates=train_stream["predicates"],
        )
        table_output = table_arm(train_stream, eval_stream, args)
        arms["table_" + name] = table_output["report"]
        outputs["table_" + name] = table_output
        diagnostics["table_" + name] = dict(
            reliability=reliability_bins(
                predict_tree(
                    table_output["model"], table_output["eval_features"], 12,
                ),
                entry_labels(eval_stream["terminal"],
                             eval_stream["entries"], HORIZON)[0],
                eval_stream["entries"]["remaining"], 12,
            ),
            uncertainty=uncertainty_report(
                table_output["model"], table_output["eval_features"],
                entry_labels(eval_stream["terminal"],
                             eval_stream["entries"], HORIZON)[0],
                eval_stream["entries"]["remaining"], 12,
            ),
        )
        logistic_output = logistic_arm(train_stream, eval_stream, args)
        arms["logistic_" + name] = logistic_output["report"]
        outputs["logistic_" + name] = logistic_output

    fusion_entries = {}
    if "table_deploy" in outputs and "logistic_deploy" in outputs:
        fusion_report, fusion_entries = fusion_arm(
            outputs["table_deploy"], outputs["logistic_deploy"], args,
        )
        arms["fuse_deploy"] = fusion_report
        arms["prob_deploy"] = probability_fusion(
            outputs["table_deploy"], outputs["logistic_deploy"], args,
        )

    for name in ("table_deploy", "table_dagger", "table_passive",
                 "logistic_deploy", "logistic_dagger", "logistic_passive",
                 "fuse_deploy", "prob_deploy"):
        row = arms[name]["12"]
        print(
            "  %-18s span12 AUC %s  band %s  base %s  tau %s  "
            "precision %s  coverage %s  recall %s" % (
                name,
                _fmt(row.get("auc")), _fmt(row.get("band_auc")),
                _fmt(row.get("base")), _fmt(row.get("tau")),
                _fmt(row.get("precision")), _fmt(row.get("coverage")),
                _fmt(row.get("recall")),
            )
        )
    for name in ("table_deploy", "table_dagger", "table_passive"):
        uncertainty = diagnostics[name]["uncertainty"]
        print(
            "  %-17s uncertainty: support AUC %s  distance AUC %s  "
            "low-support share %s  mean |error| %s" % (
                name,
                _fmt(uncertainty["auc_support"]),
                _fmt(uncertainty["auc_distance"]),
                _fmt(uncertainty["support_low"]),
                _fmt(uncertainty["mean_error"]),
            )
        )
        print("    reliability: " + "  ".join(
            "[%.1f,%.1f) n=%d pred %.2f truth %.2f" % (
                item["low"], item["high"], item["n"],
                item["mean_prediction"], item["mean_truth"],
            )
            for item in diagnostics[name]["reliability"]
        ))

    for name in ("table_deploy", "table_dagger", "table_passive",
                 "fuse_deploy", "prob_deploy"):
        row = arms[name]["16"]
        print(
            "  %-18s span16 AUC %s  band %s  precision %s  "
            "coverage %s  recall %s" % (
                name, _fmt(row.get("auc")), _fmt(row.get("band_auc")),
                _fmt(row.get("precision")), _fmt(row.get("coverage")),
                _fmt(row.get("recall")),
            )
        )

    def best(prefix, key):
        values = [
            arms[name]["12"].get(key) for name in arms
            if name.startswith(prefix)
        ]
        values = [value for value in values if value is not None]
        return max(values) if values else None

    calibrated = [
        name for name in arms
        if (arms[name]["12"].get("precision") or 0.0)
        >= args.precision_target
        and (arms[name]["12"].get("coverage") or 0.0) >= 0.05
    ]
    deploy_gain = dict(
        span_12=(
            (arms["table_deploy"]["12"].get("precision") or 0.0)
            - (arms["table_dagger"]["12"].get("precision") or 0.0)
        ),
        span_16=(
            (arms["table_deploy"]["16"].get("precision") or 0.0)
            - (arms["table_dagger"]["16"].get("precision") or 0.0)
        ),
    )
    checks = dict(
        eval_calibration=bool(calibrated),
        eval_ranking=bool(
            best("", "auc") is not None and best("", "auc") >= 0.75
            and best("", "band_auc") is not None
            and best("", "band_auc") >= 0.70
        ),
        consumption_gain=bool(
            (best("table_dagger", "auc") or 0.0)
            >= (best("table_passive", "auc") or 0.0) + 0.03
            or (best("logistic_dagger", "auc") or 0.0)
            >= (best("logistic_passive", "auc") or 0.0) + 0.03
        ),
        deploy_gain=bool(
            deploy_gain["span_12"] >= 0.05
            or deploy_gain["span_16"] >= 0.05
        ),
        uncertainty=bool(any(
            diagnostics[name]["uncertainty"]["auc_support"] is not None
            and (
                diagnostics[name]["uncertainty"]["auc_support"] >= 0.60
                or (
                    diagnostics[name]["uncertainty"]["auc_distance"]
                    is not None
                    and diagnostics[name]["uncertainty"]["auc_distance"]
                    <= 0.40
                )
            )
            for name in diagnostics
        )),
    )
    fusion = arms.get("fuse_deploy", {}).get("16", {})
    fusion_gate = dict(
        precision=fusion.get("precision"),
        coverage=fusion.get("coverage"),
        recall=fusion.get("recall"),
        target=float(args.precision_target),
        pass_=bool(
            (fusion.get("precision") or 0.0) >= float(args.precision_target)
        ),
    )
    checks["fusion_span16"] = fusion_gate["pass_"]
    probability = arms.get("prob_deploy", {}).get("16", {})
    probability_gate = dict(
        precision=probability.get("precision"),
        coverage=probability.get("coverage"),
        recall=probability.get("recall"),
        target=float(args.precision_target),
        pass_=bool(
            (probability.get("precision") or 0.0)
            >= float(args.precision_target)
        ),
    )
    checks["prob_span16"] = probability_gate["pass_"]
    verdict = (
        "pass" if checks["eval_calibration"] and checks["eval_ranking"]
        else "ranking_only" if checks["eval_ranking"] else "fail"
    )
    print(
        "== E2.1 verdict: %s  (calibration %s, ranking %s, "
        "consumption gain %s, deploy gain %s %.3f/%.3f, "
        "uncertainty %s, fusion span16 precision %s/%s, "
        "prob span16 precision %s) ==" % (
            verdict,
            "pass" if checks["eval_calibration"] else "fail",
            "pass" if checks["eval_ranking"] else "fail",
            "pass" if checks["consumption_gain"] else "fail",
            "pass" if checks["deploy_gain"] else "fail",
            deploy_gain["span_12"], deploy_gain["span_16"],
            "pass" if checks["uncertainty"] else "fail",
            _fmt(fusion.get("precision")), _fmt(fusion_gate["target"]),
            _fmt(probability.get("precision")),
        )
    )
    report = dict(
        seed=args.seed, dims=dims, arms=arms,
        diagnostics=diagnostics, checks=checks,
        deploy_gain=deploy_gain, verdict=verdict,
        fusion_gate=fusion_gate, probability_gate=probability_gate,
        fusion_entries=fusion_entries,
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return report


def _fmt(value):
    return "None" if value is None else "%.3f" % value


if __name__ == "__main__":
    main()
