"""E1: build the coarse span transition as a two-level state tree.

The P0-F autopsy showed the 6-step executor's escape window closes at
``remaining`` 7-12, and the span audit showed a 12-16 frame hazard over
state plus event history predicts death there (AUC 0.85-0.91 on two of
three seeds) - but as an *external* logistic fit, not as part of the
transition system.  This probe builds the same object the way the learner
would own it:

- a two-level state tree: fine nodes (k-means over standardized state plus
  event-history features) and coarse parent nodes (clustered fine centers,
  the "iconic" level);
- every node stores a stopping-time histogram (death at step 1..H or
  survive H) with support counts;
- prediction shrinks the fine node toward its coarse parent, and the
  parent toward the global prior - every observed transition updates both
  its fine node and its ancestor (the sibling pool), which is the credit
  structure the tree is for.

It is validated read-only against the span audit's external fit:

1. table trained on the passive stream, evaluated on the planner DAgger
   stream (the same transfer arm as the logistic reference);
2. table trained on the planner stream with episode folds (the
   decision-time distribution itself), reporting calibration transfer;
3. ablations: state-only vs state+event-history, fine-only vs fine+coarse
   (does the sibling pooling help?), and a node-count sweep.

No planner path is touched and no score column changes; the default stack
stays as it is.

Usage::

    python tree_transition_probe.py --seed 100 --report tree-100.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from endpoint_correction_probe import rank_auc
from event_span_probe import (
    _load_planner, _stream, transfer_block,
)
from event_transition_probe import EVENT_LEVELS, history_features
from event_warning import calibrate_threshold
from outcome_level_probe import (
    clock_report, entry_labels, episode_tables, fold_splits,
)
from substrate_generic import collect
from substrate_nodes import LAG


HORIZON = 16


def kmeans(features, k, iterations=20, seed=0):
    """Small deterministic k-means++ used for the tree levels."""
    features = np.asarray(features, dtype=float)
    count = len(features)
    k = max(1, min(int(k), count))
    rng = np.random.RandomState(int(seed))
    centers = [features[rng.randint(count)]]
    for _ in range(1, k):
        difference = (
            features[:, None, :] - np.asarray(centers)[None, :, :]
        )
        distance = np.einsum("ijk,ijk->ij", difference, difference).min(1)
        total = float(distance.sum())
        probability = (
            distance / total if total > 0.0
            else np.full(count, 1.0 / count)
        )
        centers.append(features[rng.choice(count, p=probability)])
    centers = np.asarray(centers)
    for _ in range(int(iterations)):
        assignment = _assign(features, centers)
        updated = centers.copy()
        for node in range(k):
            members = assignment == node
            if members.any():
                updated[node] = features[members].mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def _assign(features, centers):
    difference = features[:, None, :] - centers[None, :, :]
    return np.einsum("ijk,ijk->ij", difference, difference).argmin(axis=1)


def _standardize(features, mean=None, scale=None):
    features = np.asarray(features, dtype=float)
    if mean is None:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
    return (features - mean) / scale, mean, scale


def fit_tree(features, step, remaining, horizon=HORIZON, k_fine=32,
             k_coarse=4, alpha=4.0, seed=0, use_coarse=True):
    """Fit the two-level tree with per-node stopping-time histograms."""
    features = np.asarray(features, dtype=float)
    stand, mean, scale = _standardize(features)
    centers = kmeans(stand, k_fine, seed=seed)
    assignment = _assign(stand, centers)
    death = (step > 0) & (step <= int(horizon))
    survive = (step == 0) & (remaining >= int(horizon))
    fine_hist = np.zeros((len(centers), int(horizon) + 1), dtype=float)
    fine_total = np.zeros(len(centers), dtype=float)
    np.add.at(fine_hist, (assignment[death], step[death]), 1.0)
    np.add.at(fine_total, assignment[death], 1.0)
    np.add.at(fine_total, assignment[survive], 1.0)
    parent = None
    coarse_hist = coarse_total = None
    if use_coarse:
        coarse_centers = kmeans(
            centers, min(int(k_coarse), len(centers)), seed=seed,
        )
        parent = _assign(centers, coarse_centers)
        coarse_hist = np.zeros(
            (len(coarse_centers), int(horizon) + 1), dtype=float,
        )
        coarse_total = np.zeros(len(coarse_centers), dtype=float)
        for node in range(len(centers)):
            coarse_hist[parent[node]] += fine_hist[node]
            coarse_total[parent[node]] += fine_total[node]
    return dict(
        mean=mean, scale=scale, centers=centers, parent=parent,
        fine_hist=fine_hist, fine_total=fine_total,
        coarse_hist=coarse_hist, coarse_total=coarse_total,
        global_hist=fine_hist.sum(axis=0),
        global_total=float(fine_total.sum()),
        horizon=int(horizon), alpha=float(alpha),
    )


def predict_tree(model, features, window):
    """F(death within `window` | node) with ancestor shrinkage."""
    features = np.asarray(features, dtype=float)
    stand = (features - model["mean"]) / model["scale"]
    fine = _assign(stand, model["centers"])
    upto = int(window)
    global_hits = float(model["global_hist"][1:upto + 1].sum())
    global_p = (
        global_hits / model["global_total"]
        if model["global_total"] > 0.0 else 0.0
    )
    fine_hits = model["fine_hist"][:, 1:upto + 1].sum(axis=1)
    fine_total = model["fine_total"]
    if model["parent"] is None:
        prediction = (
            fine_hits[fine] + model["alpha"] * global_p
        ) / (fine_total[fine] + model["alpha"])
        return prediction
    parent = model["parent"]
    coarse_hits = model["coarse_hist"][:, 1:upto + 1].sum(axis=1)
    coarse_total = model["coarse_total"]
    coarse_p = (
        coarse_hits[parent[fine]] + model["alpha"] * global_p
    ) / (coarse_total[parent[fine]] + model["alpha"])
    return (
        fine_hits[fine] + model["alpha"] * coarse_p
    ) / (fine_total[fine] + model["alpha"])


def _valid(step, remaining, horizon=HORIZON):
    step = np.asarray(step)
    remaining = np.asarray(remaining)
    return (
        ((step > 0) & (step <= int(horizon))) | (remaining >= int(horizon))
    )


def evaluate_table(prediction, step, remaining, window, horizon=HORIZON):
    prediction = np.asarray(prediction, dtype=float)
    step = np.asarray(step)
    remaining = np.asarray(remaining)
    keep = _valid(step, remaining, horizon) & np.isfinite(prediction)
    truth = ((step > 0) & (step <= int(window))).astype(float)
    if not keep.any():
        return dict(n=0)
    values = prediction[keep]
    labels = truth[keep]
    base = float(labels.mean())
    brier = float(np.mean((values - labels) ** 2))
    base_brier = float(np.mean((base - labels) ** 2))
    band = keep & (remaining <= 2 * int(window))
    band_auc = None
    if band.any() and len(np.unique(truth[band])) > 1:
        band_auc = rank_auc(prediction[band], truth[band])
    return dict(
        n=int(keep.sum()),
        base=base,
        auc=(
            rank_auc(values, labels)
            if len(np.unique(labels)) > 1 else None
        ),
        brier=brier,
        brier_skill=1.0 - brier / base_brier if base_brier > 0 else None,
        band_auc=band_auc,
    )


def calibration_transfer(pred_train, step_train, remaining_train, window,
                        pred_test, step_test, remaining_test,
                        precision_target=0.5, horizon=HORIZON):
    """Threshold calibrated on train predictions, read on the test stream."""
    keep = _valid(step_train, remaining_train, horizon)
    truth = ((np.asarray(step_train) > 0)
             & (np.asarray(step_train) <= int(window))).astype(float)
    if not keep.any():
        return dict(tau=None, precision=None, coverage=None, recall=None)
    tau, _covered, _precision = calibrate_threshold(
        np.asarray(pred_train)[keep], truth[keep],
        precision_target=precision_target,
    )
    if tau is None:
        return dict(tau=None, precision=None, coverage=None, recall=None)
    keep_test = _valid(step_test, remaining_test, horizon)
    truth_test = ((np.asarray(step_test) > 0)
                  & (np.asarray(step_test) <= int(window))).astype(float)
    values = np.asarray(pred_test)[keep_test]
    labels = truth_test[keep_test]
    flagged = values >= float(tau)
    return dict(
        tau=float(tau),
        precision=(
            float(labels[flagged].mean()) if flagged.any() else None
        ),
        coverage=float(flagged.mean()) if len(flagged) else None,
        recall=(
            float(np.sum(flagged & (labels > 0.5))
                  / max(1.0, np.sum(labels > 0.5)))
            if len(labels) else None
        ),
    )


def stream_features(stream, use_history=True, phi=None):
    features = [
        np.asarray(stream["entries"]["phi"] if phi is None else phi,
                   dtype=float)
    ]
    if use_history:
        features.append(history_features(
            stream["masks"], stream["names"], stream["entries"],
        ))
    return np.concatenate(features, axis=1)


def transfer_arm(passive, planner_transfer, k_fine, k_coarse, use_history,
                 args):
    step_pas, _duration, _ = entry_labels(
        passive["terminal"], passive["entries"], HORIZON,
    )
    step_pln, _duration, _ = entry_labels(
        planner_transfer["terminal"], planner_transfer["entries"], HORIZON,
    )
    raw = (
        planner_transfer["entries"]["phi"]
        * planner_transfer["entries"]["scale"]
        + planner_transfer["entries"]["mean"]
    )
    phi_pln = (
        raw - passive["entries"]["mean"]
    ) / passive["entries"]["scale"]
    train = stream_features(passive, use_history)
    test = stream_features(planner_transfer, use_history, phi=phi_pln)
    model = fit_tree(
        train, step_pas, passive["entries"]["remaining"],
        horizon=HORIZON, k_fine=k_fine, k_coarse=k_coarse, seed=args.seed,
    )
    report = {}
    for window in (8, 12, 16):
        train_pred = predict_tree(model, train, window)
        test_pred = predict_tree(model, test, window)
        metrics = evaluate_table(
            test_pred, step_pln, planner_transfer["entries"]["remaining"],
            window,
        )
        metrics.update(calibration_transfer(
            train_pred, step_pas, passive["entries"]["remaining"], window,
            test_pred, step_pln, planner_transfer["entries"]["remaining"],
            precision_target=args.precision_target,
        ))
        report[str(window)] = metrics
    return report


def fold_arm(planner, k_fine, k_coarse, use_history, args, folds=5,
             use_coarse=True):
    """Episode-folded table on the planner stream itself."""
    step, _duration, _ = entry_labels(
        planner["terminal"], planner["entries"], HORIZON,
    )
    remaining = planner["entries"]["remaining"]
    features = stream_features(planner, use_history)
    episode = planner["entries"]["episode"]
    fold = fold_splits(episode, folds)
    predictions = {8: np.full(len(step), np.nan),
                   12: np.full(len(step), np.nan),
                   16: np.full(len(step), np.nan)}
    train_predictions = {8: np.full(len(step), np.nan),
                         12: np.full(len(step), np.nan),
                         16: np.full(len(step), np.nan)}
    for value in np.unique(fold):
        train = fold != value
        test = fold == value
        if not train.any() or not test.any():
            continue
        model = fit_tree(
            features[train], step[train], remaining[train],
            horizon=HORIZON, k_fine=k_fine, k_coarse=k_coarse,
            seed=args.seed + int(value), use_coarse=use_coarse,
        )
        for window in predictions:
            predictions[window][test] = predict_tree(
                model, features[test], window,
            )
            train_predictions[window][train] = predict_tree(
                model, features[train], window,
            )
    report = {}
    for window in predictions:
        metrics = evaluate_table(
            predictions[window], step, remaining, window,
        )
        valid_train = _valid(step, remaining)
        metrics.update(calibration_transfer(
            train_predictions[window][valid_train],
            step[valid_train], remaining[valid_train], window,
            predictions[window], step, remaining,
            precision_target=args.precision_target,
        ))
        report[str(window)] = metrics
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n-lags", type=int, default=8)
    parser.add_argument("--k-fine", type=int, default=32)
    parser.add_argument("--k-coarse", type=int, default=4)
    parser.add_argument("--k-sweep", default="16,32,64")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--precision-target", type=float, default=0.5)
    parser.add_argument("--cache-prefix", default="tree-planner")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    rows, _counts = collect(
        seed=args.seed, steps=args.steps, coverage=args.coverage, lag=LAG,
    )
    passive_tables = episode_tables(rows)
    planner_tables = episode_tables(_load_planner(args.seed, args))
    obs_dim = passive_tables[0]["obs"].shape[1]
    clock = clock_report(passive_tables, obs_dim)
    dims = [dim for dim, info in clock.items() if not info["excluded"]]
    passive = _stream(
        passive_tables, dims, EVENT_LEVELS, HORIZON, args.n_lags, args.stride,
    )
    planner = _stream(
        planner_tables, dims, EVENT_LEVELS, HORIZON, args.n_lags, args.stride,
    )
    planner_transfer = _stream(
        planner_tables, dims, EVENT_LEVELS, HORIZON, args.n_lags,
        args.stride, predicates=passive["predicates"],
    )
    print(
        "tree transition probe: seed %d  passive %d ep / %d deaths  "
        "planner %d ep / %d deaths  dims %s" % (
            args.seed, len(passive_tables), passive["n_deaths"],
            len(planner_tables), planner["n_deaths"], dims,
        )
    )

    reference = transfer_block(
        passive, planner_transfer, 12, args,
    )["A1_history"]
    reference_auc = (reference["planner"] or {}).get("auc")
    print(
        "  reference logistic (passive->planner, span 12): AUC %s  band %s" % (
            "None" if reference_auc is None else "%.3f" % reference_auc,
            "None" if reference["planner_band_auc"] is None
            else "%.3f" % reference["planner_band_auc"],
        )
    )

    transfer = {}
    for use_history in (True, False):
        label = "state+history" if use_history else "state_only"
        transfer[label] = transfer_arm(
            passive, planner_transfer, args.k_fine, args.k_coarse,
            use_history, args,
        )
        row = transfer[label]
        print(
            "  table transfer %-13s span12 AUC %s  band12 %s  skill %s  "
            "tau %s  precision %s  coverage %s" % (
                label,
                _fmt(row["12"].get("auc")),
                _fmt(row["12"].get("band_auc")),
                _fmt(row["12"].get("brier_skill")),
                _fmt(row["12"].get("tau")),
                _fmt(row["12"].get("precision")),
                _fmt(row["12"].get("coverage")),
            )
        )

    sweep = {}
    for k in [int(value) for value in str(args.k_sweep).split(",")]:
        sweep[str(k)] = transfer_arm(
            passive, planner_transfer, k, args.k_coarse, True, args,
        )
        print(
            "  transfer K=%-3d span12 AUC %s  band12 %s" % (
                k, _fmt(sweep[str(k)]["12"].get("auc")),
                _fmt(sweep[str(k)]["12"].get("band_auc")),
            )
        )

    folded = {}
    for use_history, use_coarse in (
        (True, True), (True, False), (False, True),
    ):
        label = "%s_%s" % (
            "state+history" if use_history else "state_only",
            "tree" if use_coarse else "fine_only",
        )
        folded[label] = fold_arm(
            planner, args.k_fine, args.k_coarse, use_history, args,
            use_coarse=use_coarse,
        )
        row = folded[label]
        print(
            "  table planner %-20s span12 AUC %s  band12 %s  skill %s  "
            "tau %s  precision %s  coverage %s" % (
                label,
                _fmt(row["12"].get("auc")),
                _fmt(row["12"].get("band_auc")),
                _fmt(row["12"].get("brier_skill")),
                _fmt(row["12"].get("tau")),
                _fmt(row["12"].get("precision")),
                _fmt(row["12"].get("coverage")),
            )
        )

    primary = transfer["state+history"]["12"]
    tree = folded["state+history_tree"]["12"]
    fine_only = folded["state+history_fine_only"]["12"]
    checks = dict(
        matches_reference=bool(
            primary.get("auc") is not None and reference_auc is not None
            and primary["auc"] >= reference_auc - 0.02
        ),
        tree_not_worse=bool(
            tree.get("auc") is not None and fine_only.get("auc") is not None
            and tree["auc"] >= fine_only["auc"] - 0.01
        ),
        calibration=bool(
            tree.get("precision") is not None
            and tree["precision"] >= args.precision_target
        ),
    )
    verdict = (
        "pass" if checks["matches_reference"] and checks["tree_not_worse"]
        else "fail"
    )
    print(
        "== E1 verdict: %s  (matches logistic %s, tree not worse %s, "
        "calibration %s) ==" % (
            verdict,
            "pass" if checks["matches_reference"] else "fail",
            "pass" if checks["tree_not_worse"] else "fail",
            "pass" if checks["calibration"] else "fail",
        )
    )
    report = dict(
        seed=args.seed, dims=dims, reference=reference,
        reference_auc=reference_auc, transfer=transfer, sweep=sweep,
        folded=folded, checks=checks, verdict=verdict,
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
