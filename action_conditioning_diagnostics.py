"""Diagnose the LOEO anomalies: k=3 blow-up and k=2 negative absolute R².

Read-only: loads the cached streams and re-uses the probe's dataset/ridge
helpers.  Prints per-dimension out-of-fold R², per-fold R², design-matrix
condition numbers and non-finite predictions for k=2 and k=3.

Usage::

    python action_conditioning_diagnostics.py --streams stage_streams.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from action_conditioning_probe import build_dataset, ridge_fit
from stage_support_probe import load_streams


def fold_details(features, target, groups, alpha=1e-3):
    predictions = np.zeros_like(target)
    folds = []
    for group in np.unique(groups):
        train = groups != group
        test = groups == group
        design = np.concatenate(
            [features[train], np.ones((int(train.sum()), 1))], axis=1
        )
        condition = float(np.linalg.cond(design))
        weights = ridge_fit(features[train], target[train], alpha=alpha)
        test_design = np.concatenate(
            [features[test], np.ones((int(test.sum()), 1))], axis=1
        )
        predicted = test_design @ weights
        predictions[test] = predicted
        folds.append(dict(
            group=int(group),
            n_train=int(train.sum()),
            n_test=int(test.sum()),
            condition=condition,
            finite=bool(np.isfinite(predicted).all()),
            max_abs=float(np.abs(predicted).max()),
        ))
    sse = ((target - predictions) ** 2).sum(axis=0)
    sst = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    per_dim = np.where(
        sst > 0.0, 1.0 - sse / np.maximum(sst, 1e-12), 0.0
    )
    fold_r2 = []
    for group in np.unique(groups):
        mask = groups == group
        fold_sse = ((target[mask] - predictions[mask]) ** 2).sum(axis=0)
        fold_sst = ((target[mask] - target[mask].mean(axis=0)) ** 2).sum(axis=0)
        fold_r2.append(float(np.mean(np.where(
            fold_sst > 0.0,
            1.0 - fold_sse / np.maximum(fold_sst, 1e-12), 0.0,
        ))))
    return predictions, per_dim, np.asarray(fold_r2), folds


def robust_details(features, target, groups, alpha=1.0):
    """Per-fold standardisation, zero-variance dims dropped, pooled+median."""
    keep_target = target.std(axis=0) > 1e-12
    target_kept = target[:, keep_target]
    predictions = np.zeros_like(target_kept)
    fold_r2 = []
    for group in np.unique(groups):
        train = groups != group
        test = groups == group
        train_features = features[train]
        test_features = features[test]
        mean = train_features.mean(axis=0)
        scale = train_features.std(axis=0)
        keep = scale > 1e-12
        train_features = (train_features[:, keep] - mean[keep]) / scale[keep]
        test_features = (test_features[:, keep] - mean[keep]) / scale[keep]
        train_target = target_kept[train]
        target_mean = train_target.mean(axis=0)
        target_scale = np.maximum(train_target.std(axis=0), 1e-12)
        weights = ridge_fit(
            train_features,
            (train_target - target_mean) / target_scale,
            alpha=alpha,
        )
        test_design = np.concatenate(
            [test_features, np.ones((int(test.sum()), 1))], axis=1
        )
        predicted = (
            test_design @ weights
        ) * target_scale + target_mean
        predictions[test] = predicted
        fold_sse = ((target_kept[test] - predicted) ** 2).sum(axis=0)
        fold_sst = (
            (target_kept[test] - target_kept[test].mean(axis=0)) ** 2
        ).sum(axis=0)
        fold_r2.append(float(np.mean(np.where(
            fold_sst > 0.0,
            1.0 - fold_sse / np.maximum(fold_sst, 1e-12), 0.0,
        ))))
    sse = ((target_kept - predictions) ** 2).sum(axis=0)
    sst = ((target_kept - target_kept.mean(axis=0)) ** 2).sum(axis=0)
    pooled = float(np.mean(np.where(
        sst > 0.0, 1.0 - sse / np.maximum(sst, 1e-12), 0.0,
    )))
    return pooled, float(np.median(fold_r2)), float(np.min(fold_r2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", required=True)
    parser.add_argument("--phases", default="")
    parser.add_argument("--k", default="2,3")
    args = parser.parse_args()
    streams = load_streams(args.streams)
    wanted = [
        int(value) for value in args.k.split(",") if value.strip()
    ]
    phases = [
        int(value) for value in args.phases.split(",") if value.strip()
    ] or list(range(1, len(streams) + 1))
    for index, rows in enumerate(streams):
        if index + 1 not in phases:
            continue
        print("\n== phase %d: %d frames ==" % (index + 1, len(rows)))
        for k in wanted:
            states, actions, targets, groups, _sequences = build_dataset(
                rows, k
            )
            if len(targets) == 0:
                print("  k=%d: no windows" % k)
                continue
            print(
                "  k=%d: %d windows / %d episodes; target scale per dim %s"
                % (k, len(targets), len(np.unique(groups)),
                   np.round(targets.std(axis=0), 3).tolist())
            )
            for label, features in (
                ("state", states),
                ("state+action",
                 np.concatenate([states, actions], axis=1)),
            ):
                predictions, per_dim, fold_r2, folds = fold_details(
                    features, targets, groups
                )
                bad = [fold for fold in folds if not fold["finite"]]
                worst = max(folds, key=lambda fold: fold["condition"])
                print(
                    "    %-12s per-dim R2 %s"
                    % (label, np.round(per_dim, 3).tolist())
                )
                print(
                    "    %-12s fold R2 min %.3f mean %.3f max %.3f  "
                    "nonfinite %d  worst cond %.1e (ep %d)"
                    % (label, float(fold_r2.min()), float(fold_r2.mean()),
                       float(fold_r2.max()), len(bad),
                       worst["condition"], worst["group"])
                )
                pooled, median, minimum = robust_details(
                    features, targets, groups
                )
                print(
                    "    %-12s robust: pooled R2 %+.3f  median fold %+.3f  "
                    "min fold %+.3f"
                    % (label, pooled, median, minimum)
                )


if __name__ == "__main__":
    main()
