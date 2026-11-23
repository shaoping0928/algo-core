"""Separate ranking quality from threshold calibration in the span reports.

The E2.2 six-seed failure mixes two questions:

* can the readout rank near-death entries above safe ones (AUC / top-of-list
  precision)?
* does the deploy-stream threshold place the cut in the right spot
  (precision at the calibrated tau)?

The fusion reports now store per-entry predictions, labels, step and
remaining, so both can be read without another collection run.  Fixed-coverage
precision is the ranking-only view: take the top q fraction of the valid eval
entries by score and report the label rate, independent of tau.  If top-decile
precision is high while calibrated precision is low, the readout is a ranker
with a threshold problem; if top-decile precision is low too, the ranking
itself fails on that seed.

Read-only: no model is fitted and no planner path changes.

Usage::

    python span_threshold_probe.py --pattern "consume-e22-*.json"
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from endpoint_correction_probe import rank_auc

HORIZON = 16
WINDOWS = (8, 12, 16)
ARMS = ("table", "logistic", "prediction", "prob")
COVERAGES = (0.05, 0.10, 0.20, 0.30)


def valid_mask(step, remaining, horizon=HORIZON):
    step = np.asarray(step)
    remaining = np.asarray(remaining)
    return (((step > 0) & (step <= int(horizon)))
            | (remaining >= int(horizon)))


def score_arm(values, label, coverage):
    values = np.asarray(values, dtype=float)
    label = np.asarray(label, dtype=float)
    keep = np.isfinite(values)
    values = values[keep]
    label = label[keep]
    if not len(values):
        return dict(n=0, precision=None, recall=None, threshold=None)
    count = max(1, int(round(float(coverage) * len(values))))
    order = np.argsort(-values, kind="mergesort")[:count]
    flagged = np.zeros(len(values), dtype=bool)
    flagged[order] = True
    positives = float(label.sum())
    return dict(
        n=int(count),
        precision=float(label[flagged].mean()) if count else None,
        recall=(
            float(label[flagged].sum() / positives)
            if positives > 0.0 else None
        ),
        threshold=float(values[order[-1]]) if count else None,
    )


def analyze_window(item, window):
    step = np.asarray(item["step"])
    remaining = np.asarray(item["remaining"])
    keep = valid_mask(step, remaining)
    label = ((step > 0) & (step <= int(window))).astype(float)
    report = dict(
        n=int(keep.sum()),
        base=float(label[keep].mean()) if keep.any() else None,
        arms={},
    )
    if not keep.any():
        return report
    for arm in ARMS:
        if arm == "prob":
            values = 0.5 * np.asarray(item["table"], dtype=float) + 0.5 * (
                np.asarray(item["logistic"], dtype=float)
            )
        else:
            values = np.asarray(item[arm], dtype=float)
        values = values[keep]
        truth = label[keep]
        metrics = dict(
            auc=(
                rank_auc(values, truth)
                if len(np.unique(truth)) > 1 else None
            ),
            brier=float(np.mean((values - truth) ** 2)),
        )
        base = float(truth.mean())
        base_brier = float(np.mean((base - truth) ** 2))
        metrics["brier_skill"] = (
            1.0 - metrics["brier"] / base_brier if base_brier > 0 else None
        )
        metrics["top"] = {
            "%.2f" % coverage: score_arm(values, truth, coverage)
            for coverage in COVERAGES
        }
        # Equal-mass calibration error over ten bins.
        edges = np.linspace(0.0, 1.0, 11)
        error = 0.0
        for low, high in zip(edges[:-1], edges[1:]):
            chunk = (values >= low) & (values < high)
            if chunk.any():
                error += chunk.mean() * abs(
                    float(values[chunk].mean())
                    - float(truth[chunk].mean())
                )
        metrics["ece"] = float(error)
        report["arms"][arm] = metrics
    return report


def summarize(rows, window):
    """Cross-seed medians for each arm at each fixed coverage."""
    output = {}
    for arm in ARMS:
        aucs = [row["arms"][arm]["auc"] for row in rows
                if row["arms"][arm]["auc"] is not None]
        values = {}
        for coverage in COVERAGES:
            key = "%.2f" % coverage
            precisions = [row["arms"][arm]["top"][key]["precision"]
                          for row in rows
                          if row["arms"][arm]["top"][key]["precision"]
                          is not None]
            values[key] = dict(
                median=float(np.median(precisions)) if precisions else None,
                minimum=float(np.min(precisions)) if precisions else None,
                pass_=int(sum(1 for value in precisions if value >= 0.5)),
            )
        output[arm] = dict(
            auc_median=float(np.median(aucs)) if aucs else None,
            coverages=values,
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="consume-e22-*.json")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--exclude", default="12k")
    args = parser.parse_args()

    paths = [
        path for path in sorted(glob.glob(args.pattern))
        if args.exclude not in path
    ]
    if not paths:
        raise SystemExit("no reports matched %s" % args.pattern)
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        entries = data.get("fusion_entries", {}).get(str(args.window))
        if not entries:
            print("%s: no per-entry data for span %d" % (path, args.window))
            continue
        result = analyze_window(entries, args.window)
        rows.append(result)
        print("== seed %s (n=%d, base %.3f) ==" % (
            data["seed"], result["n"], result["base"],
        ))
        for arm in ARMS:
            metrics = result["arms"][arm]
            top = "  ".join(
                "top%d%% P %.3f R %.3f" % (
                    int(100 * float(key)),
                    metrics["top"][key]["precision"],
                    metrics["top"][key]["recall"],
                )
                for key in ("0.05", "0.10", "0.20")
            )
            print("  %-11s AUC %s  skill %s  ECE %s  %s" % (
                arm,
                _fmt(metrics["auc"]), _fmt(metrics["brier_skill"]),
                _fmt(metrics["ece"]), top,
            ))
    if not rows:
        raise SystemExit("no rows parsed")
    print("== cross-seed summary (span %d) ==" % args.window)
    summary = summarize(rows, args.window)
    for arm in ARMS:
        item = summary[arm]
        print("  %-11s AUC median %s" % (arm, _fmt(item["auc_median"])))
        for coverage in COVERAGES:
            key = "%.2f" % coverage
            values = item["coverages"][key]
            print("    top%-4s median %s  min %s  >=0.5 %d/%d" % (
                key, _fmt(values["median"]), _fmt(values["minimum"]),
                values["pass_"], len(rows),
            ))


def _fmt(value):
    return "n/a" if value is None else "%.3f" % value


if __name__ == "__main__":
    main()
