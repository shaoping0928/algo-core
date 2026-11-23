"""P1-B: event-level transition / early-warning audit.

Batch 8's second experiment, read-only.  The outcome audit established six
cross-seed-stable event levels; this probe asks whether they carry
*transition* value beyond being readable state markers:

1. precursor leads: for every terminal event, which levels fired within the
   preceding window, how early, and how much that beats each level's own
   event density (the lift);
2. terminal hazard with and without causal event-history features
   (time since last firing, count in a trailing window, fired-recently flag)
   over leave-episode folds, with a within-episode feature-shuffle placebo
   applied to the history block only;
3. next-event top-1: a modal-successor baseline against the argmax of
   state-only per-level hazards over a shared fold split.

No frame-level planner, no 8-step rollout, no score column is touched.  All
features are causal (only events at or before the entry row).  Failure
readings are explicit: a zero history increment means the current state
already carries the information; zero lead means no warning value.

Usage::

    python event_transition_probe.py --seed 101 --steps 5000 \
        --report event-warning-101.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from endpoint_correction_probe import fit_logistic, predict_logistic, rank_auc
from outcome_level_probe import (
    build_entries, candidate_predicates, episode_tables, entry_labels,
    event_mask, fold_splits, stream_stats, time_basis, window_metrics,
)
from substrate_generic import collect
from substrate_nodes import LAG

EVENT_LEVELS = ("x0_sign", "x1_high", "x0_high", "x5_low", "x3_low")


def build_level_masks(tables, dims, names):
    stats = stream_stats(tables, dims)
    available = {
        item["name"]: item for item in candidate_predicates(stats, dims)
    }
    masks = []
    found = []
    for name in ["terminal"] + list(names):
        if name == "terminal":
            predicate = dict(name="terminal", kind="terminal")
        elif name in available:
            predicate = available[name]
        else:
            continue
        masks.append([event_mask(table, predicate) for table in tables])
        found.append(name)
    return found, masks


def lead_analysis(masks, names, tables, w_pre):
    terminal_rows = [np.flatnonzero(table["done"]) for table in tables]
    report = {}
    for position, name in enumerate(names):
        if name == "terminal":
            continue
        covered = []
        leads = []
        density = []
        for index, table in enumerate(tables):
            events = np.flatnonzero(masks[position][index])
            total = len(table["actions"])
            if total > 1:
                density.append(len(events) / float(total - 1))
            for terminal in terminal_rows[index]:
                earlier = events[events < terminal]
                if len(earlier):
                    gap = int(terminal - earlier[-1])
                    if gap <= int(w_pre):
                        covered.append(1)
                        leads.append(gap)
                    else:
                        covered.append(0)
                else:
                    covered.append(0)
        if not covered:
            report[name] = None
            continue
        coverage = float(np.mean(covered))
        expected = (
            1.0 - float(np.exp(-float(np.mean(density)) * float(w_pre)))
            if density else None
        )
        report[name] = dict(
            n_terminal=int(len(covered)),
            coverage=coverage,
            expected_coverage=expected,
            lift=(
                coverage / expected
                if expected is not None and expected > 0 else None
            ),
            median_lead=float(np.median(leads)) if leads else None,
            p90_lead=float(np.quantile(leads, 0.90)) if leads else None,
            n_leads=int(len(leads)),
        )
    return report


def history_features(masks, names, entries, cap=32, count_w=16, recent_w=4):
    count = len(entries["row"])
    features = np.zeros((count, 3 * len(names)), dtype=float)
    for position in range(len(names)):
        times = [np.flatnonzero(item) for item in masks[position]]
        base = 3 * position
        for index in range(count):
            table = int(entries["table"][index])
            row = int(entries["row"][index])
            events = times[table]
            earlier = events[events <= row]
            if len(earlier):
                since = min(int(row - earlier[-1]), int(cap))
            else:
                since = int(cap)
            features[index, base] = float(since) / float(cap)
            window = events[
                (events > row - int(count_w)) & (events <= row)
            ]
            features[index, base + 1] = float(len(window)) / float(count_w)
            features[index, base + 2] = 1.0 if len(window) else 0.0
    return features


def _person_period_rows(blocks, event_step, duration, l_max):
    """Stack static per-entry blocks with a per-step time basis."""
    keep = duration >= 1
    counts = duration[keep]
    static = np.concatenate(blocks, axis=1)[keep]
    static_rows = np.repeat(static, counts, axis=0)
    steps = np.concatenate([
        np.arange(1, item + 1, dtype=float) for item in counts
    ])
    labels = (
        np.repeat(np.asarray(event_step)[keep], counts).astype(float)
        == steps
    ).astype(float)
    features = np.concatenate(
        [static_rows, time_basis(steps, l_max)], axis=1,
    )
    return features, labels


def _oof_curve(blocks, event_step, duration, episode, l_max, folds,
               l2=0.01, iterations=30, permute_block=None, seed=0):
    """Out-of-fold survival curves for one feature block set.

    ``blocks`` is a list of per-entry matrices; the time basis is appended
    per step.  ``permute_block`` optionally within-episode shuffles one
    training block (placebo).
    """
    fold = fold_splits(episode, folds)
    count = blocks[0].shape[0]
    out = np.full((count, int(l_max)), np.nan)
    rng = np.random.RandomState(int(seed))
    for value in np.unique(fold):
        train = fold != value
        test = fold == value
        if not train.any() or not test.any():
            continue
        train_blocks = [block[train] for block in blocks]
        if permute_block is not None:
            target = list(train_blocks)
            train_episode = episode[train]
            permuted = target[permute_block].copy()
            for item in np.unique(train_episode):
                index = np.flatnonzero(train_episode == item)
                permuted[index] = permuted[rng.permutation(index)]
            target[permute_block] = permuted
            train_blocks = target
        train_duration = duration[train]
        train_features, train_labels = _person_period_rows(
            train_blocks, event_step[train], train_duration, l_max,
        )
        mean = train_features.mean(axis=0)
        scale = np.where(
            train_features.std(axis=0) > 1e-9,
            train_features.std(axis=0), 1.0,
        )
        coefficients = fit_logistic(
            (train_features - mean) / scale, train_labels,
            l2=l2, iterations=iterations,
        )
        test_blocks = [block[test] for block in blocks]
        rows = len(test_blocks[0])
        survival = np.ones(rows, dtype=float)
        for step in range(1, int(l_max) + 1):
            design = np.concatenate(
                test_blocks
                + [np.tile(time_basis([step], l_max), (rows, 1))],
                axis=1,
            )
            hazard = predict_logistic(coefficients, (design - mean) / scale)
            survival = survival * (1.0 - hazard)
            out[test, step - 1] = 1.0 - survival
    return out


def eval_terminal_arms(phi, lags, hist, event_step, duration, episode,
                       remaining, l_max, folds, window, args):
    arms = {}
    state_block = [phi, lags]
    history_block = [phi, lags, hist]
    for name, blocks, permute in (
        ("A0_state", state_block, None),
        ("A1_history", history_block, None),
        ("A1p_shuffled", history_block, 2),
    ):
        curve = _oof_curve(
            blocks, event_step, duration, episode, l_max, folds,
            l2=args.l2, iterations=args.iterations, permute_block=permute,
            seed=args.seed,
        )
        arms[name] = {
            int(window): window_metrics(
                curve, event_step, remaining, window,
            )
        }
    return arms


def eval_level_curve(phi, lags, event_step, episode, l_max, folds,
                     l2=0.01, iterations=30):
    duration = np.where(
        np.asarray(event_step) > 0,
        np.minimum(np.asarray(event_step), int(l_max)),
        int(l_max),
    )
    return _oof_curve(
        [phi, lags], event_step, duration, episode, l_max, folds,
        l2=l2, iterations=iterations,
    )


def next_event_sequences(masks, names, tables):
    """First-firing event sequence per episode: (row, level position)."""
    sequences = []
    for index, table in enumerate(tables):
        events = []
        for position in range(1, len(names)):
            for row in np.flatnonzero(masks[position][index]):
                events.append((int(row), position))
        events.sort()
        merged = []
        for row, position in events:
            if merged and merged[-1][0] == row:
                continue
            merged.append((row, position))
        for terminal in np.flatnonzero(table["done"]):
            merged.append((int(terminal), 0))
        merged.sort()
        deduped = []
        for row, position in merged:
            if deduped and deduped[-1][0] == row:
                continue
            deduped.append((row, position))
        sequences.append(deduped)
    return sequences


def modal_successor(sequences, names):
    counts = {}
    for sequence in sequences:
        for (_, source), (_, target) in zip(sequence[:-1], sequence[1:]):
            key = (source, target)
            counts[key] = counts.get(key, 0) + 1
    per_source = {}
    for (source, target), count in counts.items():
        entry = per_source.setdefault(source, {})
        entry[target] = entry.get(target, 0) + count
    total = 0
    correct = 0
    detail = {}
    for source, successors in sorted(per_source.items()):
        best_target, best_count = max(
            successors.items(), key=lambda item: item[1],
        )
        source_total = sum(successors.values())
        total += source_total
        correct += best_count
        detail[names[source]] = dict(
            n=source_total,
            modal=names[best_target],
            accuracy=best_count / source_total,
        )
    return dict(
        accuracy=correct / total if total else None,
        n=total,
        per_source=detail,
    )


def hazard_next_top1(level_curves, sequences, names, entries, window):
    next_map = {}
    for index in range(len(entries["row"])):
        sequence = sequences[int(entries["table"][index])]
        row = int(entries["row"][index])
        for event_row, target in sequence:
            if event_row > row:
                if event_row - row <= int(window):
                    next_map[index] = target
                break
    total = 0
    correct = 0
    scores = level_curves[:, :, int(window) - 1]
    for index, target in next_map.items():
        total += 1
        correct += int(int(np.argmax(scores[:, index])) == target)
    return dict(
        n=total,
        accuracy=correct / total if total else None,
        chance=1.0 / len(names) if names else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--l-max", type=int, default=8)
    parser.add_argument("--n-lags", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--w-pre", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    rows, _counts = collect(
        seed=args.seed, steps=args.steps, coverage=args.coverage, lag=LAG,
    )
    tables = episode_tables(rows)
    obs_dim = tables[0]["obs"].shape[1]
    dims = [dim for dim in range(obs_dim) if dim != 6]
    names, masks = build_level_masks(tables, dims, EVENT_LEVELS)
    entries = build_entries(tables, dims, args.n_lags, args.stride)
    total_rows = int(sum(len(table["actions"]) for table in tables))
    print(
        "event warning probe: seed %d, %d rows, %d episodes, levels %s" % (
            args.seed, total_rows, len(tables), names,
        )
    )

    leads = lead_analysis(masks, names, tables, args.w_pre)
    for name, row in leads.items():
        if row is None:
            print("  lead %-10s n/a" % name)
            continue
        print(
            "  lead %-10s coverage %.2f (expected %.2f, lift %.2f)  "
            "median %s p90 %s (n=%d)" % (
                name, row["coverage"], row["expected_coverage"] or 0.0,
                row["lift"] or 0.0,
                "None" if row["median_lead"] is None
                else "%.0f" % row["median_lead"],
                "None" if row["p90_lead"] is None
                else "%.0f" % row["p90_lead"],
                row["n_leads"],
            )
        )

    terminal_mask = masks[0]
    event_step, duration, _full = entry_labels(
        terminal_mask, entries, args.l_max,
    )
    hist = history_features(masks, names, entries)
    arms = eval_terminal_arms(
        entries["phi"], entries["lags"], hist, event_step, duration,
        entries["episode"], entries["remaining"], args.l_max, args.folds,
        args.window, args,
    )
    for name, row in arms.items():
        metric = row.get(int(args.window)) or {}
        print(
            "  %-12s auc%d %s  skill %s  n %s" % (
                name, args.window,
                "None" if metric.get("auc") is None
                else "%.3f" % metric["auc"],
                "None" if metric.get("brier_skill") is None
                else "%+.3f" % metric["brier_skill"],
                metric.get("n"),
            )
        )

    sequences = next_event_sequences(masks, names, tables)
    modal = modal_successor(sequences, names)
    print(
        "  next-event modal baseline: accuracy %s  n %d" % (
            "None" if modal["accuracy"] is None
            else "%.3f" % modal["accuracy"],
            modal["n"],
        )
    )
    level_curves = np.stack([
        eval_level_curve(
            entries["phi"], entries["lags"],
            entry_labels(masks[position], entries, args.l_max)[0],
            entries["episode"], args.l_max, folds=args.folds,
            l2=args.l2, iterations=args.iterations,
        )
        for position in range(len(names))
    ], axis=0)
    top1 = hazard_next_top1(
        level_curves, sequences, names, entries, args.window,
    )
    print(
        "  next-event hazard top-1: accuracy %s  chance %s  n %d" % (
            "None" if top1["accuracy"] is None
            else "%.3f" % top1["accuracy"],
            "None" if top1["chance"] is None else "%.3f" % top1["chance"],
            top1["n"],
        )
    )

    strong_levels = [
        name for name, row in leads.items()
        if row is not None and (
            (row["lift"] or 0.0) >= 2.0
            or ((row["coverage"] or 0.0) >= 0.5
                and (row["median_lead"] or 0.0) >= 2.0)
        )
    ]
    auc0 = (arms["A0_state"].get(int(args.window)) or {}).get("auc")
    auc1 = (arms["A1_history"].get(int(args.window)) or {}).get("auc")
    aucp = (arms["A1p_shuffled"].get(int(args.window)) or {}).get("auc")
    skill1 = (
        arms["A1_history"].get(int(args.window)) or {}
    ).get("brier_skill")
    delta_auc = (
        auc1 - auc0 if auc0 is not None and auc1 is not None else None
    )
    delta_placebo = (
        auc1 - aucp if auc1 is not None and aucp is not None else None
    )
    checks = dict(
        leads=len(strong_levels) >= 2,
        history=(
            delta_auc is not None and delta_auc >= 0.02
            and skill1 is not None and skill1 > 0
        ),
        placebo=delta_placebo is not None and delta_placebo >= 0.01,
        top1=(
            top1["accuracy"] is not None and top1["chance"] is not None
            and top1["n"] >= 50
            and top1["accuracy"] >= 1.5 * top1["chance"]
        ),
    )
    verdict = "pass" if all(checks.values()) else "fail"
    print(
        "== P1-B verdict: %s  (strong leads %d: %s; dAUC %s; "
        "dPlacebo %s; top1 %s) ==" % (
            verdict, len(strong_levels), strong_levels,
            "None" if delta_auc is None else "%+.3f" % delta_auc,
            "None" if delta_placebo is None else "%+.3f" % delta_placebo,
            "pass" if checks["top1"] else "fail",
        )
    )
    summary = dict(
        seed=args.seed, rows=total_rows, episodes=len(tables),
        levels=names, leads=leads, arms=arms,
        modal=modal, top1=top1,
        delta_auc=delta_auc, delta_placebo=delta_placebo,
        strong_levels=strong_levels, checks=checks, verdict=verdict,
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return summary


if __name__ == "__main__":
    main()
