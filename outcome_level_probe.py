"""Event/stage outcome audit: which outcome predicates are learnable levels?

Batch 8's first experiment, read-only.  It does **not** touch the frame-level
planner, the 8-step rollout, or any score column.  It asks the question the
plan always intended: on the passive behaviour stream, which outcome
predicates behave like learnable transition levels in the sense of

    level = (outcome predicate O, outcome alphabet A_O)
    span  = P(T | entry, option)   (stopping-time distribution; duration is
                                    an output, not a fixed horizon)

For every candidate predicate the audit fits one distributed-lag
cause-specific hazard on person-period rows

    lambda(t | s, a_lags) = sigmoid(b.phi(s) + sum_l d_l a_{t-l} + g(t))

trained once and queried at any window, then runs the five tests:

1. event support            (distinct events, episodes with events);
2. action specificity       (model contrast vs within-episode action shuffle);
3. non-clock                (full vs time-only hazard AUC);
4. post-event regime change (state mean shift after vs before, vs random);
5. cross-episode calibration(leave-episode-fold AUC / Brier skill).

Verdicts: ``level`` (all five pass), ``boundary`` (everything but action
specificity: a world/boundary covariate, not a decision level), ``reject``.
No task vocabulary enters: predicates are generated from the terminal flag
and generic observation statistics (quantile crossings, sign changes,
extrema, jumps); the only excluded coordinates are near-constant-increment
counters.  Evaluation never looks at the 8-step rollout or frame endpoints.

Usage::

    python outcome_level_probe.py --seed 101 --steps 5000 \
        --report outcome-level-101.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from endpoint_correction_probe import fit_logistic, predict_logistic, rank_auc
from substrate_generic import collect
from substrate_nodes import LAG

WINDOWS = (4, 8, 12, 16)


def time_basis(steps, l_max):
    steps = np.asarray(steps, dtype=float).reshape(-1)
    return np.column_stack([
        np.log1p(steps), np.sqrt(steps), steps / float(l_max),
    ])


def episode_tables(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), []).append(row)
    tables = []
    for episode in sorted(grouped):
        chunk = grouped[episode]
        tables.append(dict(
            episode=episode,
            obs=np.asarray([item["obs"] for item in chunk], dtype=float),
            actions=np.asarray([
                int(np.asarray(item["history"], dtype=float).reshape(-1)[0])
                for item in chunk
            ], dtype=int),
            done=np.asarray([
                bool(item.get("done", False)) for item in chunk
            ], dtype=bool),
        ))
    return tables


def clock_report(tables, obs_dim, tolerance=0.05, jump_ratio=3.0):
    """Per-dimension clock audit: is this coordinate a monotone counter?

    A coordinate is a clock when its increments are uniformly tiny and it
    never jumps: a near-constant drift with no resets.  Sawtooth structural
    coordinates (constant drift *with* occasional large jumps) are not
    clocks, whatever world they come from.  The rule is stated without any
    task vocabulary; the per-dimension numbers are reported so the exclusion
    is auditable.
    """
    stats = {}
    for dim in range(obs_dim):
        values = np.concatenate([table["obs"][:, dim] for table in tables])
        deltas = np.concatenate([
            np.diff(table["obs"][:, dim]) for table in tables
            if len(table["obs"]) > 1
        ])
        if float(np.ptp(values)) < 1e-9 or not len(deltas):
            info = dict(
                std_ratio=None, jump_ratio=None, excluded=True,
                reason="frozen",
            )
        else:
            mean_abs = float(np.mean(np.abs(deltas)))
            std_ratio = (
                float(np.std(deltas)) / mean_abs if mean_abs > 1e-12 else None
            )
            jump = (
                float(np.max(np.abs(deltas))) / mean_abs
                if mean_abs > 1e-12 else None
            )
            excluded = bool(
                std_ratio is not None
                and std_ratio < tolerance
                and jump is not None
                and jump < jump_ratio
            )
            info = dict(
                std_ratio=std_ratio,
                jump_ratio=jump,
                excluded=excluded,
                reason=(
                    "counter" if excluded
                    else "jump" if (std_ratio is not None
                                    and std_ratio < tolerance)
                    else "varying"
                ),
            )
        stats[dim] = info
    return stats


def clock_dimensions(tables, obs_dim, tolerance=0.05, jump_ratio=3.0):
    report = clock_report(tables, obs_dim, tolerance, jump_ratio)
    return sorted(
        dim for dim, info in report.items() if info["excluded"]
    )


def stream_stats(tables, dims):
    """Per-dimension generic statistics used to seed event candidates.

    The jump threshold is the midpoint between the typical and the largest
    increment: tail quantiles are degenerate when a coordinate's increments
    take only two values (e.g. a constant drift plus one discrete impulse),
    and a midpoint rule separates the modes without naming either.
    """
    stats = {}
    for dim in dims:
        values = np.concatenate([table["obs"][:, dim] for table in tables])
        deltas = np.concatenate([
            np.abs(np.diff(table["obs"][:, dim])) for table in tables
            if len(table["obs"]) > 1
        ])
        stats[dim] = dict(
            q10=float(np.quantile(values, 0.10)),
            q90=float(np.quantile(values, 0.90)),
            jump=(
                0.5 * (
                    float(np.quantile(deltas, 0.50)) + float(deltas.max())
                ) if len(deltas) else 0.0
            ),
            spans_zero=bool(values.min() < 0.0 < values.max()),
        )
    return stats


def candidate_predicates(stats, dims):
    predicates = [dict(name="terminal", kind="terminal")]
    for dim in dims:
        info = stats[dim]
        predicates.append(dict(
            name="x%d_low" % dim, kind="cross_low", dim=dim, thr=info["q10"],
        ))
        predicates.append(dict(
            name="x%d_high" % dim, kind="cross_high", dim=dim,
            thr=info["q90"],
        ))
        if info["spans_zero"]:
            predicates.append(dict(name="x%d_sign" % dim, kind="sign", dim=dim))
        predicates.append(dict(
            name="x%d_extr" % dim, kind="extremum", dim=dim,
        ))
        if info["jump"] > 1e-9:
            predicates.append(dict(
                name="x%d_jump" % dim, kind="jump", dim=dim,
                thr=info["jump"],
            ))
    return predicates


def event_mask(table, predicate):
    if predicate["kind"] == "terminal":
        return table["done"].copy()
    values = table["obs"][:, predicate["dim"]]
    previous = np.concatenate([values[:1], values[:-1]])
    if predicate["kind"] == "cross_low":
        return (values < predicate["thr"]) & (previous >= predicate["thr"])
    if predicate["kind"] == "cross_high":
        return (values > predicate["thr"]) & (previous <= predicate["thr"])
    if predicate["kind"] == "sign":
        return (values > 0.0) != (previous > 0.0)
    if predicate["kind"] == "extremum":
        mask = np.zeros(len(values), dtype=bool)
        if len(values) >= 3:
            middle = values[1:-1]
            mask[1:-1] = (
                (middle > values[:-2]) & (middle > values[2:])
            ) | ((middle < values[:-2]) & (middle < values[2:]))
        return mask
    if predicate["kind"] == "jump":
        mask = np.zeros(len(values), dtype=bool)
        if len(values) > 1:
            mask[1:] = np.abs(np.diff(values)) > predicate["thr"]
        return mask
    raise ValueError("unknown predicate kind %r" % predicate["kind"])


def _jaccard(left, right):
    intersection = int(np.sum(left & right))
    union = int(np.sum(left | right))
    return intersection / union if union else 0.0


def dedupe_predicates(predicates, masks, jaccard, cap):
    flat = [
        np.concatenate([item.astype(bool) for item in mask])
        for mask in masks
    ]
    order = sorted(
        range(len(predicates)),
        key=lambda index: (
            0 if predicates[index]["kind"] == "terminal" else
            1 if predicates[index]["kind"] == "jump" else 2,
            -int(flat[index].sum()),
        ),
    )
    kept = []
    for index in order:
        if len(kept) >= int(cap):
            break
        if any(
            _jaccard(flat[index], flat[other]) >= float(jaccard)
            for other in kept
        ):
            continue
        kept.append(index)
    return (
        [predicates[index] for index in kept],
        [masks[index] for index in kept],
    )


def build_entries(tables, dims, n_lags, stride):
    raw = np.concatenate([table["obs"][:, dims] for table in tables])
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    phi_rows = []
    lag_rows = []
    episode = []
    row_index = []
    table_index = []
    remaining = []
    for index, table in enumerate(tables):
        total = len(table["actions"])
        for start in range(int(stride), total, int(stride)):
            phi_rows.append((table["obs"][start, dims] - mean) / scale)
            lag_rows.append([
                float(table["actions"][start - lag])
                if start - lag >= 0 else 0.0
                for lag in range(int(n_lags))
            ])
            episode.append(int(table["episode"]))
            row_index.append(start)
            table_index.append(index)
            remaining.append(total - 1 - start)
    return dict(
        phi=np.asarray(phi_rows, dtype=float),
        lags=np.asarray(lag_rows, dtype=float),
        episode=np.asarray(episode, dtype=int),
        row=np.asarray(row_index, dtype=int),
        table=np.asarray(table_index, dtype=int),
        remaining=np.asarray(remaining, dtype=int),
        mean=mean, scale=scale, dims=list(dims),
    )


def entry_labels(mask, entries, l_max):
    """Next occurrence of the predicate after each entry, within its episode."""
    times = [np.flatnonzero(item) for item in mask]
    count = len(entries["row"])
    event_step = np.zeros(count, dtype=int)
    duration = np.zeros(count, dtype=int)
    full = np.full(count, -1, dtype=int)
    for index in range(count):
        table = int(entries["table"][index])
        row = int(entries["row"][index])
        events = times[table]
        position = int(np.searchsorted(events, row, side="right"))
        gap = int(events[position] - row) if position < len(events) else None
        if gap is not None and gap <= int(l_max):
            event_step[index] = gap
            duration[index] = gap
            full[index] = gap
        else:
            duration[index] = min(int(l_max), int(entries["remaining"][index]))
    return event_step, duration, full


def person_period(phi, lags, event_step, duration, l_max):
    keep = duration >= 1
    counts = duration[keep]
    phi_rows = np.repeat(phi[keep], counts, axis=0)
    lag_rows = np.repeat(lags[keep], counts, axis=0)
    steps = np.concatenate([
        np.arange(1, count + 1, dtype=float) for count in counts
    ])
    labels = (
        np.repeat(event_step[keep], counts).astype(float) == steps
    ).astype(float)
    features = np.concatenate(
        [phi_rows, lag_rows, time_basis(steps, l_max)], axis=1,
    )
    return features, labels


def fit_person_hazard(phi, lags, event_step, duration, l_max,
                      l2=0.01, iterations=30):
    features, labels = person_period(
        phi, lags, event_step, duration, l_max,
    )
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    coefficients = fit_logistic(
        (features - mean) / scale, labels, l2=l2, iterations=iterations,
    )
    return coefficients, mean, scale


def survival_curve(coefficients, phi, lags, l_max, override=None):
    if override is None:
        override = lags
    base = np.concatenate([phi, override], axis=1)
    count = len(phi)
    survival = np.ones(count, dtype=float)
    curve = np.zeros((count, int(l_max)), dtype=float)
    for step in range(1, int(l_max) + 1):
        design = np.concatenate([
            base, np.tile(time_basis([step], l_max), (count, 1)),
        ], axis=1)
        hazard = predict_logistic(coefficients, design)
        survival = survival * (1.0 - hazard)
        curve[:, step - 1] = 1.0 - survival
    return curve


def episode_bootstrap(values, groups, draws=200, seed=0):
    if int(draws) <= 0:
        return None
    rng = np.random.RandomState(int(seed))
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index = {
        episode: np.flatnonzero(groups == episode) for episode in unique
    }
    means = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        sample = rng.choice(unique, size=len(unique), replace=True)
        picks = np.concatenate([index[episode] for episode in sample])
        means[draw] = float(np.mean(np.asarray(values)[picks]))
    return means


def contrast_report(coefficients, phi, lags, episode, l_max, windows,
                    draws=200, seed=0):
    high = survival_curve(
        coefficients, phi, lags, l_max, override=np.ones_like(lags),
    )
    low = survival_curve(
        coefficients, phi, lags, l_max, override=np.zeros_like(lags),
    )
    delta = high - low
    report = {}
    for window in windows:
        if int(window) > int(l_max):
            continue
        values = delta[:, int(window) - 1]
        draws_values = episode_bootstrap(values, episode, draws, seed)
        report[int(window)] = dict(
            mean=float(np.mean(values)),
            ci_low=(
                float(np.quantile(draws_values, 0.025))
                if draws_values is not None else None
            ),
            ci_high=(
                float(np.quantile(draws_values, 0.975))
                if draws_values is not None else None
            ),
        )
    return report, delta


def fold_splits(episode, folds):
    unique = np.unique(episode)
    assignment = {
        value: position % int(folds)
        for position, value in enumerate(unique)
    }
    return np.asarray([assignment[value] for value in episode], dtype=int)


def fold_evaluation(phi, lags, event_step, duration, episode, l_max,
                    folds, l2=0.01, iterations=30):
    fold = fold_splits(episode, folds)
    count = len(phi)
    out_full = np.full((count, int(l_max)), np.nan)
    out_time = np.full((count, int(l_max)), np.nan)
    for value in np.unique(fold):
        train = fold != value
        test = fold == value
        if not train.any() or not test.any():
            continue
        features, labels = person_period(
            phi[train], lags[train], event_step[train], duration[train],
            l_max,
        )
        mean = features.mean(axis=0)
        scale = np.where(features.std(axis=0) > 1e-9,
                         features.std(axis=0), 1.0)
        coefficients = fit_logistic(
            (features - mean) / scale, labels, l2=l2, iterations=iterations,
        )
        time_features = features[:, -3:]
        time_mean = time_features.mean(axis=0)
        time_scale = np.where(
            time_features.std(axis=0) > 1e-9, time_features.std(axis=0), 1.0,
        )
        time_coefficients = fit_logistic(
            (time_features - time_mean) / time_scale, labels,
            l2=l2, iterations=iterations,
        )
        base = np.concatenate([phi[test], lags[test]], axis=1)
        rows = base.shape[0]
        survival = np.ones(rows, dtype=float)
        for step in range(1, int(l_max) + 1):
            design = np.concatenate([
                base, np.tile(time_basis([step], l_max), (rows, 1)),
            ], axis=1)
            hazard = predict_logistic(
                coefficients, (design - mean) / scale,
            )
            survival = survival * (1.0 - hazard)
            out_full[test, step - 1] = 1.0 - survival
        survival = np.ones(rows, dtype=float)
        for step in range(1, int(l_max) + 1):
            design = np.tile(time_basis([step], l_max), (rows, 1))
            hazard = predict_logistic(
                time_coefficients, (design - time_mean) / time_scale,
            )
            survival = survival * (1.0 - hazard)
            out_time[test, step - 1] = 1.0 - survival
    return out_full, out_time


def window_metrics(curve, event_step, remaining, window):
    column = curve[:, int(window) - 1]
    mask = np.isfinite(column) & (
        (np.asarray(event_step) > 0)
        | (np.asarray(remaining) >= int(window))
    )
    if not mask.any():
        return None
    prediction = column[mask]
    step = np.asarray(event_step)[mask]
    truth = ((step > 0) & (step <= int(window))).astype(float)
    if len(np.unique(truth)) < 2:
        return dict(
            n=int(mask.sum()), base=float(truth.mean()), auc=None,
            brier=float(np.mean((prediction - truth) ** 2)),
            brier_skill=None,
        )
    base = float(truth.mean())
    brier = float(np.mean((prediction - truth) ** 2))
    base_brier = float(np.mean((base - truth) ** 2))
    return dict(
        n=int(mask.sum()),
        base=base,
        auc=rank_auc(prediction, truth),
        brier=brier,
        brier_skill=1.0 - brier / base_brier if base_brier > 0.0 else None,
    )


def shuffle_control(phi, lags, episode, event_step, duration, l_max, windows,
                    draws=8, seed=0, l2=0.01, iterations=30):
    rng = np.random.RandomState(int(seed))
    samples = {int(window): [] for window in windows if window <= l_max}
    for _ in range(int(draws)):
        permuted = lags.copy()
        for value in np.unique(episode):
            index = np.flatnonzero(episode == value)
            permuted[index] = permuted[rng.permutation(index)]
        coefficients, _mean, _scale = fit_person_hazard(
            phi, permuted, event_step, duration, l_max, l2, iterations,
        )
        report, _delta = contrast_report(
            coefficients, phi, permuted, episode, l_max, windows,
            draws=0, seed=seed,
        )
        for window in samples:
            if int(window) in report:
                samples[window].append(abs(report[int(window)]["mean"]))
    control = {}
    for window, values in samples.items():
        control[int(window)] = dict(
            mean=float(np.mean(values)) if values else None,
            q95=float(np.quantile(values, 0.95)) if values else None,
        )
    return control


def regime_change(mask, tables, dims, mean, scale, window=8, draws=200,
                  seed=0):
    events = []
    for index, table in enumerate(tables):
        for row in np.flatnonzero(mask[index]):
            if row - window >= 0 and row + window < len(table["obs"]):
                events.append((index, int(row)))
    if len(events) < 5:
        return None
    standardized = [
        (table["obs"][:, dims] - mean) / scale for table in tables
    ]

    def distance(pairs):
        values = []
        for table, row in pairs:
            after = standardized[table][row + 1:row + 1 + window].mean(axis=0)
            before = standardized[table][row - window:row].mean(axis=0)
            values.append(float(np.linalg.norm(after - before)))
        return float(np.mean(values))

    real = distance(events)
    pool = []
    for index, table in enumerate(tables):
        total = len(table["obs"])
        for row in range(window, total - window):
            pool.append((index, row))
    if not pool:
        return None
    rng = np.random.RandomState(int(seed))
    control = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        picks = [
            pool[i] for i in rng.randint(0, len(pool), size=len(events))
        ]
        control[draw] = distance(picks)
    return dict(
        events=int(len(events)),
        real=float(real),
        control_mean=float(control.mean()),
        control_q95=float(np.quantile(control, 0.95)),
    )


def duration_statistics(event_step, full):
    observed = full[full > 0]
    return dict(
        censored_mass=float(np.mean(np.asarray(event_step) == 0)),
        p50=float(np.quantile(observed, 0.50)) if len(observed) else None,
        p90=float(np.quantile(observed, 0.90)) if len(observed) else None,
    )


def audit_predicate(predicate, mask, tables, entries, event_step, duration,
                    full, l_max, windows, args):
    phi = entries["phi"]
    lags = entries["lags"]
    episode = entries["episode"]
    remaining = entries["remaining"]
    distinct_events = int(sum(int(item.sum()) for item in mask))
    episodes = int(sum(1 for item in mask if item.any()))
    coefficients, _mean, _scale = fit_person_hazard(
        phi, lags, event_step, duration, l_max, args.l2, args.iterations,
    )
    lag_profile = [
        float(value) for value in coefficients[
            phi.shape[1]:phi.shape[1] + lags.shape[1]
        ]
    ]
    contrast, _delta = contrast_report(
        coefficients, phi, lags, episode, l_max, windows,
        draws=args.bootstrap, seed=args.seed,
    )
    control = shuffle_control(
        phi, lags, episode, event_step, duration, l_max, windows,
        draws=args.shuffle_draws, seed=args.seed + 1,
        l2=args.l2, iterations=args.iterations,
    )
    out_full, out_time = fold_evaluation(
        phi, lags, event_step, duration, episode, l_max,
        folds=args.folds, l2=args.l2, iterations=args.iterations,
    )
    metrics = {}
    for window in windows:
        if int(window) > int(l_max):
            continue
        metrics[int(window)] = window_metrics(
            out_full, event_step, remaining, window,
        )
    time_metrics = window_metrics(out_time, event_step, remaining, 8)
    if predicate["kind"] == "terminal":
        regime = None
        regime_required = False
    else:
        regime = regime_change(
            mask, tables, entries["dims"], entries["mean"], entries["scale"],
            window=8, draws=args.bootstrap, seed=args.seed + 2,
        )
        regime_required = True
    support_ok = (
        distinct_events >= args.min_events and episodes >= args.min_episodes
    )
    action_ok = bool(
        contrast.get(8) and control.get(8)
        and control[8]["q95"] is not None
        and abs(contrast[8]["mean"]) > control[8]["q95"]
    )
    nonclock_ok = bool(
        metrics.get(8) and metrics[8]["auc"] is not None
        and time_metrics is not None and time_metrics["auc"] is not None
        and (metrics[8]["auc"] - time_metrics["auc"]) >= 0.05
    )
    regime_ok = bool(regime is not None and regime["real"] > regime[
        "control_q95"
    ]) if regime_required else True
    calibration = metrics.get(8)
    calibration_ok = bool(
        calibration is not None and calibration["brier_skill"] is not None
        and calibration["brier_skill"] > 0.0
        and calibration["auc"] is not None and calibration["auc"] >= 0.6
    )
    if support_ok and action_ok and nonclock_ok and regime_ok \
            and calibration_ok:
        verdict = "level"
    elif support_ok and nonclock_ok and regime_ok and calibration_ok:
        verdict = "boundary"
    else:
        verdict = "reject"
    return dict(
        name=predicate["name"], kind=predicate["kind"],
        events=distinct_events, episodes=episodes,
        duration=duration_statistics(event_step, full),
        lag_profile=lag_profile,
        contrast=contrast, shuffle=control,
        metrics=metrics, time_metrics_8=time_metrics, regime=regime,
        flags=dict(
            support=support_ok, action=action_ok, nonclock=nonclock_ok,
            regime=regime_ok, calibration=calibration_ok,
        ),
        verdict=verdict,
    )


def event_transitions(names, masks, tables):
    transitions = {}
    for index, table in enumerate(tables):
        events = []
        for position in range(len(names)):
            for row in np.flatnonzero(masks[position][index]):
                events.append((int(row), position))
        events.sort()
        for (first, source), (second, target) in zip(events[:-1], events[1:]):
            key = (names[source], names[target])
            entry = transitions.setdefault(key, [0, 0.0])
            entry[0] += 1
            entry[1] += float(second - first)
    aggregated = {}
    for (source, target), (count, total) in transitions.items():
        aggregated.setdefault(source, {})[target] = dict(
            count=count,
            mean_duration=total / count if count else None,
        )
    return aggregated


def stage_structure(boundary_name, masks, names, tables, entries,
                    terminal_step, l_max):
    if boundary_name is None:
        return None
    position = names.index(boundary_name)
    mask = masks[position]
    boundaries = [np.flatnonzero(item) for item in mask]
    gaps = np.concatenate([
        np.diff(item) for item in boundaries if len(item) > 1
    ]) if any(len(item) > 1 for item in boundaries) else np.asarray([])
    stage_index = np.zeros(len(entries["row"]), dtype=int)
    for index in range(len(entries["row"])):
        table = int(entries["table"][index])
        row = int(entries["row"][index])
        stage_index[index] = int(np.searchsorted(boundaries[table], row,
                                                side="right"))
    stages = {}
    known = (np.asarray(terminal_step) > 0) | (
        np.asarray(entries["remaining"]) >= int(l_max)
    )
    terminal_hit = ((np.asarray(terminal_step) > 0)
                    & (np.asarray(terminal_step) <= int(l_max)))
    for stage in range(6):
        pick = (stage_index == stage) & known
        if not pick.any():
            continue
        stages[stage] = dict(
            entries=int(pick.sum()),
            terminal_rate_8=float(terminal_hit[pick].mean()),
        )
    return dict(
        boundary=boundary_name,
        inter_event_p50=(
            float(np.quantile(gaps, 0.50)) if len(gaps) else None
        ),
        inter_event_p90=(
            float(np.quantile(gaps, 0.90)) if len(gaps) else None
        ),
        stages=stages,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--l-max", type=int, default=16)
    parser.add_argument("--n-lags", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--shuffle-draws", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--min-episodes", type=int, default=5)
    parser.add_argument("--dedupe-jaccard", type=float, default=0.8)
    parser.add_argument("--max-predicates", type=int, default=16)
    parser.add_argument("--keep-dims", default="",
                        help="comma-separated dims to force-include even if "
                             "the clock heuristic excludes them")
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    rows, _counts = collect(
        seed=args.seed, steps=args.steps, coverage=args.coverage, lag=LAG,
    )
    tables = episode_tables(rows)
    obs_dim = tables[0]["obs"].shape[1]
    clock_stats = clock_report(tables, obs_dim)
    clock = sorted(
        dim for dim, info in clock_stats.items() if info["excluded"]
    )
    if args.keep_dims:
        forced = {
            int(value) for value in str(args.keep_dims).split(",")
            if value.strip()
        }
        clock = [dim for dim in clock if dim not in forced]
    dims = [dim for dim in range(obs_dim) if dim not in clock]
    stats = stream_stats(tables, dims)
    predicates = candidate_predicates(stats, dims)
    masks = [
        [event_mask(table, predicate) for table in tables]
        for predicate in predicates
    ]
    predicates, masks = dedupe_predicates(
        predicates, masks, args.dedupe_jaccard, args.max_predicates,
    )
    entries = build_entries(tables, dims, args.n_lags, args.stride)
    total_rows = int(sum(len(table["actions"]) for table in tables))
    print(
        "outcome level audit: seed %d, %d rows, %d episodes, "
        "%d dims (clock %s), %d predicates, %d entries" % (
            args.seed, total_rows, len(tables), len(dims), clock,
            len(predicates), len(entries["row"]),
        )
    )
    reports = []
    for predicate, mask in zip(predicates, masks):
        event_step, duration, full = entry_labels(mask, entries, args.l_max)
        report = audit_predicate(
            predicate, mask, tables, entries, event_step, duration, full,
            args.l_max, WINDOWS, args,
        )
        reports.append(report)
        metrics8 = report["metrics"].get(8) or {}
        print(
            "  %-12s %-10s ev %4d ep %3d  d8 %+.4f  auc8 %s  "
            "skill8 %s  regime %s  -> %s" % (
                report["name"], report["kind"], report["events"],
                report["episodes"],
                report["contrast"].get(8, {}).get("mean", float("nan")),
                "None" if metrics8.get("auc") is None
                else "%.3f" % metrics8["auc"],
                "None" if metrics8.get("brier_skill") is None
                else "%+.3f" % metrics8["brier_skill"],
                "pass" if report["flags"]["regime"] else "n/a"
                if report["regime"] is None else "fail",
                report["verdict"],
            )
        )
    names = [report["name"] for report in reports]
    transitions = event_transitions(names, masks, tables)
    jump_names = [
        name for name, predicate in zip(names, predicates)
        if predicate["kind"] == "jump"
    ]
    boundary_name = None
    if jump_names:
        # The boundary candidate is the jump predicate with the sparsest
        # events (largest mean inter-event gap): the world's own cycle.
        best = None
        for name in jump_names:
            position = names.index(name)
            times = [
                np.flatnonzero(item) for item in masks[position]
            ]
            gaps = np.concatenate([
                np.diff(item) for item in times if len(item) > 1
            ]) if any(len(item) > 1 for item in times) else np.asarray([])
            mean_gap = float(gaps.mean()) if len(gaps) else 0.0
            if best is None or mean_gap > best[0]:
                best = (mean_gap, name)
        boundary_name = best[1] if best else None
    terminal_step, _terminal_duration, _full = entry_labels(
        masks[names.index("terminal")], entries, args.l_max,
    )
    stage = stage_structure(
        boundary_name, masks, names, tables, entries, terminal_step,
        args.l_max,
    )
    levels = [report for report in reports if report["verdict"] == "level"]
    boundaries = [
        report for report in reports if report["verdict"] == "boundary"
    ]
    print(
        "== verdict: %d level candidates, %d boundary candidates, "
        "%d rejected ==" % (
            len(levels), len(boundaries),
            len(reports) - len(levels) - len(boundaries),
        )
    )
    summary = dict(
        seed=args.seed, rows=total_rows, episodes=len(tables),
        clock_dims=clock, clock_stats=clock_stats, dims=dims,
        predicates=names,
        reports=reports, transitions=transitions, stage=stage,
        verdict=dict(
            levels=[report["name"] for report in levels],
            boundaries=[report["name"] for report in boundaries],
        ),
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return summary


if __name__ == "__main__":
    main()
