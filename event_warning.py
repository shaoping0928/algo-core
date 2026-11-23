"""Deployment-side event-warning model shared by the P1-D probes.

The P1-B audit showed that the six stable event levels carry transition
information: a terminal hazard conditioned on causal event-history features
beats the state-only hazard by +0.03 to +0.11 AUC, and four precursors lead
death by 3-5 frames.  This module turns that audit object into a deployable
warning: it fits the hazard on the passive behaviour stream, calibrates a
trigger threshold at a precision target, and exposes an online state
tracker so the evaluation loop can score each decision causally.

Nothing here touches the frame-level planner or any score column; the model
is a warning signal, not a cost term.  Predicates are the generic audit
candidates (terminal plus per-dimension crossings), stated without task
vocabulary.
"""
from __future__ import annotations

import numpy as np

from endpoint_correction_probe import fit_logistic, predict_logistic
from event_transition_probe import _person_period_rows
from outcome_level_probe import (
    build_entries, candidate_predicates, clock_report, episode_tables,
    entry_labels, event_mask, stream_stats, time_basis,
)
from substrate_generic import collect
from substrate_nodes import LAG

EVENT_LEVELS = ("x0_sign", "x1_high", "x0_high", "x5_low", "x3_low")


def _pair_event(predicate, previous, value):
    kind = predicate["kind"]
    if kind == "cross_low":
        return bool(value < predicate["thr"] and previous >= predicate["thr"])
    if kind == "cross_high":
        return bool(value > predicate["thr"] and previous <= predicate["thr"])
    if kind == "sign":
        return bool((value > 0.0) != (previous > 0.0))
    return False


class OnlineEvents:
    """Causal online replica of the audit's event masks and history features."""

    def __init__(self, names, predicates, cap=32, count_w=16, recent_w=4):
        self.names = list(names)
        self.predicates = list(predicates)
        self.cap = int(cap)
        self.count_w = int(count_w)
        self.recent_w = int(recent_w)
        self.reset()

    def reset(self):
        self.prev = None
        self.current = None
        self.current_row = 0
        self.next_row = 0
        self.last = {name: None for name in self.names}
        self.recent = {name: [] for name in self.names}

    def update(self, observation):
        """Score the row whose observation just arrived, then advance."""
        observation = np.asarray(observation, dtype=float).reshape(-1)
        row = self.next_row
        for position, name in enumerate(self.names):
            predicate = self.predicates[position]
            fired = False
            if predicate["kind"] == "terminal":
                fired = False  # marked by the caller when the world dies
            elif self.prev is not None:
                dim = predicate["dim"]
                fired = _pair_event(
                    predicate, float(self.prev[dim]), float(observation[dim])
                )
            if fired:
                self.last[name] = row
                self.recent[name].append(row)
        for name in self.names:
            self.recent[name] = [
                item for item in self.recent[name]
                if row - item <= self.count_w
            ]
        self.prev = observation.copy()
        self.current = observation.copy()
        self.current_row = row
        self.next_row = row + 1

    def features(self):
        count = len(self.names)
        features = np.zeros(3 * count, dtype=float)
        for position, name in enumerate(self.names):
            base = 3 * position
            last = self.last[name]
            if last is None:
                since = self.cap
            else:
                since = min(self.current_row - last, self.cap)
            features[base] = float(since) / float(self.cap)
            features[base + 1] = (
                float(len(self.recent[name])) / float(self.count_w)
            )
            recent = [
                item for item in self.recent[name]
                if item > self.current_row - self.recent_w
            ]
            features[base + 2] = 1.0 if recent else 0.0
        return features


def _fit_arm(phi, lags, hist, event_step, duration, l_max, l2, iterations,
             use_hist):
    blocks = [phi, lags, hist] if use_hist else [phi, lags]
    features, labels = _person_period_rows(
        blocks, event_step, duration, l_max,
    )
    mean = features.mean(axis=0)
    scale = np.where(features.std(axis=0) > 1e-9,
                     features.std(axis=0), 1.0)
    coefficients = fit_logistic(
        (features - mean) / scale, labels, l2=l2, iterations=iterations,
    )
    return coefficients, mean, scale


def _survival(coefficients, mean, scale, static, l_max):
    count = static.shape[0]
    survival = np.ones(count, dtype=float)
    curve = np.zeros((count, int(l_max)), dtype=float)
    for step in range(1, int(l_max) + 1):
        design = np.concatenate([
            static, np.tile(time_basis([step], l_max), (count, 1)),
        ], axis=1)
        hazard = predict_logistic(coefficients, (design - mean) / scale)
        survival = survival * (1.0 - hazard)
        curve[:, step - 1] = 1.0 - survival
    return curve


def calibrate_threshold(values, labels, precision_target=0.5):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=float)
    order = np.argsort(-values, kind="mergesort")
    sorted_labels = labels[order]
    hits = np.cumsum(sorted_labels)
    counts = np.arange(1, len(labels) + 1, dtype=float)
    precision = hits / counts
    eligible = precision >= float(precision_target)
    if not eligible.any():
        return None, 0, None
    take = int(counts[eligible].max())
    tau = float(values[order][take - 1])
    return tau, take, float(hits[take - 1] / take)


def fit_warning_model(seed=101, steps=5000, coverage=0.1, l_max=8, n_lags=8,
                      stride=2, l2=0.01, iterations=30,
                      precision_target=0.5, source="behavior",
                      cache="", refit=False, horizon=6):
    if source == "behavior":
        rows, _counts = collect(
            seed=seed, steps=steps, coverage=coverage, lag=LAG,
        )
    else:
        from executor_horizon_probe import horizon_config
        from probe_cache import load_or_fit
        config = horizon_config(seed, horizon)
        _world, agent, _cached = load_or_fit(
            config, cache or None, refit=bool(refit),
        )
        rows = list(agent.milestone_training_rows)
        if source == "planner":
            rows = [
                row for row in rows if int(row.get("episode", 0)) >= 50000
            ]
    tables = episode_tables(rows)
    obs_dim = tables[0]["obs"].shape[1]
    clock_stats = clock_report(tables, obs_dim)
    dims = [dim for dim, info in clock_stats.items() if not info["excluded"]]
    stats = stream_stats(tables, dims)
    available = {
        item["name"]: item for item in candidate_predicates(stats, dims)
    }
    names = ["terminal"] + [
        name for name in EVENT_LEVELS if name in available
    ]
    predicates = [dict(name="terminal", kind="terminal")] + [
        available[name] for name in names[1:]
    ]
    masks = [
        [event_mask(table, predicate) for table in tables]
        for predicate in predicates
    ]
    entries = build_entries(tables, dims, n_lags, stride)
    hist = _history_from_masks(masks, names, entries)
    event_step, duration, _full = entry_labels(masks[0], entries, l_max)
    arms = {}
    for name, use_hist in (("A0_state", False), ("A1_history", True)):
        coefficients, mean, scale = _fit_arm(
            entries["phi"], entries["lags"], hist, event_step, duration,
            l_max, l2, iterations, use_hist,
        )
        arms[name] = dict(
            coefficients=coefficients, mean=mean, scale=scale,
        )
    static = np.concatenate([entries["phi"], entries["lags"], hist], axis=1)
    trigger_values = np.zeros(len(entries["row"]), dtype=float)
    for action_value in (0.0, 1.0):
        varied = entries["lags"].copy()
        varied[:, 0] = action_value
        curve = _survival(
            arms["A1_history"]["coefficients"],
            arms["A1_history"]["mean"],
            arms["A1_history"]["scale"],
            np.concatenate([entries["phi"], varied, hist], axis=1),
            l_max,
        )
        trigger_values = np.maximum(trigger_values, curve[:, int(l_max) - 1])
    labels = (np.asarray(event_step) > 0).astype(float)
    tau, covered, precision = calibrate_threshold(
        trigger_values, labels, precision_target=precision_target,
    )
    base_rate = float(labels.mean())
    return dict(
        seed=int(seed), names=names, predicates=predicates, dims=dims,
        obs_mean=entries["mean"], obs_scale=entries["scale"],
        arms=arms, l_max=int(l_max), n_lags=int(n_lags),
        cap=int(32), count_w=int(16), recent_w=int(4),
        tau=tau, tau_coverage=(
            float(covered) / len(labels) if covered and len(labels) else None
        ),
        tau_precision=precision,
        base_rate=base_rate,
        n_entries=int(len(entries["row"])),
        episodes=len(tables),
        source=source,
    )


def _history_from_masks(masks, names, entries):
    """The P1-B history features, restricted to causal online terms."""
    count = len(entries["row"])
    cap = 32
    count_w = 16
    recent_w = 4
    features = np.zeros((count, 3 * len(names)), dtype=float)
    for position, name in enumerate(names):
        times = [np.flatnonzero(item) for item in masks[position]]
        base = 3 * position
        for index in range(count):
            table = int(entries["table"][index])
            row = int(entries["row"][index])
            events = times[table]
            earlier = events[events <= row]
            if len(earlier):
                since = min(int(row - earlier[-1]), cap)
            else:
                since = cap
            features[index, base] = float(since) / float(cap)
            window = events[
                (events > row - int(count_w)) & (events <= row)
            ]
            features[index, base + 1] = (
                float(len(window)) / float(count_w)
            )
            near = [
                item for item in window if item > row - int(recent_w)
            ]
            features[index, base + 2] = 1.0 if near else 0.0
    return features


def phi_for(model, observation):
    observation = np.asarray(observation, dtype=float).reshape(-1)
    return (observation[model["dims"]] - model["obs_mean"]) / model[
        "obs_scale"
    ]


def warning_value(model, phi_row, hist_row, known_lags, action_value,
                  arm="A1_history", window=None):
    """F(terminal within window | state, event history, first action)."""
    arm_model = model["arms"][arm]
    l_max = int(model["l_max"] if window is None else window)
    lags = np.zeros(int(model["n_lags"]), dtype=float)
    take = min(int(model["n_lags"]) - 1, len(known_lags))
    lags[0] = float(action_value)
    lags[1:1 + take] = np.asarray(known_lags[:take], dtype=float)
    static = np.concatenate(
        [phi_row.reshape(1, -1), lags.reshape(1, -1),
         hist_row.reshape(1, -1)], axis=1,
    )
    curve = _survival(
        arm_model["coefficients"], arm_model["mean"], arm_model["scale"],
        static, l_max,
    )
    return float(curve[0, l_max - 1])
