"""Can the event viewport predict death 8-16 frames ahead?

P1-B measured precursor leads as the last instance inside an 8-frame window
(``gap <= w_pre`` by construction), so its "median lead 3-5" cannot say
whether an event fires before the frame-level escape window closes.  The
P0-F autopsy located that escape window: h6 fails at ``remaining`` 7-12
(the h8 escapes all live there), and inside ``remaining <= 6`` every
candidate dies, so the event layer needs lead >= 7-12, not 3-5.

This probe measures, on both streams (passive behaviour, planner DAgger):

1. approach leads: per death, the first and last precursor instance in the
   preceding 16 frames; the share of deaths whose first instance fires at
   least 8 / 12 frames early; and a frame-level rule warning ("any strong
   precursor fired within h frames") with precision, recall and lift over
   the base rate at h = 4 / 8 / 12 / 16, plus where its hits sit in
   remaining time.
2. span scaling: the P1-B hazard (A0 state-only, A1 + event history) fitted
   on the passive stream at l_max = 8 / 12 / 16, evaluated out-of-fold on
   the passive stream and transferred to the planner stream.  AUC, Brier
   skill, the near-death band (remaining <= 2*l_max), and the calibrated
   threshold transfer (precision / coverage / recall when the passive tau
   is applied to the planner stream).
3. entry readability: per-feature AUC between corridor-entry entries
   (remaining 7-12) and far entries (remaining 17-32) on the planner
   stream, for state and event-history features.

Read-only: no planner path, score column or default changes.  The verdict
names which component the evidence points at; the thresholds are in the
report, not hidden in a score.

Usage::

    python event_span_probe.py --seed 100 --report event-span-100.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from endpoint_correction_probe import (
    fit_logistic, predict_logistic, rank_auc,
)
from event_transition_probe import (
    EVENT_LEVELS, _oof_curve, _person_period_rows, history_features,
    lead_analysis,
)
from event_warning import calibrate_threshold
from outcome_level_probe import (
    build_entries, candidate_predicates, clock_report, entry_labels,
    episode_tables, event_mask, stream_stats, time_basis, window_metrics,
)
from substrate_generic import collect
from substrate_nodes import LAG


def approach_lead_report(masks, names, tables, horizon=16):
    """First/last precursor instance before each death, up to `horizon`."""
    deaths = [np.flatnonzero(table["done"]) for table in tables]
    report = {}
    for position, name in enumerate(names):
        if name == "terminal":
            continue
        first_leads, last_leads, covered = [], [], []
        for index, table in enumerate(tables):
            events = np.flatnonzero(masks[position][index])
            for terminal in deaths[index]:
                window = events[
                    (events < terminal) & (events >= terminal - int(horizon))
                ]
                if len(window):
                    first_leads.append(int(terminal - window[0]))
                    last_leads.append(int(terminal - window[-1]))
                    covered.append(1)
                else:
                    covered.append(0)
        total = len(covered)
        report[name] = dict(
            n_deaths=int(total),
            n_with_instance=int(len(first_leads)),
            coverage=float(np.mean(covered)) if total else None,
            first_lead_median=(
                float(np.median(first_leads)) if first_leads else None
            ),
            first_lead_p90=(
                float(np.quantile(first_leads, 0.9)) if first_leads else None
            ),
            last_lead_median=(
                float(np.median(last_leads)) if last_leads else None
            ),
            share_first_ge_8=(
                float(np.sum(np.asarray(first_leads) >= 8))
                / float(total) if total else None
            ),
            share_first_ge_12=(
                float(np.sum(np.asarray(first_leads) >= 12))
                / float(total) if total else None
            ),
        )
    return report


def _since_last(masks, position, table_index, length):
    events = np.flatnonzero(masks[position][table_index])
    rows = np.arange(length)
    if not len(events):
        return np.full(length, np.inf)
    index = np.searchsorted(events, rows, side="right") - 1
    since = np.full(length, np.inf)
    found = index >= 0
    since[found] = rows[found] - events[index[found]]
    return since


def rule_warning(masks, positions, tables, horizon):
    """Any selected level fired within `horizon` frames: precision/recall/lift."""
    tp = fp = fn = tn = 0
    hits = []
    for index, table in enumerate(tables):
        deaths = np.flatnonzero(table["done"])
        if not len(deaths):
            continue
        terminal = int(deaths[0])
        length = int(terminal)
        if length <= 0:
            continue
        since = np.full(length, np.inf)
        for position in positions:
            since = np.minimum(
                since, _since_last(masks, position, index, length)
            )
        fired = since <= int(horizon)
        remaining = length - np.arange(length)
        positive = remaining <= int(horizon)
        tp += int(np.sum(fired & positive))
        fp += int(np.sum(fired & ~positive))
        fn += int(np.sum(~fired & positive))
        tn += int(np.sum(~fired & ~positive))
        hits.extend(remaining[fired & positive].tolist())
    precision = tp / float(tp + fp) if tp + fp else None
    recall = tp / float(tp + fn) if tp + fn else None
    base = (tp + fn) / float(tp + fp + fn + tn) if tp + fp + fn + tn else None
    return dict(
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        precision=precision, recall=recall, base=base,
        lift=(precision / base if precision is not None and base else None),
        hit_remaining_mean=float(np.mean(hits)) if hits else None,
        hit_remaining_ge_8=(
            float(np.mean(np.asarray(hits) >= 8)) if hits else None
        ),
    )


def band_auc(values, truth, remaining, window, factor=2.0):
    values = np.asarray(values, dtype=float)
    truth = np.asarray(truth, dtype=float)
    remaining = np.asarray(remaining, dtype=int)
    keep = np.isfinite(values) & (remaining <= int(factor * window))
    if not keep.any() or len(np.unique(truth[keep])) < 2:
        return None
    return rank_auc(values[keep], truth[keep])


def _survival_blocks(coefficients, mean, scale, blocks, l_max):
    count = blocks[0].shape[0]
    survival = np.ones(count, dtype=float)
    curve = np.zeros((count, int(l_max)), dtype=float)
    for step in range(1, int(l_max) + 1):
        design = np.concatenate(
            blocks + [np.tile(time_basis([step], l_max), (count, 1))],
            axis=1,
        )
        hazard = predict_logistic(
            coefficients, (design - mean) / scale
        )
        survival = survival * (1.0 - hazard)
        curve[:, step - 1] = 1.0 - survival
    return curve


def _fit_blocks(blocks, event_step, duration, l_max, l2, iterations):
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


def _truth_and_values(curve, event_step, remaining, window):
    column = curve[:, int(window) - 1]
    mask = np.isfinite(column) & (
        (np.asarray(event_step) > 0)
        | (np.asarray(remaining) >= int(window))
    )
    values = column[mask]
    truth = (
        (np.asarray(event_step)[mask] > 0)
        & (np.asarray(event_step)[mask] <= int(window))
    ).astype(float)
    remain = np.asarray(remaining)[mask]
    return values, truth, remain


def transfer_block(passive, planner, l_max, args, folds=5):
    """A0/A1 span model fitted on the passive stream, read on both streams."""
    ini = int(l_max)
    step_pas, dur_pas, _ = entry_labels(
        passive["terminal"], passive["entries"], ini,
    )
    step_pln, dur_pln, _ = entry_labels(
        planner["terminal"], planner["entries"], ini,
    )
    hist_pas = history_features(
        passive["masks"], passive["names"], passive["entries"],
    )
    hist_pln = history_features(
        planner["masks"], planner["names"], planner["entries"],
    )
    raw = (
        planner["entries"]["phi"] * planner["entries"]["scale"]
        + planner["entries"]["mean"]
    )
    phi_pln = (
        raw - passive["entries"]["mean"]
    ) / passive["entries"]["scale"]
    arms = {
        "A0_state": ([passive["entries"]["phi"], passive["entries"]["lags"]],
                     [phi_pln, planner["entries"]["lags"]]),
        "A1_history": ([passive["entries"]["phi"], passive["entries"]["lags"],
                        hist_pas],
                       [phi_pln, planner["entries"]["lags"], hist_pln]),
    }
    result = {}
    for name, (blocks_pas, blocks_pln) in arms.items():
        curve = _oof_curve(
            blocks_pas, step_pas, dur_pas, passive["entries"]["episode"],
            ini, folds, l2=args.l2, iterations=args.iterations,
        )
        metrics = window_metrics(
            curve, step_pas, passive["entries"]["remaining"], ini,
        )
        values, truth, remain = _truth_and_values(
            curve, step_pas, passive["entries"]["remaining"], ini,
        )
        tau, _covered, _precision = calibrate_threshold(
            values, truth, precision_target=args.precision_target,
        )
        coefficients, mean, scale = _fit_blocks(
            blocks_pas, step_pas, dur_pas, ini, args.l2, args.iterations,
        )
        transferred = _survival_blocks(
            coefficients, mean, scale, blocks_pln, ini,
        )
        plan_metrics = window_metrics(
            transferred, step_pln, planner["entries"]["remaining"], ini,
        )
        plan_values, plan_truth, plan_remain = _truth_and_values(
            transferred, step_pln, planner["entries"]["remaining"], ini,
        )
        flagged = (
            plan_values >= float(tau) if tau is not None
            else np.zeros(len(plan_values), dtype=bool)
        )
        result[name] = dict(
            passive_oof=metrics,
            passive_band_auc=band_auc(
                values, truth, remain, ini,
            ),
            planner=plan_metrics,
            planner_band_auc=band_auc(
                plan_values, plan_truth, plan_remain, ini,
            ),
            tau=float(tau) if tau is not None else None,
            planner_precision=(
                float(plan_truth[flagged].mean()) if flagged.any() else None
            ),
            planner_flag_rate=(
                float(flagged.mean()) if len(flagged) else None
            ),
            planner_recall=(
                float(np.sum(flagged & (plan_truth > 0.5))
                      / max(1.0, np.sum(plan_truth > 0.5)))
                if len(plan_truth) else None
            ),
        )
    result["delta"] = dict(
        passive_auc=(
            (result["A1_history"]["passive_oof"] or {}).get("auc", 0.0)
            - (result["A0_state"]["passive_oof"] or {}).get("auc", 0.0)
            if (result["A1_history"]["passive_oof"] or {}).get("auc")
            is not None
            and (result["A0_state"]["passive_oof"] or {}).get("auc")
            is not None else None
        ),
        planner_auc=(
            (result["A1_history"]["planner"] or {}).get("auc", 0.0)
            - (result["A0_state"]["planner"] or {}).get("auc", 0.0)
            if (result["A1_history"]["planner"] or {}).get("auc")
            is not None
            and (result["A0_state"]["planner"] or {}).get("auc")
            is not None else None
        ),
    )
    return result


def entry_readability(planner, low=7, high=12, far=17, far_high=32):
    """Per-feature AUC separating corridor-entry states from far states."""
    entries = planner["entries"]
    remaining = np.asarray(entries["remaining"], dtype=int)
    keep = (
        ((remaining >= int(low)) & (remaining <= int(high)))
        | ((remaining >= int(far)) & (remaining <= int(far_high)))
    )
    if not keep.any():
        return []
    label = (
        (remaining[keep] >= int(low)) & (remaining[keep] <= int(high))
    ).astype(float)
    if len(np.unique(label)) < 2:
        return []
    features = np.concatenate([
        entries["phi"],
        history_features(planner["masks"], planner["names"], entries),
    ], axis=1)
    names = (
        ["state_x%d" % dim for dim in planner["dims"]]
        + ["%s_%s" % (level, term) for level in planner["names"]
           for term in ("since", "count", "recent")]
    )
    report = []
    for position, name in enumerate(names):
        column = features[keep, position]
        if float(np.ptp(column)) < 1e-12:
            continue
        auc = rank_auc(column, label)
        if auc is None:
            continue
        report.append(dict(feature=name, auc=float(auc)))
    report.sort(key=lambda item: -abs(item["auc"] - 0.5))
    return report


def verdict_of(checks):
    if not checks["span_auc"]:
        return "new_predicates_needed"
    if not (checks["early_lead"] and checks["rule_lift"]):
        return "predicates_too_late"
    if not checks["span_history"]:
        return "state_only_sufficient"
    return "predicates_sufficient"


def _load_planner(seed, args):
    from probe_cache import load_or_fit
    config = dict(
        seed=int(seed), steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, dagger_rounds=1, dagger_frames=5000,
        dagger_cap=5000, dagger_horizon=8, step_stride=2,
        curiosity_weight=0.5, death_rules=True,
        death_rule_max_thresholds=256,
        death_rule_boundary_margin=0.01,
    )
    _world, agent, cached = load_or_fit(
        config, "%s-%d.pkl" % (args.cache_prefix, seed),
        refit=bool(args.refit),
    )
    print("planner model: %s" % ("cache" if cached else "fit"))
    return [
        row for row in agent.milestone_training_rows
        if int(row.get("episode", 0)) >= 50000
    ]


def _stream(tables, dims, names, l_max, n_lags, stride, predicates=None):
    if predicates is None:
        stats = stream_stats(tables, dims)
        available = {
            item["name"]: item for item in candidate_predicates(stats, dims)
        }
    else:
        available = dict(predicates)
    found = []
    masks = []
    for name in ["terminal"] + list(names):
        if name == "terminal":
            predicate = dict(name="terminal", kind="terminal")
        elif name in available:
            predicate = available[name]
        else:
            continue
        masks.append([event_mask(table, predicate) for table in tables])
        found.append(name)
    entries = build_entries(tables, dims, n_lags, stride)
    terminal = [table["done"].copy() for table in tables]
    return dict(
        tables=tables, names=found, masks=masks, entries=entries,
        terminal=terminal, dims=list(dims), predicates=available,
        n_rows=int(sum(len(table["actions"]) for table in tables)),
        n_deaths=int(sum(int(table["done"].sum()) for table in tables)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n-lags", type=int, default=8)
    parser.add_argument("--spans", default="8,12,16")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--precision-target", type=float, default=0.5)
    parser.add_argument("--cache-prefix", default="span-planner")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    spans = [int(value) for value in str(args.spans).split(",")]
    rows, _counts = collect(
        seed=args.seed, steps=args.steps, coverage=args.coverage, lag=LAG,
    )
    passive_tables = episode_tables(rows)
    planner_tables = episode_tables(_load_planner(args.seed, args))
    obs_dim = passive_tables[0]["obs"].shape[1]
    clock = clock_report(passive_tables, obs_dim)
    dims = [dim for dim, info in clock.items() if not info["excluded"]]
    names = list(EVENT_LEVELS)
    passive = _stream(
        passive_tables, dims, names, max(spans), args.n_lags, args.stride,
    )
    planner = _stream(
        planner_tables, dims, names, max(spans), args.n_lags, args.stride,
    )
    # The transfer arm reads the planner stream through the passive
    # predicates, so the fitted feature semantics are unchanged.
    planner_transfer = _stream(
        planner_tables, dims, names, max(spans), args.n_lags, args.stride,
        predicates=passive["predicates"],
    )
    print(
        "event span probe: seed %d  passive %d rows / %d episodes / "
        "%d deaths  planner %d rows / %d episodes / %d deaths  dims %s" % (
            args.seed, passive["n_rows"], len(passive_tables),
            passive["n_deaths"], planner["n_rows"], len(planner_tables),
            planner["n_deaths"], dims,
        )
    )

    base_leads = lead_analysis(
        passive["masks"], passive["names"], passive_tables, 8,
    )
    strong = sorted(
        name for name, row in base_leads.items()
        if row is not None and (row["lift"] or 0.0) >= 2.0
    )
    print("  strong levels (P1-B lift >= 2): %s" % strong)

    timing = {}
    for stream_name, stream in (("passive", passive), ("planner", planner)):
        leads = approach_lead_report(
            stream["masks"], stream["names"], stream["tables"],
            args.horizon,
        )
        positions = [
            position for position, name in enumerate(stream["names"])
            if name in strong
        ]
        rules = {
            str(horizon): rule_warning(
                stream["masks"], positions, stream["tables"], horizon,
            )
            for horizon in (4, 8, 12, 16)
        }
        timing[stream_name] = dict(leads=leads, rules=rules)
        for name in strong:
            row = leads.get(name)
            if row is None:
                continue
            print(
                "  %-8s lead %-9s deaths %d  first med %s p90 %s  "
                "share>=8 %.2f  >=12 %.2f" % (
                    stream_name, name, row["n_deaths"],
                    "None" if row["first_lead_median"] is None
                    else "%.0f" % row["first_lead_median"],
                    "None" if row["first_lead_p90"] is None
                    else "%.0f" % row["first_lead_p90"],
                    row["share_first_ge_8"] or 0.0,
                    row["share_first_ge_12"] or 0.0,
                )
            )
        for horizon, rule in rules.items():
            print(
                "  %-8s rule h%-2s precision %s recall %s lift %s  "
                "hits rem mean %s (>=8 share %s)" % (
                    stream_name, horizon,
                    "None" if rule["precision"] is None
                    else "%.2f" % rule["precision"],
                    "None" if rule["recall"] is None
                    else "%.2f" % rule["recall"],
                    "None" if rule["lift"] is None
                    else "%.2f" % rule["lift"],
                    "None" if rule["hit_remaining_mean"] is None
                    else "%.1f" % rule["hit_remaining_mean"],
                    "None" if rule["hit_remaining_ge_8"] is None
                    else "%.2f" % rule["hit_remaining_ge_8"],
                )
            )

    spans_report = {}
    for span in spans:
        spans_report[str(span)] = transfer_block(
            passive, planner_transfer, span, args,
        )
        result = spans_report[str(span)]
        print(
            "  span %-2d  passive OOF AUC %s skill %s | planner AUC %s "
            "skill %s band %s  tau %s  precision %s coverage %s  "
            "dAUC(state->hist) pas %s plan %s" % (
                span,
                _fmt(result["A1_history"]["passive_oof"], "auc"),
                _fmt(result["A1_history"]["passive_oof"], "brier_skill"),
                _fmt(result["A1_history"]["planner"], "auc"),
                _fmt(result["A1_history"]["planner"], "brier_skill"),
                _fmt(result["A1_history"], "planner_band_auc"),
                "None" if result["A1_history"]["tau"] is None
                else "%.3f" % result["A1_history"]["tau"],
                _fmt(result["A1_history"], "planner_precision"),
                _fmt(result["A1_history"], "planner_flag_rate"),
                _fmt_plain(result["delta"]["passive_auc"]),
                _fmt_plain(result["delta"]["planner_auc"]),
            )
        )

    readability = entry_readability(planner)
    print("  entry readability (rem 7-12 vs 17-32):")
    for item in readability[:6]:
        print("     %-16s AUC %.3f" % (item["feature"], item["auc"]))

    long_spans = [
        span for span in spans if span >= 12
    ] or [max(spans)]
    best_span, best = max(
        ((span, spans_report[str(span)]) for span in long_spans),
        key=lambda item: (
            (item[1]["A1_history"]["planner"] or {}).get("auc") or -1.0
        ),
    )
    checks = dict(
        early_lead=bool(any(
            (row or {}).get("share_first_ge_8") is not None
            and (row or {}).get("share_first_ge_8") >= 0.25
            for row in timing["planner"]["leads"].values()
        )),
        rule_lift=bool(any(
            (rule["lift"] or 0.0) >= 1.5
            and (rule["recall"] or 0.0) >= 0.5
            for horizon, rule in timing["planner"]["rules"].items()
            if int(horizon) >= 8
        )),
        span_auc=bool(
            (best["A1_history"]["planner"] or {}).get("auc") is not None
            and best["A1_history"]["planner"]["auc"] >= 0.75
        ),
        span_history=bool(
            best["delta"]["planner_auc"] is not None
            and best["delta"]["planner_auc"] >= 0.02
        ),
        best_span=int(best_span),
    )
    verdict = verdict_of(checks)
    print(
        "== event span verdict: %s  (checks %s) ==" % (
            verdict, json.dumps(checks),
        )
    )
    report = dict(
        seed=args.seed, spans=spans, strong=strong,
        passive=dict(rows=passive["n_rows"], episodes=len(passive_tables),
                     deaths=passive["n_deaths"]),
        planner=dict(rows=planner["n_rows"], episodes=len(planner_tables),
                     deaths=planner["n_deaths"]),
        timing=timing, spans_report=spans_report,
        readability=readability, checks=checks, verdict=verdict,
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return report


def _fmt(row, key):
    value = (row or {}).get(key)
    return "None" if value is None else "%.3f" % value


def _fmt_plain(value):
    return "None" if value is None else "%+.3f" % value


if __name__ == "__main__":
    main()
