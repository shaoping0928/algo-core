"""Score planner candidate endpoints with the span-16 state readout.

The E2 diagnostics showed the deploy-stream readout ranks states well but its
threshold does not transport across streams.  A decision-side consumer does
not need a global threshold: it needs to order the candidates of *one*
decision.  This probe therefore measures the readout inside each decision.

For every B0 decision it replays the same candidate set (predicted endpoints
and true outcomes), fits the span-16 state-only table on a deployment-
configured planner stream, scores each candidate endpoint, and reports:

* within-decision AUC of the hazard against true death within 16 frames;
* same-first-action AUC and top-1 (choose the safest candidate);
* changed / improved / worsened selections against the default fine score;
* the same numbers for the fine score itself, as the control.

Read-only: no model enters the default path; the readout is fitted only from
real executed transitions collected under the deployment configuration
(curiosity off), never from counterfactual outcomes.

Usage::

    python span_decision_probe.py --seed 101 --eval-seeds 120-124 \
        --report span-decision-101.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from decision_b0_probe import run as b0_run
from endpoint_correction_probe import (
    choose, decision_metrics, ood_reference, rank_auc, training_index,
    within_decision_metrics,
)
from endpoint_correction_probe import build_dataset
from event_span_probe import _stream
from event_transition_probe import EVENT_LEVELS
from outcome_level_probe import clock_report, entry_labels, episode_tables
from substrate_nodes import LAG, _curiosity_off, collect_planner
from tree_transition_probe import HORIZON, _assign, fit_tree, predict_tree


def fit_span_table(agent, world, seed, args):
    """Fit the state-only span table on two candidate training streams.

    The deploy stream is on-policy under the deployment configuration; the
    DAgger stream is the planner's own sampled state distribution.  Candidate
    endpoints are off-policy predictions, so which stream transports to them
    is an empirical question, not an assumption.
    """
    with _curiosity_off(agent):
        rows, _frames = collect_planner(
            int(seed), world, agent,
            episodes=int(args.deploy_frames),
            cap=5000, horizon=int(args.horizon), lag=LAG, step_stride=2,
            offset=int(args.deploy_offset),
            max_frames=int(args.deploy_frames),
        )
    dagger_rows = [
        row for row in agent.milestone_training_rows
        if int(row.get("episode", 0)) >= 50000
    ]
    readouts = {}
    for name, source in (("deploy", rows), ("dagger", dagger_rows)):
        tables = episode_tables(source)
        if not tables:
            continue
        dims = [
            dim for dim, info in clock_report(
                tables, int(agent.obs_dim),
            ).items() if not info["excluded"]
        ]
        stream = _stream(
            tables, dims, EVENT_LEVELS, HORIZON,
            int(args.n_lags), int(args.stride),
        )
        entries = stream["entries"]
        step, _duration, _ = entry_labels(
            stream["terminal"], entries, HORIZON,
        )
        model = fit_tree(
            np.asarray(entries["phi"], dtype=float), step,
            entries["remaining"], horizon=HORIZON,
            k_fine=int(args.k_fine), k_coarse=int(args.k_coarse),
            seed=int(seed),
        )
        readouts[name] = dict(
            model=model, mean=np.asarray(entries["mean"]),
            scale=np.asarray(entries["scale"]), dims=dims,
        )
        print(
            "span table[%s]: %d rows, %d episodes, dims %s, deaths %d" % (
                name, len(source), len(tables), dims, stream["n_deaths"],
            )
        )
    return readouts


def score_endpoints(readout, endpoints, window):
    """Hazard for raw candidate endpoints (higher = more dangerous)."""
    values = np.asarray(endpoints, dtype=float)[:, readout["dims"]]
    phi = (values - readout["mean"]) / readout["scale"]
    return predict_tree(readout["model"], phi, int(window))


def endpoint_support(readout, endpoints):
    """Fine-node support of each endpoint (how much stop-time data backs it)."""
    values = np.asarray(endpoints, dtype=float)[:, readout["dims"]]
    phi = (values - readout["mean"]) / readout["scale"]
    model = readout["model"]
    stand = (phi - model["mean"]) / model["scale"]
    fine = _assign(stand, model["centers"])
    return np.asarray(model["fine_total"], dtype=float)[fine]


def per_decision_auc(score, item, horizon, same_action=False):
    labels = (
        np.asarray(item["true_value"], dtype=int) <= int(horizon)
    ).astype(float)
    probability = np.asarray(score, dtype=float)
    if same_action:
        local = {
            int(index): position
            for position, index in enumerate(item["indices"])
        }
        positions = np.asarray([
            local[int(index)]
            for index in np.flatnonzero(item["same_mask"])
        ], dtype=int)
        if positions.size:
            mask = np.zeros(len(probability), dtype=bool)
            mask[positions] = True
            probability = probability[mask]
            labels = labels[mask]
    if np.unique(labels).size != 2:
        return None
    return rank_auc(probability, labels)


def selection_summary(rows, mixed_only=False):
    if mixed_only:
        rows = [row for row in rows if row.get("mixed")]
    return dict(
        decisions=len(rows),
        changed=sum(1 for row in rows if row["changed"]),
        improved=sum(1 for row in rows if row["improved"]),
        worsened=sum(1 for row in rows if row["worsened"]),
        same_action_oracle=sum(
            1 for row in rows if row["same_action_best"]
        ),
        global_oracle=sum(1 for row in rows if row["global_best"]),
        mean_chosen_value=(
            float(np.mean([row["chosen_value"] for row in rows]))
            if rows else None
        ),
        mean_same_regret=(
            float(np.mean([row["same_action_regret"] for row in rows]))
            if rows else None
        ),
        mean_global_regret=(
            float(np.mean([row["global_regret"] for row in rows]))
            if rows else None
        ),
        relevant_miss=sum(1 for row in rows if row["relevant_miss"]),
    )


def choose_rows(decisions, scores_by_decision, scale, horizon):
    rows = []
    for key, item in decisions.items():
        probability = np.asarray(scores_by_decision[key], dtype=float)
        same_local = np.asarray([
            position for position, index in enumerate(item["indices"])
            if item["same_mask"][int(index)]
        ], dtype=int)
        if not same_local.size:
            continue
        mask = np.zeros(len(probability), dtype=bool)
        mask[same_local] = True
        same_labels = (
            np.asarray(item["true_value"], dtype=int)[same_local]
            <= int(horizon)
        )
        mixed = bool(same_labels.any() and (~same_labels).any())
        position = choose(probability, mask, item["scores"])
        chosen_index = int(item["indices"][position])
        chosen_value = int(item["true_value"][position])
        row = decision_metrics(item, chosen_index, chosen_value)
        row["mixed"] = mixed
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--span", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--deploy-frames", type=int, default=5000)
    parser.add_argument("--deploy-offset", type=int, default=70000)
    parser.add_argument("--n-lags", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--k-fine", type=int, default=32)
    parser.add_argument("--k-coarse", type=int, default=4)
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
    summary = b0_run(b0_args, trace_out=records, context_out=context)
    world = context["world"]
    agent = context["agent"]
    obs_dim = int(agent.obs_dim)
    index = training_index(
        agent.milestone_training_rows, obs_dim, LAG, agent.scales,
    )
    reference = ood_reference(index)
    row_map = {
        (row["eval_seed"], row["point"]): row
        for row in summary["rows"]
    }
    decisions = {}
    for record in records:
        key = (record["eval_seed"], record["point"])
        if key not in row_map:
            continue
        built = build_dataset(
            record, row_map[key], index, reference, obs_dim,
        )
        if built is not None:
            decisions[key] = built
    if not decisions:
        raise SystemExit("no decisions")
    print(
        "span decision probe: seed %d, %d decisions, %d candidates" % (
            args.seed, len(decisions),
            sum(len(item["indices"]) for item in decisions.values()),
        )
    )

    readouts = fit_span_table(agent, world, args.seed, args)
    fine_scores = {
        key: np.asarray(item["scores"], dtype=float)[:, 0]
        for key, item in decisions.items()
    }
    uniform = {
        key: np.zeros(len(item["indices"]), dtype=float)
        for key, item in decisions.items()
    }

    arms = {}
    for source, readout in readouts.items():
        for window in (int(args.horizon), int(args.span)):
            name = "%s%d" % (source, window)
            scores = {}
            supports = {}
            for key, item in decisions.items():
                raw = np.asarray(item["endpoint"], dtype=float) * np.asarray(
                    index["scale"], dtype=float
                )
                scores[key] = score_endpoints(readout, raw, window)
                supports[key] = float(np.median(
                    endpoint_support(readout, raw)
                ))
            aucs = []
            support_aucs = []
            for key, item in decisions.items():
                auc = per_decision_auc(scores[key], item, args.horizon)
                if auc is not None:
                    aucs.append(float(auc))
                    support_aucs.append(supports[key])
            predictions = {
                key: dict(decision=item, prediction=scores[key])
                for key, item in decisions.items()
            }
            rows = choose_rows(
                decisions, scores, index["scale"], args.horizon,
            )
            correlation = None
            if len(aucs) > 2 and np.std(aucs) > 0 and np.std(support_aucs) > 0:
                correlation = float(np.corrcoef(support_aucs, aucs)[0, 1])
            arms[name] = dict(
                within=within_decision_metrics(predictions, args.horizon),
                selection=selection_summary(rows),
                mixed_selection=selection_summary(rows, mixed_only=True),
                support_auc_correlation=correlation,
            )

    for name, scores in (("fine", fine_scores), ("shuffle", uniform)):
        predictions = {
            key: dict(decision=item, prediction=scores[key])
            for key, item in decisions.items()
        }
        rows = choose_rows(decisions, scores, index["scale"], args.horizon)
        arms[name] = dict(
            within=within_decision_metrics(predictions, args.horizon),
            selection=selection_summary(rows),
            mixed_selection=selection_summary(rows, mixed_only=True),
            support_auc_correlation=None,
        )

    # B0 labels stop at horizon + 1, so only the eight-frame AUC is honest;
    # the span16 arms are rankers for the selection comparison, not a
    # differently-labelled AUC problem.
    for name in arms:
        within = arms[name]["within"]
        mixed = arms[name]["mixed_selection"]
        print(
            "  %-9s AUC %s (n=%d)  same %s (n=%d)  top1 %d  |  "
            "mixed n=%d imp %d wor %d regret %s  |  support-corr %s" % (
                name,
                _fmt(within["decision_auc_mean"]),
                within["decision_auc_count"],
                _fmt(within["same_action_auc_mean"]),
                within["same_action_auc_count"],
                within["same_action_top1"],
                mixed["decisions"], mixed["improved"],
                mixed["worsened"], _fmt(mixed["mean_same_regret"]),
                _fmt(arms[name]["support_auc_correlation"]),
            )
        )

    report = dict(
        seed=args.seed, decisions=len(decisions),
        span=args.span, arms=arms,
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return report


def _fmt(value):
    return "n/a" if value is None else "%.3f" % value


if __name__ == "__main__":
    main()
