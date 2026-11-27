"""P1-D.0: does the event warning flag imminent death on planner states?

The gate before building the combination: replay the 6-step executor's own
evaluation episodes and, at every decision, score the deployable event
warning (fitted on the passive behaviour stream, applied causally to the
running episode).  The label is death imminence — the episode ends within
eight frames of this decision — which is exactly the quantity the warning
hazard was trained on and exactly what the 6-step horizon cannot see.

Read-only: the planner, its objective and the default stack are untouched.
The gate: the warning must separate imminent-death decisions from safe ones
on the planner's own state distribution; if it does, the triggered
deepening combination (P1-D.1) is worth building; if not, the consumption
mechanism has to change.

Usage::

    python warning_trigger_probe.py --seed 100 --horizon 6 \
        --report warning-trigger-100.json
"""
from __future__ import annotations

import argparse
import json
from collections import deque

import numpy as np

from event_warning import (
    OnlineEvents, fit_warning_model, phi_for, warning_value,
)
from executor_horizon_probe import horizon_config
from latent_world import latent
from planner_rollout_quality import oracle_plan
from probe_cache import load_or_fit
from substrate_nodes import LAG, _curiosity_off, policy_result


def replay_world(fitted, eval_seed, agent, horizon, model, cap,
                 trigger=None):
    env = latent[LAG](int(eval_seed))
    events = OnlineEvents(
        model["names"], model["predicates"], cap=model["cap"],
        count_w=model["count_w"], recent_w=model["recent_w"],
    )
    issued = deque([0.0] * 8, maxlen=8)
    prefix = np.zeros(LAG, dtype=float)
    rows = []
    while not env.dead and env.t < int(cap):
        observation = np.asarray(env.obs(), dtype=float)
        events.update(observation)
        phi = phi_for(model, observation)
        hist = events.features()
        known = list(issued)[: max(0, int(model["n_lags"]) - 1)]
        value = max(
            warning_value(model, phi, hist, known, 0.0),
            warning_value(model, phi, hist, known, 1.0),
        )
        with _curiosity_off(agent):
            result = policy_result(
                env, fitted, agent, prefix, int(horizon), lag=LAG,
                step_stride=2,
            )
        plan = np.asarray(result.actions, dtype=int)
        death_step, _state, _dead = oracle_plan(env, plan)
        rows.append(dict(
            world=int(eval_seed), row=int(events.current_row),
            warning=float(value),
            plan_death=int(death_step),
            action=int(plan[0]),
        ))
        action = int(plan[0])
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        issued.appendleft(float(action))
        env.step(action)
    total = int(env.t)
    for index, row in enumerate(rows):
        row["remaining"] = total - int(row["row"])
        row["dies_within_8"] = int(row["remaining"] <= 8)
    if trigger is not None:
        for row in rows:
            row["triggered"] = bool(row["warning"] >= float(trigger))
    return rows, total


def summarize(rows, horizon, tau, tau_source):
    labels = np.asarray([row["dies_within_8"] for row in rows], dtype=float)
    values = np.asarray([row["warning"] for row in rows], dtype=float)
    invisible = np.asarray([
        1 if (row["plan_death"] >= int(horizon)) else 0 for row in rows
    ], dtype=float)
    from endpoint_correction_probe import rank_auc
    result = dict(
        decisions=int(len(rows)),
        deaths_within_8=int(labels.sum()),
        base_rate=float(labels.mean()) if len(labels) else None,
        invisible_deaths=int(invisible.sum()),
        auc_imminent=rank_auc(values, labels),
        auc_invisible=rank_auc(values, invisible),
        tau=float(tau), tau_source=tau_source,
    )
    if tau is not None:
        flagged = values >= float(tau)
        result.update(
            trigger_rate=float(flagged.mean()) if len(flagged) else None,
            flagged_deaths=int((flagged & (labels > 0.5)).sum()),
            precision_at_tau=(
                float((labels[flagged]).mean()) if flagged.any() else None
            ),
            caught=int(np.sum(flagged & (labels > 0.5))),
        )
    profile = {}
    for low, high in ((1, 4), (5, 8), (9, 16), (17, 10 ** 6)):
        mask = np.asarray([
            low <= row["remaining"] <= high for row in rows
        ], dtype=bool)
        if mask.any():
            key = "remaining_%d_%s" % (
                low, "inf" if high >= 10 ** 6 else str(high),
            )
            profile[key] = float(values[mask].mean())
    result["warning_by_remaining"] = profile
    result["caught_deaths"] = int(np.sum(
        (values >= float(tau)) & (labels > 0.5)
    )) if tau is not None else 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--precision-target", type=float, default=0.5)
    parser.add_argument("--warning-source", default="behavior",
                        choices=("behavior", "training", "planner"),
                        help="which stream the warning model is fitted on")
    parser.add_argument("--cache-prefix", default="exec")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    from death_transition_quality import seed_spec
    eval_seeds = seed_spec(args.eval_seeds)
    model = fit_warning_model(
        seed=args.seed, precision_target=args.precision_target,
        source=args.warning_source,
        cache="%s-%d-%d.pkl" % (args.cache_prefix, args.seed, args.horizon),
        refit=bool(args.refit), horizon=args.horizon,
    )
    config = horizon_config(args.seed, args.horizon)
    world_model, agent, cached = load_or_fit(
        config, "%s-%d-%d.pkl" % (args.cache_prefix, args.seed, args.horizon),
        refit=bool(args.refit),
    )
    print(
        "warning trigger probe: seed %d, warning episodes %d, "
        "tau %.4f (coverage %.3f, precision %.3f), executor model %s" % (
            args.seed, model["episodes"], model["tau"] or 0.0,
            model["tau_coverage"] or 0.0, model["tau_precision"] or 0.0,
            "cache" if cached else "fit",
        )
    )
    rows = []
    for eval_seed in eval_seeds:
        world_rows, _total = replay_world(
            world_model, eval_seed, agent, args.horizon, model,
            args.cap,
        )
        rows.extend(world_rows)
    summary = summarize(rows, args.horizon, model["tau"], "precision_target")
    print(
        "== decisions %d  deaths-within-8 %d (base %.3f)  "
        "AUC imminent %s  AUC invisible %s ==" % (
            summary["decisions"], summary["deaths_within_8"],
            summary["base_rate"],
            "None" if summary["auc_imminent"] is None
            else "%.3f" % summary["auc_imminent"],
            "None" if summary["auc_invisible"] is None
            else "%.3f" % summary["auc_invisible"],
        )
    )
    for key, value in summary["warning_by_remaining"].items():
        print("   %-18s mean warning %.3f" % (key, value))
    if summary.get("trigger_rate") is not None:
        print(
            "   tau %.3f: trigger rate %.3f  precision %.3f  "
            "caught %d/%d" % (
                summary["tau"], summary["trigger_rate"],
                summary["precision_at_tau"], summary["caught"],
                summary["deaths_within_8"],
            )
        )
    gate = dict(
        auc=bool(
            summary["auc_imminent"] is not None
            and summary["auc_imminent"] >= 0.75
        ),
        coverage=bool(
            summary.get("precision_at_tau") is not None
            and summary["precision_at_tau"] >= args.precision_target
            and summary.get("caught", 0) >= 0.6 * max(
                1, summary["deaths_within_8"]
            )
        ),
        cost=bool(
            summary.get("trigger_rate") is not None
            and summary["trigger_rate"] <= 0.35
        ),
    )
    verdict = "pass" if all(gate.values()) else "fail"
    print(
        "== P1-D.0 gate: %s  (auc %s, coverage %s, cost %s) ==" % (
            verdict,
            "pass" if gate["auc"] else "fail",
            "pass" if gate["coverage"] else "fail",
            "pass" if gate["cost"] else "fail",
        )
    )
    report = dict(
        seed=args.seed, horizon=int(args.horizon), summary=summary,
        gate=gate, verdict=verdict, rows=rows,
        warning_meta=dict(
            tau=model["tau"], tau_coverage=model["tau_coverage"],
            tau_precision=model["tau_precision"],
            base_rate=model["base_rate"], n_entries=model["n_entries"],
        ),
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return report


if __name__ == "__main__":
    main()
