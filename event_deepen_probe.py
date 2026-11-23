"""P1-D.1: the 6-step executor with event-triggered deeper replans.

The combination arm keeps the 6-step fine executor for every decision and,
only where the deployable event warning fires (the P1-D.0-calibrated
threshold), re-plans the same state at the 8-step horizon and executes the
deeper search's own first action.  The warning is a trigger, never a cost
term: the fine layer remains the only decision maker, the deeper search is
the sanctioned local-horizon expansion, and a cooldown bounds how often a
state can trigger.

Arms run under the benchmark protocol (each model trained with its own
horizon, held-out worlds 120-124, decision=search, death rules off):
A_h8 is the default stack, A_h6 the short executor alone, A_deep the
combination.  The gate is paired: A_deep must reach the h8 baseline on
frames without losing passes.

Usage::

    python event_deepen_probe.py --seed 100 --horizons 6,8 \
        --report event-deepen-100.json
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
from probe_cache import load_or_fit
from substrate_nodes import LAG, _curiosity_off, policy_result


def run_arm(seed, world_model, agent, horizon, eval_seed, model, tau,
            cooldown, deep_horizon, warn_arm="A1_history"):
    """One evaluation episode; `model` None disables the warning arm."""
    env = latent[LAG](int(eval_seed))
    events = None
    issued = deque([0.0] * 8, maxlen=8)
    prefix = np.zeros(LAG, dtype=float)
    cool = 0
    stats = dict(
        decisions=0, triggers=0, deep_plans=0, action_changes=0,
    )
    while not env.dead and env.t < 5000:
        stats["decisions"] += 1
        plan = None
        used = "h%d" % int(horizon)
        if model is None:
            with _curiosity_off(agent):
                result = policy_result(
                    env, world_model, agent, prefix, int(horizon),
                    lag=LAG, step_stride=2,
                )
            plan = np.asarray(result.actions, dtype=int)
        else:
            observation = np.asarray(env.obs(), dtype=float)
            events.update(observation)
            phi = phi_for(model, observation)
            hist = events.features()
            known = list(issued)[: max(0, int(model["n_lags"]) - 1)]
            warning = max(
                warning_value(model, phi, hist, known, 0.0, arm=warn_arm),
                warning_value(model, phi, hist, known, 1.0, arm=warn_arm),
            )
            with _curiosity_off(agent):
                result = policy_result(
                    env, world_model, agent, prefix, int(horizon),
                    lag=LAG, step_stride=2,
                )
            plan = np.asarray(result.actions, dtype=int)
            if (
                warning >= float(tau) and cool == 0
                and int(deep_horizon) > int(horizon)
            ):
                stats["triggers"] += 1
                with _curiosity_off(agent):
                    deep = policy_result(
                        env, world_model, agent, prefix, int(deep_horizon),
                        lag=LAG, step_stride=2,
                    )
                deep_plan = np.asarray(deep.actions, dtype=int)
                if len(deep_plan):
                    if int(deep_plan[0]) != int(plan[0]):
                        stats["action_changes"] += 1
                    plan = deep_plan
                    used = "deep_h%d" % int(deep_horizon)
                    stats["deep_plans"] += 1
                cool = int(cooldown)
        action = int(plan[0]) if len(plan) else 0
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        issued.appendleft(float(action))
        if model is not None and cool > 0:
            cool -= 1
        env.step(action)
    if events is not None:
        stats["warning_rows"] = int(events.next_row)
    metrics_passes = 0
    return dict(
        train_seed=int(seed), arm=used if False else "arm",
        horizon=int(horizon), eval_seed=int(eval_seed),
        frames=int(env.t), passes=int(getattr(env, "passed", 0)),
        action_rate=0.0, stats=stats,
    )


def run_seed_arms(seed, horizons, eval_seeds, args, model, tau):
    configs = {horizon: horizon_config(seed, horizon) for horizon in horizons}
    models = {}
    for horizon, config in configs.items():
        world_model, agent, cached = load_or_fit(
            config, "%s-%d-%d.pkl" % (args.cache_prefix, seed, horizon),
            refit=bool(args.refit),
        )
        models[horizon] = (world_model, agent)
        print(
            "seed %d horizon %d: model %s" % (
                seed, horizon, "cache" if cached else "fit",
            )
        )
    rows = []
    for eval_seed in eval_seeds:
        for horizon in horizons:
            row = run_arm(
                seed, models[horizon][0], models[horizon][1], horizon,
                eval_seed, None, tau, 0, horizon, args,
            )
            row["arm"] = "h%d" % int(horizon)
            rows.append(row)
            print(
                "  seed %d arm h%-2d world %d: frames %4d passes %d" % (
                    seed, horizon, eval_seed, row["frames"], row["passes"],
                )
            )
        base_horizon = min(horizons)
        row = run_arm(
            seed, models[base_horizon][0], models[base_horizon][1],
            base_horizon, eval_seed, model, tau, args.cooldown,
            args.deep_horizon, args,
        )
        row["arm"] = "deep"
        rows.append(row)
        stats = row["stats"]
        print(
            "  seed %d arm deep world %d: frames %4d passes %d  "
            "triggers %d deep %d changes %d" % (
                seed, eval_seed, row["frames"], row["passes"],
                stats["triggers"], stats["deep_plans"],
                stats["action_changes"],
            )
        )
    return rows


def summarize(rows, horizons):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["train_seed"], row["eval_seed"]), {})[
            row["arm"]
        ] = row
    arms = ["h%d" % h for h in horizons] + ["deep"]
    pairs = []
    per_seed = {}
    totals = {arm: dict(frames=[], passes=[]) for arm in arms}
    for (train_seed, eval_seed), pair in sorted(grouped.items()):
        if not all(arm in pair for arm in arms):
            continue
        entry = dict(
            train_seed=train_seed, eval_seed=eval_seed,
            frames={arm: pair[arm]["frames"] for arm in arms},
            passes={arm: pair[arm]["passes"] for arm in arms},
        )
        entry["delta_deep_h8"] = (
            entry["frames"]["deep"] - entry["frames"]["h%d" % max(horizons)]
        )
        entry["delta_deep_h6"] = (
            entry["frames"]["deep"] - entry["frames"]["h%d" % min(horizons)]
        )
        pairs.append(entry)
        for arm in arms:
            totals[arm]["frames"].append(entry["frames"][arm])
            totals[arm]["passes"].append(entry["passes"][arm])
        seed_entry = per_seed.setdefault(train_seed, dict(
            deep=[], h8=[], h6=[],
        ))
        seed_entry["deep"].append(entry["frames"]["deep"])
        seed_entry["h8"].append(entry["frames"]["h%d" % max(horizons)])
        seed_entry["h6"].append(entry["frames"]["h%d" % min(horizons)])
    means = {
        arm: dict(
            frames=float(np.mean(values["frames"]))
            if values["frames"] else None,
            passes=float(np.mean(values["passes"]))
            if values["passes"] else None,
        )
        for arm, values in totals.items()
    }
    seed_summary = {
        seed: dict(
            mean_deep=float(np.mean(values["deep"])),
            mean_h8=float(np.mean(values["h8"])),
            mean_h6=float(np.mean(values["h6"])),
        )
        for seed, values in per_seed.items()
    }
    deep_better = int(sum(
        item["delta_deep_h8"] > 0 for item in pairs
    ))
    deep_worse = int(sum(
        item["delta_deep_h8"] < 0 for item in pairs
    ))
    overall = dict(
        mean_frames=means, n=len(pairs),
        deep_vs_h8=float(np.mean([
            item["delta_deep_h8"] for item in pairs
        ])) if pairs else None,
        deep_vs_h6=float(np.mean([
            item["delta_deep_h8"] - item["delta_deep_h6"] for item in pairs
        ])) if pairs else None,
        deep_better=deep_better, deep_worse=deep_worse,
    )
    return dict(pairs=pairs, means=means, per_seed=seed_summary,
                overall=overall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--horizons", default="6,8")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--deep-horizon", type=int, default=8)
    parser.add_argument("--cooldown", type=int, default=4)
    parser.add_argument("--precision-target", type=float, default=0.5)
    parser.add_argument("--cache-prefix", default="exec")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    from death_transition_quality import seed_spec
    horizons = [int(value) for value in str(args.horizons).split(",")]
    eval_seeds = seed_spec(args.eval_seeds)
    model = fit_warning_model(
        seed=args.seed, precision_target=args.precision_target,
    )
    tau = model["tau"]
    print(
        "event deepen probe: seed %d  tau %s (coverage %.3f)  "
        "cooldown %d  deep horizon %d" % (
            args.seed, "None" if tau is None else "%.4f" % tau,
            model["tau_coverage"] or 0.0, args.cooldown,
            args.deep_horizon,
        )
    )
    rows = run_seed_arms(
        args.seed, horizons, eval_seeds, args, model, tau,
    )
    summary = summarize(rows, horizons)
    overall = summary["overall"]
    print(
        "== arms mean frames: h%d %.1f  h%d %.1f  deep %.1f  "
        "(deep vs h8 %+.1f)  better %d / worse %d of %d ==" % (
            min(horizons), overall["mean_frames"]["h%d" % min(horizons)],
            max(horizons), overall["mean_frames"]["h%d" % max(horizons)],
            overall["mean_frames"]["deep"],
            overall["deep_vs_h8"], overall["deep_better"],
            overall["deep_worse"], overall["n"],
        )
    )
    for seed, entry in sorted(summary["per_seed"].items()):
        print(
            "   seed %d: h%d %.1f  h%d %.1f  deep %.1f" % (
                seed, min(horizons), entry["mean_h6"],
                max(horizons), entry["mean_h8"], entry["mean_deep"],
            )
        )
    gate = dict(
        pooled=bool(
            overall["mean_frames"]["deep"] is not None
            and overall["mean_frames"]["h%d" % max(horizons)] is not None
            and overall["mean_frames"]["deep"]
            >= overall["mean_frames"]["h%d" % max(horizons)] - 5.0
        ),
        per_seed=all(
            entry["mean_deep"] >= entry["mean_h8"] - 5.0
            for entry in summary["per_seed"].values()
        ),
    )
    verdict = "pass" if all(gate.values()) else "fail"
    print(
        "== P1-D.1 gate: %s  (pooled %s, per-seed %s) ==" % (
            verdict,
            "pass" if gate["pooled"] else "fail",
            "pass" if gate["per_seed"] else "fail",
        )
    )
    report = dict(
        seed=args.seed, horizons=horizons, deep_horizon=int(
            args.deep_horizon
        ), cooldown=int(args.cooldown), rows=rows, summary=summary,
        gate=gate, verdict=verdict,
        warning_meta=dict(
            tau=model["tau"], tau_coverage=model["tau_coverage"],
        ),
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return report


if __name__ == "__main__":
    main()
