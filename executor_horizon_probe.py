"""P0-F: 6-step vs 8-step fine-executor baseline on the benchmark protocol.

Batch 8's executor calibration.  The plan moves the fine executor's horizon
to 6 steps; before anything is wired, this probe measures what that change
alone does on the canonical benchmark: train each seed under the standard
default configuration (behaviour stream 1500 frames, one DAgger round of
5000 frames collected with the same horizon), then evaluate on the held-out
worlds 120-124 with `decision=search` and the default objective.  Nothing
else changes: point-event pain, no death rules (the benchmark default),
margin_eps 0, veto_fraction 0.

This is executor calibration, not the event/stage design: frames and passes
are the primary rulers, the action rate guards against degenerate flapping,
and every comparison is paired per (training seed, evaluation world).  The
default stack is not modified by this probe.

Usage::

    python executor_horizon_probe.py --seed 100 --horizons 6,8 \
        --eval-seeds 120-124 --report executor-horizon-100.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from death_transition_quality import seed_spec
from probe_cache import load_or_fit
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def horizon_config(seed, horizon):
    return dict(
        seed=int(seed), steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=int(horizon), step_stride=2,
    )


def evaluate_world(seed, world, agent, horizon, eval_seed):
    metrics = {}
    with _curiosity_off(agent):
        frames, actions, _rows = evaluate(
            int(seed), world, agent, cap=5000, horizon=int(horizon),
            lag=LAG, step_stride=2, env_seed=int(eval_seed) * 10000,
            decision="search", metrics_out=metrics,
        )
    return dict(
        train_seed=int(seed),
        horizon=int(horizon),
        eval_seed=int(eval_seed),
        frames=int(frames),
        passes=int(metrics.get("passes", 0)),
        action_rate=float(np.mean(actions)) if len(actions) else 0.0,
    )


def run_pair(seed, horizon, eval_seeds, cache_prefix, refit):
    config = horizon_config(seed, horizon)
    world, agent, cached = load_or_fit(
        config, "%s-%d-%d.pkl" % (cache_prefix, seed, horizon),
        refit=bool(refit),
    )
    print(
        "seed %d horizon %d: model %s" % (
            seed, horizon, "cache" if cached else "fit",
        )
    )
    return [
        evaluate_world(seed, world, agent, horizon, eval_seed)
        for eval_seed in eval_seeds
    ]


def summarize(rows):
    horizons = sorted(set(row["horizon"] for row in rows))
    if len(horizons) != 2:
        return None
    low, high = horizons
    grouped = {}
    for row in rows:
        grouped.setdefault((row["train_seed"], row["eval_seed"]), {})[
            row["horizon"]
        ] = row
    deltas = []
    per_seed = {}
    for (train_seed, eval_seed), pair in sorted(grouped.items()):
        if low not in pair or high not in pair:
            continue
        delta_frames = pair[low]["frames"] - pair[high]["frames"]
        delta_passes = pair[low]["passes"] - pair[high]["passes"]
        deltas.append(dict(
            train_seed=train_seed, eval_seed=eval_seed,
            frames_low=pair[low]["frames"], frames_high=pair[high]["frames"],
            delta_frames=delta_frames,
            passes_low=pair[low]["passes"], passes_high=pair[high]["passes"],
            delta_passes=delta_passes,
            action_low=pair[low]["action_rate"],
            action_high=pair[high]["action_rate"],
        ))
        entry = per_seed.setdefault(train_seed, dict(frames=[], passes=[]))
        entry["frames"].append(delta_frames)
        entry["passes"].append(delta_passes)
    seed_summary = {
        seed: dict(
            mean_delta_frames=float(np.mean(values["frames"])),
            mean_delta_passes=float(np.mean(values["passes"])),
            n=len(values["frames"]),
        )
        for seed, values in per_seed.items()
    }
    overall = dict(
        mean_delta_frames=(
            float(np.mean([item["delta_frames"] for item in deltas]))
            if deltas else None
        ),
        mean_delta_passes=(
            float(np.mean([item["delta_passes"] for item in deltas]))
            if deltas else None
        ),
        improved=int(sum(item["delta_frames"] > 0 for item in deltas)),
        worsened=int(sum(item["delta_frames"] < 0 for item in deltas)),
        n=len(deltas),
        mean_frames_low=(
            float(np.mean([
                item["frames_low"] for item in deltas
            ])) if deltas else None
        ),
        mean_frames_high=(
            float(np.mean([
                item["frames_high"] for item in deltas
            ])) if deltas else None
        ),
        mean_passes_low=(
            float(np.mean([
                item["passes_low"] for item in deltas
            ])) if deltas else None
        ),
        mean_passes_high=(
            float(np.mean([
                item["passes_high"] for item in deltas
            ])) if deltas else None
        ),
        mean_action_low=(
            float(np.mean([
                item["action_low"] for item in deltas
            ])) if deltas else None
        ),
        mean_action_high=(
            float(np.mean([
                item["action_high"] for item in deltas
            ])) if deltas else None
        ),
    )
    return dict(pairs=deltas, per_seed=seed_summary, overall=overall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--horizons", default="6,8")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--cache-prefix", default="exec")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    horizons = [int(value) for value in str(args.horizons).split(",")]
    eval_seeds = seed_spec(args.eval_seeds)
    rows = []
    for horizon in horizons:
        rows.extend(
            run_pair(
                args.seed, horizon, eval_seeds, args.cache_prefix,
                args.refit,
            )
        )
    for row in rows:
        print(
            "  seed %d h%d world %d: frames %4d passes %d flap %.3f" % (
                row["train_seed"], row["horizon"], row["eval_seed"],
                row["frames"], row["passes"], row["action_rate"],
            )
        )
    summary = summarize(rows)
    if summary is not None:
        overall = summary["overall"]
        print(
            "== h%d vs h%d: mean frames %.1f vs %.1f (delta %+.1f)  "
            "mean passes %.2f vs %.2f (delta %+.2f)  "
            "improved %d / worsened %d of %d ==" % (
                min(horizons), max(horizons),
                overall["mean_frames_low"], overall["mean_frames_high"],
                overall["mean_delta_frames"],
                overall["mean_passes_low"], overall["mean_passes_high"],
                overall["mean_delta_passes"],
                overall["improved"], overall["worsened"], overall["n"],
            )
        )
        print(
            "   mean flap rate %.3f vs %.3f" % (
                overall["mean_action_low"], overall["mean_action_high"],
            )
        )
        for seed, entry in sorted(summary["per_seed"].items()):
            print(
                "   seed %d: mean delta frames %+.1f  passes %+.2f (n=%d)" % (
                    seed, entry["mean_delta_frames"],
                    entry["mean_delta_passes"], entry["n"],
                )
            )
    report = dict(seed=args.seed, horizons=horizons, rows=rows,
                  summary=summary)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return report


if __name__ == "__main__":
    main()
