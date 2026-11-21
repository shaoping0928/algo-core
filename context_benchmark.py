"""Three-model-seed paired benchmark for the generated prior.

The standard --benchmark keeps the hidden dynamics fixed (only the pipe
geometry varies), so it cannot evaluate a dynamics prior.  This benchmark
uses the same main components as the pipeline - _train builds the node
agent, decision_action runs the full search planner - on the variable-
dynamics world set, and reports paired survival for:

    global      zero context
    online      warmup context, no memory prior
    generated   simulation-selected context prior (with the confidence gate)

Seeds 100-102 are the model seeds; the training worlds and held-out worlds
are fixed, so the comparison is paired per world and episode.
"""
import argparse
import concurrent.futures as futures
import json
import os

import numpy as np

from generated_prior_probe import run


def _worker(seed):
    return seed, run(
        train_steps=1200, warmup=10, max_frames=200, episodes=2,
        horizon=8, model_seed=seed, data_seed=seed,
    )


def _summarise(reports):
    deltas_global = []
    deltas_online = []
    per_seed = {}
    for seed, report in reports:
        seed_global = []
        seed_online = []
        accepted = 0
        fallback = 0
        candidate_credibility = []
        for world, episodes in report["results"].items():
            for episode, arms in episodes.items():
                generated = arms["generated"]["frames"]
                global_frames = arms["global"]["frames"]
                online_frames = arms["online"]["frames"]
                delta_global = generated - global_frames
                delta_online = generated - online_frames
                deltas_global.append(delta_global)
                deltas_online.append(delta_online)
                seed_global.append(delta_global)
                seed_online.append(delta_online)
                prior = arms["generated"]["prior"]
                if prior is not None:
                    if prior.get("accepted"):
                        accepted += 1
                    else:
                        fallback += 1
                    candidate_credibility.append(
                        float(prior.get("credibility", 0.5))
                    )
        per_seed[seed] = dict(
            context_dim=report["context_dim"],
            candidates=report["candidates"],
            mean_global=float(np.mean(seed_global)),
            mean_online=float(np.mean(seed_online)),
            accepted=accepted,
            fallback=fallback,
            mean_candidate_credibility=(
                float(np.mean(candidate_credibility))
                if candidate_credibility else 0.0
            ),
            deltas_global=seed_global,
            deltas_online=seed_online,
        )
    return dict(
        seeds=sorted(per_seed),
        n=len(deltas_global),
        per_seed=per_seed,
        mean_global=float(np.mean(deltas_global)),
        median_global=float(np.median(deltas_global)),
        positive_global=int(sum(d > 0 for d in deltas_global)),
        mean_online=float(np.mean(deltas_online)),
        median_online=float(np.median(deltas_online)),
        positive_online=int(sum(d > 0 for d in deltas_online)),
        ties_online=int(sum(d == 0 for d in deltas_online)),
        accepted=int(sum(item["accepted"] for item in per_seed.values())),
        fallback=int(sum(item["fallback"] for item in per_seed.values())),
        mean_candidate_credibility=float(np.mean([
            item["mean_candidate_credibility"]
            for item in per_seed.values()
        ])),
        deltas_global=deltas_global,
        deltas_online=deltas_online,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100-102")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if "-" in args.seeds:
        start, end = args.seeds.split("-")
        seeds = list(range(int(start), int(end) + 1))
    else:
        seeds = [int(item) for item in args.seeds.split(",")]
    workers = min(max(int(args.jobs), 1), len(seeds), os.cpu_count() or 1)
    reports = []
    if workers <= 1:
        reports = [_worker(seed) for seed in seeds]
    else:
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            reports = list(pool.map(_worker, seeds))
    summary = _summarise(reports)
    payload = dict(summary=summary, reports={
        str(seed): report for seed, report in reports
    })
    print(json.dumps(summary, indent=2, default=str))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
