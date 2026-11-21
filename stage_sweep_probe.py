"""Find the segmentation granularity that yields decision-scale events.

The residual segmenter's split threshold (`penalty`) sets how coarse a
stage is.  At penalty 1.0 an episode is one or two stages of about ninety
frames, which is far above the ten-to-twenty-frame window a coarse
decision needs.  This probe sweeps the penalty, and at every setting
reports:

* stages per episode and the dwell-time median;
* the share of stages whose dwell time falls in the decision window;
* cross-episode informativeness on the real stream and on the
  action-shuffled control (same penalty, so the comparison is matched);
* the lead between the final stage boundary and the death.

Read-only and offline: it segments one recorded evaluation stream.
"""

import argparse
import json

import numpy as np

from stage_graph import build_stage_graph, shuffle_actions
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def last_leads(graph, counts):
    by_episode = {}
    for stage in graph.stages:
        by_episode.setdefault(stage.episode, []).append(stage)
    leads = []
    for episode, stages in by_episode.items():
        final = max(stages, key=lambda stage: stage.start)
        leads.append(int(counts[episode]) - 1 - int(final.start))
    return leads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-122")
    parser.add_argument("--penalties", default="0.02,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--window", default="8-24")
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    eval_seeds = parse_seed_spec(args.eval_seeds)
    penalties = [float(value) for value in args.penalties.split(",")]
    low, high = [int(value) for value in args.window.split("-")]

    world, agent = fit(
        seed=args.seed, steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=8, step_stride=2,
    )
    rows = []
    with _curiosity_off(agent):
        for eval_seed in eval_seeds:
            _frames, _actions, transitions = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=8,
                lag=LAG, step_stride=2, env_seed=eval_seed * 10000,
                record=True,
            )
            for row in transitions:
                row = dict(row)
                row["episode"] = int(eval_seed)
                rows.append(row)
    counts = {}
    for row in rows:
        counts[int(row["episode"])] = counts.get(int(row["episode"]), 0) + 1
    shuffled = shuffle_actions(rows, lag=LAG, seed=12345)

    results = []
    print("== stage granularity sweep: seed %d, %d frames ==" % (
        args.seed, len(rows)
    ))
    print("  penalty  stages  clusters  dwell-p50  in-window  "
          "cross3-real  cross3-shuf  last-lead")
    for penalty in penalties:
        real = build_stage_graph(rows, lag=LAG, penalty=penalty)
        control = build_stage_graph(shuffled, lag=LAG, penalty=penalty)
        dwell = [stage.frames for stage in real.stages]
        leads = last_leads(real, counts)
        entry = dict(
            penalty=penalty,
            stages=len(real.stages),
            clusters=int(real.clusters),
            dwell_p50=float(np.median(dwell)) if dwell else 0.0,
            in_window=float(np.mean([
                low <= frames <= high for frames in dwell
            ])) if dwell else 0.0,
            cross3_real=int(real.cross_episode_informative()),
            cross3_shuffle=int(control.cross_episode_informative()),
            last_lead_median=float(np.median(leads)) if leads else -1.0,
            last_lead_min=int(np.min(leads)) if leads else -1,
        )
        results.append(entry)
        print("  %7.2f  %6d  %8d  %9.1f  %9.0f%%  %11d  %12d  %9.0f" % (
            penalty, entry["stages"], entry["clusters"],
            entry["dwell_p50"], 100.0 * entry["in_window"],
            entry["cross3_real"], entry["cross3_shuffle"],
            entry["last_lead_median"],
        ))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
