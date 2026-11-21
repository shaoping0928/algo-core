"""Is the last stage boundary early enough to be a coarse decision point?

The coarse unit is an *event*: the stage segmenter splits where one
action-conditioned local model stops explaining the next observations.
If the final stage starts only a couple of frames before death, the event
carries no lead and a coarse layer has nothing to decide at; if it starts
ten or more frames out, the stage entry is the natural coarse decision
point and the fine layer only has to execute inside the stage.

Read-only: segments the recorded evaluation stream, nothing is trained.
"""

import argparse

import numpy as np

from stage_graph import build_stage_graph
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-122")
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--penalty", type=float, default=1.0)
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    eval_seeds = parse_seed_spec(args.eval_seeds)

    world, agent = fit(
        seed=args.seed, steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=8, step_stride=2,
    )
    rows = []
    with _curiosity_off(agent):
        for eval_seed in eval_seeds:
            frames, _actions, transitions = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=8,
                lag=LAG, step_stride=2, env_seed=eval_seed * 10000,
                record=True,
            )
            for row in transitions:
                row = dict(row)
                row["episode"] = int(eval_seed)
                rows.append(row)
    graph = build_stage_graph(rows, lag=LAG, penalty=args.penalty)
    summary = graph.summary()
    print("== events: seed %d, %d episodes, %d frames ==" % (
        args.seed, summary["episodes"], summary["frames"]
    ))
    print("  stages %d, clusters %d, stage length p50 %.1f, gate R2 %.3f" % (
        summary["stages"], summary["clusters"],
        summary["stage_length_p50"], summary["gate_r2"],
    ))
    by_episode = {}
    for stage in graph.stages:
        by_episode.setdefault(stage.episode, []).append(stage)
    counts = {}
    for row in rows:
        counts[int(row["episode"])] = counts.get(int(row["episode"]), 0) + 1
    print("  episode  frames  stages  final-stage-start  lead-to-death")
    leads = []
    for episode in sorted(by_episode):
        stages = sorted(by_episode[episode], key=lambda stage: stage.start)
        final = stages[-1]
        frames = counts[episode]
        lead = int(frames) - 1 - int(final.start)
        leads.append(lead)
        print("  %7d  %6d  %6d  %17d  %14d" % (
            episode, frames, len(stages), final.start, lead
        ))
    if leads:
        print("  lead: median %.1f  mean %.1f  min %d  max %d" % (
            float(np.median(leads)), float(np.mean(leads)),
            int(np.min(leads)), int(np.max(leads)),
        ))


if __name__ == "__main__":
    main()
