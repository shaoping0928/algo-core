"""What did the stage learner actually learn?  Content audit, not a sweep.

The penalty sweep is flat from 0.02 to 1.0, so the accepted boundaries are
not a tunable granularity: they are a few very strong discontinuities.
This probe prints what those boundaries are made of:

* the frame range and action counts of every stage;
* the action effect vector of every stage (what the action changes there);
* at every boundary, the scaled observation before/after, the per-dimension
  change, and the similarity gap (cross-boundary distance over the
  within-stage distance of the same states);
* the learned stage transition table.

Read-only: segments one recorded evaluation stream.
"""

import argparse

import numpy as np

from stage_graph import build_stage_graph
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def distance(left, right):
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-122")
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--cap", type=int, default=5000)
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
            _frames, _actions, transitions = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=8,
                lag=LAG, step_stride=2, env_seed=eval_seed * 10000,
                record=True,
            )
            for row in transitions:
                row = dict(row)
                row["episode"] = int(eval_seed)
                rows.append(row)
    graph = build_stage_graph(rows, lag=LAG, penalty=args.penalty)
    print("== learned stage content: %d stages, %d clusters ==" % (
        len(graph.stages), graph.clusters
    ))
    for stage in sorted(graph.stages, key=lambda s: (s.episode, s.start)):
        effect = np.round(np.asarray(stage.action_effect), 2).tolist()
        print("  ep %d  stage [%3d,%3d)  frames %3d  actions %s  "
              "effect %s" % (
                  stage.episode, stage.start, stage.end, stage.frames,
                  stage.action_counts, effect,
              ))

    print("  -- boundaries --")
    print("  ep  at   d_obs(top dim)      within  cross  ratio  "
          "nearest-before  nearest-after")
    for stage in sorted(graph.stages, key=lambda s: (s.episode, s.start)):
        if stage.start == 0:
            continue
        episode_rows = [
            row for row in rows if int(row["episode"]) == stage.episode
        ]
        cut = int(stage.start)
        if cut < 4 or cut + 4 > len(episode_rows):
            continue
        before = np.asarray([
            episode_rows[index]["obs"] for index in range(cut - 4, cut)
        ]) / agent.scales[:agent.obs_dim]
        after = np.asarray([
            episode_rows[index]["obs"] for index in range(cut, cut + 4)
        ]) / agent.scales[:agent.obs_dim]
        mean_before = before.mean(axis=0)
        mean_after = after.mean(axis=0)
        delta = np.abs(mean_after - mean_before)
        top = int(np.argmax(delta))
        within = float(np.mean([
            distance(before[i], before[j])
            for i in range(len(before)) for j in range(i + 1, len(before))
        ]))
        cross = float(np.mean([
            distance(left, right) for left in before for right in after
        ]))
        encoded_before = np.concatenate([
            np.asarray(episode_rows[cut - 1]["obs"], dtype=float),
            np.asarray(episode_rows[cut - 1]["history"], dtype=float)[1:1 + LAG][::-1],
        ]) / agent.scales
        encoded_after = np.concatenate([
            np.asarray(episode_rows[cut]["obs"], dtype=float),
            np.asarray(episode_rows[cut]["history"], dtype=float)[1:1 + LAG][::-1],
        ]) / agent.scales
        _id_before, distance_before = agent._nearest(encoded_before)
        _id_after, distance_after = agent._nearest(encoded_after)
        print("  %3d %4d   dim %d d=%.2f        %6.2f %6.2f %6.2f  "
              "%6.3f/%s  %6.3f/%s" % (
                  stage.episode, cut, top, float(delta[top]),
                  within, cross, cross / within if within > 1e-9 else 0.0,
                  distance_before, _id_before,
                  distance_after, _id_after,
              ))

    print("  -- transitions --")
    for transition in sorted(
        graph.transitions,
        key=lambda t: (-t.frames, str(t.source), t.action),
    )[:20]:
        print("    %s -a%d-> %s  frames %d  episodes %d" % (
            transition.source, transition.action, transition.target,
            transition.frames, transition.episodes,
        ))


if __name__ == "__main__":
    main()
