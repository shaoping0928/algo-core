"""A/B the two-level decision: stage-subgoal choice, frame-level execution.

The high level picks, at each stage, the successor whose entry state has the
best realised value (node association), not the successor the stream happened
to take most often.  The frame planner then keeps that subgoal for the stage
and pays a distance term toward it.  `goal` mode with weight 0 is bitwise the
default planner.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--weights", default="0.01,0.03,0.1")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)
    weights = [float(value) for value in args.weights.split(",")]

    means = {}
    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
        )
        record = {}
        with _curiosity_off(agent):
            for eval_seed in eval_seeds:
                kwargs = dict(
                    cap=5000, horizon=args.horizon, lag=LAG, step_stride=2,
                    env_seed=int(eval_seed) * 10000,
                    episode=int(eval_seed),
                )
                record.setdefault("off", []).append(
                    evaluate(seed, world, agent, **kwargs)[0]
                )
                for weight in weights:
                    record.setdefault("w%.2f" % weight, []).append(
                        evaluate(
                            seed, world, agent, milestone_mode="goal",
                            milestone_weight=weight, **kwargs
                        )[0]
                    )
        print("== seed %d: goals %d ==" % (
            seed, len(getattr(agent, "milestone_goals_by_key", {}) or {}),
        ))
        for name, lengths in record.items():
            print("  %-6s mean %6.1f  runs %s" % (
                name, np.mean(lengths), lengths,
            ))
        means[seed] = {name: float(np.mean(values))
                       for name, values in record.items()}
    if means:
        print("== means over seeds ==")
        for name in means[seeds[0]]:
            values = [means[seed][name] for seed in means]
            print("  %-6s %6.1f  per-seed %s" % (
                name, np.mean(values), [round(value, 1) for value in values],
            ))


if __name__ == "__main__":
    main()
