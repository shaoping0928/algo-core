"""A/B the margin gate: does keeping the previous action on near-ties help?

The gate only fires when the best plan's lead over the runner-up is below
`eps`; it then prefers a plan whose first action continues the current one.
Everything else (model, value readout, weights) is held fixed.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from substrate_nodes import (
    LAG, _association_value_table, _curiosity_off, evaluate, fit,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--eps", default="0.001,0.01")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)
    eps_values = [float(value) for value in args.eps.split(",")]

    means = {}
    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
        )
        values, support = _association_value_table(
            agent.milestone_training_rows, agent
        )
        agent.association_values = values
        agent.association_support = support
        base_weight = float(agent.feature_value_weight)
        record = {}
        with _curiosity_off(agent):
            for eval_seed in eval_seeds:
                kwargs = dict(
                    cap=5000, horizon=args.horizon, lag=LAG, step_stride=2,
                    env_seed=int(eval_seed) * 10000,
                    episode=int(eval_seed),
                )
                agent.value_readout = "cells"
                agent.feature_value_weight = base_weight
                record.setdefault("cells", []).append(
                    evaluate(seed, world, agent, **kwargs)[0]
                )
                agent.value_readout = "association"
                agent.feature_value_weight = args.value_weight
                record.setdefault("assoc", []).append(
                    evaluate(seed, world, agent, **kwargs)[0]
                )
                for eps in eps_values:
                    record.setdefault("gate%.0e" % eps, []).append(
                        evaluate(seed, world, agent, margin_eps=eps,
                                 **kwargs)[0]
                    )
        print("== seed %d ==" % seed)
        for name, lengths in record.items():
            print("  %-8s mean %6.1f  runs %s" % (
                name, np.mean(lengths), lengths,
            ))
        means[seed] = {name: float(np.mean(lengths))
                       for name, lengths in record.items()}
    if means:
        print("== means over seeds ==")
        for name in means[seeds[0]]:
            values = [means[seed][name] for seed in means]
            print("  %-8s %6.1f  per-seed %s" % (
                name, np.mean(values), [round(value, 1) for value in values],
            ))


if __name__ == "__main__":
    main()
