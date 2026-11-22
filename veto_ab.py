"""A/B: use the long-window readout as an endpoint outcome constraint.

The precursor-12 readout ranks death at h=12 with AUC ~0.92 in-sample, but
charging its level per step collapsed the policy.  This probe keeps the
readout's *ranking* only: within each decision, the worst fraction of
candidate endpoints is excluded, and the stepwise hazard is carried at a low
weight so the global level cannot dominate.

Paired against the default point-event stack on the same seeds.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def run(seed, world, agent, eval_seeds, horizon, risk_weight, veto):
    with _curiosity_off(agent):
        lengths = []
        for eval_seed in eval_seeds:
            lengths.append(evaluate(
                seed, world, agent, cap=5000, horizon=horizon, lag=LAG,
                step_stride=2, env_seed=int(eval_seed) * 10000,
                episode=int(eval_seed), risk_weight=risk_weight,
                veto_fraction=veto,
            )[0])
    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--step-weight", type=float, default=0.1)
    parser.add_argument("--vetoes", default="0.5,0.75")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)
    vetoes = [float(value) for value in args.vetoes.split(",")]

    rows = {}
    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
        )
        record = {"point": run(
            seed, world, agent, eval_seeds, args.horizon, 1.0, 0.0
        )}
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
            precursor_labels=True, pain_horizon=args.window,
        )
        record["precursor"] = run(
            seed, world, agent, eval_seeds, args.horizon, 1.0, 0.0
        )
        for veto in vetoes:
            record["veto%.0f" % (100 * veto)] = run(
                seed, world, agent, eval_seeds, args.horizon,
                args.step_weight, veto,
            )
        print("== seed %d ==" % seed)
        for name, lengths in record.items():
            print("  %-10s mean %6.1f  runs %s" % (
                name, np.mean(lengths), lengths,
            ))
        rows[seed] = {name: float(np.mean(values))
                      for name, values in record.items()}
    if rows:
        print("== means over seeds ==")
        for name in rows[seeds[0]]:
            values = [rows[seed][name] for seed in rows]
            print("  %-10s %6.1f  per-seed %s" % (
                name, np.mean(values), [round(value, 1) for value in values],
            ))


if __name__ == "__main__":
    main()
