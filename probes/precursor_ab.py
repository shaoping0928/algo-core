"""Paired A/B: point-event labels (default) vs precursor labels with a window.

Same protocol, same seeds, two fits per seed: the only difference is whether
the hazard readout is trained on the entered state's own pain (the default) or
on "pain within `window` frames".  The long-window readout is the one whose
lead curve reaches h=12 with AUC ~0.92.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def run(seed, world, agent, eval_seeds, horizon):
    with _curiosity_off(agent):
        lengths = []
        for eval_seed in eval_seeds:
            lengths.append(evaluate(
                seed, world, agent, cap=5000, horizon=horizon, lag=LAG,
                step_stride=2, env_seed=int(eval_seed) * 10000,
                episode=int(eval_seed),
            )[0])
    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=12)
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)

    rows = {}
    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
        )
        base = run(seed, world, agent, eval_seeds, args.horizon)
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
            precursor_labels=True, pain_horizon=args.window,
        )
        precursor = run(seed, world, agent, eval_seeds, args.horizon)
        rows[seed] = (base, precursor)
        print("== seed %d ==" % seed)
        print("  point      mean %6.1f  runs %s" % (
            np.mean(base), base,
        ))
        print("  precursor  mean %6.1f  runs %s" % (
            np.mean(precursor), precursor,
        ))
    if rows:
        base = np.mean([np.mean(value[0]) for value in rows.values()])
        precursor = np.mean([np.mean(value[1]) for value in rows.values()])
        wins = sum(
            1 for value in rows.values()
            if np.mean(value[1]) > np.mean(value[0])
        )
        print("== paired means: point %.1f  precursor %.1f  "
              "precursor better on %d/%d ==" % (
                  base, precursor, wins, len(rows),
              ))


if __name__ == "__main__":
    main()
