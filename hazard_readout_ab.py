"""A/B the hazard readout: calibrated rate vs raw weighted sum.

The composition weights are fitted; the isotonic map is a monotone
calibration of that score to empirical rates.  On a rare-event target the map
is nearly flat over the range decisions live in, so distinct candidates get
the same hazard - a ranking loss that a monotone map cannot otherwise cause.
This probe keeps the model fixed and only changes what the planner reads.
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
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)

    rows = {}
    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon,
            step_stride=2,
        )
        with _curiosity_off(agent):
            lengths = {}
            for eval_seed in eval_seeds:
                kwargs = dict(
                    cap=5000, horizon=args.horizon, lag=LAG, step_stride=2,
                    env_seed=int(eval_seed) * 10000,
                )
                agent.hazard_readout = "calibrated"
                calibrated = evaluate(seed, world, agent, **kwargs)[0]
                agent.hazard_readout = "raw"
                raw = evaluate(seed, world, agent, **kwargs)[0]
                agent.hazard_readout = "calibrated"
                lengths[eval_seed] = (calibrated, raw)
            rows[seed] = lengths
            print("== seed %d ==" % seed)
            for eval_seed, (calibrated, raw) in lengths.items():
                print("  eval %3d: calibrated %4d  raw %4d" % (
                    eval_seed, calibrated, raw,
                ))
            calibrated_mean = np.mean(
                [value[0] for value in lengths.values()]
            )
            raw_mean = np.mean([value[1] for value in lengths.values()])
            print("  mean: calibrated %.1f  raw %.1f" % (
                calibrated_mean, raw_mean,
            ))
    if rows:
        calibrated = np.mean([
            np.mean([value[0] for value in lengths.values()])
            for lengths in rows.values()
        ])
        raw = np.mean([
            np.mean([value[1] for value in lengths.values()])
            for lengths in rows.values()
        ])
        print("overall: calibrated %.1f  raw %.1f" % (calibrated, raw))


if __name__ == "__main__":
    main()
