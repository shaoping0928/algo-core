"""A/B the P0 time-in-state features on the real evaluation protocol.

Both arms use the same fit configuration and the same held-out worlds; the
only difference is whether the environment appends action ages and a phase
clock to the observation before training and evaluation.  This is an opt-in
probe: no default path changes.

The period is estimated from a raw pilot stream by the same per-episode
spectrum used by the phase-stage probe; it is never named.
"""

import argparse

import numpy as np

from substrate_generic import collect
from substrate_nodes import LAG, evaluate, fit
from time_in_state import estimate_period, make_env_factory


BASE_CONFIG = dict(
    steps=1500, epochs=1, max_nodes=1000, birth_threshold=0.45,
    dagger_rounds=1, dagger_frames=5000, dagger_cap=5000,
    dagger_horizon=8, step_stride=2, curiosity_weight=0.5,
)


def run_arm(seed, world, agent, eval_seeds, env_factory, horizon):
    frames, passes, rates = [], [], []
    for eval_seed in eval_seeds:
        metrics = {}
        length, actions, _rows = evaluate(
            seed, world, agent, cap=5000, horizon=horizon, lag=LAG,
            step_stride=2, env_seed=int(eval_seed) * 10000,
            metrics_out=metrics, env_factory=env_factory,
        )
        frames.append(int(length))
        passes.append(int(metrics.get("passes", 0)))
        rates.append(float(np.mean(actions)) if len(actions) else 0.0)
    return frames, passes, rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--features", choices=("action", "phase", "both", "control"),
        default="action",
    )
    parser.add_argument("--max-age", type=int, default=64)
    parser.add_argument(
        "--age-action", type=int, default=-1,
        help="only age this action value; -1 ages every action",
    )
    parser.add_argument("--period", type=int, default=0,
                        help="0 estimates the period from a pilot stream")
    args = parser.parse_args()
    if "-" in args.eval_seeds and "," not in args.eval_seeds:
        start, end = args.eval_seeds.split("-")
        eval_seeds = list(range(int(start), int(end) + 1))
    else:
        eval_seeds = [
            int(value) for value in args.eval_seeds.replace(",", " ").split()
            if value.strip()
        ]

    config = dict(BASE_CONFIG)
    config["seed"] = args.seed
    config["steps"] = args.steps
    if args.period > 0:
        period = int(args.period)
    else:
        rows, _ = collect(seed=args.seed, steps=args.steps, lag=LAG)
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        episodes = np.asarray([row["episode"] for row in rows])
        period = estimate_period(observations, episodes)
    print("== seed %d steps %d period %d features %s max_age %d "
          "age_action %d eval %s ==" % (
              args.seed, args.steps, period, args.features, args.max_age,
              args.age_action, args.eval_seeds,
          ))

    world_base, agent_base = fit(**config)
    base_frames, base_passes, base_rates = run_arm(
        args.seed, world_base, agent_base, eval_seeds, None, args.horizon,
    )
    if args.features == "control":
        factory = make_env_factory(
            period, lag=LAG, max_age=args.max_age,
            include_actions=True, include_phase=False,
            age_actions=(99,),
        )
    else:
        factory = make_env_factory(
            period, lag=LAG, max_age=args.max_age,
            include_actions=args.features in ("action", "both"),
            include_phase=args.features in ("phase", "both"),
            age_actions=(
                (args.age_action,) if args.age_action >= 0 else None
            ),
        )
    world_time, agent_time = fit(**config, env_factory=factory)
    time_frames, time_passes, time_rates = run_arm(
        args.seed, world_time, agent_time, eval_seeds, factory,
        args.horizon,
    )

    print()
    print("arm       frames (mean/median)  passes  action_rate  per-world")
    for name, frames, passes, rates in (
        ("baseline", base_frames, base_passes, base_rates),
        ("time", time_frames, time_passes, time_rates),
    ):
        print(
            "%-9s %8.1f / %6.1f %8d %11.4f  %s" % (
                name, float(np.mean(frames)), float(np.median(frames)),
                int(np.sum(passes)), float(np.mean(rates)),
                frames,
            )
        )
    print()
    print("paired frame delta: %s  mean %+.1f  median %+.1f" % (
        [int(t - b) for b, t in zip(base_frames, time_frames)],
        float(np.mean(np.asarray(time_frames) - np.asarray(base_frames))),
        float(np.median(np.asarray(time_frames) - np.asarray(base_frames))),
    ))
    print("paired pass delta:  %s  total %+d" % (
        [int(t - b) for b, t in zip(base_passes, time_passes)],
        int(np.sum(time_passes) - np.sum(base_passes)),
    ))


if __name__ == "__main__":
    main()
