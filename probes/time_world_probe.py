"""World-model-only time-in-state A/B.

The agent keeps its raw observation and prototypes; only the world model sees
the extra time features.  This avoids the dimension/dilution confound of the
full-state append:

* baseline: raw world + raw agent + raw env;
* time-world: world fit on augmented observations + raw agent + augmented env.

The planner passes the full state to ``world.predict`` and the raw prefix to
the agent.  The period is estimated from a raw pilot stream; the phase is
optional and currently not identifiable on the short behaviour stream.
"""

import argparse

import numpy as np

from substrate_generic import LinearObservationModel, collect
from substrate_nodes import LAG, evaluate, fit
from time_in_state import (
    ConditionedTimeWorld, estimate_period, make_control_env_factory,
    make_env_factory,
)


BASE_CONFIG = dict(
    steps=1500, epochs=1, max_nodes=1000, birth_threshold=0.45,
    dagger_rounds=1, dagger_frames=5000, dagger_cap=5000,
    dagger_horizon=8, step_stride=2, curiosity_weight=0.5,
)


class StandardizedLinearModel:
    """Per-column standardisation around the shipped linear world model."""

    def __init__(self, obs_dim: int, action_dim: int):
        self.base = LinearObservationModel(obs_dim, action_dim)
        self.obs_mean = np.zeros(obs_dim)
        self.obs_scale = np.ones(obs_dim)
        self.target_mean = np.zeros(obs_dim)
        self.target_scale = np.ones(obs_dim)

    @property
    def obs_dim(self) -> int:
        return self.base.obs_dim

    def fit(self, observations, histories, next_observations):
        observations = np.asarray(observations, dtype=float)
        next_observations = np.asarray(next_observations, dtype=float)
        self.obs_mean = observations.mean(axis=0)
        self.obs_scale = observations.std(axis=0)
        self.obs_scale = np.where(self.obs_scale > 1e-8, self.obs_scale, 1.0)
        self.target_mean = next_observations.mean(axis=0)
        self.target_scale = next_observations.std(axis=0)
        self.target_scale = np.where(
            self.target_scale > 1e-8, self.target_scale, 1.0
        )
        self.base.fit(
            (observations - self.obs_mean) / self.obs_scale,
            histories,
            (next_observations - self.target_mean) / self.target_scale,
        )
        return self

    def predict(self, observations, histories):
        normalized = (
            np.asarray(observations, dtype=float) - self.obs_mean
        ) / self.obs_scale
        return (
            self.base.predict(normalized, histories) * self.target_scale
            + self.target_mean
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


def parse_seeds(spec):
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [
        int(value) for value in spec.replace(",", " ").split()
        if value.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-121")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--features", choices=("action", "phase", "both", "control"),
        default="action",
    )
    parser.add_argument(
        "--mode", choices=("predict", "condition"), default="condition",
    )
    parser.add_argument("--max-age", type=int, default=16)
    parser.add_argument("--age-action", type=int, default=1)
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument(
        "--standardize", action="store_true",
        help="use a per-column standardized world for both arms",
    )
    args = parser.parse_args()
    eval_seeds = parse_seeds(args.eval_seeds)

    config = dict(BASE_CONFIG)
    config["seed"] = args.seed
    config["steps"] = args.steps
    rows, _ = collect(seed=args.seed, steps=args.steps, lag=LAG)
    if args.period > 0:
        period = int(args.period)
    else:
        period = estimate_period(
            np.asarray([row["obs"] for row in rows], dtype=float),
            np.asarray([row["episode"] for row in rows]),
        )
    if args.mode == "condition":
        if args.features in ("phase", "both"):
            raise SystemExit(
                "condition mode supports action/control features only"
            )
        if args.features == "control":
            age_actions = (99,)
        elif args.age_action >= 0:
            age_actions = (args.age_action,)
        else:
            age_actions = (0, 1)
        factory = make_env_factory(
            period, lag=LAG, max_age=args.max_age,
            include_actions=True, include_phase=False,
            age_actions=age_actions,
        )
    elif args.features == "control":
        age_actions = (99,)
        factory = make_control_env_factory(lag=LAG)
    else:
        age_actions = (
            (args.age_action,) if args.age_action >= 0 else (0, 1)
        )
        factory = make_env_factory(
            period, lag=LAG, max_age=args.max_age,
            include_actions=args.features in ("action", "both"),
            include_phase=args.features in ("phase", "both"),
            age_actions=age_actions,
        )
    print("== seed %d steps %d period %d features %s mode %s "
          "standardize %s eval %s ==" % (
              args.seed, args.steps, period, args.features, args.mode,
              args.standardize, args.eval_seeds,
          ))

    world_base_fit, agent = fit(**config)
    if args.standardize:
        world_base = StandardizedLinearModel(
            len(rows[0]["obs"]), len(rows[0]["history"])
        ).fit(
            [row["obs"] for row in rows],
            [row["history"] for row in rows],
            [row["next_obs"] for row in rows],
        )
    else:
        world_base = world_base_fit
    base_frames, base_passes, base_rates = run_arm(
        args.seed, world_base, agent, eval_seeds, None, args.horizon,
    )

    augmented, _ = collect(
        seed=args.seed, steps=args.steps, lag=LAG, env_factory=factory,
    )
    observations = np.asarray(
        [row["obs"] for row in augmented], dtype=float
    )
    histories = np.asarray(
        [row["history"] for row in augmented], dtype=float
    )
    next_observations = np.asarray(
        [row["next_obs"] for row in augmented], dtype=float
    )
    if args.mode == "condition":
        world_time = ConditionedTimeWorld.from_base(
            world_base, observations, histories, next_observations,
            max_age=args.max_age, age_actions=age_actions,
            residual_scale=0.0 if args.features == "control" else 1.0,
            episodes=[row["episode"] for row in augmented],
        )
        print(
            "conditioned: validation gain %+.6f  residual_scale %.3f" % (
                world_time.validation_gain, world_time.residual_scale,
            )
        )
    else:
        model_class = (
            StandardizedLinearModel if args.standardize
            else LinearObservationModel
        )
        world_time = model_class(
            observations.shape[1], histories.shape[1]
        ).fit(observations, histories, next_observations)
    time_frames, time_passes, time_rates = run_arm(
        args.seed, world_time, agent, eval_seeds, factory, args.horizon,
    )

    print()
    print("arm          frames (mean/median)  passes  action_rate  per-world")
    for name, frames, passes, rates in (
        ("baseline", base_frames, base_passes, base_rates),
        ("time-world", time_frames, time_passes, time_rates),
    ):
        print(
            "%-12s %8.1f / %6.1f %8d %11.4f  %s" % (
                name, float(np.mean(frames)), float(np.median(frames)),
                int(np.sum(passes)), float(np.mean(rates)), frames,
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
