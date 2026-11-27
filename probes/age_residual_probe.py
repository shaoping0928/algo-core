"""Test the per-target, age-only residual world at the decision level.

The wiring diagnostic showed that a shared residual model with history and an
aggregate MSE gate rejects a real vertical signal.  This probe fits one
age-only residual per vertical target (0/1/3/5), gates each target on held-out
episodes, and then asks whether the correction changes the planner's first
action, first-action gap, and true regret.

The agent remains raw.  Only the world sees the auxiliary age.
"""

import argparse
import copy

import numpy as np

from planner_rollout_quality import (
    action_score_gap, flatten_batches, oracle_plan, plan_score_gap,
)
from substrate_generic import collect
from substrate_nodes import LAG, evaluate, fit, policy_result
from time_in_state import PerTargetAgeWorld, make_env_factory
from time_world_probe import BASE_CONFIG, run_arm
from trajectory_search import best_candidate_index


def parse_seeds(spec):
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [
        int(value) for value in spec.replace(",", " ").split()
        if value.strip()
    ]


def first_gap_and_regret(trace, true_value, horizon):
    sequences = np.asarray(trace["sequences"], dtype=int)
    gap, _ = action_score_gap(trace, sequences)
    index = best_candidate_index(
        trace["scores"], trace["model_feasible"]
    )
    action = int(sequences[index, 0])
    value = int(true_value[index])
    return gap, plan_score_gap(trace), action, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-121")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--tail", type=int, default=2)
    parser.add_argument("--max-age", type=int, default=16)
    parser.add_argument("--min-gain", type=float, default=1e-3)
    args = parser.parse_args()
    eval_seeds = parse_seeds(args.eval_seeds)

    config = dict(BASE_CONFIG)
    config["seed"] = args.seed
    config["steps"] = args.steps
    factory = make_env_factory(
        1, lag=LAG, max_age=args.max_age,
        include_actions=True, include_phase=False, age_actions=(1,),
    )
    world_base, agent = fit(**config)
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
    episodes = np.asarray([row["episode"] for row in augmented])
    world_age = PerTargetAgeWorld.from_base(
        world_base, observations, histories, next_observations,
        episodes, max_age=args.max_age, age_actions=(1,),
        targets=(0, 1, 3, 5), min_gain=args.min_gain,
    )
    world_control = PerTargetAgeWorld.from_base(
        world_base, observations, histories, next_observations,
        episodes, max_age=args.max_age, age_actions=(1,),
        targets=(), min_gain=args.min_gain,
    )
    print("== per-target age residual seed %d eval %s ==" % (
        args.seed, args.eval_seeds,
    ))
    print(
        "target gains: " + " ".join(
            "%d %+.4f%s" % (
                target, world_age.gains[target],
                "*" if target in world_age.corrections else "",
            )
            for target in (0, 1, 3, 5)
        )
    )
    print("enabled targets: %s" % sorted(world_age.corrections))

    arms = {}
    for name, world, env_factory in (
        ("baseline", world_base, None),
        ("age", world_age, factory),
        ("control", world_control, factory),
    ):
        arms[name] = run_arm(
            args.seed, world, agent, eval_seeds, env_factory,
            args.horizon,
        )
    print()
    print("arm        frames  passes  action_rate")
    for name, (frames, passes, rates) in arms.items():
        print(
            "%-10s %6.1f %7d %11.4f" % (
                name, float(np.mean(frames)), int(np.sum(passes)),
                float(np.mean(rates)),
            )
        )

    # Candidate-level comparison on states reached by the baseline policy.
    records = []
    for eval_seed in eval_seeds:
        # Replay the baseline policy once more for its action trace.
        frames, actions, _rows = evaluate(
            args.seed, world_base, agent, cap=5000, horizon=args.horizon,
            lag=LAG, step_stride=2, env_seed=int(eval_seed) * 10000,
        )
        if frames < 2:
            continue
        points = sorted(set(
            list(np.linspace(2, frames - 2, args.branches).astype(int))
            + list(range(max(1, frames - args.tail), frames))
        ))
        for point in points:
            # Replay on the wrapped env so its age is aligned with the state.
            wrapped = factory(
                int(eval_seed) * 10000,
                spacing=None,
            )
            for action in actions[:point]:
                if wrapped.dead:
                    break
                wrapped.step(int(action))
            if wrapped.dead:
                continue
            base_env = wrapped.base
            recent = list(reversed(actions[max(0, point - LAG):point]))
            prefix = np.zeros(LAG, dtype=float)
            prefix[:len(recent[:LAG])] = recent[:LAG]

            traces = {}
            for name, world, env in (
                ("baseline", world_base, base_env),
                ("age", world_age, wrapped),
            ):
                batches = []
                policy_result(
                    env, world, agent, prefix, args.horizon, lag=LAG,
                    step_stride=2, candidate_callback=batches.append,
                )
                if batches:
                    traces[name] = flatten_batches(batches)
            if "baseline" not in traces or "age" not in traces:
                continue
            sequences = np.asarray(traces["baseline"]["sequences"], dtype=int)
            age_sequences = np.asarray(
                traces["age"]["sequences"], dtype=int
            )
            if sequences.shape != age_sequences.shape or not np.array_equal(
                sequences, age_sequences
            ):
                continue
            true_first = np.asarray([
                oracle_plan(base_env, sequence)[0]
                for sequence in sequences
            ], dtype=int)
            true_value = np.where(
                true_first < 0, args.horizon + 1, true_first
            )
            best_value = int(true_value.max())
            row = {"best": best_value}
            for name in ("baseline", "age"):
                gap, plan_gap, action, value = first_gap_and_regret(
                    traces[name], true_value, args.horizon,
                )
                row[name] = dict(
                    gap=gap, plan_gap=plan_gap, action=action,
                    value=value, regret=int(best_value - value),
                    oracle_best=bool(value == best_value),
                )
            records.append(row)

    if records:
        def summarize(name):
            gaps = [
                row[name]["gap"] for row in records
                if row[name]["gap"] is not None
            ]
            regs = [row[name]["regret"] for row in records]
            return dict(
                decisions=len(records),
                gap_median=float(np.median(gaps)) if gaps else float("nan"),
                gap_mean=float(np.mean(gaps)) if gaps else float("nan"),
                oracle_best=int(sum(row[name]["oracle_best"] for row in records)),
                regret_mean=float(np.mean(regs)),
                regret_max=int(max(regs)),
            )
        print()
        print("candidate diagnostics:")
        for name in ("baseline", "age"):
            row = summarize(name)
            print(
                "%-9s decisions %2d  gap median %.5f mean %.5f  "
                "oracle-best %2d  regret mean %.3f max %d" % (
                    name, row["decisions"], row["gap_median"],
                    row["gap_mean"], row["oracle_best"],
                    row["regret_mean"], row["regret_max"],
                )
            )
        changed = int(sum(
            row["baseline"]["action"] != row["age"]["action"]
            for row in records
        ))
        improved = int(sum(
            row["age"]["regret"] < row["baseline"]["regret"]
            for row in records
        ))
        worsened = int(sum(
            row["age"]["regret"] > row["baseline"]["regret"]
            for row in records
        ))
        print(
            "changed actions %d  improved %d  worsened %d" % (
                changed, improved, worsened,
            )
        )


if __name__ == "__main__":
    main()
