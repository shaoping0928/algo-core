"""How many frames before death does *any* action sequence start to help?

The earlier 12-frame result was measured on a horizon-12 policy, whose
states are less committal than the states the shipped horizon-8 policy
actually enters.  For the shipped policy, a 12-frame macro/random sample
finds no escape at all, so the question has to be re-asked without any
search assumption:

at ``k`` frames before the real death, replay the state and fly ``n``
uniform random action sequences of length ``span``.  ``survival`` is the
first death step; ``k - 1`` is what the shipped policy itself gets from
that state.  The share of random plans that beat it is the recoverability
of the state at that lead -- zero means the death is already sealed no
matter what the agent does, whatever layer is searching.

No macro dictionary, no planner, no task constants: only real transitions
and uniform action sequences.
"""

import argparse
import json

import numpy as np

from latent_world import latent
from planner_rollout_quality import oracle_plan
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def run(args):
    world, agent = fit(
        seed=args.seed, steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
    )
    rows = []
    with _curiosity_off(agent):
        for eval_seed in args.eval_seeds:
            env_seed = int(eval_seed) * 10000
            frames, actions, _ = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=args.horizon,
                lag=LAG, step_stride=2, env_seed=env_seed,
            )
            if frames >= args.cap or frames < args.max_lag + 2:
                continue
            for lag_back in range(1, args.max_lag + 1):
                point = frames - lag_back
                if point < 1:
                    continue
                env = latent[LAG](env_seed)
                for action in actions[:point]:
                    if env.dead:
                        break
                    env.step(int(action))
                if env.dead:
                    continue
                rng = np.random.RandomState(args.seed * 7919 + env_seed + lag_back)
                plans = (
                    rng.rand(args.plans, args.span) < args.action_rate
                ).astype(float)
                survival = np.empty(len(plans), dtype=int)
                for index, plan in enumerate(plans):
                    death, _state, _dead = oracle_plan(env, plan)
                    survival[index] = args.span if death < 0 else int(death)
                actual = lag_back - 1
                beat = survival > actual
                rows.append(dict(
                    eval_seed=int(eval_seed), lag_back=int(lag_back),
                    actual_survival=int(actual),
                    share_beat=float(np.mean(beat)),
                    best_survival=int(survival.max()),
                    mean_survival=float(survival.mean()),
                    mean_beat_survival=float(survival[beat].mean())
                    if beat.any() else float("nan"),
                ))
    return rows


def summarize(rows):
    out = []
    for lag_back in sorted(set(row["lag_back"] for row in rows)):
        block = [row for row in rows if row["lag_back"] == lag_back]
        out.append(dict(
            lag_back=int(lag_back),
            states=int(len(block)),
            share_beat=float(np.mean([row["share_beat"] for row in block])),
            mean_best=float(np.mean([row["best_survival"] for row in block])),
            mean_actual=float(np.mean([row["actual_survival"] for row in block])),
            mean_gain=float(np.mean([
                row["best_survival"] - row["actual_survival"] for row in block
            ])),
        ))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-122")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--plans", type=int, default=256)
    parser.add_argument("--span", type=int, default=20)
    parser.add_argument("--action-rate", type=float, default=0.3)
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    args.eval_seeds = parse_seed_spec(args.eval_seeds)

    summary = summarize(run(args))
    print("== causal horizon for the shipped policy: seed %d, "
          "%d random plans x %d frames ==" % (
              args.seed, args.plans, args.span
          ))
    print("  k   states  random-beats-policy  best  policy  best-gain")
    for block in summary:
        print("  %2d  %6d  %19.0f%%  %4.1f  %6.1f  %9.2f" % (
            block["lag_back"], block["states"],
            100.0 * block["share_beat"], block["mean_best"],
            block["mean_actual"], block["mean_gain"],
        ))
    first = [
        block["lag_back"] for block in summary if block["share_beat"] >= 0.1
    ]
    print("  first lead where >=10%% of random plans beat the policy: %s" % (
        min(first) if first else "not within %d frames" % args.max_lag
    ))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
