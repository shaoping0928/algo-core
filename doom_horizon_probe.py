"""When does the future close? Depth-to-inevitability along real deaths.

The planner searches 8 frames ahead and, at almost every state, finds a
plan that survives the whole horizon.  Yet the agent dies.  This probe
walks back along a real death trajectory and asks, at each lag k, whether
the 8-frame search still contains a plan that survives the horizon.

* survivor exists at k  -> the doom was not visible 8 frames before death
  (the death is a compounding/replanning artifact, not a horizon miss);
* no survivor at k      -> the horizon was already closed k frames early.

Nothing is learned and the environment is only replayed, never branched:
every candidate is scored in a copy of the world exactly as the planner
already does at decision time.
"""

import argparse
import json

import numpy as np

from latent_world import latent
from planner_rollout_quality import flatten_batches, oracle_plan
from substrate_nodes import LAG, _curiosity_off, evaluate, fit, policy_result


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
            if frames >= args.cap or frames < 3:
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
                recent = list(reversed(actions[max(0, point - LAG):point]))
                prefix = np.zeros(LAG, dtype=float)
                prefix[:len(recent[:LAG])] = recent[:LAG]
                batches = []
                policy_result(
                    env, world, agent, prefix, args.horizon, lag=LAG,
                    step_stride=2, candidate_callback=batches.append,
                )
                if not batches:
                    continue
                trace = flatten_batches(batches)
                feasible = (
                    np.asarray(trace["feasible"], dtype=bool)
                    & np.asarray(trace["finite"], dtype=bool)
                )
                if not feasible.any():
                    continue
                sequences = np.asarray(trace["sequences"], dtype=float)
                scores = np.asarray(trace["scores"], dtype=float)
                pool = np.flatnonzero(feasible)
                survival = np.empty(len(sequences), dtype=int)
                for index in pool:
                    death, _state, _dead = oracle_plan(env, sequences[index])
                    survival[index] = (
                        args.horizon + 1 if death < 0 else int(death)
                    )
                chosen = pool[np.argmin(scores[pool, 0])]
                anystate = np.asarray(trace["first_states"], dtype=float)
                rows.append(dict(
                    eval_seed=int(eval_seed), frames=int(frames),
                    lag_back=int(lag_back),
                    best_survival=int(survival[pool].max()),
                    mean_survival=float(survival[pool].mean()),
                    chosen_survival=int(survival[chosen]),
                    survivors=int(np.sum(survival[pool] > args.horizon)),
                    candidates=int(len(pool)),
                    state_error=float(np.abs(
                        anystate[chosen] - np.asarray(env.obs(), dtype=float)
                    ).mean()) if len(anystate) else 0.0,
                ))
    return rows


def summarize(rows, horizon):
    out = []
    for lag_back in sorted(set(row["lag_back"] for row in rows)):
        block = [row for row in rows if row["lag_back"] == lag_back]
        out.append(dict(
            lag_back=int(lag_back),
            deaths=int(len(block)),
            share_with_survivor=float(np.mean(
                [row["survivors"] > 0 for row in block]
            )),
            mean_best_survival=float(np.mean(
                [row["best_survival"] for row in block]
            )),
            mean_chosen_survival=float(np.mean(
                [row["chosen_survival"] for row in block]
            )),
            mean_survivors=float(np.mean([row["survivors"] for row in block])),
        ))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    args.eval_seeds = parse_seed_spec(args.eval_seeds)

    rows = run(args)
    summary = summarize(rows, args.horizon)
    print("== depth to inevitability: seed %d, horizon %d ==" % (
        args.seed, args.horizon
    ))
    print("  lag  deaths  survivor-exists  best-survival  chosen-survival")
    for block in summary:
        print("  %3d  %6d  %14.0f%%  %13.2f  %15.2f" % (
            block["lag_back"], block["deaths"],
            100.0 * block["share_with_survivor"],
            block["mean_best_survival"], block["mean_chosen_survival"],
        ))
    closing = [
        block["lag_back"] for block in summary
        if block["share_with_survivor"] < 0.5
    ]
    print("  first lag where most deaths have no surviving plan: %s" % (
        min(closing) if closing else "not within %d frames" % args.max_lag
    ))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(dict(summary=summary, rows=rows), handle,
                      indent=2, sort_keys=True)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
