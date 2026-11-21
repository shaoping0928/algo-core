"""Is the gap the horizon or the choice? Oracle selection at a fixed horizon.

The death probe showed a 12-frame search can find an escape, yet the
hazard-ranked planner still picks a plan that dies at the same frame as
reality.  That separates two failures:

* the horizon is too short (no escape exists in the candidate set), or
* the choice is wrong (an escape exists, the ranking misses it).

This probe keeps the candidate set and the horizon exactly as the planner
has them and replaces only the selection rule with the true survival of
each candidate (replayed in a copy of the world).  It then flies the
resulting receding-horizon controller for one episode and reports the
frames survived, next to the real planner on the same world.

The oracle never trains anything; it is a measurement of how much of the
gap lives in the ranking.
"""

import argparse
import json

import numpy as np

from latent_world import latent
from planner_rollout_quality import flatten_batches, oracle_plan
from substrate_nodes import LAG, _curiosity_off, evaluate, fit, policy_result


def fly(args, world, agent, env_seed, env):
    prefix = np.zeros(LAG, dtype=float)
    recent = [0.0] * LAG
    total = 0
    while not env.dead and env.t < args.cap:
        batches = []
        policy_result(
            env, world, agent, prefix, args.horizon, lag=LAG,
            step_stride=2, candidate_callback=batches.append,
        )
        if not batches:
            break
        trace = flatten_batches(batches)
        feasible = (
            np.asarray(trace["feasible"], dtype=bool)
            & np.asarray(trace["finite"], dtype=bool)
        )
        if not feasible.any():
            break
        sequences = np.asarray(trace["sequences"], dtype=float)
        scores = np.asarray(trace["scores"], dtype=float)
        pool = np.flatnonzero(feasible)
        survival = np.empty(len(sequences), dtype=int)
        for index in pool:
            death, _state, _dead = oracle_plan(env, sequences[index])
            survival[index] = args.horizon + 1 if death < 0 else int(death)
        # Best true survival, then the planner's own cost as the tie-break.
        best = survival[pool].max()
        tied = pool[survival[pool] == best]
        chosen = tied[np.argmin(scores[tied, 0])]
        action = int(sequences[chosen, 0])
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        recent = [float(action)] + recent[:-1]
        env.step(action)
        total += 1
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-122")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    args.eval_seeds = parse_seed_spec(args.eval_seeds)

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
            policy_frames, _actions, _rows = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=args.horizon,
                lag=LAG, step_stride=2, env_seed=env_seed,
            )
            oracle_frames = fly(
                args, world, agent, env_seed, latent[LAG](env_seed)
            )
            rows.append(dict(
                eval_seed=int(eval_seed),
                policy_frames=int(policy_frames),
                oracle_frames=int(oracle_frames),
            ))
            print("  seed %d: policy %d  oracle %d" % (
                eval_seed, policy_frames, oracle_frames
            ))
    policy = [row["policy_frames"] for row in rows]
    oracle = [row["oracle_frames"] for row in rows]
    print("== oracle selection at horizon %d, seed %d ==" % (
        args.horizon, args.seed
    ))
    print("  policy mean %.1f  oracle mean %.1f  paired %+.1f" % (
        float(np.mean(policy)), float(np.mean(oracle)),
        float(np.mean(np.asarray(oracle) - np.asarray(policy))),
    ))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(dict(rows=rows), handle, indent=2, sort_keys=True)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
