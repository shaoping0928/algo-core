"""Why does the association value hurt?  Compare two policies on one episode.

Same fitted model, same world seed; the only difference is the endpoint value
readout.  Reported: episode length, pipes passed (the world's own counter),
flap rate overall and in windows, action agreement between the two policies,
and the size of the value term the planner actually charges.

Read-only.
"""

import argparse

import numpy as np

from latent_world import latent
from substrate_nodes import (
    LAG, _association_value_table, _curiosity_off, evaluate, fit,
)


def replay(eval_seed, actions):
    env = latent[LAG](int(eval_seed) * 10000)
    for action in actions:
        if env.dead:
            break
        env.step(int(action))
    return env


def value_of_endpoints(agent, plans):
    if not plans:
        return np.zeros(0), np.zeros(0)
    encoded = []
    for plan in plans:
        actions = np.asarray(plan["actions"], dtype=float).reshape(-1)
        take = actions[-agent.lag:][::-1] if agent.lag else np.zeros(0)
        if len(take) < agent.lag:
            take = np.concatenate([np.zeros(agent.lag - len(take)), take])
        encoded.append(np.concatenate([
            np.asarray(plan["final"], dtype=float), take,
        ]) / agent.scales)
    from substrate_nodes import association_value_batch
    return association_value_batch(agent, np.asarray(encoded))


def window_rates(actions, width=30, limit=180):
    actions = np.asarray(actions, dtype=float)
    rates = []
    for start in range(0, min(len(actions), limit), width):
        block = actions[start:start + width]
        rates.append((start, float(block.mean())))
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=120)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--weight", type=float, default=1.0)
    args = parser.parse_args()

    world, agent = fit(
        seed=args.seed, steps=args.steps, epochs=1, max_nodes=1000,
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
    kwargs = dict(
        cap=5000, horizon=args.horizon, lag=LAG, step_stride=2,
        env_seed=args.eval_seed * 10000, episode=args.eval_seed,
        record=True,
    )
    with _curiosity_off(agent):
        agent.value_readout = "cells"
        agent.feature_value_weight = base_weight
        cells_plans = []
        cells_frames, cells_actions, _ = evaluate(
            args.seed, world, agent, plan_out=cells_plans, **kwargs
        )
        agent.value_readout = "association"
        agent.feature_value_weight = args.weight
        assoc_plans = []
        assoc_frames, assoc_actions, _ = evaluate(
            args.seed, world, agent, plan_out=assoc_plans, **kwargs
        )
    for name, frames, actions, plans in (
        ("cells", cells_frames, cells_actions, cells_plans),
        ("assoc", assoc_frames, assoc_actions, assoc_plans),
    ):
        env = replay(args.eval_seed, actions)
        endpoint_values, endpoint_support = value_of_endpoints(agent, plans)
        trust = endpoint_support / (endpoint_support + 150.0)
        term = args.weight * trust * endpoint_values
        print("== %s: frames %d  passed %d  flap %.3f ==" % (
            name, frames, env.passed, float(np.mean(actions)),
        ))
        print("  endpoint value: mean %.3f  p10 %.3f  p90 %.3f" % (
            float(np.mean(endpoint_values)),
            float(np.percentile(endpoint_values, 10)),
            float(np.percentile(endpoint_values, 90)),
        ))
        print("  trust: mean %.3f  p10 %.3f  p90 %.3f" % (
            float(np.mean(trust)), float(np.percentile(trust, 10)),
            float(np.percentile(trust, 90)),
        ))
        print("  charged value term: mean %.3f  p10 %.3f  p90 %.3f" % (
            float(np.mean(term)), float(np.percentile(term, 10)),
            float(np.percentile(term, 90)),
        ))
        print("  flap by window: %s" % " ".join(
            "%d:%.2f" % pair for pair in window_rates(actions)
        ))
    shared = min(len(cells_actions), len(assoc_actions))
    agreement = float(np.mean(
        np.asarray(cells_actions[:shared]) == np.asarray(assoc_actions[:shared])
    ))
    first = next(
        (index for index in range(shared)
         if cells_actions[index] != assoc_actions[index]),
        None,
    )
    print("== agreement %.3f  first divergence frame %s ==" % (
        agreement, first,
    ))


if __name__ == "__main__":
    main()
