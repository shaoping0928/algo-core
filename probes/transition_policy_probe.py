"""Transition-level policy vs the frame-level planner, same protocol.

The transition policy chooses an edge (current node -> observed successor) by
the successor's realised value net of its immediate pain rate, then executes
the action that edge was taken with.  There is no search and no rollout: the
decision unit is the transition, as in the intended architecture.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from latent_world import latent
from substrate_nodes import (
    LAG, _curiosity_off, evaluate, fit, transition_action,
)


def run_transition_policy(agent, env_seed, cap=5000, lag=LAG,
                          risk_weight=1.0):
    env = latent[lag](int(env_seed))
    prefix = np.zeros(lag, dtype=float)
    frames = 0
    while not env.dead and env.t < cap:
        action = transition_action(
            agent, env.obs(), prefix, risk_weight=risk_weight
        )
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(int(action))
        frames += 1
    return frames, int(getattr(env, "passed", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--mode", choices=("dp", "heuristic"), default="dp")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)

    means = {}
    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
        )
        if args.mode == "heuristic":
            agent.transition_policy = {}
        edges = getattr(agent, "transition_edges", {}) or {}
        with _curiosity_off(agent):
            planner, transition = [], []
            for eval_seed in eval_seeds:
                env_seed = int(eval_seed) * 10000
                planner.append(evaluate(
                    seed, world, agent, cap=5000, horizon=args.horizon,
                    lag=LAG, step_stride=2, env_seed=env_seed,
                    episode=int(eval_seed),
                )[0])
                transition.append(run_transition_policy(
                    agent, env_seed, risk_weight=args.risk_weight
                )[0])
        print("== seed %d: nodes with edges %d ==" % (seed, len(edges)))
        print("  planner    mean %6.1f  runs %s" % (
            np.mean(planner), planner,
        ))
        print("  transition mean %6.1f  runs %s" % (
            np.mean(transition), transition,
        ))
        means[seed] = (float(np.mean(planner)), float(np.mean(transition)))
    if means:
        planner = np.mean([value[0] for value in means.values()])
        transition = np.mean([value[1] for value in means.values()])
        print("== means: planner %.1f  transition %.1f ==" % (
            planner, transition,
        ))


if __name__ == "__main__":
    main()
