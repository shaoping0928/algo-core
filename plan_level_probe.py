"""Plan-level validation: at a decision, does the value term prefer the plan
that actually survives longer?

For sampled decision frames the probe takes the best continuation for each
first action (flap / no flap), replays both from the same environment state,
and compares the realised survival.  It then asks whether the cost with the
value term and the cost without it pick the better first action.

The replays are diagnostic only: nothing is learned from them, and the
training stream is untouched.
"""

import argparse
import copy

import numpy as np

from latent_world import latent
from substrate_nodes import (
    LAG, _association_value_table, _curiosity_off, evaluate, fit,
)


def best_by_first(batches, cost):
    sequences = np.vstack([np.asarray(batch["sequences"], dtype=float)
                           for batch in batches])
    first = sequences[:, 0]
    best = {}
    for action in (0.0, 1.0):
        mask = first == action
        if mask.any():
            index = int(np.flatnonzero(mask)[np.argmin(cost[mask])])
            best[action] = sequences[index]
    return best


def replay_to(eval_seed, actions, frame):
    env = latent[LAG](eval_seed)
    for action in actions[:frame]:
        if env.dead:
            break
        env.step(int(action))
    return env


def rollout(env, sequence, continue_action=0, cap=400):
    """Execute the plan, then apply one fixed continuation until death.

    The continuation is the same for both plans, so the comparison isolates
    what the plan's own eight frames did to the state.  It is a proxy, not a
    policy: no plan is rewarded for a clever continuation it did not choose.
    """
    state = copy.deepcopy(env)
    for action in sequence:
        if state.dead or state.t >= cap:
            break
        state.step(int(action))
    while not state.dead and state.t < cap:
        state.step(int(continue_action))
    return float(state.t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=120)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=40)
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
    agent.value_readout = "association"
    agent.feature_value_weight = 1.0
    with _curiosity_off(agent):
        batches_per_frame = []
        _frames, actions, _ = evaluate(
            args.seed, world, agent,
            cap=5000, horizon=args.horizon, lag=LAG, step_stride=2,
            env_seed=args.eval_seed * 10000, episode=args.eval_seed,
            record=True, candidate_out=batches_per_frame,
        )
    frames = range(0, min(len(actions), args.every * args.limit), args.every)
    flips, with_correct, without_correct, decided = [], [], [], 0
    survival_gaps = []
    for frame in frames:
        batches = batches_per_frame[frame] if frame < len(batches_per_frame) \
            else []
        if not batches:
            continue
        scores = np.vstack([np.asarray(batch["scores"], dtype=float)
                            for batch in batches])
        sequences = np.vstack([np.asarray(batch["sequences"], dtype=float)
                               for batch in batches])
        cost_with = scores[:, 0]
        cost_without = scores[:, 0] - scores[:, 3]
        plans_with = best_by_first(batches, cost_with)
        if len(plans_with) < 2:
            continue
        env = replay_to(args.eval_seed * 10000, actions, frame)
        if env.dead:
            continue
        survival = {
            action: rollout(env, plan)
            for action, plan in plans_with.items()
        }
        if survival[0.0] == survival[1.0]:
            continue
        better = 0.0 if survival[0.0] > survival[1.0] else 1.0

        def first_choice(cost):
            first = sequences[:, 0]
            totals = {
                action: float(cost[first == action].min())
                for action in (0.0, 1.0) if (first == action).any()
            }
            return min(totals, key=totals.get)

        choice_with = first_choice(cost_with)
        choice_without = first_choice(cost_without)
        decided += 1
        with_correct.append(choice_with == better)
        without_correct.append(choice_without == better)
        if choice_with != choice_without:
            flips.append(choice_with == better)
        survival_gaps.append(abs(survival[0.0] - survival[1.0]))
    decided = max(decided, 1)
    print("== seed %d eval %d: %d decisions with a decided first action ==" % (
        args.seed, args.eval_seed, decided,
    ))
    print("  cost WITH value picks the longer-lived plan: %.2f" % (
        float(np.mean(with_correct)) if with_correct else 0.0,
    ))
    print("  cost WITHOUT value picks the longer-lived plan: %.2f" % (
        float(np.mean(without_correct)) if without_correct else 0.0,
    ))
    if flips:
        print("  decisions where the two disagree: %d  value-side correct: "
              "%.2f" % (len(flips), float(np.mean(flips)),))
    else:
        print("  decisions where the two disagree: 0")
    if survival_gaps:
        print("  survival gap between the two plans: median %.1f frames" % (
            float(np.median(survival_gaps)),
        ))


if __name__ == "__main__":
    main()
