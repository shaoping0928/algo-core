"""Trace which stage of the node runtime is weak.

Stages:
  1. world model: predicted next observation vs real next observation
  2. node value: learned pain value on terminal / near-pipe states
  3. action support: do both actions have learned action edges at the decision state?
  4. decision: chosen action vs the observation's gap direction near a pipe

This is an evaluation script.  The gap-direction check is a diagnostic only;
it is never used to train the node runtime or the world model.
"""
import argparse

import numpy as np

from latent_world import latent
from substrate_generic import history_features
from substrate_nodes import fit, policy_result
from substrate_value import binary_auc

LAG = 3
OBS_NAMES = ("y", "vy", "dx1", "gap1", "dx2", "gap2")


def _node_state(agent, observation, pending):
    return np.concatenate(
        [np.asarray(observation, dtype=float), np.asarray(pending, dtype=float)]
    ) / agent.scales


def trace(seed, steps=5000, coverage=0.1, cap=1500, horizon=10,
          epochs=10, max_nodes=5000, birth_threshold=0.45,
          components=(), component_min_count=8, max_components=128,
          component_gating=False):
    world, agent = fit(
        seed, steps=steps, coverage=coverage, max_nodes=max_nodes,
        birth_threshold=birth_threshold, epochs=epochs,
        component_kinds=components,
        component_min_count=component_min_count,
        max_components=max_components,
        component_gating=component_gating,
    )
    env = latent[LAG](seed)
    agent.reset()
    prefix = np.zeros(LAG, dtype=float)
    errors = []
    values = []
    labels = []
    near = []
    distances = []
    visit_counts = []
    immediate_values = []
    nearest_values = []
    directions = []
    support = []
    near_support = []
    gated_hits = 0
    gated_checks = 0
    chosen_pain = []
    zero_pain = []
    one_pain = []
    t = 0
    while not env.dead and t < cap:
        observation = np.asarray(env.obs(), dtype=float)
        result = policy_result(env, world, agent, prefix, horizon, lag=LAG)
        action = int(result.actions[0])

        pending_before = prefix[::-1].copy()
        state = _node_state(agent, observation, pending_before)
        node_id = agent._assign(state, create=False, count=False)
        signature = agent._signature(node_id)
        scores = agent._score_actions(
            node_id, state, signature=signature
        )
        counts = [scores[a]["count"] for a in (0, 1)]
        chosen_count = float(scores[action]["count"])
        other_count = float(scores[1 - action]["count"])
        support.append((chosen_count, other_count,
                         counts[0] > 0 and counts[1] > 0))
        dx = float(observation[2])
        gap = float(observation[3])
        is_near = abs(dx) < 0.2 and dx > -0.1
        if is_near:
            near_support.append((chosen_count, other_count,
                                  counts[0] > 0 and counts[1] > 0))
        if is_near and abs(gap) > 0.02:
            expected = 1 if gap > 0.02 else 0
            directions.append(int(action == expected))
        if signature:
            gated_checks += 1
            if any((node_id, signature, a) in agent.gated_edges
                   for a in (0, 1)):
                gated_hits += 1

        history = np.concatenate([[float(action)], prefix[:LAG]])[:LAG + 1]
        predicted = world.predict(
            observation.reshape(1, -1), history.reshape(1, -1)
        )[0]
        pending_after = prefix[:LAG - 1][::-1].tolist() + [float(action)]

        # Value of the predicted endpoint and of fixed all-zero / all-one plans.
        chosen_pain.append(float(result.score[0]))
        zero_plan = np.zeros(horizon, dtype=float)
        one_plan = np.ones(horizon, dtype=float)
        for plan, bucket in ((zero_plan, zero_pain), (one_plan, one_pain)):
            hs = history_features(
                plan.reshape(1, -1), horizon, prefix,
                action_dim=LAG + 1,
            )
            endpoint = world.predict(observation.reshape(1, -1), hs)[0]
            pending_plan = hs[:, :LAG][:, ::-1][0]
            encoded = _node_state(agent, endpoint, pending_plan)
            bucket.append(agent.value_state(encoded))

        env.step(action)
        next_observation = np.asarray(env.obs(), dtype=float)
        feedback = env.feedback()
        errors.append(np.abs(predicted - next_observation))
        next_state = _node_state(agent, next_observation, pending_after)
        values.append(agent.value_state(next_state))
        nearest_node, nearest_distance = agent._nearest(next_state)
        distances.append(nearest_distance)
        visit_counts.append(
            agent.graph.nodes[nearest_node].count
            if nearest_node is not None else 0
        )
        immediate_values.append(
            agent.graph.nodes[nearest_node].immediate_pain
            if nearest_node is not None else 0.0
        )
        nearest_values.append(
            agent.graph.nodes[nearest_node].value
            if nearest_node is not None else 0.0
        )
        labels.append(bool(env.dead))
        near.append(abs(float(next_observation[2])) < 0.2
                    and float(next_observation[2]) > -0.1)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        t += 1

    errors = np.asarray(errors)
    values = np.asarray(values)
    labels = np.asarray(labels)
    near = np.asarray(near)
    distances = np.asarray(distances)
    visit_counts = np.asarray(visit_counts)
    immediate_values = np.asarray(immediate_values)
    nearest_values = np.asarray(nearest_values)
    support = np.asarray(support)
    near_support = (
        np.asarray(near_support) if near_support
        else np.zeros((0, 3))
    )
    directions = np.asarray(directions, dtype=float)
    chosen_pain = np.asarray(chosen_pain)
    zero_pain = np.asarray(zero_pain)
    one_pain = np.asarray(one_pain)

    print("seed=%d frames=%d steps=%d" % (seed, env.t, len(values)))
    print("\n[1] world model one-step error")
    for index, name in enumerate(OBS_NAMES):
        print("  %-4s mean %.2e  max %.2e"
              % (name, errors[:, index].mean(), errors[:, index].max()))

    print("\n[2] node value")
    print("  terminal AUC %.3f  near-pipe AUC %.3f  terminal mean %.3f  safe mean %.3f"
          % (
              binary_auc(values, labels),
              binary_auc(values[near], labels[near]) if near.any() else float("nan"),
              values[labels].mean() if labels.any() else float("nan"),
              values[~labels].mean() if (~labels).any() else float("nan"),
          ))
    print("  nearest distance terminal/safe %.2f/%.2f  count terminal/safe %.1f/%.1f"
          % (
              distances[labels].mean() if labels.any() else float("nan"),
              distances[~labels].mean() if (~labels).any() else float("nan"),
              visit_counts[labels].mean() if labels.any() else float("nan"),
              visit_counts[~labels].mean() if (~labels).any() else float("nan"),
          ))
    print("  nearest immediate pain terminal/safe %.3f/%.3f  nearest value %.3f/%.3f"
          % (
              immediate_values[labels].mean() if labels.any() else float("nan"),
              immediate_values[~labels].mean() if (~labels).any() else float("nan"),
              nearest_values[labels].mean() if labels.any() else float("nan"),
              nearest_values[~labels].mean() if (~labels).any() else float("nan"),
          ))
    supported = visit_counts >= 5
    if supported.any() and labels[supported].any() and (~labels[supported]).any():
        print("  in-support (count>=5) AUC %.3f over %d states"
              % (
                  binary_auc(values[supported], labels[supported]),
                  int(supported.sum()),
              ))

    print("\n[3] action-edge support")
    print("  both actions supported: %.3f  chosen count mean %.1f  other count mean %.1f"
          % (
              float(support[:, 2].mean()),
              float(support[:, 0].mean()),
              float(support[:, 1].mean()),
          ))
    if len(near_support):
        print("  near-pipe both actions: %.3f  mean chosen/other %s"
              % (
                  float(near_support[:, 2].mean()),
                  np.round(near_support[:, :2].mean(axis=0), 1).tolist(),
              ))
    print("  gated signature available: %d/%d" % (gated_hits, gated_checks))

    print("\n[4] decision direction near pipe")
    if len(directions):
        print("  directional accuracy %.3f over %d events"
              % (float(directions.mean()), len(directions)))
    else:
        print("  no near-pipe decisions")
    print("  chosen pain <= zero plan: %.3f  chosen <= one plan: %.3f"
          % (
              float(np.mean(chosen_pain <= zero_pain + 1e-9)),
              float(np.mean(chosen_pain <= one_pain + 1e-9)),
          ))
    return dict(
        frames=env.t,
        world_error=errors.mean(axis=0).tolist(),
        terminal_auc=binary_auc(values, labels),
        near_auc=(
            binary_auc(values[near], labels[near]) if near.any()
            else float("nan")
        ),
        directional_accuracy=(
            float(directions.mean()) if len(directions) else float("nan")
        ),
        support_both=float(support[:, 2].mean()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--birth-threshold", type=float, default=0.45)
    parser.add_argument("--components", default="")
    parser.add_argument("--component-min-count", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--component-gating", action="store_true")
    args = parser.parse_args()
    components = tuple(
        kind.strip() for kind in args.components.split(",") if kind.strip()
    )
    trace(
        args.seed, steps=args.steps, coverage=args.coverage, cap=args.cap,
        horizon=args.horizon, epochs=args.epochs, max_nodes=args.max_nodes,
        birth_threshold=args.birth_threshold, components=components,
        component_min_count=args.component_min_count,
        max_components=args.max_components,
        component_gating=args.component_gating,
    )


if __name__ == "__main__":
    main()
