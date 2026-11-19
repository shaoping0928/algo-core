"""Trace what changes inside the node graph during training.

This records the training process, not the post-training evaluation:

  * when and where concept nodes are born;
  * action/association/transition/memory growth per wave;
  * how many nodes have both actions supported (overall and near a pipe);
  * node value percentiles and terminal-node values;
  * component sign transitions, if components are enabled;
  * which nodes receive credit after a terminal transition.

The diagnostic may read observation geometry to name a death, but training
itself only consumes obs/action/valence/done.
"""
import argparse

import numpy as np

from node_graph import ASSOCIATION, CONCEPT, MEMORY, TRANSITION_EDGE
from node_runtime import NodeRuntime
from substrate_generic import collect

LAG = 3


def _action_support(graph, near_only=False, scales=None):
    both = 0
    total = 0
    near_total = 0
    near_both = 0
    for node in graph.nodes.values():
        if node.kind != CONCEPT or node.role == "component_state":
            continue
        actions = set()
        for action in (0, 1):
            if graph.action_edges_from(node.id, action):
                actions.add(action)
        total += 1
        supported = len(actions) == 2
        both += int(supported)
        is_near = False
        if near_only and node.features is not None and scales is not None:
            dx = float(node.features[2] * scales[2])
            is_near = abs(dx) < 0.2 and dx > -0.1
        if is_near:
            near_total += 1
            near_both += int(supported)
    return both, total, near_both, near_total


def _decode(features, scales):
    if features is None:
        return "n/a"
    values = np.asarray(features, dtype=float)[:6] * np.asarray(scales)[:6]
    return "[y=%+.2f vy=%+.2f dx=%+.2f gap=%+.2f]" % tuple(values[:4])


def _death_type(row):
    if not row["terminal"]:
        return None
    y = float(row["next_obs"][0]) + 0.5
    if y < 0.0 or y > 1.0:
        return "boundary"
    return "pipe/other"


def trace(seed, steps=5000, coverage=0.1, epochs=5, wave_episodes=10,
          max_nodes=2000, birth_threshold=0.45,
          components=(), component_min_count=8, max_components=128,
          component_gating=False, credit_samples=5):
    rows, _counts = collect(seed, steps, coverage=coverage)
    if not rows:
        raise ValueError("no transitions collected")
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    scale = observations.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    scales = np.concatenate([scale, np.ones(LAG, dtype=float)])
    agent = NodeRuntime(
        obs_dim=observations.shape[1], action_space=(0, 1), lag=LAG,
        birth_threshold=birth_threshold, max_nodes=max_nodes,
        exploration=0.0, seed=seed, scales=scales,
        component_kinds=components,
        component_min_count=component_min_count,
        max_components=max_components,
        component_gating=component_gating,
    )
    episodes = {}
    for row in rows:
        episodes.setdefault(row["episode"], []).append(row)
    ordered = [episodes[key] for key in sorted(episodes)]

    print("seed=%d transitions=%d episodes=%d epochs=%d"
          % (seed, len(rows), len(ordered), epochs))
    print("wave  ep  nodes  new  actions  assoc  trans  mem  comp  signs "
          "both  near_both  low  iso  p50/p90/max  term")
    previous_nodes = set()
    wave = 0
    seen_episodes = 0
    credits = 0
    for epoch in range(epochs):
        for episode in ordered:
            before = {
                node_id: (node.value, node.immediate_pain)
                for node_id, node in agent.graph.nodes.items()
            }
            death = next(
                (row for row in episode if row["terminal"]), None
            )
            agent.learn_episode(episode, build_memory=(epoch == 0))
            seen_episodes += 1
            if death is not None and credits < credit_samples:
                changed = []
                for node_id, node in agent.graph.nodes.items():
                    old_value, old_immediate = before.get(
                        node_id, (0.0, 0.0)
                    )
                    delta = abs(node.value - old_value) + abs(
                        node.immediate_pain - old_immediate
                    )
                    if delta > 1e-6:
                        changed.append((delta, node_id, node))
                changed.sort(key=lambda item: item[0], reverse=True)
                print("\n  credit[%s] episode=%s nodes_updated=%d"
                      % (_death_type(death), death["episode"], len(changed)))
                for delta, node_id, node in changed[:5]:
                    print("    %-10s d=%+.4f value=%.3f now=%.3f count=%d %s"
                          % (node_id, delta, node.value,
                             node.immediate_pain, node.count,
                             _decode(node.features, scales)))
                credits += 1

            if seen_episodes % wave_episodes == 0:
                wave += 1
                current_nodes = set(agent.graph.nodes)
                new_nodes = {
                    node_id for node_id in current_nodes - previous_nodes
                    if agent.graph.nodes[node_id].kind == CONCEPT
                    and agent.graph.nodes[node_id].role
                    != "component_state"
                }
                previous_nodes = current_nodes
                action_edges = len(agent.graph.action_edges)
                associations = sum(
                    edge.kind == ASSOCIATION
                    for edge in agent.graph.edges.values()
                )
                transitions = sum(
                    edge.kind == TRANSITION_EDGE
                    for edge in agent.graph.edges.values()
                )
                memories = sum(
                    node.kind == MEMORY
                    for node in agent.graph.nodes.values()
                )
                both, total, near_both, near_total = _action_support(
                    agent.graph, near_only=True, scales=scales
                )
                values = np.asarray([
                    node.value for node in agent.graph.nodes.values()
                    if node.kind == CONCEPT
                    and node.role != "component_state"
                ])
                terminal_nodes = [
                    node for node in agent.graph.nodes.values()
                    if node.kind == CONCEPT
                    and node.role != "component_state"
                    and node.immediate_pain > 0.5
                ]
                concept_nodes = [
                    node for node in agent.graph.nodes.values()
                    if node.kind == CONCEPT
                    and node.role != "component_state"
                ]
                low_count = sum(
                    node.count <= 1 for node in concept_nodes
                )
                isolated = sum(
                    not any(
                        agent.graph.action_edges_from(node.id, action)
                        for action in (0, 1)
                    )
                    for node in concept_nodes
                )
                component_states = sum(
                    node.role == "component_state"
                    for node in agent.graph.nodes.values()
                )
                print("%4d %4d %6d %4d %8d %6d %6d %5d %5d %6d "
                      "%.3f %9.3f %4d %4d  %.2f/%.2f/%.2f %d"
                      % (
                          wave, seen_episodes, total, len(new_nodes),
                          action_edges, associations, transitions,
                          memories, len(agent.components.specs),
                          component_states,
                          both / max(total, 1),
                          near_both / max(near_total, 1),
                          low_count, isolated,
                          np.percentile(values, 50),
                          np.percentile(values, 90),
                          values.max(),
                          len(terminal_nodes),
                      ))
                if new_nodes and wave <= 3:
                    samples = sorted(new_nodes)[:3]
                    for node_id in samples:
                        node = agent.graph.nodes[node_id]
                        print("    born %-10s count=%d %s"
                              % (node_id, node.count,
                                 _decode(node.features, scales)))
                if terminal_nodes and wave <= 3:
                    for node in sorted(
                        terminal_nodes,
                        key=lambda item: item.immediate_pain,
                        reverse=True,
                    )[:3]:
                        print("    terminal %-10s now=%.3f count=%d %s"
                              % (node.id, node.immediate_pain, node.count,
                                 _decode(node.features, scales)))

    print("\nfinal nodes=%d action_edges=%d components=%d component_states=%d"
          % (len(agent.graph.nodes), len(agent.graph.action_edges),
             len(agent.components.specs),
             sum(node.role == "component_state"
                 for node in agent.graph.nodes.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--wave-episodes", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--birth-threshold", type=float, default=0.45)
    parser.add_argument("--components", default="")
    parser.add_argument("--component-min-count", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--component-gating", action="store_true")
    parser.add_argument("--credit-samples", type=int, default=5)
    args = parser.parse_args()
    components = tuple(
        kind.strip() for kind in args.components.split(",") if kind.strip()
    )
    trace(
        args.seed, steps=args.steps, coverage=args.coverage,
        epochs=args.epochs, wave_episodes=args.wave_episodes,
        max_nodes=args.max_nodes,
        birth_threshold=args.birth_threshold, components=components,
        component_min_count=args.component_min_count,
        max_components=args.max_components,
        component_gating=args.component_gating,
        credit_samples=args.credit_samples,
    )


if __name__ == "__main__":
    main()
