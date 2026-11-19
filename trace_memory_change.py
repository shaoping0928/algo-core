"""Trace memory-node changes before and after real learning.

Randomly samples episodes from several seeds and records:

  * memory nodes before vs after one episode;
  * memories created, updated, or forgotten;
  * membership growth and which concept/transition nodes joined;
  * memory value/trace/count before vs after full training;
  * retention of first-pass memories after later epochs.

No targeted task collection or near-pipe curriculum is used.
"""
import argparse

import numpy as np

from node_graph import MEMBERSHIP, MEMORY
from node_runtime import NodeRuntime
from substrate_generic import collect

LAG = 3


def _memory_snapshot(agent):
    result = {}
    for node_id, node in agent.graph.nodes.items():
        if node.kind != MEMORY:
            continue
        members = []
        weights = []
        for key in agent.graph.outgoing.get(node_id, []):
            if key[2] != MEMBERSHIP:
                continue
            edge = agent.graph.edges[key]
            members.append(key[1])
            weights.append(float(edge.weight))
        result[node_id] = dict(
            value=float(node.value),
            dopamine=float(node.dopamine),
            trace=float(node.trace),
            count=int(node.count),
            members=members,
            weights=weights,
        )
    return result


def _summary(memories):
    if not memories:
        return "none"
    values = np.asarray([item["value"] for item in memories.values()])
    traces = np.asarray([item["trace"] for item in memories.values()])
    members = np.asarray([len(item["members"]) for item in memories.values()])
    return (
        "n=%d value=%.3f/%.3f trace=%.3f/%.3f members=%.1f"
        % (
            len(memories),
            float(values.mean()), float(values.max()),
            float(traces.mean()), float(traces.max()),
            float(members.mean()),
        )
    )


def _print_memory(agent, memory_id, label, before=None):
    item = _memory_snapshot(agent).get(memory_id)
    if item is None:
        print("    %s %s: gone" % (label, memory_id))
        return
    delta = ""
    if before is not None:
        delta = " dvalue=%+.3f dtrace=%+.3f dcount=%+d" % (
            item["value"] - before["value"],
            item["trace"] - before["trace"],
            item["count"] - before["count"],
        )
    print(
        "    %s %-12s value=%.3f trace=%.3f count=%d members=%d "
        "weights=%.2f/%.2f%s"
        % (
            label, memory_id, item["value"], item["trace"], item["count"],
            len(item["members"]),
            min(item["weights"]) if item["weights"] else 0.0,
            max(item["weights"]) if item["weights"] else 0.0,
            delta,
        )
    )


def trace(seed, steps=5000, epochs=5, warmup=12):
    rows, _counts = collect(seed, steps, coverage=0.1)
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    scale = observations.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    scales = np.concatenate([scale, np.ones(LAG, dtype=float)])
    agent = NodeRuntime(
        obs_dim=observations.shape[1], action_space=(0, 1), lag=LAG,
        birth_threshold=0.45, max_nodes=5000, exploration=0.0,
        seed=seed, scales=scales,
    )
    episodes = {}
    for row in rows:
        episodes.setdefault(row["episode"], []).append(row)
    ordered = [episodes[key] for key in sorted(episodes)]
    rng = np.random.RandomState(seed + 9173)
    sample_index = int(rng.randint(warmup, max(warmup + 1, len(ordered))))
    print("\nseed=%d episodes=%d sampled_episode=%d"
          % (seed, len(ordered), sample_index))

    before_episode = None
    after_episode = None
    first_pass = None
    for epoch in range(epochs):
        for index, episode in enumerate(ordered):
            if epoch == 0 and index == sample_index:
                before_episode = _memory_snapshot(agent)
                agent.learn_episode(episode, build_memory=True)
                after_episode = _memory_snapshot(agent)
                if first_pass is None:
                    first_pass = dict(after_episode)
            else:
                agent.learn_episode(
                    episode, build_memory=(epoch == 0)
                )

    print("  before episode: %s" % _summary(before_episode))
    print("  after  episode: %s" % _summary(after_episode))
    created = set(after_episode) - set(before_episode)
    forgotten = set(before_episode) - set(after_episode)
    updated = []
    for memory_id in set(before_episode) & set(after_episode):
        before = before_episode[memory_id]
        after = after_episode[memory_id]
        if (abs(after["value"] - before["value"]) > 1e-6
                or abs(after["trace"] - before["trace"]) > 1e-6
                or after["count"] != before["count"]):
            updated.append(memory_id)
    print("  created=%d updated=%d forgotten=%d"
          % (len(created), len(updated), len(forgotten)))
    for memory_id in list(created)[:3]:
        _print_memory(agent, memory_id, "created")
    for memory_id in updated[:3]:
        _print_memory(
            agent, memory_id, "updated", before_episode[memory_id]
        )
    for memory_id in list(forgotten)[:3]:
        print("    forgotten %s" % memory_id)

    final = _memory_snapshot(agent)
    print("  after full training: %s" % _summary(final))
    if first_pass:
        retained = set(first_pass) & set(final)
        lost = set(first_pass) - set(final)
        print("  first-pass memories retained=%d lost=%d"
              % (len(retained), len(lost)))
        if retained:
            changes = [
                final[memory_id]["value"] - first_pass[memory_id]["value"]
                for memory_id in retained
            ]
            counts = [
                final[memory_id]["count"] - first_pass[memory_id]["count"]
                for memory_id in retained
            ]
            print("    retained value delta mean %+.3f  count delta mean %+.1f"
                  % (float(np.mean(changes)), float(np.mean(counts))))
            sample = sorted(retained)[:2]
            for memory_id in sample:
                _print_memory(
                    agent, memory_id, "retained", first_pass[memory_id]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=12)
    args = parser.parse_args()
    for seed in range(args.seed, args.seed + args.seeds):
        trace(
            seed, steps=args.steps, epochs=args.epochs,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
