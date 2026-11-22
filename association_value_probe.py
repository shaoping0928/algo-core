"""Association-propagated value: value(s) = sum over successors of w(s->r) * V(r).

The failed readout averaged each node's own realised return-to-go, which is
confounded with where in the episode the node is visited.  This one asks a
different question: *which nodes tend to follow s*, counted over a short
window, and what is the realised quality of those successors.  It is pure
co-occurrence counting (no model, no counterfactual): every pair is two states
the stream actually occupied in that order.

Evaluation is within-episode - the pooled correlation is inflated by episodes
of different lengths, so it is not the criterion.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from substrate_nodes import LAG, _curiosity_off, _encoded_pending, evaluate, fit
from decision_diag_probe import auc


def encode(agent, rows):
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    return np.concatenate(
        [observations, _encoded_pending(rows, agent.lag)], axis=1
    ) / agent.scales


def nodes_of(agent, encoded):
    out = []
    for state in encoded:
        node_id, _distance = agent._nearest(state)
        out.append(node_id)
    return out


def alive_fraction(rows, window):
    """Share of the next `window` frames the stream was still alive."""
    alive = np.asarray(
        [0.0 if bool(row.get("terminal")) else 1.0 for row in rows], dtype=float
    )
    out = np.zeros(len(rows))
    for index in range(len(rows)):
        stop = min(index + 1 + int(window), len(rows))
        out[index] = float(alive[index + 1:stop].mean()) if stop > index + 1 \
            else 0.0
    return out


def spearman(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or np.std(left) <= 0 or np.std(right) <= 0:
        return 0.0
    return float(np.corrcoef(
        np.argsort(np.argsort(left)), np.argsort(np.argsort(right))
    )[0, 1])


def build(agent, rows_by_episode, window, min_support):
    assoc = {}
    totals = {}
    quality = {}
    quality_count = {}
    for rows in rows_by_episode:
        nodes = nodes_of(agent, encode(agent, rows))
        value = alive_fraction(rows, window)
        for index, node in enumerate(nodes):
            if node is None:
                continue
            quality[node] = quality.get(node, 0.0) + float(value[index])
            quality_count[node] = quality_count.get(node, 0) + 1
            stop = min(index + 1 + window, len(nodes))
            for other in nodes[index + 1:stop]:
                if other is None or other == node:
                    continue
                row = assoc.setdefault(node, {})
                row[other] = row.get(other, 0) + 1
                totals[node] = totals.get(node, 0) + 1
    table = {}
    for node, neighbours in assoc.items():
        if totals.get(node, 0) < min_support:
            continue
        weights = {
            other: count / float(totals[node])
            for other, count in neighbours.items()
            if quality_count.get(other, 0) > 0
        }
        mass = sum(weights.values())
        if mass <= 0.0:
            continue
        table[node] = sum(
            weight * quality[other] / quality_count[other]
            for other, weight in weights.items()
        ) / mass
    return table


def read(agent, table, rows):
    nodes = nodes_of(agent, encode(agent, rows))
    return np.asarray([
        float(table.get(node, 0.5)) if node is not None else 0.5
        for node in nodes
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)

    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon,
            step_stride=2,
        )
        groups = {}
        for row in agent.milestone_training_rows:
            groups.setdefault(int(row.get("episode", 0)), []).append(row)
        table = build(
            agent, list(groups.values()), args.window, args.min_support
        )
        with _curiosity_off(agent):
            rows = []
            for eval_seed in eval_seeds:
                _frames, _actions, transitions = evaluate(
                    seed, world, agent, cap=5000, horizon=args.horizon,
                    lag=LAG, step_stride=2,
                    env_seed=int(eval_seed) * 10000, record=True,
                    episode=int(eval_seed),
                )
                rows.extend(transitions)
        episodes = {}
        for row in rows:
            episodes.setdefault(int(row.get("episode", 0)), []).append(row)
        values, remaining, within_value, within_left, death = [], [], [], [], []
        for indexes, episode_rows in episodes.items():
            local = read(agent, table, episode_rows)
            left = np.arange(len(episode_rows) - 1, -1, -1, dtype=float)
            values.extend(local)
            remaining.extend(left)
            within_value.extend(local - local.mean())
            within_left.extend(left - left.mean())
            death.extend((left <= args.horizon).astype(int))
        values = np.asarray(values)
        death = np.asarray(death)
        print("== seed %d: %d nodes with association value ==" % (
            seed, len(table)))
        print("  spearman %+.3f (within %+.3f)  AUC(death within 8) %s  "
              "base %.2f  distinct %.0f" % (
                  spearman(values, remaining),
                  spearman(within_value, within_left),
                  None if auc(values, death) is None
                  else round(auc(values, death), 3),
                  float(death.mean()), float(len(np.unique(values))),
              ))


if __name__ == "__main__":
    main()
