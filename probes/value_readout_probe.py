"""Can the node graph supply a value parameter at all?

The endpoint valuation reads a value function that is currently empty
(`feature_value` has zero variance at `epochs=1`).  Before wiring another
readout, measure whether the information exists in the quantities the stream
already produces:

* for every training row, the realised return-to-go (the feedback actually
  received from that state to the end of its episode) and the frames left;
* aggregate per nearest prototype (node association), with the node's count as
  strength;
* read the same statistic on a held-out stream and ask whether it ranks the
  realised remaining survival, and whether it separates "death within 8".

No counterfactual is used: every row is a state the behaviour policy actually
entered, and its label is what actually happened afterwards.
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


def nearest_nodes(agent, encoded):
    nodes = []
    for state in encoded:
        node_id, _distance = agent._nearest(state)
        nodes.append(node_id)
    return nodes


def return_to_go(rows):
    """Feedback actually received from each row to the end of its episode."""
    values = np.asarray(
        [row.get("signals", (0.0, 0.0)) for row in rows], dtype=float
    )
    feedback = values[:, 0] - values[:, 1]
    return np.cumsum(feedback[::-1])[::-1]


def spearman(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or np.std(left) <= 0 or np.std(right) <= 0:
        return 0.0
    rank_left = np.argsort(np.argsort(left)).astype(float)
    rank_right = np.argsort(np.argsort(right)).astype(float)
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


def node_statistics(agent, rows):
    encoded = encode(agent, rows)
    nodes = nearest_nodes(agent, encoded)
    totals, counts, frames = {}, {}, {}
    for index, row in enumerate(rows):
        node = nodes[index]
        if node is None:
            continue
        totals[node] = totals.get(node, 0.0) + float(row["_return"])
        frames[node] = frames.get(node, 0) + float(row["_left"])
        counts[node] = counts.get(node, 0) + 1
    return {
        node: dict(
            count=counts[node],
            value=totals[node] / counts[node],
            remain=frames[node] / counts[node],
        )
        for node in counts
    }


def label(rows):
    for index, row in enumerate(rows):
        row["_left"] = float(len(rows) - 1 - index)
    values = return_to_go(rows)
    for index, row in enumerate(rows):
        row["_return"] = float(values[index])


def read(agent, stats, rows, min_count):
    encoded = encode(agent, rows)
    nodes = nearest_nodes(agent, encoded)
    values, coverage = [], []
    for node in nodes:
        entry = stats.get(node)
        if entry is None or entry["count"] < min_count:
            values.append(0.0)
            coverage.append(0.0)
        else:
            values.append(entry["value"])
            coverage.append(1.0)
    return np.asarray(values), float(np.mean(coverage))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=3)
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
        training = {}
        for row in agent.milestone_training_rows:
            training.setdefault(int(row.get("episode", 0)), []).append(row)
        for rows in training.values():
            label(rows)
        stats = node_statistics(
            agent, [row for rows in training.values() for row in rows]
        )
        print("== seed %d: %d nodes with statistics ==" % (seed, len(stats)))
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
        for episode_rows in episodes.values():
            label(episode_rows)
        values, coverage = read(agent, stats, rows, args.min_count)
        remaining = np.asarray([row["_left"] for row in rows])
        death = (remaining <= args.horizon).astype(int)
        grouped = {}
        for index, row in enumerate(rows):
            grouped.setdefault(int(row.get("episode", 0)), []).append(index)
        within_value, within_left = [], []
        for indices in grouped.values():
            if len(indices) < 3:
                continue
            local_value = values[indices]
            local_left = remaining[indices]
            within_value.extend(local_value - local_value.mean())
            within_left.extend(local_left - local_left.mean())
        print("  coverage %.2f  spearman(value, left) %+.3f  within %+.3f  "
              "AUC(death within 8) %s  base %.2f" % (
                  coverage, spearman(values, remaining),
                  spearman(within_value, within_left),
                  None if auc(values, death) is None
                  else round(auc(values, death), 3),
                  float(death.mean()),
              ))
        # The readout the planner uses today, on the same states.
        encoded = encode(agent, rows)
        mask = agent._feature_mask_table(encoded[:, :agent.obs_dim])
        current, support = agent.feature_value_batch(
            encoded, mask, return_support=True
        )
        print("  current value readout: std %.3f  support %.2f  "
              "AUC(death within 8) %s" % (
                  float(np.std(current)), float(np.mean(support)),
                  None if auc(-np.asarray(current, dtype=float), death) is None
                  else round(auc(-np.asarray(current, dtype=float), death), 3),
              ))


if __name__ == "__main__":
    main()
