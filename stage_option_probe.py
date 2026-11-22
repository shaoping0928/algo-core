"""Stage-level transitions with an outcome-based identity and an action option.

Two things the earlier stage graph did not have:

* identity from the stage's *realised outcome* (per-frame feedback, whether the
  stage ended in death), not from entry geometry or a short future signature -
  the signature was nearly identical across stages and mixed death with
  survival, so "next stage is worse" had a base rate of zero;
* an option: the action rate the stage actually used, binned by quantiles.
  The table is then (stage, option) -> next stage, which is the high-level
  transition a two-level controller would choose from.

Reported per seed: support, top-1 hit rate against its own chance level, mass on
the true successor, and the base rate of a worse next stage.  Offline only.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from milestone_graph import kmeans_labels, silhouette
from substrate_nodes import _curiosity_off, _stage_evidence_for_streams, fit


def instances(rows, boundaries):
    edges = [0] + [int(value) for value in boundaries] + [len(rows)]
    out = []
    for start, stop in zip(edges[:-1], edges[1:]):
        segment = rows[int(start):int(stop)]
        if not segment:
            continue
        actions = np.asarray(
            [float(np.asarray(row.get("history", (0.0,))).reshape(-1)[0])
             for row in segment], dtype=float,
        )
        signals = np.asarray(
            [row.get("signals", (0.0, 0.0)) for row in segment], dtype=float
        )
        feedback = signals[:, 0] - signals[:, 1]
        out.append(dict(
            rate=float(actions.mean()),
            per_frame=float(feedback.mean()),
            died=1.0 if any(bool(row.get("terminal")) for row in segment) else 0.0,
            duration=len(segment),
            entry=np.asarray(segment[0]["obs"], dtype=float),
            episode=int(segment[0].get("episode", 0)),
        ))
    return out


def cluster_stages(stages, min_support=3, max_clusters=4):
    features = np.asarray([[stage["per_frame"], stage["died"]]
                           for stage in stages], dtype=float)
    features = (features - features.mean(axis=0)) / np.where(
        features.std(axis=0) > 0, features.std(axis=0), 1.0
    )
    best = (np.zeros(len(stages), dtype=int), 1, 0.0)
    limit = min(max_clusters, len(stages) // max(min_support, 1))
    for count in range(2, int(limit) + 1):
        labels, _centers = kmeans_labels(features, count)
        support = np.bincount(labels, minlength=count)
        if support.min() < min_support:
            continue
        separation = silhouette(features, labels)
        if separation > best[2]:
            best = (labels, count, separation)
    return best[0], best[1], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--options", type=int, default=2)
    parser.add_argument("--min-support", type=int, default=3)
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)

    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
        )
        episodes = {}
        for row in agent.milestone_training_rows:
            episodes.setdefault(int(row.get("episode", 0)), []).append(row)
        usable = [rows for rows in episodes.values() if len(rows) >= 3]
        evidence = _stage_evidence_for_streams(usable, world, agent)
        boundaries_list, _signatures = evidence
        stages, pairs = [], []
        for rows, boundaries in zip(usable, boundaries_list):
            local = instances(rows, boundaries)
            for index in range(len(local) - 1):
                pairs.append((local[index], local[index + 1]))
            stages.extend(local)
        labels, count, separation = cluster_stages(
            stages, min_support=args.min_support
        )
        rates = np.asarray([stage["rate"] for stage in stages])
        cuts = np.quantile(rates, np.linspace(0, 1, args.options + 1)[1:-1])
        options = np.searchsorted(cuts, rates)

        # Re-index pairs with the clustered identity and the option bin.
        table = {}
        transitions = []
        stage_index = 0
        by_episode = {}
        for rows, boundaries in zip(usable, boundaries_list):
            local = instances(rows, boundaries)
            for item in local:
                item["stage_id"] = int(labels[stage_index])
                item["option"] = int(options[stage_index])
                stage_index += 1
            by_episode[int(local[0]["episode"])] = local
        for local in by_episode.values():
            for index in range(len(local) - 1):
                source = local[index]
                target = local[index + 1]
                key = (source["stage_id"], source["option"])
                entry = table.setdefault(key, {})
                entry[target["stage_id"]] = entry.get(target["stage_id"], 0) + 1
                transitions.append((key, target["stage_id"], target["per_frame"]))

        warned = [pair for pair in pairs if pair[1]["per_frame"] < 0]
        top1, chance, mass = 0.0, 0.0, 0.0
        for key, target, _value in transitions:
            counts = table[key]
            total = sum(counts.values())
            best = max(counts, key=counts.get)
            top1 += 1.0 if best == target else 0.0
            chance += 1.0 / len(counts)
            mass += counts.get(target, 0) / float(total)
        n = max(len(transitions), 1)
        average = float(np.mean([stage["per_frame"] for stage in stages]))
        worse = float(np.mean([value < average for _key, _target, value
                               in transitions])) if transitions else 0.0
        # Does the option change the next stage's outcome within a stage?
        by_stage = {}
        for (stage_id, option), target, value in transitions:
            by_stage.setdefault(stage_id, {}).setdefault(option, []).append(value)
        spreads = [
            max(np.mean(v) for v in options_map.values())
            - min(np.mean(v) for v in options_map.values())
            for options_map in by_stage.values() if len(options_map) > 1
        ]
        cells = len(table)
        supported = sum(
            1 for counts in table.values() if sum(counts.values()) >= 3
        )
        print("== seed %d: stages %d  identities %d (sep %.2f)  "
              "transitions %d  cells %d  cells>=3 %d ==" % (
                  seed, len(stages), count, separation, len(transitions),
                  cells, supported,
              ))
        print("  top-1 %.2f vs chance %.2f  mass %.2f  worse-next base %.2f" % (
            top1 / n, chance / n, mass / n, worse,
        ))
        print("  option spread of next-stage per-frame: %s" % (
            "none" if not spreads else "mean %.3f  max %.3f" % (
                float(np.mean(spreads)), float(np.max(spreads)),
            )
        ))


if __name__ == "__main__":
    main()
