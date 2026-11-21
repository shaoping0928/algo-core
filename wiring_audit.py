"""Wiring audit: does each link in the mechanism chain reach the decision?

Short, read-only.  Trains the current default stack on 1500 frames (no DAgger
round) and inspects every link: observations/scales, prototypes, induced
features, feature cells, hazard composition, value cells, memories, curiosity
evidence, the planner's score columns and the world model's rollout error.

No long task is run; the point is to expose links that are computed but
never consumed, or consumed with empty statistics.

Usage::

    python wiring_audit.py --seed 100 --steps 1500
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from latent_world import latent
from substrate_nodes import (
    LAG,
    _curiosity_off,
    _curiosity_potential,
    collect,
    fit,
    policy_result,
)

COLUMNS = ("cost", "-dopamine", "risk", "value", "curiosity", "q")


def spearman(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 2:
        return 0.0
    first_rank = np.argsort(np.argsort(first))
    second_rank = np.argsort(np.argsort(second))
    first_dev = first_rank - first_rank.mean()
    second_dev = second_rank - second_rank.mean()
    denominator = np.sqrt(
        (first_dev ** 2).sum() * (second_dev ** 2).sum()
    )
    return float((first_dev * second_dev).sum() / denominator) \
        if denominator > 0 else 0.0


def run(args):
    world, agent = fit(
        seed=args.seed, steps=args.steps, epochs=args.epochs,
        max_nodes=args.max_nodes, birth_threshold=0.45,
        curiosity_weight=args.curiosity_weight, memory_credit=False,
        dagger_rounds=0, feature_backfit=args.feature_backfit,
        feature_value_weight=args.feature_value_weight,
    )
    rows, _counts = collect(
        args.seed, args.steps, coverage=0.1, lag=LAG
    )
    scale = agent.scales[:agent.obs_dim]
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    scaled = observations / scale
    pending = np.asarray([
        np.asarray(row["history"], dtype=float)[1:LAG + 1][::-1]
        for row in rows
    ], dtype=float)
    encoded = np.concatenate([observations, pending], axis=1) / agent.scales
    mask_table = agent._feature_mask_table(scaled)

    checks = []

    def add(link, status, evidence):
        checks.append(dict(link=link, status=status, evidence=evidence))

    # L0 observations / scales
    constant = int(np.sum(observations.std(axis=0) <= 1e-9))
    add("observations/scales", "ok",
        "obs_dim=%d, constant dims=%d, scale range [%.3f, %.3f]"
        % (agent.obs_dim, constant, float(scale.min()), float(scale.max())))

    # L1 prototypes
    distances = []
    for state in encoded[::max(1, len(encoded) // 200)]:
        node_id, distance = agent._nearest(state)
        distances.append(distance if node_id is not None else np.inf)
    distances = np.asarray(distances)
    add("prototypes", "ok",
        "%d prototypes, nearest distance p50=%.3f p90=%.3f, >birth %.0f%%"
        % (len(agent._prototype_ids), float(np.percentile(distances, 50)),
           float(np.percentile(distances, 90)),
           100.0 * float(np.mean(distances > agent.birth_threshold))))

    # L2 features induced
    roots = list(agent.feature_ids)
    activation = []
    any_active = np.zeros(len(rows), dtype=bool)
    for node_id in roots:
        feature = agent._feature_objects.get(node_id)
        position = agent._feature_all.index(node_id)
        mask = np.asarray(mask_table[position], dtype=bool)
        activation.append(float(mask.mean()))
        any_active |= mask
    add("features induced", "ok",
        "%d roots, %d nodes, any-root rate %.1f%%"
        % (len(roots), len(agent._feature_all),
           100.0 * float(any_active.mean())))

    # L3 feature cells: written?
    action_cells = int(sum(
        len(stats) for stats in agent.feature_action_stats.values()
    ))
    action_obs = int(sum(
        cell[0] for stats in agent.feature_action_stats.values()
        for cell in stats.values()
    ))
    value_cells = int(sum(
        len(stats) for stats in agent.feature_value_stats.values()
    ))
    add("feature cells", "dead" if action_obs == 0 else "ok",
        "action cells=%d (obs=%d), value cells=%d, ledger cells=%d"
        % (action_cells, action_obs, value_cells,
           len(agent.feature_calibration)))

    # L4 hazard composition and read
    report = getattr(agent, "composition_report", None)
    if report:
        weights = {
            key: report.get(key) for key in report
            if isinstance(report.get(key), (int, float))
        }
        add("hazard composition", "ok", json.dumps(weights, default=str))
    else:
        add("hazard composition", "missing", "no composition_report")
    risk = agent.pain_signal_batch(encoded, mask_table)
    graded, _excess = agent.pain_graded_batch(encoded)
    pain = np.asarray([float(row["signals"][1]) for row in rows])
    add("hazard read", "ok",
        "pain_signal mean %.3f saturated %.0f%%, spearman %.2f; "
        "graded mean %.3f spearman %.2f"
        % (float(risk.mean()), 100.0 * float(np.mean(risk >= 0.99)),
           spearman(risk, pain), float(graded.mean()),
           spearman(graded, pain)))

    # L5 value channel
    value, support = agent.feature_value_batch(
        encoded, mask_table, return_support=True
    )
    gamma = float(agent.gamma)
    returns = np.zeros(len(rows))
    by_episode = defaultdict(list)
    for index, row in enumerate(rows):
        by_episode[int(row["episode"])].append(index)
    for indices in by_episode.values():
        running = 0.0
        for index in reversed(indices):
            running = float(rows[index]["signals"][0]) + gamma * running
            returns[index] = running
    add("value channel", "dead" if value_cells == 0 else "ok",
        "cells=%d, read mean %.3f, support mean %.1f, spearman %.2f"
        % (value_cells, float(np.mean(value)), float(np.mean(support)),
           spearman(value, returns)))

    # L6 memories
    memory_nodes = [
        node for node in agent.graph.nodes.values() if node.kind == "memory"
    ]
    retrieval = []
    for state in encoded[::max(1, len(encoded) // 200)]:
        node_id, _distance = agent._nearest(state)
        retrieval.append(
            0 if node_id is None else len(agent.memories_for([node_id]))
        )
    membership = []
    for node in memory_nodes:
        for key in agent.graph.outgoing.get(node.id, []):
            if key[2] == "membership":
                membership.append(float(agent.graph.edges[key].weight))
    add("memories", "sparse" if np.mean(retrieval) < 0.1 else "ok",
        "%d memories, retrieval %.1f%%, mean membership weight %.3f"
        % (len(memory_nodes), 100.0 * float(np.mean(np.asarray(
            retrieval) > 0)),
           float(np.mean(membership)) if membership else 0.0))

    # L7 curiosity evidence
    evidence = agent.evidence_batch(encoded, mask_table)
    phi = _curiosity_potential(evidence)
    add("curiosity", "inert" if np.allclose(phi, 1.0) else "ok",
        "evidence mean %.2f, Phi mean %.3f, distinct Phi %d"
        % (float(np.mean(evidence)), float(np.mean(phi)),
           int(len(np.unique(np.round(phi, 9))))))

    # L8 planner score columns
    env = latent[LAG](int(args.seed) * 10000)
    prefix = np.zeros(LAG, dtype=float)
    with _curiosity_off(agent):
        result = policy_result(env, world, agent, prefix, 8, lag=LAG)
    columns = {
        COLUMNS[index]: float(result.score[index])
        for index in range(len(COLUMNS))
    }
    active = [name for name, value in columns.items()
              if name in ("risk", "value", "curiosity", "q")
              and abs(value) > 1e-9]
    add("planner columns", "ok" if len(active) > 1 else "single-signal",
        "active=%s, %s" % (active, json.dumps(columns, default=str)))

    # L9 world model rollout error by step (within episode, plus held-out)
    scale_obs = scale

    def rollout_errors(source_rows):
        by_episode = defaultdict(list)
        for row in source_rows:
            by_episode[int(row["episode"])].append(row)
        collected = {1: [], 2: [], 4: [], 8: []}
        count = 0
        for episode in by_episode.values():
            for start in range(0, len(episode) - 8, 4):
                chunk = episode[start:start + 8]
                if any(bool(row["done"]) for row in chunk):
                    continue
                state = np.asarray(chunk[0]["obs"], dtype=float)
                for step in range(8):
                    history = np.asarray(
                        chunk[step]["history"], dtype=float
                    ).reshape(1, -1)
                    predicted = world.predict(
                        state.reshape(1, -1), history
                    )[0]
                    actual = np.asarray(
                        chunk[step]["next_obs"], dtype=float
                    )
                    if step + 1 in collected:
                        collected[step + 1].append(float(np.linalg.norm(
                            (predicted - actual) / scale_obs
                        )))
                    state = predicted
                count += 1
                if count >= args.windows:
                    break
            if count >= args.windows:
                break
        return {
            key: (float(np.mean(values)) if values else 0.0)
            for key, values in collected.items()
        }

    train_errors = rollout_errors(rows)
    heldout_rows, _ = collect(
        args.heldout_seed, args.heldout_frames, coverage=0.1, lag=LAG
    )
    heldout_errors = rollout_errors(heldout_rows)
    add("world model", "drift" if (
        train_errors[8] > 0.1 or heldout_errors[8] > 1.0
    ) else "ok",
        "train " + ", ".join(
            "k=%d %.4f" % (key, value)
            for key, value in train_errors.items()
        ) + " | heldout %d " % args.heldout_seed + ", ".join(
            "k=%d %.4f" % (key, value)
            for key, value in heldout_errors.items()
        ))

    # L10 learning keying
    pain_events = int(np.sum(pain > 0.5))
    add("learning labels", "ok",
        "%d rows, %d episodes, %d pain events (%.1f/1000), acting=history[%d]"
        % (len(rows), len(by_episode), pain_events,
           1000.0 * pain_events / max(len(rows), 1), LAG))

    return dict(seed=int(args.seed), checks=checks)


def print_report(report):
    print()
    print("== wiring audit: seed %d ==" % report["seed"])
    for check in report["checks"]:
        print(
            "  %-20s %-14s %s"
            % (check["link"], check["status"], check["evidence"])
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--feature-backfit", action="store_true")
    parser.add_argument("--feature-value-weight", type=float, default=0.02)
    parser.add_argument("--curiosity-weight", type=float, default=0.5)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--heldout-seed", type=int, default=120)
    parser.add_argument("--heldout-frames", type=int, default=600)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = run(args)
    print_report(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
