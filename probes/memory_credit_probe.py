"""Diagnose whether retrospective memory credit does anything at readout.

Fits the same benchmark arm twice (memory credit off/on), then runs the
held-out evaluation episodes and reports:

* how many pain events corrected an earlier memory, and how much
  eligibility mass moved (`memory_credit_stats`);
* whether those corrected memories are recalled at all on the held-out
  trajectories (active feature roots -> membership lookup), and their
  risk-rate reads;
* graph-side counts (memories, memberships, corrected-vs-other rates).

This is a read-only probe: it uses no task geometry and writes JSON.
"""
import argparse
import json

import numpy as np

from eval_protocol import parse_seed_spec
from node_graph import MEMBERSHIP, MEMORY
from substrate_nodes import _curiosity_off, evaluate, fit


def _evaluate_rows(seed, world, agent, eval_seeds, cap=3000, horizon=8):
    rows = []
    with _curiosity_off(agent):
        for eval_seed in eval_seeds:
            _frames, _actions, episode_rows = evaluate(
                seed, world, agent, cap=cap, horizon=horizon,
                step_stride=2, env_seed=eval_seed * 10000,
                record=True, episode=0,
            )
            rows.extend(episode_rows)
    return rows


def _memory_stats(agent):
    memories = [
        node_id for node_id, node in agent.graph.nodes.items()
        if node.kind == MEMORY
    ]
    corrected = agent.memory_credit_stats.get("corrected_ids", set())
    rates = {
        node_id: float(agent._pain_association(node_id))
        for node_id in memories
    }
    corrected_rates = [rates[node_id] for node_id in corrected if node_id in rates]
    other_rates = [rates[node_id] for node_id in memories if node_id not in corrected]
    memberships = sum(
        1 for key in agent.graph.edges if key[2] == MEMBERSHIP
    )
    root_set = set(agent.feature_ids)
    feature_memberships = sum(
        1 for key in agent.graph.edges
        if key[2] == MEMBERSHIP and key[1] in root_set
    )
    memories_with_feature_member = sum(
        1 for node_id in memories
        if any(member in root_set for member in agent.graph.nodes[node_id].members)
    )
    return dict(
        features=len(agent.feature_ids),
        feature_all=len(agent._feature_all),
        feature_memberships=feature_memberships,
        memories_with_feature_member=memories_with_feature_member,
        memories=len(memories),
        memberships=memberships,
        corrected_ids=len(corrected),
        events=int(agent.memory_credit_stats.get("events", 0)),
        corrected=int(agent.memory_credit_stats.get("corrected", 0)),
        eligibility_mass=round(
            float(agent.memory_credit_stats.get("weight", 0.0)), 3
        ),
        context_links=int(agent.memory_credit_stats.get("links", 0)),
        corrected_rate_mean=(
            round(float(np.mean(corrected_rates)), 4) if corrected_rates else None
        ),
        other_rate_mean=(
            round(float(np.mean(other_rates)), 4) if other_rates else None
        ),
    )


def _recall_stats(agent, rows):
    corrected = agent.memory_credit_stats.get("corrected_ids", set())
    any_memory = 0
    corrected_recall_rows = 0
    hit_ids = set()
    rates = []
    active_counts = []
    for row in rows:
        scaled = (
            np.asarray(row["obs"], dtype=float)[:agent.obs_dim]
            / agent.scales[:agent.obs_dim]
        )
        active = agent._active_feature_ids(scaled)
        active_counts.append(len(active))
        recalled = agent.memories_for(active)
        if recalled:
            any_memory += 1
        hits = set(recalled) & corrected if corrected else set()
        if hits:
            corrected_recall_rows += 1
            hit_ids |= hits
            rates.extend(float(agent._pain_association(mid)) for mid in hits)
    return dict(
        eval_rows=len(rows),
        rows_with_active_features=sum(1 for c in active_counts if c),
        mean_active_roots=(
            round(float(np.mean(active_counts)), 2) if active_counts else 0.0
        ),
        rows_recalling_any_memory=any_memory,
        rows_recalling_corrected=corrected_recall_rows,
        corrected_ids_recalled=len(hit_ids),
        corrected_recall_rate_mean=(
            round(float(np.mean(rates)), 4) if rates else None
        ),
    )


def run_arm(seed, eval_seeds, memory_credit):
    world, agent = fit(
        seed,
        steps=1500, epochs=3, max_nodes=1000,
        birth_threshold=0.45, lag=3,
        curiosity_weight=0.5,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=3000,
        dagger_frames=3000, dagger_horizon=8,
        step_stride=2,
        memory_credit=memory_credit,
    )
    rows = _evaluate_rows(seed, world, agent, eval_seeds)
    return dict(
        arm="memory-credit" if memory_credit else "control",
        train_memory=_memory_stats(agent),
        eval_recall=_recall_stats(agent, rows),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    eval_seeds = parse_seed_spec(args.eval_seeds)

    control = run_arm(args.seed, eval_seeds, False)
    credit = run_arm(args.seed, eval_seeds, True)
    report = dict(
        seed=args.seed,
        eval_seeds=[int(value) for value in eval_seeds],
        control=control,
        memory_credit=credit,
    )
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
