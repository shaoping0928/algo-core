"""Diagnostic for similarity-based stage types over integrated states.

Builds one key-weighted ``E(s)`` profile per realized stage with the normalized
association overlay, clusters train episodes by weighted cosine similarity, and
reports the held-out coverage / margin / predictive gain.  Nothing here changes
a decision.
"""
import argparse
import json

import numpy as np

from stage_type_clusters import StageSample, StageTypeClusters
from static_graph_strengths import learn_static_strengths
from static_transition import endogenous_transition
from substrate_nodes import fit


def stage_seeds(agent, episodes):
    return [
        [
            tuple(agent._active_feature_ids(
                np.asarray(stage.entry_state, dtype=float).reshape(-1)[
                    :int(agent.obs_dim)
                ] / np.asarray(agent.scales)[:int(agent.obs_dim)]
            ))
            for stage in episode
        ]
        for episode in episodes
    ]


def integrated(agent, seeds, strengths, depth):
    return [
        [
            endogenous_transition(
                agent, seeds=stage_seeds, max_steps=int(depth),
                strengths=strengths,
            )
            for stage_seeds in episode
        ]
        for episode in seeds
    ]


def key_weights(states):
    counts = {}
    total = 0
    for episode in states:
        for state in episode:
            total += 1
            for node in state.nodes:
                counts[node.node_id] = counts.get(node.node_id, 0) + 1
    if not total:
        return {}
    weights = {
        node: 1.0 / (1.0 + count / float(total))
        for node, count in counts.items()
    }
    maximum = max(weights.values(), default=1.0)
    return {
        node: value / maximum for node, value in weights.items()
    }


def outcome_of(episode, index):
    stage = episode[index]
    if stage.terminal:
        return ("terminal",)
    if stage.censored:
        return ("censored",)
    if index + 1 < len(episode):
        return tuple(episode[index + 1].key)
    return ("open",)


def build_samples(states, episodes, node_ids, weights):
    samples = []
    for episode_index, (episode_states, episode) in enumerate(
            zip(states, episodes)):
        for index, state in enumerate(episode_states):
            profile = state.vector(node_ids) * np.asarray(
                [weights.get(node, 0.0) for node in node_ids],
                dtype=float,
            )
            if not np.any(profile):
                continue
            samples.append(StageSample(
                episode=episode_index,
                profile=profile,
                outcome=outcome_of(episode, index),
                start=index,
                end=index + 1,
            ))
    return samples


def split(samples, modulus=5, held_out=4):
    train = [sample for sample in samples if sample.episode % modulus != held_out]
    validation = [
        sample for sample in samples if sample.episode % modulus == held_out
    ]
    return train, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--decay", type=float, default=0.5)
    parser.add_argument("--min-strength", type=float, default=0.05)
    parser.add_argument("--strength-min-support", type=int, default=2)
    parser.add_argument("--min-similarities", type=float, nargs="+",
                        default=[0.5, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--min-type-support", type=int, default=2)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    _world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
    )
    experience = getattr(agent, "stage_experience", None)
    if experience is None:
        raise SystemExit("the stage layer did not fit")
    episodes = list(experience.episodes)
    seeds = stage_seeds(agent, episodes)
    flat = [state for episode in seeds for state in episode]
    strengths = learn_static_strengths(
        flat, graph=getattr(agent, "graph", None),
        min_support=int(args.strength_min_support),
        membership_strength=1.0, normalized=True,
    )
    states = integrated(agent, seeds, strengths, args.depth)
    node_ids = []
    seen = set()
    for episode_states in states:
        for state in episode_states:
            for node in state.nodes:
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    node_ids.append(node.node_id)
    weights = key_weights(states)
    samples = build_samples(states, episodes, node_ids, weights)
    train, validation = split(samples)
    print(
        "stage type probe: seed %d | episodes %d | nodes %d | "
        "samples %d train %d validation %d"
        % (args.seed, len(episodes), len(node_ids), len(samples),
           len(train), len(validation))
    )
    rows = []
    for similarity in args.min_similarities:
        model = StageTypeClusters.fit(
            train, validation, min_similarity=float(similarity),
            min_coverage=float(args.min_coverage),
            min_type_support=int(args.min_type_support),
        )
        snapshot = model.snapshot()
        snapshot["min_similarity"] = float(similarity)
        rows.append(snapshot)
        print("  similarity %.2f: %s" % (similarity, snapshot))
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                steps=int(args.steps), epochs=int(args.epochs),
                max_nodes=int(args.max_nodes), depth=int(args.depth),
                decay=float(args.decay),
                min_strength=float(args.min_strength),
                strength_min_support=int(args.strength_min_support),
                min_type_support=int(args.min_type_support),
                min_coverage=float(args.min_coverage),
            ),
            strengths=strengths.snapshot(),
            nodes=int(len(node_ids)),
            samples=int(len(samples)),
            train=int(len(train)),
            validation=int(len(validation)),
            rows=rows,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
