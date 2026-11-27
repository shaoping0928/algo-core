"""Diagnostic for the non-temporal (endogenous) state transition.

Runs the closure over association/membership/derivation edges at several depths,
reports what it integrates beyond the active feature seeds, and measures whether
the integrated state produces richer recurring co-change bundles than the seed
alone.  Nothing here changes a decision.
"""
import argparse
import json
from collections import Counter

import numpy as np

from stage_change_bundles import StageChangeBundles, StaticExpansion
from static_graph_strengths import learn_static_strengths
from static_transition import endogenous_transition
from substrate_nodes import fit


def seed_sets(agent, episodes):
    result = []
    for episode in episodes:
        sets = []
        for stage in episode:
            state = np.asarray(stage.entry_state, dtype=float).reshape(-1)
            if state.size >= int(agent.obs_dim):
                sets.append(tuple(
                    agent._active_feature_ids(
                        state[:int(agent.obs_dim)]
                        / np.asarray(agent.scales)[:int(agent.obs_dim)]
                    )
                ))
            else:
                sets.append(())
        result.append(sets)
    return result


def integrated_states(agent, seeds_per_episode, max_steps, strengths=None,
                      decay=0.5, min_strength=0.05):
    per_episode = []
    for seeds in seeds_per_episode:
        states = []
        for episode_seeds in seeds:
            states.append(endogenous_transition(
                agent, seeds=episode_seeds, max_steps=max_steps,
                decay=decay, min_strength=min_strength,
                strengths=strengths,
            ))
        per_episode.append(states)
    return per_episode


def summarize_states(per_episode):
    counts = []
    kinds = {}
    converged = 0
    edge_counts = {}
    for states in per_episode:
        for state in states:
            counts.append(len(state.nodes))
            converged += int(state.converged)
            for kind, ids in state.by_kind().items():
                kinds[kind] = kinds.get(kind, 0) + len(ids)
            for kind, count in state.edge_counts:
                edge_counts[kind] = edge_counts.get(kind, 0) + int(count)
    total = max(len(counts), 1)
    return dict(
        states=int(len(counts)),
        nodes_mean=float(np.mean(counts)) if counts else 0.0,
        nodes_max=int(max(counts)) if counts else 0,
        converged_rate=float(converged) / float(total),
        kinds={kind: int(value) for kind, value in sorted(kinds.items())},
        edge_counts={
            kind: int(value) for kind, value in sorted(edge_counts.items())
        },
    )


def bundle_summary(per_episode, min_support, max_candidates):
    node_ids = []
    seen = set()
    for states in per_episode:
        for state in states:
            for node in state.nodes:
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    node_ids.append(node.node_id)
    tables = []
    for states in per_episode:
        if not states:
            continue
        rows = [state.vector(node_ids) for state in states]
        tables.append(StaticExpansion(
            node_ids=tuple(node_ids),
            strengths=np.vstack(rows),
            weights=np.ones(len(node_ids)),
        ))
    model = StageChangeBundles.fit(
        tables, min_support=int(min_support), min_bundle_size=2,
        max_bundle_size=3, min_lift=0.5,
        max_candidates=int(max_candidates), allow_truncation=True,
    )
    boundaries = [
        len(model.boundaries(table)) for table in tables
    ] if model.bundles else [0 for _ in tables]
    sizes = Counter(bundle.size for bundle in model.bundles)
    supports = [bundle.episode_support for bundle in model.bundles]
    return dict(
        nodes=int(len(node_ids)),
        bundles=int(len(model.bundles)),
        candidates=int(model.candidates),
        truncated=bool(model.truncated),
        size_counts={
            str(size): int(count) for size, count in sorted(sizes.items())
        },
        support_max=int(max(supports)) if supports else 0,
        support_min=int(min(supports)) if supports else 0,
        boundaries_per_episode=(
            float(np.mean(boundaries)) if boundaries else 0.0
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--decay", type=float, default=0.5)
    parser.add_argument("--min-strength", type=float, default=0.05)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--strength-min-support", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=200000)
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
    seeds = seed_sets(agent, episodes)
    flat = [state for episode in seeds for state in episode]
    strengths = learn_static_strengths(
        flat, graph=getattr(agent, "graph", None),
        min_support=int(args.strength_min_support),
        membership_strength=1.0,
    )
    normalized = learn_static_strengths(
        flat, graph=getattr(agent, "graph", None),
        min_support=int(args.strength_min_support),
        membership_strength=1.0, normalized=True,
    )
    print(
        "static transition probe: seed %d | episodes %d | strengths %s"
        % (args.seed, len(episodes), strengths.snapshot())
    )
    results = []
    variants = (
        ("graph", None),
        ("learned", strengths),
        ("normalized", normalized),
    )
    for variant, overlay in variants:
        for depth in args.depths:
            per_episode = integrated_states(
                agent, seeds, int(depth), strengths=overlay,
                decay=float(args.decay),
                min_strength=float(args.min_strength),
            )
            states = summarize_states(per_episode)
            bundles = bundle_summary(
                per_episode, args.min_support, args.max_candidates,
            )
            row = dict(
                variant=variant, depth=int(depth),
                states=states, bundles=bundles,
            )
            results.append(row)
            print("  %s depth %d: %s | %s" % (
                variant, depth, states, bundles,
            ))
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                steps=int(args.steps), epochs=int(args.epochs),
                max_nodes=int(args.max_nodes), decay=float(args.decay),
                min_strength=float(args.min_strength),
                min_support=int(args.min_support),
                strength_min_support=int(args.strength_min_support),
                max_candidates=int(args.max_candidates),
                depths=[int(value) for value in args.depths],
            ),
            strengths=strengths.snapshot(),
            normalized_strengths=normalized.snapshot(),
            rows=results,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
