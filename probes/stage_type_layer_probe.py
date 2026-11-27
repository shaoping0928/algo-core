"""Diagnostic for similarity layers over integrated stage types.

Builds the same normalized ``E(s)`` samples as the stage-type probe, builds the
full complete-linkage merge tree, and reports the held-out frontier chosen for a
sweep of similarity floors and parsimony tolerances.  Nothing here changes a
decision.
"""
import argparse
import json

import numpy as np

from stage_type_cluster_probe import (
    build_samples,
    integrated,
    key_weights,
    split,
    stage_seeds,
)
from stage_type_layers import StageTypeLayers
from static_graph_strengths import learn_static_strengths
from substrate_nodes import fit


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
                        default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--tolerances", type=float, nargs="+",
                        default=[0.01, 0.02])
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-layer-support", type=int, default=2)
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
        "stage type layer probe: seed %d | episodes %d | nodes %d | "
        "samples %d train %d validation %d"
        % (args.seed, len(episodes), len(node_ids), len(samples),
           len(train), len(validation))
    )
    rows = []
    for similarity in args.min_similarities:
        for tolerance in args.tolerances:
            model = StageTypeLayers.fit(
                train, validation, min_similarity=float(similarity),
                min_coverage=float(args.min_coverage),
                min_gain=float(args.min_gain),
                tolerance=float(tolerance),
                min_layer_support=int(args.min_layer_support),
            )
            snapshot = model.snapshot()
            snapshot["min_similarity"] = float(similarity)
            snapshot["tolerance"] = float(tolerance)
            rows.append(snapshot)
            print(
                "  similarity %.2f tolerance %.2f: layers %d | "
                "coverage %.3f | gain %+.3f | level %.3f | accepted %s"
                % (
                    similarity, tolerance, snapshot["layers"],
                    snapshot["coverage"], snapshot["predictive_gain"],
                    snapshot["level"], snapshot["accepted"],
                )
            )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                steps=int(args.steps), epochs=int(args.epochs),
                max_nodes=int(args.max_nodes), depth=int(args.depth),
                decay=float(args.decay),
                min_strength=float(args.min_strength),
                strength_min_support=int(args.strength_min_support),
                min_gain=float(args.min_gain),
                min_coverage=float(args.min_coverage),
                min_layer_support=int(args.min_layer_support),
                min_similarities=[float(v) for v in args.min_similarities],
                tolerances=[float(v) for v in args.tolerances],
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
