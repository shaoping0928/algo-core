"""Diagnostic for cross-fold stability selection of stage-type layers.

Builds the normalized ``E(s)`` samples, then repeats an episode-folded
similarity/tolerance sweep under several fold seeds.  The chosen configuration
is the one whose selected layer count is stable across folds and whose held-out
gain stays competitive.  Nothing here changes a decision.
"""
import argparse
import json

import numpy as np

from stage_layer_stability import LayerStabilitySelector
from stage_type_cluster_probe import (
    build_samples,
    integrated,
    key_weights,
    stage_seeds,
)
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-similarities", type=float, nargs="+",
                        default=[0.5, 0.7, 0.9])
    parser.add_argument("--tolerances", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20])
    parser.add_argument("--min-agreement", type=float, default=0.6)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--min-acceptance", type=float, default=0.8)
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
    print(
        "layer stability probe: seed %d | episodes %d | nodes %d | "
        "samples %d"
        % (args.seed, len(episodes), len(node_ids), len(samples))
    )
    results = []
    for fold_seed in args.seeds:
        selector = LayerStabilitySelector.fit(
            samples, folds=int(args.folds), seed=int(fold_seed),
            min_similarities=[float(v) for v in args.min_similarities],
            tolerances=[float(v) for v in args.tolerances],
            min_agreement=float(args.min_agreement),
            min_coverage=float(args.min_coverage),
            min_gain=float(args.min_gain),
            min_acceptance=float(args.min_acceptance),
            min_type_support=2,
        )
        snapshot = selector.snapshot()
        snapshot["fold_seed"] = int(fold_seed)
        snapshot["candidate_rows"] = [
            candidate.snapshot() for candidate in selector.candidates
        ]
        results.append(snapshot)
        chosen = snapshot["chosen"]
        if chosen is None:
            print("  fold seed %d: no stable candidate" % fold_seed)
        else:
            print(
                "  fold seed %d: similarity %.2f tolerance %.2f | "
                "layers %d (agreement %.2f) | coverage %.3f | gain %+.3f"
                % (
                    fold_seed, chosen["min_similarity"],
                    chosen["tolerance"], chosen["mode_layers"],
                    chosen["agreement"], chosen["mean_coverage"],
                    chosen["mean_gain"],
                )
            )
        for row in snapshot["candidate_rows"]:
            print(
                "    sim %.2f tol %.2f: layers %s mode %d agree %.2f "
                "coverage %.3f gain %+.3f"
                % (
                    row["min_similarity"], row["tolerance"],
                    row["layers"], row["mode_layers"], row["agreement"],
                    row["mean_coverage"], row["mean_gain"],
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
                folds=int(args.folds),
                fold_seeds=[int(v) for v in args.seeds],
                min_similarities=[float(v) for v in args.min_similarities],
                tolerances=[float(v) for v in args.tolerances],
                min_agreement=float(args.min_agreement),
                min_coverage=float(args.min_coverage),
                min_gain=float(args.min_gain),
                min_acceptance=float(args.min_acceptance),
            ),
            strengths=strengths.snapshot(),
            nodes=int(len(node_ids)),
            samples=int(len(samples)),
            rows=results,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
