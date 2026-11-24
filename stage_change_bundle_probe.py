"""Diagnostic for learned stage-change bundles over feature-node co-changes.

Builds one static expansion per realized episode from the stage experience's
per-transition feature snapshots, learns recurring co-change bundles, and
reports bundle support, the boundaries they induce, and the intervals between
them next to the existing signal-onset stages.  Nothing here changes a decision.
"""
import argparse
import json

import numpy as np

from stage_change_bundles import (
    StageChangeBundles,
    static_expansion_from_snapshots,
)
from substrate_nodes import fit


def episode_expansions(episodes):
    node_ids = []
    seen = set()
    for episode in episodes:
        for stage in episode:
            for snapshot in stage.feature_snapshots:
                for node in snapshot:
                    node = str(node)
                    if node not in seen:
                        seen.add(node)
                        node_ids.append(node)
    tables = []
    for episode in episodes:
        snapshots = []
        for stage in episode:
            snapshots.extend(stage.feature_snapshots)
        if not snapshots:
            continue
        tables.append(static_expansion_from_snapshots(node_ids, snapshots))
    return tuple(node_ids), tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--supports", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--min-bundle-size", type=int, default=2)
    parser.add_argument("--max-bundle-size", type=int, default=3)
    parser.add_argument("--min-change", type=float, default=0.5)
    parser.add_argument("--min-lift", type=float, default=0.5)
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
    node_ids, tables = episode_expansions(episodes)
    print(
        "stage change bundle probe: seed %d | episodes %d nodes %d "
        "tables %d"
        % (args.seed, len(episodes), len(node_ids), len(tables))
    )
    transitions = sum(table.frames for table in tables)
    detector_stages = sum(len(episode) for episode in episodes)
    print(
        "  transitions %d | detector stages %d"
        % (transitions, detector_stages)
    )

    model = StageChangeBundles.fit(
        tables,
        min_support=int(args.min_support),
        min_bundle_size=int(args.min_bundle_size),
        max_bundle_size=int(args.max_bundle_size),
        min_change=float(args.min_change),
        min_lift=float(args.min_lift),
        max_candidates=int(args.max_candidates),
        allow_truncation=True,
    )
    sweep = []
    for support in args.supports:
        candidate = StageChangeBundles.fit(
            tables,
            min_support=int(support),
            min_bundle_size=int(args.min_bundle_size),
            max_bundle_size=int(args.max_bundle_size),
            min_change=float(args.min_change),
            min_lift=float(args.min_lift),
            max_candidates=int(args.max_candidates),
            allow_truncation=True,
        )
        candidate_boundaries = [
            len(candidate.boundaries(table)) for table in tables
        ]
        sweep.append(dict(
            support=int(support),
            bundles=int(len(candidate.bundles)),
            candidates=int(candidate.candidates),
            truncated=bool(candidate.truncated),
            boundaries_per_episode=(
                float(np.mean(candidate_boundaries))
                if candidate_boundaries else 0.0
            ),
            bundle_sizes=sorted(
                (bundle.size for bundle in candidate.bundles), reverse=True,
            ),
        ))
    print("  support sweep: %s" % (sweep,))
    print("  bundles: %s" % (model.snapshot(),))
    for bundle in model.bundles[:10]:
        print(
            "    %s support %d/%d lift %.3f key %.3f"
            % (bundle.tokens, bundle.episode_support,
               len(tables), bundle.lift, bundle.key_score)
        )
    top_keys = sorted(
        model.key_weights.items(), key=lambda item: (-item[1], item[0]),
    )[:10]
    print("  key nodes: %s" % (top_keys,))

    boundary_counts = []
    interval_counts = []
    durations = []
    for table in tables:
        boundaries = model.boundaries(table)
        segments = model.segments(table)
        boundary_counts.append(len(boundaries))
        interval_counts.append(len(segments))
        durations.extend(segment.duration for segment in segments)
    print(
        "  boundaries/episode %.2f | intervals/episode %.2f | "
        "duration median %.1f"
        % (
            float(np.mean(boundary_counts)) if boundary_counts else 0.0,
            float(np.mean(interval_counts)) if interval_counts else 0.0,
            float(np.median(durations)) if durations else 0.0,
        )
    )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                steps=int(args.steps), epochs=int(args.epochs),
                max_nodes=int(args.max_nodes),
                min_support=int(args.min_support),
                min_bundle_size=int(args.min_bundle_size),
                max_bundle_size=int(args.max_bundle_size),
                min_change=float(args.min_change),
                min_lift=float(args.min_lift),
            ),
            episodes=int(len(episodes)),
            nodes=int(len(node_ids)),
            transitions=int(transitions),
            detector_stages=int(detector_stages),
            bundles=model.snapshot(),
            support_sweep=sweep,
            bundle_rows=[
                dict(
                    tokens=[[token[0], token[1]] for token in bundle.tokens],
                    episode_support=int(bundle.episode_support),
                    transition_support=int(bundle.transition_support),
                    lift=float(bundle.lift),
                    key_score=float(bundle.key_score),
                )
                for bundle in model.bundles[:50]
            ],
            key_nodes=[
                dict(node=node, weight=float(weight))
                for node, weight in top_keys
            ],
            boundaries_per_episode=(
                float(np.mean(boundary_counts)) if boundary_counts else 0.0
            ),
            intervals_per_episode=(
                float(np.mean(interval_counts)) if interval_counts else 0.0
            ),
            duration_median=(
                float(np.median(durations)) if durations else 0.0
            ),
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
