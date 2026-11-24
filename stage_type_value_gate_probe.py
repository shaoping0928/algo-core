"""Fixed-budget gate for the stage-type endpoint value term.

The stage-type value is a soft terminal penalty, not a frontier rank hint: the
plan's final state is resolved to a predictive-tree root, the root maps to a
type, and the type's realized value enters the objective with a small weight.
This probe compares the baseline (weight zero) with several weights on the same
worlds, horizon, cap and beam width.  Nothing changes the default path.
"""
import argparse
import json

import numpy as np

from stage_demand_selector import remaining_net_feedback
from stage_type_cluster_probe import (
    integrated,
    key_weights,
    outcome_of,
    stage_seeds,
)
from stage_type_clusters import StageSample
from stage_type_value import StageTypeValue
from static_graph_strengths import learn_static_strengths
from substrate_nodes import (
    LAG,
    _curiosity_off,
    evaluate,
    fit,
    make_stage_value_penalty,
)


def build_value_samples(states, episodes, node_ids, weights):
    weight_vector = np.asarray(
        [weights.get(node, 0.0) for node in node_ids], dtype=float,
    )
    samples = []
    for episode_index, (episode_states, episode) in enumerate(
            zip(states, episodes)):
        for index, state in enumerate(episode_states):
            profile = state.vector(node_ids) * weight_vector
            if not np.any(profile):
                continue
            stage = episode[index]
            samples.append(StageSample(
                episode=episode_index,
                profile=profile,
                outcome=outcome_of(episode, index),
                value=remaining_net_feedback(episode, index),
                terminal=bool(stage.terminal),
                censored=bool(stage.censored),
                key=tuple(stage.key),
            ))
    return samples


def split(samples, modulus=5, held_out=4):
    train = [s for s in samples if s.episode % modulus != held_out]
    validation = [s for s in samples if s.episode % modulus == held_out]
    return train, validation


def run_arm(seed, world, agent, args, stage_value=None, weight=0.0):
    frames = []
    passes = []
    for eval_seed in args.eval_seeds:
        for episode in range(int(args.episodes)):
            env_seed = int(eval_seed) * 10000 + episode
            metrics = {}
            with _curiosity_off(agent):
                length, _actions, _rows = evaluate(
                    seed, world, agent, cap=int(args.cap),
                    horizon=int(args.horizon), lag=LAG,
                    step_stride=2, env_seed=env_seed,
                    decision="search", beam_width=int(args.beam_width),
                    metrics_out=metrics,
                    stage_value=stage_value,
                    stage_value_weight=float(weight),
                )
            frames.append(int(length))
            passes.append(int(metrics.get("passes", 0)))
    return dict(
        weight=float(weight),
        frames=frames,
        mean_frames=float(np.mean(frames)) if frames else 0.0,
        median_frames=float(np.median(frames)) if frames else 0.0,
        total_passes=int(sum(passes)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[120])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--min-similarity", type=float, default=0.7)
    parser.add_argument("--min-type-support", type=int, default=2)
    parser.add_argument("--strength-min-support", type=int, default=2)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
    )
    experience = getattr(agent, "stage_experience", None)
    tree = getattr(agent, "stage_tree", None)
    if experience is None or tree is None:
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
    samples = build_value_samples(states, episodes, node_ids, weights)
    train, validation = split(samples)
    model = StageTypeValue.fit(
        train, validation, min_similarity=float(args.min_similarity),
        min_type_support=int(args.min_type_support),
    )
    root_penalties = model.root_penalties(tree)
    raw_penalty = make_stage_value_penalty(
        agent, root_penalties, np.zeros(LAG), LAG,
    )
    stats = dict(calls=0, candidates=0, non_neutral=0, minimum=1.0,
                 maximum=0.0, distinct_sum=0, distinct_max=0)

    def penalty(states, sequences):
        values = raw_penalty(states, sequences)
        stats["calls"] += 1
        stats["candidates"] += int(len(values))
        stats["non_neutral"] += int(np.sum(values != 0.5))
        stats["minimum"] = min(stats["minimum"], float(np.min(values)))
        stats["maximum"] = max(stats["maximum"], float(np.max(values)))
        distinct = int(len(np.unique(np.round(values, 6))))
        stats["distinct_sum"] += distinct
        stats["distinct_max"] = max(stats["distinct_max"], distinct)
        return values

    print(
        "stage type value gate: seed %d | beam %d | horizon %d | cap %d | "
        "%d eval worlds | value %s"
        % (args.seed, args.beam_width, args.horizon, args.cap,
           len(args.eval_seeds), model.snapshot())
    )
    print("  root penalties: %d roots" % len(root_penalties))
    rows = []
    for weight in args.weights:
        arm = run_arm(
            args.seed, world, agent, args,
            stage_value=penalty if weight > 0.0 else None,
            weight=weight,
        )
        rows.append(arm)
        print(
            "  weight %.2f: frames %.1f (median %.1f) passes %d"
            % (arm["weight"], arm["mean_frames"],
               arm["median_frames"], arm["total_passes"])
        )
    print("  penalty stats: %s" % (stats,))
    if stats["calls"]:
        print(
            "  distinct penalties per call: mean %.2f max %d"
            % (stats["distinct_sum"] / float(stats["calls"]),
               stats["distinct_max"])
        )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                eval_seeds=[int(v) for v in args.eval_seeds],
                episodes=int(args.episodes), steps=int(args.steps),
                epochs=int(args.epochs), max_nodes=int(args.max_nodes),
                depth=int(args.depth), cap=int(args.cap),
                horizon=int(args.horizon), beam_width=int(args.beam_width),
                weights=[float(v) for v in args.weights],
                min_similarity=float(args.min_similarity),
                min_type_support=int(args.min_type_support),
            ),
            value=model.snapshot(),
            root_penalties={
                repr(token): float(value)
                for token, value in sorted(
                    root_penalties.items(), key=lambda item: repr(item[0])
                )
            },
            rows=rows,
            penalty_stats=stats,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
