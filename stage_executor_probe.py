"""Fixed-budget comparison of the stage-level option executor.

The baseline is the ordinary frame planner on the same worlds, horizon and cap.
The treatment replaces frame decisions with stage-level option traces when a
supported option exists, falling back to the same frame planner otherwise.
Nothing here changes the default planner.
"""
import argparse
import json

import numpy as np

from stage_executor import run_stage_executor
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def run_baseline(seed, world, agent, args):
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
                    decision="search", beam_width=0,
                    metrics_out=metrics,
                )
            frames.append(int(length))
            passes.append(int(metrics.get("passes", 0)))
    return dict(
        frames=frames,
        mean_frames=float(np.mean(frames)) if frames else 0.0,
        median_frames=float(np.median(frames)) if frames else 0.0,
        total_passes=int(sum(passes)),
    )


def run_executor(seed, world, agent, args):
    frames = []
    passes = []
    stats = []
    for eval_seed in args.eval_seeds:
        for episode in range(int(args.episodes)):
            env_seed = int(eval_seed) * 10000 + episode
            metrics = {}
            with _curiosity_off(agent):
                length, arm_stats = run_stage_executor(
                    seed, world, agent, cap=int(args.cap),
                    horizon=int(args.horizon), lag=LAG,
                    env_seed=env_seed,
                    terminal_weight=float(args.terminal_weight),
                    duration_weight=float(args.duration_weight),
                    max_transitions=int(args.max_transitions),
                    max_option_actions=(
                        int(args.max_option_actions)
                        if int(args.max_option_actions) > 0 else None
                    ),
                    option_quotient=(
                        agent.option_quotient
                        if args.class_mode
                        and getattr(agent, "option_quotient", None)
                        and agent.option_quotient.accepted else None
                    ),
                    metrics_out=metrics,
                )
            frames.append(int(length))
            passes.append(int(metrics.get("passes", 0)))
            stats.append(arm_stats)
    return dict(
        frames=frames,
        mean_frames=float(np.mean(frames)) if frames else 0.0,
        median_frames=float(np.median(frames)) if frames else 0.0,
        total_passes=int(sum(passes)),
        stats=stats,
        selected=float(np.mean([s["selected"] for s in stats]))
        if stats else 0.0,
        fallback=float(np.mean([s["fallback"] for s in stats]))
        if stats else 0.0,
        aborted=float(np.mean([s["aborted"] for s in stats]))
        if stats else 0.0,
        option_share=float(np.mean([s["option_share"] for s in stats]))
        if stats else 0.0,
        option_length_mean=float(np.mean([
            s["option_length_mean"] for s in stats
        ])) if stats else 0.0,
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
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-transitions", type=int, default=1)
    parser.add_argument("--max-option-actions", type=int, default=0)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--duration-weight", type=float, default=0.0)
    parser.add_argument("--class-mode", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
    )
    if getattr(agent, "stage_outcome_memory", None) is None:
        raise SystemExit("the stage memory did not fit")
    print(
        "stage executor probe: seed %d | cap %d | horizon %d | "
        "%d eval worlds | terminal %.2f duration %.2f class %s"
        % (args.seed, args.cap, args.horizon, len(args.eval_seeds),
           args.terminal_weight, args.duration_weight, args.class_mode)
    )
    baseline = run_baseline(args.seed, world, agent, args)
    print(
        "  baseline: frames %.1f (median %.1f) passes %d"
        % (baseline["mean_frames"], baseline["median_frames"],
           baseline["total_passes"])
    )
    treatment = run_executor(args.seed, world, agent, args)
    print(
        "  executor: frames %.1f (median %.1f) passes %d | "
        "options %.1f/ep, share %.2f, len %.1f, aborted %.1f"
        % (
            treatment["mean_frames"], treatment["median_frames"],
            treatment["total_passes"], treatment["selected"],
            treatment["option_share"], treatment["option_length_mean"],
            treatment["aborted"],
        )
    )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                eval_seeds=[int(v) for v in args.eval_seeds],
                episodes=int(args.episodes), steps=int(args.steps),
                epochs=int(args.epochs), max_nodes=int(args.max_nodes),
                cap=int(args.cap), horizon=int(args.horizon),
                max_transitions=int(args.max_transitions),
                max_option_actions=int(args.max_option_actions),
                terminal_weight=float(args.terminal_weight),
                duration_weight=float(args.duration_weight),
                class_mode=bool(args.class_mode),
            ),
            baseline=baseline,
            executor=treatment,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
