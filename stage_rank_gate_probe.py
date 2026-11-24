"""Fixed-budget diagnostic for the online stage-demand rank hint.

The probe trains one fast runtime with the stage layer enabled, then replays
the same evaluation worlds twice at the same beam width, horizon and frame cap:
once without the hint and once with the online provider.  Because the rank
channel only changes intermediate frontier order, the final plan is still
scored and selected by the independent objective; this probe reports whether
that ordering changed behaviour (frames, progress) and how often the provider
could speak (supported / UNKNOWN reasons), plus the search's candidate
expansion.

Target-stage path recall needs a stage-tagged frontier log; this first gate
records its absence explicitly instead of inferring it from frames.
"""
import argparse
import json

import numpy as np

from stage_demand_heuristic import StageDemand
from stage_online import OnlineStageDemandProvider
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def run_arm(seed, world, agent, args, provider=None, rank_stats=None):
    frames = []
    passes = []
    candidates = []
    decisions = []
    for eval_seed in args.eval_seeds:
        for episode in range(int(args.episodes)):
            env_seed = int(eval_seed) * 10000 + episode
            plans = []
            metrics = {}
            with _curiosity_off(agent):
                length, _actions, _rows = evaluate(
                    seed, world, agent, cap=int(args.cap),
                    horizon=int(args.horizon), lag=LAG,
                    step_stride=2, env_seed=env_seed,
                    decision="search", beam_width=int(args.beam_width),
                    plan_out=plans, metrics_out=metrics,
                    stage_demand_provider=provider,
                    rank_stats=rank_stats,
                )
            frames.append(int(length))
            passes.append(int(metrics.get("passes", 0)))
            if plans:
                candidates.extend(
                    int(plan.get("candidate_count", 0)) for plan in plans
                )
                decisions.append(len(plans))
    return dict(
        frames=frames,
        passes=passes,
        mean_frames=float(np.mean(frames)) if frames else 0.0,
        median_frames=(
            float(np.median(frames)) if frames else 0.0
        ),
        total_passes=int(sum(passes)),
        mean_candidates=(
            float(np.mean(candidates)) if candidates else 0.0
        ),
        decisions=int(sum(decisions)),
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
    parser.add_argument("--stage-min-support", type=int, default=2)
    parser.add_argument("--stage-min-episodes", type=int, default=2)
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument(
        "--rank-feature-source", choices=("root", "prototype"),
        default="root",
        help="candidate feature used by the rank hint: resolved stage root "
             "or the runtime prototype",
    )
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if args.beam_width < 1:
        raise SystemExit("beam width must be positive for the rank channel")
    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
        stage_demand_min_support=int(args.stage_min_support),
        stage_demand_min_episodes=int(args.stage_min_episodes),
    )
    if getattr(agent, "stage_tree", None) is None:
        raise SystemExit("stage tree was not fitted (not enough episodes)")
    agent.stage_rank_feature_source = args.rank_feature_source
    provider = OnlineStageDemandProvider(
        agent, rank_feature_source=args.rank_feature_source,
    )
    print(
        "stage rank gate probe: seed %d  beam %d  horizon %d  cap %d  "
        "%d eval worlds | stage tree %s"
        % (
            args.seed, args.beam_width, args.horizon, args.cap,
            len(args.eval_seeds), agent.stage_tree.snapshot(),
        )
    )
    baseline = run_arm(args.seed, world, agent, args, provider=None)
    rank_stats = {}
    ranked = run_arm(
        args.seed, world, agent, args, provider=provider,
        rank_stats=rank_stats,
    )
    counts = provider.snapshot()["counts"]
    requests = max(int(counts.get("requests", 0)), 1)
    supported = int(counts.get("supported", 0))
    unknown = {
        key: int(value) for key, value in counts.items()
        if key not in ("requests", "supported")
    }
    print(
        "  baseline frames %.1f (median %.1f) passes %d candidates/dec %.2f"
        % (
            baseline["mean_frames"], baseline["median_frames"],
            baseline["total_passes"], baseline["mean_candidates"],
        )
    )
    print(
        "  ranked   frames %.1f (median %.1f) passes %d candidates/dec %.2f"
        % (
            ranked["mean_frames"], ranked["median_frames"],
            ranked["total_passes"], ranked["mean_candidates"],
        )
    )
    print(
        "  provider: %d/%d decisions supported (%.3f), unknown %s"
        % (
            supported, requests, float(supported) / requests,
            ", ".join(
                "%s=%d" % item for item in sorted(unknown.items())
            ) or "none",
        )
    )
    rank_candidates = int(rank_stats.get("candidates", 0))
    rank_supported = int(rank_stats.get("supported", 0))
    rank_nonzero = int(rank_stats.get("nonzero", 0))
    print(
        "  rank hints: %d candidate states, heuristic supported %d, "
        "nonzero priority %d, unknown %d, feature present/empty %d/%d, "
        "target present %d"
        % (
            rank_candidates, rank_supported, rank_nonzero,
            int(rank_stats.get("unknown", 0)),
            int(rank_stats.get("feature_present", 0)),
            int(rank_stats.get("feature_empty", 0)),
            int(rank_stats.get("target_present", 0)),
        )
    )
    if args.rank_feature_source == "root":
        print(
            "  root heuristic: %s"
            % (agent.stage_root_heuristic.snapshot(),)
        )
        if getattr(provider, "last", None):
            target = provider.last["target"]
            probe_demand = StageDemand(
                target_stage=target,
                stage_history=provider.last["history"],
                association_action_trace=provider.last["prefix"],
            )
            self_score = agent.stage_root_heuristic.score(
                (target[0],), probe_demand,
            )
            print(
                "  self-feature probe: %s support=%s depth=%s"
                % (
                    self_score["status"], self_score.get("support"),
                    self_score.get("context_depth"),
                )
            )
            heuristic = agent.stage_root_heuristic
            root_key = ((), None)
            print(
                "  root table targets: %s"
                % (
                    heuristic._base.get(root_key),
                )
            )
            feature = target[0]
            print(
                "  root target=%r feature=%r feature_count=%d joint=%d"
                % (
                    target, feature,
                    heuristic._feature_counts.get(
                        (root_key, feature), 0
                    ),
                    heuristic._feature_target_counts.get(
                        (root_key, feature, target), 0
                    ),
                )
            )
    print(
        "  target-path recall: not measured (needs a stage-tagged frontier log)"
    )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                eval_seeds=[int(value) for value in args.eval_seeds],
                episodes=int(args.episodes), steps=int(args.steps),
                epochs=int(args.epochs), max_nodes=int(args.max_nodes),
                coverage=float(args.coverage), cap=int(args.cap),
                horizon=int(args.horizon), lag=LAG, beam_width=int(
                    args.beam_width
                ),
                stage_min_support=int(args.stage_min_support),
                stage_min_episodes=int(args.stage_min_episodes),
                rank_feature_source=args.rank_feature_source,
            ),
            baseline=baseline,
            ranked=ranked,
            provider=dict(
                requests=int(requests), supported=supported,
                unknown=unknown,
                feature_source=provider.snapshot()["feature_source"],
                rank_feature_source=provider.snapshot()[
                    "rank_feature_source"
                ],
            ),
            rank_stats={
                str(key): int(value)
                for key, value in sorted(rank_stats.items())
            },
            stage_tree=agent.stage_tree.snapshot(),
            target_path_recall=None,
            target_path_recall_note=(
                "requires a stage-tagged frontier log; not inferred from "
                "frames"
            ),
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
