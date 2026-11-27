"""Fixed-budget diagnostic for the online stage-demand rank hint.

The probe trains one fast runtime with the stage layer enabled, then replays
the same evaluation worlds twice at the same beam width, horizon and frame cap:
once without the hint and once with the online provider.  Because the rank
channel only changes intermediate frontier order, the final plan is still
scored and selected by the independent objective; this probe reports whether
that ordering changed behaviour (frames, progress) and how often the provider
could speak (supported / UNKNOWN reasons), plus the search's candidate
expansion.

The optional frontier trace tags candidate states with the fitted stage
resolver and reports whether the demanded root appears in candidates retained
by the beam.  It is diagnostic only and does not affect ranking or scoring.
"""
import argparse
import json

import numpy as np

from stage_demand_heuristic import StageDemand
from stage_online import OnlineStageDemandProvider
from substrate_nodes import (
    LAG, _curiosity_off, evaluate, fit, history_features,
)


def run_arm(seed, world, agent, args, provider=None, rank_stats=None):
    frames = []
    passes = []
    candidates = []
    decisions = []
    failure_postmortem = []
    target_frontier = dict(
        demands=0,
        candidate_decisions=0,
        retained_decisions=0,
        final_candidate_decisions=0,
        final_retained_decisions=0,
        selected_supported_decisions=0,
        selected_target_decisions=0,
        candidate_states=0,
        resolver_supported_states=0,
        target_states=0,
        retained_target_states=0,
        final_target_states=0,
        final_retained_target_states=0,
        checkpoints={},
        rank_group_retention={},
    )
    for eval_seed in args.eval_seeds:
        for episode in range(int(args.episodes)):
            env_seed = int(eval_seed) * 10000 + episode
            plans = []
            metrics = {}
            decision_frontiers = {}

            def trace_frontier(batch):
                demand = batch.get("demand")
                resolver = getattr(agent, "stage_resolver", None)
                tree = getattr(agent, "stage_tree", None)
                if demand is None or resolver is None or tree is None:
                    return
                step = int(batch["step"])
                states = np.asarray(batch["states"], dtype=float)
                sequences = np.asarray(batch["sequences"], dtype=float)
                prefix = np.asarray(
                    batch["pending_prefix"], dtype=float
                ).reshape(-1)
                history = history_features(
                    sequences, step + 1, prefix,
                    action_dim=LAG + 1,
                )
                pending = history[:, :LAG][:, ::-1] if LAG else np.zeros(
                    (len(states), 0)
                )
                selectable = np.asarray(batch["selectable"], dtype=bool)
                resolved_features = [None] * len(states)
                supported_mask = np.zeros(len(states), dtype=bool)
                for index, (state, queue) in enumerate(zip(states, pending)):
                    resolved = resolver.resolve(
                        agent, state[:int(agent.obs_dim)], queue,
                    )
                    if resolved.get("status") != "supported":
                        continue
                    chain = tree.chain(resolved["key"])
                    if not chain:
                        continue
                    supported_mask[index] = True
                    resolved_features[index] = (
                        ("tree", str(chain[-1])),
                    )
                retained = np.asarray(batch["retained"], dtype=bool)
                rank_groups = batch.get("rank_groups")
                if rank_groups is not None:
                    selectable_mask = np.asarray(
                        batch["selectable"], dtype=bool
                    )
                    for group in set(
                            value for value in rank_groups
                            if value is not None):
                        mask = np.asarray(
                            [value == group for value in rank_groups],
                            dtype=bool,
                        )
                        key = repr(group)
                        summary = target_frontier[
                            "rank_group_retention"
                        ].setdefault(
                            key,
                            dict(
                                frontiers=0,
                                candidate_frontiers=0,
                                retained_frontiers=0,
                                candidate_states=0,
                                retained_states=0,
                            ),
                        )
                        candidates_for_group = mask & selectable_mask
                        retained_for_group = mask & retained
                        summary["frontiers"] += 1
                        summary["candidate_frontiers"] += int(
                            candidates_for_group.any()
                        )
                        summary["retained_frontiers"] += int(
                            retained_for_group.any()
                        )
                        summary["candidate_states"] += int(
                            candidates_for_group.sum()
                        )
                        summary["retained_states"] += int(
                            retained_for_group.sum()
                        )
                targets = tuple(getattr(
                    demand, "target_stages", (demand.target_stage,),
                ))
                for target in targets:
                    target = tuple(target)
                    target_mask = np.asarray([
                        feature == target for feature in resolved_features
                    ], dtype=bool)
                    selectable_target = target_mask & selectable
                    key = (int(batch["frame"]), target)
                    row = decision_frontiers.setdefault(
                        key,
                        dict(
                            candidate_any=False,
                            retained_any=False,
                            candidate_final=False,
                            retained_final=False,
                            selected_supported=False,
                            selected_target=False,
                            pending_prefix=prefix.copy(),
                            candidate_states=0,
                            resolver_supported_states=0,
                            target_states=0,
                            retained_target_states=0,
                            final_target_states=0,
                            final_retained_target_states=0,
                            checkpoints={},
                        ),
                    )
                    has_target = bool(selectable_target.any())
                    has_retained_target = bool(
                        (selectable_target & retained).any()
                    )
                    row["candidate_any"] |= has_target
                    row["retained_any"] |= has_retained_target
                    row["candidate_states"] += len(states)
                    row["resolver_supported_states"] += int(
                        supported_mask.sum()
                    )
                    row["target_states"] += int(selectable_target.sum())
                    row["retained_target_states"] += int(
                        (selectable_target & retained).sum()
                    )
                    checkpoint = row["checkpoints"].setdefault(
                        step,
                        dict(
                            candidate=False,
                            retained=False,
                            candidate_states=0,
                            retained_states=0,
                        ),
                    )
                    checkpoint["candidate"] |= has_target
                    checkpoint["retained"] |= has_retained_target
                    checkpoint["candidate_states"] += int(
                        selectable_target.sum()
                    )
                    checkpoint["retained_states"] += int(
                        (selectable_target & retained).sum()
                    )
                    if step == int(args.horizon) - 1:
                        row["candidate_final"] = has_target
                        row["retained_final"] = has_retained_target
                        row["final_target_states"] += int(
                            selectable_target.sum()
                        )
                        row["final_retained_target_states"] += int(
                            (selectable_target & retained).sum()
                        )

            with _curiosity_off(agent):
                length, actual_actions, actual_rows = evaluate(
                    seed, world, agent, cap=int(args.cap),
                    horizon=int(args.horizon), lag=LAG,
                    step_stride=2, env_seed=env_seed,
                    decision="search", beam_width=int(args.beam_width),
                    record=True,
                    plan_out=plans, metrics_out=metrics,
                    stage_demand_provider=provider,
                    rank_stats=rank_stats,
                    rank_mode=args.rank_mode,
                    rank_margin=args.rank_margin,
                    search_chunk_size=args.search_chunk_size,
                    rank_group_quota=args.rank_group_quota,
                    frontier_callback=(
                        trace_frontier if provider is not None else None
                    ),
                )
            terminal_index = next(
                (
                    index for index, transition in enumerate(actual_rows)
                    if bool(transition.get("terminal", False))
                ),
                None,
            )
            plans_by_frame = {
                int(plan["frame"]): plan for plan in plans
            }
            for (frame, target), row in decision_frontiers.items():
                plan = plans_by_frame.get(int(frame))
                if plan is None:
                    continue
                selected_actions = np.asarray(
                    plan["actions"], dtype=float
                ).reshape(1, -1)
                selected_history = history_features(
                    selected_actions, int(args.horizon),
                    row["pending_prefix"], action_dim=LAG + 1,
                )
                selected_pending = (
                    selected_history[:, :LAG][:, ::-1][0]
                    if LAG else np.zeros(0, dtype=float)
                )
                resolved = agent.stage_resolver.resolve(
                    agent,
                    np.asarray(plan["final"], dtype=float)[
                        :int(agent.obs_dim)
                    ],
                    selected_pending,
                )
                if resolved.get("status") != "supported":
                    continue
                chain = agent.stage_tree.chain(resolved["key"])
                if not chain:
                    continue
                row["selected_supported"] = True
                row["selected_target"] = (
                    (("tree", str(chain[-1])),) == target
                )
            if terminal_index is not None and not int(
                    metrics.get("passes", 0)):
                for (frame, target), row in decision_frontiers.items():
                    distance = int(terminal_index) - int(frame)
                    if distance < 0 or distance >= int(args.horizon):
                        continue
                    if frame >= len(actual_rows):
                        continue
                    plan = plans_by_frame.get(int(frame), {})
                    if row["selected_target"]:
                        reason = "selected_target_but_episode_terminated"
                        hypothesis = (
                            "audit transition-model and action-latency error"
                        )
                    elif row["retained_final"]:
                        reason = "target_retained_but_not_selected"
                        hypothesis = (
                            "test an intent-conditioned final candidate cohort"
                        )
                    elif row["candidate_final"]:
                        reason = "target_pruned_at_final_checkpoint"
                        hypothesis = (
                            "reserve a demand-guided beam slot"
                        )
                    elif row["candidate_any"]:
                        reason = "target_lost_before_final_checkpoint"
                        hypothesis = (
                            "inspect the first checkpoint where target vanishes"
                        )
                    else:
                        reason = "target_never_entered_frontier"
                        hypothesis = (
                            "review target-feature association and support"
                        )
                    transition = actual_rows[int(frame)]
                    # #region debug-point E:lagged-transition-residual
                    source_obs = np.asarray(
                        transition["obs"], dtype=float
                    ).reshape(-1)
                    next_obs = np.asarray(
                        transition["next_obs"], dtype=float
                    ).reshape(-1)
                    actual_history = np.asarray(
                        transition["history"], dtype=float
                    ).reshape(-1)
                    predicted_next = np.asarray(
                        world.predict(
                            source_obs.reshape(1, -1),
                            actual_history.reshape(1, -1),
                        ),
                        dtype=float,
                    ).reshape(-1)
                    scale = np.maximum(
                        np.abs(np.asarray(agent.scales[:len(next_obs)])),
                        1e-8,
                    )
                    normalized_transition_error = float(np.sqrt(np.mean(
                        np.square(
                            (predicted_next[:len(next_obs)] - next_obs)
                            / scale
                        )
                    )))
                    # #endregion
                    failure_postmortem.append(dict(
                        eval_seed=int(eval_seed),
                        episode=int(episode),
                        terminal_frame=int(terminal_index),
                        source_frame=int(frame),
                        frames_to_terminal=int(distance),
                        target_stage=[
                            list(token) for token in target
                        ],
                        checkpoints=[
                            dict(
                                step=int(step),
                                candidate=bool(checkpoint["candidate"]),
                                retained=bool(checkpoint["retained"]),
                                candidate_states=int(
                                    checkpoint["candidate_states"]
                                ),
                                retained_states=int(
                                    checkpoint["retained_states"]
                                ),
                            )
                            for step, checkpoint in sorted(
                                row["checkpoints"].items()
                            )
                        ],
                        selected_target=bool(row["selected_target"]),
                        selected_plan_first_action=(
                            float(np.asarray(plan["actions"])[0])
                            if plan else None
                        ),
                        selected_objective_score=(
                            plan.get("objective_score") if plan else None
                        ),
                        selected_rank_credit=(
                            float(plan.get("rank_credit", 0.0))
                            if plan else None
                        ),
                        selected_rank_group=(
                            plan.get("rank_group") if plan else None
                        ),
                        actual_action=(
                            float(actual_actions[int(frame)])
                            if int(frame) < len(actual_actions) else None
                        ),
                        source_obs=source_obs.tolist(),
                        next_obs=next_obs.tolist(),
                        predicted_next_obs=predicted_next.tolist(),
                        normalized_transition_error=(
                            normalized_transition_error
                        ),
                        actual_signals=np.asarray(
                            transition["signals"], dtype=float
                        ).tolist(),
                        reason=reason,
                        repair_hypothesis=hypothesis,
                    ))
            for row in decision_frontiers.values():
                target_frontier["demands"] += 1
                target_frontier["candidate_decisions"] += int(
                    row["candidate_any"]
                )
                target_frontier["retained_decisions"] += int(
                    row["retained_any"]
                )
                target_frontier["final_candidate_decisions"] += int(
                    row["candidate_final"]
                )
                target_frontier["final_retained_decisions"] += int(
                    row["retained_final"]
                )
                target_frontier["selected_supported_decisions"] += int(
                    row["selected_supported"]
                )
                target_frontier["selected_target_decisions"] += int(
                    row["selected_target"]
                )
                for key in (
                    "candidate_states", "resolver_supported_states",
                    "target_states", "retained_target_states",
                    "final_target_states", "final_retained_target_states",
                ):
                    target_frontier[key] += int(row[key])
                for step, checkpoint in row["checkpoints"].items():
                    summary = target_frontier["checkpoints"].setdefault(
                        int(step),
                        dict(
                            demands=0,
                            candidate_decisions=0,
                            retained_decisions=0,
                            candidate_states=0,
                            retained_states=0,
                        ),
                    )
                    summary["demands"] += 1
                    summary["candidate_decisions"] += int(
                        checkpoint["candidate"]
                    )
                    summary["retained_decisions"] += int(
                        checkpoint["retained"]
                    )
                    summary["candidate_states"] += int(
                        checkpoint["candidate_states"]
                    )
                    summary["retained_states"] += int(
                        checkpoint["retained_states"]
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
        target_frontier=target_frontier,
        failure_postmortem=failure_postmortem,
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
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-episodes", type=int, default=2)
    parser.add_argument("--dagger-cap", type=int, default=1200)
    parser.add_argument("--dagger-frames", type=int, default=0)
    parser.add_argument("--dagger-horizon", type=int, default=10)
    parser.add_argument("--stage-min-support", type=int, default=2)
    parser.add_argument("--stage-min-episodes", type=int, default=2)
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument(
        "--max-strategy-proposals", type=int, default=3,
        help="maximum supported successor proposals sharing one search",
    )
    parser.add_argument(
        "--rank-feature-source",
        choices=("root", "prototype", "hybrid"),
        default="root",
        help="candidate feature used by the rank hint: resolved stage root "
             "or the runtime prototype; hybrid combines branch roots and "
             "prototype features",
    )
    parser.add_argument("--min-target-probability", type=float, default=0.0)
    parser.add_argument("--min-prediction-support", type=int, default=0)
    parser.add_argument("--rank-mode", choices=("order", "tie"),
                        default="order")
    parser.add_argument("--rank-margin", type=float, default=0.0)
    parser.add_argument("--rank-group-quota", type=int, default=1)
    parser.add_argument(
        "--search-chunk-size", type=int, default=0,
        help="prune the beam every N frames; 0 keeps per-frame pruning",
    )
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if args.beam_width < 1:
        raise SystemExit("beam width must be positive for the rank channel")
    if args.max_strategy_proposals < 1:
        raise SystemExit("maximum strategy proposals must be positive")
    if args.rank_group_quota < 0:
        raise SystemExit("rank group quota must be non-negative")
    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        dagger_rounds=int(args.dagger_rounds),
        dagger_episodes=int(args.dagger_episodes),
        dagger_cap=int(args.dagger_cap),
        dagger_frames=int(args.dagger_frames),
        dagger_horizon=int(args.dagger_horizon),
        stage_memory=True,
        stage_memory_feature_source=(
            "hybrid" if args.rank_feature_source == "hybrid"
            else "prototype"
        ),
        stage_demand_min_support=int(args.stage_min_support),
        stage_demand_min_episodes=int(args.stage_min_episodes),
    )
    if getattr(agent, "stage_tree", None) is None:
        raise SystemExit("stage tree was not fitted (not enough episodes)")
    agent.stage_rank_feature_source = args.rank_feature_source
    provider = OnlineStageDemandProvider(
        agent, rank_feature_source=args.rank_feature_source,
        min_target_probability=args.min_target_probability,
        min_prediction_support=args.min_prediction_support,
        max_strategy_proposals=args.max_strategy_proposals,
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
    provider_report = provider.snapshot()
    counts = provider_report["counts"]
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
    frontier = ranked["target_frontier"]
    print(
        "  proposal frontier: candidate %d/%d, retained %d/%d, "
        "final-retained %d/%d, selected endpoint %d/%d "
        "(%d supported); mapped %d/%d candidate states"
        % (
            frontier["candidate_decisions"], frontier["demands"],
            frontier["retained_decisions"], frontier["demands"],
            frontier["final_retained_decisions"], frontier["demands"],
            frontier["selected_target_decisions"], frontier["demands"],
            frontier["selected_supported_decisions"],
            frontier["resolver_supported_states"],
            frontier["candidate_states"],
        )
    )
    validation_counts = {
        key: value for key, value in counts.items()
        if key.startswith("intent_validation_")
    }
    credit_counts = {
        key: value for key, value in counts.items()
        if key.startswith("strategy_credit_")
    }
    print(
        "  strategy validation: %s | credit use %s | bank %s"
        % (
            validation_counts or "none",
            credit_counts or "none",
            provider_report["strategy_credit"],
        )
    )
    if frontier["checkpoints"]:
        print(
            "  chunk checkpoints: %s"
            % " | ".join(
                "step %d candidate %d/%d retained %d/%d (%d->%d states)"
                % (
                    step,
                    values["candidate_decisions"],
                    values["demands"],
                    values["retained_decisions"],
                    values["demands"],
                    values["candidate_states"],
                    values["retained_states"],
                )
                for step, values in sorted(
                    frontier["checkpoints"].items()
                )
            )
        )
    if frontier["rank_group_retention"]:
        print(
            "  rank-group retention: %s"
            % frontier["rank_group_retention"]
        )
    postmortem = ranked["failure_postmortem"]
    if postmortem:
        reasons = {}
        for event in postmortem:
            reasons[event["reason"]] = reasons.get(event["reason"], 0) + 1
        print(
            "  failure backtrace: %d preterminal proposal checks; causes %s"
            % (
                len(postmortem),
                ", ".join(
                    "%s=%d" % item for item in sorted(reasons.items())
                ),
            )
        )
        for event in postmortem[:3]:
            print(
                "    seed %d frame %d terminal-in %d target %s reason %s "
                "action %s objective %s rank-credit %s checkpoints %s"
                % (
                    event["eval_seed"], event["source_frame"],
                    event["frames_to_terminal"], event["target_stage"],
                    event["reason"], event["actual_action"],
                    event["selected_objective_score"],
                    event["selected_rank_credit"],
                    [
                        (item["step"], item["candidate"], item["retained"])
                        for item in event["checkpoints"]
                    ],
                )
            )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                eval_seeds=[int(value) for value in args.eval_seeds],
                episodes=int(args.episodes), steps=int(args.steps),
                epochs=int(args.epochs), max_nodes=int(args.max_nodes),
                coverage=float(args.coverage), cap=int(args.cap),
                dagger_rounds=int(args.dagger_rounds),
                dagger_episodes=int(args.dagger_episodes),
                dagger_cap=int(args.dagger_cap),
                dagger_frames=int(args.dagger_frames),
                dagger_horizon=int(args.dagger_horizon),
                horizon=int(args.horizon), lag=LAG, beam_width=int(
                    args.beam_width
                ),
                stage_min_support=int(args.stage_min_support),
                stage_min_episodes=int(args.stage_min_episodes),
                rank_feature_source=args.rank_feature_source,
                min_target_probability=float(args.min_target_probability),
                min_prediction_support=int(args.min_prediction_support),
                rank_mode=args.rank_mode,
                rank_margin=float(args.rank_margin),
                search_chunk_size=int(args.search_chunk_size),
                rank_group_quota=int(args.rank_group_quota),
                max_strategy_proposals=int(args.max_strategy_proposals),
            ),
            baseline=baseline,
            ranked=ranked,
            provider=dict(
                requests=int(requests), supported=supported,
                unknown=unknown,
                feature_source=provider_report["feature_source"],
                rank_feature_source=provider_report["rank_feature_source"],
                min_target_probability=provider_report[
                    "min_target_probability"
                ],
                min_prediction_support=provider_report[
                    "min_prediction_support"
                ],
                strategy_credit=provider_report["strategy_credit"],
                last_validation=provider_report["last_validation"],
                validation_counts=validation_counts,
                credit_counts=credit_counts,
                max_strategy_proposals=provider_report[
                    "max_strategy_proposals"
                ],
            ),
            rank_stats={
                str(key): int(value)
                for key, value in sorted(rank_stats.items())
            },
            target_frontier=ranked["target_frontier"],
            stage_tree=agent.stage_tree.snapshot(),
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
