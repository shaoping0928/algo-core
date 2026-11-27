"""Episode-held-out audit of feature -> next-stage demand association.

The demand heuristic is only a rank hint, but it still has to be calibrated:
does a feature source observed before the transition actually rank the stage
that followed above the other supported sibling types?  This probe splits the
passive stream by episode, fits the runtime representation on training
episodes only, fits the key-signal detector and the heuristic on those same
training episodes, and scores held-out test episodes with the frozen objects.

Two feature sources are compared on the same representation:

* ``branch``: the runtime's induced feature roots (the sparse masks used so
  far);
* ``prototype``: the runtime's nearest learned prototype, which is available
  for every candidate state and therefore gives the association a dense,
  task-agnostic token.

Candidates are every stage type seen in training; the heuristic's own support
gates decide which ones are scoreable.  The reported top-1/top-3/MRR are over
test snapshots whose realized target is scoreable, alongside coverage, the
chance baseline, the training-majority baseline, and the realized-vs-other
priority gap.

The default samples pool action options.  Option-conditioned priority is not
evaluated: the exact realized action trace of a stage includes actions taken
after an earlier decision and would leak the future; a valid option arm needs a
decision-time-known prefix, which this sample schema does not carry.
"""
import argparse
import json
from collections import Counter, OrderedDict

from multiview_transition_probe import fit_fold_representation
from predictive_stage_tree import PredictiveStageTree, collapse_stage_episodes
from stage_demand_heuristic import (
    DemandAssociationSample,
    StageDemand,
    StageDemandHeuristic,
)
from stage_experience import StageExperienceBatch
from substrate_generic import collect
from substrate_nodes import LAG


def episode_groups(rows):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(int(row.get("episode", 0)), []).append(row)
    return grouped


def split_rows(rows):
    grouped = episode_groups(rows)
    train, validation, test = [], [], []
    for index, episode in enumerate(sorted(grouped)):
        bucket = index % 5
        if bucket < 3:
            target = train
        elif bucket == 3:
            target = validation
        else:
            target = test
        target.extend(grouped[episode])
    return train, validation, test


def candidate_types_episodes(episodes):
    return sorted(
        {stage.key for stages in episodes for stage in stages},
        key=repr,
    )


def majority_target(samples):
    counts = Counter(sample.destination_stage for sample in samples)
    if not counts:
        return None
    return max(counts, key=lambda key: (counts[key], repr(key)))


def evaluate(heuristic, samples, targets, majority):
    metrics = dict(
        samples=0,
        feature_samples=0,
        candidate_samples=0,
        realized_supported=0,
        top1=0,
        top3=0,
        reciprocal_rank=0.0,
        chance=0.0,
        majority_correct=0,
        candidates=0,
        priority_realized=0.0,
        priority_realized_n=0,
        priority_other=0.0,
        priority_other_n=0,
        no_candidates=0,
    )
    for sample in samples:
        metrics["samples"] += 1
        if not sample.active_features:
            continue
        metrics["feature_samples"] += 1
        scored = []
        for target in targets:
            demand = StageDemand(
                target_stage=target,
                stage_history=sample.stage_history,
                association_action_trace=sample.action_trace,
            )
            result = heuristic.score(sample.active_features, demand)
            if result["status"] == "supported":
                scored.append((target, float(result["priority"])))
        if not scored:
            metrics["no_candidates"] += 1
            continue
        metrics["candidate_samples"] += 1
        metrics["candidates"] += len(scored)
        scored.sort(key=lambda row: (-row[1], repr(row[0])))
        realized = sample.destination_stage
        rank = None
        realized_priority = None
        for position, (target, priority) in enumerate(scored, start=1):
            if target == realized:
                rank = position
                realized_priority = priority
                break
        if rank is None:
            continue
        metrics["realized_supported"] += 1
        metrics["chance"] += 1.0 / len(scored)
        if majority is not None and realized == majority:
            metrics["majority_correct"] += 1
        metrics["top1"] += int(rank == 1)
        metrics["top3"] += int(rank <= 3)
        metrics["reciprocal_rank"] += 1.0 / rank
        metrics["priority_realized"] += realized_priority
        metrics["priority_realized_n"] += 1
        for target, priority in scored:
            if target != realized:
                metrics["priority_other"] += priority
                metrics["priority_other_n"] += 1
    return metrics


def summary(metrics):
    realized = metrics["realized_supported"]
    feature_samples = metrics["feature_samples"]
    return dict(
        samples=int(metrics["samples"]),
        feature_samples=int(feature_samples),
        feature_coverage=(
            float(feature_samples) / metrics["samples"]
            if metrics["samples"] else 0.0
        ),
        candidate_samples=int(metrics["candidate_samples"]),
        no_candidates=int(metrics["no_candidates"]),
        realized_supported=int(realized),
        realized_coverage=(
            float(realized) / metrics["samples"]
            if metrics["samples"] else 0.0
        ),
        top1=(
            float(metrics["top1"]) / realized if realized else None
        ),
        top3=(
            float(metrics["top3"]) / realized if realized else None
        ),
        mrr=(
            float(metrics["reciprocal_rank"]) / realized
            if realized else None
        ),
        chance=(
            float(metrics["chance"]) / realized if realized else None
        ),
        majority=(
            float(metrics["majority_correct"]) / realized
            if realized else None
        ),
        top1_all_features=(
            float(metrics["top1"]) / feature_samples
            if feature_samples else None
        ),
        mean_candidates=(
            float(metrics["candidates"]) / metrics["candidate_samples"]
            if metrics["candidate_samples"] else None
        ),
        mean_priority_realized=(
            float(metrics["priority_realized"])
            / metrics["priority_realized_n"]
            if metrics["priority_realized_n"] else None
        ),
        mean_priority_other=(
            float(metrics["priority_other"])
            / metrics["priority_other_n"]
            if metrics["priority_other_n"] else None
        ),
    )


def demand_samples(episodes, mode):
    if mode == "pooled":
        return DemandAssociationSample.from_stage_episodes(episodes)
    if mode == "prefix":
        return DemandAssociationSample.from_stage_prefixes(episodes)
    raise ValueError("association mode must be pooled or prefix")


def run_arm(train, test, args, mode, context_mode):
    if context_mode == "exact":
        train_episodes = train.episodes
        test_episodes = test.episodes
    else:
        train_episodes = collapse_stage_episodes(
            train.episodes, args._tree, args.tree_collapse,
            args.tree_support, args.tree_episodes, args.tree_level,
        )
        test_episodes = collapse_stage_episodes(
            test.episodes, args._tree, args.tree_collapse,
            args.tree_support, args.tree_episodes, args.tree_level,
        )
    train_samples = demand_samples(train_episodes, mode)
    test_samples = demand_samples(test_episodes, mode)
    heuristic = StageDemandHeuristic(
        min_support=args.min_support,
        min_episodes=args.min_episodes,
        prior=args.prior,
    ).fit(train_samples)
    targets = candidate_types_episodes(train_episodes)
    majority = majority_target(train_samples)
    metrics = evaluate(
        heuristic, test_samples, targets, majority,
    )
    return dict(
        feature_source=args.feature_source,
        association_mode=mode,
        context_mode=context_mode,
        train=dict(
            episodes=len(train_episodes),
            stages=sum(len(stages) for stages in train_episodes),
            demand_samples=len(train_samples),
            stage_types=len(targets),
            majority_target=repr(majority) if majority is not None else None,
            heuristic=heuristic.snapshot(),
        ),
        test=dict(
            episodes=len(test_episodes),
            stages=sum(len(stages) for stages in test_episodes),
            demand_samples=len(test_samples),
        ),
        metrics=summary(metrics),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-episodes", type=int, default=2)
    parser.add_argument("--prior", type=float, default=4.0)
    parser.add_argument(
        "--feature-sources", default="branch,prototype",
        help="comma-separated arms; each uses the same train-only representation",
    )
    parser.add_argument(
        "--association-modes", default="pooled,prefix",
        help="pooled options and/or causal action-prefix options",
    )
    parser.add_argument(
        "--context-modes", default="exact,tree",
        help="exact stage keys and/or predictive-tree node keys",
    )
    parser.add_argument("--tree-prior", type=float, default=4.0)
    parser.add_argument("--tree-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--tree-validation-support", type=int, default=4,
    )
    parser.add_argument("--tree-support", type=int, default=2)
    parser.add_argument("--tree-episodes", type=int, default=2)
    parser.add_argument(
        "--tree-collapse", choices=("support", "level", "root"),
        default="support",
        help="how a stage key maps to a tree node: support gate, fixed "
             "ancestor level, or its component root",
    )
    parser.add_argument("--tree-level", type=int, default=1)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    rows, _counts = collect(
        args.seed, steps=args.steps, coverage=args.coverage, lag=LAG,
    )
    train_rows, validation_rows, test_rows = split_rows(rows)
    agent = fit_fold_representation(
        train_rows, args.seed, LAG, int(args.max_nodes),
    )
    print(
        "stage demand association probe: seed %d  %d rows / %d episodes  "
        "train %d / validation %d / test %d  representation train-only"
        % (
            args.seed, len(rows), len(episode_groups(rows)),
            len(episode_groups(train_rows)),
            len(episode_groups(validation_rows)),
            len(episode_groups(test_rows)),
        )
    )
    arms = []
    context_modes = [
        value.strip() for value in args.context_modes.split(",")
        if value.strip()
    ]
    for context_mode in context_modes:
        if context_mode not in ("exact", "tree"):
            raise SystemExit("unknown context mode: %s" % context_mode)
    for source in args.feature_sources.split(","):
        source = source.strip()
        if source not in ("branch", "prototype"):
            raise SystemExit("unknown feature source: %s" % source)
        args.feature_source = source
        train = StageExperienceBatch.fit(
            train_rows, agent, lag=LAG, quantile=args.quantile,
            feature_source=source,
        )
        validation = train.transform(validation_rows, agent)
        test = train.transform(test_rows, agent)
        tree = None
        if "tree" in context_modes:
            tree = PredictiveStageTree.fit(
                train.episodes, validation.episodes,
                prior=args.tree_prior,
                tolerance=args.tree_tolerance,
                min_validation_support=args.tree_validation_support,
                min_support=args.tree_support,
                min_episodes=args.tree_episodes,
            )
        args._tree = tree
        for context_mode in context_modes:
            for mode in args.association_modes.split(","):
                mode = mode.strip()
                if mode not in ("pooled", "prefix"):
                    raise SystemExit("unknown association mode: %s" % mode)
                arm = run_arm(
                    train, test, args, mode, context_mode,
                )
                arm["train"]["experience"] = train.snapshot()
                arm["test"]["experience"] = test.snapshot()
                if tree is not None:
                    arm["tree"] = tree.snapshot()
                arms.append(arm)
                metrics = arm["metrics"]
                print(
                    "  %-9s/%-6s/%-5s test %d samples  feature cov %.3f  "
                    "realized cov %.3f  top1 %s  top3 %s  mrr %s  "
                    "chance %s  majority %s  priority %s vs %s"
                    % (
                        source, mode, context_mode, metrics["samples"],
                        metrics["feature_coverage"],
                        metrics["realized_coverage"],
                        "n/a" if metrics["top1"] is None
                        else "%.3f" % metrics["top1"],
                        "n/a" if metrics["top3"] is None
                        else "%.3f" % metrics["top3"],
                        "n/a" if metrics["mrr"] is None
                        else "%.3f" % metrics["mrr"],
                        "n/a" if metrics["chance"] is None
                        else "%.3f" % metrics["chance"],
                        "n/a" if metrics["majority"] is None
                        else "%.3f" % metrics["majority"],
                        "n/a" if metrics["mean_priority_realized"] is None
                        else "%+.3f" % metrics["mean_priority_realized"],
                        "n/a" if metrics["mean_priority_other"] is None
                        else "%+.3f" % metrics["mean_priority_other"],
                    )
                )
    print(
        "  note: pooled arms pool action options; prefix arms use only the "
        "actions observed up to each snapshot (causal), with suffix backoff"
    )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                steps=int(args.steps), coverage=float(args.coverage),
                lag=LAG, max_nodes=int(args.max_nodes),
                quantile=float(args.quantile),
                min_support=int(args.min_support),
                min_episodes=int(args.min_episodes),
                prior=float(args.prior),
                context_modes=[
                    value.strip()
                    for value in args.context_modes.split(",")
                    if value.strip()
                ],
                tree_prior=float(args.tree_prior),
                tree_tolerance=float(args.tree_tolerance),
                tree_validation_support=int(args.tree_validation_support),
                tree_support=int(args.tree_support),
                tree_episodes=int(args.tree_episodes),
                tree_collapse=args.tree_collapse,
                tree_level=int(args.tree_level),
            ),
            arms=arms,
            option_conditioning="causal_prefix_with_suffix_backoff",
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
