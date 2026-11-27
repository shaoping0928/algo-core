"""Episode-held-out audit of the predictive-equivalence stage tree.

Train episodes build the stage-type leaves, validation episodes decide which
leaves may be pooled, and test episodes are scored untouched.  Three readouts
share the same test stages:

* ``leaf``: exact stage type, falling back to the all-stage base when the leaf
  lacks support;
* ``tree``: the same support gate, but first walking the accepted merged
  ancestors;
* ``base``: the all-stage profile only.

Scored channels are the realized categorical outcome (sibling key or
terminal/censoring), log-duration and feedback.  The report includes top-1
successor accuracy, per-level usage, the merge log with validation deltas, and
the count of test stage keys the tree never saw.  Diagnostic only; no planner
path consumes the tree.
"""
import argparse
import json
from collections import Counter, OrderedDict

from predictive_stage_tree import PredictiveStageTree, stage_outcome
from stage_experience import StageExperienceBatch
from substrate_generic import collect
from substrate_nodes import LAG


class FeaturelessAgent:
    """The tree probe needs rows and a lag, not the runtime's feature masks."""

    def __init__(self, lag=LAG):
        self.lag = int(lag)


def episode_groups(rows):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(int(row.get("episode", 0)), []).append(row)
    return grouped


def split_rows(rows, folds=5):
    grouped = episode_groups(rows)
    train, validation, test = [], [], []
    for index, episode in enumerate(sorted(grouped)):
        bucket = index % int(folds)
        target = (
            train if bucket < max(1, int(folds) - 2)
            else (validation if bucket < int(folds) - 1 else test)
        )
        target.extend(grouped[episode])
    return train, validation, test


def evaluate(tree, episodes, args, ancestors):
    totals = dict(
        n=0, scored=0, joint=0.0, categorical=0.0, duration=0.0,
        feedback=0.0, top1=0,
    )
    levels = Counter()
    for stages in episodes:
        stages = list(stages)
        for index, stage in enumerate(stages):
            outcome = stage_outcome(stages, index)
            totals["n"] += 1
            components = tree.components(
                stage, outcome,
                min_support=args.min_support,
                min_episodes=args.min_episodes,
                ancestors=ancestors,
            )
            prediction = tree.predict(
                stage.key,
                min_support=args.min_support,
                min_episodes=args.min_episodes,
                ancestors=ancestors,
            )
            if (components is None
                    or prediction["status"] != "supported"):
                levels["unknown"] += 1
                continue
            totals["scored"] += 1
            totals["joint"] += components["joint"]
            totals["categorical"] += components["categorical"]
            totals["duration"] += components["duration"]
            totals["feedback"] += components["feedback"]
            probabilities = prediction["probabilities"]
            best = max(
                probabilities,
                key=lambda candidate: (probabilities[candidate], repr(candidate)),
            )
            totals["top1"] += int(best == outcome)
            levels[prediction["level"]] += 1
    n = totals["n"]
    scored = totals["scored"]
    return dict(
        n=int(n),
        scored=int(scored),
        coverage=(float(scored) / n if n else 0.0),
        joint_nll=(totals["joint"] / scored if scored else None),
        categorical_nll=(
            totals["categorical"] / scored if scored else None
        ),
        duration_nll=(totals["duration"] / scored if scored else None),
        feedback_nll=(totals["feedback"] / scored if scored else None),
        top1=(float(totals["top1"]) / scored if scored else None),
        levels={
            str(level): int(count) for level, count in levels.items()
        },
    )


def base_readout(tree, episodes, args):
    """The all-stage base profile, independent of the stage key."""
    totals = dict(n=0, scored=0, joint=0.0, top1=0)
    for stages in episodes:
        stages = list(stages)
        for index, stage in enumerate(stages):
            outcome = stage_outcome(stages, index)
            totals["n"] += 1
            components = tree.components(
                stage, outcome, min_support=10 ** 9,
                min_episodes=10 ** 9, ancestors=False,
            )
            prediction = tree.predict(
                stage.key, min_support=10 ** 9,
                min_episodes=10 ** 9, ancestors=False,
            )
            if (components is None
                    or prediction["status"] != "supported"):
                continue
            totals["scored"] += 1
            totals["joint"] += components["joint"]
            probabilities = prediction["probabilities"]
            best = max(
                probabilities,
                key=lambda candidate: (probabilities[candidate], repr(candidate)),
            )
            totals["top1"] += int(best == outcome)
    scored = totals["scored"]
    return dict(
        n=int(totals["n"]),
        scored=int(scored),
        coverage=(
            float(scored) / totals["n"] if totals["n"] else 0.0
        ),
        joint_nll=(totals["joint"] / scored if scored else None),
        top1=(float(totals["top1"]) / scored if scored else None),
    )


def label(key):
    if not key:
        return "root"
    names = []
    for token in key:
        channel, polarity = token
        kind, index = channel
        names.append(
            "%s[%d]%s"
            % (
                "obs" if kind == "observation_change" else "fb",
                int(index), "+" if polarity > 0 else "-",
            )
        )
    return "+".join(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--prior", type=float, default=4.0)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--min-validation-support", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-episodes", type=int, default=2)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    rows, _counts = collect(
        args.seed, steps=args.steps, coverage=args.coverage, lag=LAG,
    )
    train_rows, validation_rows, test_rows = split_rows(rows)
    agent = FeaturelessAgent()
    train = StageExperienceBatch.fit(
        train_rows, agent, lag=LAG, quantile=args.quantile,
    )
    validation = train.transform(validation_rows, agent)
    test = train.transform(test_rows, agent)
    tree = PredictiveStageTree.fit(
        train.episodes, validation.episodes,
        prior=args.prior, tolerance=args.tolerance,
        min_validation_support=args.min_validation_support,
        min_support=args.min_support, min_episodes=args.min_episodes,
    )
    known = sum(
        1 for stages in test.episodes for stage in stages
        if stage.key in tree.leaf_of
    )
    total_test_stages = sum(len(stages) for stages in test.episodes)
    terminal = sum(
        1 for stages in test.episodes for stage in stages
        if stage.terminal or stage.censored
    )
    print(
        "stage tree probe: seed %d  %d rows / %d episodes  "
        "train %d / validation %d / test %d"
        % (
            args.seed, len(rows), len(episode_groups(rows)),
            len(train.episodes), len(validation.episodes),
            len(test.episodes),
        )
    )
    print(
        "  test stages %d, known keys %d (%.3f), terminal/censored %.3f"
        % (
            total_test_stages, known,
            float(known) / total_test_stages if total_test_stages else 0.0,
            float(terminal) / total_test_stages
            if total_test_stages else 0.0,
        )
    )
    tree_report = evaluate(tree, test.episodes, args, ancestors=True)
    leaf_report = evaluate(tree, test.episodes, args, ancestors=False)
    base_report = base_readout(tree, test.episodes, args)

    def fmt(value):
        return "n/a" if value is None else "%.3f" % value

    print(
        "  tree: %d leaves, %d merges, %d roots, max level %d"
        % (
            tree.snapshot()["stage_types"], tree.snapshot()["merges"],
            tree.snapshot()["roots"], tree.snapshot()["max_level"],
        )
    )
    for name, report in (
        ("tree", tree_report), ("leaf", leaf_report),
    ):
        print(
            "  %-5s n=%d scored=%d cov=%.3f joint=%s succ=%s "
            "dur=%s fb=%s top1=%s"
            % (
                name, report["n"], report["scored"], report["coverage"],
                fmt(report["joint_nll"]), fmt(report["categorical_nll"]),
                fmt(report["duration_nll"]), fmt(report["feedback_nll"]),
                fmt(report["top1"]),
            )
        )
    print(
        "  %-5s n=%d scored=%d cov=%.3f joint=%s top1=%s"
        % (
            "base", base_report["n"], base_report["scored"],
            base_report["coverage"], fmt(base_report["joint_nll"]),
            fmt(base_report["top1"]),
        )
    )
    print(
        "  tree levels: %s"
        % ", ".join(
            "%s=%d" % (level, count)
            for level, count in sorted(tree_report["levels"].items())
        )
    )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                steps=int(args.steps), coverage=float(args.coverage),
                lag=LAG, quantile=float(args.quantile),
                prior=float(args.prior), tolerance=float(args.tolerance),
                min_validation_support=int(args.min_validation_support),
                min_support=int(args.min_support),
                min_episodes=int(args.min_episodes),
            ),
            episodes=dict(
                train=len(train.episodes),
                validation=len(validation.episodes),
                test=len(test.episodes),
                test_stages=int(total_test_stages),
                test_known_keys=int(known),
                test_terminal_fraction=(
                    float(terminal) / total_test_stages
                    if total_test_stages else 0.0
                ),
            ),
            tree=tree.snapshot(),
            merge_log=list(tree.merge_log),
            models=dict(
                tree=tree_report, leaf=leaf_report, base=base_report,
            ),
            test_stage_histogram=[
                dict(key=label(key), transitions=int(count))
                for key, count in Counter(
                    stage.key
                    for stages in test.episodes for stage in stages
                ).most_common()
            ],
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
