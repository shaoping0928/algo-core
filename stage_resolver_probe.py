"""Episode-held-out audit of the continuous-state stage resolver.

The resolver is only useful if a state from an episode it never saw maps to the
stage that episode actually realized, and if it says UNKNOWN when it cannot.
This probe makes that check with fold-refitted representations: for each fold
the runtime's node vocabulary, the key-signal detector, and the prototype ->
stage-type table are fit on the other folds only.  The held-out episodes are
then transformed and resolved with those frozen objects.

Reported per fold and in aggregate: supported coverage, top-1 accuracy on the
supported rows, the best constant-key baseline (the training fold's most common
stage type), UNKNOWN reason counts, confusion pairs, and the realized stage-key
histogram.  This is a diagnostic; it is not a decision gate and nothing here is
wired into the planner.
"""
import argparse
import json
from collections import Counter, OrderedDict

from multiview_transition_probe import fit_fold_representation
from stage_experience import StageExperienceBatch
from stage_state_resolver import StageStateResolver
from substrate_generic import collect
from substrate_nodes import LAG


def _label(key):
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
                int(index),
                "+" if polarity > 0 else "-",
            )
        )
    return "+".join(names)


def episode_groups(rows):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(int(row.get("episode", 0)), []).append(row)
    return grouped


def stage_histogram(experience):
    counts = Counter()
    for stages in experience.episodes:
        for stage in stages:
            counts[stage.key] += int(round(float(stage.duration)))
    return counts


def fold_report(rows, seed, fold, folds, args):
    groups = episode_groups(rows)
    episode_ids = list(groups)
    fold_of = {
        episode: index % int(folds)
        for index, episode in enumerate(episode_ids)
    }
    train_rows = [
        row for episode, items in groups.items()
        if fold_of[episode] != fold for row in items
    ]
    test_rows = [
        row for episode, items in groups.items()
        if fold_of[episode] == fold for row in items
    ]
    if not train_rows or not test_rows:
        return None
    agent = fit_fold_representation(
        train_rows, seed, LAG, int(args.max_nodes),
    )
    experience = StageExperienceBatch.fit(
        train_rows, agent, lag=LAG, quantile=args.quantile,
    )
    resolver = StageStateResolver.fit_rows(
        train_rows, agent, experience=experience,
        min_support=args.min_support,
        min_episodes=args.min_episodes,
        min_probability=args.min_probability,
        min_margin=args.min_margin,
    )
    heldout = experience.transform(test_rows, agent)
    transitions = list(heldout.stage_transitions(test_rows, agent))
    train_counts = stage_histogram(experience)
    majority_key = (
        train_counts.most_common(1)[0][0] if train_counts else ()
    )
    supported = 0
    correct = 0
    base_correct = 0
    base_correct_supported = 0
    reasons = Counter()
    confusions = Counter()
    per_key = {}
    for transition in transitions:
        realized = transition.stage.key
        result = resolver.resolve(
            agent, transition.observation, transition.pending,
        )
        base_correct += int(realized == majority_key)
        stats = per_key.setdefault(realized, dict(
            n=0, supported=0, correct=0,
        ))
        stats["n"] += 1
        if result["status"] != "supported":
            reasons[result["reason"]] += 1
            continue
        supported += 1
        base_correct_supported += int(realized == majority_key)
        stats["supported"] += 1
        if result["key"] == realized:
            correct += 1
            stats["correct"] += 1
        else:
            confusions[(realized, result["key"])] += 1
    total = len(transitions)
    return dict(
        fold=int(fold),
        train_episodes=len(episode_ids) - len(set(
            episode for episode in episode_ids if fold_of[episode] == fold
        )),
        test_episodes=len(set(
            episode for episode in episode_ids if fold_of[episode] == fold
        )),
        test_rows=int(total),
        supported=int(supported),
        coverage=(float(supported) / total if total else 0.0),
        accuracy=(
            float(correct) / supported if supported else None
        ),
        base_accuracy=(float(base_correct) / total if total else None),
        base_correct=int(base_correct),
        base_accuracy_supported=(
            float(base_correct_supported) / supported if supported else None
        ),
        base_correct_supported=int(base_correct_supported),
        supported_correct=int(correct),
        majority_key=_label(majority_key),
        unknown_reasons={
            str(reason): int(count) for reason, count in reasons.items()
        },
        confusions=[
            dict(
                realized=_label(realized),
                predicted=_label(predicted),
                count=int(count),
            )
            for (realized, predicted), count in confusions.most_common(8)
        ],
        per_key=[
            dict(
                key=_label(key),
                n=int(stats["n"]),
                supported=int(stats["supported"]),
                correct=int(stats["correct"]),
            )
            for key, stats in sorted(
                per_key.items(), key=lambda item: -item[1]["n"],
            )
        ],
        stage_histogram=[
            dict(key=_label(key), transitions=int(count))
            for key, count in train_counts.most_common()
        ],
        resolver=resolver.snapshot(),
        experience=experience.snapshot(),
    )


def aggregate(folds):
    usable = [fold for fold in folds if fold]
    total = sum(fold["test_rows"] for fold in usable)
    supported = sum(fold["supported"] for fold in usable)
    correct = sum(fold["supported_correct"] for fold in usable)
    base_correct = sum(fold["base_correct"] for fold in usable)
    base_correct_supported = sum(
        fold["base_correct_supported"] for fold in usable
    )
    reasons = Counter()
    for fold in usable:
        reasons.update({
            reason: count
            for reason, count in fold["unknown_reasons"].items()
        })
    return dict(
        folds=len(usable),
        test_rows=int(total),
        supported=int(supported),
        coverage=(float(supported) / total if total else 0.0),
        accuracy=(float(correct) / supported if supported else None),
        base_accuracy=(float(base_correct) / total if total else None),
        base_accuracy_supported=(
            float(base_correct_supported) / supported
            if supported else None
        ),
        lift=(
            float(correct) / supported
            - float(base_correct_supported) / supported
            if supported else None
        ),
        unknown_reasons={
            str(reason): int(count) for reason, count in reasons.items()
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-episodes", type=int, default=2)
    parser.add_argument("--min-probability", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.1)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if args.folds < 2:
        raise SystemExit("folds must be at least 2")
    rows, _counts = collect(
        args.seed, steps=args.steps, coverage=args.coverage, lag=LAG,
    )
    groups = episode_groups(rows)
    print(
        "stage resolver probe: seed %d  %d rows / %d episodes  folds %d"
        % (args.seed, len(rows), len(groups), args.folds)
    )
    folds = []
    for fold in range(int(args.folds)):
        report = fold_report(rows, args.seed, fold, args.folds, args)
        if report is None:
            continue
        folds.append(report)
        print(
            "  fold %d  test %d ep / %d rows  supported %d (%.3f)  "
            "accuracy %s  base %s  majority %s"
            % (
                report["fold"], report["test_episodes"],
                report["test_rows"], report["supported"],
                report["coverage"],
                "n/a" if report["accuracy"] is None
                else "%.3f" % report["accuracy"],
                "n/a" if report["base_accuracy"] is None
                else "%.3f" % report["base_accuracy"],
                report["majority_key"],
            )
        )
    summary = aggregate(folds)
    print(
        "aggregate: %d folds, %d rows, coverage %.3f, accuracy %s, "
        "constant-key base (all) %s, base (supported) %s, lift %s"
        % (
            summary["folds"], summary["test_rows"], summary["coverage"],
            "n/a" if summary["accuracy"] is None
            else "%.3f" % summary["accuracy"],
            "n/a" if summary["base_accuracy"] is None
            else "%.3f" % summary["base_accuracy"],
            "n/a" if summary["base_accuracy_supported"] is None
            else "%.3f" % summary["base_accuracy_supported"],
            "n/a" if summary["lift"] is None
            else "%+.3f" % summary["lift"],
        )
    )
    print(
        "  unknown reasons: %s"
        % ", ".join(
            "%s=%d" % (reason, count)
            for reason, count in sorted(summary["unknown_reasons"].items())
        )
    )
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                folds=int(args.folds), steps=int(args.steps),
                coverage=float(args.coverage), lag=LAG,
                max_nodes=int(args.max_nodes),
                quantile=float(args.quantile),
                min_support=int(args.min_support),
                min_episodes=int(args.min_episodes),
                min_probability=float(args.min_probability),
                min_margin=float(args.min_margin),
            ),
            folds=folds,
            aggregate=summary,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
