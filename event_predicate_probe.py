"""Compare the one-slot event predicate with the existing 0/1 death rules.

The single-slot rule is the query-graph placeholder: it reads one anonymous
state slot and writes a 0/1 event at +1.  ``DeathRuleSet`` is the current
multi-literal baseline.  Both are fit on the same training episodes and
verified on the same held-out episodes.  UNKNOWN is counted separately and is
never treated as safe.
"""

import argparse

import numpy as np

from death_rules import UNKNOWN as DEATH_UNKNOWN
from death_rules import DeathRuleSet
from event_rules import make_death_event_rule
from query_graph import UNKNOWN as GRAPH_UNKNOWN
from query_graph import Query, RuleBank, Slot
from sparse_rules import fit_event_rule
from substrate_generic import collect
from substrate_nodes import LAG


def episode_split(rows, modulus=5):
    train, test = [], []
    for index, row in enumerate(rows):
        if int(row["episode"]) % int(modulus) == int(modulus) - 1:
            test.append(index)
        else:
            train.append(index)
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def metrics(prediction, labels):
    prediction = np.asarray(prediction)
    labels = np.asarray(labels, dtype=int)
    known = np.asarray([
        value != DEATH_UNKNOWN for value in prediction
    ], dtype=bool)
    positive = labels == 1
    negative = labels == 0
    true_positive = int(np.sum(known & (prediction == 1) & positive))
    false_positive = int(np.sum(known & (prediction == 1) & negative))
    true_negative = int(np.sum(known & (prediction == 0) & negative))
    false_negative = int(np.sum(known & (prediction == 0) & positive))
    tpr = true_positive / float(true_positive + false_negative + 1e-12)
    tnr = true_negative / float(true_negative + false_positive + 1e-12)
    precision = true_positive / float(
        true_positive + false_positive + 1e-12
    )
    recall = tpr
    return dict(
        total=int(len(labels)),
        unknowns=int(np.sum(~known)),
        coverage=float(known.mean()) if len(labels) else 0.0,
        precision=float(precision),
        recall=float(recall),
        balanced_accuracy=float(0.5 * (tpr + tnr)),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--modulus", type=int, default=5)
    parser.add_argument(
        "--label", choices=("pain", "terminal"), default="pain",
    )
    parser.add_argument("--max-rules", type=int, default=24)
    parser.add_argument("--max-literals", type=int, default=5)
    args = parser.parse_args()

    rows, _ = collect(seed=args.seed, steps=args.steps, lag=LAG)
    states = np.asarray([row["obs"] for row in rows], dtype=float)
    if args.label == "pain":
        labels = np.asarray([
            1 if float(row["signals"][1]) > 0.5 else 0 for row in rows
        ], dtype=int)
    else:
        labels = np.asarray([
            1 if bool(row["terminal"]) else 0 for row in rows
        ], dtype=int)
    train, test = episode_split(rows, args.modulus)
    print("== seed %d steps %d label %s ==" % (
        args.seed, args.steps, args.label,
    ))
    print(
        "rows %d  train positives %d  test positives %d" % (
            len(rows), int(labels[train].sum()), int(labels[test].sum()),
        )
    )

    fitted = fit_event_rule(
        states[train], labels[train], Slot("event", 1),
        min_support=10,
    )
    if fitted is None:
        print("single-slot rule: no supported threshold")
    else:
        rule, report = fitted
        prediction = []
        for state in states[test]:
            value = rule.predict({
                Slot("x%d" % report["column"], 0): state[report["column"]]
            })
            prediction.append(
                DEATH_UNKNOWN if value is GRAPH_UNKNOWN else value
            )
        result = metrics(prediction, labels[test])
        print(
            "single-slot: x%d %s %.6f  train balanced %.4f" % (
                report["column"], ">" if report["direction"] > 0 else "<",
                report["threshold"], report["balanced_accuracy"],
            )
        )
        print("  test " + _format(result))

    baseline = DeathRuleSet(
        max_rules=args.max_rules,
        max_literals=args.max_literals,
        min_support=2,
        max_thresholds=256,
    ).fit(states[train], labels[train])
    verify = baseline.verify(states[test], labels[test])
    print("death-rules: %d rules" % len(baseline.rules_))
    print("  test " + _format(verify))
    event_rule = make_death_event_rule(
        baseline,
        [Slot("x%d" % index, 0) for index in range(states.shape[1])],
        target_slot=Slot("event", 1),
    )
    bank = RuleBank([event_rule])
    graph_prediction = []
    for state in states[test]:
        source = {
            Slot("x%d" % index, 0): float(state[index])
            for index in range(states.shape[1])
        }
        value = bank.query(Query(
            target_slots=(Slot("event", 1),), source=source,
        )).values[Slot("event", 1)]
        graph_prediction.append(
            DEATH_UNKNOWN if value is GRAPH_UNKNOWN else value
        )
    print("  query-graph wrapper " + _format(
        metrics(graph_prediction, labels[test])
    ))
    print("  first rules:")
    print(baseline.describe(args.max_rules))


def _format(result):
    true_positive = int(result["true_positive"])
    false_positive = int(result["false_positive"])
    true_negative = int(result["true_negative"])
    false_negative = int(result["false_negative"])
    precision = true_positive / float(
        true_positive + false_positive + 1e-12
    )
    recall = true_positive / float(
        true_positive + false_negative + 1e-12
    )
    balanced = 0.5 * (
        recall + true_negative / float(
            true_negative + false_positive + 1e-12
        )
    )
    unknown = result.get("unknowns", result.get("unknown", 0))
    coverage = result.get("coverage", 1.0)
    return (
        "precision %.4f recall %.4f balanced %.4f coverage %.4f "
        "unknown %d/%d TP %d FP %d TN %d FN %d" % (
            precision, recall, balanced, coverage, unknown,
            result["total"], true_positive, false_positive,
            true_negative, false_negative,
        )
    )


if __name__ == "__main__":
    main()
