"""Cross-world pattern reuse: schema retrieval vs learning from scratch.

A synthetic relational task supplies the ground truth in a latent frame:

    death = (|z0| < a AND |z1| > b) OR (|z2| > c)

Each world applies a different per-coordinate affine observation frame
``x = z * scale + offset``.  The death condition is therefore a different
threshold problem in every world, but the relation dimensions and mode
structure are shared.

The probe learns rules on several source worlds, stores their dimension
schemas in a PatternLibrary, and then compares, on unseen worlds:

- scratch: fit all coordinates from k labelled samples,
- schema: fit only the library's relation dimensions from the same k,
- normalise: standardise the world's coordinates, then fit all dimensions,
- schema+normalise: both.

The metric is held-out terminal coverage / unknown / false positives as a
function of k.  No decision path is changed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from death_rules import UNKNOWN, DeathRuleSet
from pattern_library import PatternLibrary

RULE_KWARGS = dict(
    max_rules=12, max_literals=6, min_support=1,
    max_thresholds=64, coverage_first=True, boundary_margin=0.01,
    compose=True, compose_max_base=12, compose_max_pairs=64,
)


def make_world(rng, n):
    latent = rng.uniform(-1.0, 1.0, (n, 4))
    labels = (
        ((np.abs(latent[:, 0]) < 0.15) & (np.abs(latent[:, 1]) > 0.4))
        | (np.abs(latent[:, 2]) > 0.9)
    ).astype(int)
    scale = rng.uniform(0.5, 2.0, 4)
    offset = rng.uniform(-1.0, 1.0, 4)
    observed = latent * scale + offset
    return observed, labels


def fit_rule_set(states, labels, dims=None):
    return DeathRuleSet(
        feature_dims=dims, **RULE_KWARGS
    ).fit(states, labels)


def evaluate(rules, states, labels):
    prediction = rules.classify_death(states)
    terminal = labels == 1
    covered = int(np.sum(terminal & (prediction == 1)))
    unknown = int(np.sum(terminal & (prediction == UNKNOWN)))
    false_positive = int(np.sum((~terminal) & (prediction == 1)))
    return dict(
        terminals=int(terminal.sum()),
        covered=covered,
        unknown=unknown,
        false_positive=false_positive,
        coverage=(float(covered) / float(terminal.sum())
                  if terminal.any() else 0.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-worlds", type=int, default=8)
    parser.add_argument("--test-worlds", type=int, default=6)
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--test-samples", type=int, default=600)
    parser.add_argument("--budgets", default="5,10,20,50,100")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    library = PatternLibrary()
    source_rules = []
    for world in range(args.source_worlds):
        states, labels = make_world(rng, args.samples)
        rules = fit_rule_set(states, labels)
        source_rules.append(rules)
        library.add_rules(rules, world=world)
    schema_dims = library.schema_dims(1)
    print("library patterns:")
    print(library.describe(16))
    print("schema dims:", schema_dims)

    budgets = [int(value) for value in args.budgets.split(",")]
    arms = ("scratch", "schema", "normalise", "schema_normalise")
    summary = {
        str(budget): {arm: [] for arm in arms}
        for budget in budgets
    }
    for world in range(args.test_worlds):
        test_states, test_labels = make_world(rng, args.test_samples)
        for budget in budgets:
            train_states, train_labels = make_world(rng, budget)
            mean = train_states.mean(axis=0)
            scale = train_states.std(axis=0)
            scale = np.where(scale > 1e-8, scale, 1.0)
            normalized_train = (train_states - mean) / scale
            normalized_test = (test_states - mean) / scale
            fits = {
                "scratch": fit_rule_set(train_states, train_labels),
                "schema": fit_rule_set(
                    train_states, train_labels,
                    dims=schema_dims or None,
                ),
                "normalise": fit_rule_set(
                    normalized_train, train_labels
                ),
                "schema_normalise": fit_rule_set(
                    normalized_train, train_labels,
                    dims=schema_dims or None,
                ),
            }
            for arm, rules in fits.items():
                metrics = evaluate(
                    rules,
                    normalized_test if "normalise" in arm else test_states,
                    test_labels,
                )
                summary[str(budget)][arm].append(metrics)

    output = {}
    for budget in budgets:
        output[str(budget)] = {}
        print("\n== budget %d ==" % budget)
        for arm in arms:
            rows = summary[str(budget)][arm]
            entry = dict(
                coverage=float(np.mean([row["coverage"] for row in rows])),
                unknown=float(np.mean([row["unknown"] for row in rows])),
                false_positive=float(np.mean([
                    row["false_positive"] for row in rows
                ])),
                terminals=float(np.mean([
                    row["terminals"] for row in rows
                ])),
            )
            output[str(budget)][arm] = entry
            print(
                "  %-18s coverage %.3f  unknown %.2f  fp %.2f"
                % (
                    arm, entry["coverage"], entry["unknown"],
                    entry["false_positive"],
                )
            )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(dict(
                schema_dims=list(schema_dims),
                patterns=[
                    dict(dims=list(pattern.dims),
                         label=pattern.label,
                         support=pattern.support,
                         text=pattern.text)
                    for pattern in library.patterns
                ],
                results=output,
            ), handle, indent=2, default=str)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
