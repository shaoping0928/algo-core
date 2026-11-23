"""Measure full versus key-node queries against actual readouts.

A key-node query is only useful if the selected slots are the ones the readout
declares.  This probe fits the sparse frame rules on training episodes and
then, on held-out rows, compares:

* the full seven-target query;
* fixed key subsets of one, two and three targets;
* two readout-declared subsets (vertical and horizontal).

It reports visited rules, completion (fraction of requested slots filled),
and the standardized error on the filled slots.  The fixed subsets are not a
recommendation: the point is that an arbitrary subset can miss a readout's
required slot and return UNKNOWN.
"""

import argparse

import numpy as np

from query_graph import UNKNOWN, Query, Readout, RuleBank, Slot
from sparse_rules import fit_sparse_scopes, make_rules
from substrate_generic import collect
from substrate_nodes import LAG


def key_indices(count):
    order = (0, 2, 3, 1, 4, 5, 6)
    return order[:int(count)]


def split(rows, modulus=5):
    train, test = [], []
    for row in rows:
        if int(row["episode"]) % int(modulus) == int(modulus) - 1:
            test.append(row)
        else:
            train.append(row)
    return train, test


def source_slots(row):
    observation = np.asarray(row["obs"], dtype=float)
    history = np.asarray(row["history"], dtype=float)
    source = {
        Slot("x%d" % index, 0): float(value)
        for index, value in enumerate(observation)
    }
    source.update({
        Slot("a%d" % index, 0): float(value)
        for index, value in enumerate(history)
    })
    return source


def run_query(bank, row, targets, readout=None):
    result = bank.query(Query(
        target_slots=tuple(targets),
        source=source_slots(row),
        readout=readout,
        budget=200,
    ))
    actual = np.asarray(row["next_obs"], dtype=float)
    scale = np.asarray(row["obs"], dtype=float).std()
    errors = []
    for target in targets:
        value = result.values.get(target)
        if value is None or value is UNKNOWN:
            continue
        errors.append(
            abs(float(value) - float(actual[target.time - 1]))
            / max(float(scale), 1e-8)
        )
    completion = 1.0 - result.unknown / float(max(len(targets), 1))
    return result, completion, (
        float(np.mean(errors)) if errors else float("nan")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--modulus", type=int, default=5)
    args = parser.parse_args()
    rows, _ = collect(seed=args.seed, steps=args.steps, lag=LAG)
    train, test = split(rows, args.modulus)
    observations = np.asarray([row["obs"] for row in train], dtype=float)
    histories = np.asarray([row["history"] for row in train], dtype=float)
    next_observations = np.asarray(
        [row["next_obs"] for row in train], dtype=float
    )
    scopes = fit_sparse_scopes(
        observations, histories, next_observations, max_terms=6
    )
    rules = make_rules(
        scopes, observations.shape[1], histories.shape[1],
    )
    bank = RuleBank(rules)
    scale = np.where(
        observations.std(axis=0) > 1e-8, observations.std(axis=0), 1.0
    )

    vertical = Readout(
        "vertical",
        required_slots=(Slot("y0", 1), Slot("y1", 1)),
        function=lambda values: (
            values[Slot("y0", 1)], values[Slot("y1", 1)]
        ),
    )
    horizontal = Readout(
        "horizontal",
        required_slots=(Slot("y2", 1), Slot("y4", 1)),
        function=lambda values: (
            values[Slot("y2", 1)], values[Slot("y4", 1)]
        ),
    )

    print("== seed %d steps %d train %d test %d ==" % (
        args.seed, args.steps, len(train), len(test),
    ))
    print(
        "query                 targets  visited  applied  completion  "
        "std_error  readout"
    )
    rows_to_run = [
        ("full", tuple(Slot("y%d" % index, 1) for index in range(7)), None),
        ("fixed m=1", tuple(
            Slot("y%d" % index, 1) for index in key_indices(1)
        ), None),
        ("fixed m=2", tuple(
            Slot("y%d" % index, 1) for index in key_indices(2)
        ), None),
        ("fixed m=3", tuple(
            Slot("y%d" % index, 1) for index in key_indices(3)
        ), None),
        ("vertical", (Slot("y0", 1), Slot("y1", 1)), vertical),
        ("horizontal", (Slot("y2", 1), Slot("y4", 1)), horizontal),
    ]
    for name, targets, readout in rows_to_run:
        visited = applied = unknown = 0
        completion = []
        errors = []
        readout_unknown = 0
        for row in test:
            result, fraction, error = run_query(
                bank, row, targets, readout=readout,
            )
            visited += result.visited
            applied += result.applied
            unknown += result.unknown
            completion.append(fraction)
            if not np.isnan(error):
                errors.append(error)
            if readout is not None and result.readout_unknown:
                readout_unknown += 1
        print(
            "%-20s %8d %8.2f %8.2f %11.3f %9.4f  %s" % (
                name, len(targets), visited / float(max(len(test), 1)),
                applied / float(max(len(test), 1)),
                float(np.mean(completion)) if completion else 0.0,
                float(np.mean(errors)) if errors else float("nan"),
                "unknown" if readout is not None and readout_unknown
                else ("ok" if readout is not None else "-"),
            )
        )


if __name__ == "__main__":
    main()
