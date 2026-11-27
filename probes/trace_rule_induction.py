"""Induce and verify pain rules from a batch of transitions.

The inducer is task-agnostic: candidate antecedents are thresholds over raw
observation features and their magnitudes; rules are conjunctions of those
predicates.  The target is the world pain signal, never a task label.

Induction uses a train split of episodes.  Verification uses held-out episodes,
so a rule has to survive data it was not selected on.  A rule is kept only when
its precision exceeds the base pain rate with a binomial z-score above a
learned-ish margin and when it has enough support.

    python trace_rule_induction.py --seed 100 --steps 5000 --coverage 0.1

Diagnostic only: it does not train or change the node runtime.
"""
import argparse

import numpy as np

from substrate_generic import collect

OBS_NAMES = ("f0", "f1", "f2", "f3", "f4", "f5")
QUANTILES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40,
             0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98)


class Predicate:
    def __init__(self, feature, magnitude, comparison, threshold, mask):
        self.feature = int(feature)
        self.magnitude = bool(magnitude)
        self.comparison = comparison
        self.threshold = float(threshold)
        self.mask = mask

    def label(self):
        name = OBS_NAMES[self.feature] if self.feature < len(OBS_NAMES) \
            else "f%d" % self.feature
        if self.magnitude:
            name = "|%s|" % name
        return "%s %s %.3f" % (name, self.comparison, self.threshold)


def candidates(x):
    result = []
    for feature in range(x.shape[1]):
        for magnitude in (False, True):
            values = np.abs(x[:, feature]) if magnitude else x[:, feature]
            for quantile in QUANTILES:
                threshold = float(np.quantile(values, quantile))
                for comparison, mask in (
                    ("<", values < threshold),
                    (">", values > threshold),
                ):
                    if mask.all() or not mask.any():
                        continue
                    result.append(Predicate(
                        feature, magnitude, comparison, threshold, mask
                    ))
    # Deduplicate identical masks.
    unique = {}
    for predicate in result:
        unique.setdefault(predicate.mask.tobytes(), predicate)
    return list(unique.values())


def stats(mask, target):
    support = int(mask.sum())
    if support <= 0:
        return dict(support=0, hits=0.0, precision=0.0, z=0.0)
    hits = float(target[mask].sum())
    precision = hits / support
    base = float(target.mean()) if len(target) else 0.0
    variance = max(base * (1.0 - base), 1e-12)
    z = (hits - support * base) / np.sqrt(support * variance)
    return dict(support=support, hits=hits, precision=precision, z=float(z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--source", choices=("obs", "next_obs"),
                        default="next_obs")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-train-support", type=int, default=20)
    parser.add_argument("--min-test-support", type=int, default=5)
    parser.add_argument("--min-z", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    transitions, _ = collect(
        args.seed, args.steps, coverage=args.coverage, lag=args.lag
    )
    x = np.asarray([row[args.source] for row in transitions], dtype=float)
    target = np.asarray(
        [row["signals"][1] > 0.5 for row in transitions], dtype=float
    )
    episodes = np.asarray([row["episode"] for row in transitions])
    order = np.unique(episodes)
    split = max(int(len(order) * args.train_fraction), 1)
    train_episodes = set(int(value) for value in order[:split])
    train = np.asarray([int(value) in train_episodes for value in episodes])
    test = ~train

    train_x, train_y = x[train], target[train]
    test_x, test_y = x[test], target[test]
    base_train = float(train_y.mean())
    base_test = float(test_y.mean())
    print("transitions=%d episodes=%d train/test=%d/%d" % (
        len(x), len(order), int(train.sum()), int(test.sum())
    ))
    print("pain base rate train=%.4f test=%.4f" % (base_train, base_test))

    predicates = candidates(train_x)
    print("candidate predicates=%d" % len(predicates))

    # Rebuild test masks with the same thresholds (not test quantiles).
    test_masks = []
    for predicate in predicates:
        values = (
            np.abs(test_x[:, predicate.feature])
            if predicate.magnitude else test_x[:, predicate.feature]
        )
        test_masks.append(
            values < predicate.threshold
            if predicate.comparison == "<" else values > predicate.threshold
        )

    singles = []
    for index, predicate in enumerate(predicates):
        train_stats = stats(predicate.mask, train_y)
        if train_stats["support"] < args.min_train_support:
            continue
        test_stats = stats(test_masks[index], test_y)
        singles.append((predicate, train_stats, test_stats))
    singles.sort(key=lambda item: (item[1]["z"], item[1]["support"]),
                 reverse=True)

    print("\nsingle predicates (train z / test precision):")
    for predicate, train_stats, test_stats in singles[:args.top]:
        print("  %-22s support=%4d p=%.3f z=%5.1f | test n=%3d p=%.3f" % (
            predicate.label(), train_stats["support"],
            train_stats["precision"], train_stats["z"],
            test_stats["support"], test_stats["precision"],
        ))

    # Conjunctions are searched over every predicate with enough support.
    # Filtering by single-predicate significance would miss a rule whose parts
    # are individually weak but jointly decisive (a pure conjunction).
    pool = [
        item for item in singles
        if item[1]["support"] >= args.min_train_support
    ]
    print("\npair pool=%d" % len(pool))

    pairs = []
    seen = set()
    for first in range(len(pool)):
        for second in range(first + 1, len(pool)):
            predicate_a, stats_a, _ = pool[first]
            predicate_b, stats_b, _ = pool[second]
            train_mask = predicate_a.mask & predicate_b.mask
            key = train_mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            train_stats = stats(train_mask, train_y)
            if train_stats["support"] < args.min_train_support:
                continue
            if train_stats["z"] < max(args.min_z, 3.0):
                continue
            if train_stats["precision"] <= base_train * 2.0:
                continue
            index_a = predicates.index(predicate_a)
            index_b = predicates.index(predicate_b)
            test_mask = test_masks[index_a] & test_masks[index_b]
            test_stats = stats(test_mask, test_y)
            if test_stats["support"] < args.min_test_support:
                continue
            pairs.append((predicate_a, predicate_b, train_stats, test_stats))

    pairs.sort(key=lambda item: (item[3]["precision"], item[2]["z"]),
               reverse=True)
    print("\nverified pair rules (selected on train, checked on test):")
    for predicate_a, predicate_b, train_stats, test_stats in pairs[:args.top]:
        print("  %-20s AND %-20s train n=%4d p=%.3f lift=%.1fx z=%5.1f | test n=%3d p=%.3f" % (
            predicate_a.label(), predicate_b.label(),
            train_stats["support"], train_stats["precision"],
            train_stats["precision"] / max(base_train, 1e-9),
            train_stats["z"], test_stats["support"],
            test_stats["precision"],
        ))


if __name__ == "__main__":
    main()
