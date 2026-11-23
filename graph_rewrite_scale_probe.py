"""Measure the size and search complexity of query-driven transition rewriting.

This is a read-only scale probe, not a learner change.  It asks the question
that comes before building the graph-rewrite engine:

    if every target slot can be filled by a local rule from a context
    cluster, how many rules exist, how many can fill the same slot, and how
    large is the search tree under a support/confidence gate?

The probe builds two views from the same stream:

* per-target sparse scopes (forward selection over state + action window);
* per-context local rules (one sparse model per target per k-means regime).

It then reports the rule count, the per-target fan-out, the shared-regime and
independent-regime combination counts, the ambiguity rate, and the size of the
existing prototype transition graph.  No task vocabulary is used: state and
action slots are anonymous indices.
"""

import argparse

import numpy as np

from substrate_generic import collect
from substrate_nodes import LAG, fit
from query_graph import (
    Applicability,
    Query,
    RuleBank,
    Slot,
    make_linear_rule,
)


def standardize(matrix):
    """Zero mean, unit spread; constant columns are left as zeros."""
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (matrix - mean) / scale


def design(matrix, columns):
    """Intercept plus the selected columns."""
    if not columns:
        return np.ones((len(matrix), 1), dtype=float)
    return np.column_stack([
        np.ones(len(matrix), dtype=float),
        np.asarray(matrix, dtype=float)[:, list(columns)],
    ])


def ols(matrix, target, ridge=1e-8):
    gram = matrix.T @ matrix + ridge * np.eye(matrix.shape[1])
    return np.linalg.solve(gram, matrix.T @ target)


def subset_fit(matrix, target, columns):
    weights = ols(design(matrix, columns), target)
    prediction = design(matrix, columns) @ weights
    residual = target - prediction
    return weights, prediction, float(residual @ residual)


def forward_select(matrix, target, max_terms=6, bic_gain=1.0):
    """Greedy BIC forward selection; returns selected columns and fit."""
    n_rows, n_columns = matrix.shape
    selected = []
    remaining = list(range(n_columns))
    weights, prediction, sse = subset_fit(matrix, target, selected)
    bic = n_rows * np.log(sse / n_rows + 1e-12)
    best = (bic, list(selected), weights, prediction, sse)
    while remaining and len(selected) < int(max_terms):
        candidate = None
        for column in remaining:
            columns = selected + [column]
            weights, prediction, sse = subset_fit(
                matrix, target, columns
            )
            score = (
                n_rows * np.log(sse / n_rows + 1e-12)
                + len(columns) * np.log(n_rows)
            )
            if candidate is None or score < candidate[0]:
                candidate = (score, columns, weights, prediction, sse)
        if candidate is None or candidate[0] >= best[0] - float(bic_gain):
            break
        best = candidate
        selected = list(candidate[1])
        remaining.remove(selected[-1])
    bic, selected, weights, prediction, sse = best
    variance = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - sse / (variance + 1e-12) if variance > 0.0 else 0.0
    return selected, weights, float(r2), float(bic)


def rule_fit(matrix, target, max_terms=6, seed=0):
    """Fit on 80% of the rows and report the held-out R2."""
    n_rows = len(matrix)
    if n_rows < 8:
        return None
    order = np.random.RandomState(seed).permutation(n_rows)
    cut = max(4, int(round(0.8 * n_rows)))
    train, test = order[:cut], order[cut:]
    selected, weights, r2_in, _ = forward_select(
        matrix[train], target[train], max_terms=max_terms
    )
    if len(test) < 3:
        return selected, weights, r2_in, float("nan")
    prediction = design(matrix[test], selected) @ weights
    residual = target[test] - prediction
    base = target[test] - target[train].mean()
    r2_out = 1.0 - float(residual @ residual) / (
        float(base @ base) + 1e-12
    )
    return selected, weights, r2_in, float(r2_out)


def kmeans(matrix, k, seed=0, iterations=40):
    """Small deterministic Lloyd implementation; no sklearn dependency."""
    n_rows = len(matrix)
    k = max(1, min(int(k), n_rows))
    rng = np.random.RandomState(seed)
    centers = matrix[rng.choice(n_rows, k, replace=False)].copy()
    labels = np.zeros(n_rows, dtype=int)
    for _ in range(int(iterations)):
        distances = (
            (matrix[:, None, :] - centers[None, :, :]) ** 2
        ).sum(axis=-1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = labels == cluster
            if np.any(members):
                centers[cluster] = matrix[members].mean(axis=0)
    return labels, centers


def analyze_regimes(matrix, target, labels, k, min_support, r2_min,
                    max_terms):
    """Fit one sparse rule per (target, regime) and count support."""
    n_targets = target.shape[1]
    supported = np.zeros((k, n_targets), dtype=bool)
    r2_in = np.full((k, n_targets), np.nan)
    r2_out = np.full((k, n_targets), np.nan)
    selected = {}
    weights = {}
    sizes = np.bincount(labels, minlength=k)
    for cluster in range(k):
        rows = np.where(labels == cluster)[0]
        if len(rows) < int(min_support):
            continue
        for column in range(n_targets):
            fitted = rule_fit(
                matrix[rows], target[rows, column],
                max_terms=max_terms, seed=cluster * 31 + column,
            )
            if fitted is None:
                continue
            rule_columns, rule_weights, in_score, out_score = fitted
            r2_in[cluster, column] = in_score
            r2_out[cluster, column] = out_score
            if in_score >= float(r2_min) and (
                np.isnan(out_score) or out_score >= 0.0
            ):
                supported[cluster, column] = True
                selected[(cluster, column)] = rule_columns
                weights[(cluster, column)] = rule_weights
    return supported, r2_in, r2_out, selected, weights, sizes


def ambiguity_rate(matrix, target, centers, labels, supported, selected,
                   weights, tolerance=0.1, ratio=1.2):
    """Rows whose two nearest regimes disagree on any supported target."""
    if len(centers) < 2:
        return 0.0, 0
    distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    order = np.argsort(distances, axis=1)
    ambiguous_rows = 0
    conflicting_rows = 0
    for row in range(len(matrix)):
        first, second = order[row, 0], order[row, 1]
        d1 = float(distances[row, first]) ** 0.5
        d2 = float(distances[row, second]) ** 0.5
        if d2 > float(ratio) * max(d1, 1e-12):
            continue
        ambiguous_rows += 1
        conflict = False
        for column in range(target.shape[1]):
            if not (supported[first, column] and supported[second, column]):
                continue
            first_pred = design(
                matrix[row:row + 1], selected[(first, column)]
            ) @ weights[(first, column)]
            second_pred = design(
                matrix[row:row + 1], selected[(second, column)]
            ) @ weights[(second, column)]
            if abs(float(first_pred[0] - second_pred[0])) > tolerance:
                conflict = True
                break
        if conflict:
            conflicting_rows += 1
    return (
        conflicting_rows / float(max(ambiguous_rows, 1)),
        ambiguous_rows,
    )


def graph_statistics(agent):
    """Describe the prototype transition graph the shipped fit builds."""
    edges = getattr(agent, "transition_edges", {}) or {}
    out_degrees = [len(targets) for targets in edges.values()]
    counts = [
        int(entry.get("count", 0))
        for targets in edges.values()
        for entry in targets.values()
    ]
    both_actions = 0
    for targets in edges.values():
        actions = {int(entry.get("action", -1)) for entry in targets.values()}
        if 0 in actions and 1 in actions:
            both_actions += 1
    milestones = getattr(agent, "milestone_transitions", None) or []
    stages = getattr(agent, "milestone_stages", None) or {}
    keys = getattr(agent, "milestone_keys", None) or {}

    def summary(values):
        if not values:
            return "0"
        values = np.asarray(values, dtype=float)
        return "p50 %.0f p90 %.0f max %.0f" % (
            np.percentile(values, 50),
            np.percentile(values, 90),
            values.max(),
        )

    return dict(
        nodes=len(edges),
        edges=len(counts),
        out_degree=summary(out_degrees),
        support=summary(counts),
        support_ge2=int(np.sum(np.asarray(counts) >= 2)) if counts else 0,
        support_ge5=int(np.sum(np.asarray(counts) >= 5)) if counts else 0,
        both_actions=both_actions,
        milestones=len(milestones),
        stages=len(stages),
        keys=len(keys),
    )


def _key_indices(count):
    """A deterministic key-node subset: vertical, horizontal, gap first."""
    order = (0, 2, 3, 1, 4, 5, 6)
    return order[:int(count)]


def _slot_for(index, n_state):
    if index < n_state:
        return Slot("x%d" % index, 0)
    return Slot("a%d" % (index - n_state), 0)


def _global_rules(global_fit, n_state):
    rules = []
    for column, (selected, weights, r2) in global_fit.items():
        inputs = tuple(_slot_for(index, n_state) for index in selected)
        rules.append(make_linear_rule(
            name="global-y%d" % column,
            target_slot=Slot("y%d" % column, 1),
            input_slots=inputs,
            weights=weights,
            confidence=float(r2),
            support=1,
            group=0,
        ))
    return rules


def _cluster_rules(selected, weights, centers, sizes, n_state, k):
    rules = []
    all_slots = tuple(
        [Slot("x%d" % index, 0) for index in range(n_state)]
        + [Slot("a%d" % index, 0) for index in range(
            centers.shape[1] - n_state
        )]
    )
    distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=-1
    )
    distances[distances == 0.0] = np.inf
    minimum = float(np.min(distances)) if k > 1 else 1.0
    radius = max(1e-9, 0.45 * minimum)
    for (cluster, column), inputs in selected.items():
        rules.append(make_linear_rule(
            name="c%d-y%d" % (cluster, column),
            target_slot=Slot("y%d" % column, 1),
            input_slots=tuple(
                _slot_for(index, n_state) for index in inputs
            ),
            weights=weights[(cluster, column)],
            confidence=1.0,
            support=int(sizes[cluster]),
            applicability=Applicability(
                slots=all_slots,
                center=centers[cluster],
                radius=radius,
                support=int(sizes[cluster]),
            ),
            group=int(cluster),
        ))
    return rules, all_slots, radius


def _search_rows(global_fit, selected, weights, centers, sizes,
                 n_state, k):
    source = {
        Slot("x%d" % index, 0): float(centers[0][index])
        for index in range(n_state)
    }
    source.update({
        Slot("a%d" % index, 0): float(centers[0][n_state + index])
        for index in range(centers.shape[1] - n_state)
    })
    global_rules = _global_rules(global_fit, n_state)
    global_bank = RuleBank(
        global_rules,
        shared_applicability=Applicability(support=1),
    )
    cluster, all_slots, radius = _cluster_rules(
        selected, weights, centers, sizes, n_state, k,
    )
    shared_bank = RuleBank(
        cluster,
        shared_applicability=Applicability(
            slots=all_slots,
            center=centers[0],
            radius=radius,
            support=int(sizes[0]),
        ),
    )
    independent_bank = RuleBank(cluster)
    rows = []
    for count in (1, 2, 3, 7):
        targets = tuple(
            Slot("y%d" % column, 1) for column in _key_indices(count)
        )
        for label, bank, group in (
            ("global-shared", global_bank, None),
            ("cluster-shared", shared_bank, 0),
            ("independent", independent_bank, None),
        ):
            result = bank.query(Query(
                target_slots=targets,
                source=source,
                budget=500,
                group=group,
            ))
            rows.append(dict(
                bank=label, key_nodes=count, visited=result.visited,
                pruned=result.pruned, applied=result.applied,
                unknown=result.unknown, conflicts=result.conflicts,
                shared=result.shared_status,
            ))
    return rows


def analyze_size(seed, steps, regime_values, min_support, r2_min, max_terms,
                 timing):
    transitions, _ = collect(seed=seed, steps=steps, lag=LAG)
    observations = np.asarray(
        [row["obs"] for row in transitions], dtype=float
    )
    histories = np.asarray(
        [row["history"] for row in transitions], dtype=float
    )
    next_observations = np.asarray(
        [row["next_obs"] for row in transitions], dtype=float
    )
    inputs = np.concatenate([observations, histories], axis=1)
    scaled_inputs = standardize(inputs)
    scaled_target = standardize(next_observations)
    episodes = len({int(row["episode"]) for row in transitions})
    n_targets = scaled_target.shape[1]

    print()
    print("== data seed %d steps %d ==" % (seed, steps))
    print(
        "frames %d  episodes %d  obs_dim %d  action_window %d  "
        "targets %d" % (
            len(transitions), episodes, observations.shape[1],
            histories.shape[1], n_targets,
        )
    )

    print()
    print("-- per-target scope (all data) --")
    global_fit = {}
    global_heldout = []
    for column in range(n_targets):
        selected, weights, r2, _ = forward_select(
            scaled_inputs, scaled_target[:, column], max_terms=max_terms
        )
        global_fit[column] = (selected, weights, r2)
        fitted = rule_fit(
            scaled_inputs, scaled_target[:, column],
            max_terms=max_terms, seed=column,
        )
        heldout = float("nan") if fitted is None else fitted[3]
        if not np.isnan(heldout):
            global_heldout.append(heldout)
        state_inputs = [value for value in selected if value < observations.shape[1]]
        action_inputs = [
            value - observations.shape[1] for value in selected
            if value >= observations.shape[1]
        ]
        print(
            "target %d  state %s  action %s  fan_in %d  R2 %.4f  "
            "heldout %.4f" % (
                column, state_inputs, action_inputs, len(selected), r2,
                heldout,
            )
        )
    print(
        "avg fan-in %.2f  max fan-in %d  avg heldout %.4f" % (
            np.mean([len(value[0]) for value in global_fit.values()]),
            max(len(value[0]) for value in global_fit.values()),
            np.mean(global_heldout) if global_heldout else float("nan"),
        )
    )

    print()
    print("-- regime rule bank (80/20 within each cluster) --")
    print(
        "k  sizes(min/med/max)  supported  coverage  fan-out  "
        "shared  independent(log10)  heldout R2"
    )
    for k in regime_values:
        labels, centers = kmeans(scaled_inputs, k, seed=seed)
        supported, r2_in, r2_out, selected, weights, sizes = analyze_regimes(
            scaled_inputs, scaled_target, labels, k,
            min_support, r2_min, max_terms,
        )
        fan_out = supported.sum(axis=0)
        rule_count = int(fan_out.sum())
        shared = int(np.sum(np.any(supported, axis=1)))
        independent = int(np.prod(np.maximum(fan_out, 1)))
        weighted = []
        weights_sizes = []
        for cluster in range(k):
            valid = ~np.isnan(r2_out[cluster])
            if valid.any() and sizes[cluster] > 0:
                weighted.append(float(np.nanmean(r2_out[cluster][valid])))
                weights_sizes.append(int(sizes[cluster]))
        heldout = (
            float(np.average(weighted, weights=weights_sizes))
            if weighted else float("nan")
        )
        print(
            "%d  %d/%d/%d  %d  %d/%d  %s  %d  %.2f  %.3f" % (
                k, sizes.min(), int(np.median(sizes)), sizes.max(),
                rule_count, int(np.sum(fan_out > 0)), n_targets,
                ",".join(str(int(value)) for value in fan_out),
                shared, np.log10(max(independent, 1)), heldout,
            )
        )
        if k == regime_values[-1]:
            ratio, ambiguous = ambiguity_rate(
                scaled_inputs, scaled_target, centers, labels,
                supported, selected, weights,
            )
            print(
                "ambiguity: %d rows near a second regime, %.1f%% of them "
                "disagree by >0.1 scaled units" % (
                    ambiguous, 100.0 * ratio,
                )
            )
            ordered = np.sort(fan_out)[::-1]
            prefix = 1
            combos = []
            for count in range(1, n_targets + 1):
                prefix *= max(int(ordered[count - 1]), 1)
                combos.append(
                    "m=%d %.2f" % (count, np.log10(max(prefix, 1)))
                )
            print(
                "key-node budget (independent log10, shared=%d): %s" % (
                    shared, "  ".join(combos),
                )
            )
            rows = _search_rows(
                global_fit, selected, weights, centers, sizes,
                observations.shape[1], k,
            )
            print()
            print("-- actual search (one context, budget 500) --")
            print(
                "bank            keys  visited  pruned  applied  unknown  "
                "conflicts  shared"
            )
            for row in rows:
                print(
                    "%-15s %4d %8d %7d %8d %8d %10d  %s" % (
                        row["bank"], row["key_nodes"], row["visited"],
                        row["pruned"], row["applied"], row["unknown"],
                        row["conflicts"], row["shared"],
                    )
                )

    if timing:
        world, agent = fit(
            seed=seed, steps=steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45,
        )
        stats = graph_statistics(agent)
        print()
        print("-- existing prototype graph (same fit) --")
        print(
            "nodes %d  edges %d  out-degree %s  support %s  "
            "support>=2 %d  support>=5 %d  both-actions nodes %d" % (
                stats["nodes"], stats["edges"], stats["out_degree"],
                stats["support"], stats["support_ge2"],
                stats["support_ge5"], stats["both_actions"],
            )
        )
        print(
            "milestone edges %d  stages %d  keys %d" % (
                stats["milestones"], stats["stages"], stats["keys"],
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", default="1500,5000")
    parser.add_argument("--regimes", default="2,4,8,16")
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--r2-min", type=float, default=0.5)
    parser.add_argument("--max-terms", type=int, default=6)
    parser.add_argument(
        "--timing", action="store_true",
        help="also fit the shipped model and describe its prototype graph",
    )
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    regimes = [
        int(value) for value in args.regimes.split(",") if value.strip()
    ]
    for size in steps:
        analyze_size(
            args.seed, size, regimes, args.min_support, args.r2_min,
            args.max_terms, args.timing,
        )


if __name__ == "__main__":
    main()
