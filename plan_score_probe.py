"""B0.5: diagnose the plan/suffix score that hides the oracle-best suffix.

B0 showed that eight of seed 101's ten decision-relevant misses are not
first-action problems: the selected and oracle-best plans either start with
the same action, or both first actions can reach the same best true survival
when each is given its own best suffix.  The remaining question is which part
of the fine score puts the better suffix behind the selected one.

This probe reuses the B0 decision protocol and, for every relevant miss,
records:

* the selected, same-action-best, and oracle-best score vectors;
* the regret decomposition into action regret (best under the selected first
  action vs best overall) and suffix regret (selected plan vs best under the
  selected first action);
* global and same-action ablations that drop risk, value, curiosity, or q,
  plus a few reference arms;
* the score rank of the same-action-best suffix among feasible candidates.

It trains no new model and changes no default path.  If no single column
recovers the best suffix, the next hypothesis is world-prediction drift or a
plan-level endpoint readout, not another scalar weight.

Usage::

    python plan_score_probe.py --seed 101 --eval-seeds 120-124 \
        --branches 10 --tail 8 --cache b0-101.pkl \
        --report plan-score-101.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from decision_b0_probe import run as b0_run
from trajectory_search import best_candidate_index

COLUMNS = (("risk", 2), ("value", 3), ("curiosity", 4), ("q", 5))

ABLATIONS = (
    ("drop_risk", ("drop", 2)),
    ("drop_value", ("drop", 3)),
    ("drop_curiosity", ("drop", 4)),
    ("drop_q", ("drop", 5)),
    ("drop_value_curiosity", ("drop", 3, 4)),
    ("only_risk", ("only", 2)),
    ("only_value", ("only", 3)),
)


def ablation_cost(scores, spec):
    """Cost vector for one ablation; lower is better."""
    scores = np.asarray(scores, dtype=float)
    cost = scores[:, 0].copy()
    if spec[0] == "drop":
        for column in spec[1:]:
            cost -= scores[:, column]
    elif spec[0] == "only":
        cost = scores[:, spec[1]].copy()
    else:
        raise ValueError("unknown ablation %r" % (spec,))
    return cost


def select(scores, mask, spec):
    cost = ablation_cost(scores, spec)
    return int(best_candidate_index(cost.reshape(-1, 1), mask))


def score_vector(column, index):
    return float(column[int(index)])


def diagnose(record, row):
    """Score-column and regret decomposition for one decision-relevant miss."""
    sequences = np.asarray(record["sequences"], dtype=int)
    scores = np.asarray(record["scores"], dtype=float)
    feasible = np.asarray(record["feasible"], dtype=bool)
    finite = np.asarray(record["trace"]["finite"], dtype=bool)
    usable = feasible & finite
    true_value = np.asarray(record["true_value"], dtype=int)
    selected_action = int(record["selected_action"])
    oracle_index = int(record["oracle_index"])
    best_value = int(record["best_value"])
    selected_value = int(row["selected_value"])

    selected_match = np.flatnonzero(np.all(
        sequences == np.asarray(record["result"].actions, dtype=int), axis=1
    ))
    if not len(selected_match):
        return dict(
            eval_seed=record["eval_seed"], point=record["point"],
            **{"class": row["class"]},
            error="selected plan not found in trace",
        )
    selected_index = int(selected_match[0])

    same_mask = usable & (sequences[:, 0] == selected_action)
    if not same_mask.any():
        return dict(
            eval_seed=record["eval_seed"], point=record["point"],
            **{"class": row["class"]},
            error="selected action has no feasible plan",
        )
    same_best_value = int(true_value[same_mask].max())
    same_best_mask = same_mask & (true_value == same_best_value)
    same_best_index = int(best_candidate_index(scores, same_best_mask))
    same_best_score = float(scores[same_best_index, 0])
    same_best_rank = int(np.sum(scores[same_mask, 0] < same_best_score))
    selected_score = float(scores[selected_index, 0])

    action_regret = int(best_value - same_best_value)
    suffix_regret = int(same_best_value - selected_value)
    if action_regret < 0 or suffix_regret < 0:
        raise ValueError("negative regret decomposition")

    vectors = {}
    for name, column in COLUMNS:
        vectors[name] = dict(
            selected=score_vector(scores[:, column], selected_index),
            same_best=score_vector(scores[:, column], same_best_index),
            oracle=score_vector(scores[:, column], oracle_index),
            same_best_minus_selected=float(
                scores[same_best_index, column] - scores[selected_index, column]
            ),
            oracle_minus_selected=float(
                scores[oracle_index, column] - scores[selected_index, column]
            ),
        )

    gap = same_best_score - selected_score
    positive = {
        name: max(0.0, vectors[name]["same_best_minus_selected"])
        for name, _column in COLUMNS
    }
    total_positive = float(sum(positive.values()))
    shares = {
        name: (value / total_positive if total_positive > 0.0 else 0.0)
        for name, value in positive.items()
    }

    global_arms = {}
    same_arms = {}
    for name, spec in ABLATIONS:
        global_index = select(scores, usable, spec)
        same_index = select(scores, same_mask, spec)
        global_cost = ablation_cost(scores, spec)
        global_spread = float(
            global_cost[usable].max() - global_cost[usable].min()
        ) if usable.any() else 0.0
        same_cost = ablation_cost(scores, spec)
        same_spread = float(
            same_cost[same_mask].max() - same_cost[same_mask].min()
        ) if same_mask.any() else 0.0
        global_arms[name] = dict(
            true_value=int(true_value[global_index]),
            action=int(sequences[global_index, 0]),
            hits_global_best=bool(true_value[global_index] == best_value),
            regret=int(best_value - true_value[global_index]),
            cost_spread=global_spread,
            degenerate=bool(global_spread <= 1e-9),
        )
        same_arms[name] = dict(
            true_value=int(true_value[same_index]),
            action=int(sequences[same_index, 0]),
            hits_same_action_best=bool(
                true_value[same_index] == same_best_value
            ),
            suffix_regret=int(
                same_best_value - true_value[same_index]
            ),
            cost_spread=same_spread,
            degenerate=bool(same_spread <= 1e-9),
        )

    global_fixes = [
        name for name, arm in global_arms.items()
        if arm["hits_global_best"] and not arm["degenerate"]
    ]
    same_fixes = [
        name for name, arm in same_arms.items()
        if arm["hits_same_action_best"] and not arm["degenerate"]
    ]
    raw_global_fixes = [
        name for name, arm in global_arms.items()
        if arm["hits_global_best"]
    ]
    raw_same_fixes = [
        name for name, arm in same_arms.items()
        if arm["hits_same_action_best"]
    ]
    baseline_global_index = int(best_candidate_index(scores, usable))
    baseline_same_index = int(best_candidate_index(scores, same_mask))
    return dict(
        eval_seed=record["eval_seed"], point=record["point"],
        **{"class": row["class"]},
        selected_action=selected_action,
        oracle_action=int(record["oracle_action"]),
        selected_value=selected_value,
        same_best_value=same_best_value,
        oracle_best_value=best_value,
        total_regret=int(row["regret"]),
        action_regret=action_regret,
        suffix_regret=suffix_regret,
        selected_index=selected_index,
        same_best_index=same_best_index,
        oracle_index=oracle_index,
        selected_score=selected_score,
        same_best_score=same_best_score,
        same_best_score_gap=float(gap),
        same_best_rank=same_best_rank,
        baseline_same_index=baseline_same_index,
        baseline_same_value=int(true_value[baseline_same_index]),
        baseline_global_index=baseline_global_index,
        baseline_global_value=int(true_value[baseline_global_index]),
        columns=vectors,
        positive_score_shares=shares,
        global_arms=global_arms,
        same_action_arms=same_arms,
        global_fixes=global_fixes,
        same_action_fixes=same_fixes,
        raw_global_fixes=raw_global_fixes,
        raw_same_action_fixes=raw_same_fixes,
    )


def summarize(diagnostics):
    valid = [item for item in diagnostics if "error" not in item]
    errors = [item for item in diagnostics if "error" in item]
    plan_level = [
        item for item in valid
        if item.get("action_regret") == 0
    ]
    action_level = [
        item for item in valid
        if item.get("action_regret", 0) > 0
    ]
    def mode_counts(items, key):
        counts = {name: 0 for name, _spec in ABLATIONS}
        any_count = 0
        for item in items:
            fixes = item.get(key, [])
            if fixes:
                any_count += 1
            for name in fixes:
                counts[name] += 1
        return counts, any_count
    global_counts, global_any = mode_counts(
        valid, "global_fixes"
    )
    same_counts, same_any = mode_counts(
        valid, "same_action_fixes"
    )
    plan_global_counts, plan_global_any = mode_counts(
        plan_level, "global_fixes"
    )
    plan_same_counts, plan_same_any = mode_counts(
        plan_level, "same_action_fixes"
    )
    columns = {}
    active_columns = []
    for name, _column in COLUMNS:
        nonzero = sum(
            abs(item["columns"][name]["same_best_minus_selected"]) > 1e-9
            for item in plan_level
        )
        columns[name] = dict(
            nonzero_plan_level=int(nonzero),
            mean_same_best_minus_selected=float(np.mean([
                item["columns"][name]["same_best_minus_selected"]
                for item in plan_level
            ])) if plan_level else None,
            mean_oracle_minus_selected=float(np.mean([
                item["columns"][name]["oracle_minus_selected"]
                for item in valid
            ])) if valid else None,
            mean_positive_share=float(np.mean([
                item["positive_score_shares"][name]
                for item in plan_level
            ])) if plan_level else None,
        )
        if nonzero:
            active_columns.append(name)
    if len(active_columns) == 1 and plan_same_any < 5:
        verdict = "single_active_column_misranks"
    elif plan_same_any >= 5:
        verdict = "single_column"
    elif plan_same_any >= 1:
        verdict = "partial_single_column"
    else:
        verdict = "no_single_column"
    return dict(
        misses=int(len(valid)),
        errors=int(len(errors)),
        plan_level=int(len(plan_level)),
        action_level=int(len(action_level)),
        plan_level_suffix_regret_positive=int(sum(
            item["suffix_regret"] > 0 for item in plan_level
        )),
        plan_level_same_best_rank_mean=float(np.mean([
            item["same_best_rank"] for item in plan_level
        ])) if plan_level else None,
        plan_level_same_best_rank_median=float(np.median([
            item["same_best_rank"] for item in plan_level
        ])) if plan_level else None,
        global_fix_counts=global_counts,
        global_any_fix=global_any,
        same_action_fix_counts=same_counts,
        same_action_any_fix=same_any,
        plan_level_global_fix_counts=plan_global_counts,
        plan_level_global_any_fix=plan_global_any,
        plan_level_same_action_fix_counts=plan_same_counts,
        plan_level_same_action_any_fix=plan_same_any,
        columns=columns,
        active_columns=active_columns,
        verdict=verdict,
        rows=diagnostics,
    )


def print_report(summary):
    print(
        "== B0.5 plan/suffix score: %d misses, %d plan-level, "
        "%d action-level ==" % (
            summary["misses"], summary["plan_level"],
            summary["action_level"],
        )
    )
    print(
        "   plan-level suffix regret >0: %d / %d; "
        "same-action-best score rank median %s mean %s" % (
            summary["plan_level_suffix_regret_positive"],
            summary["plan_level"],
            "None" if summary["plan_level_same_best_rank_median"] is None
            else "%.1f" % summary["plan_level_same_best_rank_median"],
            "None" if summary["plan_level_same_best_rank_mean"] is None
            else "%.1f" % summary["plan_level_same_best_rank_mean"],
        )
    )
    print(
        "   global fixes any %d / %d  %s" % (
            summary["global_any_fix"], summary["misses"],
            summary["global_fix_counts"],
        )
    )
    print(
        "   same-action fixes any %d / %d  %s" % (
            summary["same_action_any_fix"], summary["misses"],
            summary["same_action_fix_counts"],
        )
    )
    print(
        "   plan-level global fixes any %d / %d  %s" % (
            summary["plan_level_global_any_fix"],
            summary["plan_level"],
            summary["plan_level_global_fix_counts"],
        )
    )
    print(
        "   plan-level same-action fixes any %d / %d  %s" % (
            summary["plan_level_same_action_any_fix"],
            summary["plan_level"],
            summary["plan_level_same_action_fix_counts"],
        )
    )
    print(
        "   active columns: %s (nondegenerate fixes only)" % (
            summary["active_columns"],
        )
    )
    for name, values in summary["columns"].items():
        print(
            "   column %-10s nonzero %d/%d  "
            "same-best-minus-selected %+.4f  "
            "oracle-minus-selected %+.4f  positive share %.2f" % (
                name,
                values["nonzero_plan_level"],
                summary["plan_level"],
                values["mean_same_best_minus_selected"] or 0.0,
                values["mean_oracle_minus_selected"] or 0.0,
                values["mean_positive_share"] or 0.0,
            )
        )
    print("   verdict: %s" % summary["verdict"])
    for item in summary["rows"]:
        if "error" in item:
            print(
                "   seed %d @%d: %s  ERROR %s" % (
                    item["eval_seed"], item["point"], item["class"],
                    item["error"],
                )
            )
            continue
        deltas = " ".join(
            "%s%+.3f" % (
                name[0],
                item["columns"][name]["same_best_minus_selected"],
            ) for name, _column in COLUMNS
        )
        print(
            "   seed %d @%d: %-22s action-regret %d suffix-regret %d  "
            "same-best rank %d score-gap %.4f  deltas %s  "
            "global %s  same %s" % (
                item["eval_seed"], item["point"], item["class"],
                item["action_regret"], item["suffix_regret"],
                item["same_best_rank"], item["same_best_score_gap"],
                deltas, item["global_fixes"], item["same_action_fixes"],
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--cache", default="")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    b0_args = argparse.Namespace(
        seed=args.seed, eval_seeds=args.eval_seeds,
        branches=args.branches, tail=args.tail,
        horizon=args.horizon, margin=args.margin,
        neighbours=64, min_support=4, min_episodes=2,
        ood_quantile=0.95, ood_high_quantile=0.99,
        cache=args.cache, refit=bool(args.refit), report="",
    )
    records = []
    b0_summary = b0_run(b0_args, trace_out=records)
    relevant = {
        (row["eval_seed"], row["point"]): row
        for row in b0_summary["rows"] if row["decision_relevant"]
    }
    diagnostics = []
    for record in records:
        key = (record["eval_seed"], record["point"])
        if key not in relevant:
            continue
        diagnostics.append(diagnose(record, relevant[key]))
    summary = summarize(diagnostics)
    print_report(summary)
    summary["b0"] = dict(
        decisions=b0_summary["decisions"],
        misses=b0_summary["misses"],
        plan_level=b0_summary["plan_level_misses"],
        positive_action_contrast=b0_summary[
            "positive_action_contrast_misses"
        ],
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return summary


if __name__ == "__main__":
    main()
