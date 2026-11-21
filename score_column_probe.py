"""Which score column misranks? Leave-one-column-out on real decisions.

For each decision the probe records the candidate plans' full score vectors
(the same columns the planner sums), the oracle survival of every plan
(replaying it in a copy of the world), and whether the planner's pick is
decision-relevant (a plan with strictly better true survival exists).

For misses it re-selects under four ablations - dropping risk, value,
curiosity or Q from the scalarized cost - and reports which ablation lands
on the oracle-best plan.  This is a diagnostic only: nothing is learned and
the default planner is unchanged.

Usage::

    python score_column_probe.py --seed 101 --eval-seeds 120-124
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from latent_world import latent
from planner_rollout_quality import flatten_batches, oracle_plan
from substrate_nodes import (
    LAG,
    _curiosity_off,
    evaluate,
    fit,
    policy_result,
)

COLUMNS = {
    "risk": 2,
    "value": 3,
    "curiosity": 4,
    "q": 5,
}


def decision_points(frames, branches, tail):
    if frames < 2:
        return []
    points = set(
        int(value) for value in np.linspace(2, frames - 2, branches)
    )
    points.update(range(max(1, frames - tail), frames))
    return sorted(points)


def run(args):
    world, agent = fit(
        seed=args.seed, steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
    )
    rows = []
    with _curiosity_off(agent):
        for eval_seed in args.eval_seeds:
            env_seed = int(eval_seed) * 10000
            frames, actions, _ = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=args.horizon,
                lag=LAG, step_stride=2, env_seed=env_seed,
            )
            for point in decision_points(frames, args.branches, args.tail):
                env = latent[LAG](env_seed)
                for action in actions[:point]:
                    if env.dead:
                        break
                    env.step(int(action))
                if env.dead:
                    continue
                recent = list(reversed(actions[max(0, point - LAG):point]))
                prefix = np.zeros(LAG, dtype=float)
                prefix[:len(recent[:LAG])] = recent[:LAG]
                batches = []
                result = policy_result(
                    env, world, agent, prefix, args.horizon, lag=LAG,
                    step_stride=2, candidate_callback=batches.append,
                )
                if not batches:
                    continue
                trace = flatten_batches(batches)
                scores = np.asarray(trace["scores"], dtype=float)
                feasible = (
                    np.asarray(trace["feasible"], dtype=bool)
                    & np.asarray(trace["finite"], dtype=bool)
                )
                if not feasible.any():
                    continue
                sequences = np.asarray(trace["sequences"], dtype=float)
                true_first = np.empty(len(sequences), dtype=int)
                for index, sequence in enumerate(sequences):
                    true_first[index], _state, _dead = oracle_plan(
                        env, sequence
                    )
                predicted_first = np.asarray(
                    trace["first_death"], dtype=int
                )
                survival = np.where(
                    true_first < 0, args.horizon + 1, true_first
                )
                best = int(survival[feasible].max())
                best_all = int(survival.max())
                chosen = int(np.flatnonzero(feasible)[
                    np.argmin(scores[feasible, 0])
                ])
                feasible_costs = scores[feasible, 0]
                best_cost = float(feasible_costs.min())
                ties = int(np.sum(
                    np.abs(feasible_costs - best_cost) <= 1e-9
                ))
                near_ties = int(np.sum(
                    np.abs(feasible_costs - best_cost) <= 0.01
                ))
                oracle_all = np.flatnonzero(survival == best_all)
                oracle_all_feasible = bool(
                    feasible[oracle_all].any()
                ) if len(oracle_all) else False
                oracle_index = int(oracle_all[0]) if len(oracle_all) else 0
                entry = dict(
                    eval_seed=int(eval_seed), point=int(point),
                    candidates=int(feasible.sum()), best_survival=best,
                    best_all_survival=best_all,
                    oracle_all_feasible=oracle_all_feasible,
                    chosen_survival=int(survival[chosen]),
                    chosen_predicted_death=int(predicted_first[chosen]),
                    oracle_predicted_death=int(
                        predicted_first[oracle_index]
                    ),
                    chosen_true_death=int(true_first[chosen]),
                    oracle_true_death=int(true_first[oracle_index]),
                    miss=bool(int(survival[chosen]) < best),
                    miss_vs_all=bool(
                        int(survival[chosen]) < best_all
                    ),
                    cost_spread=float(
                        feasible_costs.max() - best_cost
                    ),
                    exact_ties=ties,
                    near_ties=near_ties,
                    tie_share=float(ties) / float(len(feasible_costs)),
                    distinct_costs=int(len(np.unique(
                        np.round(feasible_costs, 9)
                    ))),
                )
                if entry["miss_vs_all"] and not oracle_all_feasible:
                    entry["failure"] = "model_infeasible_oracle"
                elif entry["miss"]:
                    entry["failure"] = "score_misrank"
                if entry.get("failure") == "score_misrank":
                    ablations = {}
                    for name, column in COLUMNS.items():
                        cost = scores[:, 0] - scores[:, column]
                        index = int(np.flatnonzero(feasible)[
                            np.argmin(cost[feasible])
                        ])
                        ablations["drop_" + name] = dict(
                            survival=int(survival[index]),
                            oracle=bool(int(survival[index]) == best),
                        )
                    index = int(np.flatnonzero(feasible)[
                        np.argmin(scores[feasible, 2])
                    ])
                    ablations["only_risk"] = dict(
                        survival=int(survival[index]),
                        oracle=bool(int(survival[index]) == best),
                    )
                    cost = scores[:, 0] - scores[:, 3] - scores[:, 4]
                    index = int(np.flatnonzero(feasible)[
                        np.argmin(cost[feasible])
                    ])
                    ablations["drop_value_curiosity"] = dict(
                        survival=int(survival[index]),
                        oracle=bool(int(survival[index]) == best),
                    )
                    entry["ablations"] = ablations
                rows.append(entry)
    misses = [row for row in rows if row["miss"]]
    misranks = [row for row in rows if row.get("failure") == "score_misrank"]
    infeasible = [
        row for row in rows
        if row.get("failure") == "model_infeasible_oracle"
    ]
    counts = {key: 0 for key in list(COLUMNS) + ["value_curiosity", "only_risk"]}
    fixed = 0
    for row in misses:
        any_fix = False
        for key in COLUMNS:
            if row["ablations"]["drop_" + key]["oracle"]:
                counts[key] += 1
                any_fix = True
        if row["ablations"]["drop_value_curiosity"]["oracle"]:
            counts["value_curiosity"] += 1
            any_fix = True
        if row["ablations"]["only_risk"]["oracle"]:
            counts["only_risk"] += 1
            any_fix = True
        if any_fix:
            fixed += 1
    model_safer = 0
    for row in misses:
        chosen_pred = (
            args.horizon + 1 if row["chosen_predicted_death"] < 0
            else row["chosen_predicted_death"]
        )
        oracle_pred = (
            args.horizon + 1 if row["oracle_predicted_death"] < 0
            else row["oracle_predicted_death"]
        )
        if chosen_pred > oracle_pred:
            model_safer += 1
    return dict(
        seed=int(args.seed),
        decisions=len(rows),
        misses=len(misses),
        misranks=len(misranks),
        model_infeasible_oracle=len(infeasible),
        miss_vs_all=sum(1 for row in rows if row.get("miss_vs_all")),
        miss_rate=(float(len(misses)) / len(rows)) if rows else 0.0,
        misses_where_model_thought_chosen_safer=model_safer,
        mean_tie_share=(
            float(np.mean([row["tie_share"] for row in rows]))
            if rows else 0.0
        ),
        decisions_with_over_half_ties=sum(
            1 for row in rows if row["tie_share"] > 0.5
        ),
        mean_cost_spread=(
            float(np.mean([row["cost_spread"] for row in rows]))
            if rows else 0.0
        ),
        mean_distinct_costs=(
            float(np.mean([row["distinct_costs"] for row in rows]))
            if rows else 0.0
        ),
        oracle_fix_counts=counts,
        misses_fixed_by_any_ablation=fixed,
        rows=rows,
    )


def print_summary(report):
    print()
    print(
        "== seed %d: %d decisions, %d feasible misses, %d score misranks, "
        "%d model-infeasible oracle, %d misses vs all (%.0f%%) =="
        % (report["seed"], report["decisions"], report["misses"],
           report["misranks"], report["model_infeasible_oracle"],
           report["miss_vs_all"], 100.0 * report["miss_rate"])
    )
    print("  misses fixed by dropping one column (landing on oracle-best):")
    for key in ("risk", "value", "curiosity", "q"):
        print("    drop %-10s %d" % (key, report["oracle_fix_counts"][key]))
    print("    drop value+curiosity %d"
          % report["oracle_fix_counts"]["value_curiosity"])
    print("    only risk            %d"
          % report["oracle_fix_counts"]["only_risk"])
    print("  any ablation fixes: %d / %d"
          % (report["misses_fixed_by_any_ablation"], report["misses"]))
    print(
        "  risk resolution: mean tie share %.2f  decisions >50%% ties %d / %d"
        % (report["mean_tie_share"],
           report["decisions_with_over_half_ties"], report["decisions"])
    )
    print(
        "  cost spread mean %.3f  distinct costs/decision %.2f"
        % (report["mean_cost_spread"], report["mean_distinct_costs"])
    )
    print(
        "  misses where the model thought the chosen plan was safer: %d / %d"
        % (report["misses_where_model_thought_chosen_safer"],
           report["misses"])
    )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--tail", type=int, default=12)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    args.eval_seeds = parse_seed_spec(args.eval_seeds)
    report = run(args)
    print_summary(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
