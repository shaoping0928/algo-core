"""Measure the death-side signal on the plans the planner actually evaluates.

The earlier quality probe sampled control-policy branches and asked whether a
single transition was labelled correctly.  The planner, however, scores whole
candidate action sequences and chooses one of them.  This probe therefore:

- replays the deterministic control policy to sampled states;
- runs the real ``policy_result`` search twice, once with the baseline
  objective and once with the death-side tri-state callback;
- records every candidate sequence through ``candidate_callback``;
- evaluates each candidate in the real world as an oracle;
- compares plan-level death prediction, first-death-step error, and whether
  the selected first action is the oracle-best action.

The metric core consumes only generic records (sequence, predicted/true first
death, scores).  The benchmark world supplies oracle rollouts only; no task
vocabulary or geometry enters the search or the rule learner.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from death_transition_quality import seed_spec
from latent_world import latent
from substrate_nodes import LAG, _curiosity_off, evaluate, fit, policy_result

CONFIG = dict(
    seed=100, steps=1500, epochs=1, max_nodes=1000,
    birth_threshold=0.45, dagger_rounds=1, dagger_frames=5000,
    dagger_cap=5000, dagger_horizon=8, step_stride=2,
    curiosity_weight=0.5, death_rules=True,
    death_rule_max_thresholds=256, death_rule_boundary_margin=0.01,
)


def status_callback(rules, obs_dim, full=False):
    """Return 1/0/-1 status per candidate and step.

    ``full=False`` is the death side the planner reads: 1 or unknown.  The
    full classifier is a diagnostic comparator and is never used for the
    planner's hard constraint.
    """
    def callback(states, _sequences, _step):
        observed = np.asarray(states, dtype=float)[:, :int(obs_dim)]
        if full:
            return np.asarray(rules.classify(observed), dtype=int)
        return np.asarray(rules.classify_death(observed), dtype=int)
    return callback


def oracle_plan(env, sequence):
    """True first-death step and final state of one fixed plan."""
    child = copy.deepcopy(env)
    first_death = -1
    for step, action in enumerate(sequence):
        child.step(int(action))
        if child.dead:
            first_death = step
            break
    return first_death, np.asarray(child.obs(), dtype=float), bool(child.dead)


def flatten_batches(batches):
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in batches[0]
    }


def plan_death_counts(predicted, truth):
    predicted = np.asarray(predicted, dtype=int)
    truth = np.asarray(truth, dtype=int)
    pred_dead = predicted >= 0
    true_dead = truth >= 0
    return dict(
        total=int(len(truth)),
        true_death=int(true_dead.sum()),
        predicted_death=int(pred_dead.sum()),
        tp=int(np.sum(pred_dead & true_dead)),
        fp=int(np.sum(pred_dead & ~true_dead)),
        fn=int(np.sum(~pred_dead & true_dead)),
        tn=int(np.sum(~pred_dead & ~true_dead)),
        exact=int(np.sum(true_dead & pred_dead & (truth == predicted))),
        within_one=int(np.sum(
            true_dead & pred_dead & (np.abs(truth - predicted) <= 1)
        )),
        false_negative=int(np.sum(true_dead & ~pred_dead)),
        false_positive=int(np.sum(~true_dead & pred_dead)),
    )


def action_best(sequences, first_death, horizon, feasible=None):
    """Best first death per first action; survival is better than any step.

    With a feasibility mask, the best *available* plan is used when the
    action has any feasible suffix; only when every suffix dies does the
    action fall back to its latest death.  This mirrors the search, which
    treats death as infeasible and unknown as still available.
    """
    first = np.asarray(sequences[:, 0], dtype=int)
    values = np.where(
        np.asarray(first_death, dtype=int) < 0,
        int(horizon) + 1,
        np.asarray(first_death, dtype=int),
    )
    best = {}
    for action in sorted(set(first.tolist())):
        mask = first == action
        if feasible is not None and np.any(mask & feasible):
            mask = mask & feasible
        best[action] = int(values[mask].max())
    return best


def action_rank(sequences, ranking_scores, actions, feasible=None):
    """Best (lowest) predicted ranking score per first action."""
    first = np.asarray(sequences[:, 0], dtype=int)
    scores = np.asarray(ranking_scores, dtype=float)
    ranked = {}
    for action in actions:
        mask = first == action
        if not np.any(mask):
            continue
        if feasible is not None and np.any(mask & feasible):
            mask = mask & feasible
        ranked[action] = float(scores[mask, 0].min())
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--tail", type=int, default=8,
                        help="also sample the last N frames of each episode")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--full-status", action="store_true",
                        help="use the safe side as a diagnostic tri-state; "
                             "the planner's hard constraint still reads only "
                             "the death side")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    config = dict(CONFIG)
    config["death_rule_boundary_margin"] = args.margin
    world, agent = fit(**config)
    rules = agent.death_rules
    agent.death_rule_hard = False
    death_status = status_callback(
        rules, agent.obs_dim, full=bool(args.full_status)
    )
    seeds = seed_spec(args.eval_seeds)

    plan_rows = []
    selected_rows = []
    contrast_rows = []
    examples = []
    decisions = 0

    for eval_seed in seeds:
        env_seed = int(eval_seed) * 10000
        with _curiosity_off(agent):
            frames, actions, _rows = evaluate(
                args.seed, world, agent, cap=5000, horizon=args.horizon,
                lag=LAG, step_stride=2, env_seed=env_seed, record=False,
                episode=0,
            )
        if frames < 2:
            continue
        points = sorted(set(
            list(np.linspace(2, frames - 2, args.branches).astype(int))
            + list(range(max(1, frames - args.tail), frames))
        ))
        for point in points:
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

            baseline = policy_result(
                env, world, agent, prefix, args.horizon, lag=LAG,
                step_stride=2,
            )
            tri_batches = []
            tri = policy_result(
                env, world, agent, prefix, args.horizon, lag=LAG,
                step_stride=2, death_status_batch=death_status,
                candidate_callback=tri_batches.append,
            )
            if not tri_batches:
                continue
            trace = flatten_batches(tri_batches)
            sequences = trace["sequences"]
            predicted_first = trace["first_death"]
            true_first = np.empty(len(sequences), dtype=int)
            for index, sequence in enumerate(sequences):
                true_first[index], _state, _dead = oracle_plan(env, sequence)

            plan_rows.append(plan_death_counts(predicted_first, true_first))

            best_true = action_best(sequences, true_first, args.horizon)
            best_pred = action_best(
                sequences, predicted_first, args.horizon,
                feasible=trace["feasible"],
            )
            ranked_pred = action_rank(
                sequences, trace["ranking_scores"], sorted(best_true),
                feasible=trace["feasible"],
            )
            selected_action = int(tri.actions[0])
            baseline_action = int(baseline.actions[0])
            best_value = max(best_true.values())
            best_actions = {
                action for action, value in best_true.items()
                if value == best_value
            }
            selected_true, selected_state, _dead = oracle_plan(
                env, tri.actions
            )
            baseline_true, _baseline_state, _dead = oracle_plan(
                env, baseline.actions
            )
            selected_oracle_best = selected_action in best_actions
            baseline_oracle_best = baseline_action in best_actions
            decisions += 1
            selected_rows.append(dict(
                tri_oracle_best=bool(selected_oracle_best),
                baseline_oracle_best=bool(baseline_oracle_best),
                tri_predicted_death=bool(tri.first_death >= 0),
                baseline_predicted_death=bool(baseline.first_death >= 0),
                selected_true_death=int(selected_true),
                baseline_true_death=int(baseline_true),
                selected_predicted_death=int(tri.first_death),
                selected_unknown_count=int(tri.unknown_count),
                selected_first_unknown=int(tri.first_unknown),
                best_actions=sorted(int(action) for action in best_actions),
                selected_action=selected_action,
                baseline_action=baseline_action,
            ))
            if len(best_true) == 2 and best_true[0] != best_true[1]:
                saving = 0 if best_true[0] > best_true[1] else 1
                fatal = 1 - saving
                pred_saving = best_pred.get(saving, -1)
                pred_fatal = best_pred.get(fatal, -1)
                rank_saving = ranked_pred.get(saving, float("inf"))
                rank_fatal = ranked_pred.get(fatal, float("inf"))
                contrast_rows.append(dict(
                    saving=int(saving),
                    true_saving=int(best_true[saving]),
                    true_fatal=int(best_true[fatal]),
                    pred_saving=int(pred_saving),
                    pred_fatal=int(pred_fatal),
                    death_contrast_correct=bool(pred_saving > pred_fatal),
                    score_contrast_correct=bool(rank_saving < rank_fatal),
                    selected_action=selected_action,
                    selected_is_saving=bool(selected_action == saving),
                    baseline_is_saving=bool(baseline_action == saving),
                ))
            selected_missed = bool(
                selected_true >= 0 and tri.first_death < 0
            )
            if len(examples) < 12 and (
                selected_missed or not selected_oracle_best
            ):
                examples.append(dict(
                    eval_seed=int(eval_seed),
                    point=int(point),
                    selected_action=selected_action,
                    baseline_action=baseline_action,
                    selected_missed=selected_missed,
                    selected_oracle_best=bool(selected_oracle_best),
                    best_actions=sorted(
                        int(action) for action in best_actions
                    ),
                    predicted_first_death=int(tri.first_death),
                    selected_true_first_death=int(selected_true),
                    selected_unknown_count=int(tri.unknown_count),
                    sequence=[int(value) for value in tri.actions],
                    terminal_state=[
                        float(value) for value in selected_state
                    ],
                ))

    plan_total = {
        key: int(sum(row[key] for row in plan_rows))
        for key in plan_rows[0]
    } if plan_rows else {}
    if plan_total:
        tp = plan_total["tp"]
        fp = plan_total["fp"]
        fn = plan_total["fn"]
        tn = plan_total["tn"]
        plan_total["precision"] = (
            float(tp) / float(tp + fp) if tp + fp else 0.0
        )
        plan_total["recall"] = (
            float(tp) / float(tp + fn) if tp + fn else 0.0
        )
        plan_total["false_positive_rate"] = (
            float(fp) / float(fp + tn) if fp + tn else 0.0
        )
        plan_total["within_one_of_predicted"] = (
            float(plan_total["within_one"]) / float(tp)
            if tp else 0.0
        )
    selection = dict(
        decisions=decisions,
        tri_oracle_best=int(sum(
            row["tri_oracle_best"] for row in selected_rows
        )),
        baseline_oracle_best=int(sum(
            row["baseline_oracle_best"] for row in selected_rows
        )),
        tri_predicted_death=int(sum(
            row["tri_predicted_death"] for row in selected_rows
        )),
        selected_true_death=int(sum(
            row["selected_true_death"] >= 0 for row in selected_rows
        )),
        selected_missed=int(sum(
            row["selected_true_death"] >= 0
            and row["selected_predicted_death"] < 0
            for row in selected_rows
        )),
        mean_selected_unknown_count=float(np.mean([
            row["selected_unknown_count"] for row in selected_rows
        ])) if selected_rows else 0.0,
        tri_selected_survives=int(sum(
            row["selected_true_death"] < 0 for row in selected_rows
        )),
        baseline_selected_survives=int(sum(
            row["baseline_true_death"] < 0 for row in selected_rows
        )),
        tri_later=int(sum(
            (args.horizon + 1 if row["selected_true_death"] < 0
             else row["selected_true_death"])
            > (args.horizon + 1 if row["baseline_true_death"] < 0
               else row["baseline_true_death"])
            for row in selected_rows
        )),
        tri_earlier=int(sum(
            (args.horizon + 1 if row["selected_true_death"] < 0
             else row["selected_true_death"])
            < (args.horizon + 1 if row["baseline_true_death"] < 0
               else row["baseline_true_death"])
            for row in selected_rows
        )),
        tri_ties=int(sum(
            (args.horizon + 1 if row["selected_true_death"] < 0
             else row["selected_true_death"])
            == (args.horizon + 1 if row["baseline_true_death"] < 0
                else row["baseline_true_death"])
            for row in selected_rows
        )),
    )
    contrast = dict(
        comparable=len(contrast_rows),
        death_contrast_correct=int(sum(
            row["death_contrast_correct"] for row in contrast_rows
        )),
        score_contrast_correct=int(sum(
            row["score_contrast_correct"] for row in contrast_rows
        )),
        selected_saving=int(sum(
            row["selected_is_saving"] for row in contrast_rows
        )),
        baseline_saving=int(sum(
            row["baseline_is_saving"] for row in contrast_rows
        )),
    )
    summary = dict(
        plan_death=plan_total,
        selection=selection,
        action_contrast=contrast,
        examples=examples,
    )
    print(json.dumps(summary, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
