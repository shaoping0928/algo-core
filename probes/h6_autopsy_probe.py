"""P0-F autopsy: at the h6 executor's own decisions, prediction or decision?

The horizon A/B showed the 6-step executor loses 57.6 frames on average and
collapses seed 100 to a 72-frame floor with an unchanged flap rate - "timing
wrong", not "flapping more".  This probe runs the actual h6 evaluation
episodes and, at sampled decision points along the real trajectory, opens
the search: every candidate the executor scored, the score columns it used,
and the real-world oracle fate of each candidate.  The same states are
re-scored with the h8 model as a control, so "the extra two steps" can be
separated from "the score is wrong at the same states".

The benchmark protocol is honoured exactly (no death rules, decision=search,
curiosity off, margin/veto off); nothing is wired or changed.

Failure taxonomy per sampled decision:

- ``in_window_missed``   the selected plan dies inside the 6-step window and
  a better candidate existed (regret > 0): the score failed to rank the
  escape above a plan it could see dying.  Sub-split by the oracle-best
  plan's score gap: near-tie (<= 0.03), challenge (<= 0.1), large, or vetoed
  as infeasible.
- ``in_window_doomed``   the selected plan dies inside the window and every
  candidate dies too (regret = 0): no in-window escape exists; only a
  longer view (or an event/stage layer) could help.
- ``replan_death``       the selected plan is oracle-safe for its whole
  window, yet the episode died within `horizon` frames of this decision:
  the receding horizon kept re-planning into death the window never saw.
- ``post_window_death``  window-safe, death landed just past the window
  edge (within 2x horizon).
- ``fine``               window-safe and death is not imminent.

Prediction quality of the danger readout is measured directly: the pooled
rank-AUC of the score's risk column against the oracle "dies inside the
window" label over all scored candidates, plus the per-decision risk
contrast between the two first actions.

Usage::

    python h6_autopsy_probe.py --seed 100 --arms 6,8 \
        --report h6-autopsy-100.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from death_transition_quality import seed_spec
from endpoint_correction_probe import rank_auc
from executor_horizon_probe import horizon_config
from latent_world import latent
from planner_rollout_quality import (
    action_best, flatten_batches, oracle_best_details, oracle_plan,
)
from probe_cache import load_or_fit
from substrate_nodes import LAG, _curiosity_off, evaluate, policy_result
from trajectory_search import best_candidate_index

RISK_COLUMN = 2


def decision_points(frames, branches, tail):
    if frames < 2:
        return []
    points = set(
        int(value) for value in np.linspace(2, frames - 2, branches)
    )
    points.update(range(max(1, frames - tail), frames))
    return sorted(points)


def open_decision(env, world, agent, prefix, horizon, actions, point):
    """Trace one decision: model candidates, real oracle fates, score columns."""
    batches = []
    with _curiosity_off(agent):
        result = policy_result(
            env, world, agent, prefix, int(horizon), lag=LAG,
            step_stride=2, candidate_callback=batches.append,
        )
    if not batches:
        return None
    trace = flatten_batches(batches)
    sequences = np.asarray(trace["sequences"], dtype=int)
    scores = np.asarray(trace["scores"], dtype=float)
    ranking = np.asarray(trace["ranking_scores"], dtype=float)
    finite = np.asarray(trace["finite"], dtype=bool)
    true_first = np.asarray([
        oracle_plan(env, sequence)[0] for sequence in sequences
    ], dtype=int)
    selected_plan = np.asarray(result.actions, dtype=int)
    matches = np.flatnonzero((sequences == selected_plan).all(axis=1))
    selected_index = int(matches[0]) if len(matches) else None
    selected_true, _state, _dead = oracle_plan(env, selected_plan)
    true_value = np.where(true_first < 0, horizon + 1, true_first)
    best_value = int(true_value.max()) if len(true_value) else horizon + 1
    selected_value = int(horizon + 1 if selected_true < 0 else selected_true)
    selected_score = (
        float(ranking[selected_index, 0])
        if selected_index is not None else None
    )
    selected_risk = (
        float(scores[selected_index, RISK_COLUMN])
        if selected_index is not None else None
    )
    details = oracle_best_details(
        trace, sequences, true_first, horizon, selected_score or 0.0,
    )
    best_true = action_best(sequences, true_first, horizon)
    best_by_action = {}
    risk_by_action = {}
    for action in sorted(set(sequences[:, 0].tolist())):
        mask = (sequences[:, 0] == action) & finite
        if not mask.any():
            continue
        best_by_action[int(action)] = int(true_value[mask].max())
        risk_by_action[int(action)] = float(np.min(scores[mask, RISK_COLUMN]))
    row = dict(
        horizon=int(horizon),
        candidates=int(len(sequences)),
        finite=int(finite.sum()),
        selected_action=int(selected_plan[0]),
        selected_true=int(selected_true),
        selected_value=selected_value,
        selected_score=selected_score,
        selected_risk=selected_risk,
        best_value=int(best_value),
        regret=int(best_value - selected_value),
        oracle_best_action=int(details["oracle_best_action"]),
        oracle_best_feasible=bool(details["oracle_best_feasible"]),
        oracle_best_rank=int(details["oracle_best_rank"]),
        oracle_best_score_gap=float(details["oracle_best_score_gap"]),
        true_by_action=best_by_action,
        risk_by_action=risk_by_action,
        risk_auc=rank_auc(scores[:, RISK_COLUMN], (true_first >= 0).astype(float))
        if len(set(true_first.tolist())) > 1 else None,
        plan=[int(value) for value in selected_plan],
        plan_followed=(
            bool(len(selected_plan) > 1 and point + 1 < len(actions)
                 and int(selected_plan[1]) == int(actions[point + 1]))
            if point + 1 < len(actions) else None
        ),
    )
    return row


def classify_row(row, horizon):
    if row["selected_true"] >= 0:
        if row["regret"] > 0:
            gap = float(row["oracle_best_score_gap"])
            if not row["oracle_best_feasible"]:
                return "in_window_missed_veto"
            if gap <= 0.03:
                return "in_window_missed_near_tie"
            if gap <= 0.1:
                return "in_window_missed_challenge"
            return "in_window_missed_large_gap"
        return "in_window_doomed"
    if row["remaining"] <= horizon:
        return "replan_death"
    if row["remaining"] <= 2 * horizon:
        return "post_window_death"
    return "fine"


def summarize_autopsy(rows, horizon):
    classes = {}
    for row in rows:
        label = classify_row(row, horizon)
        row["class"] = label
        classes[label] = classes.get(label, 0) + 1
    pooled_risk = []
    pooled_dead = []
    contrast = dict(comparable=0, risk_contrast_correct=0)
    for row in rows:
        if row["risk_auc"] is not None:
            pooled_risk.append(row)
        actions = sorted(row["risk_by_action"])
        if len(actions) == 2 and len(row["true_by_action"]) == 2:
            low_risk = min(actions, key=lambda a: row["risk_by_action"][a])
            high_risk = max(actions, key=lambda a: row["risk_by_action"][a])
            if (
                row["true_by_action"][low_risk]
                != row["true_by_action"][high_risk]
            ):
                contrast["comparable"] += 1
                if (
                    row["true_by_action"][low_risk]
                    > row["true_by_action"][high_risk]
                ):
                    contrast["risk_contrast_correct"] += 1
    risk_values, dead_labels = [], []
    misses = [row for row in rows if row["class"].startswith("in_window")]
    imminent = [row for row in rows if row["remaining"] <= 3]
    return dict(
        decisions=int(len(rows)),
        classes=classes,
        pooled_risk_auc_points=int(len(pooled_risk)),
        mean_risk_auc=(
            float(np.mean([row["risk_auc"] for row in pooled_risk]))
            if pooled_risk else None
        ),
        risk_contrast=contrast,
        in_window_deaths=int(sum(
            row["selected_true"] >= 0 for row in rows
        )),
        imminent_blind=int(sum(
            row["selected_true"] < 0 for row in imminent
        )),
        imminent=int(len(imminent)),
        plan_followed_rate=(
            float(np.mean([
                row["plan_followed"] for row in rows
                if row["plan_followed"] is not None
            ]))
            if any(row["plan_followed"] is not None for row in rows)
            else None
        ),
        mean_remaining_by_class={
            label: float(np.mean([
                row["remaining"] for row in rows if row["class"] == label
            ]))
            for label in classes
        },
    )


def compare_arms(rows_low, rows_high):
    """Matched (world, decision) comparison of the two horizons.

    The h6 run's own timeline (`remaining`) classifies the low arm; the
    high arm is judged only by what its search saw at the same state: a
    plan that survives its full window versus one that dies inside it.
    """
    index = {
        (row["eval_seed"], row["point"]): row for row in rows_high
    }
    matched = []
    for row in rows_low:
        twin = index.get((row["eval_seed"], row["point"]))
        if twin is None:
            continue
        matched.append((row, twin))
    low_horizon = rows_low[0]["horizon"] if rows_low else None
    escapes = 0
    doomed_both = 0
    worse_high = 0
    for low, high in matched:
        low_dying = (
            low["selected_true"] >= 0
            or low["remaining"] <= 2 * int(low["horizon"])
        )
        high_safe = high["selected_true"] < 0
        if low_dying and high_safe:
            escapes += 1
        elif low_dying:
            doomed_both += 1
        elif high["selected_true"] >= 0:
            worse_high += 1
    return dict(
        matched=len(matched),
        low_horizon=low_horizon,
        high_escapes_low_death=escapes,
        both_dying=doomed_both,
        high_dies_where_low_survives=worse_high,
    )


def run_seed(seed, horizons, eval_seeds, args):
    models = {}
    for horizon in horizons:
        config = horizon_config(seed, horizon)
        world, agent, cached = load_or_fit(
            config, "%s-%d-%d.pkl" % (args.cache_prefix, seed, horizon),
            refit=bool(args.refit),
        )
        models[horizon] = (world, agent)
        print(
            "seed %d horizon %d: model %s" % (
                seed, horizon, "cache" if cached else "fit",
            )
        )
    rows = []
    world_rows = []
    for eval_seed in eval_seeds:
        env_seed = int(eval_seed) * 10000
        world, agent = models[min(horizons)]
        with _curiosity_off(agent):
            frames, actions, _ = evaluate(
                seed, world, agent, cap=5000, horizon=min(horizons),
                lag=LAG, step_stride=2, env_seed=env_seed, record=False,
                episode=0,
            )
        world_rows.append(dict(
            train_seed=seed, eval_seed=eval_seed, frames=int(frames),
            action_rate=float(np.mean(actions)) if len(actions) else 0.0,
        ))
        print(
            "  seed %d h%d world %d: frames %d" % (
                seed, min(horizons), eval_seed, frames,
            )
        )
        if frames < 3:
            continue
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
            remaining = int(frames - point)
            for horizon in horizons:
                world_h, agent_h = models[horizon]
                row = open_decision(
                    env, world_h, agent_h, prefix, horizon, actions, point,
                )
                if row is None:
                    continue
                row.update(
                    train_seed=seed, eval_seed=eval_seed, point=point,
                    remaining=remaining,
                )
                rows.append(row)
    return rows, world_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--arms", default="6,8")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=6)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--cache-prefix", default="autopsy")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    horizons = sorted(int(value) for value in str(args.arms).split(","))
    eval_seeds = seed_spec(args.eval_seeds)
    rows, world_rows = run_seed(args.seed, horizons, eval_seeds, args)
    rows_low = [row for row in rows if row["horizon"] == horizons[0]]
    rows_high = [row for row in rows if row["horizon"] == horizons[-1]]
    summary_low = summarize_autopsy(rows_low, horizons[0])
    summary_high = summarize_autopsy(rows_high, horizons[-1])
    comparison = compare_arms(rows_low, rows_high)
    print(
        "== seed %d  frames per world %s ==" % (
            args.seed,
            "/".join(str(row["frames"]) for row in world_rows),
        )
    )
    print(
        "== h%d autopsy: %d decisions  classes %s ==" % (
            horizons[0], summary_low["decisions"], summary_low["classes"],
        )
    )
    print(
        "   pooled risk AUC %s (over %d decidable decisions)  "
        "risk first-action contrast %d/%d" % (
            "None" if summary_low["mean_risk_auc"] is None
            else "%.3f" % summary_low["mean_risk_auc"],
            summary_low["pooled_risk_auc_points"],
            summary_low["risk_contrast"]["risk_contrast_correct"],
            summary_low["risk_contrast"]["comparable"],
        )
    )
    print(
        "   in-window plan deaths %d  imminent decisions %d "
        "(window-blind %d)  plan followed %.2f" % (
            summary_low["in_window_deaths"], summary_low["imminent"],
            summary_low["imminent_blind"],
            summary_low["plan_followed_rate"] or 0.0,
        )
    )
    print(
        "== control h%d at the same states: matched %d  "
        "h%d escapes %d  both die %d  h%d dies where h%d survives %d ==" % (
            horizons[-1], comparison["matched"], horizons[-1],
            comparison["high_escapes_low_death"], comparison["both_dying"],
            horizons[-1], horizons[0],
            comparison["high_dies_where_low_survives"],
        )
    )
    if summary_high["decisions"]:
        print(
            "== h%d classes %s  pooled risk AUC %s ==" % (
                horizons[-1], summary_high["classes"],
                "None" if summary_high["mean_risk_auc"] is None
                else "%.3f" % summary_high["mean_risk_auc"],
            )
        )
    fatal = sorted(
        (row for row in rows_low if row["remaining"] <= 3),
        key=lambda row: (row["eval_seed"], row["point"]),
    )
    for row in fatal[:10]:
        print(
            "   fatal h%d w%d@%d rem %d: %-28s sel a%d true %d risk %.4f  "
            "risk_by_action %s  true_by_action %s" % (
                row["horizon"], row["eval_seed"], row["point"],
                row["remaining"], row["class"], row["selected_action"],
                row["selected_true"], row["selected_risk"] or 0.0,
                {key: round(value, 4) for key, value in
                 row["risk_by_action"].items()},
                row["true_by_action"],
            )
        )
    report = dict(
        seed=args.seed, horizons=horizons, world_rows=world_rows,
        summary={
            str(horizons[0]): summary_low,
            str(horizons[-1]): summary_high,
        },
        comparison=comparison,
        rows=rows,
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return report


if __name__ == "__main__":
    main()
