"""B0.6: is the risk column wrong because of the world rollout or the readout?

B0.5 showed that the fine score on the plan-level misses is effectively one
active column (risk): value, curiosity and q contribute zero, and dropping
any column or subset does not recover the oracle-best suffix.  The remaining
question is where the risk ordering breaks:

* if risk evaluated on the *true* trajectory ranks the selected suffix as
  riskier than the same-action-best suffix, the readout is right and the
  predicted rollout is the problem;
* if risk is already inverted on true states, the readout/calibration is the
  problem and plan-level endpoint re-ranking would only inherit it.

This probe replays each plan-level miss twice with the planner's own stride,
pending-action encoding, scales and risk-decay/error factors: once through
`world.predict`, once through the real environment.  It also checks that the
replayed predicted risk equals the planner's risk column, so a mismatch is a
wiring error rather than a finding.

Read-only; no model is trained and no default path changes.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from decision_b0_probe import run as b0_run
from plan_score_probe import diagnose
from substrate_generic import history_features
from substrate_nodes import LAG, terminal_hazard


def _risk_at(agent, state, pending):
    encoded = np.concatenate([
        np.asarray(state, dtype=float)[:, :int(agent.obs_dim)], pending,
    ], axis=1) / agent.scales
    return float(agent.pain_signal_batch(encoded)[0])


def _factor(agent, step, decay):
    value = float(decay) ** int(step)
    if bool(getattr(agent, "risk_error_weight", False)):
        errors = list(getattr(agent, "step_errors", ()) or ())
        if errors:
            value /= 1.0 + float(errors[min(int(step), len(errors) - 1)])
    return value


def _terminal_factor(agent, horizon, decay):
    value = float(decay) ** int(horizon)
    if bool(getattr(agent, "risk_error_weight", False)):
        errors = list(getattr(agent, "step_errors", ()) or ())
        if errors:
            value /= 1.0 + float(errors[
                min(int(horizon) - 1, len(errors) - 1)
            ])
    return value


def _terminal_risk(agent, state, history, horizon, lag, decay):
    """The terminal hazard the planner adds to the risk column."""
    pending = history[:, :int(lag)][:, ::-1]
    encoded = np.concatenate([
        np.asarray(state, dtype=float)[:, :int(agent.obs_dim)], pending,
    ], axis=1) / agent.scales
    mask_table = agent._feature_mask_table(encoded[:, :int(agent.obs_dim)])
    pain, _dopamine = agent.valence_state_batch(encoded)
    signal = agent.pain_signal_batch(encoded, mask_table)
    risk = terminal_hazard(
        pain, signal, 1.0, pain_weight=None, lane="max",
    )
    return float(risk[0]) * _terminal_factor(agent, horizon, decay)


def predicted_rollout(env, sequence, world, agent, prefix, horizon,
                      lag=LAG, stride=2, decay=1.0):
    """Replay the planner's predicted states and risk column."""
    state = np.asarray(env.obs(), dtype=float).reshape(1, -1)
    total = 0.0
    terms = []
    states = []
    for step in range(int(horizon)):
        history = history_features(
            np.asarray(sequence, dtype=float).reshape(1, -1), step + 1,
            prefix, action_dim=int(lag) + 1,
        )
        state = np.asarray(world.predict(state, history), dtype=float)
        states.append(state[0].copy())
        if step % int(stride) == 0:
            pending = history[:, :int(lag)][:, ::-1]
            risk = _risk_at(agent, state, pending) * _factor(
                agent, step, decay
            )
            total += risk
            terms.append(risk)
    terminal_history = history_features(
        np.asarray(sequence, dtype=float).reshape(1, -1), int(horizon),
        prefix, action_dim=int(lag) + 1,
    )
    terminal = _terminal_risk(
        agent, state, terminal_history, horizon, lag, decay,
    )
    return dict(
        step_total=float(total), terminal=float(terminal),
        total=float(total + terminal), terms=terms, states=states,
    )


def true_rollout(env, sequence, agent, prefix, horizon, lag=LAG, stride=2,
                 decay=1.0):
    """Replay the real trajectory and evaluate risk on its true states."""
    child = copy.deepcopy(env)
    total = 0.0
    terms = []
    states = []
    death = -1
    for step in range(int(horizon)):
        if child.dead:
            break
        history = history_features(
            np.asarray(sequence, dtype=float).reshape(1, -1), step + 1,
            prefix, action_dim=int(lag) + 1,
        )
        child.step(int(sequence[step]))
        if child.dead:
            death = step
            break
        state = np.asarray(child.obs(), dtype=float).reshape(1, -1)
        states.append(state[0].copy())
        if step % int(stride) == 0:
            pending = history[:, :int(lag)][:, ::-1]
            risk = _risk_at(agent, state, pending) * _factor(
                agent, step, decay
            )
            total += risk
            terms.append(risk)
    if death >= 0:
        terminal = 1.0
    else:
        state = np.asarray(child.obs(), dtype=float).reshape(1, -1)
        terminal_history = history_features(
            np.asarray(sequence, dtype=float).reshape(1, -1),
            int(horizon), prefix, action_dim=int(lag) + 1,
        )
        terminal = _terminal_risk(
            agent, state, terminal_history, horizon, lag, decay,
        )
    return dict(
        step_total=float(total), terminal=float(terminal),
        total=float(total + terminal), terms=terms, states=states,
        death=int(death),
    )


def _mean(values):
    return float(np.mean(values)) if values else None


def diagnose_risk(record, row, indices, world, agent, horizon, stride):
    """Compare predicted and true risk for the selected / same-action-best pair."""
    sequences = np.asarray(record["sequences"], dtype=int)
    scores = np.asarray(record["scores"], dtype=float)
    selected_index = int(indices["selected_index"])
    best_index = int(indices["same_best_index"])
    selected_sequence = sequences[selected_index]
    best_sequence = sequences[best_index]
    prefix = np.asarray(record["prefix"], dtype=float)
    env = record["env"]
    pred_selected = predicted_rollout(
        env, selected_sequence, world, agent, prefix, horizon, LAG, stride,
    )
    pred_best = predicted_rollout(
        env, best_sequence, world, agent, prefix, horizon, LAG, stride,
    )
    true_selected = true_rollout(
        env, selected_sequence, agent, prefix, horizon, LAG, stride,
    )
    true_best = true_rollout(
        env, best_sequence, agent, prefix, horizon, LAG, stride,
    )
    scale = np.asarray(agent.scales[:int(agent.obs_dim)], dtype=float)
    common = min(len(pred_selected["states"]), len(true_selected["states"]))
    errors = []
    for step in range(common):
        difference = (
            pred_selected["states"][step][:int(agent.obs_dim)]
            - true_selected["states"][step][:int(agent.obs_dim)]
        ) / scale
        errors.append(float(np.linalg.norm(difference)))
    selected_risk_column = float(scores[selected_index, 2])
    best_risk_column = float(scores[best_index, 2])
    pred_match_selected = abs(pred_selected["total"] - selected_risk_column)
    pred_match_best = abs(pred_best["total"] - best_risk_column)
    true_selected_mean = _mean(true_selected["terms"])
    true_best_mean = _mean(true_best["terms"])
    true_step_margin = (
        float(true_selected_mean - true_best_mean)
        if true_selected_mean is not None
        and true_best_mean is not None else None
    )
    first_true_selected = (
        true_selected["terms"][0] if true_selected["terms"] else None
    )
    first_true_best = (
        true_best["terms"][0] if true_best["terms"] else None
    )
    first_pred_selected = (
        pred_selected["terms"][0] if pred_selected["terms"] else None
    )
    first_pred_best = (
        pred_best["terms"][0] if pred_best["terms"] else None
    )
    predicted_death_selected = int(
        np.asarray(record["trace"]["first_death"], dtype=int)[selected_index]
    )
    predicted_death_best = int(
        np.asarray(record["trace"]["first_death"], dtype=int)[best_index]
    )
    true_step_correct = (
        bool(true_best_mean < true_selected_mean)
        if true_selected_mean is not None
        and true_best_mean is not None else None
    )
    true_total_correct = bool(
        true_best["total"] < true_selected["total"]
    )
    true_terminal_correct = bool(
        true_best["terminal"] < true_selected["terminal"]
    )
    pred_total_correct = bool(
        pred_best["total"] < pred_selected["total"]
    )
    pred_terminal_correct = bool(
        pred_best["terminal"] < pred_selected["terminal"]
    )
    return dict(
        eval_seed=int(record["eval_seed"]),
        point=int(record["point"]),
        **{"class": row["class"]},
        action_regret=int(indices["action_regret"]),
        suffix_regret=int(indices["suffix_regret"]),
        selected_action=int(indices["selected_action"]),
        selected_index=selected_index,
        same_best_index=best_index,
        pred_selected=pred_selected["total"],
        pred_best=pred_best["total"],
        pred_step_selected=pred_selected["step_total"],
        pred_step_best=pred_best["step_total"],
        pred_terminal_selected=pred_selected["terminal"],
        pred_terminal_best=pred_best["terminal"],
        pred_margin=float(pred_selected["total"] - pred_best["total"]),
        pred_step_margin=float(
            pred_selected["step_total"] - pred_best["step_total"]
        ),
        pred_terminal_margin=float(
            pred_selected["terminal"] - pred_best["terminal"]
        ),
        true_selected_mean=true_selected_mean,
        true_best_mean=true_best_mean,
        true_step_margin=true_step_margin,
        true_total_selected=true_selected["total"],
        true_total_best=true_best["total"],
        true_total_margin=float(
            true_selected["total"] - true_best["total"]
        ),
        true_terminal_selected=true_selected["terminal"],
        true_terminal_best=true_best["terminal"],
        true_terminal_margin=float(
            true_selected["terminal"] - true_best["terminal"]
        ),
        first_pred_selected=first_pred_selected,
        first_pred_best=first_pred_best,
        first_pred_margin=(
            float(first_pred_selected - first_pred_best)
            if first_pred_selected is not None
            and first_pred_best is not None else None
        ),
        first_true_selected=first_true_selected,
        first_true_best=first_true_best,
        first_true_margin=(
            float(first_true_selected - first_true_best)
            if first_true_selected is not None
            and first_true_best is not None else None
        ),
        risk_pred_correct=pred_total_correct,
        risk_pred_terminal_correct=pred_terminal_correct,
        risk_true_correct=true_total_correct,
        risk_true_step_correct=true_step_correct,
        risk_true_terminal_correct=true_terminal_correct,
        risk_first_true_correct=(
            bool(first_true_best < first_true_selected)
            if first_true_selected is not None
            and first_true_best is not None else None
        ),
        risk_first_pred_correct=(
            bool(first_pred_best < first_pred_selected)
            if first_pred_selected is not None
            and first_pred_best is not None else None
        ),
        selected_risk_column=selected_risk_column,
        best_risk_column=best_risk_column,
        pred_match_selected=bool(pred_match_selected <= 1e-6),
        pred_match_best=bool(pred_match_best <= 1e-6),
        pred_match_error=max(pred_match_selected, pred_match_best),
        true_selected_terms=len(true_selected["terms"]),
        true_best_terms=len(true_best["terms"]),
        true_selected_death=true_selected["death"],
        true_best_death=true_best["death"],
        predicted_death_selected=predicted_death_selected,
        predicted_death_best=predicted_death_best,
        state_error_mean=_mean(errors),
        state_error_max=float(max(errors)) if errors else None,
    )


def summarize(diagnostics):
    valid = [
        item for item in diagnostics
        if "error" not in item and item.get("true_step_margin") is not None
    ]
    plan_level = [
        item for item in valid
        if item["action_regret"] == 0 and item["suffix_regret"] > 0
    ]
    true_total_correct = sum(
        bool(item["risk_true_correct"]) for item in plan_level
    )
    true_step_correct = sum(
        bool(item["risk_true_step_correct"]) for item in plan_level
    )
    true_terminal_correct = sum(
        bool(item["risk_true_terminal_correct"]) for item in plan_level
    )
    first_true_correct = sum(
        bool(item["risk_first_true_correct"]) for item in plan_level
    )
    pred_total_correct = sum(
        bool(item["risk_pred_correct"]) for item in plan_level
    )
    pred_terminal_correct = sum(
        bool(item["risk_pred_terminal_correct"]) for item in plan_level
    )
    match = sum(
        item["pred_match_selected"] and item["pred_match_best"]
        for item in plan_level
    )
    if match < len(plan_level):
        verdict = "replication_mismatch"
    else:
        need = len(plan_level) // 2 + 1 if plan_level else 1
        if true_step_correct >= need:
            verdict = "world_prediction_drift"
        elif true_terminal_correct >= need and pred_terminal_correct < need:
            verdict = "step_readout_flat_endpoint_drift"
        elif true_terminal_correct < need:
            verdict = "readout_miscalibrated"
        else:
            verdict = "mixed"
    return dict(
        misses=int(len(valid)),
        plan_level=int(len(plan_level)),
        risk_true_total_correct=int(true_total_correct),
        risk_true_step_correct=int(true_step_correct),
        risk_true_terminal_correct=int(true_terminal_correct),
        risk_first_true_correct=int(first_true_correct),
        risk_pred_total_correct=int(pred_total_correct),
        risk_pred_terminal_correct=int(pred_terminal_correct),
        pred_match=int(match),
        true_step_margin_mean=(
            float(np.mean([
                item["true_step_margin"] for item in plan_level
            ])) if plan_level else None
        ),
        true_total_margin_mean=(
            float(np.mean([
                item["true_total_margin"] for item in plan_level
            ])) if plan_level else None
        ),
        true_terminal_margin_mean=(
            float(np.mean([
                item["true_terminal_margin"] for item in plan_level
            ])) if plan_level else None
        ),
        first_true_margin_mean=(
            float(np.mean([
                item["first_true_margin"] for item in plan_level
                if item["first_true_margin"] is not None
            ])) if plan_level else None
        ),
        pred_step_margin_mean=(
            float(np.mean([
                item["pred_step_margin"] for item in plan_level
            ])) if plan_level else None
        ),
        pred_total_margin_mean=(
            float(np.mean([
                item["pred_margin"] for item in plan_level
            ])) if plan_level else None
        ),
        pred_terminal_margin_mean=(
            float(np.mean([
                item["pred_terminal_margin"] for item in plan_level
            ])) if plan_level else None
        ),
        state_error_mean=(
            float(np.mean([
                item["state_error_mean"] for item in plan_level
                if item["state_error_mean"] is not None
            ])) if plan_level else None
        ),
        verdict=verdict,
        rows=diagnostics,
    )


def print_report(summary):
    print(
        "== B0.6 risk readout: %d plan-level misses, %d valid ==" % (
            summary["plan_level"], summary["misses"],
        )
    )
    print(
        "   correct ordering: true total %d  true step %d  "
        "true terminal %d  first-step true %d  pred total %d  "
        "pred terminal %d  (of %d)" % (
            summary["risk_true_total_correct"],
            summary["risk_true_step_correct"],
            summary["risk_true_terminal_correct"],
            summary["risk_first_true_correct"],
            summary["risk_pred_total_correct"],
            summary["risk_pred_terminal_correct"],
            summary["plan_level"],
        )
    )
    print(
        "   margins: true step %s  true total %s  true terminal %s  "
        "first true %s" % (
            "None" if summary["true_step_margin_mean"] is None
            else "%+.4f" % summary["true_step_margin_mean"],
            "None" if summary["true_total_margin_mean"] is None
            else "%+.4f" % summary["true_total_margin_mean"],
            "None" if summary["true_terminal_margin_mean"] is None
            else "%+.4f" % summary["true_terminal_margin_mean"],
            "None" if summary["first_true_margin_mean"] is None
            else "%+.4f" % summary["first_true_margin_mean"],
        )
    )
    print(
        "   predicted margins: step %s  total %s  terminal %s  "
        "state error %s  replay match %d/%d" % (
            "None" if summary["pred_step_margin_mean"] is None
            else "%+.4f" % summary["pred_step_margin_mean"],
            "None" if summary["pred_total_margin_mean"] is None
            else "%+.4f" % summary["pred_total_margin_mean"],
            "None" if summary["pred_terminal_margin_mean"] is None
            else "%+.4f" % summary["pred_terminal_margin_mean"],
            "None" if summary["state_error_mean"] is None
            else "%.4f" % summary["state_error_mean"],
            summary["pred_match"], summary["plan_level"],
        )
    )
    print("   verdict: %s" % summary["verdict"])
    for item in summary["rows"]:
        if "error" in item:
            print(
                "   seed %d @%d ERROR %s" % (
                    item["eval_seed"], item["point"], item["error"],
                )
            )
            continue
        print(
            "   seed %d @%d: %-22s suffix %d  "
            "pred step %+.4f term %+.4f total %+.4f  "
            "true step %s term %s total %s  first true %s/%s  "
            "state-err %s  death true %d/%d pred %d/%d" % (
                item["eval_seed"], item["point"], item["class"],
                item["suffix_regret"],
                item["pred_step_margin"], item["pred_terminal_margin"],
                item["pred_margin"],
                "None" if item["true_step_margin"] is None
                else "%+.4f" % item["true_step_margin"],
                "%+.4f" % item["true_terminal_margin"],
                "%+.4f" % item["true_total_margin"],
                "None" if item["first_true_selected"] is None
                else "%.4f" % item["first_true_selected"],
                "None" if item["first_true_best"] is None
                else "%.4f" % item["first_true_best"],
                "None" if item["state_error_mean"] is None
                else "%.3f" % item["state_error_mean"],
                item["true_selected_death"], item["true_best_death"],
                item["predicted_death_selected"],
                item["predicted_death_best"],
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
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
    context = {}
    b0_summary = b0_run(
        b0_args, trace_out=records, context_out=context,
    )
    world = context["world"]
    agent = context["agent"]
    relevant = {
        (row["eval_seed"], row["point"]): row
        for row in b0_summary["rows"] if row["decision_relevant"]
    }
    diagnostics = []
    for record in records:
        key = (record["eval_seed"], record["point"])
        if key not in relevant:
            continue
        row = relevant[key]
        indices = diagnose(record, row)
        if "error" in indices:
            diagnostics.append(dict(
                eval_seed=record["eval_seed"], point=record["point"],
                **{"class": row["class"]}, error=indices["error"],
            ))
            continue
        diagnostics.append(diagnose_risk(
            record, row, indices, world, agent, args.horizon, args.stride,
        ))
    summary = summarize(diagnostics)
    print_report(summary)
    summary["b0"] = dict(
        decisions=b0_summary["decisions"],
        misses=b0_summary["misses"],
        plan_level=b0_summary["plan_level_misses"],
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return summary


if __name__ == "__main__":
    main()
