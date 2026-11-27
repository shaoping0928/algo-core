"""Apply the query-graph event predicate to the planner's candidate endpoints.

This is an opt-in decision-side probe; it changes no default path.  For each
decision it records the real search trace with no death veto, evaluates every
candidate in the real world as an oracle, and then compares three selections:

* baseline: scores + the model's feasibility mask;
* veto-death: baseline mask plus the event predicate's death=1;
* veto-death-or-unknown: baseline mask plus death=1 or UNKNOWN.

The event predicate is the existing ``DeathRuleSet`` wrapped as a query-graph
rule (``event_rules.make_death_event_rule``).  The probe reports whether the
veto changes the first action, whether it improves or worsens the oracle
survival, and how often it removes an oracle-best plan.

The key target is seed 101, whose failure was a score misranking rather than
a death-recall miss.  A useful result here is therefore not only lower
decision-relevant miss/regret, but also a low oracle-best veto rate.
"""

import argparse

import numpy as np

from death_transition_quality import seed_spec
from event_rules import make_death_event_rule
from latent_world import latent
from planner_rollout_quality import (
    flatten_batches, oracle_plan, probe_config,
)
from probe_cache import load_or_fit
from query_graph import UNKNOWN, Query, RuleBank, Slot
from substrate_nodes import (
    LAG, _curiosity_off, evaluate, policy_result,
)
from trajectory_search import best_candidate_index


def endpoint_event(bank, state, target):
    source = {
        Slot("x%d" % index, 0): float(value)
        for index, value in enumerate(state)
    }
    result = bank.query(Query(
        target_slots=(target,), source=source, budget=32,
    ))
    value = result.values.get(target)
    if value is UNKNOWN:
        return -1
    return int(value)


def event_quality(prediction, true_death):
    prediction = np.asarray(prediction, dtype=int)
    truth = np.asarray(true_death, dtype=bool)
    answered = prediction >= 0
    positive = prediction == 1
    true_positive = int(np.sum(answered & positive & truth))
    false_positive = int(np.sum(answered & positive & ~truth))
    false_negative = int(np.sum(answered & ~positive & truth))
    true_negative = int(np.sum(answered & ~positive & ~truth))
    precision = true_positive / float(true_positive + false_positive + 1e-12)
    recall = true_positive / float(true_positive + false_negative + 1e-12)
    return dict(
        candidates=int(len(prediction)),
        unknown=int(np.sum(~answered)),
        precision=float(precision),
        recall=float(recall),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )


def select_arm(scores, base_mask, event, mode):
    if mode == "baseline":
        mask = base_mask.copy()
    elif mode == "veto-death":
        mask = base_mask & (event != 1)
    elif mode == "veto-death-unknown":
        mask = base_mask & (event == 0)
    else:
        raise ValueError("unknown arm %s" % mode)
    fallback = False
    if not mask.any():
        mask = base_mask.copy()
        fallback = True
    if not mask.any():
        return None, fallback
    return int(best_candidate_index(scores, mask)), fallback


def summarize(records, horizon):
    def arm_rows(mode):
        rows = [row[mode] for row in records]
        return dict(
            oracle_best=int(sum(row["oracle_best"] for row in rows)),
            survives=int(sum(row["survives"] for row in rows)),
            decision_relevant_misses=int(sum(
                row["regret"] > 0 for row in rows
            )),
            mean_regret=float(np.mean([row["regret"] for row in rows])),
            max_regret=int(max(row["regret"] for row in rows)),
        )

    summary = {
        "decisions": len(records),
        "baseline": arm_rows("baseline"),
        "veto-death": arm_rows("veto-death"),
        "veto-death-unknown": arm_rows("veto-death-unknown"),
        "changed": int(sum(
            row["baseline"]["action"] != row["veto-death"]["action"]
            for row in records
        )),
        "improved": int(sum(
            row["veto-death"]["regret"] < row["baseline"]["regret"]
            for row in records
        )),
        "worsened": int(sum(
            row["veto-death"]["regret"] > row["baseline"]["regret"]
            for row in records
        )),
        "vetoed_oracle_best": int(sum(
            row["baseline"]["oracle_best"]
            and row["baseline"]["action"] != row["veto-death"]["action"]
            for row in records
        )),
        "fallback": int(sum(
            row["veto-death"]["fallback"] for row in records
        )),
        "event": {
            key: sum(row["event"][key] for row in records)
            for key in (
                "candidates", "unknown", "true_positive",
                "false_positive", "false_negative", "true_negative",
            )
        },
    }
    event = summary["event"]
    event["precision"] = event["true_positive"] / float(
        event["true_positive"] + event["false_positive"] + 1e-12
    )
    event["recall"] = event["true_positive"] / float(
        event["true_positive"] + event["false_negative"] + 1e-12
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--tail", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--cache", default="")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    config = probe_config(args)
    world, agent, cached = load_or_fit(
        config, args.cache or None, refit=bool(args.refit)
    )
    print("model cache: %s" % ("hit" if cached else "fit"))
    agent.death_rule_hard = False
    rules = getattr(agent, "death_rules", None)
    if rules is None:
        raise SystemExit("the fitted agent has no death rules")
    target = Slot("event", 1)
    bank = RuleBank([make_death_event_rule(
        rules,
        [Slot("x%d" % index, 0) for index in range(agent.obs_dim)],
        target_slot=target,
    )])
    records = []
    for eval_seed in seed_spec(args.eval_seeds):
        env_seed = int(eval_seed) * 10000
        with _curiosity_off(agent):
            frames, actions, _rows = evaluate(
                args.seed, world, agent, cap=5000,
                horizon=args.horizon, lag=LAG, step_stride=2,
                env_seed=env_seed, record=False, episode=0,
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

            batches = []
            policy_result(
                env, world, agent, prefix, args.horizon, lag=LAG,
                step_stride=2, candidate_callback=batches.append,
            )
            if not batches:
                continue
            trace = flatten_batches(batches)
            sequences = np.asarray(trace["sequences"], dtype=int)
            scores = np.asarray(trace["scores"], dtype=float)
            base_mask = (
                np.asarray(trace["model_feasible"], dtype=bool)
                & np.asarray(trace["finite"], dtype=bool)
            )
            if not base_mask.any():
                continue
            true_first = np.asarray([
                oracle_plan(env, sequence)[0]
                for sequence in sequences
            ], dtype=int)
            true_value = np.where(
                true_first < 0, args.horizon + 1, true_first
            )
            best_value = int(true_value.max())
            best_actions = {
                int(sequences[index, 0])
                for index in np.flatnonzero(true_value == best_value)
            }
            events = np.asarray([
                endpoint_event(bank, state, target)
                for state in np.asarray(trace["final_states"], dtype=float)
            ], dtype=int)
            record = {"event": event_quality(events, true_first >= 0)}
            baseline = None
            for mode in ("baseline", "veto-death", "veto-death-unknown"):
                index, fallback = select_arm(
                    scores, base_mask, events, mode,
                )
                if index is None:
                    continue
                action = int(sequences[index, 0])
                value = int(true_value[index])
                row = dict(
                    action=action,
                    value=value,
                    regret=int(best_value - value),
                    oracle_best=bool(action in best_actions),
                    survives=bool(value > args.horizon),
                    fallback=bool(fallback),
                )
                record[mode] = row
            if "baseline" not in record:
                continue
            if "veto-death" not in record:
                record["veto-death"] = dict(record["baseline"])
            if "veto-death-unknown" not in record:
                record["veto-death-unknown"] = dict(record["baseline"])
            records.append(record)
            if baseline is None:
                baseline = record["baseline"]

    summary = summarize(records, args.horizon)
    print("== seed %d eval %s decisions %d ==" % (
        args.seed, args.eval_seeds, summary["decisions"],
    ))
    for mode in ("baseline", "veto-death", "veto-death-unknown"):
        row = summary[mode]
        print(
            "%-20s oracle-best %2d/%2d  survives %2d/%2d  "
            "relevant-miss %2d  mean-regret %5.2f  max %d" % (
                mode, row["oracle_best"], summary["decisions"],
                row["survives"], summary["decisions"],
                row["decision_relevant_misses"], row["mean_regret"],
                row["max_regret"],
            )
        )
    print(
        "veto-death: changed %d  improved %d  worsened %d  "
        "oracle-best-vetoed %d  fallback %d" % (
            summary["changed"], summary["improved"],
            summary["worsened"], summary["vetoed_oracle_best"],
            summary["fallback"],
        )
    )
    event = summary["event"]
    print(
        "endpoint event: precision %.4f recall %.4f unknown %d/%d "
        "TP %d FP %d FN %d TN %d" % (
            event["precision"], event["recall"], event["unknown"],
            event["candidates"], event["true_positive"],
            event["false_positive"], event["false_negative"],
            event["true_negative"],
        )
    )


if __name__ == "__main__":
    main()
