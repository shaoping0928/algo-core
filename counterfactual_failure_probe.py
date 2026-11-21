"""Learn death rules from counterfactual plan failures.

The actual control trajectory rarely enters a terminal state the rules miss:
the planner usually dies where the rules already fire, and the candidate
plans that would die are never executed.  This probe therefore uses the
search itself as the data source:

1. sample control-policy states on refinement worlds;
2. run the real search and record every candidate plan;
3. oracle-evaluate each plan and keep the true terminal states;
4. split them into "missed" (death side UNKNOWN) and "covered";
5. add the missed terminals as positives and test on held-out worlds.

The test set has two views: the actual control transitions (terminal
coverage plus non-terminal false alarms) and counterfactual terminal states
from the held-out worlds' candidate plans.  Refinement and test worlds are
disjoint.  No task vocabulary or geometry enters the rule learner or the
loop.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from death_failure_probe import CONFIG, collect_states, test_report
from death_transition_quality import seed_spec
from latent_world import latent
from planner_rollout_quality import (
    flatten_batches,
    oracle_plan,
    status_callback,
)
from substrate_nodes import LAG, _curiosity_off, evaluate, fit, policy_result


def sample_points(frames, branches, tail):
    return sorted(set(
        list(np.linspace(2, frames - 2, branches).astype(int))
        + list(range(max(1, frames - tail), frames))
    ))


def dedupe_terminals(states, limit=0):
    unique = {}
    for state in states:
        key = tuple(np.round(np.asarray(state, dtype=float), 6).tolist())
        unique.setdefault(key, np.asarray(state, dtype=float))
    values = list(unique.values())
    if limit and len(values) > limit:
        index = np.linspace(0, len(values) - 1, limit).astype(int)
        values = [values[int(i)] for i in index]
    if not values:
        return np.zeros((0, 0))
    return np.asarray(values, dtype=float)


def collect_counterfactual(rules, agent, world, seeds, horizon,
                           branches, tail, cap, max_examples=0,
                           full_status=False):
    """Counterfactual terminal states from sampled search decisions."""
    status = status_callback(
        rules, agent.obs_dim, full=bool(full_status)
    )
    missed = []
    covered = []
    decisions = 0
    candidates = 0
    for eval_seed in seeds:
        env_seed = int(eval_seed) * 10000
        with _curiosity_off(agent):
            frames, actions, _rows = evaluate(
                CONFIG["seed"], world, agent, cap=cap, horizon=horizon,
                lag=LAG, step_stride=2, env_seed=env_seed, record=False,
                episode=0,
            )
        if frames < 2:
            continue
        for point in sample_points(frames, branches, tail):
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
                env, world, agent, prefix, horizon, lag=LAG,
                step_stride=2, death_status_batch=status,
                candidate_callback=batches.append,
            )
            if not batches:
                continue
            trace = flatten_batches(batches)
            decisions += 1
            candidates += len(trace["sequences"])
            for sequence in trace["sequences"]:
                first_death, terminal_state, _dead = oracle_plan(
                    env, sequence
                )
                if first_death < 0:
                    continue
                prediction = int(rules.classify_death(
                    np.asarray(terminal_state, dtype=float)[None]
                )[0])
                if prediction == 1:
                    covered.append(terminal_state)
                else:
                    missed.append(terminal_state)
    missed_states = dedupe_terminals(missed, max_examples)
    covered_states = dedupe_terminals(covered, max_examples)
    return dict(
        missed=missed_states,
        covered=covered_states,
        decisions=decisions,
        candidates=candidates,
    )


def terminal_report(rules, states):
    if len(states) == 0:
        return dict(terminals=0, covered=0, unknown=0, coverage=0.0)
    prediction = rules.classify_death(np.asarray(states, dtype=float))
    covered = int(np.sum(prediction == 1))
    return dict(
        terminals=int(len(states)),
        covered=covered,
        unknown=int(len(states) - covered),
        coverage=float(covered) / float(len(states)),
    )


def add_positives(rules, states, count, rng):
    if count <= 0 or len(states) == 0:
        return 0
    count = min(int(count), len(states))
    if count < len(states):
        index = rng.choice(len(states), count, replace=False)
        selected = states[index]
    else:
        selected = states
    rules.add_examples(
        np.asarray(selected, dtype=float),
        np.ones(len(selected), dtype=int),
    )
    return int(len(selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--refine-seeds", default="120-121")
    parser.add_argument("--test-seeds", default="122-124")
    parser.add_argument("--branches", type=int, default=6)
    parser.add_argument("--tail", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--cap", type=int, default=2000)
    parser.add_argument("--max-examples", type=int, default=48)
    parser.add_argument("--random-trials", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20261121)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    config = dict(CONFIG)
    config["seed"] = args.seed
    world, agent = fit(**config)
    agent.death_rule_hard = False
    base_rules = agent.death_rules
    if base_rules is None:
        raise SystemExit("death_rules are required")

    refine = collect_counterfactual(
        base_rules, agent, world, seed_spec(args.refine_seeds),
        args.horizon, args.branches, args.tail, args.cap,
        max_examples=args.max_examples,
    )
    test_counter = collect_counterfactual(
        base_rules, agent, world, seed_spec(args.test_seeds),
        args.horizon, args.branches, args.tail, args.cap,
        max_examples=0,
    )
    test_actual_states, test_actual_labels = collect_states(
        agent, world, seed_spec(args.test_seeds),
        cap=args.cap, horizon=args.horizon,
    )
    test_missed = test_counter["missed"]
    test_covered = test_counter["covered"]
    test_all = (
        np.vstack([test_missed, test_covered])
        if len(test_missed) and len(test_covered)
        else (test_missed if len(test_missed) else test_covered)
    )
    base_actual = test_report(
        base_rules, test_actual_states, test_actual_labels
    )
    base_counter = terminal_report(base_rules, test_all)
    base_counter_missed = terminal_report(base_rules, test_missed)

    missed = refine["missed"]
    covered = refine["covered"]
    all_states = (
        np.vstack([missed, covered])
        if len(missed) and len(covered)
        else (missed if len(missed) else covered)
    )

    failure_rules = copy.deepcopy(base_rules)
    failure_added = add_positives(
        failure_rules, missed, len(missed), np.random.RandomState(0)
    )
    failure_actual = test_report(
        failure_rules, test_actual_states, test_actual_labels
    )
    failure_counter = terminal_report(failure_rules, test_all)
    failure_counter_missed = terminal_report(
        failure_rules, test_missed
    )

    rng = np.random.RandomState(args.random_seed)
    random_actual = []
    random_counter = []
    random_counter_missed = []
    random_added = []
    for _trial in range(max(int(args.random_trials), 1)):
        rules = copy.deepcopy(base_rules)
        count = min(len(missed), len(covered)) if len(covered) else 0
        random_added.append(add_positives(
            rules, covered, count, rng
        ))
        random_actual.append(test_report(
            rules, test_actual_states, test_actual_labels
        ))
        random_counter.append(terminal_report(rules, test_all))
        random_counter_missed.append(
            terminal_report(rules, test_missed)
        )

    all_rules = copy.deepcopy(base_rules)
    all_added = add_positives(
        all_rules, all_states, min(len(all_states), args.max_examples),
        np.random.RandomState(1),
    )
    all_actual = test_report(
        all_rules, test_actual_states, test_actual_labels
    )
    all_counter = terminal_report(all_rules, test_all)
    all_counter_missed = terminal_report(all_rules, test_missed)

    summary = dict(
        refine=dict(
            decisions=int(refine["decisions"]),
            candidates=int(refine["candidates"]),
            missed=int(len(missed)),
            covered=int(len(covered)),
        ),
        test=dict(
            actual=dict(
                states=int(len(test_actual_states)),
                terminals=int(test_actual_labels.sum()),
                base=base_actual,
                failure=dict(
                    added=int(failure_added), **failure_actual
                ),
                random=dict(
                    added=random_added,
                    coverage_mean=float(np.mean([
                        row["coverage"] for row in random_actual
                    ])),
                    false_alarm_mean=float(np.mean([
                        row["false_alarm"] for row in random_actual
                    ])),
                ),
                all=dict(added=int(all_added), **all_actual),
            ),
            counterfactual=dict(
                decisions=int(test_counter["decisions"]),
                candidates=int(test_counter["candidates"]),
                terminals=int(len(test_all)),
                missed=int(len(test_missed)),
                covered=int(len(test_covered)),
                base=dict(
                    **base_counter,
                    missed_coverage=base_counter_missed["coverage"],
                    missed_covered=base_counter_missed["covered"],
                ),
                failure=dict(
                    added=int(failure_added), **failure_counter,
                    missed_coverage=failure_counter_missed["coverage"],
                    missed_covered=failure_counter_missed["covered"],
                ),
                random=dict(
                    coverage_mean=float(np.mean([
                        row["coverage"] for row in random_counter
                    ])),
                    missed_coverage_mean=float(np.mean([
                        row["coverage"] for row in random_counter_missed
                    ])),
                    missed_covered_mean=float(np.mean([
                        row["covered"] for row in random_counter_missed
                    ])),
                ),
                all=dict(
                    added=int(all_added), **all_counter,
                    missed_coverage=all_counter_missed["coverage"],
                    missed_covered=all_counter_missed["covered"],
                ),
            ),
        ),
    )
    print(json.dumps(summary, indent=2, default=str))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
