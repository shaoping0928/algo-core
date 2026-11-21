"""Do real planner failures improve death rules on unseen worlds?

The probe fits the benchmark agent on a source world, collects the states the
planner actually enters on separate refinement worlds, and compares three ways
of adding the same kind of data to the 0/1 death rules:

- failure: missed terminals become positives and false alarms become
  negatives (``death_rule_loop.refine_once``);
- random: the same number of randomly drawn refinement states, labelled as
  usual;
- terminal: every terminal state from the refinement worlds.

Evaluation is on held-out worlds the refinement never sees.  The metric is
the death side only: terminal coverage, unknown terminals, and non-terminal
false alarms.  No decision path is changed.  The benchmark world supplies the
stream; the rule learner and the loop see only states and 0/1 labels.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

import substrate_nodes as sn
from death_rule_loop import refine_once
from death_transition_quality import seed_spec
from substrate_nodes import LAG, _curiosity_off, evaluate, fit

CONFIG = dict(
    seed=100, steps=1500, epochs=1, max_nodes=1000,
    birth_threshold=0.45, dagger_rounds=1, dagger_frames=5000,
    dagger_cap=5000, dagger_horizon=8, step_stride=2,
    curiosity_weight=0.5, death_rules=True,
    death_rule_max_thresholds=256, death_rule_boundary_margin=0.01,
)


def collect_states(agent, world, seeds, cap=5000, horizon=8):
    """Entered states and point-event terminal labels from planner runs."""
    states = []
    labels = []
    for eval_seed in seeds:
        env_seed = int(eval_seed) * 10000
        with _curiosity_off(agent):
            _frames, _actions, rows = evaluate(
                CONFIG["seed"], world, agent, cap=cap, horizon=horizon,
                lag=LAG, step_stride=2, env_seed=env_seed, record=True,
                episode=0,
            )
        for row in rows:
            states.append(np.asarray(row["next_obs"], dtype=float))
            labels.append(
                1 if float(row["signals"][1]) > 0.5 else 0
            )
    if not states:
        return np.zeros((0, 0)), np.zeros(0, dtype=int)
    return np.asarray(states), np.asarray(labels, dtype=int)


def test_report(rules, states, labels):
    report = rules.coverage_report(states, labels)
    terminal = int(report["positive"])
    covered = int(report["death_covered_positive"])
    negatives = int(report["negative"])
    false_alarm = int(report["death_covered_negative"])
    return dict(
        states=int(report["total"]),
        terminals=terminal,
        covered=covered,
        unknown_terminal=terminal - covered,
        coverage=float(covered) / float(terminal) if terminal else 0.0,
        negatives=negatives,
        false_alarm=false_alarm,
        false_alarm_rate=(
            float(false_alarm) / float(negatives) if negatives else 0.0
        ),
    )


def add_random(rules, states, labels, count, rng):
    if count <= 0 or len(states) == 0:
        return 0
    count = min(int(count), len(states))
    indices = rng.choice(len(states), count, replace=False)
    rules.add_examples(states[indices], labels[indices])
    return int(count)


def add_all_terminals(rules, states, labels):
    terminals = labels == 1
    count = int(terminals.sum())
    if count:
        rules.add_examples(states[terminals], labels[terminals])
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--refine-seeds", default="120-121")
    parser.add_argument("--test-seeds", default="122-124")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--cap", type=int, default=5000)
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

    refine_states, refine_labels = collect_states(
        agent, world, seed_spec(args.refine_seeds),
        cap=args.cap, horizon=args.horizon,
    )
    test_states, test_labels = collect_states(
        agent, world, seed_spec(args.test_seeds),
        cap=args.cap, horizon=args.horizon,
    )
    if len(refine_states) == 0 or len(test_states) == 0:
        raise SystemExit("no refinement or test states collected")

    base_test = test_report(base_rules, test_states, test_labels)
    refine_before = base_rules.coverage_report(
        refine_states, refine_labels
    )
    failure_rules = copy.deepcopy(base_rules)
    failure_round = refine_once(
        failure_rules, refine_states, refine_labels
    )
    failure_test = test_report(
        failure_rules, test_states, test_labels
    )
    added = (
        failure_round.added_missed + failure_round.added_false_alarm
    )
    positive_rules = copy.deepcopy(base_rules)
    positive_round = refine_once(
        positive_rules, refine_states, refine_labels,
        include_false_alarms=False,
    )
    positive_test = test_report(
        positive_rules, test_states, test_labels
    )
    positive_added = positive_round.added_missed

    rng = np.random.RandomState(args.random_seed)
    random_tests = []
    random_added = []
    random_positive_tests = []
    random_positive_added = []
    for _trial in range(max(int(args.random_trials), 1)):
        rules = copy.deepcopy(base_rules)
        random_added.append(add_random(
            rules, refine_states, refine_labels, added, rng
        ))
        random_tests.append(test_report(rules, test_states, test_labels))
        rules = copy.deepcopy(base_rules)
        random_positive_added.append(add_random(
            rules, refine_states, refine_labels, positive_added, rng
        ))
        random_positive_tests.append(
            test_report(rules, test_states, test_labels)
        )

    terminal_rules = copy.deepcopy(base_rules)
    terminal_added = add_all_terminals(
        terminal_rules, refine_states, refine_labels
    )
    terminal_test = test_report(
        terminal_rules, test_states, test_labels
    )

    summary = dict(
        refine=dict(
            states=int(len(refine_states)),
            terminals=int(refine_labels.sum()),
            coverage=float(refine_before["death_coverage"]),
        ),
        test=dict(
            states=int(len(test_states)),
            terminals=int(test_labels.sum()),
            base=base_test,
            failure=dict(added=int(added), **failure_test),
            failure_positives=dict(
                added=int(positive_added),
                **positive_test,
            ),
            random=dict(
                added=random_added,
                coverage_mean=float(np.mean([
                    row["coverage"] for row in random_tests
                ])),
                coverage_min=float(np.min([
                    row["coverage"] for row in random_tests
                ])),
                coverage_max=float(np.max([
                    row["coverage"] for row in random_tests
                ])),
                false_alarm_mean=float(np.mean([
                    row["false_alarm"] for row in random_tests
                ])),
            ),
            random_positives=dict(
                added=random_positive_added,
                coverage_mean=float(np.mean([
                    row["coverage"] for row in random_positive_tests
                ])),
                coverage_min=float(np.min([
                    row["coverage"] for row in random_positive_tests
                ])),
                coverage_max=float(np.max([
                    row["coverage"] for row in random_positive_tests
                ])),
                false_alarm_mean=float(np.mean([
                    row["false_alarm"] for row in random_positive_tests
                ])),
            ),
            terminal=dict(
                added=int(terminal_added),
                **terminal_test,
            ),
        ),
        failure_round=dict(
            added_missed=failure_round.added_missed,
            added_false_alarm=failure_round.added_false_alarm,
            refine_coverage_before=float(
                failure_round.before["death_coverage"]
            ),
            refine_coverage_after=float(
                failure_round.after["death_coverage"]
            ),
            refine_false_alarm_before=int(
                failure_round.before["death_covered_negative"]
            ),
            refine_false_alarm_after=int(
                failure_round.after["death_covered_negative"]
            ),
        ),
    )
    print(json.dumps(summary, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
