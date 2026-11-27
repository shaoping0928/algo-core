"""Compare absolute-coordinate and relative-feature death rules.

The same terminal transitions are used for every arm; only the state
representation changes:

- raw coordinates,
- raw + all pairwise differences,
- pairwise differences only,
- raw + normalized pairwise differences.

The death-rule learner is otherwise identical.  Evaluation reports
terminal coverage, unknown terminals, false positives and the learned
rules on held-out worlds.  No decision path is changed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import substrate_nodes as sn
from death_rules import UNKNOWN, DeathRuleSet
from node_features import expand_observation
from substrate_nodes import LAG, _curiosity_off, evaluate

TRAIN_KWARGS = dict(
    seed=100, lag=LAG, max_nodes=1000, birth_threshold=0.45,
    component_kinds=(), component_min_count=8, max_components=128,
    component_weight=0.5, component_gating=False, memory_weight=0.1,
    association_rate=0.0, induce_features=True, precursor_labels=False,
    pain_horizon=None, voi_target=3, feature_value_weight=0.02,
    value_horizon=None, ood_trust_scale=0.0, obs_trust=1.0,
    feature_action_prior=2.0, feature_value_prior=4.0, value_features=True,
    value_label_quantile=0.8, subgraph_cells=False, predicate_readouts=False,
    event_memory_risk=True, contrast_objective=False,
    value_trust_samples=150.0, control_risk_weight=0.0,
    value_bank_only=False, importance_weighting=False, voi_threat_min=0.25,
    curiosity_weight=0.5, memory_credit=False, rule_min_coverage=0.0,
    rule_min_primitive_coverage=0.0, pairwise_features=False,
    branch_transitions=False, action_edge_weight=0.0,
    action_edge_trust_samples=50.0, compose_readout=True,
    iterative_induction=False, value_dispersion=False, epochs=1,
)


def seed_spec(text):
    seeds = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(part))
    return seeds


def pairwise_differences(states):
    expanded = expand_observation(states)
    return expanded[:, states.shape[1]:]


def normalized_differences(states):
    states = np.asarray(states, dtype=float)
    width = states.shape[1]
    parts = []
    for first in range(width):
        for second in range(first + 1, width):
            left = states[:, first]
            right = states[:, second]
            scale = np.abs(left) + np.abs(right) + 1e-6
            parts.append((left - right) / scale)
    return np.column_stack(parts)


VARIANTS = {
    "raw": lambda states: np.asarray(states, dtype=float),
    "raw_plus_diff": lambda states: expand_observation(states),
    "diff_only": pairwise_differences,
    "raw_plus_normdiff": lambda states: np.concatenate(
        [np.asarray(states, dtype=float), normalized_differences(states)],
        axis=1,
    ),
}


def collect_training(seed, steps, dagger_frames):
    transitions, _ = sn.collect(
        seed, steps, coverage=0.1, lag=LAG
    )
    if dagger_frames > 0:
        world, agent = sn._train(transitions, **TRAIN_KWARGS)
        planner, _frames = sn.collect_planner(
            seed, world, agent, episodes=max(2, dagger_frames),
            cap=5000, horizon=8, lag=LAG, step_stride=2,
            offset=50000, max_frames=dagger_frames,
        )
        transitions = list(transitions) + planner
    return transitions


def collect_eval(world, agent, seed, seeds, cap=5000):
    states = []
    labels = []
    for eval_seed in seeds:
        with _curiosity_off(agent):
            _frames, _actions, rows = evaluate(
                seed, world, agent, cap=cap, horizon=8, lag=LAG,
                step_stride=2, env_seed=eval_seed * 10000,
                record=True, episode=0,
            )
        for row in rows:
            states.append(np.asarray(row["next_obs"], dtype=float))
            labels.append(1 if float(row["signals"][1]) > 0.5 else 0)
    return np.asarray(states), np.asarray(labels, dtype=int)


def terminal_metrics(rules, states, labels):
    prediction = rules.classify_death(states)
    terminal = labels == 1
    tp = int(np.sum(terminal & (prediction == 1)))
    unknown = int(np.sum(terminal & (prediction == UNKNOWN)))
    fp = int(np.sum((~terminal) & (prediction == 1)))
    full = rules.classify(states)
    false_safe = int(np.sum(terminal & (full == 0)))
    return dict(
        terminals=int(terminal.sum()),
        terminal_covered=tp,
        terminal_unknown=unknown,
        terminal_false_safe=false_safe,
        safe_false_positive=fp,
        coverage=(float(tp) / float(terminal.sum())
                  if terminal.any() else 0.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--dagger-frames", type=int, default=5000)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--max-thresholds", type=int, default=96)
    parser.add_argument("--max-rules", type=int, default=24)
    parser.add_argument("--max-literals", type=int, default=5)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    transitions = collect_training(
        args.seed, args.steps, args.dagger_frames
    )
    train_states = np.asarray(
        [row["next_obs"] for row in transitions], dtype=float
    )
    train_labels = np.asarray(
        [1 if float(row["signals"][1]) > 0.5 else 0
         for row in transitions], dtype=int
    )
    world, agent = sn._train(
        transitions, **TRAIN_KWARGS
    )
    eval_states, eval_labels = collect_eval(
        world, agent, args.seed, seed_spec(args.eval_seeds)
    )
    print("training transitions: %d  positives=%d" % (
        len(train_states), int(train_labels.sum())
    ))
    print("evaluation states: %d  terminals=%d" % (
        len(eval_states), int(eval_labels.sum())
    ))
    report = dict(
        config=vars(args),
        training=dict(
            states=int(len(train_states)),
            positives=int(train_labels.sum()),
        ),
        evaluation=dict(
            states=int(len(eval_states)),
            terminals=int(eval_labels.sum()),
        ),
        arms={},
    )
    for name, transform in VARIANTS.items():
        train_x = transform(train_states)
        eval_x = transform(eval_states)
        rules = DeathRuleSet(
            max_rules=args.max_rules,
            max_literals=args.max_literals,
            min_support=2,
            max_thresholds=args.max_thresholds,
            coverage_first=True,
            boundary_margin=0.01,
        ).fit(train_x, train_labels)
        metrics = terminal_metrics(rules, eval_x, eval_labels)
        print("\n== %s ==" % name)
        print(json.dumps(metrics, indent=2))
        print("training coverage:")
        print(json.dumps(rules.coverage_, indent=2))
        print("death rules:")
        print(rules.describe(12))
        print("modes:")
        print(rules.describe_modes(1, limit=4))
        report["arms"][name] = dict(
            metrics=metrics,
            training_coverage=rules.coverage_,
            rules=[rule.text() for rule in rules.rules_],
            modes=[
                dict(signature=list(mode["signature"]),
                     rules=[rule.text() for rule in mode["rules"]])
                for mode in rules.modes(1)
            ],
        )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
