"""Offline validation of the 0/1 death-condition learner.

The probe fits :class:`death_rules.DeathRuleSet` on the real transition
stream (entered states labelled 1 when the transition was terminal) and
reports:

- the rules and the discovered coordinate signatures (modes),
- held-out episode verification,
- cross-world verification on held-out worlds,
- a counterexample-refinement round on separate worlds, then re-testing
  on worlds the refinement never saw,
- a comparison against the current probabilistic hazard read on the
  planner's own evaluation trajectories, including terminal states the
  rule catches while the hazard reads zero.

No decision path is changed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import substrate_nodes as sn
from death_rules import UNKNOWN, DeathRuleSet
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


def encode_next(agent, observation, row):
    """Encode the entered state with the action history that produced it."""
    observation = np.asarray(observation, dtype=float).reshape(-1)
    history = np.asarray(row["history"], dtype=float).reshape(-1)
    pending = history[1:1 + agent.lag]
    if pending.size < agent.lag:
        pending = np.concatenate([
            pending, np.zeros(agent.lag - pending.size)
        ])
    if agent.lag:
        state = np.concatenate([observation, pending[::-1]])
    else:
        state = observation
    return state / agent.scales


def episode_split(rows, every=5):
    episodes = sorted({int(row["episode"]) for row in rows})
    mapping = {episode: index for index, episode in enumerate(episodes)}
    test = np.asarray([
        mapping[int(row["episode"])] % every == every - 1 for row in rows
    ])
    return ~test, test


def collect_world_states(seeds, steps, coverage=0.1):
    states = []
    labels = []
    rows_out = []
    for eval_seed in seeds:
        rows, _ = sn.collect(
            eval_seed, steps, coverage=coverage, lag=LAG
        )
        for row in rows:
            states.append(np.asarray(row["next_obs"], dtype=float))
            labels.append(
                1 if float(row["signals"][1]) > 0.5 else 0
            )
            rows_out.append(row)
    return np.asarray(states), np.asarray(labels, dtype=int), rows_out


def hazard_reads(agent, states, rows):
    values = np.zeros(len(states), dtype=float)
    for index, row in enumerate(rows):
        encoded = encode_next(agent, states[index], row)
        values[index] = float(agent.pain_signal_batch(encoded[None])[0])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--refine-seeds", default="120-121")
    parser.add_argument("--test-seeds", default="122-124")
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--max-thresholds", type=int, default=256)
    parser.add_argument("--max-rules", type=int, default=24)
    parser.add_argument("--max-literals", type=int, default=5)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--planner-eval", action="store_true",
                        help="also run the trained planner on the test "
                             "worlds and compare the rule with the hazard")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    transitions, _counts = sn.collect(
        args.seed, args.steps, coverage=args.coverage, lag=LAG
    )
    states = np.asarray([row["next_obs"] for row in transitions], dtype=float)
    labels = np.asarray(
        [1 if float(row["signals"][1]) > 0.5 else 0 for row in transitions],
        dtype=int,
    )
    train, test = episode_split(transitions)
    print("training transitions: %d  positives=%d  held-out positives=%d"
          % (len(states), int(labels[train].sum()), int(labels[test].sum())))
    rules = DeathRuleSet(
        max_rules=args.max_rules,
        max_literals=args.max_literals,
        min_support=2,
        max_thresholds=args.max_thresholds,
    ).fit(states[train], labels[train])
    print("\nrules:")
    print(rules.describe(24))
    print("\nmodes:")
    print(rules.describe_modes(1))
    print("\nheld-out episodes:")
    print(json.dumps(rules.verify(states[test], labels[test]), indent=2))
    print("\nall training states:")
    print(json.dumps(rules.verify(states, labels), indent=2))

    refine_states, refine_labels, _refine_rows = collect_world_states(
        seed_spec(args.refine_seeds), args.eval_steps, args.coverage
    )
    test_states, test_labels, test_rows = collect_world_states(
        seed_spec(args.test_seeds), args.eval_steps, args.coverage
    )
    print("\nrefine worlds (%s): %d states, %d terminal" % (
        args.refine_seeds, len(refine_states), int(refine_labels.sum()),
    ))
    print("test worlds (%s): %d states, %d terminal" % (
        args.test_seeds, len(test_states), int(test_labels.sum()),
    ))
    before = rules.verify(test_states, test_labels)
    print("\ntest before refinement:")
    print(json.dumps(before, indent=2))
    refinement = []
    for round_index in range(int(args.refine_rounds)):
        prediction = rules.classify(refine_states)
        wrong = prediction != refine_labels
        if not wrong.any():
            break
        rules.add_examples(refine_states[wrong], refine_labels[wrong])
        after = rules.verify(test_states, test_labels)
        refinement.append(dict(
            round=round_index + 1,
            added=int(wrong.sum()),
            false_positive=int(after["false_positive"]),
            false_negative=int(after["false_negative"]),
            unknown=int(after["unknown"]),
        ))
        print("\nrefinement round %d: added %d counterexamples" % (
            round_index + 1, int(wrong.sum()),
        ))
        print(json.dumps(after, indent=2))
    print("\nrefinement trace:")
    print(json.dumps(refinement, indent=2))
    print("\nrules after behavior refinement:")
    print(rules.describe(24))
    print("\nmodes after behavior refinement:")
    print(rules.describe_modes(1))
    rules_after_behavior = [rule.text() for rule in rules.rules_]

    world, agent = sn._train(transitions, **TRAIN_KWARGS)
    hazard = hazard_reads(agent, test_states, test_rows)
    rule_prediction = rules.classify(test_states)
    death = test_labels == 1
    print("\nbehavior test worlds vs hazard:")
    print("terminal states: %d" % int(death.sum()))
    print("rule catches: %d" % int(np.sum(death & (rule_prediction == 1))))
    print("rule says safe: %d" % int(np.sum(death & (rule_prediction == 0))))
    print("rule unknown: %d" % int(np.sum(death & (rule_prediction == UNKNOWN))))
    print("hazard<0.1 terminal states: %d" % int(np.sum(
        death & (hazard < 0.1)
    )))
    print("hazard mean on terminal: %.4f" % (
        float(np.mean(hazard[death])) if death.any() else 0.0
    ))
    print("hazard mean on safe: %.4f" % (
        float(np.mean(hazard[~death])) if (~death).any() else 0.0
    ))

    planner = None
    planner_refinement = []
    if args.planner_eval:
        # Refine on the decision distribution too: behavior transitions
        # and planner transitions are not the same distribution.
        planner_refine_states = []
        planner_refine_labels = []
        with _curiosity_off(agent):
            for eval_seed in seed_spec(args.refine_seeds):
                _frames, _actions, rows = evaluate(
                    args.seed, world, agent, cap=5000, horizon=8, lag=LAG,
                    step_stride=2, env_seed=eval_seed * 10000,
                    record=True, episode=0,
                )
                for row in rows:
                    planner_refine_states.append(
                        np.asarray(row["next_obs"], dtype=float)
                    )
                    planner_refine_labels.append(
                        1 if float(row["signals"][1]) > 0.5 else 0
                    )
        planner_refine_states = np.asarray(planner_refine_states)
        planner_refine_labels = np.asarray(
            planner_refine_labels, dtype=int
        )
        planner_refinement = []
        for round_index in range(int(args.refine_rounds)):
            prediction = rules.classify(planner_refine_states)
            wrong = prediction != planner_refine_labels
            if not wrong.any():
                break
            rules.add_examples(
                planner_refine_states[wrong],
                planner_refine_labels[wrong],
            )
            planner_refinement.append(dict(
                round=round_index + 1,
                added=int(wrong.sum()),
            ))
        print("\nplanner refinement trace:")
        print(json.dumps(planner_refinement, indent=2))
        print("\nrules after planner refinement:")
        print(rules.describe(24))

        planner_states = []
        planner_labels = []
        planner_rows = []
        with _curiosity_off(agent):
            for eval_seed in seed_spec(args.test_seeds):
                _frames, _actions, rows = evaluate(
                    args.seed, world, agent, cap=5000, horizon=8, lag=LAG,
                    step_stride=2, env_seed=eval_seed * 10000,
                    record=True, episode=0,
                )
                for row in rows:
                    planner_states.append(
                        np.asarray(row["next_obs"], dtype=float)
                    )
                    planner_labels.append(
                        1 if float(row["signals"][1]) > 0.5 else 0
                    )
                    planner_rows.append(row)
        planner_states = np.asarray(planner_states)
        planner_labels = np.asarray(planner_labels, dtype=int)
        planner_hazard = hazard_reads(agent, planner_states, planner_rows)
        planner_prediction = rules.classify(planner_states)
        terminal = planner_labels == 1
        blind = terminal & (planner_hazard < 0.1)
        planner = dict(
            states=int(len(planner_states)),
            terminal=int(terminal.sum()),
            rule_catches=int(np.sum(terminal & (
                planner_prediction == 1
            ))),
            rule_safe=int(np.sum(terminal & (
                planner_prediction == 0
            ))),
            rule_unknown=int(np.sum(terminal & (
                planner_prediction == UNKNOWN
            ))),
            hazard_blind=int(blind.sum()),
            rule_catches_blind=int(np.sum(blind & (
                planner_prediction == 1
            ))),
            hazard_terminal_mean=(
                float(np.mean(planner_hazard[terminal]))
                if terminal.any() else 0.0
            ),
            hazard_safe_mean=(
                float(np.mean(planner_hazard[~terminal]))
                if (~terminal).any() else 0.0
            ),
        )
        print("\nplanner evaluation vs hazard:")
        print(json.dumps(planner, indent=2))

    if args.report:
        report = dict(
            config=vars(args),
            rules=[rule.text() for rule in rules.rules_],
            modes=[
                dict(signature=list(mode["signature"]),
                     rules=[rule.text() for rule in mode["rules"]])
                for mode in rules.modes(1)
            ],
            heldout=rules.verify(states[test], labels[test]),
            all_states=rules.verify(states, labels),
            test_before=before,
            refinement=refinement,
            test_after=rules.verify(test_states, test_labels),
            rules_after_behavior=rules_after_behavior,
            rules_after_planner=[rule.text() for rule in rules.rules_],
            behavior_hazard=dict(
                terminal=int(death.sum()),
                rule_catches=int(np.sum(death & (rule_prediction == 1))),
                hazard_blind=int(np.sum(death & (hazard < 0.1))),
                terminal_mean=(
                    float(np.mean(hazard[death])) if death.any() else 0.0
                ),
                safe_mean=(
                    float(np.mean(hazard[~death])) if (~death).any() else 0.0
                ),
            ),
            planner=planner,
            planner_refinement=planner_refinement,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
