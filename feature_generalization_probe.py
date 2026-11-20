"""Why do induced feature roots stop firing on held-out worlds?

For one fitted arm this reports, per root:

* the induction statistics (train support/precision, held-out test
  support/precision from the same world);
* activation rate on the behavior stream (training distribution);
* activation rate on the held-out evaluation decision states;
* for AND/OR roots, the activation rate of every primitive, so the
  conjunct that kills the root is visible (threshold brittleness).

Read-only; no task geometry.
"""
import argparse
import json

import numpy as np

from eval_protocol import parse_seed_spec
from node_features import Primitive
from substrate_nodes import _curiosity_off, collect, evaluate, fit


def _primitives(feature):
    if isinstance(feature, Primitive):
        return [feature]
    children = getattr(feature, "children", ())
    result = []
    for child in children:
        result.extend(_primitives(child))
    return result


def _rate(mask):
    mask = np.asarray(mask, dtype=bool)
    return float(mask.mean()) if mask.size else 0.0


def run(seed, eval_seeds, memory_credit,
        rule_min_coverage=0.0, rule_min_primitive_coverage=0.0,
        rule_validation_seed=0, rule_validation_frames=600):
    world, agent = fit(
        seed,
        steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, lag=3,
        curiosity_weight=0.5,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=8,
        step_stride=2,
        memory_credit=memory_credit,
        rule_min_coverage=rule_min_coverage,
        rule_min_primitive_coverage=rule_min_primitive_coverage,
        rule_validation_seed=rule_validation_seed,
        rule_validation_frames=rule_validation_frames,
    )
    behavior, _counts = collect(seed, 1500, coverage=0.1, lag=3)
    behavior_obs = np.asarray(
        [row["obs"] for row in behavior], dtype=float
    )
    eval_rows = []
    with _curiosity_off(agent):
        for eval_seed in eval_seeds:
            _frames, _actions, rows = evaluate(
                seed, world, agent, cap=3000, horizon=8,
                step_stride=2, env_seed=eval_seed * 10000,
                record=True, episode=0,
            )
            eval_rows.extend(rows)
    eval_obs = np.asarray([row["obs"] for row in eval_rows], dtype=float)

    rules = list(agent.rules.rules)
    roots = []
    for index, rule in enumerate(rules):
        root_id = (
            agent.feature_ids[index]
            if index < len(agent.feature_ids) else None
        )
        behavior_rate = _rate(rule.feature.mask(behavior_obs))
        eval_rate = _rate(rule.feature.mask(eval_obs))
        primitives = []
        for primitive in _primitives(rule.feature):
            primitives.append(dict(
                label=primitive.label(),
                behavior_rate=round(_rate(primitive.mask(behavior_obs)), 4),
                eval_rate=round(_rate(primitive.mask(eval_obs)), 4),
            ))
        bottleneck = min(
            primitives, key=lambda item: item["eval_rate"]
        ) if primitives else None
        roots.append(dict(
            root_id=root_id,
            label=rule.feature.label(),
            support=int(rule.support),
            precision=round(float(rule.precision), 3),
            test_support=int(rule.test_support),
            test_precision=round(float(rule.test_precision), 3),
            behavior_rate=round(behavior_rate, 4),
            eval_rate=round(eval_rate, 4),
            bottleneck=bottleneck,
            primitives=primitives,
        ))
    zero_eval = sum(1 for item in roots if item["eval_rate"] <= 0.0)
    rare_child = sum(
        1 for item in roots
        if item["bottleneck"] and item["bottleneck"]["eval_rate"] < 0.05
    )
    return dict(
        arm="memory-credit" if memory_credit else "control",
        roots=roots,
        summary=dict(
            roots=len(roots),
            zero_eval_roots=zero_eval,
            roots_with_rare_child=rare_child,
            mean_behavior_rate=round(float(np.mean(
                [item["behavior_rate"] for item in roots]
            )) if roots else 0.0, 4),
            mean_eval_rate=round(float(np.mean(
                [item["eval_rate"] for item in roots]
            )) if roots else 0.0, 4),
            eval_rows=len(eval_rows),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--memory-credit", action="store_true")
    parser.add_argument("--rule-min-coverage", type=float, default=0.0)
    parser.add_argument(
        "--rule-min-primitive-coverage", type=float, default=0.0
    )
    parser.add_argument("--rule-validation-seed", type=int, default=0)
    parser.add_argument("--rule-validation-frames", type=int, default=600)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = run(
        args.seed, parse_seed_spec(args.eval_seeds), args.memory_credit,
        rule_min_coverage=args.rule_min_coverage,
        rule_min_primitive_coverage=args.rule_min_primitive_coverage,
        rule_validation_seed=args.rule_validation_seed,
        rule_validation_frames=args.rule_validation_frames,
    )
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
