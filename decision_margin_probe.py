"""How close are the candidates at a decision?  Measure the margin, not the mean.

If the best and second-best plans differ by less than the size of a term the
planner is being asked to weigh, then that term is not "a small correction" -
it is deciding the argmin.  This reports, per decision, the spread of the
hazard column and of the scalarized cost across candidates, and the margin
between the best and the runner-up.
"""

import argparse

import numpy as np

from substrate_nodes import (
    LAG, _association_value_table, _curiosity_off, evaluate, fit,
)


def summarize(batches):
    if not batches:
        return None
    scores = np.vstack([
        np.asarray(batch["scores"], dtype=float) for batch in batches
    ])
    if len(scores) < 2:
        return None
    cost = scores[:, 0]
    risk = scores[:, 2]
    order = np.argsort(cost)
    margin = float(cost[order[1]] - cost[order[0]])
    return dict(
        candidates=len(scores),
        cost_spread=float(cost.max() - cost.min()),
        cost_margin=margin,
        risk_spread=float(risk.max() - risk.min()),
        near_tie=float(margin < 1e-3),
    )


def report(name, decisions):
    rows = [row for row in decisions if row is not None]
    if not rows:
        print("== %s: no decisions ==" % name)
        return
    margin = np.asarray([row["cost_margin"] for row in rows])
    spread = np.asarray([row["cost_spread"] for row in rows])
    risk_spread = np.asarray([row["risk_spread"] for row in rows])
    near = np.asarray([row["near_tie"] for row in rows])
    print("== %s: %d decisions, %d candidates each ==" % (
        name, len(rows), rows[0]["candidates"],
    ))
    print("  hazard spread: p10 %.4f  p50 %.4f  p90 %.4f" % tuple(
        np.percentile(risk_spread, [10, 50, 90])
    ))
    print("  cost spread:   p10 %.4f  p50 %.4f  p90 %.4f" % tuple(
        np.percentile(spread, [10, 50, 90])
    ))
    print("  best-vs-second margin: p10 %.4f  p50 %.4f  p90 %.4f  "
          "margin<1e-3 on %.2f of decisions" % (
              float(np.percentile(margin, 10)),
              float(np.percentile(margin, 50)),
              float(np.percentile(margin, 90)),
              float(near.mean()),
          ))


def first_action_stats(batches):
    """Best continuation for each first action, and the gap between them."""
    if not batches:
        return None
    scores = np.vstack([
        np.asarray(batch["scores"], dtype=float) for batch in batches
    ])
    sequences = np.vstack([
        np.asarray(batch["sequences"], dtype=float) for batch in batches
    ])
    cost = scores[:, 0]
    first = sequences[:, 0]
    best = {}
    for action in np.unique(first):
        mask = first == action
        best[float(action)] = float(cost[mask].min())
    if len(best) < 2:
        return None
    left, right = sorted(best)
    gap = abs(best[left] - best[right])
    return dict(
        gap=float(gap),
        better=left if best[left] <= best[right] else right,
        best=best,
    )


def report_first(name, rows):
    rows = [row for row in rows if row is not None]
    if not rows:
        print("== %s: no first-action pairs ==" % name)
        return
    gap = np.asarray([row["gap"] for row in rows])
    better = np.asarray([row["better"] for row in rows])
    print("== %s: first-action gap p10 %.4f  p50 %.4f  p90 %.4f  "
          "gap<1e-3 on %.2f  flap preferred on %.2f ==" % (
              name,
              float(np.percentile(gap, 10)),
              float(np.percentile(gap, 50)),
              float(np.percentile(gap, 90)),
              float((gap < 1e-3).mean()),
              float((better == 1).mean()),
          ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=120)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--weight", type=float, default=1.0)
    args = parser.parse_args()

    world, agent = fit(
        seed=args.seed, steps=args.steps, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
    )
    values, support = _association_value_table(
        agent.milestone_training_rows, agent
    )
    agent.association_values = values
    agent.association_support = support
    kwargs = dict(
        cap=5000, horizon=args.horizon, lag=LAG, step_stride=2,
        env_seed=args.eval_seed * 10000, episode=args.eval_seed,
        record=True,
    )
    with _curiosity_off(agent):
        for name, readout, weight in (
            ("cells", "cells", 0.0),
            ("assoc", "association", args.weight),
        ):
            agent.value_readout = readout
            agent.feature_value_weight = weight
            decisions = []
            output = []
            evaluate(
                args.seed, world, agent, candidate_out=output, **kwargs
            )
            for batches in output:
                decisions.append(summarize(batches))
            report(name, decisions)
            report_first(name, [first_action_stats(batches)
                                for batches in output])


if __name__ == "__main__":
    main()
