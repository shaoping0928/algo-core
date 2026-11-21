"""Can a cheap coarse pass see what the 8-frame fine pass cannot?

The fine search is already optimal inside its horizon (oracle selection
ties it exactly), and the death is sealed about ten frames ahead, so all
remaining headroom is in a long-range, coarse layer.

This probe builds that layer as temporal abstraction only:

* a **macro action** is a fixed action pattern over ``k`` frames
  (all-still, all-flap, flap-then-glide, glide-then-flap);
* the coarse pass chains ``steps`` of them, covering ``steps * k`` frames
  with ``len(macros) ** steps`` leaves (64 for 4 macros x 3 steps);
* its cost is the hazard the planner already reads, accumulated along
  the predicted trajectory -- the world model is the same one.

For every state it reports, across the macro plans:

* ``rho`` - rank correlation between coarse cost and true survival
  (negative is correct: cheaper must mean longer);
* ``best_survival`` - true survival of the coarse-cheapest macro plan;
* ``ceiling`` - true survival of the best macro plan in the set;
so the caller can tell apart "the macro set cannot express an escape"
from "the coarse cost cannot rank the escapes".

Nothing is learned; the coarse plans are only replayed for measurement.
"""

import argparse
import itertools
import json

import numpy as np

from latent_world import latent
from substrate_generic import history_features
from substrate_nodes import LAG, _curiosity_off, evaluate, fit, policy_result

MACRO_SETS = {
    4: {
        "still": lambda k: [0] * k,
        "flap": lambda k: [1] * k,
        "flap_first": lambda k: [1] + [0] * (k - 1),
        "flap_last": lambda k: [0] * (k - 1) + [1],
    },
    8: {
        "still": lambda k: [0] * k,
        "flap": lambda k: [1] * k,
        "flap_first": lambda k: [1] + [0] * (k - 1),
        "flap_last": lambda k: [0] * (k - 1) + [1],
        "flap_second": lambda k: [0, 1] + [0] * (k - 2),
        "flap_third": lambda k: [0, 0, 1] + [0] * (k - 3),
        "flap_two_first": lambda k: [1, 1] + [0] * (k - 2),
        "flap_two_last": lambda k: [0] * (k - 2) + [1, 1],
    },
}


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left, right):
    a = rankdata(left)
    b = rankdata(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else 0.0


def coarse_plans(k, steps, macro_count=4):
    macros = MACRO_SETS[int(macro_count)]
    names = list(macros)
    plans = []
    labels = []
    for combination in itertools.product(names, repeat=steps):
        plan = []
        for name in combination:
            plan.extend(macros[name](k))
        plans.append(np.asarray(plan, dtype=float))
        labels.append("+".join(combination))
    return np.asarray(plans), labels


def coarse_cost(agent, world, prefix, start, plans):
    """Hazard accumulated along each macro plan's predicted trajectory."""
    states = np.broadcast_to(
        np.asarray(start, dtype=float), (len(plans), len(start))
    ).copy()
    total = np.zeros(len(plans), dtype=float)
    for step in range(plans.shape[1]):
        history = history_features(
            plans, step + 1, prefix, action_dim=agent.lag + 1
        )
        pending = history[:, :agent.lag][:, ::-1]
        encoded = np.concatenate([states, pending], axis=1) / agent.scales
        total += np.asarray(agent.pain_signal_batch(encoded), dtype=float)
        states = np.asarray(world.predict(states, history), dtype=float)
    return total


def run(args):
    world, agent = fit(
        seed=args.seed, steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
    )
    plans, labels = coarse_plans(
        args.macro_frames, args.macro_steps, args.macro_count
    )
    span = args.macro_frames * args.macro_steps
    rows = []
    with _curiosity_off(agent):
        for eval_seed in args.eval_seeds:
            env_seed = int(eval_seed) * 10000
            frames, actions, _ = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=args.horizon,
                lag=LAG, step_stride=2, env_seed=env_seed,
            )
            if frames >= args.cap or frames < span + 2:
                continue
            for lag_back in range(args.min_lag, args.max_lag + 1):
                point = frames - lag_back
                if point < 1:
                    continue
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
                cost = coarse_cost(
                    agent, world, prefix, env.obs(), plans
                )
                survival = np.empty(len(plans), dtype=int)
                from planner_rollout_quality import (
                    flatten_batches, oracle_plan,
                )
                for index, plan in enumerate(plans):
                    death, _state, _dead = oracle_plan(env, plan)
                    survival[index] = (
                        span + 1 if death < 0 else int(death)
                    )
                cheapest = int(np.argmin(cost))
                control = None
                if args.control_plans:
                    rng = np.random.RandomState(args.seed * 1000 + point)
                    controls = (
                        rng.rand(args.control_plans, span) < 0.5
                    ).astype(float)
                    control_survival = np.empty(len(controls), dtype=int)
                    for index, plan in enumerate(controls):
                        death, _state, _dead = oracle_plan(env, plan)
                        control_survival[index] = (
                            span + 1 if death < 0 else int(death)
                        )
                    control = dict(
                        best=int(control_survival.max()),
                        mean=float(control_survival.mean()),
                        share=float(np.mean(control_survival > span)),
                    )
                # The fine layer's own choice at the same state, for the
                # head-to-head: the plan the 8-frame search executes.
                batches = []
                policy_result(
                    env, world, agent, prefix, args.horizon, lag=LAG,
                    step_stride=2, candidate_callback=batches.append,
                )
                fine = None
                if batches:
                    trace = flatten_batches(batches)
                    feasible = (
                        np.asarray(trace["feasible"], dtype=bool)
                        & np.asarray(trace["finite"], dtype=bool)
                    )
                    if feasible.any():
                        pool = np.flatnonzero(feasible)
                        scores = np.asarray(trace["scores"], dtype=float)
                        sequences = np.asarray(
                            trace["sequences"], dtype=float
                        )
                        chosen = pool[np.argmin(scores[pool, 0])]
                        death, _state, _dead = oracle_plan(
                            env, sequences[chosen]
                        )
                        fine = args.horizon + 1 if death < 0 else int(death)
                rows.append(dict(
                    eval_seed=int(eval_seed), lag_back=int(lag_back),
                    rho=spearman(cost, survival),
                    best_survival=int(survival[cheapest]),
                    ceiling=int(survival.max()),
                    fine_survival=fine,
                    actual_survival=int(lag_back - 1),
                    best_label=labels[cheapest],
                    plans=int(len(plans)),
                    macro_share=float(np.mean(survival > span)),
                    macro_mean=float(survival.mean()),
                    control=control,
                ))
    return rows


def summarize(rows):
    out = []
    for lag_back in sorted(set(row["lag_back"] for row in rows)):
        block = [row for row in rows if row["lag_back"] == lag_back]
        gains = [
            row["best_survival"] - row["actual_survival"] for row in block
        ]
        ceiling_gains = [
            row["ceiling"] - row["actual_survival"] for row in block
        ]
        out.append(dict(
            lag_back=int(lag_back),
            states=int(len(block)),
            mean_rho=float(np.mean([row["rho"] for row in block])),
            share_ranked=float(np.mean([row["rho"] <= -0.2 for row in block])),
            mean_best=float(np.mean([row["best_survival"] for row in block])),
            mean_ceiling=float(np.mean([row["ceiling"] for row in block])),
            mean_gain=float(np.mean(gains)),
            share_gain=float(np.mean([gain > 0 for gain in gains])),
            mean_ceiling_gain=float(np.mean(ceiling_gains)),
            share_ceiling_gain=float(np.mean(
                [gain > 0 for gain in ceiling_gains]
            )),
            macro_share=float(np.mean([row["macro_share"] for row in block])),
            control_share=float(np.mean([
                row["control"]["share"] for row in block
                if row["control"] is not None
            ])) if any(row["control"] is not None for row in block)
            else float("nan"),
        ))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-122")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--macro-frames", type=int, default=4)
    parser.add_argument("--macro-steps", type=int, default=3)
    parser.add_argument("--macro-count", type=int, choices=(4, 8), default=4)
    parser.add_argument("--control-plans", type=int, default=0)
    parser.add_argument("--min-lag", type=int, default=8)
    parser.add_argument("--max-lag", type=int, default=14)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    args.eval_seeds = parse_seed_spec(args.eval_seeds)

    rows = run(args)
    summary = summarize(rows)
    print("== coarse pass: seed %d, %d macros x %d steps x %d frames, "
          "%d leaves ==" % (
              args.seed, args.macro_count, args.macro_steps,
              args.macro_frames, rows[0]["plans"] if rows else 0,
          ))
    print("  k  states  rho  coarse-best  ceiling  gain-vs-actual  "
          "ceiling-gain  macro-survive  random-survive")
    for block in summary:
        print("  %2d  %6d  %5.2f  %11.2f  %7.2f  %14.2f  %12.2f  "
              "%13.0f%%  %14.0f%%" % (
                  block["lag_back"], block["states"], block["mean_rho"],
                  block["mean_best"], block["mean_ceiling"],
                  block["mean_gain"], block["mean_ceiling_gain"],
                  100.0 * block["macro_share"],
                  100.0 * block["control_share"],
              ))
    if summary:
        print("  rho correct (<=-0.2): %.0f%% | coarse beats actual: %.0f%% "
              "| ceiling beats actual: %.0f%%" % (
                  100.0 * np.mean([b["share_ranked"] for b in summary]),
                  100.0 * np.mean([b["share_gain"] for b in summary]),
                  100.0 * np.mean([b["share_ceiling_gain"] for b in summary]),
              ))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(dict(summary=summary, rows=rows), handle,
                      indent=2, sort_keys=True)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
