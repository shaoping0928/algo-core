"""How many frames of warning does each learned channel actually give?

The planner searches 8 frames ahead, so a channel that only rises inside
that window cannot change the choice that caused the death.  This probe
replays real death episodes and reads every channel at each recorded
transition, bucketed by the true distance to death.

Reported per bucket ``k`` (frames before the fatal transition):

* ``composed`` - the stacked hazard the planner sums;
* ``prototype`` / ``feature`` - the two hazard channels;
* ``value`` - the feature value channel (long-run return read);
* ``mc`` - the prototype MC pain value the endpoint uses;
* ``dopamine`` - the positive-valence channel.

A channel only helps a search with horizon H if it separates the fatal
neighbourhood at k > H.  Nothing is learned here.
"""

import argparse
import json

import numpy as np

from substrate_nodes import LAG, _curiosity_off, _encoded_pending, evaluate, fit


def channels_at(agent, encoded):
    mask_table = agent._feature_mask_table(encoded[:, :agent.obs_dim])
    channels = agent._hazard_channels(encoded, mask_table)
    value, support = agent.feature_value_batch(
        encoded, mask_table, return_support=True
    )
    mc, dopamine = agent.valence_state_batch(encoded)
    channels["composed"] = np.asarray(
        agent.pain_signal_batch(encoded, mask_table), dtype=float
    )
    channels["value"] = np.asarray(value, dtype=float)
    channels["value_support"] = np.asarray(support, dtype=float)
    channels["mc"] = np.asarray(mc, dtype=float)
    channels["dopamine"] = np.asarray(dopamine, dtype=float)
    return channels


def run(args):
    world, agent = fit(
        seed=args.seed, steps=1500, epochs=args.epochs, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
        feature_backfit=args.feature_backfit,
    )
    buckets = {}
    with _curiosity_off(agent):
        for eval_seed in args.eval_seeds:
            env_seed = int(eval_seed) * 10000
            frames, _actions, transitions = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=args.horizon,
                lag=LAG, step_stride=2, env_seed=env_seed, record=True,
            )
            if not transitions:
                continue
            observation = np.asarray(
                [row["obs"] for row in transitions], dtype=float
            )
            pending = _encoded_pending(transitions, LAG)
            encoded = np.concatenate(
                [observation, pending], axis=1
            ) / agent.scales
            reads = channels_at(agent, encoded)
            died = bool(transitions[-1]["done"]) or frames < args.cap
            if not died:
                continue
            for index in range(len(transitions)):
                distance = len(transitions) - 1 - index
                if distance > args.max_lead:
                    continue
                entry = buckets.setdefault(distance, {})
                for name, values in reads.items():
                    entry.setdefault(name, []).append(float(values[index]))
    return buckets


def summarize(buckets):
    summary = {}
    for distance in sorted(buckets):
        summary[distance] = {
            name: float(np.mean(values))
            for name, values in buckets[distance].items()
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--feature-backfit", action="store_true")
    parser.add_argument("--max-lead", type=int, default=20)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    args.eval_seeds = parse_seed_spec(args.eval_seeds)

    summary = summarize(run(args))
    names = ["composed", "prototype", "feature", "value", "mc", "dopamine"]
    print("== warning lead: seed %d, horizon %d ==" % (
        args.seed, args.horizon
    ))
    print("  k  " + "  ".join("%10s" % name for name in names))
    for distance in sorted(summary):
        row = summary[distance]
        print("%3d  " % distance + "  ".join(
            "%10.4f" % row.get(name, float("nan")) for name in names
        ))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
