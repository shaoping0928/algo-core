"""A/B the two hazard parameters kept separate: realised pain vs potential risk.

The terminal term is `max(pain, risk)` today.  The two are different
quantities - the state's realised outcome statistic and the calibrated
probability of reaching pain inside a window - so this probe compares:

* ``max``      - the historical composition (one shared weight);
* ``pain``     - the realised-outcome parameter alone;
* ``both``     - realised outcome plus a small weight on the potential risk.

Everything else is held fixed (same fitted model, same evaluation seeds).
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--risk-weight", type=float, default=0.1)
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)

    configs = [
        ("max", dict(risk_lane="max", risk_weight=1.0)),
        ("pain", dict(risk_lane="separate", pain_weight=1.0,
                      risk_weight=0.0)),
        ("both", dict(risk_lane="separate", pain_weight=1.0,
                      risk_weight=args.risk_weight)),
    ]
    means = {}
    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon,
            step_stride=2,
        )
        with _curiosity_off(agent):
            record = {}
            for name, kwargs in configs:
                lengths = []
                for eval_seed in eval_seeds:
                    length = evaluate(
                        seed, world, agent, cap=5000, horizon=args.horizon,
                        lag=LAG, step_stride=2,
                        env_seed=int(eval_seed) * 10000, **kwargs,
                    )[0]
                    lengths.append(length)
                record[name] = lengths
            print("== seed %d ==" % seed)
            for name, lengths in record.items():
                print("  %-5s mean %6.1f  runs %s" % (
                    name, np.mean(lengths), lengths,
                ))
            means[seed] = {name: np.mean(values)
                           for name, values in record.items()}
    if means:
        print("== means over seeds ==")
        for name in [config[0] for config in configs]:
            values = [means[seed][name] for seed in means]
            print("  %-5s %6.1f  per-seed %s" % (
                name, np.mean(values), [round(value, 1) for value in values],
            ))


if __name__ == "__main__":
    main()
