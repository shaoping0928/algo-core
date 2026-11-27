"""A/B the stage-layer plan term against the default planner.

The term is inert at weight zero (bitwise), so the baseline is the same model
with ``milestone_mode="off"``.  For each training seed the probe reports the
episode lengths under the default planner and under the endpoint-stage risk
term at several weights.

Read-only; nothing is written to disk.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--weights", default="0.1,0.3")
    parser.add_argument("--stage", default="strength")
    parser.add_argument("--boundary-k", type=float, default=3.0)
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)
    weights = [float(value) for value in args.weights.split(",")]

    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon,
            step_stride=2, milestone_stage=args.stage,
            milestone_boundary_k=args.boundary_k,
        )
        stats = getattr(agent, "milestone_stages", {}) or {}
        print("== train %d: stages %s identities %s sources %s branch %.2f "
              "eta2 %.2f ==" % (
                  seed, stats.get("stages"), stats.get("identities"),
                  stats.get("sources"), float(stats.get("branch", 0.0)),
                  float(stats.get("feedback_eta2", 0.0)),
              ))
        with _curiosity_off(agent):
            rows = []
            for eval_seed in eval_seeds:
                env_seed = int(eval_seed) * 10000
                kw = dict(cap=args.cap, horizon=args.horizon, lag=LAG,
                          step_stride=2, env_seed=env_seed)
                base = evaluate(seed, world, agent, **kw)[0]
                line = [("base", base)]
                for weight in weights:
                    length = evaluate(
                        seed, world, agent, milestone_weight=weight,
                        milestone_mode="plan", **kw,
                    )[0]
                    line.append(("w%.2f" % weight, length))
                print("  eval %3d: %s" % (
                    eval_seed,
                    "  ".join("%s %d" % pair for pair in line),
                ))
                rows.append([value for _label, value in line])
            if rows:
                means = np.asarray(rows, dtype=float).mean(axis=0)
                labels = ["base"] + ["w%.2f" % weight for weight in weights]
                print("  mean: %s" % "  ".join(
                    "%s %.1f" % (label, value)
                    for label, value in zip(labels, means)
                ))


if __name__ == "__main__":
    main()
