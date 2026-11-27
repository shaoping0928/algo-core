"""How many pipes does the shipped planner actually pass?

The milestone line's root-cause note claims the planner dies before the
first pipe (78 frames) and never recycles a pipe.  The death times in the
benchmark are all of the form 72 + 80k, which would instead mean pipe
passages.  This probe settles it on the default protocol: it fits the
planner exactly as the benchmark does, runs the evaluation episodes, and
replays the executed actions to read the world's own ``passed`` counter.

Reported per episode: frames, pipes passed, and the distance between the
death and the next expected pipe (78 + 80 * passed), so "fell short" and
"died at the pipe" can be told apart.

Read-only: no mechanism is changed.
"""

import argparse

import numpy as np

from latent_world import latent
from substrate_nodes import LAG, _curiosity_off, evaluate, fit

FIRST_PIPE = 78.0   # (1.0 - BIRD_X) / SPEED
SPACING = 80.0      # SPACING / SPEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)

    rows = []
    for train_seed in seeds:
        world, agent = fit(
            seed=train_seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon,
            step_stride=2,
        )
        with _curiosity_off(agent):
            for eval_seed in eval_seeds:
                env_seed = int(eval_seed) * 10000
                frames, actions, _ = evaluate(
                    train_seed, world, agent, cap=args.cap,
                    horizon=args.horizon, lag=LAG, step_stride=2,
                    env_seed=env_seed,
                )
                env = latent[LAG](env_seed)
                for action in actions:
                    if env.dead:
                        break
                    env.step(int(action))
                next_pipe = FIRST_PIPE + SPACING * env.passed
                rows.append(dict(
                    train=int(train_seed), eval=int(eval_seed),
                    frames=int(frames), passed=int(env.passed),
                    next_pipe=float(next_pipe),
                    died_before_next=bool(frames < next_pipe),
                ))
                print("  train %3d eval %3d: frames %4d  pipes passed %d  "
                      "next pipe %.0f  %s" % (
                          train_seed, eval_seed, frames, env.passed,
                          next_pipe,
                          "died before it" if frames < next_pipe
                          else "died at/after it",
                      ))
    passed = np.asarray([row["passed"] for row in rows])
    frames = np.asarray([row["frames"] for row in rows])
    print("== %d episodes ==" % len(rows))
    print("  frames mean %.1f  pipes passed mean %.2f  max %d" % (
        float(frames.mean()), float(passed.mean()), int(passed.max())
    ))
    print("  share of episodes passing >= 1 pipe: %.0f%%" % (
        100.0 * float(np.mean(passed >= 1))
    ))
    for train in sorted(set(row["train"] for row in rows)):
        block = [row for row in rows if row["train"] == train]
        print("  train %d: frames %.1f  pipes %.2f" % (
            train, float(np.mean([row["frames"] for row in block])),
            float(np.mean([row["passed"] for row in block])),
        ))


if __name__ == "__main__":
    main()
