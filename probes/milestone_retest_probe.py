"""Re-test the milestone layer after the identity/memory fixes.

For each training seed: fit with the benchmark protocol (the model is built
from the training stream), then run the evaluation episodes and replay them
through the *causal* tracker with the frozen threshold/scale/frame.

Reported per episode: frames, pipes passed (the world's own counter),
milestones opened, distinct identities visited, and coverage of the
decision-time identities against the learned key set.  Also reported once
per model: bin count, axis-time correlation, and the learned key set.

Read-only.
"""

import argparse

import numpy as np

from latent_world import latent
from milestone_graph import MilestoneTracker, milestone_coverage
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,102")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stage",
                        choices=("strength", "signature", "geometry"),
                        default="strength")
    parser.add_argument("--boundary-k", type=float, default=3.0)
    parser.add_argument("--no-eval", action="store_true",
                        help="only build/measure the stage layer")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)

    for train_seed in seeds:
        world, agent = fit(
            seed=train_seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon,
            step_stride=2, milestone_stage=args.stage,
            milestone_boundary_k=args.boundary_k,
        )
        frame = agent.milestone_frame
        print("== train %d: bins %s  time_corr %s (pooled %s)  keys %s  "
              "goals %s ==" % (
                  train_seed,
                  None if frame is None else frame.get("bins"),
                  None if frame is None
                  else round(float(frame.get("time_corr", 0)), 3),
                  None if frame is None
                  else round(float(frame.get("time_corr_pooled", 0)), 3),
                  len(agent.milestone_keys or []),
                  None if agent.milestone_goals is None
                  else len(agent.milestone_goals),
              ))
        if frame is None:
            print("  no milestone frame (degenerate stream)")
            continue
        print("  clock dims %s  axis %s" % (
            np.flatnonzero(np.asarray(frame.get("time_dir", []),
                                      dtype=float)).tolist(),
            np.round(np.asarray(frame.get("axis", [])), 2).tolist(),
        ))
        stats = getattr(agent, "milestone_stages", {}) or {}
        print("  stage %s: boundaries %s  stages %s  identities %s  "
              "dwell %.1f  sources %s  branch %.2f  branching %s  "
              "eta2 %.2f" % (
                  args.stage, stats.get("boundaries"), stats.get("stages"),
                  stats.get("identities"),
                  float(stats.get("dwell", 0.0)), stats.get("sources"),
                  float(stats.get("branch", 0.0)), stats.get("branching"),
                  float(stats.get("feedback_eta2", 0.0)),
              ))
        clusters = getattr(agent, "milestone_clusters", None)
        if clusters:
            print("  clusters %s  supports %s" % (
                clusters.get("k"), clusters.get("supports"),
            ))
        if args.no_eval:
            continue
        tracker = MilestoneTracker(
            agent.milestone_threshold, frame, scale=agent.milestone_scale
        )
        with _curiosity_off(agent):
            for eval_seed in eval_seeds:
                env_seed = int(eval_seed) * 10000
                frames, actions, transitions = evaluate(
                    train_seed, world, agent, cap=args.cap,
                    horizon=args.horizon, lag=LAG, step_stride=2,
                    env_seed=env_seed, record=True,
                )
                keys = []
                opened = 0
                for row in transitions:
                    if tracker.observe(row["obs"]):
                        opened += 1
                    keys.append(tracker.key)
                env = latent[LAG](env_seed)
                for action in actions:
                    if env.dead:
                        break
                    env.step(int(action))
                coverage = milestone_coverage(keys, agent.milestone_keys or ())
                print("  eval %3d: frames %4d  pipes %d  opened %2d  "
                      "keys %2d  coverage %.2f" % (
                          eval_seed, frames, env.passed, opened,
                          len({key for key in keys if key is not None}),
                          coverage["fraction"],
                      ))


if __name__ == "__main__":
    main()
