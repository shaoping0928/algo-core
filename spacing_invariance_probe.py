"""Time-scale invariance of the decision policy (2026-11-22).

The claim under test: a passage is a **time-independent milestone**.  The
geometry the bird has to fly through is the same whether the pipes are close
together or far apart; only the number of frames between two passages moves.
A policy that reasons at the right level therefore should not care how far
apart the pipes are.

The probe scores a policy on held-out worlds at several pipe spacings, reporting
both the frame count and the progress-milestone count at each.  Two policies
are available:

* ``--policy learned`` (default) trains one model per seed on the default world
  and re-scores the **same** model at every spacing;
* ``--policy reference`` runs the recovered-dynamics lookahead controller
  (``latent_world.make_policy``) with no training at all.  A strong controller
  is the sensitive instrument: the learned baseline reaches only ~0.3
  milestones per episode, so its milestone signal is too sparse to resolve an
  invariance, while the reference controller passes dozens.

A frame-horizon planner is expected to fall apart as the spacing grows, because
its rollout can never reach the next passage.  A milestone-level decision (or a
controller that reasons per passage) is expected to hold: milestones per
passage stay put while frames per passage scale with the spacing.

    python spacing_invariance_probe.py --seeds 24 --jobs 6 --report si.json
    python spacing_invariance_probe.py --policy reference --eval-seeds 120-139

`--spacings` is a comma list; the first entry is the reference arm.  The
default world's spacing (0.80 = 80 frames between passages) is the first.

Nothing here trains on the evaluation worlds.  The learner never reads the
spacing parameter; it sees the world's ordinary observation channels, whose
values carry the (now larger) distance to the pipe after the next one.  The
milestone count is the world's own progress counter, reported out of band.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from eval_protocol import cluster_summary, paired_delta, parse_seed_spec
from flappy import SPEED as NOMINAL_SPEED
from substrate_nodes import evaluate, fit

DEFAULT_JOBS = min(8, os.cpu_count() or 1)


def _pass_rate(frames, passes, spacing):
    """Passages achieved per passage the world actually offered.

    A passage takes ``spacing / speed`` frames, so a fixed frame budget
    contains fewer of them as the spacing grows.  Raw milestone counts
    therefore fall with the spacing for any policy; dividing by the number of
    passages *offered* over the episode is the spacing-free competence.
    """
    offered = float(frames) * NOMINAL_SPEED / float(spacing)
    if offered <= 0.0:
        return 0.0
    return float(passes) / offered


def _evaluate_spacings(seed, args):
    """Train one model, then score it at every spacing on held-out worlds."""
    world, agent = fit(
        seed, steps=args.steps, epochs=args.epochs,
        max_nodes=args.max_nodes, lag=args.lag,
    )
    rows = []
    for spacing in args.spacings:
        for eval_seed in args.eval_seeds:
            for episode in range(args.eval_episodes):
                env_seed = int(eval_seed) * 10000 + episode
                metrics = {}
                frames, actions, _ = evaluate(
                    seed, world, agent, cap=args.cap, horizon=args.horizon,
                    lag=args.lag, env_seed=env_seed, decision=args.decision,
                    metrics_out=metrics, spacing=spacing,
                )
                rows.append(dict(
                    train_seed=int(seed),
                    spacing=float(spacing),
                    eval_seed=int(eval_seed),
                    episode=int(episode),
                    env_seed=int(env_seed),
                    frames=int(frames),
                    passes=int(metrics.get("passes", 0)),
                    pass_rate=_pass_rate(
                        frames, metrics.get("passes", 0), spacing
                    ),
                    action_rate=(
                        float(np.mean(actions)) if actions else 0.0
                    ),
                ))
    return rows


def _reference_rows(args):
    """Run the recovered-dynamics lookahead controller at every spacing.

    No training: the controller's rules are identified once from a separate
    behaviour stream, then replayed on each evaluation world and spacing.  It
    is the sensitive instrument for the invariance question because it passes
    many milestones per episode.
    """
    from latent_world import latent, make_policy, recover

    recovered = recover(0)
    act, target = make_policy(recovered, 0, args.reference_horizon)
    rows = []
    for eval_seed in args.eval_seeds:
        for episode in range(args.eval_episodes):
            env_seed = int(eval_seed) * 10000 + episode
            for spacing in args.spacings:
                env = latent[0](env_seed, spacing=spacing)
                env.max_t = args.cap
                actions = []
                while not env.dead:
                    obs = env.obs()
                    action = act(
                        float(obs[0]) + 0.5, float(obs[1]) / 10.0,
                        target(env), [], 0,
                    )
                    actions.append(action)
                    env.step(action)
                rows.append(dict(
                    train_seed=int(eval_seed),
                    spacing=float(spacing),
                    eval_seed=int(eval_seed),
                    episode=int(episode),
                    env_seed=int(env_seed),
                    frames=int(env.t),
                    passes=int(env.passed),
                    pass_rate=_pass_rate(env.t, env.passed, spacing),
                    action_rate=(
                        float(np.mean(actions)) if actions else 0.0
                    ),
                ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure the time-scale invariance of a decision policy"
    )
    parser.add_argument("--seed", type=int, default=100,
                        help="first training seed")
    parser.add_argument("--seeds", type=int, default=24,
                        help="number of training seeds")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="concurrent training seeds")
    parser.add_argument("--steps", type=int, default=2000,
                        help="offline training steps per seed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=6,
                        help="search horizon (the frame-scoped decision depth)")
    parser.add_argument("--decision", default="search",
                        help="search | stage | cellq")
    parser.add_argument("--policy", default="learned",
                        choices=("learned", "reference"),
                        help="learned = train the node runtime; reference = "
                             "the recovered-dynamics lookahead controller")
    parser.add_argument("--reference-horizon", type=int, default=10,
                        help="lookahead of the reference controller")
    parser.add_argument("--cap", type=int, default=600,
                        help="frame cap per evaluation episode")
    parser.add_argument("--eval-seeds", default="200-203")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--spacings", default="0.8,1.6,2.4",
                        help="comma list of evaluation pipe spacings; the "
                             "first is the reference arm")
    parser.add_argument("--report", default="",
                        help="write the rows and summaries as JSON")
    args = parser.parse_args(argv)

    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.cap < 1:
        raise ValueError("--cap must be positive")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be positive")
    args.eval_seeds = parse_seed_spec(args.eval_seeds)
    args.spacings = [
        float(part) for part in str(args.spacings).split(",") if part.strip()
    ]
    if not args.spacings:
        raise ValueError("--spacings must name at least one spacing")
    if any(value <= 0.0 for value in args.spacings):
        raise ValueError("--spacings must all be positive")
    if len(set(args.spacings)) != len(args.spacings):
        raise ValueError("--spacings must be distinct")

    seeds = list(range(args.seed, args.seed + args.seeds))
    if args.policy == "reference":
        rows = _reference_rows(args)
    else:
        tasks = [(seed, args) for seed in seeds]
        rows = []
        if args.jobs == 1:
            for seed in seeds:
                rows.extend(_evaluate_spacings(seed, args))
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for chunk in pool.map(_pickle_helper, tasks):
                    rows.extend(chunk)
    rows.sort(key=lambda row: (row["train_seed"], row["spacing"],
                               row["eval_seed"], row["episode"]))

    by_spacing = {}
    for row in rows:
        by_spacing.setdefault(row["spacing"], []).append(row)

    reference = args.spacings[0]
    print("\nspacing  frames  (mean of all evaluation episodes)")
    summaries = {}
    for spacing in args.spacings:
        subset = by_spacing[spacing]
        frame_summary = cluster_summary(
            subset, value="frames", cluster="train_seed"
        )
        pass_summary = cluster_summary(
            subset, value="passes", cluster="train_seed"
        )
        rate_summary = cluster_summary(
            subset, value="pass_rate", cluster="train_seed"
        )
        summaries[str(spacing)] = dict(
            frames=frame_summary, passes=pass_summary, pass_rate=rate_summary
        )
        print("  %5.2f  frames mean %7.2f median %7.1f  | milestones mean %6.2f"
              "  | passages achieved / offered %.3f  CI [%.3f, %.3f]"
              % (spacing, frame_summary["mean"], frame_summary["median"],
                 pass_summary["mean"], rate_summary["mean"],
                 rate_summary["cluster_ci_low"],
                 rate_summary["cluster_ci_high"]))

    print("\npaired delta vs spacing %.2f (per training seed, left - right)"
          % reference)
    deltas = {}
    reference_rows = by_spacing[reference]
    for spacing in args.spacings[1:]:
        subset = by_spacing[spacing]
        frame_delta = paired_delta(
            subset, reference_rows, key="train_seed", value="frames"
        )
        pass_delta = paired_delta(
            subset, reference_rows, key="train_seed", value="passes"
        )
        rate_delta = paired_delta(
            subset, reference_rows, key="train_seed", value="pass_rate"
        )
        deltas[str(spacing)] = dict(
            frames=frame_delta, passes=pass_delta, pass_rate=rate_delta
        )
        print("  %5.2f  frames %+8.2f CI [%+.2f, %+.2f]  "
              "| milestones %+6.2f  | per-offered rate %+.3f CI [%+.3f, %+.3f]"
              % (spacing, frame_delta["mean"],
                 frame_delta["summary"]["ci_low"],
                 frame_delta["summary"]["ci_high"], pass_delta["mean"],
                 rate_delta["mean"], rate_delta["summary"]["ci_low"],
                 rate_delta["summary"]["ci_high"]))

    if args.report:
        report = dict(
            config=dict(
                seed=args.seed, seeds=args.seeds, steps=args.steps,
                epochs=args.epochs, max_nodes=args.max_nodes, lag=args.lag,
                horizon=args.horizon, decision=args.decision, cap=args.cap,
                eval_seeds=args.eval_seeds, eval_episodes=args.eval_episodes,
                spacings=args.spacings, reference=reference,
            ),
            rows=rows,
            per_spacing=summaries,
            paired_vs_reference=deltas,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("wrote %s" % args.report)
    return rows


def _pickle_helper(task):
    """Unpack a (seed, args) pair inside the worker process."""
    seed, args = task
    return _evaluate_spacings(seed, args)


if __name__ == "__main__":
    main()
