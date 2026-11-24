"""Diagnostic for the exhaustive stage-level (non-frame) option search.

Two measurements, neither of which changes a decision:

* offline: for every realized stage in the fitted experience, enumerate the
  complete option tree at the requested depth and report breadth, depth,
  terminal reachability and unresolved mass;
* online: at real decision points, build the causal stage context with the
  online tracker and run the same enumeration.  The probe returns ``None`` to
  the planner, so the trajectory and every action stay on the baseline path.
"""
import argparse
import json

import numpy as np

from stage_online import OnlineStageTracker
from stage_option_search import (
    StageOptionSearch,
    distribution_quantile,
)
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def summarize(results, horizon=None):
    if not results:
        return dict(searches=0)
    counts = {
        key: int(sum(result.counts.get(key, 0) for result in results))
        for key in (
            "queries", "supported_options", "unknown_options",
            "backoff_refused", "nodes", "unique_nodes", "edges", "leaves",
            "leaf_terminal", "leaf_censored", "leaf_open", "unresolved",
            "elapsed_convolutions",
        )
    }
    counts["max_elapsed_support_seen"] = max(
        (int(result.counts.get("max_elapsed_support_seen", 0))
         for result in results),
        default=0,
    )
    terminal_plans = sum(
        1 for result in results
        if any(leaf.outcome == "terminal" for leaf in result.leaves)
    )
    dead_end = sum(
        1 for result in results
        if result.leaves == () and result.unresolved
    )
    per_search_elapsed = []
    per_search_terminal = []
    per_search_q90 = []
    per_search_within = []
    for result in results:
        mass = sum(leaf.probability for leaf in result.leaves)
        per_search_terminal.append(sum(
            leaf.probability for leaf in result.leaves
            if leaf.outcome == "terminal"
        ))
        aggregated = {}
        for leaf in result.leaves:
            for time, probability in leaf.elapsed_distribution:
                aggregated[time] = (
                    aggregated.get(time, 0.0)
                    + float(probability) * float(leaf.probability)
                )
        if mass > 0.0:
            per_search_elapsed.append(sum(
                leaf.probability * leaf.elapsed_mean
                for leaf in result.leaves
            ) / mass)
            elapsed = tuple(sorted(
                (time, probability / mass)
                for time, probability in aggregated.items()
            ))
            per_search_q90.append(distribution_quantile(elapsed, 0.9))
            if horizon is not None:
                per_search_within.append(sum(
                    probability for time, probability in elapsed
                    if time <= float(horizon)
                ))
        else:
            per_search_elapsed.append(0.0)
            per_search_q90.append(0.0)
            if horizon is not None:
                per_search_within.append(0.0)
    nodes = max(counts["unique_nodes"], 1)
    return dict(
        searches=len(results),
        counts=counts,
        incomplete=sum(1 for result in results if not result.complete),
        max_leaves=max((len(result.leaves) for result in results), default=0),
        max_plans=max((len(result.plans()) for result in results), default=0),
        supported_per_node=float(counts["supported_options"]) / nodes,
        unknown_per_node=float(counts["unknown_options"]) / nodes,
        terminal_reachable=terminal_plans,
        dead_end_searches=dead_end,
        terminal_mass_sum_mean=float(np.mean(per_search_terminal)),
        elapsed_mean=float(np.mean(per_search_elapsed)),
        elapsed_q90=float(np.mean(per_search_q90)),
        within_horizon=(
            None if horizon is None
            else float(np.mean(per_search_within))
        ),
        unresolved_mass=float(np.mean([
            result.unresolved_mass for result in results
        ])),
    )


class ProbeProvider:
    """Run the search at decision time and never influence the decision."""

    def __init__(self, agent, depth, max_decisions, max_leaves):
        self.tracker = OnlineStageTracker(
            agent.stage_experience.detector, agent.lag,
        )
        self.search = StageOptionSearch(
            agent.stage_outcome_memory, max_transitions=depth,
            max_leaves=max_leaves, allow_truncation=True,
        )
        self.depth = int(depth)
        self.max_decisions = int(max_decisions)
        self.results = []
        self.no_context = 0
        self.skipped = 0
        self.errors = 0

    def reset_episode(self, *_args, **_kwargs):
        self.tracker.reset()
        return self

    def observe_transition(self, transition):
        self.tracker.observe_transition(transition)

    def __call__(self, agent, _observation, _prefix, _actions):
        if len(self.results) >= self.max_decisions:
            self.skipped += 1
            return None
        context = self.tracker.context()
        if context is None:
            self.no_context += 1
            return None
        try:
            self.results.append(
                self.search.search(context["history"])
            )
        except (ValueError, RuntimeError):
            self.errors += 1
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[120])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--online-depth", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--max-leaves", type=int, default=20000)
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--offline-episodes", type=int, default=12)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
    )
    memory = getattr(agent, "stage_outcome_memory", None)
    if memory is None:
        raise SystemExit("stage memory was not fitted")
    experience = agent.stage_experience
    print(
        "stage option search probe: seed %d | memory %s"
        % (args.seed, memory.snapshot())
    )

    search = StageOptionSearch(
        memory, max_transitions=int(args.depth),
        max_leaves=int(args.max_leaves), allow_truncation=True,
    )
    offline = []
    for episode in list(experience.episodes)[:max(
            int(args.offline_episodes), 1)]:
        stages = list(episode)
        for index in range(len(stages)):
            history = tuple(stage.key for stage in stages[:index + 1])
            try:
                offline.append(search.search(history))
            except ValueError:
                continue
    print("  offline: %s" % (summarize(offline, horizon=args.horizon),))

    provider = ProbeProvider(
        agent, args.online_depth, args.max_decisions, args.max_leaves,
    )
    frames = []
    for eval_seed in args.eval_seeds:
        for episode in range(int(args.episodes)):
            env_seed = int(eval_seed) * 10000 + episode
            with _curiosity_off(agent):
                length, _actions, _rows = evaluate(
                    args.seed, world, agent, cap=int(args.cap),
                    horizon=int(args.horizon), lag=LAG,
                    step_stride=2, env_seed=env_seed,
                    decision="search", beam_width=int(args.beam_width),
                    stage_demand_provider=provider,
                )
            frames.append(int(length))
    print("  baseline frames %s" % (frames,))
    print(
        "  online: searches %d (no_context %d, errors %d, skipped %d)"
        % (len(provider.results), provider.no_context,
           provider.errors, provider.skipped)
    )
    online = summarize(provider.results, horizon=args.horizon)
    print("  online summary: %s" % (online,))
    if args.report:
        report = dict(
            seed=int(args.seed),
            config=dict(
                depth=int(args.depth), online_depth=int(args.online_depth),
                max_leaves=int(args.max_leaves),
                eval_seeds=[int(value) for value in args.eval_seeds],
                episodes=int(args.episodes), steps=int(args.steps),
                epochs=int(args.epochs), max_nodes=int(args.max_nodes),
            ),
            memory=memory.snapshot(),
            option_vocabulary=len(memory.action_options),
            offline=offline_summary(offline, horizon=args.horizon),
            online=online,
            frames=frames,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


def offline_summary(results, horizon=None):
    summary = summarize(results, horizon=horizon)
    summary["searches"] = len(results)
    return summary


if __name__ == "__main__":
    main()
