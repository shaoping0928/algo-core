"""Diagnostic for predictive-equivalence classes over realized options.

Fits the fast runtime with the stage layer, then compares the exhaustive
stage-level search over exact traces with the same search over accepted option
classes.  Nothing here changes a decision: the search is offline over realized
validation stages, and no planner path consumes it.
"""
import argparse
import json

import numpy as np

from predictive_option_quotient import OptionQuotient
from stage_option_search import (
    StageOptionSearch,
    distribution_quantile,
)
from substrate_nodes import fit


def summarize(results, horizon):
    if not results:
        return dict(searches=0)
    leaves = sum(len(result.leaves) for result in results)
    terminal = sum(
        1 for result in results
        if any(leaf.outcome == "terminal" for leaf in result.leaves)
    )
    unresolved = sum(len(result.unresolved) for result in results)
    q90 = []
    within = []
    for result in results:
        mass = sum(leaf.probability for leaf in result.leaves)
        if mass <= 0.0:
            continue
        aggregate = {}
        for leaf in result.leaves:
            for time, probability in leaf.elapsed_distribution:
                aggregate[time] = (
                    aggregate.get(time, 0.0)
                    + float(probability) * float(leaf.probability)
                )
        distribution = tuple(sorted(
            (time, probability / mass)
            for time, probability in aggregate.items()
        ))
        q90.append(distribution_quantile(distribution, 0.9))
        within.append(sum(
            probability for time, probability in distribution
            if time <= float(horizon)
        ))
    return dict(
        searches=len(results),
        leaves=int(leaves),
        mean_leaves=float(leaves) / len(results),
        mean_plans=float(np.mean([
            len(result.plans()) for result in results
        ])),
        terminal_reachable=int(terminal),
        unresolved=int(unresolved),
        elapsed_q90=float(np.mean(q90)) if q90 else 0.0,
        within_horizon=float(np.mean(within)) if within else 0.0,
        max_leaves=int(max(len(result.leaves) for result in results)),
        incomplete=int(sum(1 for result in results if not result.complete)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-leaves", type=int, default=20000)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument(
        "--tolerances", type=float, nargs="+",
        default=[0.0, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument("--option-min-support", type=int, default=1)
    parser.add_argument("--option-min-episodes", type=int, default=1)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    _world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=True, stage_memory_feature_source="prototype",
        option_tolerance=0.05,
        option_min_support=int(args.option_min_support),
        option_min_episodes=int(args.option_min_episodes),
    )
    memory = getattr(agent, "stage_outcome_memory", None)
    if memory is None:
        raise SystemExit("the stage layer did not fit")
    print(
        "option quotient probe: seed %d | memory %s"
        % (args.seed, memory.snapshot())
    )

    episodes = list(agent.stage_experience.episodes)
    train = [
        episode for index, episode in enumerate(episodes)
        if index % 5 != 4
    ]
    heldout = [
        episode for index, episode in enumerate(episodes)
        if index % 5 == 4
    ]
    if not train or not heldout:
        raise SystemExit("not enough episodes for an option-quotient split")

    sweep = []
    chosen = None
    for tolerance in args.tolerances:
        quotient = OptionQuotient.fit(
            train, heldout, prior=4.0, tolerance=float(tolerance),
            min_support=int(args.option_min_support),
            min_episodes=int(args.option_min_episodes),
        )
        snapshot = quotient.snapshot()
        snapshot["tolerance"] = float(tolerance)
        sweep.append(snapshot)
        if quotient.accepted and (
                chosen is None
                or len(quotient.classes) < len(chosen.classes)):
            chosen = quotient
        print("  tolerance %.4f: %s" % (tolerance, snapshot))
    if chosen is None:
        chosen = OptionQuotient.fit(
            train, heldout, prior=4.0, tolerance=float(args.tolerances[0]),
            min_support=int(args.option_min_support),
            min_episodes=int(args.option_min_episodes),
        )
    per_stage = []
    for stage_key in sorted(chosen.by_stage, key=repr):
        traces = sum(
            1 for key in chosen.class_of if key[0] == stage_key
        )
        classes = len(chosen.by_stage[stage_key])
        per_stage.append(dict(
            stage=repr(stage_key), traces=int(traces), classes=int(classes),
        ))
    per_stage.sort(key=lambda row: (-row["traces"], row["stage"]))
    print("  chosen: %s" % (chosen.snapshot(),))
    print("  per-stage reduction (top 8): %s" % (per_stage[:8],))

    validation = heldout[:max(int(args.validation_limit), 1)]

    trace_search = StageOptionSearch(
        memory, max_transitions=int(args.depth),
        max_leaves=int(args.max_leaves), allow_truncation=True,
    )
    trace_results = []
    for episode in validation:
        stages = list(episode)
        for index in range(len(stages)):
            history = tuple(stage.key for stage in stages[:index + 1])
            trace_results.append(trace_search.search(history))

    print("  trace mode: %s" % (summarize(trace_results, args.horizon),))
    report = dict(
        seed=int(args.seed),
        config=dict(
            depth=int(args.depth), horizon=int(args.horizon),
            max_leaves=int(args.max_leaves),
            validation=int(len(validation)), steps=int(args.steps),
            epochs=int(args.epochs), max_nodes=int(args.max_nodes),
            tolerances=[float(value) for value in args.tolerances],
            option_min_support=int(args.option_min_support),
            option_min_episodes=int(args.option_min_episodes),
        ),
        memory=memory.snapshot(),
        sweep=sweep,
        chosen=chosen.snapshot(),
        per_stage=per_stage,
        trace_mode=summarize(trace_results, args.horizon),
        class_mode=None,
        class_mode_note=None,
    )
    if not chosen.accepted:
        report["class_mode_note"] = (
            "every tested tolerance was rejected by the held-out gate; "
            "class mode not run"
        )
        print("  class mode: skipped (%s)" % report["class_mode_note"])
    else:
        class_search = StageOptionSearch(
            memory, max_transitions=int(args.depth),
            max_leaves=int(args.max_leaves),
            option_quotient=chosen, allow_truncation=True,
        )
        class_results = []
        for episode in validation:
            stages = list(episode)
            for index in range(len(stages)):
                history = tuple(stage.key for stage in stages[:index + 1])
                class_results.append(class_search.search(history))
        report["class_mode"] = summarize(class_results, args.horizon)
        report["class_mode"]["option_vocabulary"] = int(
            class_results[0].counts.get("option_vocabulary", 0)
        ) if class_results else 0
        print("  class mode: %s" % (report["class_mode"],))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
