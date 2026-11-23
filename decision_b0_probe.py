"""B0 gate: diagnose the decision-relevant misses before building a router.

This is a read-only probe.  It changes no default path and fits no coarse
outcome model.  It replays the exact ``planner_rollout_quality`` protocol
(seed, eval worlds, decision points, horizon, death-status callback), keeps
the candidates the planner actually evaluated, and for each decision-relevant
miss records:

* the fine first-action gap and plan margin;
* whether the oracle-best plan was feasible, its score gap and rank;
* whether the selected and oracle-best first actions differ;
* local action support in the training stream (rows and episodes for both
  actions), plus how many near states carry the same pending-action window;
* candidate-endpoint OOD rate against the training state distribution.

The point of B0 is not to score a router.  It decides which problem the next
layer must solve: ranking inside the fine near-tie band, coverage (the
oracle-best action has no support or its endpoint is OOD), or the fine score
itself (the miss is large or the first action is already correct).

No benchmark vocabulary or task geometry enters the calculations.  Support
and OOD are measured in the agent's own standardized observation space, and
the action labels come from the recorded stream.

Usage::

    python decision_b0_probe.py --seed 101 --eval-seeds 120-124 \
        --cache b0-101.pkl --report b0-101.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from death_transition_quality import seed_spec
from latent_world import latent
from planner_rollout_quality import (
    action_best, action_score_gap, flatten_batches, oracle_best_details,
    oracle_plan, plan_score_gap, probe_config, status_callback,
)
from probe_cache import load_or_fit
from substrate_nodes import LAG, _curiosity_off, evaluate, policy_result
from trajectory_search import best_candidate_index


def training_index(rows, obs_dim, lag, scales, neighbours=64):
    """Standardized training states and the action labels read from them.

    ``history[0]`` is the action issued at that state; the pending window is
    ``history[1:lag+1]`` (newest first), matching the runtime encoder.  The
    support question is local: how many nearby states contain each action,
    and how often the same pending window is present.
    """
    states = np.asarray(
        [np.asarray(row["obs"], dtype=float)[:obs_dim] for row in rows],
        dtype=np.float64,
    )
    scale = np.asarray(scales[:obs_dim], dtype=float)
    scale = np.where(scale > 1e-8, scale, 1.0)
    states = states / scale
    actions = np.asarray([
        int(np.asarray(row["history"], dtype=float).reshape(-1)[0])
        for row in rows
    ], dtype=int)
    acting = np.zeros(len(rows), dtype=int)
    for index, row in enumerate(rows):
        history = np.asarray(row["history"], dtype=float).reshape(-1)
        position = min(int(lag), max(history.size - 1, 0))
        acting[index] = int(history[position]) if history.size else 0
    pending = np.zeros((len(rows), int(lag)), dtype=int)
    for index, row in enumerate(rows):
        history = np.asarray(row["history"], dtype=float).reshape(-1)
        take = min(int(lag), max(history.size - 1, 0))
        if take:
            pending[index, :take] = history[1:1 + take].astype(int)
    episodes = np.asarray([
        int(row.get("episode", 0)) for row in rows
    ], dtype=int)
    return dict(
        states=states, actions=actions, acting=acting, pending=pending,
        episodes=episodes, neighbours=int(neighbours), scale=scale,
    )


def _self_neighbour_distances(states, chunk=128):
    """Distance from each training state to its nearest other training state."""
    states = np.asarray(states, dtype=np.float32)
    count = len(states)
    result = np.empty(count, dtype=np.float32)
    for start in range(0, count, chunk):
        block = states[start:start + chunk]
        difference = block[:, None, :] - states[None, :, :]
        distance = np.sqrt(
            np.einsum("ijk,ijk->ij", difference, difference)
        )
        rows = np.arange(len(block))
        distance[rows, rows + start] = np.inf
        result[start:start + len(block)] = distance.min(axis=1)
    return result.astype(float)


def _nearest_distances(states, queries, chunk=256):
    """Minimum standardized distance from each query to any training state."""
    states = np.asarray(states, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    result = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        difference = block[:, None, :] - states[None, :, :]
        distance = np.sqrt(
            np.einsum("ijk,ijk->ij", difference, difference)
        )
        result[start:start + len(block)] = distance.min(axis=1)
    return result.astype(float)


def ood_reference(index, quantile=0.95, high_quantile=0.99):
    """Held-out density reference: leave-one-episode-out nearest distances.

    Self-neighbour distances inside one episode are not a density reference:
    consecutive frames are near duplicates and would collapse the threshold
    to zero.  Measuring each episode against the other episodes asks the
    question the endpoint model actually faces: is this state represented by
    a *different* episode in the training stream?
    """
    states = index["states"]
    episodes = index["episodes"]
    distances = np.empty(len(states), dtype=float)
    for episode in np.unique(episodes):
        query = episodes == episode
        train = ~query
        if not train.any():
            distances[query] = 0.0
            continue
        distances[query] = _nearest_distances(
            states[train], states[query]
        )
    threshold = float(np.quantile(distances, float(quantile)))
    high = float(np.quantile(distances, float(high_quantile)))
    return dict(
        distances=distances,
        threshold=threshold,
        high=high,
        p50=float(np.quantile(distances, 0.5)),
        p95=float(np.quantile(distances, 0.95)),
        p99=float(np.quantile(distances, 0.99)),
    )


def percentile_of(value, reference):
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if not len(reference):
        return None
    return float(np.mean(reference <= float(value)))


def local_support(index, state, pending, neighbours=None):
    """Local action/episode support and pending-window matches at one state."""
    states = index["states"]
    k = int(index["neighbours"] if neighbours is None else neighbours)
    k = max(1, min(k, len(states)))
    difference = states - np.asarray(state, dtype=float)
    distance = np.einsum("ij,ij->i", difference, difference)
    if k < len(states):
        choice = np.argpartition(distance, k - 1)[:k]
    else:
        choice = np.arange(len(states))
    actions = index["actions"][choice]
    counts = np.bincount(actions, minlength=2)
    acting_counts = np.bincount(
        index["acting"][choice], minlength=2
    )
    episodes = index["episodes"][choice]
    episode_counts = np.asarray([
        len(set(episodes[actions == action].tolist()))
        for action in (0, 1)
    ], dtype=int)
    pending = np.asarray(pending, dtype=int).reshape(-1)
    if pending.size:
        matches = int(np.sum(
            np.all(index["pending"][choice, :pending.size] == pending, axis=1)
        ))
    else:
        matches = int(k)
    return dict(
        neighbours=int(k),
        rows_0=int(counts[0]),
        rows_1=int(counts[1]),
        acting_rows_0=int(acting_counts[0]),
        acting_rows_1=int(acting_counts[1]),
        episodes_0=int(episode_counts[0]),
        episodes_1=int(episode_counts[1]),
        pending_matches=int(matches),
        radius=float(np.sqrt(distance[choice].max())),
    )


def action_endpoint(index, trace, action, reference, obs_dim):
    """The model-best feasible candidate for one first action and its OOD flag."""
    sequences = np.asarray(trace["sequences"], dtype=int)
    scores = np.asarray(trace["scores"], dtype=float)
    feasible = (
        np.asarray(trace["feasible"], dtype=bool)
        & np.asarray(trace["finite"], dtype=bool)
    )
    mask = feasible & (sequences[:, 0] == int(action))
    if not mask.any():
        return None
    chosen = best_candidate_index(scores, mask)
    state = np.asarray(trace["final_states"][chosen], dtype=float)
    state = state[:obs_dim] / index["scale"]
    distance = float(_nearest_distances(
        index["states"], state.reshape(1, -1),
    )[0])
    return dict(
        index=int(chosen),
        distance=distance,
        ood=bool(distance > reference["threshold"]),
        ood_high=bool(distance > reference["high"]),
        percentile=percentile_of(distance, reference["distances"]),
        endpoint=[float(value) for value in state],
    )


def classify_miss(row, min_rows=4, min_episodes=2):
    """The B0 decision tree for one decision-relevant miss.

    ``true_action_contrast`` is the oracle-best true survival under the
    oracle-best first action minus that under the selected first action,
    each with its own best suffix.  When it is zero, both first actions can
    reach the same best outcome; changing the first action cannot fix the
    miss, even if the selected and oracle-best *plans* happen to start with
    different actions.  That is a plan/suffix problem, not an action-level
    routing problem.
    """
    if row["same_first_action"]:
        return "suffix_only"
    contrast = row.get("true_action_contrast")
    if contrast is None:
        return "one_visible_action"
    if int(contrast) <= 0:
        return "suffix_within_action"
    support = (
        row["support"]["rows_0"] >= min_rows
        and row["support"]["rows_1"] >= min_rows
        and row["support"]["episodes_0"] >= min_episodes
        and row["support"]["episodes_1"] >= min_episodes
    )
    if not support:
        return "coverage_support"
    endpoint = row["endpoint_ood"]
    if (
        endpoint["selected"] is None
        or endpoint["selected"]
        or endpoint["oracle_best"]
    ):
        return "coverage_endpoint"
    if row["first_action_gap"] is None:
        return "one_visible_action"
    gap = float(row["first_action_gap"])
    if gap <= 0.03:
        return "ranking"
    if gap > 0.1:
        return "fine_score_or_challenge"
    return "intermediate"


def summarize(rows, decisions, horizon):
    misses = [row for row in rows if row["decision_relevant"]]
    action_different = [row for row in misses if not row["same_first_action"]]
    same_action = [row for row in misses if row["same_first_action"]]
    positive_contrast = [
        row for row in misses
        if row.get("true_action_contrast") is not None
        and int(row["true_action_contrast"]) > 0
    ]
    zero_contrast = [
        row for row in misses
        if row.get("true_action_contrast") is not None
        and int(row["true_action_contrast"]) <= 0
    ]
    gaps = np.asarray([
        row["first_action_gap"] for row in misses
        if row["first_action_gap"] is not None
    ], dtype=float)
    classes = {}
    for row in misses:
        classes[row["class"]] = classes.get(row["class"], 0) + 1
    positive_supported = [
        row for row in positive_contrast
        if row["support_admissible"]
        and not row["endpoint_ood"]["selected"]
        and not row["endpoint_ood"]["oracle_best"]
    ]
    positive_coarse = [
        row for row in positive_supported
        if row["first_action_gap"] is not None
    ]
    positive_near_tie = [
        row for row in positive_coarse
        if float(row["first_action_gap"]) <= 0.03
    ]
    positive_challenge = [
        row for row in positive_coarse
        if 0.03 < float(row["first_action_gap"]) <= 0.1
    ]
    positive_coverage = [
        row for row in positive_contrast
        if not row["support_admissible"]
        or row["endpoint_ood"]["selected"]
        or row["endpoint_ood"]["oracle_best"]
    ]
    support_admissible = sum(
        row["support_admissible"] for row in action_different
    )
    endpoint_admissible = sum(
        not row["endpoint_ood"]["selected"]
        and not row["endpoint_ood"]["oracle_best"]
        for row in action_different
    )
    coarse_admissible = sum(
        row["support_admissible"]
        and not row["endpoint_ood"]["selected"]
        and not row["endpoint_ood"]["oracle_best"]
        for row in action_different
    )
    ood_rates = [
        row["candidate_ood_rate"] for row in rows
        if row["candidate_ood_rate"] is not None
    ]
    ood_rates_high = [
        row["candidate_ood_rate_high"] for row in rows
        if row["candidate_ood_rate_high"] is not None
    ]
    percentiles = [
        row["candidate_percentile_median"] for row in rows
        if row["candidate_percentile_median"] is not None
    ]
    ranking = classes.get("ranking", 0)
    fine = classes.get("fine_score_or_challenge", 0)
    plan_level = (
        classes.get("suffix_only", 0)
        + classes.get("suffix_within_action", 0)
    )
    coverage = (
        classes.get("coverage_support", 0)
        + classes.get("coverage_endpoint", 0)
        + classes.get("one_visible_action", 0)
    )
    if plan_level >= 5:
        verdict = "plan_level"
    elif ranking >= 7:
        verdict = "ranking"
    elif fine >= 5:
        verdict = "fine_score_or_challenge"
    elif coverage >= 5:
        verdict = "coverage"
    else:
        verdict = "mixed"
    return dict(
        decisions=int(decisions),
        misses=int(len(misses)),
        action_different_misses=int(len(action_different)),
        same_first_action_misses=int(len(same_action)),
        positive_action_contrast_misses=int(len(positive_contrast)),
        zero_action_contrast_misses=int(len(zero_contrast)),
        plan_level_misses=int(plan_level),
        action_level_positive_supported=int(len(positive_supported)),
        action_level_near_tie_supported=int(len(positive_near_tie)),
        action_level_challenge_supported=int(len(positive_challenge)),
        action_level_coverage_failures=int(len(positive_coverage)),
        gap_count=int(len(gaps)),
        gap_median=float(np.median(gaps)) if len(gaps) else None,
        gap_p10=float(np.quantile(gaps, 0.1)) if len(gaps) else None,
        gap_p90=float(np.quantile(gaps, 0.9)) if len(gaps) else None,
        gap_le_003=int(np.sum(gaps <= 0.03)) if len(gaps) else 0,
        gap_gt_01=int(np.sum(gaps > 0.1)) if len(gaps) else 0,
        classes=classes,
        support_admissible_action_different=support_admissible,
        endpoint_admissible_action_different=endpoint_admissible,
        coarse_admissible_action_different=coarse_admissible,
        candidate_ood_rate_mean=(
            float(np.mean(ood_rates)) if ood_rates else None
        ),
        candidate_ood_rate_median=(
            float(np.median(ood_rates)) if ood_rates else None
        ),
        candidate_ood_rate_high_mean=(
            float(np.mean(ood_rates_high)) if ood_rates_high else None
        ),
        candidate_percentile_median_mean=(
            float(np.mean(percentiles)) if percentiles else None
        ),
        verdict=verdict,
        calibration="not_measured_no_coarse_model",
        horizon=int(horizon),
        rows=rows,
    )


def decision_points(frames, branches, tail):
    if frames < 2:
        return []
    points = set(
        int(value) for value in np.linspace(2, frames - 2, branches)
    )
    points.update(range(max(1, frames - tail), frames))
    return sorted(points)


def run(args):
    config = probe_config(args)
    world, agent, cached = load_or_fit(
        config, args.cache or None, refit=bool(args.refit)
    )
    print("model cache: %s" % ("hit" if cached else "fit"))
    agent.death_rule_hard = False
    rules = getattr(agent, "death_rules", None)
    if rules is None:
        raise SystemExit("the fitted agent has no death rules")
    death_status = status_callback(rules, agent.obs_dim)
    obs_dim = int(agent.obs_dim)
    index = training_index(
        agent.milestone_training_rows, obs_dim, LAG, agent.scales,
        neighbours=args.neighbours,
    )
    reference = ood_reference(
        index, quantile=args.ood_quantile,
        high_quantile=args.ood_high_quantile,
    )
    print(
        "training index: %d states, held-out nearest p50 %.4f p95 %.4f "
        "p99 %.4f, OOD q95 %.4f q99 %.4f" % (
            len(index["states"]), reference["p50"], reference["p95"],
            reference["p99"], reference["threshold"], reference["high"],
        )
    )

    rows = []
    decisions = 0
    for eval_seed in seed_spec(args.eval_seeds):
        env_seed = int(eval_seed) * 10000
        with _curiosity_off(agent):
            frames, actions, _ = evaluate(
                args.seed, world, agent, cap=5000, horizon=args.horizon,
                lag=LAG, step_stride=2, env_seed=env_seed, record=False,
                episode=0,
            )
        if frames < 2:
            continue
        for point in decision_points(frames, args.branches, args.tail):
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

            batches = []
            tri = policy_result(
                env, world, agent, prefix, args.horizon, lag=LAG,
                step_stride=2, death_status_batch=death_status,
                candidate_callback=batches.append,
            )
            if not batches:
                continue
            trace = flatten_batches(batches)
            sequences = np.asarray(trace["sequences"], dtype=int)
            scores = np.asarray(trace["scores"], dtype=float)
            feasible = (
                np.asarray(trace["feasible"], dtype=bool)
                & np.asarray(trace["finite"], dtype=bool)
            )
            true_first = np.asarray([
                oracle_plan(env, sequence)[0] for sequence in sequences
            ], dtype=int)
            true_value = np.where(
                true_first < 0, args.horizon + 1, true_first
            )
            best_value = int(true_value.max())
            selected_action = int(tri.actions[0])
            selected_true = int(oracle_plan(env, tri.actions)[0])
            selected_value = (
                args.horizon + 1 if selected_true < 0 else selected_true
            )
            regret = int(best_value - selected_value)
            selected_missed = bool(selected_true >= 0 and tri.first_death < 0)
            decision_relevant = bool(selected_missed and regret > 0)
            best_true = action_best(sequences, true_first, args.horizon)
            gap, action_scores = action_score_gap(trace, sequences)
            selected_score = (
                tri.score[0] if isinstance(tri.score, tuple) else tri.score
            )
            details = oracle_best_details(
                trace, sequences, true_first, args.horizon, selected_score,
            )
            oracle_action = int(details["oracle_best_action"])
            same_action = bool(selected_action == oracle_action)

            support = local_support(
                index,
                np.asarray(env.obs(), dtype=float)[:obs_dim] / index["scale"],
                prefix[:LAG],
            )
            support_admissible = bool(
                support["rows_0"] >= args.min_support
                and support["rows_1"] >= args.min_support
                and support["episodes_0"] >= args.min_episodes
                and support["episodes_1"] >= args.min_episodes
            )
            selected_endpoint = None
            selected_state = (
                np.asarray(tri.final_state, dtype=float)[:obs_dim]
                / index["scale"]
            )
            distance = float(_nearest_distances(
                index["states"], selected_state.reshape(1, -1),
            )[0])
            selected_endpoint = dict(
                distance=distance,
                ood=bool(distance > reference["threshold"]),
                ood_high=bool(distance > reference["high"]),
                percentile=percentile_of(
                    distance, reference["distances"]
                ),
                endpoint=[float(value) for value in selected_state],
            )
            oracle_state = (
                np.asarray(
                    trace["final_states"][details["oracle_best_index"]],
                    dtype=float,
                )[:obs_dim] / index["scale"]
            )
            oracle_distance = float(_nearest_distances(
                index["states"], oracle_state.reshape(1, -1),
            )[0])
            oracle_endpoint = dict(
                distance=oracle_distance,
                ood=bool(oracle_distance > reference["threshold"]),
                ood_high=bool(oracle_distance > reference["high"]),
                percentile=percentile_of(
                    oracle_distance, reference["distances"]
                ),
                endpoint=[float(value) for value in oracle_state],
            )
            endpoint_ood = dict(
                selected=(
                    bool(selected_endpoint["ood"])
                    if selected_endpoint is not None else None
                ),
                oracle_best=bool(oracle_endpoint["ood"]),
                selected_high=(
                    bool(selected_endpoint["ood_high"])
                    if selected_endpoint is not None else None
                ),
                oracle_best_high=bool(oracle_endpoint["ood_high"]),
                selected_distance=(
                    selected_endpoint["distance"]
                    if selected_endpoint is not None else None
                ),
                oracle_best_distance=oracle_distance,
                selected_percentile=(
                    selected_endpoint["percentile"]
                    if selected_endpoint is not None else None
                ),
                oracle_best_percentile=oracle_endpoint["percentile"],
            )
            per_action = {}
            for action in (0, 1):
                per_action[action] = action_endpoint(
                    index, trace, action, reference, obs_dim,
                )
            endpoint_ood["action_0"] = (
                per_action[0]["ood"] if per_action[0] is not None else None
            )
            endpoint_ood["action_1"] = (
                per_action[1]["ood"] if per_action[1] is not None else None
            )
            endpoint_ood["action_0_high"] = (
                per_action[0]["ood_high"]
                if per_action[0] is not None else None
            )
            endpoint_ood["action_1_high"] = (
                per_action[1]["ood_high"]
                if per_action[1] is not None else None
            )
            finite_mask = np.asarray(trace["finite"], dtype=bool)
            query_states = (
                np.asarray(trace["final_states"], dtype=float)[
                    finite_mask, :obs_dim
                ] / index["scale"]
            )
            if len(query_states):
                distances = _nearest_distances(index["states"], query_states)
                candidate_ood_rate = float(
                    np.mean(distances > reference["threshold"])
                )
                candidate_ood_rate_high = float(
                    np.mean(distances > reference["high"])
                )
                candidate_distance_median = float(np.median(distances))
                candidate_percentile_median = percentile_of(
                    float(np.median(distances)), reference["distances"]
                )
            else:
                candidate_ood_rate = None
                candidate_ood_rate_high = None
                candidate_distance_median = None
                candidate_percentile_median = None
            action_contrast = None
            if (
                selected_action in best_true and oracle_action in best_true
            ):
                action_contrast = int(
                    best_true[oracle_action] - best_true[selected_action]
                )
            row = dict(
                eval_seed=int(eval_seed), point=int(point),
                candidates=int(len(sequences)),
                feasible=int(feasible.sum()),
                selected_action=selected_action,
                oracle_best_action=oracle_action,
                same_first_action=same_action,
                selected_true=int(selected_true),
                selected_value=int(selected_value),
                oracle_best_value=int(best_value),
                regret=int(regret),
                selected_missed=bool(selected_missed),
                decision_relevant=bool(decision_relevant),
                first_action_gap=gap,
                plan_margin=plan_score_gap(trace),
                action_scores=action_scores,
                action_true_best={
                    int(key): int(value) for key, value in best_true.items()
                },
                true_action_contrast=action_contrast,
                oracle_best_feasible=bool(details["oracle_best_feasible"]),
                oracle_best_score=float(details["oracle_best_score"]),
                oracle_best_score_gap=float(
                    details["oracle_best_score_gap"]
                ),
                oracle_best_rank=int(details["oracle_best_rank"]),
                support=support,
                support_admissible=support_admissible,
                endpoint_ood=endpoint_ood,
                action_endpoints={
                    int(key): value for key, value in per_action.items()
                },
                candidate_ood_rate=candidate_ood_rate,
                candidate_ood_rate_high=candidate_ood_rate_high,
                candidate_distance_median=candidate_distance_median,
                candidate_percentile_median=candidate_percentile_median,
            )
            row["class"] = classify_miss(
                row, args.min_support, args.min_episodes
            ) if decision_relevant else "not_relevant"
            rows.append(row)
            decisions += 1

    summary = summarize(rows, decisions, args.horizon)
    print(
        "== seed %d eval %s: %d decisions, %d relevant misses "
        "(%d action-different, %d suffix-only) ==" % (
            args.seed, args.eval_seeds, summary["decisions"],
            summary["misses"], summary["action_different_misses"],
            summary["same_first_action_misses"],
        )
    )
    if summary["misses"]:
        print(
            "   plan-level %d  action-contrast %d  zero-contrast %d"
            % (
                summary["plan_level_misses"],
                summary["positive_action_contrast_misses"],
                summary["zero_action_contrast_misses"],
            )
        )
        print(
            "   gap p10/p50/p90 %s  <=0.03 %d  >0.1 %d" % (
                " / ".join(
                    "None" if value is None else "%.4f" % value
                    for value in (
                        summary["gap_p10"], summary["gap_median"],
                        summary["gap_p90"],
                    )
                ),
                summary["gap_le_003"], summary["gap_gt_01"],
            )
        )
        print("   classes: %s" % summary["classes"])
        print(
            "   action-different: support-admissible %d, "
            "endpoint-admissible %d, coarse-admissible %d" % (
                summary["support_admissible_action_different"],
                summary["endpoint_admissible_action_different"],
                summary["coarse_admissible_action_different"],
            )
        )
        print(
            "   positive-contrast: supported %d, near-tie %d, "
            "challenge %d, coverage-fail %d" % (
                summary["action_level_positive_supported"],
                summary["action_level_near_tie_supported"],
                summary["action_level_challenge_supported"],
                summary["action_level_coverage_failures"],
            )
        )
        print(
            "   candidate endpoint OOD q95 mean/median %s / %s  "
            "q99 mean %s  median-percentile %s" % (
                "None" if summary["candidate_ood_rate_mean"] is None
                else "%.2f" % summary["candidate_ood_rate_mean"],
                "None" if summary["candidate_ood_rate_median"] is None
                else "%.2f" % summary["candidate_ood_rate_median"],
                "None" if summary["candidate_ood_rate_high_mean"] is None
                else "%.2f" % summary["candidate_ood_rate_high_mean"],
                "None"
                if summary["candidate_percentile_median_mean"] is None
                else "%.2f"
                % summary["candidate_percentile_median_mean"],
            )
        )
        print("   B0 verdict: %s" % summary["verdict"])
        for row in (item for item in rows if item["decision_relevant"]):
            print(
                "   seed %d @%d: %-22s sel %d oracle %d %s  "
                "gap %s margin %s rank %d  support %d/%d ep %d/%d "
                "pend %d acting %d/%d  contrast %s  "
                "ood sel/oracle %s/%s all %s" % (
                    row["eval_seed"], row["point"], row["class"],
                    row["selected_action"], row["oracle_best_action"],
                    "same" if row["same_first_action"] else "diff",
                    "None" if row["first_action_gap"] is None
                    else "%.4f" % row["first_action_gap"],
                    "None" if row["plan_margin"] is None
                    else "%.4f" % row["plan_margin"],
                    row["oracle_best_rank"],
                    row["support"]["rows_0"], row["support"]["rows_1"],
                    row["support"]["episodes_0"], row["support"]["episodes_1"],
                    row["support"]["pending_matches"],
                    row["support"]["acting_rows_0"],
                    row["support"]["acting_rows_1"],
                    "None" if row["true_action_contrast"] is None
                    else "%+d" % row["true_action_contrast"],
                    row["endpoint_ood"]["selected"],
                    row["endpoint_ood"]["oracle_best"],
                    "None" if row["candidate_ood_rate"] is None
                    else "%.2f" % row["candidate_ood_rate"],
                )
            )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--neighbours", type=int, default=64)
    parser.add_argument("--min-support", type=int, default=4)
    parser.add_argument("--min-episodes", type=int, default=2)
    parser.add_argument("--ood-quantile", type=float, default=0.95)
    parser.add_argument("--ood-high-quantile", type=float, default=0.99)
    parser.add_argument("--cache", default="")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
