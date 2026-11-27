"""Node-centric planning on Flappy using only signals, not task rules.

The runtime receives:

    observation, action, valence (dopamine/pain), episode boundary

It never reads task geometry, a collision window, or a `terminal` risk label.
A learned observation model predicts candidate trajectories; the node runtime
supplies the learned valence of their endpoints.
The node graph itself is trained from real transitions with TD(lambda) credit
assignment and action-conditioned edges.
"""
import argparse
import itertools
import json
import os
import sys
from dataclasses import replace
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np

from _paths import ensure_core_paths
ensure_core_paths()

from eval_protocol import (
    cluster_summary,
    group_summaries,
    parse_seed_spec,
)
from latent_world import SEEDS, latent
from milestone_graph import (
    MilestoneTracker,
    boundary_axis,
    frame_project,
    frame_project_axes,
    milestone_coverage,
    milestone_lookahead,
    milestone_memory,
    nearest_stage_risk,
    pooled_boundary_model,
    pooled_strength_threshold,
    segment_prototypes,
    signature_clusters,
    signature_keys,
    signature_stage_risk,
    strength_boundaries,
    strength_change_scores,
    transition_signatures,
)
from multiview_transitions import MultiViewTransitionModel
from node_features import FeatureRuleBank
from node_runtime import NodeRuntime, future_pain_labels, future_return_labels
from predictive_option_quotient import OptionQuotient
from predictive_stage_tree import (
    PredictiveStageTree,
    collapse_stage_episodes,
)
from stage_demand_heuristic import (
    DemandAssociationSample,
    StageDemandHeuristic,
)
from stage_demand_selector import (
    StageDemandSelector,
    StageOutcomeReadout,
)
from stage_experience import StageExperienceBatch
from stage_outcome_memory import StageOutcomeMemory
from stage_state_resolver import StageStateResolver
from substrate_generic import (
    LinearObservationModel,
    collect,
    history_features,
)
from trajectory_search import (
    SearchResult,
    beam_search,
    chunked_beam_search,
    exhaustive_search,
    merge_candidate_batches,
    override_first_action,
)

LAG = 3
DEFAULT_JOBS = min(8, os.cpu_count() or 1)
FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()






from chained_search import (
    _expand_clean_chunk,
    _extend_clean_frontier_one_chunk,
    _run_clean_chain_tier,
    _score_clean_frontier,
    _select_adaptive_beam,
    adaptive_chained_result,
    association_value_batch,
    chained_shallow_result,
    terminal_hazard,
)


def _association_value_table(transitions, agent, window: int = 8,
                             min_support: int = 5):
    """Value by association: what tends to follow a state, and how good it is.

    Pure counting over states the stream actually occupied: for every state,
    the nodes met within ``window`` frames after it, and each successor's
    realised alive fraction.  The value is the weighted mean of those
    successors' quality.  No model and no counterfactual - and, unlike a
    node's own mean return, not confounded with where in the episode the node
    happens to sit (the failure measured on 2026-11-22).
    """
    episodes = {}
    for row in transitions:
        episodes.setdefault(int(row.get("episode", 0)), []).append(row)
    assoc, totals = {}, {}
    quality, quality_count = {}, {}
    for rows in episodes.values():
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        encoded = np.concatenate(
            [observations, _encoded_pending(rows, agent.lag)], axis=1
        ) / agent.scales
        nodes = []
        for state in encoded:
            node_id, _distance = agent._nearest(state)
            nodes.append(node_id)
        alive = np.asarray([
            0.0 if bool(row.get("terminal")) else 1.0 for row in rows
        ])
        for index, node in enumerate(nodes):
            if node is None:
                continue
            stop = min(index + 1 + int(window), len(nodes))
            fraction = (float(alive[index + 1:stop].mean())
                        if stop > index + 1 else 0.0)
            quality[node] = quality.get(node, 0.0) + fraction
            quality_count[node] = quality_count.get(node, 0) + 1
            for other in nodes[index + 1:stop]:
                if other is None or other == node:
                    continue
                neighbours = assoc.setdefault(node, {})
                neighbours[other] = neighbours.get(other, 0) + 1
                totals[node] = totals.get(node, 0) + 1
    values = {}
    for node, neighbours in assoc.items():
        if totals.get(node, 0) < int(min_support):
            continue
        weight_sum = 0.0
        total = 0.0
        for other, count in neighbours.items():
            if quality_count.get(other, 0) <= 0:
                continue
            weight = count / float(totals[node])
            total += weight * quality[other] / quality_count[other]
            weight_sum += weight
        if weight_sum > 0.0:
            values[node] = total / weight_sum
    support = {
        node: int(quality_count[node]) for node in values
        if quality_count.get(node, 0) > 0
    }
    return values, support




def _transition_table(transitions, agent, min_support: int = 2):
    """Node-to-node transitions from the stream: the unit this policy decides on.

    For every consecutive pair of entered states, record the edge, how often it
    was taken, which action was used on it, and how often entering the target
    was immediately painful.  Nothing is predicted: every entry is a transition
    the stream actually executed.  Returns ``(edges, risk, default_action)``.
    """
    episodes = {}
    for row in transitions:
        episodes.setdefault(int(row.get("episode", 0)), []).append(row)
    raw = {}
    pain = {}
    visits = {}
    action_totals = np.zeros(2)
    for rows in episodes.values():
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        encoded = np.concatenate(
            [observations, _encoded_pending(rows, agent.lag)], axis=1
        ) / agent.scales
        signals = np.asarray(
            [row.get("signals", (0.0, 0.0)) for row in rows], dtype=float
        )
        histories = [
            np.asarray(row.get("history", ()), dtype=float).reshape(-1)
            for row in rows
        ]
        # Same delayed-action alignment used by NodeRuntime's action-conditioned
        # transition/value updates: history[lag] acted on this transition.
        actions = np.asarray([
            int(round(float(history[agent.lag])))
            if history.size > int(agent.lag) else -1
            for history in histories
        ], dtype=int)
        nodes = []
        for state in encoded:
            node_id, _distance = agent._nearest(state)
            nodes.append(node_id)
        for index, node in enumerate(nodes):
            if node is None:
                continue
            visits[node] = visits.get(node, 0) + 1
            pain[node] = pain.get(node, 0.0) + float(signals[index, 1] > 0.0)
            action = int(actions[index])
            if action < 0:
                continue
            action_totals[action] += 1
            if bool(rows[index].get("done", False)):
                continue
            if index + 1 >= len(nodes):
                continue
            target = nodes[index + 1]
            if target is None:
                continue
            edge = raw.setdefault(node, {}).setdefault(
                target, dict(count=0, actions={})
            )
            edge["count"] += 1
            edge["actions"][action] = edge["actions"].get(action, 0) + 1
    edges = {}
    for node, targets in raw.items():
        kept = {}
        for target, entry in targets.items():
            if entry["count"] < int(min_support):
                continue
            action_counts = dict(
                (int(action), int(count))
                for action, count in entry["actions"].items()
            )
            action = max(
                action_counts, key=action_counts.get
            ) if action_counts else 0
            kept[target] = dict(
                count=int(entry["count"]), action=int(action),
                actions=action_counts,
            )
        if kept:
            edges[node] = kept
    risk = {
        node: float(pain[node]) / visits[node]
        for node in visits if visits[node] > 0
    }
    default_action = int(np.argmax(action_totals)) if action_totals.any() else 0
    return edges, risk, default_action


def transition_action(agent, observation, prefix, risk_weight: float = 1.0):
    """Choose the action that realises the best observed successor transition.

    The decision is the *edge*, not an action sequence: among the successors
    the stream actually reached from this node, take the one with the best
    realised value net of its immediate pain rate, and execute the action that
    edge was taken with.  The world model is not rolled here.
    """
    pending = np.asarray(prefix, dtype=float).reshape(-1)[:int(agent.lag)][::-1]
    if len(pending) < int(agent.lag):
        pending = np.concatenate([
            np.zeros(int(agent.lag) - len(pending)), pending,
        ])
    encoded = np.concatenate([
        np.asarray(observation, dtype=float).reshape(-1), pending,
    ]) / agent.scales
    node_id, _distance = agent._nearest(encoded)
    policy = getattr(agent, "transition_policy", None) or {}
    if node_id in policy:
        return int(policy[node_id])
    edges = (getattr(agent, "transition_edges", None) or {}).get(node_id) or {}
    if not edges:
        return int(getattr(agent, "transition_default_action", 0))
    values = getattr(agent, "association_values", None) or {}
    risks = getattr(agent, "node_risk", None) or {}
    best = max(
        edges,
        key=lambda target: (
            float(values.get(target, 0.0))
            - float(risk_weight) * float(risks.get(target, 0.0))
            + 1e-6 * edges[target]["count"]
        ),
    )
    return int(edges[best]["action"])


def _transition_values(edges, risk, gamma: float = 0.95,
                       iterations: int = 60):
    """Model-based value iteration on the state-transition graph.

    The graph is the model: ``edges[s][t]`` says successor ``t`` was reached
    from ``s`` with a given count and action.  Value iteration composes those
    one-step transitions into a multi-step value, so the policy is chosen on
    the graph rather than on a frame rollout.  Reward is the world's own
    signal - alive is +1 and entering a painful state is -1:

    ``r(t) = 1 - 2 * risk(t)``

    Returns ``(policy, values)`` where ``policy[s]`` is the action with the
    best action-value at ``s``.
    """
    nodes = set(edges)
    values = {node: 0.0 for node in nodes}
    policy = {}
    for _ in range(int(iterations)):
        updated = {}
        for node, targets in edges.items():
            by_action = {}
            for target, entry in targets.items():
                action_counts = entry.get(
                    "actions", {int(entry["action"]): int(entry["count"])}
                )
                for action, count in action_counts.items():
                    by_action.setdefault(int(action), {})[target] = (
                        float(count)
                    )
            best_value = None
            best_action = None
            for action, distribution in by_action.items():
                total = sum(distribution.values())
                if total <= 0.0:
                    continue
                value = 0.0
                for target, count in distribution.items():
                    reward = 1.0 - 2.0 * float(risk.get(target, 0.0))
                    value += (count / total) * (
                        reward + float(gamma) * values.get(target, 0.0)
                    )
                if best_value is None or value > best_value:
                    best_value = value
                    best_action = action
            if best_value is not None:
                updated[node] = best_value
                policy[node] = int(best_action)
        if not updated:
            break
        values = updated
    return policy, values


def _curiosity_potential(support):
    """Curiosity potential: high where the readout's evidence is thin.

    ``濠?s) = 1/sqrt(1 + evidence(s))`` with evidence the mean observation
    count behind the state's active cells.  It is bounded in (0, 1], so
    the shaped objective cannot out-scale the hazard and value terms, and
    it needs no task knowledge - only the counters the learner already
    maintains.
    """
    support = np.asarray(support, dtype=float)
    return 1.0 / np.sqrt(1.0 + np.maximum(support, 0.0))


def _curiosity_term(phi_start, phi_end, weight, discount, horizon):
    """Potential-based curiosity shaping summed over a fixed-horizon plan.

    PBRS adds ``缂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌ｉ幋锝呅撻柛銈呭閺屾盯顢曢敐鍡欘槰闂佽鍨抽崑銈夌嵁閺嶎灔搴敆閳ь剚淇婇懖鈺冪＜闁逞屽墴瀹曞ジ鎮㈢粙鍨紟婵犵妲呴崹闈涒枖閺囥垹閱囨い蹇撶墛閻?s') - 濠?s)`` per step; over H steps the sum telescopes
    to ``缂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴闇夐柨婵嗙墱濞兼劖绻涢崨顖毿㈤棁澶愭煟濡儤鈻曢柛搴＄箳缁辨帡鎮╁畷鍥р拡缂?濠?s_H) - 濠?s_0)``.  The planner's step costs are accumulated
    without a per-step discount (``缂?= 1`` in the rollout), so the term
    enters as ``w * (濠?s_0) - 濠?s_H))``: a less-supported endpoint lowers
    the cost.  Every plan shares ``s_0``, so the start term is a constant;
    because this is a potential difference the shaped objective has the
    same optimum as the unshaped one, which is the guardrail against
    curiosity reward hacking.  `discount` is kept explicit so a future
    discounted rollout can pass its own factor.
    """
    phi_end = np.asarray(phi_end, dtype=float)
    return float(weight) * (
        float(phi_start) - float(discount) ** int(horizon) * phi_end
    )


def _apply_margin_tiebreak(result, batches, previous_action, eps):
    """Among near-tied plans, prefer one that keeps the previous action.

    The planner's top two candidates routinely differ by ~1e-3 while the terms
    it is asked to weigh are ~1e-2, so an added term decides the winner on
    noise.  This gate does not choose a different objective: it only says that
    when several plans are within `eps` of the best, the one whose first action
    continues the current one wins.  `eps` is read off the decision's own
    margin distribution (``decision_margin_probe.py``), not picked per world.
    """
    if not batches or previous_action is None or eps <= 0.0:
        return result
    scores = np.vstack([np.asarray(batch["scores"], dtype=float)
                        for batch in batches])
    sequences = np.vstack([np.asarray(batch["sequences"], dtype=float)
                           for batch in batches])
    feasible = np.concatenate([
        np.asarray(batch.get("feasible", np.ones(len(batch["scores"]), bool)),
                   dtype=bool)
        for batch in batches
    ])
    finite = np.concatenate([
        np.asarray(batch.get("finite", np.ones(len(batch["scores"]), bool)),
                   dtype=bool)
        for batch in batches
    ])
    usable = feasible & finite
    if usable.sum() < 2:
        return result
    cost = scores[:, 0]
    best = float(np.min(cost[usable]))
    near = usable & (cost <= best + float(eps))
    keep = near & (np.asarray(sequences[:, 0], dtype=float)
                   == float(previous_action))
    if not keep.any():
        return result
    index = int(np.flatnonzero(keep)[np.argmin(cost[keep])])
    actions = np.asarray(result.actions, dtype=float).copy()
    actions[:] = sequences[index]
    return replace(result, actions=actions, score=scores[index])


def make_node_demand_rank_step(agent, demand, heuristic, stats=None):
    """Adapt learned Runtime feature masks to the demand-only rank channel."""
    from stage_demand_heuristic import make_demand_rank_step

    all_features = list(getattr(agent, "_feature_all", ()))
    roots = set(getattr(agent, "feature_ids", ()))
    feature_columns = [
        (index, feature_id)
        for index, feature_id in enumerate(all_features)
        if feature_id in roots
    ]
    obs_dim = int(agent.obs_dim)
    scales = np.asarray(agent.scales[:obs_dim], dtype=float)

    def extract_features(states, _sequences, _step):
        observations = (
            np.asarray(states, dtype=float)[:, :obs_dim] / scales
        )
        masks = agent._feature_mask_table(observations)
        active_by_state = [[] for _ in range(len(observations))]
        for column, feature_id in feature_columns:
            if column >= len(masks) or masks[column] is None:
                continue
            active = np.asarray(masks[column], dtype=bool).reshape(-1)
            if active.shape != (len(observations),):
                raise ValueError(
                    "feature mask must return one value per candidate"
                )
            for index in np.flatnonzero(active):
                active_by_state[int(index)].append(feature_id)
        return active_by_state

    return make_demand_rank_step(
        heuristic, demand, feature_extractor=extract_features,
        stats=stats,
    )


def make_node_prototype_rank_step(agent, demand, heuristic,
                                  pending_prefix, lag, stats=None):
    """Adapt runtime prototype features to the demand-only rank channel.

    Candidate states are encoded with their pending action window exactly as
    the runtime's state encoder does, then mapped to the nearest learned
    prototype.  This is the same token the experience pipeline used when it
    fit a prototype-conditioned association; no task geometry enters.
    """
    from stage_demand_heuristic import make_demand_rank_step

    obs_dim = int(agent.obs_dim)
    lag = int(lag)
    scales = np.asarray(agent.scales, dtype=float)
    prefix = np.asarray(pending_prefix, dtype=float).reshape(-1)

    def extract_features(states, sequences, step):
        history = history_features(
            sequences, step + 1, prefix, action_dim=lag + 1,
        )
        pending = history[:, :lag][:, ::-1] if lag else np.zeros(
            (len(states), 0)
        )
        encoded = np.concatenate(
            [np.asarray(states, dtype=float)[:, :obs_dim], pending], axis=1,
        )
        if encoded.shape[1] != scales.size:
            raise ValueError(
                "candidate state dimension does not match the agent scales"
            )
        encoded = encoded / scales
        features = []
        for state in encoded:
            node, distance = agent._nearest(state)
            if node is None or not np.isfinite(distance):
                features.append(())
            else:
                features.append((("prototype", str(node)),))
        return features

    return make_demand_rank_step(
        heuristic, demand, feature_extractor=extract_features,
        stats=stats,
    )


def make_node_hybrid_rank_step(agent, demand, heuristic,
                               pending_prefix, lag, stats=None):
    """Combine learned feature roots and prototypes for demand ranking.

    Training and candidate-time tokens are namespaced identically. The
    high-level demand changes only beam priority; it does not alter feature
    readouts, feasibility, or the final objective score.
    """
    from stage_demand_heuristic import make_demand_rank_step

    obs_dim = int(agent.obs_dim)
    lag = int(lag)
    scales = np.asarray(agent.scales, dtype=float)
    prefix = np.asarray(pending_prefix, dtype=float).reshape(-1)
    branch = getattr(agent, "_branch_active_features", None)

    def extract_features(states, sequences, step):
        states = np.asarray(states, dtype=float)
        history = history_features(
            sequences, step + 1, prefix, action_dim=lag + 1,
        )
        pending = history[:, :lag][:, ::-1] if lag else np.zeros(
            (len(states), 0)
        )
        observations = states[:, :obs_dim]
        encoded = np.concatenate([observations, pending], axis=1)
        if encoded.shape[1] != scales.size:
            raise ValueError(
                "candidate state dimension does not match the agent scales"
            )
        encoded /= scales
        value_roots = set(getattr(agent, "value_feature_ids", ()))
        all_features = list(getattr(agent, "_feature_all", ()))
        value_masks = (
            agent._feature_mask_table(observations / scales[:obs_dim])
            if value_roots else ()
        )

        feature_sets = []
        for index, (observation, state) in enumerate(zip(observations, encoded)):
            features = []
            if branch is not None:
                features.extend(
                    ("branch", feature)
                    for feature in branch(observation)
                )
            node, distance = agent._nearest(state)
            if node is not None and np.isfinite(distance):
                features.append(("prototype", str(node)))
            for feature_id in value_roots:
                if feature_id not in all_features:
                    continue
                mask = value_masks[all_features.index(feature_id)]
                if mask is not None and bool(
                        np.asarray(mask, dtype=bool).reshape(-1)[index]):
                    features.append(("value", str(feature_id)))
            feature_sets.append(tuple(features))
        return feature_sets

    return make_demand_rank_step(
        heuristic, demand, feature_extractor=extract_features,
        stats=stats,
    )


def make_node_root_rank_step(agent, demand, heuristic,
                             pending_prefix, lag, stats=None):
    """Rank candidate states by their resolved stage root.

    The resolver maps the candidate state to a realized stage type and the
    predictive tree collapses that type to its accepted component root; the
    resulting token is the same feature used in ``from_stage_roots`` samples.
    A candidate the resolver cannot map (or a key with no tree node) gets a
    neutral empty feature, never a fabricated root.
    """
    from stage_demand_heuristic import make_demand_rank_step

    resolver = getattr(agent, "stage_resolver", None)
    tree = getattr(agent, "stage_tree", None)
    if resolver is None or tree is None:
        raise ValueError("root rank requires a fitted resolver and stage tree")
    obs_dim = int(agent.obs_dim)
    lag = int(lag)
    scales = np.asarray(agent.scales, dtype=float)
    prefix = np.asarray(pending_prefix, dtype=float).reshape(-1)

    def extract_features(states, sequences, step):
        history = history_features(
            sequences, step + 1, prefix, action_dim=lag + 1,
        )
        pending = history[:, :lag][:, ::-1] if lag else np.zeros(
            (len(states), 0)
        )
        encoded = np.concatenate(
            [np.asarray(states, dtype=float)[:, :obs_dim], pending], axis=1,
        )
        if encoded.shape[1] != scales.size:
            raise ValueError(
                "candidate state dimension does not match the agent scales"
            )
        raw_observations = np.asarray(states, dtype=float)[:, :obs_dim]
        target_stages = tuple(getattr(
            demand, "target_stages", (demand.target_stage,),
        ))
        features = []
        for observation, queue in zip(raw_observations, pending):
            resolved = resolver.resolve(agent, observation, queue)
            node_token = None
            if resolved["status"] == "supported":
                chain = tree.chain(resolved["key"])
                if chain:
                    node_token = ("tree", str(chain[-1]))
            if node_token is None:
                if stats is not None:
                    stats["feature_empty"] = (
                        int(stats.get("feature_empty", 0)) + 1
                    )
                features.append(())
                continue
            if stats is not None:
                stats["feature_present"] = (
                    int(stats.get("feature_present", 0)) + 1
                )
                if (node_token,) in target_stages:
                    stats["target_present"] = (
                        int(stats.get("target_present", 0)) + 1
                    )
            features.append((node_token,))
        return features

    return make_demand_rank_step(
        heuristic, demand, feature_extractor=extract_features,
        stats=stats,
    )


def make_stage_value_penalty(agent, root_penalties, prefix, lag,
                             neutral: float = 0.5):
    """Map final plan states to a soft stage-type value penalty.

    ``root_penalties`` maps predictive-tree root tokens to ``1 - value``.
    The callable resolves each final state through the same resolver the root
    rank adapter uses, then reads its tree root.  Unknown states get the
    neutral penalty, never a fabricated value.
    """
    resolver = getattr(agent, "stage_resolver", None)
    tree = getattr(agent, "stage_tree", None)
    if resolver is None or tree is None:
        raise ValueError("stage value needs a fitted resolver and stage tree")
    obs_dim = int(agent.obs_dim)
    lag = int(lag)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    penalties = dict(root_penalties)

    def penalty(states, sequences):
        states = np.asarray(states, dtype=float)
        sequences = np.asarray(sequences, dtype=float)
        history = history_features(
            sequences, sequences.shape[1], pending_prefix,
            action_dim=lag + 1,
        )
        pending = history[:, :lag][:, ::-1] if lag else np.zeros(
            (len(states), 0)
        )
        values = np.full(len(states), float(neutral), dtype=float)
        for index, (state, queue) in enumerate(zip(states, pending)):
            observation = np.asarray(state[:obs_dim], dtype=float)
            resolved = resolver.resolve(agent, observation, queue)
            token = None
            if resolved.get("status") == "supported":
                chain = tree.chain(resolved["key"])
                token = (
                    ("tree", "base") if not chain
                    else ("tree", str(chain[-1]))
                )
            if token is not None:
                values[index] = float(penalties.get(token, neutral))
        return values

    return penalty


def policy_result(env, world, agent, prefix, horizon, lag=LAG,
                  step_stride=2, stride_includes_lag=False,
                  death_status_batch=None, unknown_penalty=0.0,
                  near_penalty=0.0, candidate_callback=None,
                  inquiry_action=None, risk_weight=1.0,
                  risk_decay=1.0, beam_width=0, milestone_weight=0.0,
                  milestone_mode="off", milestone_rate=None,
                  milestone_goal=None, pain_weight=None, risk_lane="max",
                  previous_action=None, margin_eps=0.0, veto_fraction=0.0,
                  rank_step=None, stage_demand=None,
                  demand_heuristic=None, rank_stats=None,
                  rank_mode="order", rank_margin=0.0,
                  stage_value=None, stage_value_weight=0.0,
                  frontier_callback=None, search_chunk_size=0,
                  rank_group_quota=1):
    initial = np.asarray(env.obs(), dtype=float)
    agent_obs_dim = int(agent.obs_dim)
    if (stage_demand is None) != (demand_heuristic is None):
        raise ValueError(
            "stage_demand and demand_heuristic must be supplied together"
        )
    if stage_demand is not None:
        if rank_step is not None:
            raise ValueError(
                "supply either rank_step or stage_demand, not both"
            )
        mode = getattr(agent, "stage_rank_feature_source", None)
        if mode == "root":
            rank_step = make_node_root_rank_step(
                agent, stage_demand, demand_heuristic, prefix, lag,
                stats=rank_stats,
            )
        elif getattr(
                agent, "stage_demand_feature_source", "branch",
        ) == "hybrid":
            rank_step = make_node_hybrid_rank_step(
                agent, stage_demand, demand_heuristic, prefix, lag,
                stats=rank_stats,
            )
        elif getattr(
                agent, "stage_demand_feature_source", "branch",
        ) == "prototype":
            rank_step = make_node_prototype_rank_step(
                agent, stage_demand, demand_heuristic, prefix, lag,
                stats=rank_stats,
            )
        else:
            rank_step = make_node_demand_rank_step(
                agent, stage_demand, demand_heuristic,
                stats=rank_stats,
            )
    if rank_step is not None and int(beam_width) <= 0:
        raise ValueError("rank_step requires beam search with positive beam_width")
    if int(search_chunk_size) < 0:
        raise ValueError("search_chunk_size must be non-negative")
    if int(rank_group_quota) < 0:
        raise ValueError("rank_group_quota must be non-negative")
    if int(search_chunk_size) > 0 and int(beam_width) <= 0:
        raise ValueError(
            "search_chunk_size requires beam search with positive beam_width"
        )

    def agent_view(batch):
        """The agent's raw observation; a world may carry extra state dims."""
        return np.asarray(batch, dtype=float)[:, :agent_obs_dim]

    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    captured = []
    user_callback = candidate_callback

    def _capture(batch):
        captured.append(batch)
        if user_callback is not None:
            user_callback(batch)

    candidate_callback = _capture
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)
    lag = int(lag)
    risk_weight = float(risk_weight)
    risk_decay = float(risk_decay)
    milestone_weight = float(milestone_weight)
    error_weight = bool(getattr(agent, "risk_error_weight", False))
    step_errors = list(getattr(agent, "step_errors", ()) or ())

    def error_factor(step_index):
        if not error_weight or not step_errors:
            return 1.0
        index = min(int(step_index), len(step_errors) - 1)
        return 1.0 / (1.0 + float(step_errors[index]))
    # Decision-side curiosity: the potential at the shared start state is
    # a constant across plans, but it is evaluated so the term is exactly
    # the potential difference (PBRS identity), not a bare endpoint bonus.
    curiosity_weight = float(getattr(agent, "curiosity_weight", 0.0))
    action_edge_weight = float(getattr(agent, "action_edge_weight", 0.0))
    q_table = (
        agent.action_edge_transition_table()
        if action_edge_weight > 0.0 else None
    )
    death_rules = getattr(agent, "death_rules", None)
    death_batch = None
    if death_status_batch is None and death_rules is not None and bool(
        getattr(agent, "death_rule_hard", False)
    ):
        envelope = bool(getattr(agent, "death_rule_envelope", False))

        def death_batch(states, _sequences, step):
            observed = np.asarray(states, dtype=float)[:, :agent.obs_dim]
            predicted = death_rules.classify_death(
                observed, include_margin=envelope
            )
            return np.where(predicted == 1, int(step), -1)
    previous_holder = {"states": None}

    def previous_states_for(states, step):
        if (
            step == 0 or previous_holder["states"] is None
            or len(previous_holder["states"]) != len(states)
        ):
            previous = np.broadcast_to(
                initial, (len(states), len(initial))
            ).copy()
        else:
            previous = previous_holder["states"]
        previous_holder["states"] = np.asarray(states, dtype=float).copy()
        return previous

    phi_start = 0.0
    if curiosity_weight > 0.0:
        pending_start = (
            pending_prefix[:lag][::-1] if lag > 0
            else np.zeros(0, dtype=float)
        )
        encoded_start = np.concatenate(
            [agent_view(initial.reshape(1, -1)), pending_start.reshape(1, -1)],
            axis=1,
        ) / agent.scales
        phi_start = float(
            _curiosity_potential(agent.evidence_batch(encoded_start))[0]
        )

    action_space = tuple(getattr(agent, "action_space", (0, 1)))
    n_actions = len(action_space)
    full_exhaustive_count = (
        (n_actions ** int(horizon))
        if int(beam_width) == 0 and n_actions > 0 else 0
    )

    def _prefix_stride(count, step):
        if (
            full_exhaustive_count > 0
            and count == full_exhaustive_count
            and step + 1 < horizon
        ):
            return n_actions ** (horizon - 1 - step)
        return 1

    def predict_batch(states, sequences, step):
        stride = _prefix_stride(len(states), step)
        if stride > 1:
            history = history_features(
                sequences[::stride], step + 1, pending_prefix,
                action_dim=action_dim,
            )
            predicted = world.predict(states[::stride], history)
            return np.repeat(predicted, stride, axis=0)
        history = history_features(
            sequences, step + 1, pending_prefix, action_dim=action_dim
        )
        return world.predict(states, history)

    if getattr(agent, "contrast_objective", False):
        # F1: the per-step cost is the action contrast (per-state relative
        # action quality, absolute lethality cancels) and the endpoint cost
        # is the value term alone - the value channel books the long
        # survival that pays for traversing a genuinely hazardous band.
        # Columns 2-3 carry the objective's ledger - the hazard and value
        # shares of the scalarized column 0 - so the executed plan's
        # components can be regressed against its realized outcome without
        # a second pass (P4 objective learning).
        def evaluate_step(states, sequences, step):
            previous = previous_states_for(states, step)
            if step % step_stride != 0 and not (
                stride_includes_lag and step == lag
            ):
                return (
                    np.zeros((len(states), 6)),
                    np.isfinite(states).all(axis=1),
                )
            history = history_features(
                sequences, step + 1, pending_prefix, action_dim=action_dim
            )
            pending = history[:, :agent.lag][:, ::-1]
            encoded = np.concatenate([agent_view(states), pending], axis=1) / agent.scales
            previous_history = history_features(
                sequences, step, pending_prefix, action_dim=action_dim
            )
            previous_pending = previous_history[:, :agent.lag][:, ::-1]
            previous_encoded = np.concatenate(
                [agent_view(previous), previous_pending], axis=1
            ) / agent.scales
            threat, excess = agent.pain_graded_batch(encoded)
            q_cost = np.zeros(len(states))
            if q_table is not None:
                q_pain, q_count = agent.action_edge_transition_q_batch(
                    previous_encoded, encoded, q_table
                )
                trust = q_count / (
                    q_count + agent.action_edge_trust_samples
                )
                q_cost = action_edge_weight * trust * (
                    q_pain / (1.0 + np.abs(q_pain))
                )
            cost = threat + agent.control_risk_weight * excess + q_cost
            zeros = np.zeros(len(states))
            return (
                np.column_stack([
                    cost, zeros, threat + agent.control_risk_weight * excess,
                    zeros, zeros, q_cost,
                ]),
                np.isfinite(encoded).all(axis=1),
            )

        def evaluate_terminal(states, sequences):
            history = history_features(
                sequences, horizon, pending_prefix, action_dim=action_dim
            )
            pending = history[:, :agent.lag][:, ::-1]
            encoded = np.concatenate([agent_view(states), pending], axis=1) / agent.scales
            _pain, dopamine = agent.valence_state_batch(encoded)
            mask_table = agent._feature_mask_table(
                encoded[:, :agent.obs_dim]
            )
            value, support = agent.feature_value_batch(
                encoded, mask_table, return_support=True
            )
            # D4: the value channel speaks with the evidence behind it -
            # a state whose value cells are thin contributes proportionally
            # less to the endpoint cost.
            trust = support / (support + agent.value_trust_samples)
            value_term = -agent.feature_value_weight * trust * value
            if curiosity_weight > 0.0:
                phi_end = _curiosity_potential(
                    agent.evidence_batch(encoded, mask_table)
                )
                curiosity = _curiosity_term(
                    phi_start, phi_end, curiosity_weight, 1.0, horizon,
                )
            else:
                curiosity = np.zeros(len(states))
            return (
                np.column_stack([
                    value_term + curiosity, -dopamine,
                    np.zeros(len(states)), value_term, curiosity,
                    np.zeros(len(states)),
                ]),
                np.isfinite(encoded).all(axis=1),
            )
    else:
        def evaluate_step(states, sequences, step):
            # `lag` is the first predicted state whose acting action comes
            # from the plan rather than the pending prefix, so a plain
            # stride can step over the first point the plan can actually
            # influence.
            previous = (
                previous_states_for(states, step)
                if q_table is not None else None
            )
            if step % step_stride != 0 and not (
                stride_includes_lag and step == lag
            ):
                return (
                    np.zeros((len(states), 6)),
                    np.isfinite(states).all(axis=1),
                )
            stride = _prefix_stride(len(states), step)
            if stride > 1 and q_table is None:
                sub_states = states[::stride]
                history = history_features(
                    sequences[::stride], step + 1, pending_prefix,
                    action_dim=action_dim,
                )
                pending = history[:, :agent.lag][:, ::-1]
                encoded = (
                    np.concatenate([agent_view(sub_states), pending], axis=1)
                    / agent.scales
                )
                risk = agent.pain_signal_batch(encoded) * risk_weight * (
                    risk_decay ** step
                ) * error_factor(step)
                zeros = np.zeros(len(sub_states))
                cols = np.column_stack([
                    risk, zeros, risk, zeros, zeros, zeros,
                ])
                finite = np.isfinite(encoded).all(axis=1)
                return (
                    np.repeat(cols, stride, axis=0),
                    np.repeat(finite, stride, axis=0),
                )
            history = history_features(
                sequences, step + 1, pending_prefix, action_dim=action_dim
            )
            pending = history[:, :agent.lag][:, ::-1]
            encoded = (
                np.concatenate([agent_view(states), pending], axis=1)
                / agent.scales
            )
            risk = agent.pain_signal_batch(encoded) * risk_weight * (
                risk_decay ** step
            ) * error_factor(step)
            q_cost = np.zeros(len(states))
            if q_table is not None:
                previous_history = history_features(
                    sequences, step, pending_prefix, action_dim=action_dim
                )
                previous_pending = previous_history[:, :agent.lag][:, ::-1]
                previous_encoded = np.concatenate(
                    [agent_view(previous), previous_pending], axis=1
                ) / agent.scales
                q_pain, q_count = agent.action_edge_transition_q_batch(
                    previous_encoded, encoded, q_table
                )
                trust = q_count / (
                    q_count + agent.action_edge_trust_samples
                )
                q_cost = action_edge_weight * trust * (
                    q_pain / (1.0 + np.abs(q_pain))
                )
            zeros = np.zeros(len(states))
            return (
                np.column_stack([
                    risk + q_cost, zeros, risk, zeros, zeros, q_cost,
                ]),
                np.isfinite(encoded).all(axis=1),
            )

        def evaluate_terminal(states, sequences):
            history = history_features(
                sequences, horizon, pending_prefix, action_dim=action_dim
            )
            # `history` is newest-first; the node state stores pending
            # actions oldest-first and only the last `lag` actions remain
            # pending.
            pending = history[:, :agent.lag][:, ::-1]
            encoded = (
                np.concatenate([agent_view(states), pending], axis=1)
                / agent.scales
            )
            if isinstance(agent, NodeRuntime) and len(encoded) > 0:
                soft_weights = agent._soft_weights_batch(encoded)
                pain, dopamine = agent.valence_state_batch(
                    encoded, soft_weights=soft_weights
                )
            else:
                soft_weights = None
                pain, dopamine = agent.valence_state_batch(encoded)
            mask_table = agent._feature_mask_table(
                encoded[:, :agent.obs_dim]
            )
            if getattr(agent, "predicate_readouts", False):
                # D3: the terminal hazard comes from the predicates only;
                # the prototype valence no longer bounds it.
                if soft_weights is not None:
                    signal = agent.pain_signal_batch(
                        encoded, mask_table, soft_weights=soft_weights
                    )
                else:
                    signal = agent.pain_signal_batch(encoded, mask_table)
                risk = signal * risk_weight * (risk_decay ** horizon) * (
                    error_factor(horizon - 1)
                )
            else:
                if soft_weights is not None:
                    signal = agent.pain_signal_batch(
                        encoded, mask_table, soft_weights=soft_weights
                    )
                else:
                    signal = agent.pain_signal_batch(encoded, mask_table)
                risk = terminal_hazard(
                    pain,
                    signal,
                    risk_weight,
                    pain_weight=pain_weight,
                    lane=risk_lane,
                ) * (risk_decay ** horizon) * error_factor(horizon - 1)
            # Scalarized objective: accumulated hazard minus the
            # sequence-level value of the endpoint, so a plan that accepts
            # local risk in exchange for a long-survival alignment can win.
            # D4 everywhere: the value channel speaks with the evidence
            # behind it.  Without the trust scale an uncovered endpoint
            # would be credited with the population-average return, so the
            # planner would be paid (or charged) for states nothing was
            # ever learned about.
            if getattr(agent, "value_readout", "cells") == "association":
                value, support = association_value_batch(agent, encoded)
            else:
                value, support = agent.feature_value_batch(
                    encoded, mask_table, return_support=True
                )
            trust_samples = getattr(agent, "value_trust_samples", 150.0)
            trust = support / (support + trust_samples)
            value_term = -agent.feature_value_weight * trust * value
            if curiosity_weight > 0.0:
                phi_end = _curiosity_potential(
                    agent.evidence_batch(encoded, mask_table)
                )
                curiosity = _curiosity_term(
                    phi_start, phi_end, curiosity_weight, 1.0, horizon,
                )
            else:
                curiosity = np.zeros(len(states))
            # Milestone terms.  Two modes, both inert at weight zero:
            #   "goal" - pull the rollout endpoint toward the boundary states
            #     the agent's own stream started milestones in;
            #   "rate" - the epoch-level action commitment: the high level
            #     learned, per milestone identity, the action rate the
            #     behavior policy realised there, and the low level keeps its
            #     full per-frame freedom but is charged for deviating from
            #     that tendency.  It is a direction, not a position, so it
            #     imports no absolute state and no time scale.
            # The commitment is charged per plan step (`mean_t |a_t - rate|`,
            # not `|mean_t a_t - rate|`).  Measured (2026-11-22): every
            # candidate plan's *mean* action sits close to the small learned
            # rate, so the mean-deviation form is nearly constant across
            # candidates and cannot discriminate - it bound the decision on
            # 0-3% of frames.  The per-step form spreads an order of magnitude
            # wider across candidates (0.1 to 0.9 for `rate ~ 0.1`) and is
            # what makes the term a live constraint.
            milestone = np.zeros(len(states))
            if (milestone_weight > 0.0 and milestone_mode == "goal"):
                # The subgoal is a state the transition graph leads to (the
                # successor milestone's entry state), when the caller supplies
                # one; otherwise the modal boundary states of the stream.
                goal = milestone_goal
                if goal is None:
                    goal = getattr(agent, "milestone_goals", None)
                frame = getattr(agent, "milestone_frame", None)
                if goal is not None and frame is not None:
                    # Distance is measured in the identity's own (time-
                    # orthogonal) directions, not over every coordinate:
                    # raw coordinates drag in the offsets that merely track
                    # how far along the cycle we are.  The second direction is
                    # what tells the two sides of an identity apart instead of
                    # collapsing them onto one attractor.
                    obs_dim = agent.obs_dim
                    scales = np.asarray(agent.scales[:obs_dim], dtype=float)
                    center_raw = np.asarray(
                        frame["center"], dtype=float
                    )[:obs_dim]
                    sigma_raw = np.asarray(
                        frame.get("sigma", np.ones(obs_dim)), dtype=float
                    )[:obs_dim]
                    axes = np.asarray(
                        frame.get(
                            "axes", np.asarray([frame["axis"]], dtype=float)
                        ),
                        dtype=float,
                    )[:, :obs_dim]
                    # `encoded` is raw / scales, so the raw-frame projection
                    # becomes (encoded - center/scales) @ (axes*scales/sigma).
                    transform = axes * scales / sigma_raw
                    center_enc = center_raw / scales
                    spread = float(frame["scale"])
                    here = (
                        (encoded[:, :obs_dim] - center_enc) @ transform.T
                    ) / spread
                    raw_goal = np.atleast_2d(
                        np.asarray(goal, dtype=float)
                    )[:, :obs_dim]
                    targets = (
                        ((raw_goal - center_raw) / sigma_raw) @ axes.T
                    ) / spread
                    gap = here[:, None, :] - targets[None, :, :]
                    milestone = -milestone_weight * np.linalg.norm(
                        gap, axis=2
                    ).min(axis=1)
            elif (milestone_weight > 0.0 and milestone_mode == "rate"
                  and milestone_rate is not None):
                deviation = np.abs(
                    sequences - float(milestone_rate)
                ).mean(axis=1)
                milestone = milestone_weight * deviation
            elif (milestone_weight > 0.0
                  and milestone_mode == "track"):
                # Phase-invariant tracking: the target is the successor
                # stage's entry state, but the distance is measured over the
                # observation coordinates that are *not* clocks.  The identity
                # frame already separates those; using all of them (instead of
                # its one-dimensional projection) keeps the gap offsets that
                # the projection throws away.
                goal = milestone_goal
                if goal is None:
                    goal = getattr(agent, "milestone_goals", None)
                frame = getattr(agent, "milestone_frame", None)
                if goal is not None and frame is not None:
                    obs_dim = agent.obs_dim
                    scales = np.asarray(agent.scales[:obs_dim], dtype=float)
                    time_dir = np.asarray(
                        frame.get("time_dir", np.zeros(obs_dim)), dtype=float
                    )
                    keep = np.abs(time_dir) <= 1e-9
                    if not keep.any():
                        keep = np.ones(obs_dim, dtype=bool)
                    target = np.atleast_2d(
                        np.asarray(goal, dtype=float)
                    )[0, :obs_dim] / scales
                    difference = encoded[:, :obs_dim] - target[None, :]
                    milestone = milestone_weight * np.mean(
                        difference[:, keep] ** 2, axis=1
                    )
            elif (milestone_weight > 0.0
                  and milestone_mode == "plan"):
                # Wide, non-frame prediction: the h-step stage walk's
                # expected feedback deficit is a cost the frame rollout
                # cannot see.  It is charged per candidate, by the stage the
                # candidate's own endpoint lands in, so it can change which
                # plan wins; charging the current stage's risk would be a
                # constant across candidates.
                frame = getattr(agent, "milestone_frame", None)
                stage_states = getattr(agent, "milestone_stage_states", None)
                clusters = getattr(agent, "milestone_clusters", None)
                risks = getattr(agent, "milestone_risks", {})
                if (clusters and clusters.get("horizon")
                        and np.asarray(clusters.get("centers", ())).size):
                    # The consistent map: name the endpoint by its own
                    # action-conditioned future, exactly as the training
                    # stages were named.  Falls back to entry geometry only
                    # when the layer was built with the frame identity.
                    raw_endpoints = encoded[:, :agent.obs_dim] * np.asarray(
                        agent.scales[:agent.obs_dim]
                    )
                    endpoint_sig = transition_signatures(
                        raw_endpoints, world.predict, agent.lag,
                        int(clusters["horizon"]),
                    )
                    milestone = milestone_weight * signature_stage_risk(
                        endpoint_sig, clusters["centers"], risks
                    )
                elif (frame is not None and stage_states is not None
                        and len(stage_states)):
                    endpoint_risk = nearest_stage_risk(
                        encoded[:, :agent.obs_dim],
                        stage_states,
                        getattr(agent, "milestone_stage_keys", ()),
                        risks,
                        frame,
                        scales=np.asarray(agent.scales[:agent.obs_dim]),
                    )
                    milestone = milestone_weight * endpoint_risk
            cost = risk + value_term + curiosity + milestone
            if veto_fraction > 0.0:
                # Outcome constraint: the endpoints this decision's readout
                # ranks worst are a bad result and are excluded, rather than
                # adding their (globally high) level to every step.  The
                # cutoff is a within-decision quantile, so only the ranking is
                # used and the readout's absolute scale cannot leak into the
                # objective.
                cutoff = float(np.quantile(
                    signal, 1.0 - float(veto_fraction)
                ))
                margin = float(cost.max() - cost.min()) + 1.0
                cost = cost + np.where(
                    signal >= cutoff, 10.0 * margin, 0.0
                )
            if stage_value is not None and float(stage_value_weight) > 0.0:
                penalties = np.asarray(
                    stage_value(states, sequences), dtype=float,
                ).reshape(-1)
                if penalties.shape != (len(states),):
                    raise ValueError(
                        "stage_value must return one penalty per candidate"
                    )
                if not np.isfinite(penalties).all():
                    raise ValueError(
                        "stage_value penalties must be finite"
                    )
                cost = cost + float(stage_value_weight) * penalties
            return (
                np.column_stack([
                    cost, -dopamine, risk, value_term,
                    curiosity + milestone, np.zeros(len(states)),
                ]),
                np.isfinite(encoded).all(axis=1),
            )

    inquiry_batches = []

    def trace_callback(batch):
        inquiry_batches.append(batch)
        if candidate_callback is not None:
            candidate_callback(batch)

    if int(beam_width) > 0:
        # Long-horizon search at full per-frame action resolution: beam search
        # keeps `2 ** horizon` enumeration out of the picture while leaving the
        # timing of every step free, which is what the reduced candidate sets
        # cannot do.  The only knob is the beam width.
        if death_batch is not None or death_status_batch is not None:
            raise ValueError(
                "beam search does not take a death callback; disable the "
                "death rules or leave --beam-width at 0"
            )
        def trace_frontier(batch):
            if frontier_callback is None:
                return
            payload = dict(batch)
            payload.update(
                frame=int(getattr(env, "t", 0)),
                demand=stage_demand,
                pending_prefix=pending_prefix.copy(),
            )
            frontier_callback(payload)

        search_function = (
            beam_search if int(search_chunk_size) == 0
            else chunked_beam_search
        )
        search_options = dict(
            initial_state=initial,
            action_space=tuple(getattr(agent, "action_space", (0, 1))),
            horizon=horizon,
            transition_batch=predict_batch,
            evaluate_step=evaluate_step,
            evaluate_terminal=evaluate_terminal,
            beam_width=int(beam_width),
            score_combine="sum",
            rank_step=rank_step,
            rank_mode=rank_mode,
            rank_margin=rank_margin,
            rank_group_quota=(
                int(rank_group_quota)
                if int(search_chunk_size) > 0 else 0
            ),
            frontier_callback=(
                trace_frontier if frontier_callback is not None else None
            ),
        )
        if int(search_chunk_size) > 0:
            search_options["chunk_size"] = int(search_chunk_size)
        result = search_function(**search_options)
        return result
    result = exhaustive_search(
        initial_state=initial,
        action_space=tuple(getattr(agent, "action_space", (0, 1))),
        horizon=horizon,
        transition_batch=predict_batch,
        evaluate_step=evaluate_step,
        evaluate_terminal=evaluate_terminal,
        evaluate_death=death_batch,
        evaluate_death_status=death_status_batch,
        unknown_penalty=unknown_penalty,
        near_penalty=near_penalty,
        candidate_callback=(
            trace_callback if inquiry_action is not None
            else candidate_callback
        ),
        batch_size=2 ** horizon,
        # "sum" expands to one mode per objective column, so the ledger
        # columns (hazard and value shares) sum alongside the scalarized
        # cost and the dopamine tie-break without changing their contract.
        score_combine="sum",
    )
    if inquiry_action is not None and inquiry_batches:
        merged = merge_candidate_batches(inquiry_batches)
        # Inquiry is only for the all-unknown case: if any candidate has a
        # death hit, the search already has a preference to act on.
        if np.all(np.asarray(merged["first_death"]) < 0):
            action = inquiry_action(initial, merged)
            if action is not None:
                result = override_first_action(
                    result, inquiry_batches, int(action)
                )
    if margin_eps > 0.0 and previous_action is not None:
        result = _apply_margin_tiebreak(
            result, captured, previous_action, margin_eps
        )
    return result


def stage_schedules(action_space, blocks: int, block_frames: int) -> np.ndarray:
    """Stage-level candidate set: block-wise constant action sequences.

    A stage is a stretch whose internal steps do not matter; only what the
    stretch ends in.  The abstraction is therefore over DURATION, not over
    frames: each candidate holds one action for `block_frames` frames before
    switching, so the set is ``|A| ** blocks`` rather than ``2 ** span`` while
    shrinking the lookahead cost the same way a stage abstraction does.

    Nothing here knows what an action means or which one is "do nothing":
    every action of ``action_space`` is treated symmetrically and the set is
    built identically for any discrete action space with any number of
    actions.  The only inputs are the action alphabet and the block duration,
    which are generic hyperparameters of the same kind as `horizon`.
    """
    actions = [int(action) for action in action_space]
    blocks = int(blocks)
    block_frames = int(block_frames)
    span = blocks * block_frames
    plans = np.zeros((len(actions) ** blocks, span), dtype=float)
    for row, combo in enumerate(itertools.product(actions, repeat=blocks)):
        for index, action in enumerate(combo):
            plans[row, index * block_frames:(index + 1) * block_frames] = action
    return plans


def stage_result(env, world, agent, prefix, blocks=4, block_frames=4, lag=LAG,
                 step_stride=2, risk_weight=1.0) -> SearchResult:
    """Stage-level decision: score stage outcomes, not per-frame plans.

    Mirrors `policy_result`'s objective columns so the ledger contract holds,
    but the candidates are `stage_schedules` evaluated over a horizon the
    per-frame search cannot afford, and the executed action is the first action
    of the best-scoring schedule.  Nothing is read from the task: the actions
    come from the runtime's own `action_space` and the score uses the same
    learned hazard and value readouts as the fine search.
    """
    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)
    risk_weight = float(risk_weight)
    actions = tuple(getattr(agent, "action_space", (0, 1)))
    span = int(blocks) * int(block_frames)
    count = len(actions) ** int(blocks)
    if count > 20000:
        raise ValueError(
            "stage candidate count %d is too large; lower --stage-blocks"
            % count
        )
    plans = stage_schedules(actions, blocks, block_frames)
    states = np.broadcast_to(initial, (count, len(initial))).copy()
    risk_total = np.zeros(count, dtype=float)
    history = None
    for step in range(span):
        history = history_features(
            plans, step + 1, pending_prefix, action_dim=action_dim
        )
        pending = history[:, :agent.lag][:, ::-1]
        encoded = np.concatenate([states, pending], axis=1) / agent.scales
        if step % step_stride == 0:
            risk_total += np.asarray(
                agent.pain_signal_batch(encoded), dtype=float
            )
        states = np.asarray(world.predict(states, history), dtype=float)
    history = history_features(
        plans, span, pending_prefix, action_dim=action_dim
    )
    pending = history[:, :agent.lag][:, ::-1]
    encoded = np.concatenate([states, pending], axis=1) / agent.scales
    pain, dopamine = agent.valence_state_batch(encoded)
    mask_table = agent._feature_mask_table(encoded[:, :agent.obs_dim])
    risk_end = np.maximum(
        np.asarray(pain, dtype=float),
        np.asarray(agent.pain_signal_batch(encoded, mask_table), dtype=float),
    )
    risk = (risk_total + risk_end) * risk_weight
    if getattr(agent, "value_readout", "cells") == "association":
        value, support = association_value_batch(agent, encoded)
    else:
        value, support = agent.feature_value_batch(
            encoded, mask_table, return_support=True
        )
    trust = support / (support + getattr(agent, "value_trust_samples", 150.0))
    value_term = -agent.feature_value_weight * trust * value
    cost = risk + value_term
    finite = np.isfinite(cost) & np.isfinite(value_term)
    if not finite.any():
        raise ValueError("stage search found no finite candidate")
    ranking = np.where(finite, cost, np.inf)
    index = int(np.argmin(ranking))
    score = (
        float(cost[index]),
        0.0,
        float(risk[index]),
        float(value_term[index]),
        0.0,
        0.0,
    )
    return SearchResult(
        actions=np.asarray(plans[index], dtype=float).copy(),
        score=score,
        feasible=True,
        feasible_count=int(finite.sum()),
        candidate_count=count,
        first_state=np.asarray(states[index], dtype=float).copy(),
        final_state=np.asarray(states[index], dtype=float).copy(),
    )




def policy_action(env, world, agent, prefix, horizon, lag=LAG,
                  step_stride=2, stride_includes_lag=False):
    return int(policy_result(
        env, world, agent, prefix, horizon, lag=lag,
        step_stride=step_stride,
        stride_includes_lag=stride_includes_lag,
    ).actions[0])


def cell_q_action(env, world, agent, prefix, lag=LAG):
    """Decide without search, from the learned (feature, action) cells.

    The pending actions are already issued and fixed, so the world model is
    rolled forward `lag` steps on them alone.  The candidate action issued
    now acts `lag` steps from now, on the rolled state; both candidates are
    scored there with the same cells the search reads:

        score(a) = feature hazard(s, a) - w * sequence value(s, a)

    and the lower score wins.  This is the cell-Q-only ablation: it measures
    what the MPC shell adds beyond reading the learned values directly.
    """
    if int(lag) < 0:
        raise ValueError("lag must be non-negative")
    lag = int(lag)
    state = np.asarray(env.obs(), dtype=float).reshape(1, -1)
    pending = list(env.pending()[:lag])
    if len(pending) < lag:
        pending = [0.0] * (lag - len(pending)) + pending
    prefix = np.asarray(prefix, dtype=float).reshape(-1)
    for step in range(lag):
        applied = float(pending[step])
        columns = [applied]
        for offset in range(1, lag + 1):
            index = step - offset
            columns.append(
                float(pending[index]) if index >= 0
                else float(prefix[-index - 1])
            )
        history = np.asarray([columns], dtype=float)
        state = np.asarray(world.predict(state, history), dtype=float)
    scores = {}
    curiosity_weight = float(getattr(agent, "curiosity_weight", 0.0))
    phi_current = 0.0
    if curiosity_weight > 0.0 and lag > 0:
        current_columns = [float(pending[0])] + [
            float(item) for item in pending[1:]
        ]
        encoded_current = np.concatenate(
            [state, np.asarray([current_columns], dtype=float)], axis=1
        ) / agent.scales
        phi_current = float(_curiosity_potential(
            agent.evidence_batch(encoded_current)
        )[0])
    for action in (0, 1):
        columns = [float(action)] + [float(item) for item in pending[1:]]
        encoded = np.concatenate(
            [state, np.asarray([columns], dtype=float)], axis=1
        ) / agent.scales
        risk = float(agent.feature_risk_batch(encoded)[0])
        value = float(agent.feature_value_batch(encoded)[0])
        score = risk - agent.feature_value_weight * value
        if curiosity_weight > 0.0:
            phi_next = _curiosity_potential(
                agent.evidence_batch(encoded)
            )
            score += float(_curiosity_term(
                phi_current, phi_next, curiosity_weight, 1.0, 1,
            )[0])
        scores[action] = score
    return int(min(scores, key=lambda key: (scores[key], key)))


def decision_action(env, world, agent, prefix, horizon, lag=LAG,
                    step_stride=2, stride_includes_lag=False,
                    decision="search", ledger_out=None, risk_weight=1.0,
                    risk_decay=1.0, stage_blocks=4, stage_block_frames=4,
                    beam_width=0, milestone_weight=0.0,
                    milestone_mode="off", milestone_rate=None,
                    milestone_goal=None, pain_weight=None, risk_lane="max",
                    plan_out=None, candidate_callback=None,
                    previous_action=None, margin_eps=0.0, veto_fraction=0.0,
                    rank_step=None, stage_demand=None,
                    demand_heuristic=None, rank_stats=None,
                    rank_mode="order", rank_margin=0.0,
                    stage_value=None, stage_value_weight=0.0,
                    frontier_callback=None, search_chunk_size=0,
                    rank_group_quota=1,
                    chain_chunk_size=2, chain_hops=3, chain_beam_width=2,
                    heuristic_demand=None, heuristic_demand_weight=0.0,
                    adaptive_controller=None, chain_max_hops=5,
                    chain_wide_beam_width=6):
    """Dispatch one decision to the searched or the cell-Q-only policy.

    With `ledger_out` (a list) and the search decision, the winning
    plan's objective ledger - the hazard and value shares of the
    scalarized cost, split by `policy_result`'s third and fourth score
    columns - is appended so the executed decision's components can be
    regressed against its realized outcome (P4 objective learning).  The
    curiosity share (column 4) is a fixed shaping term and is deliberately
    not part of that ledger.

    `decision="stage"` scores block-wise constant action schedules over
    `stage_blocks * stage_block_frames` frames instead of per-frame plans over
    `horizon`: inside an uneventful stretch only the stretch's outcome
    matters, which buys a lookahead the per-frame search cannot afford at a
    fraction of its candidate count.
    """
    if decision != "search" and (
            rank_step is not None or stage_demand is not None
            or demand_heuristic is not None
            or frontier_callback is not None):
        raise ValueError(
            "demand ranking is currently supported only by beam search"
        )
    if decision == "search":
        result = policy_result(
            env, world, agent, prefix, horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
            risk_weight=risk_weight,
            risk_decay=risk_decay,
            beam_width=beam_width,
            milestone_weight=milestone_weight,
            milestone_mode=milestone_mode,
            milestone_rate=milestone_rate,
            milestone_goal=milestone_goal,
            pain_weight=pain_weight,
            risk_lane=risk_lane,
            candidate_callback=candidate_callback,
            previous_action=previous_action,
            margin_eps=margin_eps,
            veto_fraction=veto_fraction,
            rank_step=rank_step,
            stage_demand=stage_demand,
            demand_heuristic=demand_heuristic,
            rank_stats=rank_stats,
            rank_mode=rank_mode,
            rank_margin=rank_margin,
            stage_value=stage_value,
            stage_value_weight=stage_value_weight,
            frontier_callback=frontier_callback,
            search_chunk_size=search_chunk_size,
            rank_group_quota=rank_group_quota,
        )
        if ledger_out is not None:
            score = result.score
            ledger_out.append((
                float(score[2]), float(score[3]),
            ))
        if plan_out is not None:
            # The search's own answer: the state it believes the chosen plan
            # ends in.  Recorded only when asked (diagnostics), so the
            # production path is untouched.
            plan_out.append(dict(
                frame=int(env.t),
                final=np.asarray(result.final_state, dtype=float),
                actions=np.asarray(result.actions, dtype=float),
                cost=float(np.asarray(result.score).reshape(-1)[0]),
                objective_score=np.asarray(
                    result.score, dtype=float
                ).reshape(-1).tolist(),
                rank_credit=float(
                    getattr(result, "rank_priority", 0.0)
                ),
                candidate_count=int(
                    getattr(result, "candidate_count", 0)
                ),
            ))
        return int(result.actions[0])
    if decision == "chained":
        if adaptive_controller is not None:
            result = adaptive_chained_result(
                env, world, agent, prefix,
                chunk_size=chain_chunk_size,
                base_hops=chain_hops,
                max_hops=chain_max_hops,
                beam_width=chain_beam_width,
                wide_beam_width=chain_wide_beam_width,
                lag=lag,
                step_stride=step_stride,
                risk_weight=risk_weight,
                stage_value=stage_value,
                stage_value_weight=stage_value_weight,
                heuristic_demand=heuristic_demand,
                heuristic_demand_weight=heuristic_demand_weight,
                controller=adaptive_controller,
            )
        else:
            result = chained_shallow_result(
                env, world, agent, prefix,
                chunk_size=chain_chunk_size,
                n_hops=chain_hops,
                beam_width=chain_beam_width,
                lag=lag,
                step_stride=step_stride,
                risk_weight=risk_weight,
                stage_value=stage_value,
                stage_value_weight=stage_value_weight,
                heuristic_demand=heuristic_demand,
                heuristic_demand_weight=heuristic_demand_weight,
            )
        if ledger_out is not None:
            score = result.score
            ledger_out.append((float(score[2]), float(score[3])))
        return int(result.actions[0])
    if decision == "stage":
        result = stage_result(
            env, world, agent, prefix, blocks=stage_blocks,
            block_frames=stage_block_frames, lag=lag,
            step_stride=step_stride, risk_weight=risk_weight,
        )
        if ledger_out is not None:
            score = result.score
            ledger_out.append((float(score[2]), float(score[3])))
        return int(result.actions[0])
    if decision == "cellq":
        if ledger_out is not None:
            ledger_out.append((0.0, 0.0))
        return cell_q_action(env, world, agent, prefix, lag=lag)
    raise ValueError("unknown decision mode %r" % decision)


def plan_score(initial_state, world, agent, prefix, plan, horizon, lag=LAG,
               step_stride=2, stride_includes_lag=False):
    """Score one fixed plan with the same objective as `policy_result`.

    Reproduces the `exhaustive_search` scoring under
    ``score_combine=("sum", "sum")``.  Under the default objective column 0
    is the accumulated `pain_signal` hazard over the strided predicted
    states plus the terminal risk, minus the sequence-level value of the
    endpoint; under the contrast objective it is the accumulated action
    contrast minus the value term.  Column 1 is the negative terminal
    dopamine.  With a non-zero curiosity weight the scalar also carries
    ``w * (濠?s_0) - 濠?s_H))`` (the potential-difference shaping term).
    Used by diagnostics that compare fixed plans against the
    searched plan, and later by plan-level posteriors.
    """
    initial = np.asarray(initial_state, dtype=float).reshape(1, -1)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)
    lag = int(lag)
    plan = np.asarray(plan, dtype=float).reshape(1, -1)
    curiosity_weight = float(getattr(agent, "curiosity_weight", 0.0))
    phi_start = 0.0
    if curiosity_weight > 0.0:
        pending_start = (
            pending_prefix[:lag][::-1] if lag > 0
            else np.zeros(0, dtype=float)
        )
        encoded_start = np.concatenate(
            [initial, pending_start.reshape(1, -1)], axis=1
        ) / agent.scales
        phi_start = float(
            _curiosity_potential(agent.evidence_batch(encoded_start))[0]
        )
    states = initial.copy()
    total = 0.0
    contrast = getattr(agent, "contrast_objective", False)
    for step in range(horizon):
        history = history_features(
            plan, step + 1, pending_prefix, action_dim=action_dim
        )
        states = np.asarray(world.predict(states, history), dtype=float)
        if step % step_stride == 0 or (
            stride_includes_lag and step == lag
        ):
            pending = history[:, :agent.lag][:, ::-1]
            encoded = np.concatenate([states, pending], axis=1) / agent.scales
            if not np.isfinite(encoded).all():
                return (float("inf"), float("inf"))
            if contrast:
                threat, excess = agent.pain_graded_batch(encoded)
                total += threat + agent.control_risk_weight * excess
            else:
                total += float(agent.pain_signal_batch(encoded)[0])
    history = history_features(
        plan, horizon, pending_prefix, action_dim=action_dim
    )
    pending = history[:, :agent.lag][:, ::-1]
    encoded = np.concatenate([states, pending], axis=1) / agent.scales
    if not np.isfinite(encoded).all():
        return (float("inf"), float("inf"))
    pain, dopamine = agent.valence_state_batch(encoded)
    mask_table = agent._feature_mask_table(encoded[:, :agent.obs_dim])
    curiosity = 0.0
    if curiosity_weight > 0.0:
        phi_end = _curiosity_potential(
            agent.evidence_batch(encoded, mask_table)
        )
        curiosity = float(_curiosity_term(
            phi_start, phi_end, curiosity_weight, 1.0, horizon,
        )[0])
    if contrast:
        value, support = agent.feature_value_batch(
            encoded, mask_table, return_support=True
        )
        trust = support / (support + agent.value_trust_samples)
        total -= agent.feature_value_weight * trust * float(value[0])
        total += curiosity
        return (total, -float(dopamine[0]))
    if getattr(agent, "predicate_readouts", False):
        terminal = float(agent.pain_signal_batch(encoded, mask_table)[0])
    else:
        terminal = max(
            float(pain[0]),
            float(agent.pain_signal_batch(encoded, mask_table)[0]),
        )
    value = float(agent.feature_value_batch(encoded, mask_table)[0])
    total += terminal - agent.feature_value_weight * value
    total += curiosity
    return (total, -float(dopamine[0]))



from trainer import (
    _encoded_pending,
    _graded_holdout_loss,
    _milestone_goals_from_transitions,
    _milestone_model_from_transitions,
    _rankdata,
    _rollout_error_curve,
    _spearman,
    _stage_evidence_for_streams,
    _train,
    collect_planner,
    fit,
    learn_objective_weight,
    ledger_survival_report,
    value_dispersion_labels,
)
from evaluator import _curiosity_off, evaluate, run_seed
from cli import main


if __name__ == "__main__":
    main()
