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
    exhaustive_search,
    merge_candidate_batches,
    override_first_action,
)

LAG = 3
DEFAULT_JOBS = min(8, os.cpu_count() or 1)
FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()


def value_dispersion_labels(returns, quantile: float) -> np.ndarray:
    """Labels for the decision-critical regions of a return distribution.

    A state matters for decisions when its outcome is far from typical on
    either side: the top quantile (the aligned, high-return continuation)
    or the bottom quantile (the dying one).  Selecting predicates on the
    union tiles both faces of the regions where actions change the
    outcome, instead of only the cruising half of the high-return mass.
    """
    returns = np.asarray(returns, dtype=float)
    high_cut = float(np.quantile(returns, quantile))
    low_cut = float(np.quantile(returns, 1.0 - quantile))
    return ((returns > high_cut) | (returns < low_cut)).astype(float)


def _graded_holdout_loss(agent, enc: np.ndarray, labels: np.ndarray,
                         mask: np.ndarray, exclude=()) -> Optional[float]:
    """Mean absolute error of the graded threat read on held-out rows.

    The loss the structure generations must not worsen: the calibrated
    threat read against the realized window labels, over the held-out
    rows where some root fires.  `exclude` temporarily withdraws roots
    from the readout (a reversible attribute swap, so the same rows and
    calibration are read with and without a candidate generation's
    roots) - that is what makes the backfit test exact.
    """
    saved = list(getattr(agent, "feature_ids", ()))
    if not saved:
        return None
    if exclude:
        dropped = set(exclude)
        agent.feature_ids = [fid for fid in saved if fid not in dropped]
    try:
        threat, _excess = agent.pain_graded_batch(enc)
    finally:
        agent.feature_ids = saved
    scope = (threat > 0.0) & mask
    if not scope.any():
        return None
    return float(np.mean(np.abs(threat[scope] - labels[scope])))


def _encoded_pending(rows, lag: int) -> np.ndarray:
    """The pending-action window the runtime's own encoder pairs with each row.

    `encode` builds a state from the actions issued *before* it (the
    pending deque is read before the current action is appended), oldest
    first.  For a row whose history is ``[a_t, a_{t-1}, ..., a_{t-lag}]``
    (most recent first) that window is ``history[1:lag+1][::-1]``.  Using
    ``history[:lag][::-1]`` instead would pair every offline readout with
    the encoding of the *next* row's state - a one-row shift that silently
    misaligns calibration and residual scans.
    """
    window = np.zeros((len(rows), int(lag)), dtype=float)
    for index, row in enumerate(rows):
        history = np.asarray(row["history"], dtype=float).reshape(-1)
        take = min(int(lag), max(history.size - 1, 0))
        if take:
            window[index, int(lag) - take:] = history[1:1 + take][::-1]
    return window


def _train(transitions, seed, epochs, lag, max_nodes, birth_threshold,
           component_kinds, component_min_count, max_components,
           component_weight, component_gating, memory_weight,
           association_rate, induce_features, precursor_labels,
           pain_horizon, voi_target, feature_value_weight,
           value_horizon=None, ood_trust_scale=0.0, obs_trust=1.0,
           feature_action_prior=2.0, feature_value_prior=4.0,
           value_features=True, value_label_quantile=0.8,
           subgraph_cells=False, predicate_readouts=False,
           event_memory_risk=True, contrast_objective=False,
           value_trust_samples: float = 150.0, control_risk_weight: float = 0.0,
        value_bank_only: bool = False,
        value_dispersion=False, iterative_induction=False,
        compose_readout: bool = True, importance_weighting=False,
        voi_threat_min: float = 0.25, curiosity_weight: float = 0.0,
        dagger_frames: int = 0, memory_credit: bool = False,
        rule_min_coverage: float = 0.0,
        rule_min_primitive_coverage: float = 0.0,
        rule_coverage_x=None, pairwise_features: bool = False,
        branch_transitions: bool = False,
        action_edge_weight: float = 0.0,
        feature_backfit: bool = False,
        risk_error_weight: bool = False,
        memory_feature_backfill: bool = False,
        pain_gamma: float = None,
        action_edge_trust_samples: float = 50.0,
        death_rules: bool = False,
        death_rule_compose: bool = False,
        death_rule_max_thresholds: int = 256,
        death_rule_max_rules: int = 24,
        death_rule_max_literals: int = 5,
        death_rule_boundary_margin: float = 0.0,
        death_rule_envelope: bool = False,
        readout_propagation_gain: float = 0.0,
        readout_propagation_steps: int = 1,
        readout_propagation_edges: str = "transition"):
    """Fit the world model and the node runtime on a transition stream."""
    obs_dim = len(transitions[0]["obs"])
    action_dim = lag + 1
    world = LinearObservationModel(obs_dim, action_dim).fit(
        [row["obs"] for row in transitions],
        [row["history"] for row in transitions],
        [row["next_obs"] for row in transitions],
    )
    observations = np.asarray(
        [row["obs"] for row in transitions], dtype=float
    )
    # Scale only from observed state spread; no task-specific constants.
    observation_scale = observations.std(axis=0)
    observation_scale = np.where(
        observation_scale > 1e-8, observation_scale, 1.0
    )
    scales = np.concatenate([observation_scale, np.ones(lag, dtype=float)])
    coverage_scaled = None
    if rule_coverage_x is not None:
        coverage_scaled = (
            np.asarray(rule_coverage_x, dtype=float)
            / scales[:obs_dim]
        )
    agent = NodeRuntime(
        obs_dim=obs_dim, action_space=(0, 1), lag=lag,
        birth_threshold=birth_threshold, max_nodes=max_nodes,
        exploration=0.0, seed=seed, scales=scales,
        component_kinds=component_kinds,
        component_min_count=component_min_count,
        max_components=max_components,
        component_weight=component_weight,
        component_gating=component_gating,
        memory_weight=memory_weight,
        association_rate=association_rate,
        pain_horizon=pain_horizon,
        voi_target=voi_target,
        feature_value_weight=feature_value_weight,
        value_horizon=(value_horizon or None),
        pain_gamma=pain_gamma,
        ood_trust_scale=ood_trust_scale,
        obs_trust=obs_trust,
        feature_action_prior=feature_action_prior,
        feature_value_prior=feature_value_prior,
        subgraph_cells=subgraph_cells,
        predicate_readouts=predicate_readouts,
        event_memory_risk=event_memory_risk,
        contrast_objective=contrast_objective,
        value_trust_samples=value_trust_samples,
        control_risk_weight=control_risk_weight,
        value_bank_only=value_bank_only,
        importance_weighting=importance_weighting,
        voi_threat_min=voi_threat_min,
        curiosity_weight=curiosity_weight,
        point_labels=(not precursor_labels),
        memory_credit=memory_credit,
        rule_min_coverage=rule_min_coverage,
        rule_min_primitive_coverage=rule_min_primitive_coverage,
        pairwise_features=pairwise_features,
        branch_transitions=branch_transitions,
        action_edge_weight=action_edge_weight,
        action_edge_trust_samples=action_edge_trust_samples,
        readout_propagation_gain=readout_propagation_gain,
        readout_propagation_steps=readout_propagation_steps,
        readout_propagation_edges=readout_propagation_edges,
    )
    episodes = {}
    for row in transitions:
        episodes.setdefault(row["episode"], []).append(row)
    ordered = [episodes[key] for key in sorted(episodes)]

    if risk_error_weight:
        # Data-driven per-step trust: the planner conditions on a known
        # action sequence, so measure how far the fitted model drifts on
        # the real stream and down-weight the hazard read accordingly.
        agent.risk_error_weight = True
        agent.step_errors = _rollout_error_curve(
            transitions, world, lag
        )

    # Rules are induced after the first pass, then materialized as feature
    # nodes.  With point-event labels (default) the predicate describes the
    # state the action was taken from and predicts the entered state's own
    # pain; the opt-in precursor path instead uses a window.  Later epochs
    # train transitions, action edges and memories with those feature nodes
    # active, so features take part in learning rather than only in the
    # final readout.
    long_window = None
    if precursor_labels:
        window = int(pain_horizon) if pain_horizon else lag + 1
        long_window = window if window != lag + 1 else None
        states = np.asarray(
            [row["obs"] for row in transitions], dtype=float
        ) / scales[:obs_dim]
        pain = np.asarray(
            future_pain_labels(transitions, window), dtype=float
        )
        if long_window is not None:
            pain_short = np.asarray(
                future_pain_labels(transitions, lag + 1), dtype=float
            )
        else:
            pain_short = pain
    else:
        # Point-event pain: the world only reports pain on the transition
        # that ended in a collision.  The predicate describes the state the
        # action was taken from, and its label is the entered state's own
        # feedback - one step, no precursor window.
        long_window = None
        states = np.asarray(
            [row["obs"] for row in transitions], dtype=float
        ) / scales[:obs_dim]
        pain = np.asarray(
            [row["signals"][1] > 0.5 for row in transitions], dtype=float
        )
        pain_short = pain
    groups = np.asarray([row["episode"] for row in transitions])
    order = np.unique(groups)
    # Interleaved episode split: every fifth episode is held out.  A contiguous
    # tail split can leave rare causes entirely in train, so their rules can
    # never be verified.
    episode_index = np.searchsorted(order, groups)
    test = (episode_index % 5 == 4)
    train = ~test
    # Structure generations pending their held-out backfit (P3): the roots
    # a gap generation installed, tested against the exact counterfactual
    # at the next generation and rolled back if they made the held-out
    # readout worse.
    pending_roots: List[str] = []
    agent.generation_log = []

    for epoch in range(epochs):
        for episode in ordered:
            if feature_backfit and epoch > 0:
                # Feature-only replay: prototypes, edges and memories keep
                # their single-pass statistics.
                agent.replay_features(episode)
            else:
                agent.learn_episode(episode)
        if induce_features and epoch == 0:
            primary = pain_short if long_window is not None else pain
            agent.induce_rules(
                states[train], primary[train], states[test], primary[test],
                coverage_states=coverage_scaled,
            )
            # The primary bank is selected on the short-window labels, so
            # its roots must train and read with that same window: an
            # untagged root would inherit the runtime's configured window
            # and train on longer labels than its selection used - and a
            # longer window would also push it out of the graded readout.
            agent.install_feature_rules(horizon=lag + 1)
            if long_window is not None:
                bank = FeatureRuleBank(
                    min_coverage=rule_min_coverage,
                    min_primitive_coverage=rule_min_primitive_coverage,
                    pairwise_features=pairwise_features,
                )
                bank.induce(
                    states[train], pain[train], states[test], pain[test],
                    coverage_x=coverage_scaled,
                )
                agent.rules_long = bank
                agent.install_feature_rules(bank.rules, horizon=long_window)
            # Value-channel bank: the same grammar, but the labels are the
            # top of the empirical discounted-return distribution instead
            # of pain.  The selected predicates tile the states whose value
            # cells carry data - regions the pain labels never cover - so
            # the sequence-level value readout gains its own resolution.
            # Same transitions, no extra collection.
            if value_features:
                returns = np.asarray(
                    future_return_labels(transitions, agent.gamma),
                    dtype=float,
                )
                if value_dispersion:
                    high = value_dispersion_labels(
                        returns, value_label_quantile
                    )
                else:
                    cut = float(np.quantile(returns, value_label_quantile))
                    high = (returns > cut).astype(float)
                if high.sum() >= 2:
                    if not precursor_labels:
                        value_states = np.asarray(
                            [row["obs"] for row in transitions], dtype=float
                        ) / scales[:obs_dim]
                    else:
                        value_states = states
                    value_bank = FeatureRuleBank(
                    min_coverage=rule_min_coverage,
                    min_primitive_coverage=rule_min_primitive_coverage,
                    pairwise_features=pairwise_features,
                )
                    value_bank.induce(
                        value_states[train], high[train],
                        value_states[test], high[test],
                        coverage_x=coverage_scaled,
                    )
                    if value_bank.rules:
                        agent.rules_value = value_bank
                        agent.install_feature_rules(
                            value_bank.rules, channel="value"
                        )
            if memory_feature_backfill:
                # Memories predate the induced features; link them so the
                # memory risk channel has a feature-membership read path.
                agent.backfill_memory_features()
        elif induce_features and iterative_induction and (
            1 <= epoch < epochs - 1
        ):
            # F5: gap-focused re-induction - the structure search runs
            # every generation, and the selection pressure is the previous
            # generation's residual coverage.  The criterion is
            # rank-based, not level-based: the deployed readout's
            # max-combination reads are inflated well above the empirical
            # danger rates, so a level threshold would mark nothing as
            # missed; a positive counts as unexplained when its graded
            # threat read ranks below half of the negatives' reads.
            pending_all = _encoded_pending(transitions, lag)
            enc_all = np.concatenate([states, pending_all], axis=1)
            if contrast_objective:
                # The gap generations select against read levels, so the
                # monotone calibration must be current: refit it on the
                # stream with the cells as they stand now (online
                # refresh), then measure the missed positives against the
                # calibrated read.  Fitting once at the end of training
                # would leave every generation selecting against the
                # inflated levels the calibration exists to correct.
                agent.fit_readout_calibration(
                    states, pending_all, pain_short
                )
            if pending_roots:
                # LS_Boost backfitting: the previous generation stays
                # only if the held-out loss is better with its roots
                # than without them - the counterfactual is exact, the
                # same rows and calibration read with and without.
                loss_with = _graded_holdout_loss(
                    agent, enc_all, pain_short, test
                )
                loss_without = _graded_holdout_loss(
                    agent, enc_all, pain_short, test, exclude=pending_roots
                )
                rolled_back = (
                    loss_with is None or loss_without is None
                    or loss_with > loss_without
                )
                if rolled_back:
                    agent.uninstall_feature_rules(pending_roots)
                agent.generation_log.append(dict(
                    epoch=epoch,
                    tested=len(pending_roots),
                    loss_with=loss_with,
                    loss_without=loss_without,
                    rolled_back=bool(rolled_back),
                ))
                pending_roots = []
            threat_read, _excess_read = agent.pain_graded_batch(enc_all)
            negative_levels = threat_read[pain_short <= 0.5]
            cut = float(np.median(negative_levels)) if len(
                negative_levels
            ) else 0.5
            missed_short = (pain_short > 0.5) & (threat_read <= cut)
            if missed_short.sum() >= 5:
                gap_bank = FeatureRuleBank(
                    min_coverage=rule_min_coverage,
                    min_primitive_coverage=rule_min_primitive_coverage,
                    pairwise_features=pairwise_features,
                )
                gap_bank.induce(
                    states[train], (pain_short * missed_short)[train],
                    states[test], (pain_short * missed_short)[test],
                    coverage_x=coverage_scaled,
                )
                if gap_bank.rules:
                    before = set(agent.feature_ids)
                    agent.install_feature_rules(
                        gap_bank.rules, horizon=lag + 1
                    )
                    pending_roots = [
                        fid for fid in agent.feature_ids if fid not in before
                    ]
            missed_long = (pain > 0.5) & (threat_read <= cut)
            if missed_long.sum() >= 5 and long_window is not None:
                gap_long = FeatureRuleBank(
                        min_coverage=rule_min_coverage,
                        min_primitive_coverage=rule_min_primitive_coverage,
                        pairwise_features=pairwise_features,
                    )
                gap_long.induce(
                    states[train], (pain * missed_long)[train],
                    states[test], (pain * missed_long)[test],
                    coverage_x=coverage_scaled,
                )
                if gap_long.rules:
                    before = set(agent.feature_ids)
                    agent.install_feature_rules(
                        gap_long.rules, horizon=long_window
                    )
                    pending_roots.extend(
                        fid for fid in agent.feature_ids if fid not in before
                    )
    if (
        compose_readout and pain_short is not None and len(states)
        and getattr(agent, "feature_ids", ())
    ):
        # P2: the stacked non-negative composition replaces the
        # hand-written max over correlated channels - its fixed point
        # is E[max of correlated estimates], structurally above the
        # truths, and it is exactly the default objective's readout.
        # Fitted to the short-window labels (the same family the
        # channels train on), then monotone-mapped to empirical
        # rates.  The report records the fit's own error against the
        # historical max for the trace.
        pending_final = _encoded_pending(transitions, lag)
        report = agent.fit_hazard_composition(
            states, pending_final, pain_short
        )
        if report:
            agent.composition_report = report
    if contrast_objective:
        # F6: fit the monotone readout calibration on the training stream
        # itself, so the graded threat read is an empirical pain rate
        # rather than an inflated combination artifact - the objective's
        # accounting and the planner's comparisons recover their meaning.
        pending_final = _encoded_pending(transitions, lag)
        agent.fit_readout_calibration(states, pending_final, pain_short)
    if pending_roots and contrast_objective:
        # The last generation never met a backfit test inside the loop;
        # run it once against the final calibration.
        pending_final = _encoded_pending(transitions, lag)
        enc_all = np.concatenate([states, pending_final], axis=1)
        loss_with = _graded_holdout_loss(agent, enc_all, pain_short, test)
        loss_without = _graded_holdout_loss(
            agent, enc_all, pain_short, test, exclude=pending_roots
        )
        rolled_back = (
            loss_with is None or loss_without is None
            or loss_with > loss_without
        )
        if rolled_back:
            agent.uninstall_feature_rules(pending_roots)
        agent.generation_log.append(dict(
            epoch=epochs - 1,
            tested=len(pending_roots),
            loss_with=loss_with,
            loss_without=loss_without,
            rolled_back=bool(rolled_back),
        ))
    if death_rules:
        # The 0/1 death condition is trained on the entered states.  It
        # stays a pure rule set: no probability enters the planner.
        from death_rules import DeathRuleSet
        entered_states = np.asarray(
            [row["next_obs"] for row in transitions], dtype=float
        )
        terminal_labels = np.asarray(
            [
                1 if float(row["signals"][1]) > 0.5 else 0
                for row in transitions
            ],
            dtype=int,
        )
        rule_set = DeathRuleSet(
            max_rules=int(death_rule_max_rules),
            max_literals=int(death_rule_max_literals),
            min_support=2,
            max_thresholds=int(death_rule_max_thresholds),
            compose=bool(death_rule_compose),
            coverage_first=True,
            boundary_margin=float(death_rule_boundary_margin),
        ).fit(entered_states, terminal_labels)
        agent.death_rules = rule_set
        agent.death_rule_hard = True
        # Opt-in model-error envelope: the boundary margin that already
        # keeps the safe side from claiming safety can also turn a
        # near-boundary state into a death hit.  Exact death remains the
        # default.
        agent.death_rule_envelope = bool(death_rule_envelope)
        agent.death_rule_summary = dict(
            rules=[rule.text() for rule in rule_set.rules_],
            modes=[
                dict(
                    signature=list(mode["signature"]),
                    rules=[rule.text() for rule in mode["rules"]],
                )
                for mode in rule_set.modes(1)
            ],
            verification=rule_set.verify(
                entered_states, terminal_labels
            ),
            coverage=rule_set.coverage_,
        )
    return world, agent


def collect_planner(seed, world, agent, episodes=2, cap=1200, horizon=10,
                    lag=LAG, step_stride=2, stride_includes_lag=False,
                    offset=50000, max_frames=None, risk_weight=1.0,
                    risk_decay=1.0, env_factory=None):
    """Record the trajectories the planner actually executes.

    These are real transitions with real world feedback, in the same schema
    as `collect`; no counterfactual prediction enters the stream.  The world
    seeds are offset from the behavior stream so episode ids cannot collide.
    With `max_frames`, collection stops as soon as that many executed frames
    have been recorded (the last episode may overshoot); `episodes` stays
    the hard limit that bounds the number of world resets.
    """
    transitions = []
    frames = []
    total = 0
    limit = max_frames if max_frames is not None else None
    for episode in range(int(episodes)):
        env_seed = int(seed) * 10000 + int(offset) + episode
        length, _actions, rows = evaluate(
            seed, world, agent, cap=cap, horizon=horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
            env_seed=env_seed, record=True,
            episode=int(offset) + episode,
            risk_weight=risk_weight,
            risk_decay=risk_decay,
            env_factory=env_factory,
        )
        transitions.extend(rows)
        frames.append(int(length))
        total += int(length)
        if limit is not None and total >= int(limit):
            break
    return transitions, frames


def _rankdata(values) -> np.ndarray:
    """Average ranks (1-based, ties averaged) - enough for Spearman."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average = 0.5 * (index + end - 1) + 1.0
        for position in range(index, end):
            ranks[order[position]] = average
        index = end
    return ranks


def _spearman(first, second) -> float:
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    if first_rank.size < 2:
        return 0.0
    first_dev = first_rank - first_rank.mean()
    second_dev = second_rank - second_rank.mean()
    denominator = np.sqrt(
        (first_dev * first_dev).sum() * (second_dev * second_dev).sum()
    )
    if denominator <= 0.0:
        return 0.0
    return float((first_dev * second_dev).sum() / denominator)


def learn_objective_weight(rows, min_decisions: int = 8) -> Optional[float]:
    """Select the objective's value weight from the planner's own outcomes.

    P4: the trade-off between one unit of accumulated hazard and one unit
    of endpoint value - the objective's hand-tuned scalarization - is
    measured from the executed decisions: every decision row carries its
    plan's ledger (the hazard share and the value share of the
    scalarized cost) and its realized remaining survival.  For each
    candidate weight the scalarized cost rebuilds as
    ``hazard + weight * value_share`` (the value share already carries
    the trust scale, and scales linearly in the weight), and the weight
    whose rebuilt cost ranks survival best across decisions wins.
    Candidate weights are scale-free: geometric around the ratio of the
    data's own median hazard and value magnitudes, plus zero (an honest
    answer when the value channel does not predict survival).  Returns
    None when the executed data cannot rank the candidates.
    """
    hazards = []
    values = []
    outcomes = []
    last_index = {}
    for index, row in enumerate(rows):
        ledger = row.get("ledger")
        if not ledger:
            continue
        last_index[row["episode"]] = index
    for index, row in enumerate(rows):
        ledger = row.get("ledger")
        if not ledger:
            continue
        hazards.append(float(ledger[0]))
        values.append(float(ledger[1]))
        outcomes.append(float(last_index[row["episode"]] - index + 1))
    if len(hazards) < min_decisions or len(set(outcomes)) < 2:
        return None
    hazards = np.asarray(hazards, dtype=float)
    values = np.asarray(values, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    hazard_scale = float(np.median(np.abs(hazards)))
    value_scale = float(np.median(np.abs(values)))
    center = hazard_scale / max(value_scale, 1e-9)
    grid = [0.0] + [center * factor for factor in (0.25, 0.5, 1.0, 2.0, 4.0)]
    best_weight = None
    best_agreement = -np.inf
    for weight in grid:
        rebuilt = -(hazards + weight * values)
        agreement = _spearman(rebuilt, outcomes)
        if agreement > best_agreement:
            best_agreement = agreement
            best_weight = float(weight)
    return best_weight


def ledger_survival_report(rows, weight: float,
                           min_decisions: int = 8) -> Optional[dict]:
    """Monotone regression of each ledger component on realized survival.

    P4 diagnostic: the objective's components are supposed to predict
    what actually happens - a plan whose ledger says "expensive" should
    not outlive one whose ledger says "cheap".  For every executed
    decision this reports the rank agreement of each component with the
    realized remaining survival (the hazard share and the value share
    should both correlate negatively with survival: more hazard and more
    value both mean a lower scalarized cost) and the agreement of the
    scalarized cost at the objective's current weight.  A component near
    zero is not predicting survival - exactly the misfit that flags an
    objective term worth re-learning or dropping.
    """
    hazards = []
    values = []
    outcomes = []
    last_index = {}
    for index, row in enumerate(rows):
        if row.get("ledger"):
            last_index[row["episode"]] = index
    for index, row in enumerate(rows):
        ledger = row.get("ledger")
        if not ledger:
            continue
        hazards.append(float(ledger[0]))
        values.append(float(ledger[1]))
        outcomes.append(float(last_index[row["episode"]] - index + 1))
    if len(hazards) < min_decisions or len(set(outcomes)) < 2:
        return None
    hazards = np.asarray(hazards, dtype=float)
    values = np.asarray(values, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return {
        "decisions": len(hazards),
        "hazard_agreement": _spearman(-hazards, outcomes),
        "value_agreement": _spearman(values, outcomes),
        "scalar_agreement": _spearman(
            -(hazards + float(weight) * values), outcomes
        ),
    }


def _rollout_error_curve(transitions, world, lag, horizon=16,
                         max_windows=200):
    """Per-step rollout error of the fitted world model on the real stream.

    The planner conditions on a known action sequence, so the model is
    rolled forward with the recorded actions and compared with the recorded
    next observations at each step.  The curve is in observation-standard-
    deviation units and is used to down-weight the hazard read at steps the
    model cannot predict (discrete events, coordinate resets).
    """
    observations = np.asarray(
        [row["obs"] for row in transitions], dtype=float
    )
    scale = observations.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    episodes = {}
    for row in transitions:
        episodes.setdefault(row["episode"], []).append(row)
    errors = [[] for _ in range(horizon)]
    windows = 0
    for rows in episodes.values():
        start = 0
        while (start + horizon < len(rows)
               and windows < max_windows):
            chunk = rows[start:start + horizon]
            if any(bool(row["done"]) for row in chunk):
                start += 1
                continue
            state = np.asarray(chunk[0]["obs"], dtype=float)
            for step in range(horizon):
                history = np.asarray(
                    chunk[step]["history"], dtype=float
                ).reshape(1, -1)
                predicted = world.predict(
                    state.reshape(1, -1), history
                )[0]
                actual = np.asarray(chunk[step]["next_obs"], dtype=float)
                errors[step].append(float(np.linalg.norm(
                    (predicted - actual) / scale
                )))
                state = predicted
            windows += 1
            start += 4
    return [
        float(np.mean(values)) if values else 0.0 for values in errors
    ]


def fit(seed, steps=5000, coverage=0.1, max_nodes=5000,
        birth_threshold=0.45, epochs=5, lag=LAG,
        component_kinds=(), component_min_count=8,
        max_components=128, component_weight=0.5,
        component_gating=False, memory_weight=0.1,
        association_rate=0.0, induce_features=True,
        precursor_labels=False, pain_horizon=None,
        voi_fraction=0.0, voi_target=3, feature_value_weight=0.02,
        voi_threat_min=0.25,
        dagger_rounds=0, dagger_episodes=2, dagger_cap=1200,
        dagger_offset=50000, dagger_horizon=10, dagger_frames=0, step_stride=2,
        stride_includes_lag=False, value_horizon=0,
        ood_trust_scale=0.0, obs_trust=1.0,
        feature_action_prior=2.0, feature_value_prior=4.0,
        value_features=True, value_label_quantile=0.8,
        subgraph_cells=False, predicate_readouts=False,
        event_memory_risk=True, contrast_objective=False,
        value_trust_samples=150.0, value_dispersion=False,
        control_risk_weight=0.0, value_bank_only=False,
        iterative_induction=False, compose_readout=True,
        importance_weighting=False, learn_objective=True,
        curiosity_weight=0.5, memory_credit=False,
        rule_min_coverage=0.0, rule_min_primitive_coverage=0.0,
        rule_validation_seed=0, rule_validation_frames=600,
        pairwise_features=False, branch_transitions=False,
        action_edge_weight=0.0, action_edge_trust_samples=50.0,
        feature_backfit=False,
        risk_weight=1.0,
        risk_decay=1.0,
        risk_error_weight=False,
        memory_feature_backfill=False,
        pain_gamma=None,
        death_rules=False, death_rule_compose=False,
        death_rule_max_thresholds=256, death_rule_max_rules=24,
        death_rule_max_literals=5, death_rule_boundary_margin=0.0,
        death_rule_envelope=False,
        readout_propagation_gain=0.0,
        readout_propagation_steps=1,
        readout_propagation_edges="transition",
        milestone_stage="strength",
        milestone_boundary_k=3.0,
        hazard_readout="calibrated",
        value_readout="cells",
        stage_memory=False,
        stage_memory_quantile=0.98,
        stage_memory_feature_source="branch",
        stage_demand_min_support=2,
        stage_demand_min_episodes=2,
        env_factory=None):
    """Collect transitions and train the node runtime on them.

    With `voi_fraction > 0` collection runs in two stages: the first stage
    bootstraps the runtime with the plain behavior policy, the second stage
    asks that bootstrap agent where its (feature, action) support is
    deficient and overrides at most `voi_fraction * steps` actions with the
    less-supported one.  The override is additive - the behavior sampler
    still runs the rest of the time - and everything is retrained on the
    combined stream.

    With `dagger_rounds > 0` the planner's own executed trajectories are
    appended after each round and the runtime is retrained from scratch on
    the combined real-transition stream.  Only executed actions and their
    observed feedback are recorded.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    if float(readout_propagation_gain) < 0.0:
        raise ValueError("readout_propagation_gain must be non-negative")
    if int(readout_propagation_steps) < 1:
        raise ValueError("readout_propagation_steps must be positive")
    if str(readout_propagation_edges) not in (
        "transition", "association", "both"
    ):
        raise ValueError(
            "readout_propagation_edges must be transition/association/both"
        )
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if int(lag) < 0:
        raise ValueError("lag must be non-negative")
    if not 0.5 < float(stage_memory_quantile) < 1.0:
        raise ValueError("stage_memory_quantile must be between 0.5 and 1")
    if str(stage_memory_feature_source) not in ("branch", "prototype"):
        raise ValueError(
            "stage_memory_feature_source must be branch or prototype"
        )
    if int(stage_demand_min_support) < 1:
        raise ValueError("stage_demand_min_support must be positive")
    if int(stage_demand_min_episodes) < 1:
        raise ValueError("stage_demand_min_episodes must be positive")
    if int(component_min_count) < 1:
        raise ValueError("component_min_count must be positive")
    if int(max_components) < 0:
        raise ValueError("max_components must be non-negative")
    if not 0.0 <= float(voi_fraction) < 1.0:
        raise ValueError("voi_fraction must be in [0, 1)")
    if int(voi_target) < 1:
        raise ValueError("voi_target must be positive")
    if float(feature_value_weight) < 0.0:
        raise ValueError("feature_value_weight must be non-negative")
    if int(dagger_rounds) < 0:
        raise ValueError("dagger_rounds must be non-negative")
    if int(dagger_episodes) < 0:
        raise ValueError("dagger_episodes must be non-negative")
    if dagger_rounds and int(dagger_episodes) < 1:
        raise ValueError("dagger_episodes must be positive")
    if int(dagger_cap) < 1:
        raise ValueError("dagger_cap must be positive")
    if int(dagger_offset) < 0:
        raise ValueError("dagger_offset must be non-negative")
    if int(dagger_horizon) < 1:
        raise ValueError("dagger_horizon must be positive")
    if int(dagger_frames) < 0:
        raise ValueError("dagger_frames must be non-negative")
    if int(dagger_frames) > 0 and int(dagger_rounds) < 1:
        raise ValueError("dagger_frames requires at least one dagger round")
    if int(step_stride) < 1:
        raise ValueError("step_stride must be positive")
    if int(value_horizon) < 0:
        raise ValueError("value_horizon must be non-negative")
    if float(ood_trust_scale) < 0.0:
        raise ValueError("ood_trust_scale must be non-negative")
    if not 0.0 <= float(obs_trust) <= 1.0:
        raise ValueError("obs_trust must be in [0, 1]")
    if float(feature_action_prior) < 0.0:
        raise ValueError("feature_action_prior must be non-negative")
    if float(feature_value_prior) < 0.0:
        raise ValueError("feature_value_prior must be non-negative")
    if not 0.0 < float(value_label_quantile) < 1.0:
        raise ValueError("value_label_quantile must be in (0, 1)")
    if float(value_trust_samples) < 0.0:
        raise ValueError("value_trust_samples must be non-negative")
    if float(control_risk_weight) < 0.0:
        raise ValueError("control_risk_weight must be non-negative")
    if float(curiosity_weight) < 0.0:
        raise ValueError("curiosity_weight must be non-negative")
    if not 0.0 <= float(rule_min_coverage) <= 1.0:
        raise ValueError("rule_min_coverage must be in [0, 1]")
    if not 0.0 <= float(rule_min_primitive_coverage) <= 1.0:
        raise ValueError("rule_min_primitive_coverage must be in [0, 1]")
    if int(rule_validation_seed) < 0:
        raise ValueError("rule_validation_seed must be non-negative")
    if int(rule_validation_frames) < 1:
        raise ValueError("rule_validation_frames must be positive")
    lag = int(lag)
    voi_fraction = float(voi_fraction)
    if voi_fraction > 0.0:
        if steps < 2:
            raise ValueError("two-stage collection needs at least 2 steps")
        steps1 = max(1, int(round(steps * (1.0 - voi_fraction))))
        stage1, _counts = collect(
            seed, steps1, coverage=coverage, lag=lag,
            env_factory=env_factory,
        )
        if not stage1:
            raise ValueError("collection produced no transitions")
        _bootstrap_world, guide_agent = _train(
            stage1, seed=seed, epochs=epochs, lag=lag,
            max_nodes=max_nodes, birth_threshold=birth_threshold,
            component_kinds=component_kinds,
            component_min_count=component_min_count,
            max_components=max_components,
            component_weight=component_weight,
            component_gating=component_gating,
            memory_weight=memory_weight,
            association_rate=association_rate,
            induce_features=induce_features,
            precursor_labels=precursor_labels,
            pain_horizon=pain_horizon,
            voi_target=voi_target,
            feature_value_weight=feature_value_weight,
            # The guide's threat gate reads the calibrated graded threat;
            # the monotone calibration only exists when this flag is on,
            # and the bootstrap agent never plans, so the flag costs
            # nothing beyond the calibration fit.
            contrast_objective=True,
            branch_transitions=branch_transitions,
            action_edge_weight=action_edge_weight,
            action_edge_trust_samples=action_edge_trust_samples,
        )
        propensities = []
        # Re-collect the whole stream with the guide instead of splicing
        # a guided tail onto the bootstrap stream: the two-stage split
        # fragments the episodes enough to cost a third of survival at
        # full scale even when the guide never fires, and a fresh single
        # pass lets the guide's knowledge shape every decision point.
        transitions, _counts2 = collect(
            seed, steps, coverage=coverage, lag=lag,
            guide=guide_agent.counterfactual_action,
            budget=int(round(voi_fraction * steps)),
            propensities=propensities,
            env_factory=env_factory,
        )
        transitions = list(transitions)
    else:
        transitions, _counts = collect(
            seed, steps, coverage=coverage, lag=lag,
            env_factory=env_factory,
        )
    if not transitions:
        raise ValueError("collection produced no transitions")
    rule_validation_x = None
    if int(rule_validation_seed) > 0:
        # Cross-world activation check: the validation stream supplies
        # states only (no labels, no training), so a rule that only fires
        # in the training world can be rejected before it is installed.
        validation_rows, _ = collect(
            int(rule_validation_seed), steps=int(rule_validation_frames),
            coverage=coverage, lag=lag,
            env_factory=env_factory,
        )
        if validation_rows:
            rule_validation_x = np.asarray(
                [row["obs"] for row in validation_rows], dtype=float
            )
    train_kwargs = dict(
        seed=seed, epochs=epochs, lag=lag,
        max_nodes=max_nodes, birth_threshold=birth_threshold,
        component_kinds=component_kinds,
        component_min_count=component_min_count,
        max_components=max_components,
        component_weight=component_weight,
        component_gating=component_gating,
        memory_weight=memory_weight,
        association_rate=association_rate,
        induce_features=induce_features,
        precursor_labels=precursor_labels,
        pain_horizon=pain_horizon,
        voi_target=voi_target,
        feature_value_weight=feature_value_weight,
        value_horizon=value_horizon,
        pain_gamma=pain_gamma,
        ood_trust_scale=ood_trust_scale,
        obs_trust=obs_trust,
        feature_action_prior=feature_action_prior,
        feature_value_prior=feature_value_prior,
        value_features=value_features,
        value_label_quantile=value_label_quantile,
        subgraph_cells=subgraph_cells,
        predicate_readouts=predicate_readouts,
        event_memory_risk=event_memory_risk,
        contrast_objective=contrast_objective,
        value_trust_samples=value_trust_samples,
        control_risk_weight=control_risk_weight,
        value_bank_only=value_bank_only,
        value_dispersion=value_dispersion,
        iterative_induction=iterative_induction,
        compose_readout=compose_readout,
        importance_weighting=importance_weighting,
        curiosity_weight=curiosity_weight,
        dagger_frames=dagger_frames,
        memory_credit=memory_credit,
        rule_min_coverage=rule_min_coverage,
        rule_min_primitive_coverage=rule_min_primitive_coverage,
        rule_coverage_x=rule_validation_x,
        pairwise_features=pairwise_features,
        branch_transitions=branch_transitions,
        action_edge_weight=action_edge_weight,
        action_edge_trust_samples=action_edge_trust_samples,
        feature_backfit=feature_backfit,
        risk_error_weight=risk_error_weight,
        memory_feature_backfill=memory_feature_backfill,
        death_rules=death_rules,
        death_rule_compose=death_rule_compose,
        death_rule_max_thresholds=death_rule_max_thresholds,
        death_rule_max_rules=death_rule_max_rules,
        death_rule_max_literals=death_rule_max_literals,
        death_rule_boundary_margin=death_rule_boundary_margin,
        death_rule_envelope=death_rule_envelope,
        readout_propagation_gain=readout_propagation_gain,
        readout_propagation_steps=readout_propagation_steps,
        readout_propagation_edges=readout_propagation_edges,
    )
    world, agent = _train(transitions, **train_kwargs)
    dagger_log = []
    planner_rows: List[dict] = []
    for _round in range(int(dagger_rounds)):
        # A frame budget buys a fixed amount of on-policy experience even
        # when episodes die early.  Each episode contributes at least one
        # frame, so `dagger_frames` resets guarantee the budget can be
        # reached; collection still stops as soon as the budget is met.
        planner_episodes = int(dagger_episodes)
        if int(dagger_frames) > 0:
            planner_episodes = max(planner_episodes, int(dagger_frames))
        planner, _frames = collect_planner(
            seed, world, agent, episodes=planner_episodes,
            cap=dagger_cap, horizon=dagger_horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
            # Each round must get fresh episode ids: merging two rollouts
            # under one id would make the backward pass treat the second as
            # a continuation of the first.  The stride between rounds is
            # the number of episode ids a round can actually consume (the
            # frame budget can stretch `planner_episodes` far beyond
            # `dagger_episodes`), so a `dagger_episodes`-sized stride would
            # collide ids across rounds and splice distinct real episodes
            # into one list.
            offset=dagger_offset + _round * int(planner_episodes),
            max_frames=(int(dagger_frames) or None),
            risk_weight=risk_weight,
            risk_decay=risk_decay,
            env_factory=env_factory,
        )
        if not planner:
            break
        transitions = list(transitions) + planner
        dagger_log.append(dict(
            round=_round + 1,
            planner_steps=len(planner),
            planner_frames=list(_frames),
            total_steps=len(transitions),
        ))
        world, agent = _train(transitions, **train_kwargs)
        if learn_objective and planner:
            # P4: the planner's own executed decisions measure the
            # objective's trade-off; the learned weight rides into the
            # next round's planner and the returned agent.
            planner_rows.extend(planner)
            learned = learn_objective_weight(planner_rows)
            if learned is not None:
                agent.feature_value_weight = learned
                dagger_log[-1]["learned_value_weight"] = learned
            report = ledger_survival_report(
                planner_rows, agent.feature_value_weight
            )
            if report:
                dagger_log[-1]["ledger_report"] = report
    agent.dagger_log = dagger_log
    # Which form of the composed hazard the planner reads: the calibrated
    # rate (a level) or the raw weighted sum (a ranking).  The isotonic map is
    # monotone, so it cannot invert an order - but on a rare-event target it
    # collapses almost the whole score range into one plateau, which makes
    # distinct candidates indistinguishable.  Recorded on the agent so the
    # choice travels with the readout it belongs to.
    agent.hazard_readout = hazard_readout
    # The endpoint value readout: the per-feature cells (empty at epochs=1) or
    # the association value (the weighted quality of the nodes the stream met
    # after this one).  Built once, from the same realised stream.
    agent.value_readout = value_readout
    # Association value is built once: the high-level successor choice uses
    # it, and the endpoint value readout reuses it when selected.
    association_values, association_support = _association_value_table(
        transitions, agent
    )
    agent.association_values = association_values
    agent.association_support = association_support
    transition_edges, node_risk, default_action = _transition_table(
        transitions, agent
    )
    agent.transition_edges = transition_edges
    agent.node_risk = node_risk
    agent.transition_default_action = default_action
    transition_policy, transition_values = _transition_values(
        transition_edges, node_risk
    )
    agent.transition_policy = transition_policy
    agent.transition_values = transition_values
    # Unified frame-to-arbitrary-window transition measure.  It is fit from
    # realized transitions only and keeps the action that actually acted on
    # each state edge (history[lag]), rather than the action just issued.
    # The planner does not consume it by default; probes can query the same
    # fitted object at multiple windows without training separate hazards.
    agent.multiview_transitions = MultiViewTransitionModel.fit_rows(
        transitions, agent, lag=agent.lag,
    )
    if stage_memory:
        # Automatic self-supervised experience path.  The detector is fit on
        # these episodes only; per-transition feature snapshots come from the
        # runtime's frozen feature masks.  Nothing here is read by the
        # planner, so enabling it cannot change a decision.
        agent.stage_experience = StageExperienceBatch.fit(
            transitions, agent, lag=agent.lag,
            quantile=stage_memory_quantile,
            feature_source=stage_memory_feature_source,
        )
        # State -> stage type resolution fitted on the same training fold.
        # The resolver is a data-layer readout; no planner path calls it.
        agent.stage_resolver = StageStateResolver.fit_rows(
            transitions, agent, experience=agent.stage_experience,
        )
        # Stage-level outcome memory and the demand selector's readout, all
        # from realized stages.  The planner does not read either object.
        agent.stage_outcome_memory = StageOutcomeMemory().fit(
            agent.stage_experience.outcome_episodes()
        )
        agent.stage_demand_readout = StageOutcomeReadout.fit(
            agent.stage_experience.episodes
        )
        agent.stage_demand_selector = StageDemandSelector(
            agent.stage_demand_readout
        )
        agent.stage_demand_heuristic = StageDemandHeuristic(
            min_support=stage_demand_min_support,
            min_episodes=stage_demand_min_episodes,
        ).fit(agent.stage_experience.demand_samples)
        episodes = list(agent.stage_experience.episodes)
        tree_train = [
            episode for index, episode in enumerate(episodes)
            if index % 5 != 4
        ]
        tree_validation = [
            episode for index, episode in enumerate(episodes)
            if index % 5 == 4
        ]
        if tree_train and tree_validation:
            agent.stage_tree = PredictiveStageTree.fit(
                tree_train, tree_validation,
                prior=4.0, tolerance=0.0,
                min_validation_support=4,
                min_support=2, min_episodes=2,
            )
            collapsed = collapse_stage_episodes(
                tree_train, agent.stage_tree, mode="root",
            )
            agent.stage_tree_heuristic = StageDemandHeuristic(
                min_support=stage_demand_min_support,
                min_episodes=stage_demand_min_episodes,
            ).fit(
                DemandAssociationSample.from_stage_prefixes(collapsed)
            )

            def collapse_key(key):
                chain = agent.stage_tree.chain(key)
                return (
                    (("tree", "base"),)
                    if not chain
                    else (("tree", str(chain[-1])),)
                )

            agent.stage_root_heuristic = StageDemandHeuristic(
                min_support=stage_demand_min_support,
                min_episodes=stage_demand_min_episodes,
            ).fit(
                DemandAssociationSample.from_stage_roots(
                    tree_train, collapse_key, pooled=True,
                )
            )
            prototype_roots = {}
            try:
                transitions_for_roots = agent.stage_experience.stage_transitions(
                    transitions, agent,
                )
                for transition in transitions_for_roots:
                    state = np.concatenate([
                        np.asarray(transition.observation, dtype=float),
                        np.asarray(transition.pending, dtype=float),
                    ]) / np.asarray(agent.scales, dtype=float)
                    prototype, _distance = agent._nearest(state)
                    if prototype is None:
                        continue
                    token = collapse_key(transition.stage.key)[0]
                    counts = prototype_roots.setdefault(
                        str(prototype), {}
                    )
                    counts[token] = int(counts.get(token, 0)) + 1
            except ValueError:
                prototype_roots = {}
            agent.stage_prototype_roots = {
                prototype: max(
                    counts,
                    key=lambda token: (counts[token], repr(token)),
                )
                for prototype, counts in prototype_roots.items()
            }
        else:
            agent.stage_tree = None
            agent.stage_tree_heuristic = None
            agent.stage_root_heuristic = None
            agent.stage_prototype_roots = {}
        agent.stage_demand_feature_source = stage_memory_feature_source
    else:
        agent.stage_experience = None
        agent.stage_resolver = None
        agent.stage_outcome_memory = None
        agent.stage_demand_readout = None
        agent.stage_demand_selector = None
        agent.stage_demand_heuristic = None
        agent.stage_tree = None
        agent.stage_tree_heuristic = None
        agent.stage_root_heuristic = None
        agent.stage_prototype_roots = {}
        agent.stage_demand_feature_source = None
    # The milestone layer, read off the agent's own stream.  All of it is
    # inert unless the planner is asked for one of its terms.
    model = _milestone_model_from_transitions(
        transitions, world=world, agent=agent, stage=milestone_stage,
        boundary_k=milestone_boundary_k,
        successor_value=association_values,
    )
    agent.milestone_goals = model["goals"]
    agent.milestone_threshold = model["threshold"]
    agent.milestone_rate = model["rate"]
    agent.milestone_keys = model["keys"]
    agent.milestone_scale = model["scale"]
    agent.milestone_frame = model.get("frame")
    agent.milestone_goals_by_key = model["goals_by_key"]
    agent.milestone_transitions = model.get("transitions", [])
    agent.milestone_durations = model.get("durations", {})
    agent.milestone_feedbacks = model.get("feedbacks", {})
    agent.milestone_risks = model.get("risks", {})
    agent.milestone_stages = model.get("stages", {})
    agent.milestone_clusters = model.get("clusters")
    agent.milestone_stage_mode = milestone_stage
    stage_index = model.get("stage_index") or {}
    agent.milestone_stage_states = np.asarray(
        stage_index.get("states", ()), dtype=float
    )
    agent.milestone_stage_keys = [
        tuple(int(value) for value in key)
        for key in stage_index.get("keys", ())
    ]
    agent.milestone_summary = model["summary"]
    # The rows the milestone layer was read from, kept so the layer can be
    # re-measured without repeating the fit (diagnostics, offline probes).
    agent.milestone_training_rows = transitions
    return world, agent


def _stage_evidence_for_streams(rows_list, world, agent, horizon: int = 6,
                                boundary_k: float = 3.0):
    """Stage boundaries and signatures from the learned graph, per stream.

    The criterion is the one the stream itself makes available: a boundary is
    a frame where the nearest node's strength has just risen *and* the
    action-conditioned future is no longer the future that came before
    (:func:`strength_boundaries`).  Strength is the node's own count, i.e. how
    much of the stream the graph has already committed to that state; the
    future is the learned model's rollout under each action.  Neither is a
    task quantity: both are read off `agent` and `world` exactly as they were
    trained.

    The threshold is pooled across episodes (``k`` times the pooled 90th
    percentile), so every stream is segmented by the same bar, and ``k``
    defaults to the same value the geometric segmenter uses - sharing the
    statistical rule is the point; nothing here is chosen per world.

    Returns ``(boundaries_list, signatures_list)`` or ``None`` when either
    consumer is missing (offline callers and unit tests keep the geometric
    default).
    """
    if world is None or agent is None:
        return None
    evidence = []
    for rows in rows_list:
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        encoded = np.concatenate(
            [observations, _encoded_pending(rows, agent.lag)], axis=1
        ) / agent.scales
        strength = np.zeros(len(encoded), dtype=float)
        for index, state in enumerate(encoded):
            node_id, _distance = agent._nearest(state)
            node = None if node_id is None else agent.graph.nodes.get(node_id)
            if node is not None:
                strength[index] = float(node.count)
        signatures = transition_signatures(
            observations, world.predict, agent.lag, horizon
        )
        evidence.append((strength, signatures))
    threshold = pooled_strength_threshold(
        [strength_change_scores(strength, signatures)
         for strength, signatures in evidence],
        k=boundary_k,
    )
    boundaries_list = []
    signatures_list = []
    for strength, signatures in evidence:
        boundaries_list.append(np.asarray(
            strength_boundaries(strength, signatures, threshold=threshold),
            dtype=int,
        ))
        signatures_list.append(signatures)
    return boundaries_list, signatures_list


def _milestone_model_from_transitions(transitions, bins=None, top=1,
                                      min_support=3, world=None, agent=None,
                                      stage: str = "strength",
                                      horizon: int = 6,
                                      boundary_k: float = 3.0,
                                      successor_value=None):
    """Everything the milestone layer needs, read off the agent's own stream.

    * ``frame`` - the time-orthogonal identity frame, calibrated on the pooled
      stream so every episode is read through one set of directions and cuts;
    * ``goals`` - the boundary states milestones started in, most frequent
      first, taken from the *same* segmentation as the identities (the earlier
      version calibrated a second, per-episode segmentation for the goals);
    * ``rate`` - per milestone identity, the duration-weighted action rate of
      the *better half* of the stream (by realised return), not of the whole
      behaviour policy: cloning the open-loop collection policy's rate would
      only reinforce the behaviour that fails.

    ``stage`` chooses the segmentation: ``"strength"`` (default) uses the
    learned graph - a node's strength rises where the action-conditioned
    future changes (:func:`strength_boundaries`); ``"geometry"`` keeps the
    earlier boundary statistic of the observation stream.  The two are kept
    as an explicit switch so the change can be measured rather than assumed.

    Nothing about the task is read.
    """
    episodes = {}
    for row in transitions:
        episodes.setdefault(int(row.get("episode", 0)), []).append(row)
    usable_items = [
        (episode, rows) for episode, rows in episodes.items() if len(rows) >= 3
    ]
    usable = [rows for _episode, rows in usable_items]
    if not usable:
        return dict(
            goals=None, summary=None, frame=None, threshold=None,
            scale=None, rate={}, keys=[], goals_by_key={},
            transitions=[], durations={}, feedbacks={}, risks={},
            stages={}, clusters=None,
        )
    # One frame of reference for the whole stream.  The identity is only a
    # common alphabet if every episode is read through the same directions and
    # the same boundary scale; calibrating per episode (the earlier behaviour)
    # made the keys incomparable across episodes and left the decision-time
    # tracker reading a frame taken from a single episode.
    streams = [np.asarray([row["obs"] for row in rows], dtype=float)
               for rows in usable]
    # The boundary statistic is pooled from the within-episode frames only:
    # episode seams are discontinuities that are not phase resets, and
    # stitching them together would inflate the high quantile.
    model = pooled_boundary_model(streams)
    threshold = model["threshold"]
    scale = model["scale"]
    boundaries_list = None
    signatures_list = None
    if stage in ("strength", "signature"):
        evidence = _stage_evidence_for_streams(
            usable, world, agent, horizon=horizon, boundary_k=boundary_k
        )
        if evidence is not None:
            boundaries_list, signatures_list = evidence
    identity_list = None
    clusters = None
    if stage == "signature" and signatures_list is not None:
        prototypes = [
            segment_prototypes(signatures, boundaries)
            for signatures, boundaries in zip(signatures_list,
                                              boundaries_list)
        ]
        clusters = signature_clusters(prototypes, min_support=min_support)
        clusters["horizon"] = int(horizon)
        identity_list = [
            signature_keys(prototype, clusters["centers"])
            for prototype in prototypes
        ]
    boundaries_by_episode = {}
    if boundaries_list is not None:
        boundaries_by_episode = {
            episode: boundaries
            for (episode, _rows), boundaries in zip(usable_items,
                                                    boundaries_list)
        }
    frame = boundary_axis(streams, boundaries_list=boundaries_list,
                          bins=bins, threshold=threshold,
                          scale=scale, min_support=min_support)
    goals, summary = _milestone_goals_from_transitions(
        transitions, bins=bins, top=top, frame=frame,
        threshold=threshold, scale=scale,
        boundaries_by_episode=boundaries_by_episode,
    )
    collected = {}
    better = {}
    key_set = set()
    memories = []
    returns = []
    for stream_index, rows in enumerate(usable):
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        histories = [
            np.asarray(row.get("history", ()), dtype=float).reshape(-1)
            for row in rows
        ]
        # The transition obs_t -> obs_{t+1} is caused by the oldest action
        # still pending, not the action just issued at t.  NodeRuntime uses
        # this same history[lag] alignment for its action-conditioned reads.
        acting_lag = int(
            getattr(agent, "lag", max(len(histories[0]) - 1, 0))
        )
        actions = np.asarray([
            int(round(float(history[acting_lag])))
            if history.size > acting_lag else -1
            for history in histories
        ], dtype=int)
        signals = np.asarray(
            [row.get("signals", (0.0, 0.0)) for row in rows], dtype=float
        )
        outcome = signals[:, 0:1] - signals[:, 1:2]
        memories.append(milestone_memory(
            observations, actions=actions, feedback=outcome,
            boundaries=None if boundaries_list is None
            else boundaries_list[stream_index],
            identity=None if identity_list is None
            else identity_list[stream_index],
            threshold=threshold, scale=scale, frame=frame,
        ))
        returns.append(float(outcome.sum()))
    cutoff = float(np.median(returns)) if returns else 0.0
    for index, memory in enumerate(memories):
        key_set.update(tuple(int(v) for v in key) for key in memory["keys"])
        for key, value in memory["key_rate"].items():
            collected.setdefault(key, []).append(float(value))
            if returns[index] >= cutoff:
                better.setdefault(key, []).append(float(value))
    source = better or collected
    rate = {key: float(np.mean(values)) for key, values in source.items()}
    # The subgoal of a milestone, read off the transition graph: the entry
    # state of the successor the stream most often took from here.
    prototype_by_key = {}
    successor = {}
    for memory in memories:
        for key, prototype in zip(memory["keys"], memory["prototypes"]):
            prototype_by_key.setdefault(
                tuple(int(v) for v in key), np.asarray(prototype, dtype=float)
            )
        for row in memory["transitions"]:
            source_key = tuple(int(v) for v in row["key_from"])
            target_key = tuple(int(v) for v in row["key_to"])
            counts = successor.setdefault(source_key, {})
            counts[target_key] = counts.get(target_key, 0) + int(row["count"])
    goals_by_key = {}
    pending_width = int(getattr(agent, "lag", 0)) if agent is not None else 0

    def _successor_value(candidate):
        """Realised value of the successor's entry state, via node association."""
        prototype = prototype_by_key.get(candidate)
        if prototype is None or successor_value is None or agent is None:
            return None
        encoded = np.concatenate([
            np.asarray(prototype, dtype=float),
            np.zeros(pending_width),
        ]) / agent.scales
        node_id, _distance = agent._nearest(encoded)
        if node_id is None:
            return None
        return float(successor_value.get(node_id, 0.0))

    for key, counts in successor.items():
        modal = max(counts, key=lambda candidate: counts[candidate])
        best = modal
        if successor_value is not None:
            # The high-level choice: the successor whose entry state has the
            # best realised value, not the one the stream happened to take
            # most often.  The modal successor is only the fallback.
            scored = {
                candidate: _successor_value(candidate) for candidate in counts
            }
            valued = [candidate for candidate, value in scored.items()
                      if value is not None]
            if valued:
                best = max(
                    valued,
                    key=lambda candidate: (scored[candidate], counts[candidate]),
                )
        if best in prototype_by_key:
            goals_by_key[key] = prototype_by_key[best]
    # Persist the memory itself: without it "the transition table" was a
    # local variable and no consumer could ever predict a stage ahead.
    pooled = {}
    duration_totals = {}
    duration_counts = {}
    feedback_totals = {}
    for memory in memories:
        for row in memory["transitions"]:
            pair = (tuple(int(v) for v in row["key_from"]),
                    tuple(int(v) for v in row["key_to"]))
            entry = pooled.setdefault(
                pair, dict(count=0, duration=0.0, feedback=0.0)
            )
            count = int(row["count"])
            entry["count"] += count
            entry["duration"] += count * float(row["mean_duration"])
            entry["feedback"] += count * float(row["mean_feedback"])
        for key, length, outcome in zip(
            memory["keys"], memory["durations"], memory["feedback"]
        ):
            stage = tuple(int(v) for v in key)
            duration_totals[stage] = duration_totals.get(stage, 0) + int(length)
            duration_counts[stage] = duration_counts.get(stage, 0) + 1
            feedback_totals[stage] = feedback_totals.get(stage, 0.0) + float(
                outcome
            )
    transitions = [
        dict(
            count=int(entry["count"]),
            key_from=list(pair[0]),
            key_to=list(pair[1]),
            mean_duration=float(entry["duration"]) / int(entry["count"]),
            mean_feedback=float(entry["feedback"]) / int(entry["count"]),
        )
        for pair, entry in pooled.items()
    ]
    transitions.sort(key=lambda row: -row["count"])
    durations = {
        stage: float(total) / max(int(duration_counts[stage]), 1)
        for stage, total in duration_totals.items()
    }
    feedbacks = {
        stage: float(feedback_totals[stage]) / max(float(total), 1.0)
        for stage, total in duration_totals.items()
    }
    # A stage is worse than usual when its per-frame feedback sits below the
    # stream's own duration-weighted average.  The binary test "feedback <= 0"
    # almost never fires: a fifty-frame stage that ends in one bad frame still
    # averages positive, so every charge collapsed to zero and the term became
    # a constant across plans (measured 2026-11-22: all stage risks 0.00, the
    # plan term bit-identical at every weight).  The deficit is scaled by the
    # worst observed deficit, so the charge is in [0, 1] whatever the feedback
    # scale is - a property of the stream, not a chosen constant.
    total_feedback = float(sum(feedback_totals.values()))
    total_frames = float(sum(duration_totals.values()))
    center = total_feedback / total_frames if total_frames > 0.0 else 0.0
    deficit = {
        stage: max(0.0, center - float(value))
        for stage, value in feedbacks.items()
    }
    worst = max(deficit.values()) if deficit else 0.0
    key_cost = {
        stage: (value / worst if worst > 0.0 else 0.0)
        for stage, value in deficit.items()
    }
    risks = {}
    for stage in key_set:
        walk = milestone_lookahead(
            transitions, stage, feedbacks, steps=2, key_cost=key_cost
        )
        risks[stage] = float(walk["cost"])
    # What the segmentation is worth, measured from the segmentation itself:
    # how many frames a stage lasts and how often one stage leads to more than
    # one successor.  A usable stage graph branches; a chain does not.
    sources = {}
    for row in transitions:
        sources.setdefault(tuple(row["key_from"]), set()).add(
            tuple(row["key_to"])
        )
    # Does the identity carry information?  Each stage contributes its own
    # per-frame feedback; the between-identity share of that variance (eta^2)
    # says whether stages are outcome classes or arbitrary labels.
    stage_rows = []
    stage_states = []
    stage_labels = []
    for stream_index, memory in enumerate(memories):
        edges = np.concatenate([
            [0], np.asarray(memory["boundaries"], dtype=int),
            [len(streams[stream_index])],
        ])
        for index, key in enumerate(memory["keys"]):
            start = int(edges[index])
            stage_states.append(
                streams[stream_index][start] if start < len(streams[stream_index])
                else streams[stream_index][-1]
            )
            stage_labels.append(tuple(int(value) for value in key))
            stage_rows.append((
                tuple(int(value) for value in key),
                float(memory["feedback"][index])
                / max(float(memory["durations"][index]), 1.0),
            ))
    feedback_eta2 = 0.0
    if stage_rows:
        values = np.asarray([value for _key, value in stage_rows])
        grand = float(values.mean())
        spread = float(np.sum((values - grand) ** 2))
        if spread > 1e-12:
            between = 0.0
            for key in set(entry[0] for entry in stage_rows):
                group = np.asarray(
                    [value for candidate, value in stage_rows if candidate == key]
                )
                between += len(group) * float((group.mean() - grand) ** 2)
            feedback_eta2 = float(between / spread)
    stages = dict(
        streams=len(memories),
        boundaries=int(sum(len(memory["boundaries"]) for memory in memories)),
        stages=int(sum(len(memory["keys"]) for memory in memories)),
        identities=len(key_set),
        dwell=float(np.mean([
            float(np.mean(memory["durations"])) for memory in memories
        ])) if memories else 0.0,
        sources=len(sources),
        branch=float(np.mean([len(v) for v in sources.values()]))
        if sources else 0.0,
        branching=int(sum(1 for v in sources.values() if len(v) > 1)),
        feedback_eta2=float(feedback_eta2),
    )
    return dict(
        goals=goals, summary=summary, frame=frame,
        threshold=threshold, scale=scale, rate=rate, keys=sorted(key_set),
        goals_by_key=goals_by_key, transitions=transitions,
        durations=durations, feedbacks=feedbacks, risks=risks,
        stages=stages, clusters=clusters,
        stage_index=dict(
            states=np.asarray(stage_states, dtype=float).tolist(),
            keys=[list(key) for key in stage_labels],
        ),
    )


def _milestone_goals_from_transitions(transitions, bins=2, top=1, frame=None,
                                      threshold=None, scale=None,
                                      boundaries_by_episode=None):
    """Goal states for the milestone shaping term, from the agent's own stream.

    The training stream is segmented into milestones and each milestone's
    starting state becomes a candidate goal.  Candidates are ranked by how
    often the stream actually started a milestone there: with this world's
    feedback unable to tell milestones apart (the only feedback is survival
    per frame, so a milestone's total feedback equals its length), frequency
    is the informative learned statistic and the modal boundary state is the
    prototypical place to head for.  Nothing about the task is read - the
    states are observations the runtime already saw.

    When `frame` is supplied the milestones are the ones read through that
    shared frame, so the goals and the identities come from the same
    segmentation; without it the per-episode default is kept for callers that
    only want prototype states.

    Returns ``(goals, summary)`` or ``(None, None)`` when the stream carries
    no milestones.
    """
    episodes = {}
    for row in transitions:
        episodes.setdefault(int(row.get("episode", 0)), []).append(row)
    prototypes = []
    for episode, rows in episodes.items():
        if not rows:
            continue
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        signals = np.asarray(
            [row.get("signals", (0.0, 0.0)) for row in rows], dtype=float
        )
        outcome = (signals[:, 0] - signals[:, 1]).reshape(-1, 1)
        boundaries = None
        if boundaries_by_episode is not None:
            boundaries = boundaries_by_episode.get(episode)
        if frame is not None:
            memory = milestone_memory(
                observations, feedback=outcome, frame=frame,
                boundaries=boundaries, threshold=threshold, scale=scale,
            )
        else:
            memory = milestone_memory(
                observations, feedback=outcome, bins=bins,
                boundaries=boundaries,
            )
        prototypes.extend(
            np.asarray(prototype, dtype=float)
            for prototype in memory["prototypes"]
        )
    if not prototypes:
        return None, None
    prototypes = np.asarray(prototypes, dtype=float)
    # The modal boundary state wins: identical prototypes are counted.
    first, counts = {}, []
    for prototype in prototypes:
        key = tuple(np.round(prototype, 6).tolist())
        if key not in first:
            first[key] = len(counts)
            counts.append(0)
        counts[first[key]] += 1
    ranking = sorted(range(len(counts)), key=lambda index: -counts[index])
    chosen = ranking[:max(int(top), 1)]
    keys = list(first.keys())
    goals = np.asarray([np.asarray(keys[index], dtype=float)
                        for index in chosen])
    summary = dict(
        candidates=len(prototypes),
        distinct=len(counts),
        goals=goals.tolist(),
        counts=[int(counts[index]) for index in chosen],
    )
    return goals, summary


def terminal_hazard(pain, risk, risk_weight, pain_weight=None, lane="max"):
    """The endpoint's hazard from two *different* quantities.

    * ``pain`` - the state's realised outcome statistic: how often entering
      this state was actually painful.  It carries no window: it is the
      outcome the world already reported here.
    * ``risk`` - the calibrated probability of reaching pain inside the
      readout's window.  It is an estimate about the future, not an outcome.

    They are different parameters of the valuation function, so they may be
    carried with different weights.  ``lane="max"`` keeps the historical
    composition (the realised outcome replaces the estimate when it is
    larger, one shared weight); ``lane="separate"`` keeps the two terms, each
    with its own weight, which is the form that lets the value of a certain
    outcome differ from the price of a mere possibility.
    """
    pain = np.asarray(pain, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if lane == "separate":
        weight = float(risk_weight if pain_weight is None else pain_weight)
        return weight * pain + float(risk_weight) * risk
    return np.maximum(pain, risk) * float(risk_weight)


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


def association_value_batch(agent, encoded):
    """Read the association value at each state through its nearest prototype."""
    encoded = np.asarray(encoded, dtype=float)
    table = getattr(agent, "association_values", None) or {}
    support_map = getattr(agent, "association_support", None) or {}
    values = np.zeros(len(encoded))
    supports = np.zeros(len(encoded))
    if not table:
        return values, supports
    for index, state in enumerate(encoded):
        node_id, _distance = agent._nearest(state)
        if node_id is None:
            continue
        value = table.get(node_id)
        if value is None:
            continue
        values[index] = float(value)
        supports[index] = float(support_map.get(node_id, 0))
    return values, supports


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


@contextmanager
def _curiosity_off(agent):
    """Evaluation reads the objective without the curiosity term.

    Curiosity is a training-side exploration incentive: it changes which
    plans the planner executes while collecting on-policy data, and those
    executed transitions are real.  The ruler must not be shaped, so every
    evaluation episode runs with the weight set to zero and the training
    value restored afterwards.
    """
    saved = float(getattr(agent, "curiosity_weight", 0.0))
    agent.curiosity_weight = 0.0
    try:
        yield
    finally:
        agent.curiosity_weight = saved


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
        scaled = encoded / scales
        features = []
        for observation, queue, encoded_state in zip(
            raw_observations, pending, scaled,
        ):
            resolved = resolver.resolve(agent, observation, queue)
            node_token = None
            if resolved["status"] == "supported":
                chain = tree.chain(resolved["key"])
                if chain:
                    node_token = ("tree", str(chain[-1]))
            if node_token is None:
                # Rank ordering may see predicted states outside the
                # resolver's OOD radius.  Fall back to the root most often
                # observed on the nearest training prototype; this is a
                # read-only rank hint, never a stage assertion.
                prototype_roots = getattr(
                    agent, "stage_prototype_roots", {},
                )
                nearest = getattr(agent, "_nearest", None)
                if nearest is not None and prototype_roots:
                    prototype, _distance = nearest(encoded_state)
                    if prototype is not None:
                        node_token = prototype_roots.get(str(prototype))
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
                if (node_token,) == demand.target_stage:
                    stats["target_present"] = (
                        int(stats.get("target_present", 0)) + 1
                    )
            features.append((node_token,))
        return features

    return make_demand_rank_step(
        heuristic, demand, feature_extractor=extract_features,
        stats=stats,
    )


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
                  rank_mode="order", rank_margin=0.0):
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

    def predict_batch(states, sequences, step):
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
            encoded = (
                np.concatenate([agent_view(states), pending], axis=1)
                / agent.scales
            )
            previous_history = history_features(
                sequences, step, pending_prefix, action_dim=action_dim
            )
            previous_pending = previous_history[:, :agent.lag][:, ::-1]
            previous_encoded = np.concatenate(
                [agent_view(previous), previous_pending], axis=1
            ) / agent.scales
            risk = agent.pain_signal_batch(encoded) * risk_weight * (
                risk_decay ** step
            ) * error_factor(step)
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
            pain, dopamine = agent.valence_state_batch(encoded)
            mask_table = agent._feature_mask_table(
                encoded[:, :agent.obs_dim]
            )
            if getattr(agent, "predicate_readouts", False):
                # D3: the terminal hazard comes from the predicates only;
                # the prototype valence no longer bounds it.
                signal = agent.pain_signal_batch(encoded, mask_table)
                risk = signal * risk_weight * (risk_decay ** horizon) * (
                    error_factor(horizon - 1)
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
        result = beam_search(
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
        )
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
                    rank_mode="order", rank_margin=0.0):
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
            or demand_heuristic is not None):
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
                candidate_count=int(
                    getattr(result, "candidate_count", 0)
                ),
            ))
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


def evaluate(seed, world, agent, cap=6000, horizon=10, lag=LAG,
             step_stride=2, stride_includes_lag=False, env_seed=None,
             record=False, episode=0, decision="search",
             risk_weight=1.0, risk_decay=1.0, stage_blocks=4,
             stage_block_frames=4, beam_width=0, metrics_out=None,
             spacing=None, milestone_weight=0.0, milestone_mode="off",
             pain_weight=None, risk_lane="max", plan_out=None,
             candidate_out=None, margin_eps=0.0, veto_fraction=0.0,
             env_factory=None, rank_step=None,
             stage_demand_provider=None, rank_stats=None,
             rank_mode="order", rank_margin=0.0):
    """Run one deterministic evaluation episode.

    ``env_seed`` selects the world instance; when omitted the historical
    ``seed`` world is used.  With ``record=True`` the real transitions the
    planner actually executed are returned in the same schema as ``collect``
    (``obs``/``next_obs``/``history``/``signals``/``done``/``episode``), which
    is what the closed-loop (DAgger) path consumes.  No counterfactual
    prediction is recorded.

    An internally constructed ``stage_demand_provider`` is called before each
    decision with
    the current observation, pending-action prefix, and actions already
    executed. It returns ``(StageDemand, StageDemandHeuristic)`` or ``None``.
    If it implements ``observe_transition`` / ``reset_episode``, those hooks
    receive only actual episode transitions and boundaries.
    It is not read from the task configuration.
    """
    if cap < 1:
        raise ValueError("cap must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if step_stride < 1:
        raise ValueError("step_stride must be positive")
    if rank_step is not None and stage_demand_provider is not None:
        raise ValueError(
            "supply either a fixed rank_step or stage_demand_provider"
        )
    if stage_demand_provider is not None and int(beam_width) <= 0:
        raise ValueError(
            "stage_demand_provider requires positive beam_width"
        )
    lag = int(lag)
    env = (env_factory if env_factory is not None else latent[lag])(
        seed if env_seed is None else int(env_seed),
        spacing=None if spacing is None else float(spacing),
    )
    reset_provider = getattr(stage_demand_provider, "reset_episode", None)
    if reset_provider is not None:
        reset_provider(
            int(episode),
            int(seed if env_seed is None else env_seed),
        )
    prefix = np.zeros(lag, dtype=float)
    recent = [0.0] * lag
    actions = []
    last_action = None
    transitions = []
    # Epoch-level commitment: the tracker tells the planner which milestone
    # identity the stream is in right now, and the rate learned for that
    # identity is the action tendency the plan should honour.  Built for
    # either milestone mode so the coverage of the *decision-time* stream
    # against the memory learned on the *behaviour* stream is measurable -
    # a shift that used to make an unseen identity silently skip its term.
    tracker = None
    if milestone_mode in ("goal", "rate", "track"):
        threshold = getattr(agent, "milestone_threshold", None)
        frame = getattr(agent, "milestone_frame", None)
        if threshold is not None and frame is not None:
            tracker = MilestoneTracker(
                threshold, frame, scale=getattr(agent, "milestone_scale", None)
            )
    visited = [] if tracker is not None else None
    while not env.dead and env.t < cap:
        obs = np.asarray(env.obs(), dtype=float)
        agent_obs = obs[:int(agent.obs_dim)]
        active_rate = None
        active_goal = None
        if tracker is not None:
            tracker.observe(agent_obs)
            visited.append(tracker.key)
            if milestone_mode == "rate":
                active_rate = getattr(agent, "milestone_rate", {}).get(
                    tracker.key
                )
            elif milestone_mode in ("goal", "track"):
                # The subgoal is where the transition graph leads from here,
                # resolved per frame rather than fixed at fit time.
                active_goal = getattr(
                    agent, "milestone_goals_by_key", {}
                ).get(tracker.key)
        stage_demand = None
        demand_heuristic = None
        if stage_demand_provider is not None:
            request = stage_demand_provider(
                agent, agent_obs.copy(), prefix.copy(), tuple(actions),
            )
            if request is not None:
                if not isinstance(request, tuple) or len(request) != 2:
                    raise TypeError(
                        "stage_demand_provider must return "
                        "(StageDemand, StageDemandHeuristic) or None"
                    )
                stage_demand, demand_heuristic = request
        ledger: List[Tuple[float, float]] = []
        ledger_out = ledger if record else None
        collected = None
        callback = None
        if candidate_out is not None:
            collected = []
            callback = collected.append
        action = decision_action(
            env, world, agent, prefix, horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
            decision=decision, ledger_out=ledger_out,
            risk_weight=risk_weight,
            risk_decay=risk_decay,
            stage_blocks=stage_blocks,
            stage_block_frames=stage_block_frames,
            beam_width=beam_width,
            milestone_weight=milestone_weight,
            milestone_mode=milestone_mode,
            milestone_rate=active_rate,
            milestone_goal=active_goal,
            pain_weight=pain_weight,
            risk_lane=risk_lane,
            plan_out=plan_out,
            candidate_callback=callback,
            previous_action=last_action,
            margin_eps=margin_eps,
            veto_fraction=veto_fraction,
            rank_step=rank_step,
            stage_demand=stage_demand,
            demand_heuristic=demand_heuristic,
            rank_stats=rank_stats,
            rank_mode=rank_mode,
            rank_margin=rank_margin,
        )
        if candidate_out is not None:
            candidate_out.append(collected)
        history = np.asarray([float(action)] + recent[:lag], dtype=float)
        prediction_hook = getattr(agent, "prediction_branch", None) is not None
        if prediction_hook:
            predicted = world.predict(
                obs.reshape(1, -1), history.reshape(1, -1)
            )[0]
            agent.set_prediction(predicted[:int(agent.obs_dim)])
        actions.append(action)
        last_action = int(action)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(action)
        observe_provider = getattr(
            stage_demand_provider, "observe_transition", None
        )
        next_obs = None
        if prediction_hook or record or observe_provider is not None:
            next_obs = np.asarray(env.obs(), dtype=float)
        if prediction_hook:
            agent.observe_prediction(next_obs[:int(agent.obs_dim)])
        if record or observe_provider is not None:
            feedback = env.feedback()
            transition_record = dict(
                obs=obs,
                next_obs=next_obs,
                history=history,
                signals=np.asarray(
                    [feedback["dopamine"], feedback["pain"]], dtype=float
                ),
                terminal=bool(env.dead),
                done=bool(env.dead),
                episode=int(episode),
                # The planner's choice is deterministic: its rows are
                # already on the target policy, so their IS weight is one.
                propensity=1.0,
                # The executed plan's objective ledger (hazard share,
                # value share of the scalarized cost): the row's realized
                # remaining survival is the outcome these components must
                # predict (P4 objective learning).
                ledger=(ledger[0] if ledger else (0.0, 0.0)),
            )
            if record:
                transitions.append(transition_record)
            if observe_provider is not None:
                observe_provider(dict(transition_record))
        recent = [float(action)] + recent[:-1]
    if metrics_out is not None:
        # Task-level progress the world reports but the learner never sees:
        # how many progress milestones the episode reached.  Frames count
        # time; milestones count outcomes, which is the scale a frame-horizon
        # planner cannot reach on its own.
        metrics_out["passes"] = int(getattr(env, "passed", 0))
        if visited is not None:
            # How much of the planner's own stream the milestone memory -
            # learned on the behaviour stream - can actually speak to.
            metrics_out["milestone_coverage"] = milestone_coverage(
                visited, getattr(agent, "milestone_keys", ())
            )
    return env.t, actions, transitions


def run_seed(task):
    """Train one model and score it on the requested evaluation worlds.

    ``task`` is a plain dict so the worker payload stays picklable and new
    protocol fields do not require positional shuffling.  ``eval_seeds`` is
    explicit: when it is empty the historical self-evaluation is kept, with a
    warning printed by ``main``.  Each trained model is scored on every
    evaluation seed and every requested episode of that seed.
    """
    seed = int(task["seed"])
    steps = int(task.get("steps", 5000))
    coverage = float(task.get("coverage", 0.1))
    cap = int(task.get("cap", 6000))
    max_nodes = int(task.get("max_nodes", 5000))
    birth_threshold = float(task.get("birth_threshold", 0.45))
    horizon = int(task.get("horizon", 10))
    epochs = int(task.get("epochs", 5))
    lag = int(task.get("lag", LAG))
    component_kinds = tuple(task.get("component_kinds", ()))
    component_min_count = int(task.get("component_min_count", 8))
    max_components = int(task.get("max_components", 128))
    component_weight = float(task.get("component_weight", 0.5))
    component_gating = bool(task.get("component_gating", False))
    memory_weight = float(task.get("memory_weight", 0.1))
    association_rate = float(task.get("association_rate", 0.0))
    induce_features = bool(task.get("induce_features", True))
    precursor_labels = bool(task.get("precursor_labels", False))
    pain_horizon = int(task.get("pain_horizon", 0))
    voi_fraction = float(task.get("voi_fraction", 0.0))
    voi_target = int(task.get("voi_target", 3))
    feature_value_weight = float(task.get("feature_value_weight", 0.02))
    step_stride = int(task.get("step_stride", 2))
    stride_includes_lag = bool(task.get("stride_includes_lag", False))
    decision = str(task.get("decision", "search"))
    stage_blocks = int(task.get("stage_blocks", 4))
    stage_block_frames = int(task.get("stage_block_frames", 4))
    beam_width = int(task.get("beam_width", 0))
    eval_spacing = float(task.get("eval_spacing", 0.0))
    milestone_weight = float(task.get("milestone_weight", 0.0))
    milestone_mode = str(task.get("milestone_mode", "off"))
    ood_trust_scale = float(task.get("ood_trust_scale", 0.0))
    obs_trust = float(task.get("obs_trust", 1.0))
    feature_action_prior = float(task.get("feature_action_prior", 2.0))
    feature_value_prior = float(task.get("feature_value_prior", 4.0))
    value_features = bool(task.get("value_features", True))
    value_label_quantile = float(task.get("value_label_quantile", 0.8))
    subgraph_cells = bool(task.get("subgraph_cells", False))
    predicate_readouts = bool(task.get("predicate_readouts", False))
    event_memory_risk = bool(task.get("event_memory_risk", True))
    contrast_objective = bool(task.get("contrast_objective", False))
    value_trust_samples = float(task.get("value_trust_samples", 150.0))
    value_dispersion = bool(task.get("value_dispersion", False))
    control_risk_weight = float(task.get("control_risk_weight", 0.0))
    value_bank_only = bool(task.get("value_bank_only", False))
    iterative_induction = bool(task.get("iterative_induction", False))
    compose_readout = bool(task.get("compose_readout", True))
    importance_weighting = bool(task.get("importance_weighting", False))
    learn_objective = bool(task.get("learn_objective", True))
    curiosity_weight = float(task.get("curiosity_weight", 0.5))
    memory_credit = bool(task.get("memory_credit", False))
    rule_min_coverage = float(task.get("rule_min_coverage", 0.0))
    rule_min_primitive_coverage = float(
        task.get("rule_min_primitive_coverage", 0.0)
    )
    rule_validation_seed = int(task.get("rule_validation_seed", 0))
    rule_validation_frames = int(task.get("rule_validation_frames", 600))
    pairwise_features = bool(task.get("pairwise_features", False))
    branch_transitions = bool(task.get("branch_transitions", False))
    feature_backfit = bool(task.get("feature_backfit", False))
    action_edge_weight = float(task.get("action_edge_weight", 0.0))
    action_edge_trust_samples = float(
        task.get("action_edge_trust_samples", 50.0)
    )
    death_rules = bool(task.get("death_rules", False))
    risk_weight = float(task.get("risk_weight", 1.0))
    risk_decay = float(task.get("risk_decay", 1.0))
    risk_error_weight = bool(task.get("risk_error_weight", False))
    memory_feature_backfill = bool(
        task.get("memory_feature_backfill", False)
    )
    death_rule_compose = bool(task.get("death_rule_compose", False))
    death_rule_max_thresholds = int(
        task.get("death_rule_max_thresholds", 256)
    )
    death_rule_max_rules = int(task.get("death_rule_max_rules", 24))
    death_rule_max_literals = int(
        task.get("death_rule_max_literals", 5)
    )
    death_rule_boundary_margin = float(
        task.get("death_rule_boundary_margin", 0.0)
    )
    death_rule_envelope = bool(
        task.get("death_rule_envelope", False)
    )
    readout_propagation_gain = float(
        task.get("readout_propagation_gain", 0.0)
    )
    readout_propagation_steps = int(
        task.get("readout_propagation_steps", 1)
    )
    readout_propagation_edges = str(
        task.get("readout_propagation_edges", "transition")
    )
    voi_threat_min = float(task.get("voi_threat_min", 0.25))
    eval_episodes = int(task.get("eval_episodes", 1))
    eval_seeds = list(task.get("eval_seeds", ()))
    dagger_rounds = int(task.get("dagger_rounds", 0))
    dagger_episodes = int(task.get("dagger_episodes", 2))
    dagger_cap = int(task.get("dagger_cap", 1200))
    dagger_offset = int(task.get("dagger_offset", 50000))
    dagger_horizon = int(task.get("dagger_horizon", horizon))
    dagger_frames = int(task.get("dagger_frames", 0))

    value_horizon = int(task.get("value_horizon", 0))
    stage_memory = bool(task.get("stage_memory", False))
    stage_memory_quantile = float(task.get("stage_memory_quantile", 0.98))
    stage_memory_feature_source = str(
        task.get("stage_memory_feature_source", "branch")
    )

    world, agent = fit(
        seed, steps=steps, coverage=coverage,
        max_nodes=max_nodes, birth_threshold=birth_threshold,
        epochs=epochs, lag=lag,
        component_kinds=component_kinds,
        component_min_count=component_min_count,
        max_components=max_components,
        component_weight=component_weight,
        component_gating=component_gating,
        memory_weight=memory_weight,
        association_rate=association_rate,
        induce_features=induce_features,
        precursor_labels=precursor_labels,
        pain_horizon=(pain_horizon or None),
        voi_fraction=voi_fraction,
        voi_target=voi_target,
        feature_value_weight=feature_value_weight,
        dagger_rounds=dagger_rounds,
        dagger_episodes=dagger_episodes,
        dagger_cap=dagger_cap,
        dagger_offset=dagger_offset,
        dagger_horizon=dagger_horizon,
        dagger_frames=dagger_frames,
        step_stride=step_stride,
        stride_includes_lag=stride_includes_lag,
        value_horizon=value_horizon,
        ood_trust_scale=ood_trust_scale,
        obs_trust=obs_trust,
        feature_action_prior=feature_action_prior,
        feature_value_prior=feature_value_prior,
        value_features=value_features,
        value_label_quantile=value_label_quantile,
        subgraph_cells=subgraph_cells,
        predicate_readouts=predicate_readouts,
        event_memory_risk=event_memory_risk,
        contrast_objective=contrast_objective,
        value_trust_samples=value_trust_samples,
        value_dispersion=value_dispersion,
        control_risk_weight=control_risk_weight,
        value_bank_only=value_bank_only,
        iterative_induction=iterative_induction,
        compose_readout=compose_readout,
        importance_weighting=importance_weighting,
        learn_objective=learn_objective,
        voi_threat_min=voi_threat_min,
        curiosity_weight=curiosity_weight,
        memory_credit=memory_credit,
        rule_min_coverage=rule_min_coverage,
        rule_min_primitive_coverage=rule_min_primitive_coverage,
        rule_validation_seed=rule_validation_seed,
        rule_validation_frames=rule_validation_frames,
        pairwise_features=pairwise_features,
        branch_transitions=branch_transitions,
        feature_backfit=feature_backfit,
        risk_weight=risk_weight,
        risk_decay=risk_decay,
        risk_error_weight=risk_error_weight,
        memory_feature_backfill=memory_feature_backfill,
        action_edge_weight=action_edge_weight,
        action_edge_trust_samples=action_edge_trust_samples,
        death_rules=death_rules,
        death_rule_compose=death_rule_compose,
        death_rule_max_thresholds=death_rule_max_thresholds,
        death_rule_max_rules=death_rule_max_rules,
        death_rule_max_literals=death_rule_max_literals,
        death_rule_boundary_margin=death_rule_boundary_margin,
        death_rule_envelope=death_rule_envelope,
        readout_propagation_gain=readout_propagation_gain,
        readout_propagation_steps=readout_propagation_steps,
        readout_propagation_edges=readout_propagation_edges,
        stage_memory=stage_memory,
        stage_memory_quantile=stage_memory_quantile,
        stage_memory_feature_source=stage_memory_feature_source,
    )
    explicit_eval = bool(eval_seeds)
    if not explicit_eval:
        eval_seeds = [seed]
    rows = []
    # The ruler is unshaped: curiosity shapes training-time execution (its
    # transitions are real data), never the evaluation episode.
    with _curiosity_off(agent):
        for eval_seed in eval_seeds:
            for episode in range(eval_episodes):
                if explicit_eval:
                    # Match `collect`'s world-naming scheme so evaluation
                    # worlds are reproducible and disjoint from the
                    # training stream.
                    env_seed = int(eval_seed) * 10000 + episode
                elif episode == 0:
                    # Historical self-evaluation world.
                    env_seed = int(eval_seed)
                else:
                    env_seed = int(eval_seed) * 10000 + episode
                metrics = {}
                frames, actions, _ = evaluate(
                    seed, world, agent, cap=cap, horizon=horizon, lag=lag,
                    step_stride=step_stride,
                    stride_includes_lag=stride_includes_lag,
                    env_seed=env_seed,
                    decision=decision,
                    risk_weight=risk_weight,
                    risk_decay=risk_decay,
                    stage_blocks=stage_blocks,
                    stage_block_frames=stage_block_frames,
                    beam_width=beam_width,
                    metrics_out=metrics,
                    spacing=eval_spacing if eval_spacing > 0.0 else None,
                    milestone_weight=milestone_weight,
                    milestone_mode=milestone_mode,
                )
                rows.append(dict(
                    train_seed=seed,
                    eval_seed=int(eval_seed),
                    episode=episode,
                    env_seed=int(env_seed),
                    frames=int(frames),
                    passes=int(metrics.get("passes", 0)),
                    action_rate=(
                        float(np.mean(actions)) if actions else 0.0
                    ),
                    milestone_coverage=metrics.get(
                        "milestone_coverage", {}
                    ).get("fraction"),
                ))
    return dict(
        seed=seed,
        frames=rows[0]["frames"],
        action_rate=rows[0]["action_rate"],
        evals=rows,
        graph=agent.snapshot(),
        prediction=agent.prediction_summary(),
        stage_memory=(
            agent.stage_experience.snapshot()
            if getattr(agent, "stage_experience", None) is not None else None
        ),
        stage_resolver=(
            agent.stage_resolver.snapshot()
            if getattr(agent, "stage_resolver", None) is not None else None
        ),
        stage_demand=(
            agent.stage_demand_selector.snapshot()
            if getattr(agent, "stage_demand_selector", None) is not None
            else None
        ),
        stage_demand_heuristic=(
            agent.stage_demand_heuristic.snapshot()
            if getattr(agent, "stage_demand_heuristic", None) is not None
            else None
        ),
        stage_tree=(
            agent.stage_tree.snapshot()
            if getattr(agent, "stage_tree", None) is not None else None
        ),
        stage_tree_heuristic=(
            agent.stage_tree_heuristic.snapshot()
            if getattr(agent, "stage_tree_heuristic", None) is not None
            else None
        ),
        dagger=getattr(agent, "dagger_log", []),
        death_rules=getattr(agent, "death_rule_summary", None),
        milestone=getattr(agent, "milestone_summary", None),
        top_nodes=agent.top_nodes(5),
        top_edges=agent.top_edges(5),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=6000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--birth-threshold", type=float, default=0.45)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lag", type=int, default=LAG)
    parser.add_argument("--components", default="",
                        help="comma-separated runtime component kinds")
    parser.add_argument("--component-min-count", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--component-weight", type=float, default=0.5)
    parser.add_argument("--component-gating", action="store_true")
    parser.add_argument("--memory-weight", type=float, default=0.1)
    parser.add_argument("--association-rate", type=float, default=0.0)
    parser.add_argument("--no-features", action="store_true",
                        help="disable the compositional feature layer")
    parser.add_argument("--instant-labels", action="store_true",
                        help="deprecated alias: point-event pain is now the "
                             "default, so this flag is accepted but has no "
                             "effect")
    parser.add_argument("--precursor-labels", action="store_true",
                        help="opt into the historical precursor window "
                             "(P(pain within k), k = --pain-horizon); it "
                             "assigns pain to normal-flight states before a "
                             "collision, which is not world feedback")
    parser.add_argument("--pain-horizon", type=int, default=0,
                        help="precursor label window in entered states "
                             "(only with --precursor-labels); "
                             "0 uses the default lag + 1")
    parser.add_argument("--voi-fraction", type=float, default=0.0,
                        help="fraction of collection steps guided by the "
                             "counterfactual budget (two-stage collect)")
    parser.add_argument("--voi-target", type=int, default=3,
                        help="how many samples a (feature, action) cell "
                             "should have before the guide stops testing it")
    parser.add_argument("--voi-threat-min", type=float, default=0.25,
                        help="minimum calibrated threat rate a state must "
                             "read before the counterfactual guide tests "
                             "an action there (decision-critical targeting)")
    parser.add_argument("--feature-value-weight", type=float, default=0.02,
                        help="how much sequence-level value trades against "
                             "accumulated hazard in the search objective")
    parser.add_argument("--risk-weight", type=float, default=1.0,
                        help="scale on the accumulated hazard column "
                             "(1.0 = current behaviour)")
    parser.add_argument("--risk-decay", type=float, default=1.0,
                        help="per-step decay on the hazard column: later "
                             "predicted steps count less (1.0 = off)")
    parser.add_argument("--risk-error-weight", action="store_true",
                        help="down-weight the hazard read at steps where "
                             "the world model's measured rollout error is "
                             "large (default off)")
    parser.add_argument("--memory-feature-backfill", action="store_true",
                        help="after induction, link event memories to the "
                             "feature roots active at their members (the "
                             "memory risk channel is otherwise empty under "
                             "epochs=1)")
    parser.add_argument("--eval-seeds", default="",
                        help="held-out seed spec, e.g. 120-139; empty keeps "
                             "the legacy self-evaluation")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="episodes per evaluation seed")
    parser.add_argument("--step-stride", type=int, default=2,
                        help="query every Nth predicted state in search")
    parser.add_argument("--stride-includes-lag", action="store_true",
                        help="also query step == lag, the first predicted "
                             "state the plan action can affect")
    parser.add_argument("--decision", choices=("search", "cellq", "stage"),
                        default="search",
                        help="search = full MPC planning; cellq = read the "
                             "learned (feature, action) cells directly; "
                             "stage = score block-wise constant action "
                             "schedules over blocks * block-frames frames")
    parser.add_argument("--stage-blocks", type=int, default=4,
                        help="number of action blocks a stage-level "
                             "(--decision stage) schedule holds")
    parser.add_argument("--stage-block-frames", type=int, default=4,
                        help="frames each stage-level action block lasts")
    parser.add_argument("--beam-width", type=int, default=0,
                        help="beam width for the searched decision: 0 keeps "
                             "the exhaustive 2**horizon enumeration (bitwise "
                             "unchanged); positive values switch to beam "
                             "search, which keeps per-frame action timing at "
                             "a bounded candidate count")
    parser.add_argument("--eval-spacing", type=float, default=0.0,
                        help="pipe spacing of the evaluation worlds (0 = the "
                             "world default).  Raising it stretches the same "
                             "task over more frames, which is how the "
                             "time-scale invariance of a decision policy is "
                             "measured; training worlds are unaffected")
    parser.add_argument("--milestone-weight", type=float, default=0.0,
                        help="weight of the milestone shaping term: the "
                             "rollout endpoint is pulled toward the boundary "
                             "states the agent's own stream started "
                             "milestones in.  0 (default) leaves the "
                             "objective bitwise unchanged")
    parser.add_argument("--milestone-mode",
                        choices=("off", "goal", "rate", "plan", "track"),
                        default="off",
                        help="how the milestone term attaches: 'off' is "
                             "inert; 'goal' pulls a rollout endpoint toward "
                             "the modal boundary states of the agent's own "
                             "stream; 'rate' honours, per milestone identity, "
                             "the duration-weighted action rate the behavior "
                             "policy realized there - an epoch-level action "
                             "commitment that carries no position and no "
                             "time scale; 'plan' charges each candidate the "
                             "expected feedback deficit of the stage its own "
                             "endpoint lands in (the two-stage walk of the "
                             "learned transition table); the frame horizon "
                             "is unchanged")
    parser.add_argument("--milestone-stage",
                        choices=("geometry", "strength", "signature"),
                        default="strength",
                        help="how stages are cut: 'geometry' uses the "
                             "observation stream's boundary statistic; "
                             "'strength' uses the learned graph (a node's "
                             "strength rises where the action-conditioned "
                             "future changes) and keeps the frame identity; "
                             "'signature' additionally names each stage by "
                             "the pooled cluster of its future")
    parser.add_argument("--milestone-boundary-k", type=float, default=3.0,
                        help="stage-boundary threshold in pooled score "
                             "quantiles; the same statistical rule (three "
                             "quantiles above the 90th) the geometric "
                             "segmenter uses")
    parser.add_argument("--hazard-readout",
                        choices=("calibrated", "raw"), default="calibrated",
                        help="what the planner reads as hazard: 'calibrated' "
                             "is the isotonic rate (a level; on a rare-event "
                             "target it collapses most of the score range "
                             "into one plateau), 'raw' is the weighted "
                             "channel sum (a ranking, uncalibrated)")
    parser.add_argument("--value-readout",
                        choices=("cells", "association"), default="cells",
                        help="what the planner reads as endpoint value: "
                             "'cells' is the per-feature cell (empty at "
                             "epochs=1); 'association' is the counting "
                             "readout - the weighted realised quality of the "
                             "nodes the stream met after this one")
    parser.add_argument("--veto-fraction", type=float, default=0.0,
                        help="exclude the worst fraction of candidate "
                             "endpoints by the hazard readout within each "
                             "decision (0 = off); the cutoff is a "
                             "within-decision quantile, so only the ranking "
                             "is used")
    parser.add_argument("--value-horizon", type=int, default=0,
                        help="k-step bootstrap target for the sequence-value "
                             "cells; 0 keeps the full-episode MC return")
    parser.add_argument("--ood-trust-scale", type=float, default=0.0,
                        help="shrink prototype readouts toward the population "
                             "baseline beyond this distance (0 = off; the "
                             "birth threshold is a natural setting)")
    parser.add_argument("--obs-trust", type=float, default=1.0,
                        help="weight of the observation-anchored prototype/"
                             "memory channel against induced feature channels")
    parser.add_argument("--feature-action-prior", type=float, default=2.0,
                        help="pseudo-counts shrinking pain cells toward the "
                             "feature scalar; lower = more trust in verified "
                             "action-conditioned cells")
    parser.add_argument("--feature-value-prior", type=float, default=4.0,
                        help="pseudo-counts shrinking value cells toward the "
                             "baseline; lower = more trust in verified "
                             "success cells")
    parser.add_argument("--no-value-features", action="store_true",
                        help="keep the value readout on the pain-coverage "
                             "roots instead of adding a dedicated "
                             "high-return bank")
    parser.add_argument("--value-quantile", type=float, default=0.8,
                        help="return-distribution quantile that defines the "
                             "high-return labels for value-channel "
                             "induction")
    parser.add_argument("--subgraph-cells", action="store_true",
                        help="train and read feature statistics on rule "
                             "branches (most specific active member wins); "
                             "measured inert on survival, off by default")
    parser.add_argument("--predicate-readouts", action="store_true",
                        help="drop the prototype channel from the risk "
                             "readout; prototypes stay as event memory")
    parser.add_argument("--no-event-memory-risk", action="store_true",
                        help="drop the feature-keyed event-memory readout "
                             "from the risk max-combine")
    parser.add_argument("--contrast-objective", action="store_true",
                        help="F1: per-step cost is the action contrast and "
                             "the endpoint cost is the value term alone, "
                             "so genuinely hazardous bands no longer "
                             "penalize the plans that must cross them")
    parser.add_argument("--value-trust-samples", type=float, default=150.0,
                        help="D4: mean value-cell sample count at which the "
                             "value channel earns half weight; 0 disables "
                             "the trust scaling")
    parser.add_argument("--value-dispersion-labels", action="store_true",
                        help="induce the value bank on both return extremes "
                             "(top and bottom quantiles) so its predicates "
                             "tile the decision-critical regions")
    parser.add_argument("--control-risk-weight", type=float, default=0.0,
                        help="threat grading: weight of the controllable "
                             "excess (the avoidable share a plan takes by "
                             "its own action) on top of the real threat; "
                             "0 keeps controllable risk free to take")
    parser.add_argument("--value-bank-only", action="store_true",
                        help="read the value channel from the dedicated "
                             "value-bank roots only, without the risk "
                             "bank's drifting-continuation cells")
    parser.add_argument("--curiosity-weight", type=float, default=0.5,
                        help="decision-side curiosity: the maximum absolute "
                             "share the potential-difference term may take "
                             "in the objective (phi in (0,1] so the term is "
                             "bounded by +/-w); it shapes training-time "
                             "execution only - evaluation always runs with "
                             "w=0. 0 disables it")
    parser.add_argument("--memory-credit", action="store_true",
                        help="on a pain event, correct earlier memories' "
                             "risk rates by an eligibility trace and link "
                             "the event memory back to the preceding "
                             "context; the pain itself stays a point event")
    parser.add_argument("--rule-min-coverage", type=float, default=0.0,
                        help="induction gate: minimum fraction of the "
                             "validation states a candidate rule must "
                             "activate on (0 = off)")
    parser.add_argument("--rule-min-primitive-coverage", type=float,
                        default=0.0,
                        help="induction gate: every conjunct/branch "
                             "primitive must activate on at least this "
                             "fraction of the validation states "
                             "(0 = off)")
    parser.add_argument("--rule-validation-seed", type=int, default=0,
                        help="cross-world activation check: collect this "
                             "many frames from an independent world seed "
                             "and use its states (no labels) for the "
                             "coverage gates; 0 = use the training split")
    parser.add_argument("--rule-validation-frames", type=int, default=600,
                        help="frames collected from --rule-validation-seed")
    parser.add_argument("--pairwise-features", dest="pairwise_features",
                        action="store_true",
                        help="add generic pairwise coordinate differences "
                             "to the induction grammar (six-seed A/B null; "
                             "opt-in)")
    parser.add_argument("--no-pairwise-features", dest="pairwise_features",
                        action="store_false",
                        help="deprecated alias; pairwise differences are "
                             "off by default")
    parser.add_argument("--feature-backfit", action="store_true",
                        help="replay the stream with features only after "
                             "induction: feature cells, value cells and the "
                             "prediction ledger learn; prototypes, edges and "
                             "memories keep their single-pass statistics")
    parser.set_defaults(pairwise_features=False)
    parser.add_argument("--branch-transitions", action="store_true",
                        help="add decayed recent branch sources to "
                             "action-edge evidence (structure-only: the "
                             "planner does not read action-edge Q yet)")
    parser.add_argument("--action-edge-weight", type=float, default=0.0,
                        help="weight of the normalized action-edge Q term "
                             "in the planner cost (0 = the Q read is off)")
    parser.add_argument("--action-edge-trust", type=float, default=50.0,
                        help="evidence pseudo-count for trusting the "
                             "action-edge Q read")
    parser.add_argument("--death-rules", action="store_true",
                        help="train a 0/1 death-condition rule set and use "
                             "it as a hard planner constraint; off by "
                             "default")
    parser.add_argument("--death-rule-compose", action="store_true",
                        help="allow nested AND/OR/XOR/NOT composition in "
                             "the death-rule learner")
    parser.add_argument("--death-rule-thresholds", type=int, default=256,
                        help="candidate threshold budget per coordinate")
    parser.add_argument("--death-rule-max-rules", type=int, default=24,
                        help="maximum death/safe rules per class")
    parser.add_argument("--death-rule-max-literals", type=int, default=5,
                        help="maximum literals in one conjunctive rule")
    parser.add_argument("--death-rule-margin", type=float, default=0.0,
                        help="boundary margin as a fraction of each "
                             "coordinate's observed range; states inside "
                             "the margin are UNKNOWN, not safe")
    parser.add_argument("--death-rule-envelope", action="store_true",
                        help="treat states inside the boundary margin as "
                             "death for the planner's hard constraint "
                             "(model-error-aware envelope); off by default")
    parser.add_argument("--readout-propagation", type=float, default=0.0,
                        help="relax the readout's soft-neighbour seed "
                             "weights over learned graph edges before "
                             "reading values/rates, so the readout reflects "
                             "the topological neighbourhood instead of the "
                             "Euclidean one (0 = off, the geometric readout)")
    parser.add_argument("--readout-propagation-steps", type=int, default=1,
                        help="relaxation steps for --readout-propagation")
    parser.add_argument("--readout-propagation-edges",
                        choices=("transition", "association", "both"),
                        default="transition",
                        help="which learned edges carry the readout "
                             "relaxation")
    parser.add_argument("--iterative-induction", action="store_true",
                        help="F5: re-induce gap banks every epoch, with "
                             "the previous generation's missed positives "
                             "as the next generation's targets")
    parser.add_argument("--no-compose-readout", dest="compose_readout",
                        action="store_false",
                        help="keep the hand-written max-combination in "
                             "pain_signal_batch instead of the stacked "
                             "non-negative composition fitted to the "
                             "precursor labels")
    parser.add_argument("--importance-weighting", action="store_true",
                        help="weight each window label by the clipped "
                             "product of uniform-target propensity "
                             "ratios over the window's governing "
                             "actions, rebalancing the cells' "
                             "state-action mix away from the behavior "
                             "policy's occupancy")
    parser.add_argument("--no-learn-objective", dest="learn_objective",
                        action="store_false",
                        help="keep the hand-set value weight in the "
                             "planning objective instead of selecting "
                             "it from the planner's executed decisions "
                             "by rank agreement with realized survival")
    parser.add_argument("--report", default="",
                        help="write per-episode rows and summaries as JSON")
    parser.add_argument("--dagger-rounds", type=int, default=0,
                        help="closed-loop rounds: record the planner's own "
                             "executed trajectories and retrain")
    parser.add_argument("--dagger-episodes", type=int, default=2,
                        help="planner episodes recorded per DAgger round")
    parser.add_argument("--dagger-cap", type=int, default=1200,
                        help="frame cap per recorded planner episode")
    parser.add_argument("--dagger-offset", type=int, default=50000,
                        help="episode-id offset for recorded planner data")
    parser.add_argument("--dagger-horizon", type=int, default=0,
                        help="search horizon while recording planner data; "
                             "0 uses --horizon")
    parser.add_argument("--dagger-frames", type=int, default=0,
                        help="on-policy frame budget per DAgger round: stop "
                             "recording once this many executed frames were "
                             "collected (0 = use --dagger-episodes as-is)")
    parser.add_argument("--stage-memory", action="store_true",
                        help="build the self-supervised stage experience from "
                             "the realized stream (read-only; planner unwired)")
    parser.add_argument("--stage-memory-quantile", type=float, default=0.98,
                        help="upper-tail quantile for the generic key-signal "
                             "detector's channels")
    parser.add_argument("--stage-feature-source",
                        choices=("branch", "prototype"), default="branch",
                        help="read-only feature source for stage demand "
                             "samples: induced feature roots (sparse) or the "
                             "runtime's nearest prototype (dense)")
    parser.add_argument("--benchmark", action="store_true",
                        help="standard ruler: 1500 offline frames replayed "
                             "once, then one 5000-frame free-control round "
                             "and retrain; evaluation cap 5000 frames")
    parser.add_argument("--fast", action="store_true",
                        help="small development preset: 1500 steps, 3 epochs, "
                             "1000 nodes, 400-frame evaluation")
    args = parser.parse_args()
    if args.fast:
        args.steps = 1500
        args.epochs = 3
        args.max_nodes = 1000
        args.cap = 400
        args.horizon = 8
    if args.benchmark:
        # The benchmark fixes both halves of the ruler: the offline leg is
        # the fast bootstrap (1500 frames replayed once - no multi-epoch
        # overfitting) and the online leg is one free-control round of 5000
        # executed frames.  The evaluation cap matches the online budget,
        # so a model that survives is not censored.
        args.steps = 1500
        args.epochs = 1
        args.max_nodes = 1000
        args.horizon = 8
        args.cap = 5000
        args.dagger_cap = 5000
        if args.dagger_rounds == 0:
            args.dagger_rounds = 1
        if args.dagger_frames == 0:
            args.dagger_frames = 5000
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.coverage <= 1.0:
        raise ValueError("--coverage must be in [0, 1]")
    if args.cap < 1:
        raise ValueError("--cap must be positive")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.stage_blocks < 1:
        raise ValueError("--stage-blocks must be positive")
    if args.stage_block_frames < 1:
        raise ValueError("--stage-block-frames must be positive")
    if args.beam_width < 0:
        raise ValueError("--beam-width must be non-negative")
    if args.eval_spacing < 0.0:
        raise ValueError("--eval-spacing must be non-negative")
    if args.milestone_weight < 0.0:
        raise ValueError("--milestone-weight must be non-negative")
    if args.milestone_boundary_k <= 0.0:
        raise ValueError("--milestone-boundary-k must be positive")
    if not (0.0 <= args.veto_fraction < 1.0):
        raise ValueError("--veto-fraction must be in [0, 1)")
    if args.max_nodes < 1:
        raise ValueError("--max-nodes must be positive")
    if args.birth_threshold <= 0.0:
        raise ValueError("--birth-threshold must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not 0.5 < args.stage_memory_quantile < 1.0:
        raise ValueError("--stage-memory-quantile must be in (0.5, 1)")
    if args.lag < 0:
        raise ValueError("--lag must be non-negative")
    if args.pain_horizon < 0:
        raise ValueError("--pain-horizon must be non-negative")
    if not 0.0 <= args.voi_fraction < 1.0:
        raise ValueError("--voi-fraction must be in [0, 1)")
    if args.voi_target < 1:
        raise ValueError("--voi-target must be positive")
    if args.feature_value_weight < 0.0:
        raise ValueError("--feature-value-weight must be non-negative")
    if args.curiosity_weight < 0.0:
        raise ValueError("--curiosity-weight must be non-negative")
    if not 0.0 <= args.rule_min_coverage <= 1.0:
        raise ValueError("--rule-min-coverage must be in [0, 1]")
    if not 0.0 <= args.rule_min_primitive_coverage <= 1.0:
        raise ValueError("--rule-min-primitive-coverage must be in [0, 1]")
    if args.rule_validation_seed < 0:
        raise ValueError("--rule-validation-seed must be non-negative")
    if args.rule_validation_frames < 1:
        raise ValueError("--rule-validation-frames must be positive")
    if not 0.0 < args.value_quantile < 1.0:
        raise ValueError("--value-quantile must be in (0, 1)")
    if args.component_min_count < 1:
        raise ValueError("--component-min-count must be positive")
    if args.max_components < 0:
        raise ValueError("--max-components must be non-negative")
    if args.component_weight < 0.0:
        raise ValueError("--component-weight must be non-negative")
    if not 0.0 <= args.memory_weight <= 1.0:
        raise ValueError("--memory-weight must be in [0, 1]")
    if not 0.0 <= args.association_rate <= 1.0:
        raise ValueError("--association-rate must be in [0, 1]")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be positive")
    if args.step_stride < 1:
        raise ValueError("--step-stride must be positive")
    if args.dagger_rounds < 0:
        raise ValueError("--dagger-rounds must be non-negative")
    if args.dagger_rounds and args.dagger_episodes < 1:
        raise ValueError("--dagger-episodes must be positive")
    if args.dagger_cap < 1:
        raise ValueError("--dagger-cap must be positive")
    if args.dagger_offset < 0:
        raise ValueError("--dagger-offset must be non-negative")
    if args.dagger_horizon < 0:
        raise ValueError("--dagger-horizon must be non-negative")
    if args.dagger_horizon == 0:
        args.dagger_horizon = args.horizon
    if args.dagger_frames < 0:
        raise ValueError("--dagger-frames must be non-negative")
    if args.dagger_frames > 0 and args.dagger_rounds < 1:
        raise ValueError("--dagger-frames requires --dagger-rounds")
    if args.value_horizon < 0:
        raise ValueError("--value-horizon must be non-negative")
    if args.ood_trust_scale < 0.0:
        raise ValueError("--ood-trust-scale must be non-negative")
    if not 0.0 <= args.obs_trust <= 1.0:
        raise ValueError("--obs-trust must be in [0, 1]")
    if args.feature_action_prior < 0.0:
        raise ValueError("--feature-action-prior must be non-negative")
    if args.feature_value_prior < 0.0:
        raise ValueError("--feature-value-prior must be non-negative")
    component_kinds = tuple(
        kind.strip() for kind in args.components.split(",") if kind.strip()
    )
    eval_seeds = parse_seed_spec(args.eval_seeds)

    seeds = [seed for seed in SEEDS if seed >= args.seed][:args.seeds]
    if len(seeds) < args.seeds:
        seeds = list(range(args.seed, args.seed + args.seeds))
    if eval_seeds:
        overlap = sorted(set(seeds) & set(eval_seeds))
        if overlap:
            print(
                "warning: evaluation seeds overlap training seeds: %s"
                % ",".join(str(seed) for seed in overlap)
            )
    else:
        print(
            "warning: --eval-seeds is empty; using legacy self-evaluation "
            "(training seed == evaluation seed)"
        )
    tasks = [
        dict(
            seed=seed,
            steps=args.steps,
            coverage=args.coverage,
            cap=args.cap,
            max_nodes=args.max_nodes,
            birth_threshold=args.birth_threshold,
            horizon=args.horizon,
            epochs=args.epochs,
            lag=args.lag,
            component_kinds=component_kinds,
            component_min_count=args.component_min_count,
            max_components=args.max_components,
            component_weight=args.component_weight,
            component_gating=args.component_gating,
            memory_weight=args.memory_weight,
            association_rate=args.association_rate,
            induce_features=not args.no_features,
            precursor_labels=args.precursor_labels,
            pain_horizon=args.pain_horizon,
            voi_fraction=args.voi_fraction,
            voi_target=args.voi_target,
            voi_threat_min=args.voi_threat_min,
            feature_value_weight=args.feature_value_weight,
            step_stride=args.step_stride,
            stride_includes_lag=args.stride_includes_lag,
            decision=args.decision,
            stage_blocks=args.stage_blocks,
            stage_block_frames=args.stage_block_frames,
            beam_width=args.beam_width,
            eval_spacing=args.eval_spacing,
            milestone_weight=args.milestone_weight,
            milestone_mode=args.milestone_mode,
                veto_fraction=args.veto_fraction,
            value_horizon=args.value_horizon,
            ood_trust_scale=args.ood_trust_scale,
            obs_trust=args.obs_trust,
            feature_action_prior=args.feature_action_prior,
            feature_value_prior=args.feature_value_prior,
            value_features=not args.no_value_features,
            value_label_quantile=args.value_quantile,
            subgraph_cells=args.subgraph_cells,
            predicate_readouts=args.predicate_readouts,
            event_memory_risk=not args.no_event_memory_risk,
            contrast_objective=args.contrast_objective,
            value_trust_samples=args.value_trust_samples,
            value_dispersion=args.value_dispersion_labels,
            control_risk_weight=args.control_risk_weight,
            value_bank_only=args.value_bank_only,
            iterative_induction=args.iterative_induction,
            compose_readout=args.compose_readout,
            importance_weighting=args.importance_weighting,
            learn_objective=args.learn_objective,
            curiosity_weight=args.curiosity_weight,
            memory_credit=args.memory_credit,
            rule_min_coverage=args.rule_min_coverage,
            rule_min_primitive_coverage=args.rule_min_primitive_coverage,
            rule_validation_seed=args.rule_validation_seed,
            rule_validation_frames=args.rule_validation_frames,
            pairwise_features=args.pairwise_features,
            branch_transitions=args.branch_transitions,
            feature_backfit=args.feature_backfit,
            risk_weight=args.risk_weight,
            risk_decay=args.risk_decay,
            risk_error_weight=args.risk_error_weight,
            memory_feature_backfill=args.memory_feature_backfill,
            action_edge_weight=args.action_edge_weight,
            action_edge_trust_samples=args.action_edge_trust,
            death_rules=args.death_rules,
            death_rule_compose=args.death_rule_compose,
            death_rule_max_thresholds=args.death_rule_thresholds,
            death_rule_max_rules=args.death_rule_max_rules,
            death_rule_max_literals=args.death_rule_max_literals,
            death_rule_boundary_margin=args.death_rule_margin,
            death_rule_envelope=args.death_rule_envelope,
            readout_propagation_gain=args.readout_propagation,
            readout_propagation_steps=args.readout_propagation_steps,
            readout_propagation_edges=args.readout_propagation_edges,
            milestone_stage=args.milestone_stage,
            milestone_boundary_k=args.milestone_boundary_k,
            hazard_readout=args.hazard_readout,
            value_readout=args.value_readout,
            eval_seeds=eval_seeds,
            eval_episodes=args.eval_episodes,
            dagger_rounds=args.dagger_rounds,
            dagger_episodes=args.dagger_episodes,
            dagger_cap=args.dagger_cap,
            dagger_offset=args.dagger_offset,
            dagger_horizon=args.dagger_horizon,
            dagger_frames=args.dagger_frames,
            stage_memory=args.stage_memory,
            stage_memory_quantile=args.stage_memory_quantile,
            stage_memory_feature_source=args.stage_feature_source,
        )
        for seed in seeds
    ]
    def _drain(futures):
        """Collect worker results, streaming one line per finished model."""
        rows = []
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            frames = [item["frames"] for item in row["evals"]]
            print(
                " model %3d done: held-out mean %.1f over %d worlds"
                % (row["seed"], float(np.mean(frames)), len(frames)),
                flush=True,
            )
        return rows

    if args.jobs == 1:
        rows = []
        for task in tasks:
            row = run_seed(task)
            frames = [item["frames"] for item in row["evals"]]
            print(
                " model %3d done: held-out mean %.1f over %d worlds"
                % (row["seed"], float(np.mean(frames)), len(frames)),
                flush=True,
            )
            rows.append(row)
    elif FREE_THREADED:
        # The free-threaded build runs seed-level Python and numpy work in
        # parallel without pickling graphs between workers.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows = _drain([
                pool.submit(run_seed, task) for task in tasks
            ])
    else:
        # A GIL build still gets true parallelism from independent seeds; the
        # payload per seed is small (plain dicts and lists).
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = _drain([
                pool.submit(run_seed, task) for task in tasks
            ])
    rows.sort(key=lambda row: row["seed"])

    print("train  concepts  graph_nodes  action_edges  components")
    for row in rows:
        graph = row["graph"]
        print(" %4d  %8d  %11d  %12d  %10d"
              % (row["seed"], graph["prototypes"], graph["nodes"],
                 graph["action_edges"], graph["components"]))

    for row in rows:
        if row["dagger"]:
            parts = [
                "r%d +%d steps (%s frames)"
                % (item["round"], item["planner_steps"],
                   "/".join(str(value) for value in item["planner_frames"]))
                for item in row["dagger"]
            ]
            print(" dagger seed %d: %s" % (row["seed"], ", ".join(parts)))

    eval_rows = [row for result in rows for row in result["evals"]]
    print("\ntrain  eval  ep  env_seed  frames  passes  action_rate")
    for row in eval_rows:
        print(" %5d  %4d  %2d  %8d  %6d  %6d  %10.3f"
              % (row["train_seed"], row["eval_seed"], row["episode"],
                 row["env_seed"], row["frames"], row["passes"],
                 row["action_rate"]))

    per_train = group_summaries(eval_rows, group="train_seed")
    per_eval = group_summaries(eval_rows, group="eval_seed")
    overall = cluster_summary(eval_rows, cluster="train_seed")
    per_train_passes = group_summaries(
        eval_rows, value="passes", group="train_seed"
    )
    overall_passes = cluster_summary(
        eval_rows, value="passes", cluster="train_seed"
    )
    print("\nframes per training seed (mean over evaluation worlds)")
    for key, summary in per_train.items():
        print(" train %4s  n=%2d  mean %6.1f  median %6.1f  range %d-%d"
              % (key, summary["n"], summary["mean"], summary["median"],
                 summary["min"], summary["max"]))
    print("frames per evaluation seed (mean over training seeds)")
    for key, summary in per_eval.items():
        print(" eval  %4s  n=%2d  mean %6.1f  median %6.1f"
              % (key, summary["n"], summary["mean"], summary["median"]))
    print("\nframes overall: n=%d mean %.1f median %.1f IQR %.1f [%.1f, %.1f] "
          "range %d-%d"
          % (overall["n"], overall["mean"], overall["median"],
             overall["iqr"], overall["q1"], overall["q3"],
             overall["min"], overall["max"]))
    print("bootstrap 95%% CI over %d training seeds: [%.1f, %.1f]"
          % (overall["clusters"], overall["cluster_ci_low"],
             overall["cluster_ci_high"]))
    print("\nprogress milestones overall: n=%d mean %.2f median %.1f "
          "range %d-%d  (share of episodes with progress: %.3f)"
          % (overall_passes["n"], overall_passes["mean"],
             overall_passes["median"], int(overall_passes["min"]),
             int(overall_passes["max"]),
             float(np.mean([row["passes"] > 0 for row in eval_rows]))))
    print("bootstrap 95%% CI over %d training seeds: [%.2f, %.2f]"
          % (overall_passes["clusters"], overall_passes["cluster_ci_low"],
             overall_passes["cluster_ci_high"]))
    coverages = [row["milestone_coverage"] for row in eval_rows
                 if row.get("milestone_coverage") is not None]
    if coverages:
        # The milestone memory is learned on the behaviour stream and applied
        # to the planner's; this is how much of the planner's own stream it
        # can actually speak to.  Low coverage is a distribution shift, not a
        # weight to tune.
        coverages = np.asarray(coverages, dtype=float)
        print("milestone memory coverage of the planner's stream: "
              "mean %.3f median %.3f min %.3f  (fraction of frames whose "
              "identity the memory had seen)"
              % (coverages.mean(), float(np.median(coverages)),
                 coverages.min()))
    if args.stage_memory:
        census = rows[0].get("stage_memory")
        if census:
            # Read-only diagnostic: how many realized episodes produced
            # stages, how many stages carry a signal key (the rest share the
            # empty root key), and how often the internal feature masks fired.
            print(
                "stage experience (seed %d): %d/%d episodes, %d stages "
                "(%.2f/episode), keyed %.2f, feature coverage %.2f, "
                "%d demand samples"
                % (
                    rows[0]["seed"], census["episodes"],
                    census["episodes"] + census["episodes_skipped"],
                    census["stages"], census["mean_stages_per_episode"],
                    (
                        float(census["keyed_stages"]) / census["stages"]
                        if census["stages"] else 0.0
                    ),
                    census["feature_coverage"], census["demand_samples"],
                )
            )
        resolver = rows[0].get("stage_resolver")
        if resolver:
            print(
                "stage resolver (seed %d): %d prototypes, %d stage types, "
                "%d training rows, radius %.3f, gates support>=%d / "
                "episodes>=%d / p>=%.2f / margin>=%.2f"
                % (
                    rows[0]["seed"], resolver["prototypes"],
                    resolver["stage_types"], resolver["training_rows"],
                    resolver["radius"], resolver["min_support"],
                    resolver["min_episodes"], resolver["min_probability"],
                    resolver["min_margin"],
                )
            )
        selector = rows[0].get("stage_demand")
        if selector:
            readout = selector["readout"]
            print(
                "stage demand selector (seed %d): readout samples %d, "
                "target mean %+.3f std %.3f, min outcome episodes %d, "
                "planner unwired"
                % (
                    rows[0]["seed"], readout["samples"],
                    readout["target_mean"], readout["target_std"],
                    selector["min_outcome_episodes"],
                )
            )
        heuristic = rows[0].get("stage_demand_heuristic")
        if heuristic:
            print(
                "stage demand heuristic (seed %d): %d samples, "
                "%d context-feature pairs, %d context outcomes, "
                "feature source %s"
                % (
                    rows[0]["seed"], heuristic["samples"],
                    heuristic["context_feature_pairs"],
                    heuristic["context_outcomes"],
                    rows[0].get("stage_memory", {}).get("feature_source"),
                )
            )
        tree = rows[0].get("stage_tree")
        if tree:
            print(
                "stage tree (seed %d): %d types, %d merges, %d roots, "
                "max level %d"
                % (
                    rows[0]["seed"], tree["stage_types"], tree["merges"],
                    tree["roots"], tree["max_level"],
                )
            )
        tree_heuristic = rows[0].get("stage_tree_heuristic")
        if tree_heuristic:
            print(
                "stage tree heuristic (seed %d): %d samples, "
                "%d context-feature pairs, %d context outcomes"
                % (
                    rows[0]["seed"], tree_heuristic["samples"],
                    tree_heuristic["context_feature_pairs"],
                    tree_heuristic["context_outcomes"],
                )
            )
    if overall["clusters"] < 5:
        print("warning: fewer than 5 trained models; the interval is "
              "descriptive, not a significance test")

    if args.report:
        report = dict(
            config=dict(
                training_seeds=seeds,
                eval_seeds=eval_seeds,
                eval_episodes=args.eval_episodes,
                steps=args.steps,
                coverage=args.coverage,
                cap=args.cap,
                horizon=args.horizon,
                epochs=args.epochs,
                max_nodes=args.max_nodes,
                lag=args.lag,
                step_stride=args.step_stride,
                stride_includes_lag=args.stride_includes_lag,
                decision=args.decision,
                stage_blocks=args.stage_blocks,
                stage_block_frames=args.stage_block_frames,
                beam_width=args.beam_width,
                milestone_weight=args.milestone_weight,
                milestone_mode=args.milestone_mode,
                veto_fraction=args.veto_fraction,
                eval_spacing=args.eval_spacing,
                value_horizon=args.value_horizon,
                ood_trust_scale=args.ood_trust_scale,
                obs_trust=args.obs_trust,
                feature_action_prior=args.feature_action_prior,
                feature_value_prior=args.feature_value_prior,
                feature_value_weight=args.feature_value_weight,
                features=not args.no_features,
                precursor_labels=args.precursor_labels,
                value_features=not args.no_value_features,
                value_label_quantile=args.value_quantile,
                subgraph_cells=args.subgraph_cells,
                predicate_readouts=args.predicate_readouts,
                event_memory_risk=not args.no_event_memory_risk,
                contrast_objective=args.contrast_objective,
                value_trust_samples=args.value_trust_samples,
                value_dispersion=args.value_dispersion_labels,
                control_risk_weight=args.control_risk_weight,
                value_bank_only=args.value_bank_only,
                iterative_induction=args.iterative_induction,
                self_evaluation=not bool(eval_seeds),
                dagger_rounds=args.dagger_rounds,
                dagger_episodes=args.dagger_episodes,
                dagger_cap=args.dagger_cap,
                dagger_offset=args.dagger_offset,
                dagger_horizon=args.dagger_horizon,
                stage_memory=args.stage_memory,
                stage_memory_quantile=args.stage_memory_quantile,
                stage_feature_source=args.stage_feature_source,
            ),
            rows=eval_rows,
            milestones=[
                dict(seed=row["seed"], summary=row["milestone"])
                for row in rows
            ],
            dagger=[
                dict(seed=row["seed"], rounds=row["dagger"])
                for row in rows
            ],
            stage_memory=[
                dict(seed=row["seed"], census=row["stage_memory"])
                for row in rows
                if row.get("stage_memory") is not None
            ],
            stage_resolver=[
                dict(seed=row["seed"], census=row["stage_resolver"])
                for row in rows
                if row.get("stage_resolver") is not None
            ],
            stage_demand=[
                dict(seed=row["seed"], census=row["stage_demand"])
                for row in rows
                if row.get("stage_demand") is not None
            ],
            stage_demand_heuristic=[
                dict(seed=row["seed"], census=row["stage_demand_heuristic"])
                for row in rows
                if row.get("stage_demand_heuristic") is not None
            ],
            stage_tree=[
                dict(seed=row["seed"], census=row["stage_tree"])
                for row in rows
                if row.get("stage_tree") is not None
            ],
            stage_tree_heuristic=[
                dict(seed=row["seed"], census=row["stage_tree_heuristic"])
                for row in rows
                if row.get("stage_tree_heuristic") is not None
            ],
            per_train_seed=per_train,
            per_eval_seed=per_eval,
            overall=overall,
            per_train_seed_passes=per_train_passes,
            overall_passes=overall_passes,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("wrote %s" % args.report)

    print("\nlearned high-pain nodes (seed %d):" % rows[0]["seed"])
    for node in rows[0]["top_nodes"]:
        features = node["features"]
        preview = (
            "n/a" if features is None
            else "[" + ", ".join("%+.2f" % value for value in features[:6]) + "]"
        )
        print("  %-8s count=%4d pain=%.3f dopamine=%.2f state=%s"
              % (node["id"], node["count"], node["pain"],
                 node["dopamine"], preview))

    print("\nlearned high-pain action edges (seed %d):" % rows[0]["seed"])
    for edge in rows[0]["top_edges"]:
        print("  %-8s -a%d-> %-8s count=%4d pain=%.3f dopamine=%.2f"
              % (edge["source"], edge["action"], edge["target"],
                 edge["count"], edge["pain"], edge["dopamine"]))


if __name__ == "__main__":
    main()
