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
        readout_propagation_edges: str = "transition",
        online_feature_learning: bool = False,
        online_feature_options=None):
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
        online_feature_learning=online_feature_learning,
        online_feature_options=online_feature_options,
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
                agent.learn_episode(
                    episode, learn_online=(epoch == 0),
                )
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
        option_tolerance=0.05,
        option_min_support=1,
        option_min_episodes=1,
        online_feature_learning=False,
        online_feature_options=None,
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
    if str(stage_memory_feature_source) not in (
            "branch", "prototype", "hybrid"):
        raise ValueError(
            "stage_memory_feature_source must be branch/prototype/hybrid"
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
            online_feature_learning=online_feature_learning,
            online_feature_options=online_feature_options,
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
        online_feature_learning=online_feature_learning,
        online_feature_options=online_feature_options,
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
        (
            agent.stage_experience,
            stage_train,
            stage_validation,
        ) = StageExperienceBatch.fit_with_validation(
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
        tree_train = list(stage_train.episodes)
        tree_validation = list(stage_validation.episodes)
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
        else:
            agent.stage_tree = None
            agent.stage_tree_heuristic = None
            agent.stage_root_heuristic = None
        # Predictive-equivalence classes over realized options.  Merging is a
        # training-profile model-selection decision; the quotient as a whole is
        # accepted or rejected on validation episodes.  Nothing here is read by
        # the planner, and a rejected quotient is never used by the search.
        if tree_train and tree_validation:
            agent.option_quotient = OptionQuotient.fit(
                tree_train, tree_validation,
                prior=4.0, tolerance=float(option_tolerance),
                min_support=int(option_min_support),
                min_episodes=int(option_min_episodes),
            )
        else:
            agent.option_quotient = None
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
        agent.option_quotient = None
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
             rank_mode="order", rank_margin=0.0,
             stage_value=None, stage_value_weight=0.0,
             frontier_callback=None, search_chunk_size=0,
             rank_group_quota=1,
             chain_chunk_size=2, chain_hops=3, chain_beam_width=2,
             heuristic_demand=None, heuristic_demand_weight=0.0,
             adapt_world=False, adaptive_controller=None,
             chain_max_hops=5, chain_wide_beam_width=6):
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
    observe_online_features = (
        getattr(agent, "observe_online_transition", None)
        if getattr(agent, "online_feature_learner", None) is not None
        else None
    )
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
            stage_value=stage_value,
            stage_value_weight=stage_value_weight,
            frontier_callback=frontier_callback,
            search_chunk_size=search_chunk_size,
            rank_group_quota=rank_group_quota,
            chain_chunk_size=chain_chunk_size,
            chain_hops=chain_hops,
            chain_beam_width=chain_beam_width,
            heuristic_demand=heuristic_demand,
            heuristic_demand_weight=heuristic_demand_weight,
            adaptive_controller=adaptive_controller,
            chain_max_hops=chain_max_hops,
            chain_wide_beam_width=chain_wide_beam_width,
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
        if (prediction_hook or record or observe_provider is not None
                or observe_online_features is not None or adapt_world):
            next_obs = np.asarray(env.obs(), dtype=float)
        if adapt_world and hasattr(world, "observe"):
            world.observe(obs, history, next_obs)
        if prediction_hook:
            agent.observe_prediction(next_obs[:int(agent.obs_dim)])
        if (record or observe_provider is not None
                or observe_online_features is not None):
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
            if observe_online_features is not None:
                online_transition = dict(transition_record)
                online_transition["episode"] = (
                    int(seed if env_seed is None else env_seed),
                    int(episode),
                )
                observe_online_features(online_transition)
        recent = [float(action)] + recent[:-1]
    if observe_online_features is not None:
        agent.online_feature_learner.end_episode(
            terminal=bool(env.dead),
        )
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
    online_feature_learning = bool(
        task.get("online_feature_learning", False)
    )
    regime_value_weight = float(task.get("regime_value_weight", 0.0))
    regime_n_regimes = int(task.get("regime_n_regimes", 16))
    regime_min_dwell = int(task.get("regime_min_dwell", 6))
    regime_partition = str(task.get("regime_partition", "survival_tree"))
    regime_k_stages = int(task.get("regime_k_stages", 8))
    regime_mode = str(task.get("regime_mode", "lcb"))
    regime_lcb_z = float(task.get("regime_lcb_z", 1.0))
    regime_value_key = str(task.get("regime_value_key", "cum_survival"))
    regime_soft_top_k = int(task.get("regime_soft_top_k", 4))
    regime_kernel_tau = float(task.get("regime_kernel_tau", 0.7))
    regime_min_coverage = float(task.get("regime_min_coverage", 0.80))
    regime_ood_policy = str(task.get("regime_ood_policy", "neutral_mean"))
    regime_online_every = int(task.get("regime_online_every", 0))
    regime_online_frames = int(task.get("regime_online_frames", 12000))
    regime_failure_weight = float(task.get("regime_failure_weight", 0.0))
    regime_failure_sigma = float(task.get("regime_failure_sigma", 0.25))
    regime_failure_prior = float(task.get("regime_failure_prior", 4.0))
    regime_abstain_lcb = float(task.get("regime_abstain_lcb", 0.0))
    world_variables = bool(task.get("world_variables", False))
    world_var_reset_episode = bool(task.get("world_var_reset_episode", False))
    world_var_sigma = float(task.get("world_var_sigma", 0.75))
    world_var_abstain_lcb = float(task.get("world_var_abstain_lcb", 0.0))
    chain_chunk_size = int(task.get("chain_chunk_size", 2))
    chain_hops = int(task.get("chain_hops", 3))
    chain_beam_width = int(task.get("chain_beam_width", 2))
    chain_demand_weight = float(task.get("chain_demand_weight", 0.10))
    chain_demand_online_every = int(task.get("chain_demand_online_every", 0))
    chain_demand_ema = float(task.get("chain_demand_ema", 0.30))
    chain_demand_online_frames = int(task.get("chain_demand_online_frames", 12000))
    chain_demand_min_frames = int(task.get("chain_demand_min_frames", 600))
    chain_adaptive = bool(task.get("chain_adaptive", False))
    chain_max_hops = int(task.get("chain_max_hops", 5))
    chain_wide_beam_width = int(task.get("chain_wide_beam_width", 6))
    chain_conf_threshold = float(task.get("chain_conf_threshold", 0.72))
    chain_conf_decay = float(task.get("chain_conf_decay", 0.80))
    chain_meta_lcb = float(task.get("chain_meta_lcb", 0.65))
    chain_conf_quantile = float(task.get("chain_conf_quantile", 0.0))
    online_feature_options = task.get("online_feature_options")

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
        online_feature_learning=online_feature_learning,
        online_feature_options=online_feature_options,
    )
    explicit_eval = bool(eval_seeds)
    eval_world = world
    world_var_node = None
    if world_variables:
        from node_variables import DynamicsClassNode
        training_rows_for_vars = getattr(agent, "milestone_training_rows", ())
        if training_rows_for_vars:
            world_var_node = DynamicsClassNode(
                world, training_rows_for_vars,
                abstain_lcb=world_var_abstain_lcb,
            )
            eval_world = world_var_node
    regime_evaluator = None
    regime_online_updater = None
    regime_stage_value = None
    regime_stats = None
    feature_demand = None
    online_feature_demand = None
    heuristic_demand_cb = None
    adaptive_controller = None
    online_episode_id = 0
    if regime_value_weight > 0.0:
        from stage_regime import (
            AdaptiveSearchController,
            CalibratedRegimeTransitionModel,
            ControllableFeatureDemand,
            OnlineFeatureDemandUpdater,
            OnlineRegimeContinuationUpdater,
            RegimeContinuationEvaluator,
            StageRegimeSummarizer,
            TrunkPointFailureMemory,
        )
        training_rows = getattr(agent, "milestone_training_rows", ())
        summarizer = StageRegimeSummarizer(
            n_regimes=regime_n_regimes,
            min_dwell=regime_min_dwell,
            partition=regime_partition,
        ).fit_rows(training_rows, lag=lag)
        regime_episodes = summarizer.transform_rows(training_rows, lag=lag)
        regime_model = CalibratedRegimeTransitionModel(
            max_depth=2, min_episodes=2,
        ).fit(regime_episodes, use_action=True)
        regime_values = regime_model.continuation_values(
            len(summarizer.centers_),
            horizon_stages=regime_k_stages,
            mode=regime_mode,
            min_action_episodes=4,
            lcb_z=regime_lcb_z,
        )
        regime_evaluator = RegimeContinuationEvaluator(
            summarizer,
            regime_values,
            min_regime_episodes=2,
            min_decision_coverage=regime_min_coverage,
            ood_policy=regime_ood_policy,
            value_key=regime_value_key,
            soft_top_k=regime_soft_top_k,
            kernel_tau=regime_kernel_tau,
        )
        regime_stats = {}
        if regime_online_every > 0:
            failure_mem = None
            if regime_failure_weight > 0.0:
                failure_mem = TrunkPointFailureMemory(
                    summarizer,
                    sigma_mult=regime_failure_sigma,
                    prior=regime_failure_prior,
                    var_sigma=world_var_sigma,
                )
                failure_mem.ingest_rows(training_rows, include_failures=False)
            regime_online_updater = OnlineRegimeContinuationUpdater(
                summarizer,
                regime_episodes,
                horizon_stages=regime_k_stages,
                mode=regime_mode,
                min_action_episodes=4,
                lcb_z=regime_lcb_z,
                max_depth=2,
                min_episodes=2,
                min_regime_episodes=2,
                min_decision_coverage=regime_min_coverage,
                ood_policy=regime_ood_policy,
                value_key=regime_value_key,
                soft_top_k=regime_soft_top_k,
                kernel_tau=regime_kernel_tau,
                update_every=regime_online_every,
                max_online_frames=regime_online_frames,
                initial_evaluator=regime_evaluator,
                failure_memory=failure_mem,
                failure_memory_weight=regime_failure_weight,
                var_sigma=world_var_sigma,
                stage_abstain_lcb=regime_abstain_lcb,
            )
            regime_stage_value = regime_online_updater.as_stage_value_callback(
                stats=regime_stats,
                var_signature_getter=(
                    world_var_node.world_signature
                    if world_var_node is not None else None
                ),
            )
            online_episode_id = (
                max((int(r.get("episode", 0)) for r in training_rows), default=0)
                + 1000
            )
        else:
            regime_stage_value = regime_evaluator.as_stage_value_callback(
                stats=regime_stats,
            )
        if decision == "chained" and chain_demand_weight > 0.0:
            feature_demand = ControllableFeatureDemand().fit_rows(
                world, agent, training_rows, lag=lag,
            )
            if chain_demand_online_every > 0:
                online_feature_demand = OnlineFeatureDemandUpdater(
                    feature_demand,
                    update_every=chain_demand_online_every,
                    ema_alpha=chain_demand_ema,
                    max_online_frames=chain_demand_online_frames,
                    min_online_frames=chain_demand_min_frames,
                    var_sigma=world_var_sigma,
                )
                heuristic_demand_cb = online_feature_demand.distance_batch
            else:
                heuristic_demand_cb = feature_demand.distance_batch
        if decision == "chained" and chain_adaptive:
            adaptive_controller = AdaptiveSearchController(
                conf_threshold=chain_conf_threshold,
                conf_decay=chain_conf_decay,
                conf_quantile=chain_conf_quantile,
                meta_lcb=chain_meta_lcb,
            )

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
                if world_var_node is not None and world_var_reset_episode:
                    world_var_node.reset_variables()
                if adaptive_controller is not None:
                    stage_gate = (
                        regime_online_updater.stage_abstain_gate
                        if regime_online_updater is not None else None
                    )
                    adaptive_controller.begin_episode(stage_gate)
                record_online = (
                    regime_online_updater is not None
                    or online_feature_demand is not None
                )
                frames, actions, ep_rows = evaluate(
                    seed, eval_world, agent, cap=cap, horizon=horizon, lag=lag,
                    step_stride=step_stride,
                    stride_includes_lag=stride_includes_lag,
                    env_seed=env_seed,
                    record=record_online,
                    episode=online_episode_id if record_online else 0,
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
                    stage_value=regime_stage_value,
                    stage_value_weight=regime_value_weight,
                    chain_chunk_size=chain_chunk_size,
                    chain_hops=chain_hops,
                    chain_beam_width=chain_beam_width,
                    heuristic_demand=heuristic_demand_cb,
                    heuristic_demand_weight=(
                        chain_demand_weight
                        if heuristic_demand_cb is not None else 0.0
                    ),
                    adapt_world=(world_var_node is not None),
                    adaptive_controller=adaptive_controller,
                    chain_max_hops=chain_max_hops,
                    chain_wide_beam_width=chain_wide_beam_width,
                )
                if adaptive_controller is not None:
                    adaptive_controller.record_episode(int(frames))
                if record_online:
                    online_episode_id += 1
                    if regime_online_updater is not None:
                        regime_online_updater.record_episode_rows(
                            ep_rows,
                            lag=lag,
                            var_signature=(
                                world_var_node.world_signature()
                                if world_var_node is not None else None
                            ),
                        )
                        regime_evaluator = regime_online_updater.evaluator
                    if online_feature_demand is not None:
                        online_feature_demand.record_episode_rows(ep_rows)
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
        online_features=(
            agent.online_feature_learner.snapshot()
            if getattr(agent, "online_feature_learner", None) is not None
            else None
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
        regime_continuation=(
            dict(
                weight=regime_value_weight,
                evaluator=regime_evaluator.snapshot(),
                stats=regime_stats,
                online=(
                    regime_online_updater.snapshot()
                    if regime_online_updater is not None else None
                ),
                adaptive_search=(
                    adaptive_controller.snapshot()
                    if adaptive_controller is not None else None
                ),
                online_demand=(
                    online_feature_demand.snapshot()
                    if online_feature_demand is not None else None
                ),
            )
            if regime_evaluator is not None else None
        ),
        world_variables=(
            world_var_node.snapshot()
            if world_var_node is not None else None
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



from cli import main


if __name__ == "__main__":
    main()
