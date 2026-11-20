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
import json
import os
import sys
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
from node_features import FeatureRuleBank
from node_runtime import NodeRuntime, future_pain_labels, future_return_labels
from substrate_generic import (
    LinearObservationModel,
    collect,
    history_features,
)
from trajectory_search import exhaustive_search

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
        rule_coverage_x=None, pairwise_features: bool = True):
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
    )
    episodes = {}
    for row in transitions:
        episodes.setdefault(row["episode"], []).append(row)
    ordered = [episodes[key] for key in sorted(episodes)]

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
    return world, agent


def collect_planner(seed, world, agent, episodes=2, cap=1200, horizon=10,
                    lag=LAG, step_stride=2, stride_includes_lag=False,
                    offset=50000, max_frames=None):
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
        pairwise_features=True):
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
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if int(lag) < 0:
        raise ValueError("lag must be non-negative")
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
        stage1, _counts = collect(seed, steps1, coverage=coverage, lag=lag)
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
        )
        transitions = list(transitions)
    else:
        transitions, _counts = collect(seed, steps, coverage=coverage, lag=lag)
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
            # a continuation of the first.
            offset=dagger_offset + _round * int(dagger_episodes),
            max_frames=(int(dagger_frames) or None),
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
    return world, agent


def _curiosity_potential(support):
    """Curiosity potential: high where the readout's evidence is thin.

    ``Φ(s) = 1/sqrt(1 + evidence(s))`` with evidence the mean observation
    count behind the state's active cells.  It is bounded in (0, 1], so
    the shaped objective cannot out-scale the hazard and value terms, and
    it needs no task knowledge - only the counters the learner already
    maintains.
    """
    support = np.asarray(support, dtype=float)
    return 1.0 / np.sqrt(1.0 + np.maximum(support, 0.0))


def _curiosity_term(phi_start, phi_end, weight, discount, horizon):
    """Potential-based curiosity shaping summed over a fixed-horizon plan.

    PBRS adds ``γΦ(s') - Φ(s)`` per step; over H steps the sum telescopes
    to ``γ^H Φ(s_H) - Φ(s_0)``.  The planner's step costs are accumulated
    without a per-step discount (``γ = 1`` in the rollout), so the term
    enters as ``w * (Φ(s_0) - Φ(s_H))``: a less-supported endpoint lowers
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


def policy_result(env, world, agent, prefix, horizon, lag=LAG,
                  step_stride=2, stride_includes_lag=False):
    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)
    lag = int(lag)
    # Decision-side curiosity: the potential at the shared start state is
    # a constant across plans, but it is evaluated so the term is exactly
    # the potential difference (PBRS identity), not a bare endpoint bonus.
    curiosity_weight = float(getattr(agent, "curiosity_weight", 0.0))
    phi_start = 0.0
    if curiosity_weight > 0.0:
        pending_start = (
            pending_prefix[:lag][::-1] if lag > 0
            else np.zeros(0, dtype=float)
        )
        encoded_start = np.concatenate(
            [initial.reshape(1, -1), pending_start.reshape(1, -1)], axis=1
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
            if step % step_stride != 0 and not (
                stride_includes_lag and step == lag
            ):
                return (
                    np.zeros((len(states), 5)),
                    np.isfinite(states).all(axis=1),
                )
            history = history_features(
                sequences, step + 1, pending_prefix, action_dim=action_dim
            )
            pending = history[:, :agent.lag][:, ::-1]
            encoded = np.concatenate([states, pending], axis=1) / agent.scales
            threat, excess = agent.pain_graded_batch(encoded)
            cost = threat + agent.control_risk_weight * excess
            zeros = np.zeros(len(states))
            return (
                np.column_stack([cost, zeros, cost, zeros, zeros]),
                np.isfinite(encoded).all(axis=1),
            )

        def evaluate_terminal(states, sequences):
            history = history_features(
                sequences, horizon, pending_prefix, action_dim=action_dim
            )
            pending = history[:, :agent.lag][:, ::-1]
            encoded = np.concatenate([states, pending], axis=1) / agent.scales
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
                ]),
                np.isfinite(encoded).all(axis=1),
            )
    else:
        def evaluate_step(states, sequences, step):
            # `lag` is the first predicted state whose acting action comes
            # from the plan rather than the pending prefix, so a plain
            # stride can step over the first point the plan can actually
            # influence.
            if step % step_stride != 0 and not (
                stride_includes_lag and step == lag
            ):
                return (
                    np.zeros((len(states), 5)),
                    np.isfinite(states).all(axis=1),
                )
            history = history_features(
                sequences, step + 1, pending_prefix, action_dim=action_dim
            )
            pending = history[:, :agent.lag][:, ::-1]
            encoded = (
                np.concatenate([states, pending], axis=1) / agent.scales
            )
            risk = agent.pain_signal_batch(encoded)
            zeros = np.zeros(len(states))
            return (
                np.column_stack([risk, zeros, risk, zeros, zeros]),
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
                np.concatenate([states, pending], axis=1) / agent.scales
            )
            pain, dopamine = agent.valence_state_batch(encoded)
            mask_table = agent._feature_mask_table(
                encoded[:, :agent.obs_dim]
            )
            if getattr(agent, "predicate_readouts", False):
                # D3: the terminal hazard comes from the predicates only;
                # the prototype valence no longer bounds it.
                risk = agent.pain_signal_batch(encoded, mask_table)
            else:
                risk = np.maximum(
                    pain, agent.pain_signal_batch(encoded, mask_table)
                )
            # Scalarized objective: accumulated hazard minus the
            # sequence-level value of the endpoint, so a plan that accepts
            # local risk in exchange for a long-survival alignment can win.
            # D4 everywhere: the value channel speaks with the evidence
            # behind it.  Without the trust scale an uncovered endpoint
            # would be credited with the population-average return, so the
            # planner would be paid (or charged) for states nothing was
            # ever learned about.
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
            cost = risk + value_term + curiosity
            return (
                np.column_stack([
                    cost, -dopamine, risk, value_term, curiosity,
                ]),
                np.isfinite(encoded).all(axis=1),
            )

    result = exhaustive_search(
        initial_state=initial,
        action_space=(0, 1),
        horizon=horizon,
        transition_batch=predict_batch,
        evaluate_step=evaluate_step,
        evaluate_terminal=evaluate_terminal,
        batch_size=2 ** horizon,
        # "sum" expands to one mode per objective column, so the ledger
        # columns (hazard and value shares) sum alongside the scalarized
        # cost and the dopamine tie-break without changing their contract.
        score_combine="sum",
    )
    return result


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
                    decision="search", ledger_out=None):
    """Dispatch one decision to the searched or the cell-Q-only policy.

    With `ledger_out` (a list) and the search decision, the winning
    plan's objective ledger - the hazard and value shares of the
    scalarized cost, split by `policy_result`'s third and fourth score
    columns - is appended so the executed decision's components can be
    regressed against its realized outcome (P4 objective learning).  The
    curiosity share (column 4) is a fixed shaping term and is deliberately
    not part of that ledger.
    """
    if decision == "search":
        result = policy_result(
            env, world, agent, prefix, horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
        )
        if ledger_out is not None:
            score = result.score
            ledger_out.append((
                float(score[2]), float(score[3]),
            ))
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
    ``w * (Φ(s_0) - Φ(s_H))`` (the potential-difference shaping term).
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
             record=False, episode=0, decision="search"):
    """Run one deterministic evaluation episode.

    ``env_seed`` selects the world instance; when omitted the historical
    ``seed`` world is used.  With ``record=True`` the real transitions the
    planner actually executed are returned in the same schema as ``collect``
    (``obs``/``next_obs``/``history``/``signals``/``done``/``episode``), which
    is what the closed-loop (DAgger) path consumes.  No counterfactual
    prediction is recorded.
    """
    if cap < 1:
        raise ValueError("cap must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if step_stride < 1:
        raise ValueError("step_stride must be positive")
    lag = int(lag)
    env = latent[lag](seed if env_seed is None else int(env_seed))
    prefix = np.zeros(lag, dtype=float)
    recent = [0.0] * lag
    actions = []
    transitions = []
    while not env.dead and env.t < cap:
        obs = np.asarray(env.obs(), dtype=float)
        ledger: List[Tuple[float, float]] = []
        ledger_out = ledger if record else None
        action = decision_action(
            env, world, agent, prefix, horizon, lag=lag,
            step_stride=step_stride,
            stride_includes_lag=stride_includes_lag,
            decision=decision, ledger_out=ledger_out,
        )
        history = np.asarray([float(action)] + recent[:lag], dtype=float)
        actions.append(action)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(action)
        if record:
            feedback = env.feedback()
            transitions.append(dict(
                obs=obs,
                next_obs=np.asarray(env.obs(), dtype=float),
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
            ))
        recent = [float(action)] + recent[:-1]
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
    pairwise_features = bool(task.get("pairwise_features", True))
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
                frames, actions, _ = evaluate(
                    seed, world, agent, cap=cap, horizon=horizon, lag=lag,
                    step_stride=step_stride,
                    stride_includes_lag=stride_includes_lag,
                    env_seed=env_seed,
                    decision=decision,
                )
                rows.append(dict(
                    train_seed=seed,
                    eval_seed=int(eval_seed),
                    episode=episode,
                    env_seed=int(env_seed),
                    frames=int(frames),
                    action_rate=(
                        float(np.mean(actions)) if actions else 0.0
                    ),
                ))
    return dict(
        seed=seed,
        frames=rows[0]["frames"],
        action_rate=rows[0]["action_rate"],
        evals=rows,
        graph=agent.snapshot(),
        dagger=getattr(agent, "dagger_log", []),
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
    parser.add_argument("--decision", choices=("search", "cellq"),
                        default="search",
                        help="search = full MPC planning; cellq = read the "
                             "learned (feature, action) cells directly")
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
    parser.add_argument("--no-pairwise-features", dest="pairwise_features",
                        action="store_false",
                        help="disable the generic pairwise coordinate "
                             "differences in the induction grammar")
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
    if args.max_nodes < 1:
        raise ValueError("--max-nodes must be positive")
    if args.birth_threshold <= 0.0:
        raise ValueError("--birth-threshold must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
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
            eval_seeds=eval_seeds,
            eval_episodes=args.eval_episodes,
            dagger_rounds=args.dagger_rounds,
            dagger_episodes=args.dagger_episodes,
            dagger_cap=args.dagger_cap,
            dagger_offset=args.dagger_offset,
            dagger_horizon=args.dagger_horizon,
            dagger_frames=args.dagger_frames,
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
    print("\ntrain  eval  ep  env_seed  frames  action_rate")
    for row in eval_rows:
        print(" %5d  %4d  %2d  %8d  %6d  %10.3f"
              % (row["train_seed"], row["eval_seed"], row["episode"],
                 row["env_seed"], row["frames"], row["action_rate"]))

    per_train = group_summaries(eval_rows, group="train_seed")
    per_eval = group_summaries(eval_rows, group="eval_seed")
    overall = cluster_summary(eval_rows, cluster="train_seed")
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
            ),
            rows=eval_rows,
            dagger=[
                dict(seed=row["seed"], rounds=row["dagger"])
                for row in rows
            ],
            per_train_seed=per_train,
            per_eval_seed=per_eval,
            overall=overall,
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
