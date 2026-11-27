"""Training loop, world-model fitting, objective calibration, and milestone induction."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from _paths import ensure_core_paths
ensure_core_paths()

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

LAG = 3


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
    observed_actions = sorted({int(row["history"][0]) for row in transitions})
    action_space = tuple(observed_actions) if len(observed_actions) >= 2 else (0, 1)
    agent = NodeRuntime(
        obs_dim=obs_dim, action_space=action_space, lag=lag,
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
    from evaluator import evaluate
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
    from substrate_nodes import (
        _association_value_table,
        _transition_table,
        _transition_values,
    )
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

