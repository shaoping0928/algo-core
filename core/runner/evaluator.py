"""Episode evaluation loop and per-seed experiment orchestration."""
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

from trainer import (
    _milestone_goals_from_transitions,
    _milestone_model_from_transitions,
    _rollout_error_curve,
    _stage_evidence_for_streams,
    fit,
    ledger_survival_report,
)

LAG = 3


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
        from substrate_nodes import decision_action
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
    chain_demand_rel_pairs = int(task.get("chain_demand_rel_pairs", 0))
    chain_demand_min_terminal = int(task.get("chain_demand_min_terminal", 0))
    world_online_refit_gain = float(task.get("world_online_refit_gain", 0.0))
    world_online_min_frames = int(task.get("world_online_min_frames", 1500))
    world_online_every = int(task.get("world_online_every", 5))
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
    world_refit_gate = None
    if world_online_refit_gain > 0.0 and world_var_node is None:
        from world_refit import OnlineWorldRefitGate
        base_rows_refit = getattr(agent, "milestone_training_rows", ())
        if base_rows_refit:
            world_refit_gate = OnlineWorldRefitGate(
                eval_world,
                base_rows_refit,
                seed=seed,
                min_rel_gain=world_online_refit_gain,
                min_frames=world_online_min_frames,
                update_every=world_online_every,
            )
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
            feature_demand = ControllableFeatureDemand(
                max_rel_pairs=chain_demand_rel_pairs,
            ).fit_rows(
                world, agent, training_rows, lag=lag,
            )
            if chain_demand_online_every > 0:
                online_feature_demand = OnlineFeatureDemandUpdater(
                    feature_demand,
                    update_every=chain_demand_online_every,
                    ema_alpha=chain_demand_ema,
                    max_online_frames=chain_demand_online_frames,
                    min_online_frames=chain_demand_min_frames,
                    min_terminal_episodes=chain_demand_min_terminal,
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
                    or world_refit_gate is not None
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
                    if world_refit_gate is not None and world_refit_gate.record_episode_rows(ep_rows):
                        eval_world = world_refit_gate.world
                        if (
                            decision == "chained"
                            and chain_demand_weight > 0.0
                            and feature_demand is not None
                        ):
                            from stage_regime import ControllableFeatureDemand, OnlineFeatureDemandUpdater
                            feature_demand = ControllableFeatureDemand(
                                max_rel_pairs=chain_demand_rel_pairs,
                            ).fit_rows(
                                eval_world, agent, world_refit_gate.combined_rows(), lag=lag,
                            )
                            if chain_demand_online_every > 0:
                                online_feature_demand = OnlineFeatureDemandUpdater(
                                    feature_demand,
                                    update_every=chain_demand_online_every,
                                    ema_alpha=chain_demand_ema,
                                    max_online_frames=chain_demand_online_frames,
                                    min_online_frames=chain_demand_min_frames,
                                    min_terminal_episodes=chain_demand_min_terminal,
                                    var_sigma=world_var_sigma,
                                )
                                online_feature_demand.record_episode_rows(ep_rows)
                                heuristic_demand_cb = online_feature_demand.distance_batch
                            else:
                                heuristic_demand_cb = feature_demand.distance_batch
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
        world_refit=(
            world_refit_gate.snapshot()
            if world_refit_gate is not None else None
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

