"""Chained shallow search and dual-confidence adaptive segmented planning.

Strictly task-agnostic planning primitives extracted from ``substrate_nodes.py``:
  - ``terminal_hazard`` / ``association_value_batch``: endpoint readout helpers
  - ``chained_shallow_result``: fixed multi-hop active-pruning segmented search
  - ``adaptive_chained_result``: dual-confidence decaying-threshold adaptive deepening
"""
from __future__ import annotations

import numpy as np

from node_runtime import NodeRuntime
from substrate_generic import LAG, history_features
from trajectory_search import SearchResult


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


def _expand_step_actions(states, action_arr, n_act, action_top_k, action_proposer):
    if int(action_top_k) > 0 and n_act > int(action_top_k) and action_proposer is not None:
        cand = np.asarray(action_proposer(states, int(action_top_k)), dtype=float)
        return int(cand.shape[1]), cand.reshape(-1)
    return n_act, np.tile(action_arr, len(states))


def chained_shallow_result(
    env, world, agent, prefix, chunk_size=2, n_hops=3, beam_width=2,
    lag=LAG, step_stride=2, risk_weight=1.0,
    stage_value=None, stage_value_weight=0.0,
    heuristic_demand=None, heuristic_demand_weight=0.0,
    diverse_first_action=True,
    action_top_k: int = 0,
    action_proposer=None,
) -> SearchResult:
    """Chained 2-3 frame active-pruning shallow search.

    Chains ``n_hops`` short searches of ``chunk_size`` frames, pruning to the
    top ``beam_width`` frontier states at each intermediate hop boundary.
    ``heuristic_demand`` (when provided) is used strictly for intermediate
    pruning/ranking and NEVER alters the final objective score used to choose
    among the surviving terminal candidates.
    """
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if int(n_hops) < 1:
        raise ValueError("n_hops must be positive")
    if int(beam_width) < 1:
        raise ValueError("beam_width must be positive")
    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    action_space = tuple(getattr(agent, "action_space", (0, 1)))
    n_act = len(action_space)
    action_arr = np.asarray(action_space, dtype=float)
    action_dim = int(lag) + 1
    step_stride = max(int(step_stride), 1)
    risk_weight = float(risk_weight)
    obs_dim = int(agent.obs_dim)
    scales = agent.scales
    chunk_size = int(chunk_size)
    n_hops = int(n_hops)
    beam_width = int(beam_width)
    horizon = chunk_size * n_hops
    is_rt = isinstance(agent, NodeRuntime)
    sv_weight = float(stage_value_weight)
    hd_weight = float(heuristic_demand_weight)
    proposer = (
        action_proposer if action_proposer is not None
        else getattr(agent, "propose_actions_batch", None)
    )
    top_k_act = int(action_top_k)

    states = initial.reshape(1, -1)
    sequences = np.zeros((1, horizon), dtype=float)
    cum_risk = np.zeros(1, dtype=float)
    first_states = None
    candidate_count = 0

    for hop in range(n_hops):
        start_step = hop * chunk_size
        end_step = start_step + chunk_size
        for step in range(start_step, end_step):
            k_act, step_actions = _expand_step_actions(
                states, action_arr, n_act, top_k_act, proposer,
            )
            next_states = np.repeat(states, k_act, axis=0)
            next_seqs = np.repeat(sequences, k_act, axis=0)
            next_seqs[:, step] = step_actions
            history = history_features(
                next_seqs, step + 1, pending_prefix, action_dim=action_dim,
            )
            next_states = np.asarray(
                world.predict(next_states, history), dtype=float
            )
            if first_states is None:
                first_states = next_states.copy()
            else:
                first_states = np.repeat(first_states, k_act, axis=0)
            candidate_count += len(next_states)
            next_risk = np.repeat(cum_risk, k_act)
            if step % step_stride == 0 and step < horizon - 1:
                pending = history[:, :agent.lag][:, ::-1]
                encoded = np.concatenate(
                    [next_states[:, :obs_dim], pending], axis=1
                ) / scales
                next_risk = next_risk + np.asarray(
                    agent.pain_signal_batch(encoded), dtype=float
                ) * risk_weight
            states = next_states
            sequences = next_seqs
            cum_risk = next_risk

        if hop < n_hops - 1:
            # Intermediate active pruning: heuristic ranker only (separate from
            # the true terminal objective).
            h_score = cum_risk.copy()
            if stage_value is not None and sv_weight > 0.0:
                pen = np.asarray(
                    stage_value(states, sequences), dtype=float
                ).reshape(-1)
                h_score = h_score + sv_weight * pen
            if heuristic_demand is not None and hd_weight > 0.0:
                h_score = h_score + hd_weight * np.asarray(
                    heuristic_demand(states), dtype=float
                ).reshape(-1)
            finite = np.isfinite(h_score) & np.isfinite(states).all(axis=1)
            if finite.any():
                h_score = np.where(finite, h_score, np.inf)
            else:
                h_score = np.zeros(len(states), dtype=float)
            k_keep = min(beam_width, len(states))
            if diverse_first_action and hop == 0 and k_keep >= n_act:
                chosen = []
                used = set()
                per_a = max(1, k_keep // n_act)
                for a_val in action_space:
                    idxs = np.flatnonzero(sequences[:, 0] == float(a_val))
                    if len(idxs):
                        for ii in idxs[np.argsort(h_score[idxs])[:per_a]]:
                            chosen.append(int(ii))
                            used.add(int(ii))
                for ii in np.argsort(h_score):
                    if len(chosen) >= k_keep:
                        break
                    if int(ii) not in used:
                        chosen.append(int(ii))
                        used.add(int(ii))
                keep = np.asarray(chosen[:k_keep], dtype=int)
            else:
                keep = np.argsort(h_score)[:k_keep]
            states = states[keep]
            sequences = sequences[keep]
            cum_risk = cum_risk[keep]
            first_states = first_states[keep]

    # Final terminal evaluation using strictly the true calibrated objective.
    history = history_features(
        sequences, horizon, pending_prefix, action_dim=action_dim,
    )
    pending = history[:, :agent.lag][:, ::-1]
    encoded = np.concatenate([states[:, :obs_dim], pending], axis=1) / scales
    if is_rt and len(encoded) > 0:
        sw = agent._soft_weights_batch(encoded)
        pain, dopamine = agent.valence_state_batch(encoded, soft_weights=sw)
        mask_table = agent._feature_mask_table(encoded[:, :obs_dim])
        signal = agent.pain_signal_batch(encoded, mask_table, soft_weights=sw)
    else:
        pain, dopamine = agent.valence_state_batch(encoded)
        mask_table = agent._feature_mask_table(encoded[:, :obs_dim])
        signal = agent.pain_signal_batch(encoded, mask_table)
    term_risk = terminal_hazard(pain, signal, risk_weight)
    risk = cum_risk + term_risk
    if getattr(agent, "value_readout", "cells") == "association":
        value, support = association_value_batch(agent, encoded)
    else:
        value, support = agent.feature_value_batch(
            encoded, mask_table, return_support=True
        )
    trust = support / (support + getattr(agent, "value_trust_samples", 150.0))
    value_term = -agent.feature_value_weight * trust * value
    cost = risk + value_term
    if stage_value is not None and sv_weight > 0.0:
        penalties = np.asarray(
            stage_value(states, sequences), dtype=float
        ).reshape(-1)
        if penalties.shape != (len(states),) or not np.isfinite(penalties).all():
            raise ValueError("stage_value must return finite penalties")
        cost = cost + sv_weight * penalties
    finite = np.isfinite(cost) & np.isfinite(encoded).all(axis=1)
    if not finite.any():
        raise ValueError("chained shallow search found no finite candidate")
    ranking_cost = np.where(finite, cost, np.inf)
    order = np.lexsort((-dopamine, ranking_cost))
    best = int(order[0])
    score = (
        float(cost[best]),
        float(-dopamine[best]),
        float(risk[best]),
        float(value_term[best]),
        0.0,
        0.0,
    )
    return SearchResult(
        actions=np.asarray(sequences[best], dtype=float).copy(),
        score=score,
        feasible=True,
        feasible_count=int(finite.sum()),
        candidate_count=int(candidate_count),
        first_state=np.asarray(first_states[best], dtype=float).copy(),
        final_state=np.asarray(states[best], dtype=float).copy(),
    )


def _select_adaptive_beam(
    states, seqs, first_states, cum_risk, path_sp, path_dem,
    score, beam_width, action_space, keep_diverse,
):
    """Prune frontier states to ``beam_width``, optionally preserving first-action diversity.

    When ``keep_diverse`` is True (used at hop boundaries up to and past ``lag``),
    allocates an equal quota per distinct root action before filling remaining
    slots globally.  This prevents premature single-action beam collapse before
    the root action has physically taken effect in the dynamics.
    """
    finite = np.isfinite(score) & np.isfinite(states).all(axis=1)
    safe = (
        np.where(finite, score, np.inf)
        if finite.any() else np.zeros(len(states), dtype=float)
    )
    k = min(int(beam_width), len(states))
    n_act = len(action_space)
    if keep_diverse and k >= n_act:
        chosen, used = [], set()
        per_a = max(1, k // n_act)
        for a_val in action_space:
            idxs = np.flatnonzero(seqs[:, 0] == float(a_val))
            if len(idxs):
                for ii in idxs[np.argsort(safe[idxs])[:per_a]]:
                    chosen.append(int(ii))
                    used.add(int(ii))
        # Fill any leftover beam budget with the globally best remaining trajectories.
        for ii in np.argsort(safe):
            if len(chosen) >= k:
                break
            if int(ii) not in used:
                chosen.append(int(ii))
                used.add(int(ii))
        keep = np.asarray(chosen[:k], dtype=int)
    else:
        keep = np.argsort(safe)[:k]
    return (
        states[keep],
        seqs[keep],
        first_states[keep] if first_states is not None else None,
        cum_risk[keep],
        path_sp[keep] if path_sp is not None else None,
        path_dem[keep] if path_dem is not None else None,
        safe[keep],
    )


def _expand_clean_chunk(
    states, seqs, first_states, cum_risk, path_sp, path_dem,
    start_step, chunk_size, pending_prefix, world, agent,
    lag=LAG, step_stride=2, risk_weight=0.0,
    action_top_k: int = 0, action_proposer=None,
):
    """Roll the current beam forward by ``chunk_size`` steps without recomputing earlier hops.

    Adaptive tiers pass ``risk_weight=0.0`` once the online evaluator is verified
    reliable, avoiding distortion from frozen offline pain readouts during deeper
    multi-step lookahead.
    """
    action_space = tuple(getattr(agent, "action_space", (0, 1)))
    n_act = len(action_space)
    action_arr = np.asarray(action_space, dtype=float)
    proposer = (
        action_proposer if action_proposer is not None
        else getattr(agent, "propose_actions_batch", None)
    )
    top_k_act = int(action_top_k)
    action_dim = int(lag) + 1
    obs_dim = int(agent.obs_dim)
    scales = agent.scales
    end_step = int(start_step) + int(chunk_size)
    if seqs.shape[1] < end_step:
        # Grow action buffer on demand when incrementally extending past base_hops.
        pad = np.zeros((len(seqs), end_step - seqs.shape[1]), dtype=float)
        seqs = np.concatenate([seqs, pad], axis=1)
    added = 0
    for step in range(int(start_step), end_step):
        k_act, step_actions = _expand_step_actions(
            states, action_arr, n_act, top_k_act, proposer,
        )
        next_states = np.repeat(states, k_act, axis=0)
        next_seqs = np.repeat(seqs, k_act, axis=0)
        next_seqs[:, step] = step_actions
        hist = history_features(
            next_seqs, step + 1, pending_prefix, action_dim=action_dim,
        )
        next_states = np.asarray(world.predict(next_states, hist), dtype=float)
        if first_states is None:
            first_states = next_states.copy()
        else:
            first_states = np.repeat(first_states, k_act, axis=0)
        added += len(next_states)
        next_risk = np.repeat(cum_risk, k_act)
        if path_sp is not None:
            path_sp = np.repeat(path_sp, k_act, axis=0)
        if path_dem is not None:
            path_dem = np.repeat(path_dem, k_act, axis=0)
        if risk_weight > 0.0 and step % step_stride == 0 and step < end_step - 1:
            pend = hist[:, :agent.lag][:, ::-1]
            enc = np.concatenate([next_states[:, :obs_dim], pend], axis=1) / scales
            next_risk = next_risk + np.asarray(
                agent.pain_signal_batch(enc), dtype=float
            ) * risk_weight
        states, seqs, cum_risk = next_states, next_seqs, next_risk
    return states, seqs, first_states, cum_risk, path_sp, path_dem, added


def _score_clean_frontier(
    states, seqs, cum_risk, path_sp, path_dem, horizon,
    stage_value, stage_value_weight, heuristic_demand, heuristic_demand_weight,
    lag=LAG,
):
    """Score frontier consistently across all controllable hops (``horizon > lag``).

    Pre-lag states are identical across root actions and are excluded from the
    path-mean accumulator so early uncontrollable steps cannot dilute downstream
    controllable differences.
    """
    if stage_value is not None and stage_value_weight > 0.0:
        sp = stage_value_weight * np.asarray(
            stage_value(states, seqs), dtype=float
        ).reshape(-1)
    else:
        sp = np.zeros(len(states), dtype=float)
    if heuristic_demand is not None and heuristic_demand_weight > 0.0:
        dm = heuristic_demand_weight * np.asarray(
            heuristic_demand(states), dtype=float
        ).reshape(-1)
    else:
        dm = np.zeros(len(states), dtype=float)

    if int(horizon) > int(lag):
        col_sp = sp.reshape(-1, 1)
        col_dm = dm.reshape(-1, 1)
        full_sp = (
            col_sp if path_sp is None else np.concatenate([path_sp, col_sp], axis=1)
        )
        full_dm = (
            col_dm if path_dem is None else np.concatenate([path_dem, col_dm], axis=1)
        )
        sp_eff = np.mean(full_sp, axis=1)
        dm_eff = np.mean(full_dm, axis=1)
    else:
        full_sp, full_dm = path_sp, path_dem
        sp_eff, dm_eff = sp, dm

    cost = cum_risk + sp_eff + dm_eff
    finite = np.isfinite(cost) & np.isfinite(states).all(axis=1)
    if finite.any():
        cost = np.where(finite, cost, np.inf)
    return cost, full_sp, full_dm, finite


def _run_clean_chain_tier(
    initial, world, agent, pending_prefix, chunk_size, n_hops, beam_width,
    diverse_hop1, stage_value, stage_value_weight,
    heuristic_demand, heuristic_demand_weight, lag=LAG, step_stride=2,
    action_top_k: int = 0, action_proposer=None,
):
    """Evaluate one segmented search tier and compute its monotonic bottleneck confidence."""
    action_space = tuple(getattr(agent, "action_space", (0, 1)))
    states = initial.reshape(1, -1)
    seqs = np.zeros((1, int(chunk_size) * int(n_hops)), dtype=float)
    first_states = None
    cum_risk = np.zeros(1, dtype=float)
    path_sp, path_dem = None, None
    total_cands = 0

    for hop in range(int(n_hops)):
        start_step = hop * int(chunk_size)
        end_step = start_step + int(chunk_size)
        (
            states, seqs, first_states, cum_risk, path_sp, path_dem, added,
        ) = _expand_clean_chunk(
            states, seqs, first_states, cum_risk, path_sp, path_dem,
            start_step=start_step, chunk_size=chunk_size,
            pending_prefix=pending_prefix, world=world, agent=agent,
            lag=lag, step_stride=step_stride, risk_weight=0.0,
            action_top_k=action_top_k, action_proposer=action_proposer,
        )
        total_cands += added
        cost, path_sp, path_dem, finite = _score_clean_frontier(
            states, seqs, cum_risk, path_sp, path_dem, horizon=end_step,
            stage_value=stage_value, stage_value_weight=stage_value_weight,
            heuristic_demand=heuristic_demand,
            heuristic_demand_weight=heuristic_demand_weight,
            lag=lag,
        )
        if hop < int(n_hops) - 1:
            keep_div = (hop == 0) or (hop == 1 and diverse_hop1)
            (
                states, seqs, first_states, cum_risk, path_sp, path_dem, _,
            ) = _select_adaptive_beam(
                states, seqs, first_states, cum_risk, path_sp, path_dem,
                cost, beam_width=beam_width, action_space=action_space,
                keep_diverse=keep_div,
            )

    if not finite.any():
        raise ValueError("adaptive chained search found no finite candidate")
    best = int(np.argmin(cost))
    r_best = max(0.0, float(cum_risk[best]))
    # Bottleneck (max) post-lag penalty ensures confidence is monotonically
    # non-increasing with depth, preventing cross-horizon optimism illusion.
    worst_sp = (
        float(np.max(np.clip(path_sp[best], 0.0, 1.0)))
        if (path_sp is not None and path_sp.shape[1] > 0) else 0.0
    )
    conf = float(np.exp(-r_best) * (1.0 - worst_sp))
    frontier = (
        states, seqs, first_states, cum_risk, path_sp, path_dem,
        cost, int(chunk_size) * int(n_hops),
    )
    return best, cost, finite, conf, total_cands, frontier


def _extend_clean_frontier_one_chunk(
    frontier, chunk_size, beam_width, world, agent, pending_prefix,
    stage_value, stage_value_weight, heuristic_demand, heuristic_demand_weight,
    lag=LAG, step_stride=2,
    action_top_k: int = 0, action_proposer=None,
):
    """Incrementally extend an existing frontier by one chunk (zero recompute of earlier hops)."""
    states, seqs, first_states, cum_risk, path_sp, path_dem, cost, cur_h = frontier
    action_space = tuple(getattr(agent, "action_space", (0, 1)))
    (
        states, seqs, first_states, cum_risk, path_sp, path_dem, _,
    ) = _select_adaptive_beam(
        states, seqs, first_states, cum_risk, path_sp, path_dem,
        cost, beam_width=beam_width, action_space=action_space,
        keep_diverse=True,
    )
    (
        states, seqs, first_states, cum_risk, path_sp, path_dem, added,
    ) = _expand_clean_chunk(
        states, seqs, first_states, cum_risk, path_sp, path_dem,
        start_step=cur_h, chunk_size=chunk_size,
        pending_prefix=pending_prefix, world=world, agent=agent,
        lag=lag, step_stride=step_stride, risk_weight=0.0,
        action_top_k=action_top_k, action_proposer=action_proposer,
    )
    new_h = int(cur_h) + int(chunk_size)
    cost, path_sp, path_dem, finite = _score_clean_frontier(
        states, seqs, cum_risk, path_sp, path_dem, horizon=new_h,
        stage_value=stage_value, stage_value_weight=stage_value_weight,
        heuristic_demand=heuristic_demand,
        heuristic_demand_weight=heuristic_demand_weight,
        lag=lag,
    )
    if not finite.any():
        raise ValueError("adaptive chained deepening found no finite candidate")
    best = int(np.argmin(cost))
    r_best = max(0.0, float(cum_risk[best]))
    worst_sp = (
        float(np.max(np.clip(path_sp[best], 0.0, 1.0)))
        if (path_sp is not None and path_sp.shape[1] > 0) else 0.0
    )
    conf = float(np.exp(-r_best) * (1.0 - worst_sp))
    next_frontier = (
        states, seqs, first_states, cum_risk, path_sp, path_dem, cost, new_h,
    )
    return best, cost, finite, conf, added, next_frontier


def adaptive_chained_result(
    env, world, agent, prefix, chunk_size=2, base_hops=3, max_hops=5,
    beam_width=4, wide_beam_width=6, lag=LAG, step_stride=2, risk_weight=1.0,
    stage_value=None, stage_value_weight=0.0,
    heuristic_demand=None, heuristic_demand_weight=0.0,
    controller=None,
    action_top_k: int = 0,
    action_proposer=None,
) -> SearchResult:
    """Dual-confidence decaying-threshold adaptive segmented search.

    When ``controller`` is provided and its outer causal gates are not active,
    delegates to conservative :func:`chained_shallow_result`.  When active,
    evaluates Tier 0 (``chunk_size x base_hops``, ``beam_width``) on the clean
    path-consistent online objective and releases immediately if its monotonic
    bottleneck confidence clears ``tau_0``; otherwise escalates to
    ``base_hops + 1`` with ``wide_beam_width`` (preserving action diversity
    past ``lag``) and incrementally extends the frontier one chunk at a time up
    to ``max_hops`` under strictly decaying thresholds ``tau_k = tau_0 * decay**k``.
    The final tier always releases, guaranteeing finite termination.
    """
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if int(base_hops) < 1:
        raise ValueError("base_hops must be positive")
    if int(max_hops) < int(base_hops):
        raise ValueError("max_hops must be >= base_hops")
    if int(beam_width) < 1 or int(wide_beam_width) < 1:
        raise ValueError("beam widths must be positive")

    if controller is not None and not controller.active:
        controller.record_step(False)
        return chained_shallow_result(
            env, world, agent, prefix,
            chunk_size=chunk_size, n_hops=base_hops, beam_width=beam_width,
            lag=lag, step_stride=step_stride, risk_weight=risk_weight,
            stage_value=stage_value, stage_value_weight=stage_value_weight,
            heuristic_demand=heuristic_demand,
            heuristic_demand_weight=heuristic_demand_weight,
            action_top_k=action_top_k,
            action_proposer=action_proposer,
        )

    initial = np.asarray(env.obs(), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float).reshape(-1)
    sv_w = float(stage_value_weight)
    hd_w = float(heuristic_demand_weight)
    total_candidates = 0

    # Tier 0: fast narrow segmented search
    best, cost, finite, conf, cands, frontier = _run_clean_chain_tier(
        initial, world, agent, pending_prefix,
        chunk_size=chunk_size, n_hops=base_hops, beam_width=beam_width,
        diverse_hop1=False, stage_value=stage_value, stage_value_weight=sv_w,
        heuristic_demand=heuristic_demand, heuristic_demand_weight=hd_w,
        lag=lag, step_stride=step_stride,
        action_top_k=action_top_k, action_proposer=action_proposer,
    )
    total_candidates += cands
    if controller is not None:
        tau0, decay = controller.observe_tier0_confidence(conf)
    else:
        tau0, decay = 0.72, 0.80

    exit_tier = 0
    is_final_t0 = int(base_hops) >= int(max_hops)
    thresh0 = (
        controller.threshold_for_tier(0, tau0, decay, is_final_t0)
        if controller is not None
        else (-1.0 if is_final_t0 else tau0)
    )
    if conf < thresh0:
        # Tier 1: deeper + wider with hop-1 action diversity
        next_hops = int(base_hops) + 1
        exit_tier = 1
        best, cost, finite, conf, cands, frontier = _run_clean_chain_tier(
            initial, world, agent, pending_prefix,
            chunk_size=chunk_size, n_hops=next_hops, beam_width=wide_beam_width,
            diverse_hop1=True, stage_value=stage_value, stage_value_weight=sv_w,
            heuristic_demand=heuristic_demand, heuristic_demand_weight=hd_w,
            lag=lag, step_stride=step_stride,
            action_top_k=action_top_k, action_proposer=action_proposer,
        )
        total_candidates += cands
        cur_hops = next_hops
        # Tier 2..K: zero-recompute incremental frontier deepening up to max_hops
        while cur_hops < int(max_hops):
            thresh_k = (
                controller.threshold_for_tier(exit_tier, tau0, decay, False)
                if controller is not None
                else tau0 * (decay ** exit_tier)
            )
            if conf >= thresh_k:
                break
            exit_tier += 1
            cur_hops += 1
            best, cost, finite, conf, cands, frontier = _extend_clean_frontier_one_chunk(
                frontier, chunk_size=chunk_size, beam_width=wide_beam_width,
                world=world, agent=agent, pending_prefix=pending_prefix,
                stage_value=stage_value, stage_value_weight=sv_w,
                heuristic_demand=heuristic_demand, heuristic_demand_weight=hd_w,
                lag=lag, step_stride=step_stride,
                action_top_k=action_top_k, action_proposer=action_proposer,
            )
            total_candidates += cands

    if controller is not None:
        controller.record_step(True, tier_index=exit_tier)

    states, seqs, first_states = frontier[0], frontier[1], frontier[2]
    score = (float(cost[best]), 0.0, 0.0, 0.0, 0.0, 0.0)
    return SearchResult(
        actions=np.asarray(seqs[best], dtype=float).copy(),
        score=score,
        feasible=True,
        feasible_count=int(finite.sum()),
        candidate_count=int(total_candidates),
        first_state=np.asarray(first_states[best], dtype=float).copy(),
        final_state=np.asarray(states[best], dtype=float).copy(),
    )

