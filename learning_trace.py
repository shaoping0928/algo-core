"""Frame-level learning trace: what changes around a failure frame.

This is an observational probe.  It runs the real benchmark fit path
(1500 offline frames, one epoch, one 5000-frame DAgger round, retrain),
then replays a held-out world with the trained agents and records, for
every decision:

- the planner's chosen plan and its six-column objective ledger,
- the nearest prototype, its visit count and learned value,
- the active feature roots and their learned cells,
- the memories the state retrieves and their values,
- the transition the world model predicts and the one the world took,
- the action edge the realized transition updates.

It then prints a few random frames plus the frames just before the death
frame, and reports what the episode's backward pass actually changed:
node values, action edges, feature cells, and created memories.  Nothing
in the learner is modified; every wrapper is read-only.

Usage::

    python learning_trace.py --seed 100 --eval-seed 120 --window 8
"""
from __future__ import annotations

import argparse
import json
from itertools import product

import numpy as np

import substrate_nodes as sn
from node_graph import MEMORY
from substrate_nodes import LAG, _curiosity_off, evaluate, fit

# Per-agent training-episode traces, keyed by id(agent) while the agent
# is alive.  The wrapper only records one offline death and one planner
# death per agent, so the snapshots stay bounded.
LEARN_STATE = {}


def node_label(agent, node_id):
    """Human-readable label for a node, preferring feature predicates."""
    feature = agent._feature_objects.get(node_id)
    if feature is not None:
        try:
            return feature.label()
        except Exception:
            pass
    node = agent.graph.nodes.get(node_id)
    if node is not None and node.role:
        return "%s[%s]" % (node_id, node.role)
    return str(node_id)


def pending_from_history(history, lag):
    """Most-recent-first action list from a transition row's history."""
    history = np.asarray(history, dtype=float).reshape(-1)
    pending = history[1:1 + int(lag)]
    if pending.size < int(lag):
        pending = np.concatenate([
            pending, np.zeros(int(lag) - pending.size, dtype=float)
        ])
    return pending


def encode_state(agent, observation, pending):
    """Encode an observation with an explicit most-recent-first prefix."""
    observation = np.asarray(observation, dtype=float).reshape(-1)
    lag = int(agent.lag)
    if lag:
        pending = np.asarray(pending, dtype=float).reshape(-1)[:lag]
        if pending.size < lag:
            pending = np.concatenate([
                pending, np.zeros(lag - pending.size, dtype=float)
            ])
        # The runtime's pending deque is oldest-to-newest.
        state = np.concatenate([observation, pending[::-1]])
    else:
        state = observation
    return state / agent.scales


def feature_read(agent, node_id, acting):
    """Learned statistics behind one feature root under one acting action."""
    node = agent.graph.nodes.get(node_id)
    if node is None:
        return None
    action_cells = agent.feature_action_stats.get(node_id, {})
    cell = action_cells.get(int(acting), [0, 0.0])
    count, hits = float(cell[0]), float(cell[1])
    value_cell = agent.feature_value_stats.get(node_id, {}).get(
        int(acting), [0, 0.0, {}]
    )
    value_count = float(value_cell[0])
    value_total = float(value_cell[1])
    return dict(
        id=node_id,
        label=node_label(agent, node_id),
        marginal_count=float(node.count),
        marginal_pain=float(node.value),
        action=int(acting),
        action_count=count,
        action_pain=(hits / count if count > 0 else None),
        value_count=value_count,
        value_mean=(value_total / value_count if value_count > 0 else None),
        credibility=float(agent.prediction_credibility(node_id)),
    )


def frame_context(agent, obs, prefix, action, acting):
    """Read every learned surface the runtime exposes for one state."""
    encoded = encode_state(agent, obs, prefix)
    prototype, distance = agent._nearest(encoded)
    roots = agent._active_feature_ids(encoded)
    roots = [node_id for node_id in roots if node_id in agent.feature_ids]
    memories = agent.memories_for([prototype] + roots)
    mask_table = agent._feature_mask_table(encoded[None, :agent.obs_dim])
    risk, excess = agent.pain_graded_batch(encoded[None])
    value, support = agent.feature_value_batch(
        encoded[None], mask_table, return_support=True
    )
    phi = float(sn._curiosity_potential(
        agent.evidence_batch(encoded[None], mask_table)
    )[0])
    node = agent.graph.nodes.get(prototype)
    edges = []
    for edge in agent.graph.action_edges_from(prototype, int(acting)):
        edges.append(dict(
            target=edge.target,
            count=int(edge.count),
            q_pain=float(edge.q_pain),
            q_dopamine=float(edge.q_dopamine),
        ))
    edges.sort(key=lambda item: -item["count"])
    memory_rows = []
    for memory_id, weight in sorted(
        memories.items(), key=lambda item: -item[1]
    ):
        memory = agent.graph.nodes.get(memory_id)
        if memory is None:
            continue
        memory_rows.append(dict(
            id=memory_id,
            value=float(memory.value),
            dopamine=float(memory.dopamine),
            count=int(memory.count),
            trace=float(memory.trace),
            weight=float(weight),
        ))
    return dict(
        encoded=encoded,
        prototype=prototype,
        distance=float(distance),
        prototype_count=(int(node.count) if node is not None else 0),
        prototype_value=(float(node.value) if node is not None else 0.0),
        prototype_dopamine=(
            float(node.dopamine) if node is not None else 0.0
        ),
        active_roots=[feature_read(agent, node_id, acting) for node_id in roots],
        root_count=len(roots),
        memory=memory_rows,
        risk=float(risk[0]),
        excess=float(excess[0]),
        value=float(value[0]),
        support=float(support[0]),
        phi=phi,
        action_edges=edges[:8],
    )


def trace_episode(seed, world, agent, env_seed, cap, horizon, step_stride,
                  lag=LAG):
    """Run one evaluation episode and record each decision and outcome."""
    records = []
    original = sn.policy_result

    def traced_policy_result(env, world_, agent_, prefix, horizon_, **kwargs):
        obs = np.asarray([list(env.obs())], dtype=float)
        prefix_copy = np.asarray(prefix, dtype=float).copy()
        result = original(
            env, world_, agent_, prefix, horizon_, **kwargs
        )
        action = int(result.actions[0])
        acting = int(round(float(prefix_copy[lag - 1]))) if lag else action
        context = frame_context(
            agent_, obs[0], prefix_copy, action, acting
        )
        context.pop("encoded")
        records.append(dict(
            frame=int(env.t),
            obs=np.asarray(obs[0], dtype=float).copy(),
            prefix=prefix_copy,
            action=action,
            acting=acting,
            plan=np.asarray(result.actions, dtype=float).copy(),
            score=[float(item) for item in np.asarray(result.score).reshape(-1)],
            feasible=bool(result.feasible),
            feasible_count=int(result.feasible_count),
            candidate_count=int(result.candidate_count),
            **context,
        ))
        return result

    sn.policy_result = traced_policy_result
    try:
        with _curiosity_off(agent):
            frames, _actions, rows = evaluate(
                seed, world, agent, cap=cap, horizon=horizon, lag=lag,
                step_stride=step_stride, env_seed=int(env_seed),
                record=True, episode=0,
            )
    finally:
        sn.policy_result = original

    # Attach the realized outcome and the next state's reads.  The next
    # decision's prefix is [action] + prefix[:-1]; this is exactly the
    # runtime's pending deque at the next state.
    for index, record in enumerate(records):
        row = rows[index]
        next_prefix = np.concatenate([
            [float(record["action"])], record["prefix"][:-1]
        ]) if len(record["prefix"]) else np.zeros(0)
        next_encoded = encode_state(agent, row["next_obs"], next_prefix)
        next_prototype, next_distance = agent._nearest(next_encoded)
        next_roots = [
            node_id for node_id in agent._active_feature_ids(next_encoded)
            if node_id in agent.feature_ids
        ]
        next_memories = agent.memories_for([next_prototype] + next_roots)
        history = np.asarray(row["history"], dtype=float)
        prediction = world.predict(
            record["obs"][None, :], history[None, :]
        )[0]
        error = np.abs(prediction - np.asarray(row["next_obs"], dtype=float))
        edge = agent.graph.action_edges.get((
            record["prototype"], next_prototype, int(record["acting"])
        ))
        transition = agent.graph.edge_node(
            record["prototype"], next_prototype
        )
        transition_node = (
            agent.graph.nodes.get(transition) if transition else None
        )
        record.update(
            next_obs=np.asarray(row["next_obs"], dtype=float).copy(),
            pain=float(row["signals"][1]),
            dopamine=float(row["signals"][0]),
            done=bool(row["done"]),
            next_prototype=next_prototype,
            next_distance=float(next_distance),
            next_roots=[
                node_label(agent, node_id) for node_id in next_roots
            ],
            next_memories=len(next_memories),
            prediction_error=float(error.mean()),
            prediction_error_max=float(error.max()),
            prediction=prediction,
            action_edge_count=(int(edge.count) if edge else 0),
            action_edge_q=(float(edge.q_pain) if edge else 0.0),
            transition_node=(
                transition if transition_node is not None else None
            ),
            transition_value=(
                float(transition_node.value)
                if transition_node is not None else 0.0
            ),
        )
    return frames, records, rows


def best_costs_by_first_action(agent, world, record, horizon, lag=LAG,
                               step_stride=2):
    """Best plan cost for each first action, the planner's action margin."""
    costs = {0: [], 1: []}
    for plan in product((0, 1), repeat=horizon):
        cost, _ = sn.plan_score(
            record["obs"], world, agent, record["prefix"], plan, horizon,
            lag=lag, step_stride=step_stride,
        )
        if np.isfinite(cost):
            costs[plan[0]].append(float(cost))
    summary = {}
    for action, values in costs.items():
        values = sorted(values)
        summary[action] = dict(
            best=(values[0] if values else None),
            top3=[float(value) for value in values[:3]],
            count=len(values),
        )
    if summary[0]["best"] is not None and summary[1]["best"] is not None:
        summary["margin_1_minus_0"] = (
            summary[1]["best"] - summary[0]["best"]
        )
    else:
        summary["margin_1_minus_0"] = None
    return summary


def read_with_agent(agent, record):
    """Re-read a recorded state with another trained agent."""
    encoded = encode_state(agent, record["obs"], record["prefix"])
    prototype, distance = agent._nearest(encoded)
    roots = [
        node_id for node_id in agent._active_feature_ids(encoded)
        if node_id in agent.feature_ids
    ]
    memories = agent.memories_for([prototype] + roots)
    mask_table = agent._feature_mask_table(encoded[None, :agent.obs_dim])
    risk, _excess = agent.pain_graded_batch(encoded[None])
    value, support = agent.feature_value_batch(
        encoded[None], mask_table, return_support=True
    )
    return dict(
        prototype=prototype,
        distance=float(distance),
        roots=len(roots),
        memories=len(memories),
        risk=float(risk[0]),
        value=float(value[0]),
        support=float(support[0]),
    )


def snapshot_feature_cells(agent, focus_roots):
    cells = {}
    for node_id in focus_roots:
        action_rows = {}
        for action, cell in agent.feature_action_stats.get(node_id, {}).items():
            action_rows[int(action)] = [float(cell[0]), float(cell[1])]
        value_rows = {}
        for action, cell in agent.feature_value_stats.get(node_id, {}).items():
            value_rows[int(action)] = [float(cell[0]), float(cell[1])]
        cells[node_id] = dict(action=action_rows, value=value_rows)
    return cells


def snapshot_episode(agent, rows, window=12):
    """Everything the episode's backward pass could change."""
    focus_nodes = {}
    focus_edges = set()
    focus_roots = set()
    focus_memories = set()
    focus_rows = []
    for index, row in enumerate(rows[-int(window):], start=max(
        0, len(rows) - int(window)
    )):
        pending = pending_from_history(row["history"], agent.lag)
        encoded = encode_state(agent, row["obs"], pending)
        prototype, distance = agent._nearest(encoded)
        roots = [
            node_id for node_id in agent._active_feature_ids(encoded)
            if node_id in agent.feature_ids
        ]
        memories = agent.memories_for([prototype] + roots)
        next_encoded = encode_state(
            agent, row["next_obs"],
            np.concatenate([[float(row["history"][0])], pending[:-1]])
            if len(pending) else np.zeros(0),
        )
        next_prototype, _next_distance = agent._nearest(next_encoded)
        acting = int(round(float(np.asarray(row["history"])[agent.lag]))) \
            if len(row["history"]) > agent.lag else int(row["history"][0])
        focus_rows.append(dict(
            index=int(index),
            frame=int(index),
            prototype=prototype,
            distance=float(distance),
            roots=[node_label(agent, node_id) for node_id in roots],
            memories=len(memories),
            acting=acting,
            pain=float(row["signals"][1]),
            done=bool(row["done"]),
        ))
        for node_id in [prototype] + roots + list(memories):
            if node_id is not None:
                focus_nodes[node_id] = True
                node = agent.graph.nodes.get(node_id)
                if node is not None and node.kind == MEMORY:
                    focus_memories.add(node_id)
        focus_roots.update(roots)
        focus_edges.add((prototype, next_prototype, acting))

    nodes = {
        node_id: (
            float(node.value), float(node.dopamine), int(node.count),
            float(node.immediate_pain), float(node.trace),
        )
        for node_id, node in agent.graph.nodes.items()
    }
    edges = {
        key: (int(edge.count), float(edge.q_pain), float(edge.q_dopamine))
        for key, edge in agent.graph.action_edges.items()
    }
    feature_cells = snapshot_feature_cells(agent, focus_roots)
    memory_nodes = {
        node_id: (float(node.value), int(node.count), float(node.trace))
        for node_id, node in agent.graph.nodes.items()
        if node.kind == MEMORY
    }
    return dict(
        focus_nodes=set(focus_nodes),
        focus_edges=set(focus_edges),
        focus_roots=set(focus_roots),
        focus_memories=set(focus_memories),
        focus_rows=focus_rows,
        nodes=nodes,
        edges=edges,
        feature_cells=feature_cells,
        memory_nodes=memory_nodes,
        counts=dict(
            nodes=len(agent.graph.nodes),
            action_edges=len(agent.graph.action_edges),
            memories=sum(
                1 for node in agent.graph.nodes.values()
                if node.kind == MEMORY
            ),
        ),
    )


def episode_deltas(agent, before):
    """What the backward pass changed, focused on the pre-death frames."""
    after_nodes = {
        node_id: (
            float(node.value), float(node.dopamine), int(node.count),
            float(node.immediate_pain), float(node.trace),
        )
        for node_id, node in agent.graph.nodes.items()
    }
    after_edges = {
        key: (int(edge.count), float(edge.q_pain), float(edge.q_dopamine))
        for key, edge in agent.graph.action_edges.items()
    }
    after_cells = snapshot_feature_cells(agent, before["focus_roots"])
    after_memories = {
        node_id: (float(node.value), int(node.count), float(node.trace))
        for node_id, node in agent.graph.nodes.items()
        if node.kind == MEMORY
    }
    node_rows = []
    new_node_rows = []
    for node_id, new in after_nodes.items():
        old = before["nodes"].get(node_id)
        if old is None:
            if abs(new[0]) > 1e-12 or new[2] > 0:
                new_node_rows.append(dict(
                    id=node_id,
                    label=node_label(agent, node_id),
                    count=int(new[2]),
                    value=float(new[0]),
                    dopamine=float(new[1]),
                ))
            continue
        if abs(new[0] - old[0]) <= 1e-12 and abs(
            new[1] - old[1]
        ) <= 1e-12:
            continue
        node_rows.append(dict(
            id=node_id,
            label=node_label(agent, node_id),
            new=False,
            value_before=old[0],
            value_after=new[0],
            value_delta=new[0] - old[0],
            dopamine_before=old[1],
            dopamine_after=new[1],
            dopamine_delta=new[1] - old[1],
            count_before=old[2],
            count_after=new[2],
        ))
    for row in new_node_rows:
        row.update(
            new=True,
            value_before=0.0,
            value_after=row["value"],
            value_delta=row["value"],
            dopamine_before=0.0,
            dopamine_after=row["dopamine"],
            dopamine_delta=row["dopamine"],
            count_before=0,
            count_after=row["count"],
        )
    node_rows.extend(new_node_rows)
    node_rows.sort(key=lambda item: -abs(item["value_delta"]))
    edge_rows = []
    new_edge_rows = []
    for key, new in after_edges.items():
        old = before["edges"].get(key)
        if old is None:
            if new[0] > 0:
                new_edge_rows.append(dict(
                    source=key[0], target=key[1], action=key[2],
                    count_before=0, count_after=new[0],
                    q_before=0.0, q_after=new[1],
                    q_delta=new[1], new=True,
                ))
            continue
        if old[0] == new[0] and abs(old[1] - new[1]) <= 1e-12:
            continue
        edge_rows.append(dict(
            source=key[0],
            target=key[1],
            action=key[2],
            count_before=old[0],
            count_after=new[0],
            q_before=old[1],
            q_after=new[1],
            q_delta=new[1] - old[1],
            new=False,
        ))
    for key in before["focus_edges"]:
        old = before["edges"].get(key)
        new = after_edges.get(key)
        if old is None and new is None:
            continue
        old = old or (0, 0.0, 0.0)
        new = new or (0, 0.0, 0.0)
        if old[0] == new[0] and abs(old[1] - new[1]) <= 1e-12:
            continue
        edge_rows.append(dict(
            source=key[0],
            target=key[1],
            action=key[2],
            count_before=old[0],
            count_after=new[0],
            q_before=old[1],
            q_after=new[1],
            q_delta=new[1] - old[1],
            new=old[0] == 0,
        ))
    edge_rows.sort(key=lambda item: -abs(item["q_delta"]))
    new_edge_rows.sort(key=lambda item: -item["count_after"])
    focus_new_edges = [
        row for row in new_edge_rows
        if row["source"] in before["focus_nodes"]
    ][:8]
    cell_rows = []
    for node_id in before["focus_roots"]:
        old = before["feature_cells"].get(node_id, {})
        new = after_cells.get(node_id, {})
        for action in sorted(set(old.get("action", {})) | set(
            new.get("action", {})
        )):
            old_cell = old.get("action", {}).get(action, [0.0, 0.0])
            new_cell = new.get("action", {}).get(action, [0.0, 0.0])
            cell_rows.append(dict(
                id=node_id,
                label=node_label(agent, node_id),
                action=int(action),
                count_before=old_cell[0],
                count_after=new_cell[0],
                hits_before=old_cell[1],
                hits_after=new_cell[1],
                pain_before=(
                    old_cell[1] / old_cell[0] if old_cell[0] > 0 else None
                ),
                pain_after=(
                    new_cell[1] / new_cell[0] if new_cell[0] > 0 else None
                ),
            ))
    cell_rows.sort(key=lambda item: -(
        item["count_after"] - item["count_before"]
    ))
    created = sorted(set(after_memories) - set(before["memory_nodes"]))
    memory_rows = []
    for memory_id in created:
        value, count, trace = after_memories[memory_id]
        memory_rows.append(dict(
            id=memory_id, value=value, count=count, trace=trace
        ))
    changed = sum(
        1 for node_id, old in before["nodes"].items()
        if node_id in after_nodes
        and abs(after_nodes[node_id][0] - old[0]) > 1e-12
    )
    changed_values = [
        abs(after_nodes[node_id][0] - old[0])
        for node_id, old in before["nodes"].items()
        if node_id in after_nodes
        and abs(after_nodes[node_id][0] - old[0]) > 1e-12
    ]
    return dict(
        counts_before=before["counts"],
        counts_after=dict(
            nodes=len(agent.graph.nodes),
            action_edges=len(agent.graph.action_edges),
            memories=sum(
                1 for node in agent.graph.nodes.values()
                if node.kind == MEMORY
            ),
        ),
        changed_nodes=changed,
        new_nodes_with_value=len(new_node_rows),
        new_action_edges=len(new_edge_rows),
        mean_abs_value_delta=(
            float(np.mean(changed_values)) if changed_values else 0.0
        ),
        node_rows=node_rows[:10],
        edge_rows=edge_rows[:10],
        new_edge_rows=new_edge_rows[:10],
        focus_new_edges=focus_new_edges,
        cell_rows=cell_rows[:10],
        created_memories=memory_rows,
        lost_memories=sorted(set(before["memory_nodes"]) - set(after_memories)),
    )


def install_learning_wrapper():
    """Wrap `learn_episode`; record one offline and one planner death."""
    original = sn.NodeRuntime.learn_episode

    def traced_learn(self, rows):
        rows = list(rows)
        episode = int(rows[0]["episode"]) if rows else -1
        death = bool(rows) and bool(rows[-1]["done"]) and any(
            float(row["signals"][1]) > 0.5 for row in rows
        )
        slot = (
            "planner" if episode >= 50000 else "offline"
        )
        state = LEARN_STATE.setdefault(
            id(self), dict(offline=None, planner=None)
        )
        trace = None
        # The very first episodes run before any prototype exists, so their
        # pre-update state is empty and the snapshot would miss everything.
        # Wait until the structure is populated, then keep the first death.
        mature = (
            len(self.graph.nodes) >= 32
            or len(self._prototype_ids) >= 16
        )
        if death and state.get(slot) is None and mature:
            before = snapshot_episode(self, rows)
            trace = dict(
                episode=episode,
                slot=slot,
                frames=len(rows),
                before=before,
            )
            state[slot] = trace
        original(self, rows)
        if trace is not None:
            trace["delta"] = episode_deltas(self, before)

    sn.NodeRuntime.learn_episode = traced_learn
    return original


def run_fit(seed, steps, epochs, max_nodes, horizon, dagger_rounds,
            dagger_frames, dagger_cap, coverage):
    """Run the benchmark fit path and capture every trained agent."""
    captured = []
    original_train = sn._train

    def traced_train(transitions, **kwargs):
        world, agent = original_train(transitions, **kwargs)
        agent._trace_phase = len(captured)
        agent._trace_transitions = len(transitions)
        agent._trace_learn = LEARN_STATE.pop(id(agent), {})
        captured.append(dict(
            phase=len(captured),
            world=world,
            agent=agent,
            transitions=len(transitions),
        ))
        return world, agent

    sn._train = traced_train
    try:
        fit(
            seed=seed, steps=steps, coverage=coverage,
            epochs=epochs, max_nodes=max_nodes,
            birth_threshold=0.45,
            dagger_rounds=dagger_rounds,
            dagger_frames=dagger_frames,
            dagger_cap=dagger_cap,
            dagger_horizon=horizon,
            step_stride=2,
            curiosity_weight=0.5,
            memory_credit=False,
        )
    finally:
        sn._train = original_train
    return captured


def survival(world, agent, seed, eval_seeds, cap, horizon):
    rows = []
    with _curiosity_off(agent):
        for eval_seed in eval_seeds:
            frames, actions, _ = evaluate(
                seed, world, agent, cap=cap, horizon=horizon, lag=LAG,
                step_stride=2, env_seed=int(eval_seed) * 10000,
            )
            rows.append(dict(
                eval_seed=int(eval_seed),
                frames=int(frames),
                action_rate=float(np.mean(actions)) if actions else 0.0,
            ))
    return rows


def episode_summary(records):
    if not records:
        return dict(frames=0, death_frame=None)
    pain_rows = [record for record in records if record["pain"] > 0.5]
    death = records[-1]["done"] or records[-1]["pain"] > 0.5
    return dict(
        frames=len(records),
        death_frame=len(records) if death else None,
        pain_rows=len(pain_rows),
        action_rate=float(np.mean([record["action"] for record in records])),
        root_firing=float(np.mean([
            record["root_count"] > 0 for record in records
        ])),
        memory_retrieval=float(np.mean([
            len(record["memory"]) > 0 for record in records
        ])),
        mean_risk=float(np.mean([record["risk"] for record in records])),
        mean_value=float(np.mean([record["value"] for record in records])),
        mean_support=float(np.mean([record["support"] for record in records])),
        mean_prediction_error=float(np.mean([
            record["prediction_error"] for record in records
        ])),
        mean_feasible_share=float(np.mean([
            record["feasible_count"] / max(record["candidate_count"], 1)
            for record in records
        ])),
    )


def select_frames(records, window, random_frames, rng):
    last = len(records) - 1
    death_window = list(range(max(0, last - int(window)), last + 1))
    candidates = [
        index for index in range(len(records))
        if index not in set(death_window)
    ]
    picked = []
    if candidates:
        picked = sorted(rng.choice(
            candidates, size=min(int(random_frames), len(candidates)),
            replace=False,
        ).tolist())
    return picked, death_window


def print_frame(record):
    print(
        "  frame %-4d action=%d acting=%d  risk=%.3f value=%.4f "
        "support=%.1f  proto=%s d=%.2f  roots=%d mem=%d  "
        "pred_err=%.3f  pain=%.0f%s"
        % (
            record["frame"], record["action"], record["acting"],
            record["risk"], record["value"], record["support"],
            record["prototype"], record["distance"], record["root_count"],
            len(record["memory"]), record["prediction_error"],
            record["pain"], " DEAD" if record["done"] else "",
        )
    )
    print(
        "    score hazard=%.3f -dopamine=%.3f risk=%.3f value=%.3f "
        "curiosity=%.3f q=%.3f  feasible=%d/%d"
        % tuple(record["score"] + [
            record["feasible_count"], record["candidate_count"]
        ])
    )
    roots = record["active_roots"][:4]
    if roots:
        print("    roots: " + " | ".join(
            "%s P=%.2f n=%.0f valn=%.0f" % (
                root["label"], root["marginal_pain"],
                root["marginal_count"], root["value_count"],
            ) for root in roots
        ))
    else:
        print("    roots: (none)")
    if record["memory"]:
        print("    memory: " + " | ".join(
            "%s v=%.2f n=%d w=%.2f" % (
                memory["id"], memory["value"], memory["count"],
                memory["weight"],
            ) for memory in record["memory"][:3]
        ))
    print(
        "    transition: %s -> %s (d=%.2f) edge_n=%d q=%.2f  "
        "pred_err_max=%.3f"
        % (
            record["prototype"], record["next_prototype"],
            record["next_distance"], record["action_edge_count"],
            record["action_edge_q"], record["prediction_error_max"],
        )
    )
    if record["action_edges"]:
        print("    learned edges: " + " | ".join(
            "->%s n=%d q=%.2f" % (
                edge["target"], edge["count"], edge["q_pain"]
            ) for edge in record["action_edges"][:4]
        ))


def emit_window(agent, world, eval_seed, records, window, random_frames,
                rng, label, horizon):
    """Print one sampled/death window and its plan-cost margins."""
    picked, death_window = select_frames(
        records, window, random_frames, rng
    )
    print("-" * 78)
    print("%s (world %d)" % (label, eval_seed))
    print("random frames: %s  death window: %s" % (
        picked, death_window
    ))
    costs_by_frame = {}
    for index in sorted(set(picked) | set(death_window)):
        record = records[index]
        print_frame(record)
        costs_by_frame[index] = best_costs_by_first_action(
            agent, world, record, horizon
        )
        print(
            "    plan costs: best(a=0)=%s best(a=1)=%s margin(1-0)=%s"
            % (
                ("%.3f" % costs_by_frame[index][0]["best"]
                 if costs_by_frame[index][0]["best"] is not None
                 else "None"),
                ("%.3f" % costs_by_frame[index][1]["best"]
                 if costs_by_frame[index][1]["best"] is not None
                 else "None"),
                ("%.3f" % costs_by_frame[index]["margin_1_minus_0"]
                 if costs_by_frame[index]["margin_1_minus_0"] is not None
                 else "None"),
            )
        )
        print()
    return picked, death_window, costs_by_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=120)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--dagger-frames", type=int, default=5000)
    parser.add_argument("--dagger-cap", type=int, default=5000)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--random-frames", type=int, default=4)
    parser.add_argument("--report", default="")
    parser.add_argument("--trace-seeds", default="",
                        help="comma-separated eval seeds to trace; "
                             "default: --eval-seed")
    args = parser.parse_args()
    eval_seeds = []
    for part in args.eval_seeds.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            eval_seeds.extend(range(int(first), int(last) + 1))
        else:
            eval_seeds.append(int(part))
    trace_seeds = (
        [int(item) for item in args.trace_seeds.split(",") if item.strip()]
        if args.trace_seeds else list(eval_seeds)
    )

    original_learn = install_learning_wrapper()
    try:
        captured = run_fit(
            args.seed, args.steps, args.epochs, args.max_nodes,
            args.horizon, args.dagger_rounds, args.dagger_frames,
            args.dagger_cap, args.coverage,
        )
    finally:
        sn.NodeRuntime.learn_episode = original_learn

    report = dict(
        config=vars(args),
        phases=[],
    )
    phase_records = {}
    rng = np.random.RandomState(args.seed + 7919)
    for phase in captured:
        agent = phase["agent"]
        world = phase["world"]
        print()
        print("=" * 78)
        print("phase %d: %d training transitions" % (
            phase["phase"], phase["transitions"]
        ))
        print("structure: prototypes=%d nodes=%d action_edges=%d "
              "memories=%d feature_roots=%d value_roots=%d" % (
                  len(agent._prototype_ids), len(agent.graph.nodes),
                  len(agent.graph.action_edges),
                  sum(1 for node in agent.graph.nodes.values()
                      if node.kind == MEMORY),
                  len(agent.feature_ids), len(agent.value_feature_ids),
              ))
        feature_obs = sum(
            cell[0] for stats in agent.feature_action_stats.values()
            for cell in stats.values()
        )
        value_obs = sum(
            cell[0] for stats in agent.feature_value_stats.values()
            for cell in stats.values()
        )
        print("feature cells: action observations=%d  value observations=%d"
              % (feature_obs, value_obs))
        survival_rows = survival(
            world, agent, args.seed, eval_seeds, args.cap, args.horizon
        )
        print("held-out survival: %s  mean=%.1f median=%.1f" % (
            [row["frames"] for row in survival_rows],
            float(np.mean([row["frames"] for row in survival_rows])),
            float(np.median([row["frames"] for row in survival_rows])),
        ))

        candidates = []
        fallback = None
        for eval_seed in trace_seeds:
            frames, records, _rows = trace_episode(
                args.seed, world, agent, int(eval_seed) * 10000,
                args.cap, args.horizon, 2,
            )
            summary = episode_summary(records)
            tail = records[-min(len(records), args.window + 1):]
            mean_distance = float(np.mean([
                record["distance"] for record in tail
            ])) if tail else float("inf")
            summary["tail_mean_distance"] = mean_distance
            print("traced world %d: %s" % (eval_seed, json.dumps(summary)))
            if summary["death_frame"] is not None:
                candidates.append((
                    mean_distance, int(eval_seed), records, summary
                ))
            if fallback is None or len(records) > len(fallback[2]):
                fallback = (
                    mean_distance, int(eval_seed), records, summary
                )
        if candidates:
            # Report two failures when they differ: the most in-distribution
            # one (closest prototypes) and the one whose own reads actually
            # signalled danger.  A death with no local evidence and a death
            # with a rising hazard are different failure stories.
            focus = min(candidates, key=lambda item: item[0])

            def signal(item):
                tail = item[2][-min(len(item[2]), args.window + 1):]
                return (
                    float(np.mean([row["risk"] for row in tail])),
                    float(np.mean([row["root_count"] > 0 for row in tail])),
                )

            signaled = max(candidates, key=signal)
        elif fallback is not None:
            focus = fallback
            signaled = None
        else:
            continue

        eval_seed, records, summary = focus[1:]
        picked, death_window, costs_by_frame = emit_window(
            agent, world, eval_seed, records, args.window,
            args.random_frames, rng, "closest failure", args.horizon,
        )
        signaled_report = None
        phase_records[phase["phase"]] = dict(
            focus=(eval_seed, records),
            signaled=None,
        )
        if signaled is not None and signaled[1] != focus[1]:
            sig_seed, sig_records, sig_summary = signaled[1:]
            phase_records[phase["phase"]]["signaled"] = (
                sig_seed, sig_records
            )
            sig_picked, sig_window, sig_costs = emit_window(
                agent, world, sig_seed, sig_records, args.window,
                args.random_frames, rng, "most signaled failure",
                args.horizon,
            )
            signaled_report = dict(
                eval_seed=sig_seed,
                summary=sig_summary,
                random_frames=sig_picked,
                death_window=sig_window,
                plan_costs={
                    str(index): sig_costs[index] for index in sig_costs
                },
            )

        # Training feedback for this phase's exemplar death episode.
        learn_traces = getattr(agent, "_trace_learn", {}) or {}
        learn_trace = learn_traces.get("planner") or learn_traces.get(
            "offline"
        )
        if learn_trace:
            delta = learn_trace["delta"]
            print("-" * 78)
            print(
                "training exemplar: episode %d (%s) frames=%d  "
                "nodes %d->%d  action_edges %d->%d  memories %d->%d"
                % (
                    learn_trace["episode"], learn_trace["slot"],
                    learn_trace["frames"],
                    delta["counts_before"]["nodes"],
                    delta["counts_after"]["nodes"],
                    delta["counts_before"]["action_edges"],
                    delta["counts_after"]["action_edges"],
                    delta["counts_before"]["memories"],
                    delta["counts_after"]["memories"],
                )
            )
            print(
                "  changed nodes=%d  new nodes with value=%d  mean|delta "
                "value|=%.5f"
                % (
                    delta["changed_nodes"],
                    delta["new_nodes_with_value"],
                    delta["mean_abs_value_delta"],
                )
            )
            print("  pre-death focus rows:")
            for row in learn_trace["before"]["focus_rows"][-6:]:
                print(
                    "    row %-4d proto=%s d=%.2f acting=%d pain=%.0f "
                    "roots=%d mem=%d  %s"
                    % (
                        row["index"], row["prototype"], row["distance"],
                        row["acting"], row["pain"], len(row["roots"]),
                        row["memories"], ",".join(row["roots"][:2]),
                    )
                )
            print("  node value updates (top 8, including newly born):")
            for row in delta["node_rows"][:8]:
                print(
                    "    %s: %.4f -> %.4f (d=%+.4f)  dopamine %+.4f%s"
                    % (
                        row["label"], row["value_before"],
                        row["value_after"], row["value_delta"],
                        row["dopamine_delta"],
                        "  NEW" if row.get("new") else "",
                    )
                )
            print("  action-edge updates:")
            for row in delta["edge_rows"][:6]:
                print(
                    "    %s -[%d]-> %s: n %d->%d  q %.3f -> %.3f (d=%+.3f)%s"
                    % (
                        row["source"], row["action"], row["target"],
                        row["count_before"], row["count_after"],
                        row["q_before"], row["q_after"], row["q_delta"],
                        "  NEW" if row.get("new") else "",
                    )
                )
            print("  new action edges: %d" % delta["new_action_edges"])
            for row in delta["new_edge_rows"][:4]:
                print(
                    "    %s -[%d]-> %s: n=%d q=%.3f"
                    % (
                        row["source"], row["action"], row["target"],
                        row["count_after"], row["q_after"],
                    )
                )
            if delta.get("focus_new_edges"):
                print("  new edges out of the pre-death prototypes:")
                for row in delta["focus_new_edges"]:
                    print(
                        "    %s -[%d]-> %s: n=%d q=%.3f"
                        % (
                            row["source"], row["action"], row["target"],
                            row["count_after"], row["q_after"],
                        )
                    )
            print("  feature-cell updates: %d rows" % len(delta["cell_rows"]))
            for row in delta["cell_rows"][:6]:
                print(
                    "    %s a=%d: n %.0f->%.0f pain %s -> %s"
                    % (
                        row["label"], row["action"], row["count_before"],
                        row["count_after"],
                        ("%.2f" % row["pain_before"]
                         if row["pain_before"] is not None else "None"),
                        ("%.2f" % row["pain_after"]
                         if row["pain_after"] is not None else "None"),
                    )
                )
            print("  created memories: %d  lost: %d" % (
                len(delta["created_memories"]), len(delta["lost_memories"])
            ))
            for row in delta["created_memories"][:4]:
                print(
                    "    %s value=%.2f count=%d trace=%.2f"
                    % (row["id"], row["value"], row["count"], row["trace"])
                )

        report["phases"].append(dict(
            phase=phase["phase"],
            transitions=phase["transitions"],
            structure=dict(
                prototypes=len(agent._prototype_ids),
                nodes=len(agent.graph.nodes),
                action_edges=len(agent.graph.action_edges),
                memories=sum(
                    1 for node in agent.graph.nodes.values()
                    if node.kind == MEMORY
                ),
                feature_roots=len(agent.feature_ids),
                value_roots=len(agent.value_feature_ids),
                feature_action_observations=feature_obs,
                feature_value_observations=value_obs,
            ),
            survival=survival_rows,
            traced_world=eval_seed,
            traced_summary=summary,
            random_frames=picked,
            death_window=death_window,
            signaled=signaled_report,
            frames=[
                {key: value for key, value in record.items()
                 if key not in ("encoded", "prediction", "obs", "next_obs",
                                "prefix", "plan", "active_roots", "memory",
                                "action_edges")}
                for record in records
            ],
            plan_costs={
                str(index): costs_by_frame[index]
                for index in costs_by_frame
            },
            training_exemplar=(
                dict(
                    episode=learn_trace["episode"],
                    slot=learn_trace["slot"],
                    frames=learn_trace["frames"],
                    delta=learn_trace["delta"],
                ) if learn_trace else None
            ),
        ))

    # Cross-phase reads on the exact states each phase failed on.  This is
    # the same physical observation and pending-action prefix, re-read by
    # the offline-only agent and the DAgger-trained agent, so the change in
    # the learned surfaces is visible without a policy confound.
    if len(captured) >= 2:
        first = captured[0]
        second = captured[-1]
        shifts = []
        for label, phase_data in (
            ("phase-0 signaled", phase_records.get(0, {}).get("signaled")),
            ("phase-1 signaled", phase_records.get(1, {}).get("signaled")),
            ("phase-1 closest", phase_records.get(1, {}).get("focus")),
        ):
            if phase_data is None:
                continue
            eval_seed, records = phase_data
            last = len(records) - 1
            window = records[max(0, last - args.window):]
            print()
            print("-" * 78)
            print("cross-phase reads: %s (world %d)" % (label, eval_seed))
            for record in window:
                old = read_with_agent(first["agent"], record)
                new = read_with_agent(second["agent"], record)
                shifts.append(dict(
                    label=label,
                    eval_seed=eval_seed,
                    frame=record["frame"],
                    risk_before=old["risk"], risk_after=new["risk"],
                    value_before=old["value"], value_after=new["value"],
                    support_before=old["support"],
                    support_after=new["support"],
                    roots_before=old["roots"], roots_after=new["roots"],
                    memories_before=old["memories"],
                    memories_after=new["memories"],
                    distance_before=old["distance"],
                    distance_after=new["distance"],
                    prototype_before=old["prototype"],
                    prototype_after=new["prototype"],
                ))
                print(
                    "  frame %-4d risk %.3f->%.3f  value %.4f->%.4f  "
                    "roots %d->%d  mem %d->%d  d %.2f->%.2f"
                    % (
                        record["frame"], old["risk"], new["risk"],
                        old["value"], new["value"], old["roots"],
                        new["roots"], old["memories"], new["memories"],
                        old["distance"], new["distance"],
                    )
                )
        report["cross_phase"] = shifts

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
    print()
    print("report written to %s" % args.report if args.report else "")


if __name__ == "__main__":
    main()
