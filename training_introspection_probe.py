"""Introspection probe: watch the substrate learn at several time points.

The probe runs the real benchmark fit path (`fit`, DAgger rounds included)
with observational wrappers only - no learning behavior is changed:

- ``_train`` is wrapped so every phase (the offline leg, then the retrain
  after each planner round) is snapshotted as soon as it finishes.
- ``NodeRuntime.learn_episode`` is wrapped to record the per-episode value
  correction magnitudes (mean |delta value| / |delta dopamine| over touched
  nodes) - the feedback that runs on every transition, pain or not.
- ``NodeRuntime._record_prediction`` is wrapped to record every pre-update
  prediction against the actual outcome, so error-correction dynamics can
  be traced within a phase (quarters of the stream) and split by whether
  the actual outcome was a pain event or normal flight.

At each snapshot the probe measures: structure growth, feature extraction
(root count and activation coverage on a fixed probe stream), memory
growth, the four objective reads (threat, sequence value, evidence, valence)
at a fixed set of states, prediction calibration (the readout's own
prediction-vs-actual ledger), and decision behavior (held-out survival and
action rate).

Usage::

    python training_introspection_probe.py --seed 100 --rounds 2 \
        --eval-seeds 120-124 --report introspect.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import node_runtime
import substrate_nodes as sn
from eval_protocol import parse_seed_spec
from node_graph import MEMORY
from node_runtime import NodeRuntime
from substrate_nodes import (
    LAG,
    _curiosity_off,
    _curiosity_potential,
    _spearman,
    collect,
    evaluate,
    fit,
)

STATE = {"phase": 0, "episode": 0, "row": 0}
EPISODE_TRACE = []
PRED_TRACE = []


def install_wrappers():
    """Wrap the three learning entry points with recording-only shims."""
    orig_learn = NodeRuntime.learn_episode
    orig_record = NodeRuntime._record_prediction

    def learn_episode(self, rows):
        before = {
            node_id: (node.value, node.dopamine)
            for node_id, node in self.graph.nodes.items()
        }
        pain = float(sum(
            float(np.asarray(row["signals"], dtype=float)[1])
            for row in rows
        ))
        orig_learn(self, rows)
        value_delta = []
        dopamine_delta = []
        changed = 0
        for node_id, node in self.graph.nodes.items():
            old = before.get(node_id)
            if old is None:
                continue
            value_change = abs(node.value - old[0])
            dopamine_change = abs(node.dopamine - old[1])
            if value_change > 1e-12 or dopamine_change > 1e-12:
                changed += 1
            value_delta.append(value_change)
            dopamine_delta.append(dopamine_change)
        changed_value = [
            change for change in value_delta if change > 1e-12
        ]
        EPISODE_TRACE.append(dict(
            phase=STATE["phase"],
            episode=STATE["episode"],
            frames=len(rows),
            pain=pain,
            nodes=len(value_delta),
            changed=changed,
            value_delta=(
                float(np.mean(changed_value)) if changed_value else 0.0
            ),
            dopamine_delta=float(np.mean([
                change for change in dopamine_delta if change > 1e-12
            ])) if changed else 0.0,
        ))
        STATE["episode"] += 1

    def record_prediction(self, root_ids, acting, actual, state):
        orig_record(self, root_ids, acting, actual, state)
        row = STATE["row"]
        STATE["row"] += 1
        for node_id in root_ids:
            # The original call only reads the cells, so re-reading gives
            # the same pre-update prediction the learner compared against.
            prediction, count = self._feature_cell_read(node_id, int(acting))
            PRED_TRACE.append(dict(
                phase=STATE["phase"],
                row=row,
                node=node_id,
                prediction=float(prediction),
                actual=float(actual),
                count=int(count),
            ))

    NodeRuntime.learn_episode = learn_episode
    NodeRuntime._record_prediction = record_prediction
    return orig_learn, orig_record


def quarter_means(values):
    """Mean of each quarter of a sequence (in order)."""
    if not values:
        return []
    values = np.asarray(values, dtype=float)
    edges = np.linspace(0, len(values), 5).astype(int)
    return [
        float(np.mean(values[edges[i]:edges[i + 1]]))
        if edges[i + 1] > edges[i] else 0.0
        for i in range(4)
    ]


def phase_pred_trace(phase, total_rows):
    """Per-phase error-correction statistics from the recorded predictions."""
    records = [item for item in PRED_TRACE if item["phase"] == phase]
    if not records:
        return dict(pairs=0, rows_covered=0, share_rows=0.0)
    rows_covered = len({item["row"] for item in records})
    gaps = [
        abs(item["prediction"] - item["actual"]) for item in records
    ]
    gaps_normal = [
        gap for item, gap in zip(records, gaps) if item["actual"] < 0.5
    ]
    gaps_pain = [
        gap for item, gap in zip(records, gaps) if item["actual"] >= 0.5
    ]
    order = sorted(records, key=lambda item: item["row"])
    return dict(
        pairs=len(records),
        rows_covered=rows_covered,
        share_rows=(
            float(rows_covered) / float(total_rows) if total_rows else 0.0
        ),
        pain_pairs=sum(1 for item in records if item["actual"] >= 0.5),
        mean_gap=float(np.mean(gaps)),
        mean_gap_normal=(
            float(np.mean(gaps_normal)) if gaps_normal else 0.0
        ),
        mean_gap_pain=(
            float(np.mean(gaps_pain)) if gaps_pain else 0.0
        ),
        surprise_pairs=sum(1 for gap in gaps if gap > 0.02),
        gap_quarters=quarter_means(gaps),
    )


def phase_episode_trace(phase):
    """Per-phase value-correction statistics from the episode records."""
    records = [item for item in EPISODE_TRACE if item["phase"] == phase]
    if not records:
        return dict(episodes=0)
    order = [item["episode"] for item in records]
    records = [item for _, item in sorted(zip(order, records))]
    return dict(
        episodes=len(records),
        mean_frames=float(np.mean([item["frames"] for item in records])),
        mean_pain=float(np.mean([item["pain"] for item in records])),
        pain_episodes=sum(
            1 for item in records if item["pain"] > 0.0
        ),
        changed_share=float(np.mean([
            item["changed"] / item["nodes"]
            for item in records if item["nodes"]
        ])),
        value_delta_quarters=quarter_means(
            [item["value_delta"] for item in records]
        ),
        dopamine_delta_quarters=quarter_means(
            [item["dopamine_delta"] for item in records]
        ),
    )


def snapshot(agent, world, phase, total_rows, args, probe_rows):
    """Everything worth reading off one freshly trained agent."""
    structure = agent.snapshot()
    scale = agent.scales[:agent.obs_dim]

    # Fixed probe stream (same states at every phase, same world seed).
    observations = np.asarray(
        [row["obs"] for row in probe_rows], dtype=float
    ) / scale
    pending = np.asarray(
        [
            np.asarray(row["history"], dtype=float)[1:agent.lag + 1][::-1]
            for row in probe_rows
        ],
        dtype=float,
    )
    encoded = np.concatenate([observations, pending], axis=1) / agent.scales
    mask_table = agent._feature_mask_table(encoded[:, :agent.obs_dim])

    # Feature extraction: which roots exist and how often they fire.
    any_active = np.zeros(len(probe_rows), dtype=bool)
    activation = []
    index_of = {node_id: i for i, node_id in enumerate(agent._feature_all)}
    for node_id in agent.feature_ids:
        feature = agent._feature_objects.get(node_id)
        position = index_of.get(node_id)
        if feature is None or position is None:
            continue
        mask = np.asarray(mask_table[position], dtype=bool)
        rate = float(np.mean(mask))
        any_active |= mask
        cell_counts = [
            int(entry[0]) for entry in
            agent.feature_action_stats.get(node_id, {}).values()
        ]
        activation.append(dict(
            node=node_id,
            label=feature.label(),
            rate=rate,
            cell_observations=int(sum(cell_counts)),
        ))

    # Objective reads at the probe states.
    risk, excess = agent.pain_graded_batch(encoded)
    value, support = agent.feature_value_batch(
        encoded, mask_table, return_support=True
    )
    phi = _curiosity_potential(
        agent.evidence_batch(encoded, mask_table)
    )
    pain_read, dopamine_read = agent.valence_state_batch(encoded)

    # Realized outcomes on the probe stream.
    pain_actual = np.asarray(
        [float(row["signals"][1]) for row in probe_rows], dtype=float
    )
    gamma = float(agent.gamma)
    returns = np.zeros(len(probe_rows), dtype=float)
    by_episode = defaultdict(list)
    for index, row in enumerate(probe_rows):
        by_episode[int(row["episode"])].append(index)
    for indices in by_episode.values():
        running = 0.0
        for index in reversed(indices):
            running = (
                float(probe_rows[index]["signals"][0]) + gamma * running
            )
            returns[index] = running

    finite = np.isfinite(risk)
    pain_mask = pain_actual > 0.5
    reads = dict(
        risk_mean=float(np.nanmean(risk[finite])),
        risk_pain_rows=(
            float(np.nanmean(risk[pain_mask & finite]))
            if pain_mask.any() else 0.0
        ),
        risk_safe_rows=float(np.nanmean(risk[(~pain_mask) & finite])),
        risk_spearman=float(_spearman(
            risk[finite], pain_actual[finite]
        )),
        value_mean=float(np.nanmean(value[np.isfinite(value)])),
        value_spearman=float(_spearman(
            value[np.isfinite(value)], returns[np.isfinite(value)]
        )),
        value_support_mean=float(np.nanmean(support)),
        phi_mean=float(np.nanmean(phi)),
        phi_p90=float(np.nanpercentile(phi, 90)),
        dopamine_read_mean=float(np.nanmean(dopamine_read)),
        pain_actual_mean=float(np.mean(pain_actual)),
    )

    # Prediction calibration: the readout's own ledger.
    report = agent.prediction_report()
    cells = report["cells"]
    gaps = [abs(cell["gap"]) for cell in cells]
    weights = [cell["n"] for cell in cells]
    prediction = dict(
        cells=len(cells),
        observations=int(sum(weights)),
        mean_abs_gap=float(np.mean(gaps)) if gaps else 0.0,
        weighted_abs_gap=(
            float(np.average(gaps, weights=weights)) if gaps else 0.0
        ),
        miscalibrated=int(report["miscalibrated"]),
        surprise_buffer=int(report["buffer_size"]),
        top_gaps=[
            dict(label=cell["feature"], action=int(cell["action"]),
                 n=int(cell["n"]), prediction=float(cell["pred"]),
                 actual=float(cell["actual"]), gap=float(cell["gap"]))
            for cell in cells[:5]
        ],
    )

    # Experience-channel ledger: did the per-experience updates run at all?
    experience = dict(
        action_stat_cells=int(sum(
            len(stats) for stats in
            agent.feature_action_stats.values()
        )),
        action_stat_observations=int(sum(
            cell[0]
            for stats in agent.feature_action_stats.values()
            for cell in stats.values()
        )),
        value_cells=int(sum(
            len(stats) for stats in agent.feature_value_stats.values()
        )),
        calibration_cells=len(agent.feature_calibration),
        recorded_predictions=sum(
            entry[0] for entry in agent.feature_calibration.values()
        ),
    )

    # Decisions: same states, what the prototype Q layer would pick.
    agent.reset()
    decisions = []
    order = np.argsort(np.where(finite, risk, -np.inf))[::-1]
    picked = list(order[:4]) + list(order[-4:])
    for index in picked:
        if not finite[index]:
            continue
        node_id, scores = agent.decision(
            np.asarray(probe_rows[index]["obs"], dtype=float)
        )
        cost = {action: scores[action]["cost"] for action in scores}
        best = min(cost, key=cost.get)
        decisions.append(dict(
            row=int(index),
            risk=float(risk[index]),
            value=float(value[index]),
            support=float(support[index]),
            phi=float(phi[index]),
            pain_actual=float(pain_actual[index]),
            action=int(best),
            margin=float(
                sorted(cost.values())[1] - sorted(cost.values())[0]
            ) if len(cost) > 1 else 0.0,
            cost={str(action): float(cost[action]) for action in cost},
        ))

    # Behavior: held-out worlds, the unshaped ruler.
    evals = []
    with _curiosity_off(agent):
        for eval_seed in args.eval_seeds:
            frames, actions, _rows = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=args.horizon,
                lag=LAG, step_stride=2,
                env_seed=int(eval_seed) * 10000,
            )
            evals.append(dict(
                seed=int(eval_seed),
                frames=int(frames),
                action_rate=float(np.mean(actions)) if actions else 0.0,
            ))

    # Memory growth and reinforcement.
    memory_nodes = [
        node for node in agent.graph.nodes.values()
        if node.kind == MEMORY
    ]
    counts = [int(node.count) for node in memory_nodes]
    memories = dict(
        count=len(memory_nodes),
        reinforced=sum(1 for count in counts if count > 1),
        mean_count=float(np.mean(counts)) if counts else 0.0,
        max_count=int(max(counts)) if counts else 0,
        mean_abs_value=(
            float(np.mean([abs(node.value) for node in memory_nodes]))
            if memory_nodes else 0.0
        ),
    )

    # Prototype value spread: how far the value channel has differentiated.
    concept_values = [
        node.value for node in agent.graph.nodes.values()
        if node.kind == "concept" and node.count > 1
    ]
    value_spread = dict(
        p50=float(np.percentile(concept_values, 50)),
        p90=float(np.percentile(concept_values, 90)),
        max=float(np.max(concept_values)),
    ) if concept_values else dict(p50=0.0, p90=0.0, max=0.0)

    return dict(
        phase=phase,
        frames=int(total_rows),
        structure=structure,
        feature_weight=float(agent.feature_value_weight),
        rules=[
            dict(label=rule.label(), support=int(rule.support),
                 precision=float(rule.precision),
                 test_support=int(rule.test_support),
                 test_precision=float(rule.test_precision))
            for rule in agent.rules.rules
        ],
        activation=dict(
            any_root_rate=float(np.mean(any_active)),
            per_root=activation,
        ),
        reads=reads,
        prediction=prediction,
        experience=experience,
        pred_trace=phase_pred_trace(phase, total_rows),
        episode_trace=phase_episode_trace(phase),
        memories=memories,
        value_spread=value_spread,
        evals=evals,
        eval_mean=float(np.mean([item["frames"] for item in evals])),
        eval_median=float(np.median([item["frames"] for item in evals])),
        decisions=decisions,
    )


def run(args):
    install_wrappers()
    probe_rows, _counts = collect(
        args.seed, args.probe_frames, coverage=0.1, lag=LAG
    )
    snapshots = []
    orig_train = sn._train

    def traced_train(transitions, **kwargs):
        STATE["phase"] += 1
        STATE["episode"] = 0
        STATE["row"] = 0
        world, agent = orig_train(transitions, **kwargs)
        snapshots.append(snapshot(
            agent, world, STATE["phase"], len(transitions), args, probe_rows
        ))
        return world, agent

    sn._train = traced_train
    try:
        fit(
            seed=args.seed, steps=args.steps, coverage=0.1,
            epochs=args.epochs, max_nodes=args.max_nodes,
            birth_threshold=0.45,
            curiosity_weight=args.curiosity_weight, memory_credit=False,
            dagger_rounds=args.rounds, dagger_episodes=2,
            dagger_cap=args.dagger_cap, dagger_frames=args.dagger_frames,
            dagger_horizon=args.horizon, step_stride=2,
        )
    finally:
        sn._train = orig_train
    return dict(
        seed=int(args.seed),
        rounds=int(args.rounds),
        phases=snapshots,
        episode_trace=EPISODE_TRACE,
    )


def print_summary(report):
    print()
    for phase in report["phases"]:
        structure = phase["structure"]
        activation = phase["activation"]
        reads = phase["reads"]
        prediction = phase["prediction"]
        trace = phase["pred_trace"]
        episodes = phase["episode_trace"]
        print("== phase %d  (%d frames) ==" % (phase["phase"], phase["frames"]))
        print(
            "  structure: prototypes %d nodes %d edges %d memories %d "
            "rules %d feature_nodes %d"
            % (structure["prototypes"], structure["nodes"],
               structure["action_edges"], structure["memories"],
               structure["rules"], structure["feature_nodes"])
        )
        print(
            "  feature extraction: roots %d  any-root coverage %.2f%%"
            % (structure["rules"], 100.0 * activation["any_root_rate"])
        )
        for root in activation["per_root"]:
            print(
                "    %s  stream %.2f%%  cell_obs %d"
                % (root["label"], 100.0 * root["rate"],
                   root["cell_observations"])
            )
        print(
            "  prediction: cells %d  mean|gap| %.3f  miscal %d  surprise %d"
            % (prediction["cells"], prediction["mean_abs_gap"],
               prediction["miscalibrated"], prediction["surprise_buffer"])
        )
        if trace.get("pairs"):
            print(
                "  error trace: rows covered %.0f%%  mean|gap| %.3f "
                "(normal %.3f / pain %.3f)  quarters %s"
                % (100.0 * trace["share_rows"], trace["mean_gap"],
                   trace["mean_gap_normal"], trace["mean_gap_pain"],
                   ["%.3f" % value for value in trace["gap_quarters"]])
            )
        if episodes.get("episodes"):
            quarters = episodes["value_delta_quarters"]
            print(
                "  value correction |dv| of touched nodes: %s  "
                "(episodes %d, pain %d, touched %.0f%%)"
                % (["%.4f" % value for value in quarters],
                   episodes["episodes"], episodes["pain_episodes"],
                   100.0 * episodes["changed_share"])
            )
        print(
            "  reads: risk %.3f (pain rows %.3f / safe %.3f, rho %.2f)  "
            "value %.3f (rho %.2f, support %.1f)  phi %.2f"
            % (reads["risk_mean"], reads["risk_pain_rows"],
               reads["risk_safe_rows"], reads["risk_spearman"],
               reads["value_mean"], reads["value_spearman"],
               reads["value_support_mean"], reads["phi_mean"])
        )
        experience = phase["experience"]
        print(
            "  experience channels: action cells %d (obs %d)  value cells %d"
            "  calibration cells %d (predictions %d)"
            % (experience["action_stat_cells"],
               experience["action_stat_observations"],
               experience["value_cells"], experience["calibration_cells"],
               experience["recorded_predictions"])
        )
        print(
            "  memories: %d (reinforced %d)  eval mean %.1f median %.1f"
            % (phase["memories"]["count"], phase["memories"]["reinforced"],
               phase["eval_mean"], phase["eval_median"])
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1,
                        help="replay passes in the fit path: 1 is the "
                             "current benchmark (features are induced "
                             "after the only pass, so the experience "
                             "channels stay empty); 2 gives the induced "
                             "features one training pass - an "
                             "observation arm, not a benchmark change")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--dagger-frames", type=int, default=5000)
    parser.add_argument("--dagger-cap", type=int, default=5000)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--probe-frames", type=int, default=600)
    parser.add_argument("--curiosity-weight", type=float, default=0.5)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    args.eval_seeds = parse_seed_spec(args.eval_seeds)
    report = run(args)
    print_summary(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
