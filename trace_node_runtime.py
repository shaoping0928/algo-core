"""Trace which stage of the node runtime is weak, on the deployed path.

All probes run against the states the planner actually visits:

  1. world model: predicted next observation vs real next observation
  2. node value: learned pain value on terminal / near-pipe states
  3. coverage: how far the planner states are from any trained prototype,
     and whether the training data contains both actions nearby
  4. decision: chosen plan vs fixed all-zero / all-one plans, scored with
     the same objective as the search (`plan_score`)
  5. precursor risk: does the risk readout fire on the states that precede
     a death within k steps (the metric the terminal AUC replaces)
  6. delayed-action identification: which history slot the world model
     reacts to at the planner states

The gap-direction check is a diagnostic only; it is never used to train the
node runtime or the world model.  The task constants below exist only so the
reports are readable; the learner's task-freedom is enforced by a guard test.
"""
import argparse

import numpy as np

from latent_world import latent
from flappy import GAP
from substrate_generic import collect, history_features
from substrate_nodes import fit, plan_score, policy_result
from substrate_value import binary_auc

LAG = 3
OBS_NAMES = ("y", "vy", "dx1", "gap1", "dx2", "gap2")

# --- task constants, DIAGNOSTICS ONLY --------------------------------------
# The measurement layer may name the world's geometry (collision band,
# boundary limits, near-pipe window) so its reports are readable.  Nothing
# here is imported by or fed into the learner.
DIAG_COLLISION_WINDOW = 0.06   # world's |pipe_x - bird_x| collision band
DIAG_BOUNDARY = (0.0, 1.0)     # world's y limits
DIAG_NEAR_WINDOW = 0.2         # "near pipe" band for the support metrics


def _node_state(agent, observation, pending):
    return np.concatenate(
        [np.asarray(observation, dtype=float), np.asarray(pending, dtype=float)]
    ) / agent.scales


def _death_type(observation, died, reached_cap):
    """Classify the final state; measurement only, never used for learning."""
    if not died or reached_cap:
        return "cap"
    y = float(observation[0]) + 0.5
    if abs(float(observation[2])) < DIAG_COLLISION_WINDOW \
            and abs(float(observation[3])) > GAP / 2:
        return "pipe"
    low, high = DIAG_BOUNDARY
    if y < low or y > high:
        return "boundary"
    return "other"


def trace(seed, steps=5000, coverage=0.1, cap=1500, horizon=10,
          epochs=10, max_nodes=5000, birth_threshold=0.45,
          components=(), component_min_count=8, max_components=128,
          component_gating=False, pain_horizon=None, voi_fraction=0.0):
    world, agent = fit(
        seed, steps=steps, coverage=coverage, max_nodes=max_nodes,
        birth_threshold=birth_threshold, epochs=epochs,
        component_kinds=components,
        component_min_count=component_min_count,
        max_components=max_components,
        component_gating=component_gating,
        pain_horizon=pain_horizon,
        voi_fraction=voi_fraction,
    )
    # The same deterministic transitions `fit` trained on; used only to
    # measure whether both actions exist near the planner states.
    train_transitions, _ = collect(seed, steps, coverage=coverage, lag=LAG)
    obs_dim = len(agent.scales) - agent.lag
    train_states = np.asarray(
        [row["obs"] for row in train_transitions], dtype=float
    ) / agent.scales[:obs_dim]
    train_actions = np.asarray(
        [int(row["history"][0]) for row in train_transitions]
    )
    knn = 8

    env = latent[LAG](seed)
    agent.reset()
    prefix = np.zeros(LAG, dtype=float)
    errors = []
    values = []
    labels = []
    near = []
    distances = []
    visit_counts = []
    immediate_values = []
    nearest_values = []
    directions = []
    edge_support = []
    near_edge_support = []
    gated_hits = 0
    gated_checks = 0
    decision_risks = []
    feature_risks = []
    knn_rows = []
    near_knn_rows = []
    lag_slots = []
    lag_separations = []
    plan_rows = []
    t = 0
    while not env.dead and t < cap:
        observation = np.asarray(env.obs(), dtype=float)
        result = policy_result(env, world, agent, prefix, horizon, lag=LAG)
        action = int(result.actions[0])

        pending_before = prefix[::-1].copy()
        state = _node_state(agent, observation, pending_before)
        obs_state = state[:obs_dim].reshape(1, -1)
        node_id, nearest_distance = agent._nearest(state)
        signature = agent._signature(node_id)
        scores = agent._score_actions(
            node_id, state, signature=signature
        )
        counts = [scores[a]["count"] for a in (0, 1)]
        chosen_count = float(scores[action]["count"])
        other_count = float(scores[1 - action]["count"])
        edge_support.append((chosen_count, other_count,
                             counts[0] > 0 and counts[1] > 0))
        dx = float(observation[2])
        gap = float(observation[3])
        is_near = abs(dx) < DIAG_NEAR_WINDOW \
            and dx > -0.5 * DIAG_NEAR_WINDOW
        if is_near:
            near_edge_support.append((chosen_count, other_count,
                                      counts[0] > 0 and counts[1] > 0))
        if is_near and abs(gap) > 0.02:
            expected = 1 if gap > 0.02 else 0
            directions.append(int(action == expected))
        if signature:
            gated_checks += 1
            if any((node_id, signature, a) in agent.gated_edges
                   for a in (0, 1)):
                gated_hits += 1

        # Planner-state coverage and risk activation.
        decision_risks.append(
            float(agent.pain_signal_batch(state.reshape(1, -1))[0])
        )
        feature_risks.append(
            float(agent._feature_risk_batch(state.reshape(1, -1))[0])
        )
        distances.append(float(nearest_distance))
        if len(train_states) >= knn:
            d2 = ((train_states - obs_state[0]) ** 2).sum(axis=1)
            nearest = np.argpartition(d2, knn - 1)[:knn]
            local = np.bincount(train_actions[nearest], minlength=2)
            knn_rows.append((int(local[0]), int(local[1])))
            if is_near:
                near_knn_rows.append((int(local[0]), int(local[1])))

        # Which history slot moves the prediction at this state?
        if t % 5 == 0:
            base_history = np.zeros(LAG + 1, dtype=float)
            base = world.predict(
                observation.reshape(1, -1), base_history.reshape(1, -1)
            )[0]
            deltas = np.zeros(LAG + 1, dtype=float)
            for slot in range(LAG + 1):
                flipped = base_history.copy()
                flipped[slot] = 1.0
                moved = world.predict(
                    observation.reshape(1, -1), flipped.reshape(1, -1)
                )[0]
                deltas[slot] = float(
                    np.linalg.norm((moved - base) / agent.scales[:obs_dim])
                )
            top = int(np.argmax(deltas))
            lag_slots.append(top)
            lag_separations.append(
                float(deltas[top] / (np.median(np.delete(deltas, top)) + 1e-12))
            )

        # Same-scale comparison against the fixed all-zero / all-one plans.
        score = result.score
        chosen_risk = float(score[0] if np.ndim(score) else score)
        zero_risk = plan_score(
            observation, world, agent, prefix, np.zeros(horizon, dtype=float),
            horizon, lag=LAG,
        )[0]
        one_risk = plan_score(
            observation, world, agent, prefix, np.ones(horizon, dtype=float),
            horizon, lag=LAG,
        )[0]
        plan_rows.append((chosen_risk, zero_risk, one_risk))

        history = np.concatenate([[float(action)], prefix[:LAG]])[:LAG + 1]
        predicted = world.predict(
            observation.reshape(1, -1), history.reshape(1, -1)
        )[0]
        pending_after = prefix[:LAG - 1][::-1].tolist() + [float(action)]

        env.step(action)
        next_observation = np.asarray(env.obs(), dtype=float)
        errors.append(np.abs(predicted - next_observation))
        next_state = _node_state(agent, next_observation, pending_after)
        values.append(agent.value_state(next_state))
        nearest_node, _ = agent._nearest(next_state)
        visit_counts.append(
            agent.graph.nodes[nearest_node].count
            if nearest_node is not None else 0
        )
        immediate_values.append(
            agent.graph.nodes[nearest_node].immediate_pain
            if nearest_node is not None else 0.0
        )
        nearest_values.append(
            agent.graph.nodes[nearest_node].value
            if nearest_node is not None else 0.0
        )
        labels.append(bool(env.dead))
        near.append(abs(float(next_observation[2])) < DIAG_NEAR_WINDOW
                    and float(next_observation[2]) > -0.5 * DIAG_NEAR_WINDOW)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        t += 1

    errors = np.asarray(errors)
    values = np.asarray(values)
    labels = np.asarray(labels)
    near = np.asarray(near)
    distances = np.asarray(distances)
    visit_counts = np.asarray(visit_counts)
    immediate_values = np.asarray(immediate_values)
    nearest_values = np.asarray(nearest_values)
    edge_support = np.asarray(edge_support)
    near_edge_support = (
        np.asarray(near_edge_support) if near_edge_support
        else np.zeros((0, 3))
    )
    decision_risks = np.asarray(decision_risks)
    feature_risks = np.asarray(feature_risks)
    knn_rows = np.asarray(knn_rows, dtype=int) if knn_rows else np.zeros((0, 2), dtype=int)
    near_knn_rows = (
        np.asarray(near_knn_rows, dtype=int) if near_knn_rows
        else np.zeros((0, 2), dtype=int)
    )
    plan_rows = np.asarray(plan_rows)
    died = bool(env.dead) and env.t < cap
    death = _death_type(next_observation if t else observation, died, t >= cap)

    print("seed=%d frames=%d steps=%d death=%s"
          % (seed, env.t, len(values), death))
    print("\n[1] world model one-step error")
    for index, name in enumerate(OBS_NAMES):
        print("  %-4s mean %.2e  max %.2e"
              % (name, errors[:, index].mean(), errors[:, index].max()))

    print("\n[2] node value")
    print("  terminal AUC %.3f  near-pipe AUC %.3f  terminal mean %.3f  safe mean %.3f"
          % (
              binary_auc(values, labels),
              binary_auc(values[near], labels[near]) if near.any() else float("nan"),
              values[labels].mean() if labels.any() else float("nan"),
              values[~labels].mean() if (~labels).any() else float("nan"),
          ))
    print("  nearest distance terminal/safe %.2f/%.2f  count terminal/safe %.1f/%.1f"
          % (
              distances[labels].mean() if labels.any() else float("nan"),
              distances[~labels].mean() if (~labels).any() else float("nan"),
              visit_counts[labels].mean() if labels.any() else float("nan"),
              visit_counts[~labels].mean() if (~labels).any() else float("nan"),
          ))
    print("  nearest immediate pain terminal/safe %.3f/%.3f  nearest value %.3f/%.3f"
          % (
              immediate_values[labels].mean() if labels.any() else float("nan"),
              immediate_values[~labels].mean() if (~labels).any() else float("nan"),
              nearest_values[labels].mean() if labels.any() else float("nan"),
              nearest_values[~labels].mean() if (~labels).any() else float("nan"),
          ))
    supported = visit_counts >= 5
    if supported.any() and labels[supported].any() and (~labels[supported]).any():
        print("  in-support (count>=5) AUC %.3f over %d states"
              % (
                  binary_auc(values[supported], labels[supported]),
                  int(supported.sum()),
              ))

    print("\n[3] planner-state coverage")
    ood = distances > birth_threshold
    print("  out of birth radius: %.3f  median distance %.2f  max %.2f"
          % (float(ood.mean()), float(np.median(distances)), float(distances.max())))
    print("  risk active (>0): %.3f  (>0.05): %.3f"
          % (float((decision_risks > 0).mean()),
             float((decision_risks > 0.05).mean())))
    print("  feature risk active (>0): %.3f" % float((feature_risks > 0).mean()))
    if len(knn_rows):
        both = (knn_rows[:, 0] > 0) & (knn_rows[:, 1] > 0)
        print("  training data both actions in %d nearest: %.3f  mean min-count %.2f"
              % (knn, float(both.mean()), float(knn_rows.min(axis=1).mean())))
        if len(near_knn_rows):
            near_both = (
                (near_knn_rows[:, 0] > 0) & (near_knn_rows[:, 1] > 0)
            )
            print("  near-pipe both actions in %d nearest: %.3f"
                  % (knn, float(near_both.mean())))

    print("\n[4] action edges and plans")
    print("  edge both actions supported: %.3f  chosen count mean %.1f  other count mean %.1f"
          % (
              float(edge_support[:, 2].mean()),
              float(edge_support[:, 0].mean()),
              float(edge_support[:, 1].mean()),
          ))
    if len(near_edge_support):
        print("  near-pipe edge both actions: %.3f  mean chosen/other %s"
              % (
                  float(near_edge_support[:, 2].mean()),
                  np.round(near_edge_support[:, :2].mean(axis=0), 1).tolist(),
              ))
    print("  gated signature available: %d/%d" % (gated_hits, gated_checks))
    print("  chosen plan risk <= zero plan: %.3f  <= one plan: %.3f"
          % (
              float(np.mean(plan_rows[:, 0] <= plan_rows[:, 1] + 1e-9)),
              float(np.mean(plan_rows[:, 0] <= plan_rows[:, 2] + 1e-9)),
          ))
    print("  mean plan risk chosen/zero/one %.4f/%.4f/%.4f"
          % tuple(plan_rows.mean(axis=0)))

    print("\n[5] precursor risk (death within k decision steps)")
    if died:
        total = len(decision_risks)
        for k in range(1, LAG + 2):
            precursor = np.zeros(total, dtype=bool)
            precursor[max(0, total - k):] = True
            safe = ~precursor
            if not safe.any():
                continue
            print(
                "  k=%d  risk AUC %.3f  feature AUC %.3f  "
                "precursor risk %.3f vs safe %.3f"
                % (
                    k,
                    binary_auc(decision_risks, precursor),
                    binary_auc(feature_risks, precursor),
                    decision_risks[precursor].mean(),
                    decision_risks[safe].mean(),
                )
            )
    else:
        print("  no death within the cap")

    print("\n[6] delayed-action identification")
    if lag_slots:
        counts = np.bincount(lag_slots, minlength=LAG + 1)
        print("  dominant history slot counts %s (slot %d = the lag-applied action)"
              % (counts.tolist(), LAG))
        print("  mean separation (max/median other) %.2f"
              % float(np.mean(lag_separations)))

    print("\n[7] decision direction near pipe")
    directions = np.asarray(directions, dtype=float)
    if len(directions):
        print("  directional accuracy %.3f over %d events"
              % (float(directions.mean()), len(directions)))
    else:
        print("  no near-pipe decisions")

    print("\n[8] prediction calibration (readout vs actual)")
    report = agent.prediction_report()
    print("  cells %d  miscalibrated %d  surprise buffer %d"
          % (len(report["cells"]), report["miscalibrated"],
             report["buffer_size"]))
    for entry in report["cells"]:
        if entry["miscalibrated"]:
            print("    a=%d n=%4d pred=%.3f actual=%.3f brier=%.3f  %s"
                  % (entry["action"], entry["n"], entry["pred"],
                     entry["actual"], entry["brier"],
                     entry["feature"][:46]))
    for item in report["buffer"][:5]:
        print("    surprise score=%.2f a=%d pred=%.2f actual=%.0f conf=%.2f"
              % (item["score"], item["acting"], item["prediction"],
                 item["actual"], item["confidence"]))
    return dict(
        frames=env.t,
        death=death,
        world_error=errors.mean(axis=0).tolist(),
        terminal_auc=binary_auc(values, labels),
        near_auc=(
            binary_auc(values[near], labels[near]) if near.any()
            else float("nan")
        ),
        ood_frac=float(ood.mean()),
        risk_active_frac=float((decision_risks > 0).mean()),
        feature_active_frac=float((feature_risks > 0).mean()),
        knn_both=float(
            (((knn_rows[:, 0] > 0) & (knn_rows[:, 1] > 0)).mean())
            if len(knn_rows) else float("nan")
        ),
        precursor_auc=(
            float(binary_auc(decision_risks,
                             np.arange(len(decision_risks))
                             >= len(decision_risks) - (LAG + 1)))
            if died else float("nan")
        ),
        directional_accuracy=(
            float(directions.mean()) if len(directions) else float("nan")
        ),
        support_both=float(edge_support[:, 2].mean()),
        lag_slot=(
            int(np.bincount(lag_slots, minlength=LAG + 1).argmax())
            if lag_slots else -1
        ),
        prediction_miscalibrated=int(report["miscalibrated"]),
        surprise_buffer=int(report["buffer_size"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--birth-threshold", type=float, default=0.45)
    parser.add_argument("--components", default="")
    parser.add_argument("--component-min-count", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--component-gating", action="store_true")
    parser.add_argument("--pain-horizon", type=int, default=0,
                        help="precursor label window; 0 uses the default")
    parser.add_argument("--voi-fraction", type=float, default=0.0,
                        help="counterfactual collection budget fraction")
    parser.add_argument("--fast", action="store_true",
                        help="small development preset matching substrate_nodes")
    args = parser.parse_args()
    if args.fast:
        args.steps = 1500
        args.epochs = 3
        args.max_nodes = 1000
        args.cap = 400
        args.horizon = 8
    components = tuple(
        kind.strip() for kind in args.components.split(",") if kind.strip()
    )
    trace(
        args.seed, steps=args.steps, coverage=args.coverage, cap=args.cap,
        horizon=args.horizon, epochs=args.epochs, max_nodes=args.max_nodes,
        birth_threshold=args.birth_threshold, components=components,
        component_min_count=args.component_min_count,
        max_components=args.max_components,
        component_gating=args.component_gating,
        pain_horizon=(args.pain_horizon or None),
        voi_fraction=args.voi_fraction,
    )


if __name__ == "__main__":
    main()
