"""Flappy adapter for generic search and world-provided valence feedback.

This adapter supplies state encoding, a learned one-step transition predictor, and an evaluator
to `trajectory_search.exhaustive_search`. By default, the value head learns from the world's
dopamine/pain channels; `--task-baseline` opts into the hand-specified gap/collision score.
The search core itself has no Flappy knowledge.

Sandbox predictions are not fed to the world model; only real transitions update it.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from latent_world import BIRD_X, GAP, SEEDS, SPEED, latent
from predict_layer import LayerPair
from substrate_value import (
    InputNovelty,
    ReturnReplay,
    TerminalRiskHead,
    TransitionReplay,
    WorldFeedbackValueHead,
    discounted_returns,
    fit_terminal_risk,
    fit_value_head_returns,
    planned_substrate_features,
    substrate_features,
    value_probe_metrics,
)
from trajectory_search import beam_search, exhaustive_search

LAG = 3
H = 14
LONG_H = 78
BEAM_WIDTH = 256
PRACTICE = 5000
VALUE_UPDATES = 3000
VALUE_BATCH = 128
TERMINAL_FRACTION = 0.15
NEAR_PIPE_FRACTION = 0.0
RISK_INTERACTION_DIM = 0
RISK_UPDATES = 8000
ACTION_SPACE = (0, 1)
ACTION_DIM = LAG + 1
RISK_ACTION_DIM = 0
DEFAULT_JOBS = min(8, os.cpu_count() or 1)


def action_history(sequences, steps):
    """Latest action, then previous actions; shape (n, ACTION_DIM)."""
    sequences = np.asarray(sequences, dtype=float)
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape (candidates, horizon)")
    result = np.zeros((len(sequences), ACTION_DIM), dtype=float)
    for offset in range(ACTION_DIM):
        index = steps - 1 - offset
        if index >= 0:
            result[:, offset] = sequences[:, index]
    return result


def nearest_pipe(env):
    """Signed horizontal distance and gap offset of the nearest pipe."""
    px, gap = min(env.pipes, key=lambda pipe: abs(pipe[0] - BIRD_X))
    return float(px - BIRD_X), float(gap - env.y)


def observe_transition(
    lp, feedback_head, novelty, delta, event, observation, next_observation,
    world_feedback, terminal=False, learn_value=True, learn_novelty=True,
    action_history_vector=None,
):
    """Learn only from the world feedback attached to a real transition."""
    novelty_score = (
        novelty.observe(lp.hist) if learn_novelty else novelty.score(lp.hist)
    )
    features = substrate_features(lp, novelty_score=novelty_score)
    feedback_features = np.concatenate(
        [features, np.asarray(observation, dtype=float).reshape(-1)]
    )
    error = float(delta - lp.predict())
    lp.observe(delta, event)
    next_features = substrate_features(
        lp, novelty_score=novelty.score(lp.hist)
    )
    next_feedback_features = np.concatenate(
        [next_features, np.asarray(next_observation, dtype=float).reshape(-1)]
    )
    signals = np.asarray(
        [world_feedback["dopamine"], world_feedback["pain"]], dtype=float
    )
    if action_history_vector is None:
        risk_features = next_feedback_features
    else:
        risk_features = np.concatenate(
            [next_feedback_features, np.asarray(action_history_vector, dtype=float)]
        )
    transition = dict(
        features=feedback_features,
        signals=signals,
        next_features=next_feedback_features,
        risk_features=risk_features,
        terminal=bool(terminal),
    )
    if learn_value:
        feedback_head.update(
            feedback_features, signals, next_feedback_features, terminal=terminal
        )
    return error, transition


def practice(seed, steps=PRACTICE, value_updates=VALUE_UPDATES,
             risk_updates=RISK_UPDATES, terminal_fraction=TERMINAL_FRACTION,
             near_pipe_fraction=NEAR_PIPE_FRACTION,
             risk_interaction_dim=RISK_INTERACTION_DIM,
             risk_action_dim=RISK_ACTION_DIM):
    lp = LayerPair(k_max=8)
    novelty = InputNovelty(feature_dim=len(lp.hist))
    sample_env = latent[LAG](seed * 10000)
    feature_dim = len(substrate_features(lp)) + len(sample_env.obs())
    risk_feature_dim = feature_dim + risk_action_dim
    feedback_head = WorldFeedbackValueHead(feature_dim=feature_dim, discount=0.99)
    risk_head = TerminalRiskHead(
        feature_dim=risk_feature_dim, interaction_dim=risk_interaction_dim
    )
    done = ep = 0
    probe = []
    replay = TransitionReplay(capacity=max(steps * 2, 1000), seed=seed)
    return_replay = ReturnReplay(capacity=max(steps * 2, 1000), seed=seed + 1)
    coverage = dict(
        episodes=0, frames=0, pipe_deaths=0, boundary_deaths=0,
        cap_deaths=0, pipe_window_steps=0, high_y_steps=0, low_y_steps=0,
        episode_lengths=[],
    )
    while done < steps:
        env = latent[LAG](seed * 10000 + ep)
        rng = np.random.RandomState(seed * 10000 + ep)
        lp.hist = [0.0] * 8
        holdout = (ep % 5 == 4)
        episode_transitions = []
        while not env.dead and done < steps:
            gains = np.abs(lp.h)
            total_gain = float(gains.sum())
            confidence = float(gains.max() / total_gain) if total_gain > 1e-9 else 0.0
            if total_gain > 1e-9 and (lp.t >= 20 or confidence >= 0.25):
                lag_index = int(np.argmax(gains))
                gain = float(lp.h[lag_index])
                center_rate = float(np.clip(-lp.b / gain, 0.0, 1.0))
            else:
                # A generic low flap-rate prior keeps early exploratory episodes alive
                # long enough to reach later world events; it is not a goal or reward.
                center_rate = 0.1
            # Explore around the model-implied neutral action rate; uncertainty widens the
            # exploration band. No task target or collision rule enters data collection.
            explore = 0.03 + 0.10 * (1.0 - confidence)
            action_rate = float(np.clip(
                center_rate + rng.uniform(-explore, explore), 0.0, 1.0
            ))
            action = int(rng.rand() < action_rate)
            vy0 = env.vy
            y0, t0 = env.y, env.t
            pipe_dx, gap_dy = nearest_pipe(env)
            if abs(pipe_dx) < 0.1:
                coverage["pipe_window_steps"] += 1
            if y0 > 0.9:
                coverage["high_y_steps"] += 1
            elif y0 < 0.1:
                coverage["low_y_steps"] += 1
            observation = env.obs()
            env.step(action)
            next_observation = env.obs()
            world_feedback = env.feedback()
            history_vector = None
            if risk_action_dim:
                history_vector = np.concatenate(
                    [[float(action)], np.asarray(lp.hist[:LAG], dtype=float)]
                )[:risk_action_dim]
            _error, transition = observe_transition(
                lp, feedback_head, novelty, env.vy - vy0, float(action),
                observation, next_observation, world_feedback, env.dead,
                learn_value=False, learn_novelty=not holdout,
                action_history_vector=history_vector,
            )
            risk_features = (
                transition["risk_features"] if risk_action_dim
                else transition["features"]
            )
            if holdout:
                probe.append(dict(
                    features=transition["features"],
                    risk_features=risk_features,
                    terminal=transition["terminal"],
                    y=y0, vy=vy0, t=t0, pipe_dx=pipe_dx, gap_dy=gap_dy,
                    episode=ep,
                ))
            else:
                replay.add(
                    risk_features, transition["signals"],
                    risk_features, transition["terminal"],
                    metadata=dict(pipe_dx=pipe_dx, gap_dy=gap_dy, y=y0, t=t0),
                )
                episode_transitions.append(transition)
            done += 1
        if not holdout and episode_transitions:
            returns = discounted_returns(
                episode_transitions, feedback_head.discount
            )
            for transition, target in zip(episode_transitions, returns):
                return_replay.add(transition["features"], target)
        coverage["episodes"] += 1
        coverage["frames"] += env.t
        coverage["episode_lengths"].append(env.t)
        if env.dead:
            if env.t >= env.max_t:
                coverage["cap_deaths"] += 1
            elif env.y < 0.0 or env.y > 1.0:
                coverage["boundary_deaths"] += 1
            else:
                coverage["pipe_deaths"] += 1
        ep += 1
    fit_value_head_returns(
        feedback_head, return_replay, value_updates, batch_size=VALUE_BATCH
    )
    fit_terminal_risk(
        risk_head, replay, risk_updates, batch_size=VALUE_BATCH,
        terminal_fraction=terminal_fraction,
        near_pipe_fraction=near_pipe_fraction,
    )
    coverage["return_samples"] = len(return_replay)
    j = int(np.argmax(np.abs(lp.h)))
    return (lp, j + 1, float(lp.h[j]), feedback_head, risk_head, novelty, probe,
            replay, coverage)


def target_gap(env):
    # Select the pipe closest to reaching the bird during this short horizon.
    px_target = BIRD_X + H * SPEED
    _, gc = min(env.pipes, key=lambda p: abs(p[0] - px_target))
    return gc


def score_text(score):
    if isinstance(score, tuple):
        return "[" + ", ".join("%+.3e" % value for value in score) + "]"
    return "%+.3e" % score


def rollout_candidates(
    env, b, f, lag, lp, feedback_head, risk_head, novelty, value_mode,
    engine, horizon, beam_width, risk_aggregate="terminal",
    risk_action_dim=RISK_ACTION_DIM,
):
    """Bind the learned transition and selected value evaluator to the generic search core."""
    pending = np.asarray(env.pending()[:lag], dtype=float)
    if len(pending) < lag:
        pending = np.pad(pending, (0, lag - len(pending)))
    initial_state = np.asarray([env.y, env.vy], dtype=float)
    gc = target_gap(env) if value_mode == "task" else None

    def predict_batch(states, sequences, step):
        y, vy = states[:, 0], states[:, 1]
        if step < lag:
            impulse = np.full(len(states), pending[step], dtype=float)
        else:
            impulse = sequences[:, step - lag]
        next_vy = vy + b + f * impulse
        next_y = y + next_vy
        return np.column_stack([next_y, next_vy])

    def feedback_features(states, sequences, steps):
        features = planned_substrate_features(
            lp, sequences[:, :steps], novelty
        )
        pipe1, pipe2 = env.pipes[0], env.pipes[-1]
        pipe1_x = pipe1[0] - steps * SPEED
        pipe2_x = pipe2[0] - steps * SPEED
        ys, vys = states[:, 0], states[:, 1]
        observations = np.column_stack(
            [
                ys - 0.5,
                vys * 10.0,
                np.full(len(states), pipe1_x - BIRD_X),
                pipe1[1] - ys,
                np.full(len(states), pipe2_x - BIRD_X),
                pipe2[1] - ys,
            ]
        )
        return np.concatenate([features, observations], axis=1)

    def risk_features(states, sequences, steps):
        base = feedback_features(states, sequences, steps)
        if not risk_action_dim:
            return base
        return np.concatenate(
            [base, action_history(sequences, steps)[:, :risk_action_dim]],
            axis=1,
        )

    def evaluate_step(states, sequences, step):
        if value_mode == "feedback":
            features = risk_features(states, sequences, step + 1)
            if risk_aggregate == "crossing":
                steps = step + 1
                crossing = any(
                    abs(px - steps * SPEED - BIRD_X) < 0.06
                    for px, _gap in env.pipes
                )
            else:
                crossing = True
            if engine == "beam" or risk_aggregate in ("sum", "max", "crossing"):
                if crossing:
                    # Query the learned pain/risk head at the predicted pipe-crossing
                    # moment, not at every harmless step or only at the endpoint.
                    risk = risk_head.probability(features)
                    zeros = np.zeros(len(states))
                    scores = np.column_stack([risk, zeros, zeros])
                else:
                    scores = np.zeros((len(states), 3))
            else:
                scores = np.zeros((len(states), 3))
            return scores, np.isfinite(features).all(axis=1)

        ys = states[:, 0]
        feasible = (ys >= 0.0) & (ys <= 1.0)
        # Task-specific pipe constraints belong to the adapter, not the search core.
        for px, pipe_gc in env.pipes:
            pipe_x = px - (step + 1) * SPEED
            if abs(pipe_x - BIRD_X) < 0.06:
                feasible &= np.abs(ys - pipe_gc) <= GAP / 2.0
        # Optional task-baseline scorer; not used by the world-feedback mode.
        cost = (ys - gc) ** 2
        return cost, feasible

    def evaluate_terminal(states, sequences):
        features = feedback_features(states, sequences, horizon)
        risk = risk_head.probability(risk_features(states, sequences, horizon))
        values = feedback_head.value(features)
        # Immediate pain first, then the longer-horizon pain return, then dopamine.
        scores = np.column_stack([risk, values[:, 1], -values[:, 0]])
        return scores, np.isfinite(features).all(axis=1)

    combine = (
        ("max", "sum", "sum")
        if risk_aggregate in ("max", "crossing") else "sum"
    )
    if engine == "beam":
        result = beam_search(
            initial_state=initial_state,
            action_space=ACTION_SPACE,
            horizon=horizon,
            transition_batch=predict_batch,
            evaluate_step=evaluate_step,
            evaluate_terminal=evaluate_terminal,
            beam_width=beam_width,
            score_combine=combine,
        )
    else:
        result = exhaustive_search(
            initial_state=initial_state,
            action_space=ACTION_SPACE,
            horizon=horizon,
            transition_batch=predict_batch,
            evaluate_step=evaluate_step,
            evaluate_terminal=evaluate_terminal if value_mode == "feedback" else None,
            batch_size=len(ACTION_SPACE) ** horizon,
            score_combine=combine,
        )
    return result, gc


def run(seed, value_mode="feedback", engine="exhaustive", horizon=H,
        beam_width=BEAM_WIDTH, practice_steps=PRACTICE,
        value_updates=VALUE_UPDATES, risk_updates=RISK_UPDATES,
        terminal_fraction=TERMINAL_FRACTION,
        near_pipe_fraction=NEAR_PIPE_FRACTION,
        risk_interaction_dim=RISK_INTERACTION_DIM,
        risk_action_dim=RISK_ACTION_DIM,
        risk_aggregate="terminal", show=False):
    if value_mode == "task" and engine != "exhaustive":
        raise ValueError("the task baseline uses exhaustive search")
    lp, lag, f, feedback_head, risk_head, novelty, probe, replay, coverage = practice(
        seed, steps=practice_steps, value_updates=value_updates,
        risk_updates=risk_updates, terminal_fraction=terminal_fraction,
        near_pipe_fraction=near_pipe_fraction,
        risk_interaction_dim=risk_interaction_dim,
        risk_action_dim=risk_action_dim,
    )
    b = lp.b
    env = latent[LAG](seed)
    lp.hist = [0.0] * 8
    t = searches = feasible_total = candidate_total = no_feasible = 0
    max_model_error = 0.0
    selected = []
    search_memory = []

    while not env.dead and t < 6000:
        plan, gc = rollout_candidates(
            env, b, f, lag, lp, feedback_head, risk_head, novelty,
            value_mode, engine, horizon, beam_width,
            risk_aggregate=risk_aggregate, risk_action_dim=risk_action_dim,
        )
        searches += 1
        feasible_total += plan.feasible_count
        candidate_total += plan.candidate_count
        if not plan.feasible:
            no_feasible += 1
        seq = plan.actions
        action = int(seq[0])
        predicted_next_y = float(plan.first_state[0])
        if show and len(selected) < 8:
            selected.append((t, "".join(str(int(x)) for x in seq[:10]),
                             float(plan.final_state[0]), gc, plan.feasible,
                             plan.score))

        pending_before = tuple(int(x) for x in env.pending())
        applied = int(pending_before[0])  # next impulse: oldest action currently pending
        y0, vy0 = env.y, env.vy
        observation = env.obs()
        env.step(action)
        actual_next_y = float(env.y)
        next_observation = env.obs()
        world_feedback = env.feedback()
        observed_delta = env.vy - vy0
        max_model_error = max(max_model_error, abs(observed_delta - (b + f * applied)))
        # Search experience is stored separately from factual transition data.
        # It records the counterfactual plan and then the real one-step verification.
        search_memory.append({
            "t": t,
            "state": (float(y0), float(vy0)),
            "pending": pending_before,
            "candidate_count": plan.candidate_count,
            "feasible_count": plan.feasible_count,
            "chosen_sequence": tuple(int(x) for x in seq),
            "predicted_endpoint": float(plan.final_state[0]),
            "target_gap": float(gc) if gc is not None else None,
            "value_score": plan.score,
            "predicted_next_y": predicted_next_y,
            "actual_next_y": actual_next_y,
            "one_step_error": actual_next_y - predicted_next_y,
        })
        observe_transition(
            lp, feedback_head, novelty, observed_delta, float(action),
            observation, next_observation, world_feedback, env.dead,
            learn_value=False,
        )
        # Refresh the learned parameters from real data only.
        j = int(np.argmax(np.abs(lp.h)))
        lag, f, b = j + 1, float(lp.h[j]), float(lp.b)
        t += 1

    return dict(frames=env.t, lag=lag, f=f, b=b, searches=searches,
                search_engine=engine, search_horizon=horizon,
                feasible_fraction=feasible_total / max(candidate_total, 1),
                no_feasible=no_feasible, model_error=max_model_error, selected=selected,
                search_memory=search_memory, replay_size=len(replay),
                coverage=coverage,
                value_probe=value_probe_metrics(
                    probe, feedback_head, risk_head, gap=GAP
                ))


def report(rows, value_mode):
    seeds = SEEDS[:len(rows)]
    candidate_metric = "finite fraction" if value_mode == "feedback" else "feasible fraction"
    failure_metric = "nonfinite" if value_mode == "feedback" else "no-feasible"
    print(" seed   frames   lag   learned F   %-17s %s steps   max model error"
          % (candidate_metric, failure_metric))
    for seed, result in zip(seeds, rows):
        print(" %4d   %6d   %3d   %.6f        %.3f              %5d             %.2e"
              % (seed, result["frames"], result["lag"], result["f"],
                 result["feasible_fraction"], result["no_feasible"],
                 result["model_error"]))

    selected_validity = "finite" if value_mode == "feedback" else "feasible"
    print("\nselected candidates from seed %d (prefix, predicted endpoint, "
          "value target, %s, score):" % (seeds[0], selected_validity))
    for t, prefix, endpoint, gc, ok, score in rows[0]["selected"]:
        target = "world-feedback" if gc is None else "gap=%.4f" % gc
        print("  t=%3d  %s  y_pred=%.4f  %s  %s=%s  score=%s"
              % (t, prefix, endpoint, target, selected_validity, ok,
                 score_text(score)))

    print("\nQ1  learned lag 3 and F=.020: %s (mean lag %.2f, F %.6f)"
          % ("ok" if all(r["lag"] == LAG and abs(r["f"] - .02) < .002 for r in rows)
             else "MISS", np.mean([r["lag"] for r in rows]),
             np.mean([r["f"] for r in rows])))
    print("Q2  candidate rollouts pass the active numeric/feasibility check: %s "
          "(mean fraction %.3f)"
          % ("ok" if np.mean([r["feasible_fraction"] for r in rows]) > 0 else "MISS",
             np.mean([r["feasible_fraction"] for r in rows])))
    print("Q3  steps with no candidate passing the active check: %d across %d seeds"
          % (sum(r["no_feasible"] for r in rows), len(rows)))
    print("Q4  external survival evaluation: mean %.1f, median %.1f frames; "
          "seed range %d-%d"
          % (np.mean([r["frames"] for r in rows]),
             np.median([r["frames"] for r in rows]),
             min(r["frames"] for r in rows), max(r["frames"] for r in rows)))
    print("    reference only: the hand-derived exact-model rule scored 4887, "
          "but it is not a learned search policy")
    print("Q5  one-step model/world discrepancy: %.2e"
          % max(r["model_error"] for r in rows))
    mem = [item for r in rows for item in r["search_memory"]]
    print("Q6  separate sandbox-memory records: %d; mean |predicted next y - real "
          "next y| %.2e"
          % (len(mem), np.mean([abs(item["one_step_error"]) for item in mem])))
    print("    each record stores state, pending queue, candidate count, feasible "
          "count, selected plan,")
    print("    predicted endpoint, target, and the real one-step verification; only "
          "that real transition")
    print("    updates the world model.")

    print("\nPractice coverage:")
    print(" seed episodes mean_frames pipe_deaths boundary_deaths cap pipe-window "
          "high-y low-y")
    for seed, result in zip(seeds, rows):
        coverage = result["coverage"]
        lengths = coverage["episode_lengths"]
        mean_frames = float(np.mean(lengths)) if lengths else 0.0
        print(" %4d %8d %11.1f %11d %15d %3d %11d %6d %5d"
              % (seed, coverage["episodes"], mean_frames,
                 coverage["pipe_deaths"], coverage["boundary_deaths"],
                 coverage["cap_deaths"], coverage["pipe_window_steps"],
                 coverage["high_y_steps"], coverage["low_y_steps"]))
    coverage_rows = [r["coverage"] for r in rows]
    print("Q8  coverage: mean episode %.1f frames; pipe deaths %d; boundary "
          "deaths %d; pipe-window steps %d"
          % (np.mean([np.mean(c["episode_lengths"]) for c in coverage_rows]),
             sum(c["pipe_deaths"] for c in coverage_rows),
             sum(c["boundary_deaths"] for c in coverage_rows),
             sum(c["pipe_window_steps"] for c in coverage_rows)))

    print("\nValue holdout (every fifth practice episode is not used for value "
          "updates):")
    print(" seed      n  terminal  pain AUC  risk AUC  near AUC  near-in  "
          "near-out  Brier")
    for seed, result in zip(seeds, rows):
        probe = result["value_probe"]
        if not probe["n"]:
            print(" %4d     0      0       n/a       n/a       n/a      n/a"
                  "       n/a    n/a" % seed)
            continue
        print(" %4d  %5d  %8d  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %6.3f"
              % (seed, probe["n"], probe["terminal"], probe["pain_auc"],
                 probe["risk_auc"], probe["near_pipe_risk_auc"],
                 probe["near_inside_risk"], probe["near_outside_risk"],
                 probe["risk_brier"]))
    probes = [r["value_probe"] for r in rows if r["value_probe"]["n"]]
    if probes:
        print("Q7  value holdout: pain AUC %.3f; risk AUC %.3f; near-pipe risk "
              "AUC %.3f; near-in/out risk %.3f/%.3f; risk Brier %.3f"
              % (np.nanmean([p["pain_auc"] for p in probes]),
                 np.nanmean([p["risk_auc"] for p in probes]),
                 np.nanmean([p["near_pipe_risk_auc"] for p in probes]),
                 np.nanmean([p["near_inside_risk"] for p in probes]),
                 np.nanmean([p["near_outside_risk"] for p in probes]),
                 np.nanmean([p["risk_brier"] for p in probes])))


def _run_task(task):
    return run(**task)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-baseline", action="store_true")
    parser.add_argument("--engine", choices=("exhaustive", "beam"),
                        default="exhaustive")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--practice-steps", type=int, default=PRACTICE)
    parser.add_argument("--value-updates", type=int, default=VALUE_UPDATES)
    parser.add_argument("--risk-updates", type=int, default=RISK_UPDATES)
    parser.add_argument("--terminal-fraction", type=float,
                        default=TERMINAL_FRACTION)
    parser.add_argument("--near-pipe-fraction", type=float,
                        default=NEAR_PIPE_FRACTION)
    parser.add_argument("--risk-interaction-dim", type=int,
                        default=RISK_INTERACTION_DIM)
    parser.add_argument("--risk-action-dim", type=int,
                        default=RISK_ACTION_DIM)
    parser.add_argument("--risk-aggregate",
                        choices=("terminal", "sum", "max", "crossing"),
                        default="terminal")
    args = parser.parse_args(argv)

    value_mode = "task" if args.task_baseline else "feedback"
    engine = "exhaustive" if value_mode == "task" else args.engine
    if args.horizon is not None:
        horizon = args.horizon
    elif engine == "beam":
        horizon = LONG_H
    else:
        horizon = H
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")

    print("sandbox value mode: %s" % value_mode)
    print("sandbox engine: %s; horizon=%d; beam_width=%d; jobs=%d"
          % (engine, horizon, args.beam_width, args.jobs))
    print("practice=%d real steps/seed; value_updates=%d; risk_updates=%d; "
          "terminal_fraction=%.2f; near_pipe_fraction=%.2f; "
          "risk_interaction_dim=%d; risk_action_dim=%d; risk_aggregate=%s; "
          "sandbox predictions are not training samples\n"
          % (args.practice_steps, args.value_updates, args.risk_updates,
             args.terminal_fraction, args.near_pipe_fraction,
             args.risk_interaction_dim, args.risk_action_dim,
             args.risk_aggregate))

    tasks = [
        dict(
            seed=seed, value_mode=value_mode, engine=engine, horizon=horizon,
            beam_width=args.beam_width, practice_steps=args.practice_steps,
            value_updates=args.value_updates, risk_updates=args.risk_updates,
            terminal_fraction=args.terminal_fraction,
            near_pipe_fraction=args.near_pipe_fraction,
            risk_interaction_dim=args.risk_interaction_dim,
            risk_action_dim=args.risk_action_dim,
            risk_aggregate=args.risk_aggregate, show=(seed == SEEDS[0]),
        )
        for seed in SEEDS[:args.seeds]
    ]
    if args.jobs == 1:
        rows = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_run_task, tasks))
    report(rows, value_mode)


if __name__ == "__main__":
    main()
