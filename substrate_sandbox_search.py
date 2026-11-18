"""Flappy adapter for generic search and world-provided valence feedback.

This adapter supplies state encoding, a learned one-step transition predictor, and an evaluator
to `trajectory_search.exhaustive_search`. By default, the value head learns from the world's
dopamine/pain channels; `--task-baseline` opts into the hand-specified gap/collision score.
The search core itself has no Flappy knowledge.

Sandbox predictions are not fed to the world model; only real transitions update it.
"""
import numpy as np
import sys

from latent_world import BIRD_X, GAP, SEEDS, SPEED, latent
from predict_layer import LayerPair
from substrate_value import (
    InputNovelty,
    WorldFeedbackValueHead,
    planned_substrate_features,
    substrate_features,
)
from trajectory_search import exhaustive_search

LAG = 3
H = 14
PRACTICE = 5000
ACTION_SPACE = (0, 1)
VALUE_MODE = "task" if "--task-baseline" in sys.argv[1:] else "feedback"


def observe_transition(
    lp, feedback_head, novelty, delta, event, observation, next_observation,
    world_feedback, terminal=False
):
    """Learn only from the world feedback attached to a real transition."""
    novelty_score = novelty.observe(lp.hist)
    features = substrate_features(lp, novelty_score=novelty_score)
    feedback_features = np.concatenate(
        [features, np.asarray(observation, dtype=float).reshape(-1)]
    )
    error = float(delta - lp.predict())
    lp.observe(delta, event)
    next_features = substrate_features(lp, novelty_score=novelty.score(lp.hist))
    next_feedback_features = np.concatenate(
        [next_features, np.asarray(next_observation, dtype=float).reshape(-1)]
    )
    signals = np.asarray(
        [world_feedback["dopamine"], world_feedback["pain"]], dtype=float
    )
    feedback_head.update(
        feedback_features, signals, next_feedback_features, terminal=terminal
    )
    return error


def practice(seed, steps=PRACTICE):
    lp = LayerPair(k_max=8)
    novelty = InputNovelty(feature_dim=len(lp.hist))
    sample_env = latent[LAG](seed * 10000)
    feedback_head = WorldFeedbackValueHead(
        feature_dim=len(substrate_features(lp)) + len(sample_env.obs()),
        discount=0.99,
    )
    done = ep = 0
    while done < steps:
        env = latent[LAG](seed * 10000 + ep)
        rng = np.random.RandomState(seed * 10000 + ep)
        lp.hist = [0.0] * 8
        while not env.dead and done < steps:
            gains = np.abs(lp.h)
            total_gain = float(gains.sum())
            confidence = float(gains.max() / total_gain) if total_gain > 1e-9 else 0.0
            if lp.t >= 50 and confidence >= 0.5:
                lag_index = int(np.argmax(gains))
                gain = float(lp.h[lag_index])
                center_rate = float(np.clip(-lp.b / gain, 0.0, 1.0))
            else:
                center_rate = 0.5
            # Explore around the model-implied neutral action rate; uncertainty widens the
            # exploration band. No task target or collision rule enters data collection.
            explore = 0.05 + 0.20 * (1.0 - confidence)
            action_rate = float(np.clip(
                center_rate + rng.uniform(-explore, explore), 0.0, 1.0
            ))
            action = int(rng.rand() < action_rate)
            vy0 = env.vy
            observation = env.obs()
            env.step(action)
            next_observation = env.obs()
            world_feedback = env.feedback()
            observe_transition(
                lp, feedback_head, novelty, env.vy - vy0, float(action),
                observation, next_observation, world_feedback, env.dead,
            )
            done += 1
        ep += 1
    j = int(np.argmax(np.abs(lp.h)))
    return lp, j + 1, float(lp.h[j]), feedback_head, novelty


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
    env, b, f, lag, lp, feedback_head, novelty, value_mode
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

    def evaluate_step(states, sequences, step):
        if value_mode == "feedback":
            features = planned_substrate_features(
                lp, sequences[:, :step + 1], novelty
            )
            pipe1, pipe2 = env.pipes[0], env.pipes[-1]
            pipe1_x = pipe1[0] - (step + 1) * SPEED
            pipe2_x = pipe2[0] - (step + 1) * SPEED
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
            feedback_features = np.concatenate(
                [features, observations], axis=1
            )
            world_values = feedback_head.value(feedback_features)
            scores = feedback_head.preference_scores(world_values)
            return scores, np.isfinite(feedback_features).all(axis=1)

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

    result = exhaustive_search(
        initial_state=initial_state,
        action_space=ACTION_SPACE,
        horizon=H,
        transition_batch=predict_batch,
        evaluate_step=evaluate_step,
        batch_size=len(ACTION_SPACE) ** H,
    )
    return result, gc


def run(seed, show=False):
    lp, lag, f, feedback_head, novelty = practice(seed)
    b = lp.b
    env = latent[LAG](seed)
    lp.hist = [0.0] * 8
    t = searches = feasible_total = candidate_total = no_feasible = 0
    max_model_error = 0.0
    selected = []
    search_memory = []

    while not env.dead and t < 6000:
        plan, gc = rollout_candidates(
            env, b, f, lag, lp, feedback_head, novelty,
            VALUE_MODE,
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
        )
        # Refresh the learned parameters from real data only.
        j = int(np.argmax(np.abs(lp.h)))
        lag, f, b = j + 1, float(lp.h[j]), float(lp.b)
        t += 1

    return dict(frames=env.t, lag=lag, f=f, b=b, searches=searches,
                feasible_fraction=feasible_total / max(candidate_total, 1),
                no_feasible=no_feasible, model_error=max_model_error, selected=selected,
                search_memory=search_memory)


print("sandbox value mode: %s" % VALUE_MODE)
print("sandbox: %d action sequences (H=%d); generic search core; execute first action; re-search"
      % (len(ACTION_SPACE) ** H, H))
print("practice=%d real steps/seed; sandbox predictions are not training samples\n" % PRACTICE)
candidate_metric = "valid fraction" if VALUE_MODE == "feedback" else "feasible fraction"
failure_metric = "invalid" if VALUE_MODE == "feedback" else "no-feasible"
print(" seed   frames   lag   learned F   %-17s %s steps   max model error"
      % (candidate_metric, failure_metric))
rows = []
for seed in SEEDS[:8]:
    result = run(seed, show=(seed == SEEDS[0]))
    rows.append(result)
    print(" %4d   %6d   %3d   %.6f        %.3f              %5d             %.2e"
          % (seed, result["frames"], result["lag"], result["f"],
             result["feasible_fraction"], result["no_feasible"], result["model_error"]))

selected_validity = "valid" if VALUE_MODE == "feedback" else "feasible"
print("\nselected candidates from seed %d (prefix, predicted endpoint, value target, %s, score):"
      % (SEEDS[0], selected_validity))
for t, prefix, endpoint, gc, ok, score in rows[0]["selected"]:
    target = "world-feedback" if gc is None else "gap=%.4f" % gc
    print("  t=%3d  %s  y_pred=%.4f  %s  %s=%s  score=%s"
          % (t, prefix, endpoint, target, selected_validity, ok, score_text(score)))

print("\nQ1  learned lag 3 and F=.020: %s (mean lag %.2f, F %.6f)"
      % ("ok" if all(r["lag"] == LAG and abs(r["f"] - .02) < .002 for r in rows) else "MISS",
         np.mean([r["lag"] for r in rows]), np.mean([r["f"] for r in rows])))
print("Q2  candidate rollouts pass the active validity check: %s (mean fraction %.3f)"
      % ("ok" if np.mean([r["feasible_fraction"] for r in rows]) > 0 else "MISS",
         np.mean([r["feasible_fraction"] for r in rows])))
print("Q3  steps with no candidate passing the validity check: %d across %d seeds"
      % (sum(r["no_feasible"] for r in rows), len(rows)))
print("Q4  external survival evaluation: mean %.1f, median %.1f frames; seed range %d-%d"
      % (np.mean([r["frames"] for r in rows]),
         np.median([r["frames"] for r in rows]),
         min(r["frames"] for r in rows), max(r["frames"] for r in rows)))
print("    reference only: the hand-derived exact-model rule scored 4887, but it is not a learned search policy")
print("Q5  one-step model/world discrepancy: %.2e"
      % max(r["model_error"] for r in rows))
mem = [item for r in rows for item in r["search_memory"]]
print("Q6  separate sandbox-memory records: %d; mean |predicted next y - real next y| %.2e"
      % (len(mem), np.mean([abs(item["one_step_error"]) for item in mem])))
print("    each record stores state, pending queue, candidate count, feasible count, selected plan,")
print("    predicted endpoint, target, and the real one-step verification; only that real transition")
print("    updates the world model.")
