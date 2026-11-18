"""Trace whether the world-feedback policy acts on pipe geometry.

The diagnostic runs one trained seed and records decisions near the nearest pipe:

    gap_dy = gap_centre - bird_y

If the gap is above the bird, a flap is the action that can raise it toward the gap.
If the gap is below, not flapping is the directional action. The script reports whether
the selected actions follow that rule and whether the learned risk head assigns less
risk to "all flap" than "all no-flap".

This is a behaviour trace, not a policy-gradient training loop. The value and risk
heads are trained from real transitions; the search uses the learned transition model.
"""
import argparse

import numpy as np

from latent_world import BIRD_X, SPEED, latent
from substrate_sandbox_search import (
    BEAM_WIDTH,
    H,
    LAG,
    nearest_pipe,
    observe_transition,
    practice,
    rollout_candidates,
)
from substrate_value import planned_substrate_features


def endpoint_values(env, lp, feedback_head, risk_head, novelty, lag, b, f, seq):
    """Endpoint risk/value for one action sequence under the learned transition model."""
    pending = np.asarray(env.pending()[:lag], dtype=float)
    if len(pending) < lag:
        pending = np.pad(pending, (0, lag - len(pending)))
    y, vy = float(env.y), float(env.vy)
    seq = np.asarray(seq, dtype=float).reshape(-1)
    for step in range(H):
        impulse = pending[step] if step < lag else seq[step - lag]
        vy = vy + b + f * impulse
        y = y + vy

    pipe1, pipe2 = env.pipes[0], env.pipes[-1]
    features = planned_substrate_features(lp, seq.reshape(1, -1), novelty)
    observations = np.asarray([[
        y - 0.5,
        vy * 10.0,
        pipe1[0] - H * SPEED - BIRD_X,
        pipe1[1] - y,
        pipe2[0] - H * SPEED - BIRD_X,
        pipe2[1] - y,
    ]])
    x = np.concatenate([features, observations], axis=1)
    return {
        "endpoint_y": y,
        "risk": float(risk_head.probability(x)[0]),
        "value": feedback_head.value(x)[0].copy(),
    }


def directional_summary(events):
    """Action/gap agreement near pipes. Gap offsets under 0.02 are ignored."""
    usable = [event for event in events if abs(event["gap_dy"]) > 0.02]
    need_up = [event for event in usable if event["gap_dy"] > 0.02]
    need_down = [event for event in usable if event["gap_dy"] < -0.02]
    correct = (
        [event["action"] == 1 for event in need_up]
        + [event["action"] == 0 for event in need_down]
    )
    return {
        "usable": len(usable),
        "need_up": len(need_up),
        "need_up_flap_rate": (
            float(np.mean([event["action"] for event in need_up]))
            if need_up else float("nan")
        ),
        "need_down": len(need_down),
        "need_down_flap_rate": (
            float(np.mean([event["action"] for event in need_down]))
            if need_down else float("nan")
        ),
        "directional_accuracy": (
            float(np.mean(correct)) if correct else float("nan")
        ),
        "one_better_than_zero": sum(
            event["one_risk"] < event["zero_risk"] for event in events
        ),
        "events": len(events),
        "mean_risk_zero": (
            float(np.mean([event["zero_risk"] for event in events]))
            if events else float("nan")
        ),
        "mean_risk_one": (
            float(np.mean([event["one_risk"] for event in events]))
            if events else float("nan")
        ),
        "mean_risk_chosen": (
            float(np.mean([event["chosen_risk"] for event in events]))
            if events else float("nan")
        ),
    }


def trace(seed, risk_aggregate="terminal", distance=0.25, show_events=12):
    lp, lag, f, feedback_head, risk_head, novelty, probe, replay, coverage = practice(seed)
    b = lp.b
    env = latent[LAG](seed)
    lp.hist = [0.0] * 8
    events = []
    t = 0
    while not env.dead and t < 6000:
        pipe_dx, gap_dy = nearest_pipe(env)
        plan, _ = rollout_candidates(
            env, b, f, lag, lp, feedback_head, risk_head, novelty,
            "feedback", "exhaustive", H, BEAM_WIDTH,
            risk_aggregate=risk_aggregate,
        )
        action = int(plan.actions[0])
        zero = endpoint_values(
            env, lp, feedback_head, risk_head, novelty, lag, b, f,
            np.zeros(H, dtype=float),
        )
        one = endpoint_values(
            env, lp, feedback_head, risk_head, novelty, lag, b, f,
            np.ones(H, dtype=float),
        )
        chosen = endpoint_values(
            env, lp, feedback_head, risk_head, novelty, lag, b, f,
            plan.actions,
        )
        if abs(pipe_dx) < distance and pipe_dx > -0.15:
            events.append(dict(
                t=t, y=float(env.y), vy=float(env.vy),
                pipe_dx=pipe_dx, gap_dy=gap_dy, action=action,
                chosen_risk=chosen["risk"], zero_risk=zero["risk"],
                one_risk=one["risk"], zero_y=zero["endpoint_y"],
                one_y=one["endpoint_y"], chosen_y=chosen["endpoint_y"],
                score=plan.score,
            ))

        vy0 = env.vy
        observation = env.obs()
        env.step(action)
        next_observation = env.obs()
        world_feedback = env.feedback()
        observe_transition(
            lp, feedback_head, novelty, env.vy - vy0, float(action),
            observation, next_observation, world_feedback, env.dead,
            learn_value=False,
        )
        j = int(np.argmax(np.abs(lp.h)))
        lag, f, b = j + 1, float(lp.h[j]), float(lp.b)
        t += 1

    summary = directional_summary(events)
    print("seed %d: frames=%d near_pipe_events=%d" % (seed, env.t, len(events)))
    print("directional summary:", summary)
    print("first %d near-pipe decisions:" % show_events)
    for event in events[:show_events]:
        print(
            " t=%3d dx=%+.3f gap=%+.3f act=%d "
            "risk(zero/one/chosen)=%.3f/%.3f/%.3f "
            "y(zero/one/chosen)=%+.3f/%+.3f/%+.3f"
            % (event["t"], event["pipe_dx"], event["gap_dy"], event["action"],
               event["zero_risk"], event["one_risk"],
               event["chosen_risk"], event["zero_y"], event["one_y"],
               event["chosen_y"])
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", nargs="?", type=int, default=100)
    parser.add_argument("--risk-aggregate", choices=("terminal", "sum", "max"),
                        default="terminal")
    parser.add_argument("--distance", type=float, default=0.25)
    parser.add_argument("--show-events", type=int, default=12)
    args = parser.parse_args()
    trace(args.seed, args.risk_aggregate, args.distance, args.show_events)


if __name__ == "__main__":
    main()
