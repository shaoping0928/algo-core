"""Task-agnostic prototype: raw observations -> learned world/value/risk -> search.

The planner receives only:

    obs_t, action_t, feedback_t, terminal_t

It never reads Flappy's pipe list, gap centre, bird x, speed, or collision window.
The world model learns the full observation transition from real transitions:

    (obs_t, action history) -> obs_{t+1}

Value and terminal risk are learned from the same observation/action history. Search
rolls out the learned model and chooses actions; no task reward formula is supplied.
"""
import argparse
from collections import deque

import numpy as np

from latent_world import latent
from predict_layer import LayerPair
from substrate_value import (
    ReturnReplay,
    TerminalRiskHead,
    TransitionReplay,
    WorldFeedbackValueHead,
    discounted_returns,
    fit_terminal_risk,
    fit_value_head_returns,
)
from trajectory_search import exhaustive_search

LAG = 3
H = 14
PRACTICE = 5000
ACTION_DIM = LAG + 1
VALUE_UPDATES = 3000
VALUE_BATCH = 128
RISK_UPDATES = 6000
TERMINAL_FRACTION = 0.15
PRIORITY_ALPHA = 1.0


class LinearObservationModel:
    """Ridge regression from (obs, action history) to the next raw observation."""

    def __init__(self, obs_dim, action_dim, ridge=1e-8):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.ridge = float(ridge)
        self.weights = None
        self.train_error = float("nan")
        self.holdout_error = float("nan")

    def _features(self, observations, histories):
        observations = np.asarray(observations, dtype=float)
        histories = np.asarray(histories, dtype=float)
        if observations.shape[-1] != self.obs_dim:
            raise ValueError("observation dimension mismatch")
        if histories.shape[-1] != self.action_dim:
            raise ValueError("action-history dimension mismatch")
        return np.concatenate(
            [observations, histories, np.ones(observations.shape[:-1] + (1,))],
            axis=-1,
        )

    def fit(self, observations, histories, next_observations, holdout=0.1):
        observations = np.asarray(observations, dtype=float)
        histories = np.asarray(histories, dtype=float)
        next_observations = np.asarray(next_observations, dtype=float)
        if len(observations) < 2:
            raise ValueError("need at least two transitions")
        order = np.arange(len(observations))
        # A deterministic tail holdout keeps the report reproducible without a task RNG.
        cut = max(1, int(round(len(order) * (1.0 - holdout))))
        train, held = order[:cut], order[cut:]
        x_train = self._features(observations[train], histories[train])
        y_train = next_observations[train]
        gram = x_train.T @ x_train + self.ridge * np.eye(x_train.shape[1])
        self.weights = np.linalg.solve(gram, x_train.T @ y_train)
        self.train_error = float(
            np.max(np.abs(x_train @ self.weights - y_train))
        )
        if len(held):
            x_held = self._features(observations[held], histories[held])
            self.holdout_error = float(
                np.max(np.abs(x_held @ self.weights - next_observations[held]))
            )
        return self

    def predict(self, observations, histories):
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        return self._features(observations, histories) @ self.weights


def history_features(sequences, steps, prefix, action_dim=ACTION_DIM):
    """Newest action first; fill the pre-plan prefix for early rollout steps."""
    sequences = np.asarray(sequences, dtype=float)
    prefix = np.asarray(prefix, dtype=float).reshape(-1)
    result = np.zeros((len(sequences), int(action_dim)), dtype=float)
    for offset in range(int(action_dim)):
        index = steps - 1 - offset
        if index >= 0:
            result[:, offset] = sequences[:, index]
        elif -index - 1 < len(prefix):
            result[:, offset] = prefix[-index - 1]
    return result


def collect(seed, steps=PRACTICE, coverage=0.0, lag=LAG):
    """Collect real transitions using only observation/action/feedback."""
    lp = LayerPair(k_max=8)
    rng = np.random.RandomState(seed * 10000)
    transitions = []
    action_counts = {}
    done = episode = 0
    while done < steps:
        env = latent[int(lag)](seed * 10000 + episode)
        lp.hist = [0.0] * 8
        while not env.dead and done < steps:
            gains = np.abs(lp.h)
            total = float(gains.sum())
            confidence = float(gains.max() / total) if total > 1e-9 else 0.0
            if total > 1e-9 and (lp.t >= 20 or confidence >= 0.25):
                gain = float(lp.h[int(np.argmax(gains))])
                center = float(np.clip(-lp.b / gain, 0.0, 1.0))
            else:
                center = 0.1
            explore = 0.03 + 0.10 * (1.0 - confidence)
            obs = np.asarray(env.obs(), dtype=float)
            action = int(rng.rand() < np.clip(
                center + rng.uniform(-explore, explore), 0.0, 1.0
            ))
            if coverage > 0.0 and rng.rand() < coverage:
                # Generic action-coverage exploration: choose the less-used action
                # in the current observation bucket. This uses no task geometry.
                key = tuple(np.round(obs, 1))
                counts = action_counts.setdefault(key, [0, 0])
                action = int(counts[1] < counts[0] or (
                    counts[1] == counts[0] and rng.rand() < 0.5
                ))
            key = tuple(np.round(obs, 1))
            action_counts.setdefault(key, [0, 0])[action] += 1
            vy0 = env.vy
            history = np.asarray([action] + lp.hist[:int(lag)], dtype=float)
            env.step(action)
            next_obs = np.asarray(env.obs(), dtype=float)
            feedback = env.feedback()
            signals = np.asarray(
                [feedback["dopamine"], feedback["pain"]], dtype=float
            )
            lp.observe(env.vy - vy0, float(action))
            transitions.append(dict(
                obs=obs, next_obs=next_obs, history=history,
                signals=signals, terminal=bool(env.dead),
                done=bool(env.dead),
                episode=episode,
            ))
            done += 1
        episode += 1
    return transitions, action_counts


def fit(transitions, seed):
    if not transitions:
        raise ValueError("need at least one transition to fit")
    obs_dim = len(transitions[0]["obs"])
    world = LinearObservationModel(obs_dim, ACTION_DIM).fit(
        [row["obs"] for row in transitions],
        [row["history"] for row in transitions],
        [row["next_obs"] for row in transitions],
    )
    observations = np.asarray([row["obs"] for row in transitions], dtype=float)
    histories = np.asarray([row["history"] for row in transitions], dtype=float)
    next_observations = np.asarray(
        [row["next_obs"] for row in transitions], dtype=float
    )
    errors = np.abs(
        world.predict(observations, histories) - next_observations
    ).mean(axis=1)
    scale = float(np.median(errors))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    priorities = 1.0 + np.minimum(errors / scale, 10.0)

    value_replay = ReturnReplay(capacity=len(transitions) + 1, seed=seed)
    risk_replay = TransitionReplay.for_stream(len(transitions), seed=seed)
    value_head = WorldFeedbackValueHead(
        feature_dim=obs_dim + ACTION_DIM, discount=0.99, seed=seed
    )
    risk_head = TerminalRiskHead(feature_dim=obs_dim + ACTION_DIM)
    by_episode = {}
    for index, row in enumerate(transitions):
        by_episode.setdefault(row["episode"], []).append((index, row))
        risk = np.concatenate([row["next_obs"], row["history"]])
        risk_replay.add(
            risk, row["signals"], risk, row["terminal"],
            priority=float(priorities[index]),
        )
    for episode in by_episode.values():
        rows = [row for _index, row in episode]
        targets = discounted_returns(rows, value_head.discount)
        for (index, row), target in zip(episode, targets):
            value_head_features = np.concatenate([row["obs"], row["history"]])
            value_replay.add(
                value_head_features, target, priority=float(priorities[index])
            )
    fit_value_head_returns(
        value_head, value_replay, VALUE_UPDATES, batch_size=VALUE_BATCH,
        priority_alpha=PRIORITY_ALPHA,
    )
    fit_terminal_risk(
        risk_head, risk_replay, RISK_UPDATES, batch_size=VALUE_BATCH,
        terminal_fraction=TERMINAL_FRACTION, priority_alpha=PRIORITY_ALPHA,
    )
    return world, value_head, risk_head


def policy_action(env, world, value_head, risk_head, prefix):
    """One generic MPC step; only raw observation and learned models are used."""
    initial = np.asarray(env.obs(), dtype=float)
    sequences = np.zeros((1, H), dtype=float)
    pending_prefix = np.asarray(prefix, dtype=float)

    def predict_batch(states, sequences, step):
        history = history_features(sequences, step + 1, pending_prefix)
        return world.predict(states, history)

    def evaluate_step(states, sequences, step):
        history = history_features(sequences, step + 1, pending_prefix)
        risk = risk_head.probability(np.concatenate([states, history], axis=1))
        zeros = np.zeros(len(states))
        return np.column_stack([risk, zeros, zeros]), np.ones(
            len(states), dtype=bool
        )

    def evaluate_terminal(states, sequences):
        history = history_features(sequences, H, pending_prefix)
        risk = risk_head.probability(np.concatenate([states, history], axis=1))
        value = value_head.value(np.concatenate([states, history], axis=1))
        return np.column_stack([risk, value[:, 1], -value[:, 0]]), np.ones(
            len(states), dtype=bool
        )

    result = exhaustive_search(
        initial_state=initial,
        action_space=(0, 1),
        horizon=H,
        transition_batch=predict_batch,
        evaluate_step=evaluate_step,
        evaluate_terminal=evaluate_terminal,
        batch_size=2 ** H,
        score_combine=("max", "sum", "sum"),
    )
    return int(result.actions[0])


def evaluate(seed, world, value_head, risk_head, cap=6000):
    env = latent[LAG](seed)
    prefix = np.zeros(LAG, dtype=float)
    actions = []
    while not env.dead and env.t < cap:
        action = policy_action(env, world, value_head, risk_head, prefix)
        actions.append(action)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        env.step(action)
    return env.t, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=PRACTICE)
    parser.add_argument("--coverage", type=float, default=0.0)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.coverage <= 1.0:
        raise ValueError("--coverage must be in [0, 1]")
    transitions, action_counts = collect(
        args.seed, args.steps, coverage=args.coverage
    )
    world, value_head, risk_head = fit(transitions, args.seed)
    frames, actions = evaluate(args.seed, world, value_head, risk_head)
    both = sum(1 for counts in action_counts.values()
               if counts[0] > 0 and counts[1] > 0)
    print("seed", args.seed, "transitions", len(transitions), "frames", frames)
    print("one-step max error train %.2e holdout %.2e"
          % (world.train_error, world.holdout_error))
    print("actions executed", len(actions), "flap rate %.3f"
          % (np.mean(actions) if actions else 0.0))
    print("coverage buckets %d/%d have both actions"
          % (both, len(action_counts)))


if __name__ == "__main__":
    main()
