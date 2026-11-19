"""Compare raw, difference, and buffered observation signals for world prediction.

This is an offline ablation on real episodes collected by the task-agnostic prototype.
It does not change the controller. For each signal mode we fit the same linear transition

    signal_features(window) + action history -> next raw observation

and report one-step errors and multi-step errors on held-out episodes. The multi-step
probe recurses on predicted observations but keeps the recorded action history
(teacher-forced actions); it is not a closed-loop action rollout.
"""
import argparse

import numpy as np

from substrate_generic import ACTION_DIM, LinearObservationModel, collect

WINDOW = 5
BETAS = (0.5, 0.2, 0.1, 0.05, 0.02)
NAMES = ("y", "vy", "dx1", "gap1", "dx2", "gap2")


def signal_features(window, mode):
    """window: (..., WINDOW, obs_dim) -> feature channels."""
    window = np.asarray(window, dtype=float)
    raw = window[..., -1, :]
    parts = [raw]
    if mode in ("diff", "all"):
        d1 = window[..., -1, :] - window[..., -2, :]
        d2 = (window[..., -1, :]
              - 2.0 * window[..., -2, :]
              + window[..., -3, :])
        parts.extend([d1, d2])
    if mode in ("buffer", "all"):
        for beta in BETAS:
            u = np.zeros_like(raw)
            for index in range(window.shape[-2]):
                u = (1.0 - beta) * u + beta * window[..., index, :]
            parts.append(u)
    return np.concatenate(parts, axis=-1)


def episode_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["episode"], []).append(row)
    return [grouped[key] for key in sorted(grouped)]


def make_samples(episodes):
    windows = []
    histories = []
    targets = []
    for episode in episodes:
        observations = [row["obs"] for row in episode]
        for index, row in enumerate(episode):
            start = max(0, index - WINDOW + 1)
            window = observations[start:index + 1]
            if len(window) < WINDOW:
                window = [window[0]] * (WINDOW - len(window)) + window
            windows.append(np.asarray(window, dtype=float))
            histories.append(row["history"])
            targets.append(row["next_obs"])
    return (
        np.asarray(windows, dtype=float),
        np.asarray(histories, dtype=float),
        np.asarray(targets, dtype=float),
    )


def one_step_errors(model, episodes, mode):
    windows, histories, targets = make_samples(episodes)
    features = signal_features(windows, mode)
    prediction = model.predict(features, histories)
    errors = np.abs(prediction - targets)
    return errors


def multi_step_errors(model, episodes, mode, horizon=14, step=7):
    errors = []
    for episode in episodes:
        observations = [row["obs"] for row in episode]
        for start in range(0, max(1, len(episode) - horizon), step):
            start_window = observations[max(0, start - WINDOW + 1):start + 1]
            if len(start_window) < WINDOW:
                start_window = [start_window[0]] * (WINDOW - len(start_window)) + start_window
            window = np.asarray(start_window, dtype=float)
            for offset in range(horizon):
                index = start + offset
                if index >= len(episode):
                    break
                features = signal_features(window[None, ...], mode)
                prediction = model.predict(features, episode[index]["history"][None, ...])[0]
                target = episode[index]["next_obs"]
                errors.append(np.abs(prediction - target))
                window = np.concatenate([window[1:], prediction[None, :]], axis=0)
    return np.asarray(errors, dtype=float)


def run(seed, steps, coverage):
    rows, _counts = collect(seed, steps, coverage=coverage)
    episodes = episode_rows(rows)
    train = [episode for index, episode in enumerate(episodes) if index % 5 != 4]
    held = [episode for index, episode in enumerate(episodes) if index % 5 == 4]
    print("seed=%d episodes=%d train=%d held=%d" % (
        seed, len(episodes), len(train), len(held)
    ))
    print("mode     one-step mean/max        multi-step mean/max")
    for mode in ("raw", "diff", "buffer", "all"):
        windows, histories, targets = make_samples(train)
        features = signal_features(windows, mode)
        model = LinearObservationModel(features.shape[-1], ACTION_DIM).fit(
            features, histories, targets
        )
        one = one_step_errors(model, held, mode)
        multi = multi_step_errors(model, held, mode)
        print("%-7s  %.2e / %.2e          %.2e / %.2e"
              % (mode, one.mean(), one.max(), multi.mean(), multi.max()))
        for index, name in enumerate(NAMES[:one.shape[1]]):
            if index in (0, 1, 2, 3):
                print("  %-4s one %.2e/% .2e  multi %.2e/% .2e"
                      % (name, one[:, index].mean(), one[:, index].max(),
                         multi[:, index].mean(), multi[:, index].max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.coverage <= 1.0:
        raise ValueError("--coverage must be in [0, 1]")
    run(args.seed, args.steps, args.coverage)


if __name__ == "__main__":
    main()
