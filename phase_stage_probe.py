"""Is a phase-relative stage state more informative than an absolute one?

The absolute partition has the right dwell time but its cells are partly
clock cells; removing the linear trend threw the danger signal away with
it.  This probe tests the remaining hypothesis: hold the *phase* of the
world cycle, and describe the state only by its deviation from what is
typical at that phase.

Measured with one protocol for every representation, so the comparison is
fair despite different cell counts:

* leave-one-episode-out prediction of ``pain within 8 steps``;
* Brier score and rank AUC against the base rate;
* eta squared over the same cells, for reference.

Representations compared: absolute states (k-means), phase only, and
phase x relative-state (each phase bin split into a few clusters on the
deviation from that bin's mean).  The period is estimated from the
observations themselves by autocorrelation, never named.
"""

import argparse

import numpy as np

from stage_graph import _kmeans
from substrate_nodes import LAG, _curiosity_off, _encoded_pending, evaluate, fit


def estimate_period(observations, low=6, high=80):
    z = np.asarray(observations, dtype=float)
    z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-9)
    count = len(z)
    curves = []
    for column in range(z.shape[1]):
        spectrum = np.abs(np.fft.rfft(z[:, column], n=2 * count)) ** 2
        autocorrelation = np.fft.irfft(spectrum)[:count]
        if autocorrelation[0] > 1e-12:
            autocorrelation = autocorrelation / autocorrelation[0]
        curves.append(autocorrelation)
    curve = np.mean(curves, axis=0)
    high = min(high, count // 3)
    if high <= low:
        return 1, curve
    return low + int(np.argmax(curve[low:high])), curve


def horizon_labels(rows, horizon=8):
    episodes = {}
    for index, row in enumerate(rows):
        episodes.setdefault(int(row["episode"]), []).append(index)
    y = np.zeros(len(rows), dtype=float)
    for indices in episodes.values():
        for position, index in enumerate(indices):
            window = indices[position:position + horizon]
            if any(float(rows[other]["signals"][1]) > 0.5 for other in window):
                y[index] = 1.0
    return y


def auc(scores, labels):
    labels = np.asarray(labels, dtype=float)
    positive = labels > 0.5
    n1 = int(positive.sum())
    n0 = int(len(labels) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def leave_one_episode_out(cells, y, episodes):
    """Predict y from the cell mean fitted on the other episodes."""
    predicted = np.zeros(len(y), dtype=float)
    for episode in np.unique(episodes):
        holdout = episodes == episode
        fit_rows = ~holdout
        for cell in np.unique(cells):
            target = holdout & (cells == cell)
            if not target.any():
                continue
            members = fit_rows & (cells == cell)
            predicted[target] = (
                float(y[members].mean()) if members.any() else float(y[fit_rows].mean())
            )
    return predicted


def eta_squared(cells, y):
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for cell in np.unique(cells):
        members = y[cells == cell]
        between += len(members) * (members.mean() - y.mean()) ** 2
    return float(between / total)


def report(name, cells, y, episodes):
    predicted = leave_one_episode_out(cells, y, episodes)
    brier = float(np.mean((predicted - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))
    print("  %-22s cells %3d  Brier %.5f (base %.5f)  AUC %.3f  eta2 %.3f" % (
        name, int(len(np.unique(cells))), brier, base,
        auc(predicted, y), eta_squared(cells, y),
    ))
    return brier, auc(predicted, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--phase-bins", type=int, default=8)
    parser.add_argument("--per-bin", type=int, default=3)
    parser.add_argument("--absolute", type=int, default=24)
    parser.add_argument("--cap", type=int, default=5000)
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    eval_seeds = parse_seed_spec(args.eval_seeds)

    world, agent = fit(
        seed=args.seed, steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=8, step_stride=2,
    )
    rows = []
    with _curiosity_off(agent):
        for eval_seed in eval_seeds:
            _frames, _actions, transitions = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=8,
                lag=LAG, step_stride=2, env_seed=eval_seed * 10000,
                record=True,
            )
            for row in transitions:
                row = dict(row)
                row["episode"] = int(eval_seed)
                rows.append(row)
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    episodes = np.asarray([int(row["episode"]) for row in rows])
    y = horizon_labels(rows)
    pending = _encoded_pending(rows, LAG)
    encoded = np.concatenate(
        [observations, pending], axis=1
    ) / agent.scales

    period, curve = estimate_period(observations)
    order = np.argsort(episodes)
    phase_index = np.zeros(len(rows), dtype=int)
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        phase_index[indices] = np.arange(len(indices)) % period
    phase = (phase_index.astype(float) / period)
    phase_bin = np.minimum(
        (phase * args.phase_bins).astype(int), args.phase_bins - 1
    )

    print("== phase-relative stages: seed %d, %d frames, %d episodes ==" % (
        args.seed, len(rows), len(eval_seeds)
    ))
    print("  estimated period %d frames (autocorrelation peak %.3f), "
          "y8 base %.3f" % (period, float(curve[period]), float(y.mean())))
    report("absolute", np.asarray(
        _kmeans(encoded, args.absolute, seed=0), dtype=int
    ), y, episodes)
    report("phase only", phase_bin, y, episodes)

    # Phase x relative state: subtract the phase bin's typical state first,
    # then split each bin by the deviation.
    relative = encoded.copy()
    for index in range(args.phase_bins):
        members = phase_bin == index
        if members.any():
            relative[members] -= relative[members].mean(axis=0)
    cells = np.zeros(len(rows), dtype=int)
    for index in range(args.phase_bins):
        members = np.flatnonzero(phase_bin == index)
        if len(members) < args.per_bin:
            cells[members] = index
            continue
        local = _kmeans(relative[members], args.per_bin, seed=0)
        cells[members] = index * args.per_bin + np.asarray(local, dtype=int)
    report("phase x relative", cells, y, episodes)


if __name__ == "__main__":
    main()
