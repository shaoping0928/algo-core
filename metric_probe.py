"""Does geometric closeness mean outcome closeness?

The prototype channel reads ``E[pain | 经过这个点]`` and the similarity
kernel is a fixed Gaussian over scaled observation space.  That assumes
two states that are close in coordinates are the same *situation*.  This
probe tests the assumption directly on the planner's own standardized
state (``obs / scales`` plus the pending-action window, built exactly as
``_encoded_pending`` does):

1. cross-episode nearest neighbour: is the neighbour's outcome closer to
   the query's outcome than a random state's?
2. k-nearest-neighbour outcome prediction: does the geometry predict
   ``pain within H steps`` at all (AUC, Brier)?
3. same-action vs different-action neighbours: does restricting the
   neighbour pool to the action actually taken sharpen the prediction?
   That is the "the missing dimension is the action" test.
4. near-duplicate pairs: at matched geometry, do different actions lead
   to different outcomes?
5. channel check: how do the prototype / feature / composed reads rank
   against the horizon label (not just immediate pain)?

Only real transitions are used; nothing here trains on counterfactuals.
"""

import argparse
import json

import numpy as np

from substrate_generic import collect
from substrate_nodes import LAG, _encoded_pending, fit


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, the tie-safe input to the AUC formula."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and sorted_values[end] == sorted_values[position]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1) + 1.0
        position = end
    return ranks


def auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    positive = labels > 0.5
    n1 = int(positive.sum())
    n0 = int(len(labels) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = _ranks(scores)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(left, right) -> float:
    a = _ranks(left)
    b = _ranks(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else 0.0


def horizon_labels(rows, horizon: int):
    """``y`` = pain within the next H transitions, ``t`` = steps to it."""
    episodes = {}
    for index, row in enumerate(rows):
        episodes.setdefault(int(row["episode"]), []).append(index)
    y = np.zeros(len(rows), dtype=float)
    t = np.full(len(rows), float(horizon + 1))
    for indices in episodes.values():
        for position, index in enumerate(indices):
            window = indices[position:position + horizon]
            for step, other in enumerate(window):
                if float(rows[other]["signals"][1]) > 0.5:
                    y[index] = 1.0
                    t[index] = float(step + 1)
                    break
    return y, t


def episode_progress(rows):
    """Steps since the episode started: a generic temporal coordinate.

    The survival hazard rises with time in an episode, and observations
    encode that time, so a geometric neighbour can look predictive only
    because it sits at the same phase of another episode.  The time match
    below removes exactly that confound.
    """
    progress = np.zeros(len(rows), dtype=float)
    episodes = {}
    for index, row in enumerate(rows):
        episodes.setdefault(int(row["episode"]), []).append(index)
    for indices in episodes.values():
        for position, index in enumerate(indices):
            progress[index] = float(position)
    return progress


def neighbour_tables(states, episodes, actions):
    states = np.asarray(states, dtype=float)
    episodes = np.asarray(episodes)
    actions = np.asarray(actions)
    norms = np.einsum("ij,ij->i", states, states)
    squared = norms[:, None] + norms[None, :] - 2.0 * (states @ states.T)
    np.maximum(squared, 0.0, out=squared)
    distance = np.sqrt(squared)
    cross = episodes[:, None] != episodes[None, :]
    same_action = actions[:, None] == actions[None, :]
    return distance, cross, same_action


def knn_predict(distance, mask, y, take, sigma):
    """Weighted mean outcome over the nearest `take` admissible neighbours."""
    allowed = np.where(mask, distance, np.inf)
    count = min(int(take), allowed.shape[1])
    indices = np.argpartition(allowed, count - 1, axis=1)[:, :count]
    rows = np.arange(len(y))[:, None]
    nearest = allowed[rows, indices]
    weights = np.exp(-0.5 * nearest / (sigma * sigma))
    weights = np.where(np.isfinite(nearest), weights, 0.0)
    total = weights.sum(axis=1)
    predicted = (weights * y[indices]).sum(axis=1) / np.where(
        total > 0.0, total, 1.0
    )
    return predicted, nearest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--duplicate-quantile", type=float, default=0.01)
    parser.add_argument("--report", type=str, default="")
    args = parser.parse_args()

    world, agent = fit(
        seed=args.seed, steps=args.steps, epochs=1,
        max_nodes=1000, dagger_rounds=0,
    )
    rows, _counts = collect(
        args.seed, args.steps, coverage=0.1, lag=LAG
    )
    obs = np.asarray([row["obs"] for row in rows], dtype=float)
    pending = _encoded_pending(rows, LAG)
    states = np.concatenate([obs, pending], axis=1) / agent.scales
    episodes = np.asarray([int(row["episode"]) for row in rows])
    actions = np.asarray([int(row["history"][0]) for row in rows])

    y, t = horizon_labels(rows, args.horizon)
    base = float(y.mean())
    print("== metric probe: seed %d, %d rows, %d episodes ==" % (
        args.seed, len(rows), len(set(episodes.tolist()))
    ))
    print("  horizon %d: base rate %.4f, mean time-to-pain %.2f" % (
        args.horizon, base, float(np.where(y > 0.5, t, args.horizon + 1).mean())
    ))
    print("  action rate %.3f" % float(actions.mean()))

    distance, cross, same_action = neighbour_tables(
        states, episodes, actions
    )
    finite = distance[cross]
    print("  cross-episode distance: p1 %.3f p10 %.3f p50 %.3f p90 %.3f" % (
        float(np.quantile(finite, 0.01)),
        float(np.quantile(finite, 0.10)),
        float(np.quantile(finite, 0.50)),
        float(np.quantile(finite, 0.90)),
    ))

    # 1. nearest neighbour outcome agreement, against random pairs and the
    #    majority-class baseline.
    predicted, nearest = knn_predict(distance, cross, y, 1, args.sigma)
    neighbour_y = np.where(
        np.isfinite(nearest[:, 0]), predicted, np.nan
    )
    valid = np.isfinite(neighbour_y)
    agree = float(np.mean((neighbour_y[valid] > 0.5) == (y[valid] > 0.5)))
    chance = base * base + (1.0 - base) * (1.0 - base)
    majority = max(base, 1.0 - base)
    print("  1. nearest neighbour outcome agreement %.3f "
          "(random %.3f, majority %.3f)" % (agree, chance, majority))

    # 2. kNN outcome prediction.
    predicted, _nearest = knn_predict(
        distance, cross, y, args.neighbors, args.sigma
    )
    knn_auc = auc(predicted, y)
    brier = float(np.mean((predicted - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    print("  2. kNN (cross-episode) AUC %.3f  Brier %.4f (base %.4f)" % (
        knn_auc, brier, brier_base
    ))

    # 3. same-action pool: does conditioning on the taken action sharpen it?
    same_pred, _ = knn_predict(
        distance, cross & same_action, y, args.neighbors, args.sigma
    )
    diff_pred, _ = knn_predict(
        distance, cross & ~same_action, y, args.neighbors, args.sigma
    )
    print("  3. kNN same-action AUC %.3f | different-action AUC %.3f" % (
        auc(same_pred, y), auc(diff_pred, y)
    ))
    print("     AUC(|action=1 - action=0| prediction) %.3f" % auc(
        np.abs(same_pred - diff_pred), y
    ))

    # 3b. time-matched control: the same neighbour pools, but restricted to
    #     states at the same phase of their episode (+-1 step).  If the
    #     geometry AUC collapses here, the kernel was only recovering time.
    progress = episode_progress(rows)
    same_time = np.abs(progress[:, None] - progress[None, :]) <= 1.0
    timed_pred, _ = knn_predict(
        distance, cross & same_time, y, args.neighbors, args.sigma
    )
    timed_same, _ = knn_predict(
        distance, cross & same_time & same_action, y, args.neighbors,
        args.sigma,
    )
    timed_diff, _ = knn_predict(
        distance, cross & same_time & ~same_action, y, args.neighbors,
        args.sigma,
    )
    print("  3b. time-matched kNN AUC %.3f | same-action %.3f | "
          "different-action %.3f" % (
              auc(timed_pred, y), auc(timed_same, y), auc(timed_diff, y)
          ))
    print("      AUC(progress alone) %.3f" % auc(progress, y))

    # 4. near-duplicate geometry: same place, different action.
    cutoff = float(np.quantile(finite, args.duplicate_quantile))
    near = np.where(cross, distance, np.inf) <= cutoff
    pairs = np.argwhere(np.triu(near, k=1))
    if len(pairs):
        left, right = pairs[:, 0], pairs[:, 1]
        same = actions[left] == actions[right]
        agreement = (y[left] > 0.5) == (y[right] > 0.5)
        print("  4. near-duplicate pairs (d<=%.3f): n=%d" % (
            cutoff, len(pairs)
        ))
        print("     outcome agreement same-action %.3f (n=%d) | "
              "diff-action %.3f (n=%d) | all %.3f" % (
                  float(agreement[same].mean()) if same.any() else float("nan"),
                  int(same.sum()),
                  float(agreement[~same].mean()) if (~same).any() else float("nan"),
                  int((~same).sum()),
                  float(agreement.mean()),
              ))
        # Same place *and* same phase: the action is the only difference.
        matched = np.abs(progress[left] - progress[right]) <= 1.0
        if matched.any():
            same_m = same[matched]
            agree_m = agreement[matched]
            print("     time-matched: n=%d | same-action %.3f (n=%d) | "
                  "diff-action %.3f (n=%d)" % (
                      int(matched.sum()),
                      float(agree_m[same_m].mean()) if same_m.any()
                      else float("nan"),
                      int(same_m.sum()),
                      float(agree_m[~same_m].mean()) if (~same_m).any()
                      else float("nan"),
                      int((~same_m).sum()),
                  ))
            flips = (y[left] != y[right]) & matched
            print("     time-matched outcome flips %.3f (n=%d)" % (
                float(np.mean(flips[matched])), int(flips.sum())
            ))
        outcome = y[left] - y[right]
        flip = float(np.mean(outcome != 0.0))
        print("     outcome flips between duplicates %.3f" % flip)
    else:
        print("  4. near-duplicate pairs: none at d<=%.3f" % cutoff)

    # 5. channel ranking against the horizon label.
    mask_table = agent._feature_mask_table(states[:, :agent.obs_dim])
    channels = agent._hazard_channels(states, mask_table)
    for name, values in channels.items():
        print("  5. channel %-10s rho(y_H) %.3f  rho(t) %.3f  mean %.4f" % (
            name, spearman(values, y), spearman(values, t),
            float(np.mean(values)),
        ))
    composed = agent.pain_signal_batch(states, mask_table)
    print("     composed     rho(y_H) %.3f  rho(t) %.3f  mean %.4f" % (
        spearman(composed, y), spearman(composed, t), float(composed.mean())
    ))

    payload = dict(
        seed=args.seed, rows=len(rows), horizon=args.horizon,
        base_rate=base, action_rate=float(actions.mean()),
        nn_agreement=agree, chance=chance, majority=majority,
        knn_auc=knn_auc, brier=brier, brier_base=brier_base,
        knn_auc_same_action=auc(same_pred, y),
        knn_auc_diff_action=auc(diff_pred, y),
        knn_auc_action_contrast=auc(np.abs(same_pred - diff_pred), y),
        knn_auc_time_matched=auc(timed_pred, y),
        knn_auc_time_matched_same_action=auc(timed_same, y),
        knn_auc_time_matched_diff_action=auc(timed_diff, y),
        auc_progress=auc(progress, y),
        duplicate_cutoff=cutoff,
        channels={
            name: dict(
                rho_horizon=spearman(values, y),
                rho_time=spearman(values, t),
                mean=float(np.mean(values)),
            )
            for name, values in channels.items()
        },
        composed=dict(
            rho_horizon=spearman(composed, y),
            rho_time=spearman(composed, t),
            mean=float(composed.mean()),
        ),
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
