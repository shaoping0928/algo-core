"""Why is the stage table so small?  Diagnose the segmentation, not the consumer.

For a fitted model it reports, on the training stream:

* episode-length distribution (a stage graph can only learn from episodes that
  contain more than one stage);
* the strength x signature boundary score: quantiles, non-zero share, and how
  many boundaries fire at several thresholds, pooled and per stream;
* the support/silhouette curve of the signature clustering at several rollout
  horizons, so the horizon and the cluster count are chosen from data rather
  than guessed.

Read-only.
"""

import argparse

import numpy as np

from milestone_graph import (
    silhouette,
    kmeans_labels,
    segment_prototypes,
    strength_boundaries,
    transition_signatures,
)
from substrate_nodes import LAG, _encoded_pending, _curiosity_off, fit


def stream_evidence(rows, world, agent, horizon):
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    encoded = np.concatenate(
        [observations, _encoded_pending(rows, agent.lag)], axis=1
    ) / agent.scales
    strength = np.zeros(len(encoded), dtype=float)
    for index, state in enumerate(encoded):
        node_id, _distance = agent._nearest(state)
        node = None if node_id is None else agent.graph.nodes.get(node_id)
        if node is not None:
            strength[index] = float(node.count)
    signatures = transition_signatures(
        observations, world.predict, agent.lag, horizon
    )
    return strength, signatures


def score_series(strength, signatures):
    change = np.abs(np.diff(strength, prepend=strength[0]))
    step = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
    score = np.zeros(len(strength))
    score[1:] = change[1:] * step
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,102")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizons", default="6,12,24")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    seeds = parse_seed_spec(args.seeds)
    horizons = [int(v) for v in args.horizons.split(",")]

    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=8, step_stride=2,
        )
        with _curiosity_off(agent):
            episodes = {}
            for row in agent.milestone_training_rows:
                episodes.setdefault(int(row.get("episode", 0)), []).append(row)
            usable = [rows for rows in episodes.values() if len(rows) >= 3]
            lengths = np.asarray([len(rows) for rows in usable])
            print("== seed %d: %d episodes, length median %.0f p90 %.0f "
                  "max %d ==" % (
                      seed, len(usable), np.median(lengths),
                      np.percentile(lengths, 90), lengths.max(),
                  ))
            for horizon in horizons:
                evidence = [
                    stream_evidence(rows, world, agent, horizon)
                    for rows in usable
                ]
                scores = [score_series(strength, signatures)
                          for strength, signatures in evidence]
                pooled = np.concatenate(scores)
                nonzero = float((pooled > 0).mean())
                quantiles = np.percentile(
                    pooled, [50, 90, 99]
                )
                prototypes = [
                    segment_prototypes(
                        signatures,
                        strength_boundaries(strength, signatures),
                    )
                    for strength, signatures in evidence
                ]
                points = np.vstack([p for p in prototypes if len(p)])
                curve = []
                for count in range(2, 6):
                    labels, _centers = kmeans_labels(points, count)
                    supports = np.bincount(labels, minlength=count)
                    curve.append((count, int(supports.min()),
                                  round(silhouette(points, labels), 3)))
                print("  h %2d: score q50 %.3g q90 %.3g q99 %.3g  nonzero "
                      "%.2f  silhouette %s" % (
                          horizon, quantiles[0], quantiles[1], quantiles[2],
                          nonzero, curve,
                      ))
                for k in (1.0, 2.0, 3.0):
                    shares = []
                    for strength, signatures in evidence:
                        score = score_series(strength, signatures)
                        threshold = k * float(np.quantile(score[1:], 0.90))
                        fired = int(np.sum(score > threshold))
                        shares.append(fired)
                    print("    k %.1f: boundaries %d  episodes with one %d/%d"
                          % (k, int(np.sum(shares)),
                             int(np.sum(np.asarray(shares) > 0)),
                             len(shares)))


if __name__ == "__main__":
    main()
