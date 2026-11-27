"""Can a frame-horizon planner see a stage?  Measure the reach, don't assume.

The stage table prices whole stages; the planner rolls a handful of frames.
This probe asks whether that roll can even leave the stage it started in:

* for sampled training states, map the state to its stage (nearest known
  boundary state),
* roll every action sequence the planner can consider over its own horizon,
* map each endpoint back to a stage,
* report how often all endpoints share one stage, how many distinct stages
  the endpoints span, and the spread of the per-stage risk they would be
  charged.

If the endpoints never span a stage, a stage-level charge is a constant
across candidates and cannot change the decision - which is a fact about the
time scales, not about the weight.

Read-only.
"""

import argparse
import itertools

import numpy as np

from eval_protocol import parse_seed_spec
from milestone_graph import nearest_stage_key, transition_signatures
from substrate_generic import history_features
from substrate_nodes import LAG, fit


def roll(world, observation, pending, sequence, horizon):
    state = np.asarray(observation, dtype=float).reshape(1, -1)
    batch = np.asarray([sequence], dtype=float)
    for step in range(int(horizon)):
        history = history_features(
            batch, step, pending, action_dim=len(pending) + 1
        )
        state = np.asarray(world.predict(state, history), dtype=float)
    return state[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--stage", default="strength")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)

    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon,
            step_stride=2, milestone_stage=args.stage,
        )
        frame = agent.milestone_frame
        states = agent.milestone_stage_states
        keys = agent.milestone_stage_keys
        risks = agent.milestone_risks or {}
        if frame is None or len(states) == 0:
            print("== seed %d: no stage index ==" % seed)
            continue
        rows = list(agent.milestone_training_rows)
        step = max(len(rows) // max(args.samples, 1), 1)
        rows = rows[::step][:args.samples]
        # The planner's own candidate set at step_stride=2: each action is
        # held for two frames, so the search enumerates horizon/2 choices.
        blocks = list(itertools.product(
            (0.0, 1.0), repeat=max(args.horizon // 2, 1)
        ))
        sequences = [
            np.repeat(np.asarray(block, dtype=float), 2)
            for block in blocks
        ]
        same, spanned, spreads, distances = [], [], [], []
        clusters = agent.milestone_clusters
        signature_mode = bool(
            clusters and np.asarray(clusters.get("centers", ())).size
            and clusters.get("horizon")
        )
        cluster_spanned, cluster_spreads = [], []
        for row in rows:
            observation = np.asarray(row["obs"], dtype=float)
            history = np.asarray(row.get("history", ()), dtype=float).reshape(-1)
            pending = history[1:1 + agent.lag]
            if len(pending) < agent.lag:
                pending = np.concatenate([
                    np.zeros(agent.lag - len(pending)), pending,
                ])
            here = nearest_stage_key(states, keys, frame, observation)
            endpoints = [
                roll(world, observation, pending, sequence, args.horizon)
                for sequence in sequences
            ]
            endpoint_keys = [
                nearest_stage_key(states, keys, frame, endpoint)
                for endpoint in endpoints
            ]
            values = np.asarray([
                float(risks.get(key, 0.0)) for key in endpoint_keys
            ])
            same.append(float(np.mean([key == here for key in endpoint_keys])))
            spanned.append(len(set(endpoint_keys)))
            spreads.append(float(values.max() - values.min()))
            offsets = np.linalg.norm(
                (np.asarray(endpoints) - np.asarray(observation)) / np.asarray(
                    frame.get("sigma", np.ones(len(observation)))
                ), axis=1,
            )
            distances.append(float(offsets.mean()))
            if signature_mode:
                centers = np.asarray(clusters["centers"], dtype=float)
                horizon = int(clusters["horizon"])
                endpoint_sig = transition_signatures(
                    np.asarray(endpoints, dtype=float), world.predict,
                    agent.lag, horizon,
                )
                labels = np.argmin(np.linalg.norm(
                    endpoint_sig[:, None, :] - centers[None, :, :], axis=2
                ), axis=1)
                cluster_spanned.append(len(set(int(v) for v in labels)))
                local = np.asarray([
                    float(risks.get((int(label),), 0.0)) for label in labels
                ])
                cluster_spreads.append(float(local.max() - local.min()))
        print("== seed %d: %d states, horizon %d, %d sequences, stage risk "
              "range %.2f..%.2f ==" % (
                  seed, len(rows), args.horizon, len(sequences),
                  min(risks.values()), max(risks.values()),
              ))
        print("  endpoints sharing the current stage: %.2f" % np.mean(same))
        print("  distinct endpoint stages: mean %.2f  max %d" % (
            np.mean(spanned), int(np.max(spanned)),
        ))
        print("  endpoint risk spread: mean %.4f  nonzero %.2f" % (
            np.mean(spreads), float(np.mean(np.asarray(spreads) > 1e-9)),
        ))
        print("  endpoint travel (sigmas): mean %.2f" % np.mean(distances))
        if signature_mode:
            print("  signature clusters: distinct endpoint labels mean %.2f "
                  " max %d; risk spread mean %.4f  nonzero %.2f" % (
                      np.mean(cluster_spanned), int(np.max(cluster_spanned)),
                      np.mean(cluster_spreads),
                      float(np.mean(np.asarray(cluster_spreads) > 1e-9)),
                  ))


if __name__ == "__main__":
    main()
