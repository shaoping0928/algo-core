"""Does a generated prior improve adaptation and survival in a new world?

For each held-out world the probe runs the same warmup actions, then branches
into three arms with the same planner and node policy:

    global     context cleared: the online fast channel starts from zero
    online     the warmup's own context estimate, no memory prior
    generated  simulate every remembered regime under the warmup actions and
               pick the one whose predictions match the observed warmup;
               use that regime as the context prior

The last arm is the outward use of the summary: generation is used for
system identification, not for display.  All arms share the same environment
seed and warmup actions; only the context prior differs.
"""
import argparse
import copy
import json

import numpy as np

from context_model import ContextualLinearModel
from dynamics_context_probe import (
    HELDOUT_WORLDS,
    LAG,
    OBS_DIM,
    TRAIN_WORLDS,
    _behavior_action,
    _flatten,
    collect_world,
)
from generative_expansion import GenerativeExpander
from hierarchical_memory import HierarchicalMemory
from latent_world import latent
from substrate_generic import LayerPair
from substrate_nodes import _train, decision_action


def _node_policy(agent):
    def policy(observation, history):
        pending = np.asarray(history[1:1 + LAG], dtype=float)
        if len(pending) < LAG - 1:
            pending = np.concatenate([
                pending, np.zeros(LAG - 1 - len(pending))
            ])
        scores = {}
        for action in (0, 1):
            columns = np.concatenate([[float(action)], pending[:LAG - 1]])
            encoded = np.concatenate([
                np.asarray(observation, dtype=float), columns
            ]).reshape(1, -1) / agent.scales
            risk = float(agent.feature_risk_batch(encoded)[0])
            value = float(agent.feature_value_batch(encoded)[0])
            scores[action] = risk - agent.feature_value_weight * value
        return int(min(scores, key=lambda key: (scores[key], key)))
    return policy


class _SequencePolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def __call__(self, observation, history):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return int(action)


def _collect_streams(steps, data_seed=None):
    return [
        collect_world(
            seed, gravity, flap, drag, speed, steps=steps,
            behavior_seed=(
                None if data_seed is None
                else int(data_seed) * 1000000 + seed
            ),
        )
        for seed, gravity, flap, drag, speed in TRAIN_WORLDS
    ]


def _build(steps=1200, model_seed=200, data_seed=None):
    streams = _collect_streams(steps, data_seed=data_seed)
    observations, histories, targets, _worlds = _flatten(streams)
    context_model = ContextualLinearModel(
        OBS_DIM, LAG + 1, context_dim=4, context_ridge=1e-2,
        window=48, change_factor=1.0, segment_size=800,
    ).fit(observations, histories, targets)
    context_dim = (
        int(context_model.basis.shape[0])
        if context_model.basis is not None else 0
    )
    rows = [row for stream in streams for row in stream]
    _world, agent = _train(
        rows, seed=model_seed, epochs=2, lag=LAG, max_nodes=800,
        birth_threshold=0.45, component_kinds=(), component_min_count=8,
        max_components=0, component_weight=0.5, component_gating=False,
        memory_weight=0.1, association_rate=0.0, induce_features=True,
        precursor_labels=False, pain_horizon=None, voi_target=3,
        feature_value_weight=0.05,
    )
    memory = HierarchicalMemory(
        context_dim=max(context_dim, 1), observation_dim=OBS_DIM,
        window=3, salience_z=1.0, phase_split=0.5,
        world_radius=0.1, world_capacity=8, world_fanout=2,
        keyword_dim=1,
    )
    replay_model = copy.deepcopy(context_model)
    for stream in streams:
        for row in stream:
            update = replay_model.observe(
                row["obs"], row["history"], row["next_obs"]
            )
            memory.observe(
                observation=row["obs"], feedback=row["signals"],
                error=update["error"], context=update["context"],
                done=row["done"], episode=row["episode"],
            )
        memory.end_episode()
    return context_model, context_dim, agent, memory


def _warmup(seed, parameters, steps):
    env = latent[LAG](seed, **parameters)
    layer_pair = LayerPair(k_max=8)
    layer_pair.hist = [0.0] * 8
    rng = np.random.RandomState(seed * 37 + 11)
    actions = []
    observations = [np.asarray(env.obs(), dtype=float)]
    for _ in range(steps):
        if env.dead:
            break
        action = _behavior_action(env, layer_pair, rng)
        actions.append(action)
        vy0 = env.vy
        env.step(action)
        layer_pair.observe(env.vy - vy0, float(action))
        observations.append(np.asarray(env.obs(), dtype=float))
    return actions, observations


def _replay_warmup(seed, parameters, actions, model):
    env = latent[LAG](seed, **parameters)
    recent = [0.0] * LAG
    for action in actions:
        if env.dead:
            break
        observation = np.asarray(env.obs(), dtype=float)
        history = np.asarray([action] + recent[:LAG], dtype=float)
        env.step(action)
        next_observation = np.asarray(env.obs(), dtype=float)
        model.observe(observation, history, next_observation)
        recent = [float(action)] + recent[:-1]
    return env, recent


def _candidates(memory, context_dim):
    result = []
    if context_dim <= 0:
        return result
    for leaf_id in memory.world_tree.leaves:
        leaf = memory.world_tree.nodes.get(leaf_id)
        if leaf is None or leaf.count <= 0.0:
            continue
        result.append((
            leaf.node_id, leaf.mean[:context_dim].copy(), leaf.count
        ))
    return result


def _generated_choice(expander, candidates, actions, observations, agent):
    target = np.asarray(observations[1:], dtype=float)
    if not candidates or not target.size:
        return None
    ranked = []
    for node_id, context, count in candidates:
        policy = _SequencePolicy(actions)
        flow = expander.expand(
            context, start_obs=observations[0], horizon=len(target),
            policy=policy,
        )
        if len(flow) < len(target):
            continue
        predictions = np.stack([
            row["prediction"] for row in flow[:len(target)]
        ])
        distance = float(np.mean(np.linalg.norm(
            predictions - target, axis=1
        )))
        # Per-node prediction ability inside this candidate's simulation:
        # a candidate whose predicted roots rarely match the observed
        # roots is less trustworthy even if its numeric distance is small.
        hits = {}
        counts = {}
        for predicted, actual in zip(predictions, target):
            predicted_roots = set(agent.prediction_nodes(predicted))
            actual_roots = set(agent.prediction_nodes(actual))
            for root in predicted_roots:
                counts[root] = counts.get(root, 0) + 1
                if root in actual_roots:
                    hits[root] = hits.get(root, 0) + 1
        if counts:
            credibility = float(np.mean([
                (hits.get(root, 0) + 1.0) / (counts[root] + 2.0)
                for root in counts
            ]))
        else:
            credibility = 0.5
        adjusted = distance * (1.0 + (1.0 - credibility))
        ranked.append((
            adjusted, distance, credibility, node_id, context, count
        ))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    best = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else best[0]
    return dict(
        distance=best[0], raw_distance=best[1], credibility=best[2],
        node=best[3], context=best[4], support=best[5], second=second,
        margin=(second - best[0]) / second if second > 0.0 else 0.0,
    )


def _plan(env, model, agent, recent, max_frames, horizon=8):
    prefix = np.asarray(recent[:LAG], dtype=float)
    errors = []
    agent.reset_prediction_stats()
    while not env.dead and env.t < max_frames:
        observation = np.asarray(env.obs(), dtype=float)
        action = decision_action(
            env, model, agent, prefix, horizon=horizon, lag=LAG
        )
        history = np.asarray([action] + recent[:LAG], dtype=float)
        prediction = model.predict(
            observation.reshape(1, -1), history.reshape(1, -1)
        )[0]
        agent.set_prediction(prediction)
        env.step(action)
        next_observation = np.asarray(env.obs(), dtype=float)
        agent.observe_prediction(next_observation)
        errors.append(float(np.linalg.norm(
            next_observation - prediction
        )))
        model.observe(observation, history, next_observation)
        prefix = np.concatenate([[float(action)], prefix[:-1]])
        recent = [float(action)] + recent[:-1]
    return env.t, errors, agent.prediction_summary()


def run(train_steps=1200, warmup=10, max_frames=120, episodes=2,
        horizon=8, model_seed=200, data_seed=None):
    context_model, context_dim, agent, memory = _build(
        train_steps, model_seed=model_seed, data_seed=data_seed
    )
    expander = GenerativeExpander(
        context_model, memory=memory, policy=_node_policy(agent),
        horizon=warmup, phase_len=4,
    )
    candidates = _candidates(memory, context_dim)
    results = {}
    for world_index, (world_seed, gravity, flap, drag, speed) in enumerate(
        HELDOUT_WORLDS
    ):
        parameters = dict(
            gravity=gravity, flap=flap, drag=drag, speed=speed
        )
        rows = {}
        for episode in range(episodes):
            seed = world_seed * 10000 + episode
            actions, observations = _warmup(seed, parameters, warmup)
            arms = {}
            for arm in ("global", "online", "generated"):
                model = copy.deepcopy(context_model)
                env, recent = _replay_warmup(
                    seed, parameters, actions, model
                )
                prior_info = None
                if arm == "global":
                    model.reset_context()
                elif arm == "generated":
                    choice = _generated_choice(
                        expander, candidates, actions, observations, agent
                    )
                    if choice is not None:
                        # Use the generated regime only when the simulation
                        # both distinguishes it from the runner-up and its
                        # predicted roots are credible; otherwise fall back
                        # to the warmup's own context.
                        accepted = bool(
                            choice["margin"] >= 0.05
                            and choice["credibility"] >= 0.4
                        )
                        if accepted:
                            model.apply_prior(
                                choice["context"],
                                precision=min(
                                    choice["support"], 20.0
                                ) / 10.0,
                            )
                            prior_info = dict(
                                node=int(choice["node"]),
                                distance=float(choice["distance"]),
                                raw_distance=float(choice["raw_distance"]),
                                credibility=float(choice["credibility"]),
                                support=float(choice["support"]),
                                margin=float(choice["margin"]),
                                accepted=True,
                            )
                        else:
                            prior_info = dict(
                                node=int(choice["node"]),
                                distance=float(choice["distance"]),
                                raw_distance=float(choice["raw_distance"]),
                                credibility=float(choice["credibility"]),
                                support=float(choice["support"]),
                                margin=float(choice["margin"]),
                                accepted=False,
                                fallback=(
                                    "ambiguous"
                                    if choice["margin"] < 0.05
                                    else "low_credibility"
                                ),
                            )
                frames, errors, prediction = _plan(
                    env, model, agent, recent, max_frames,
                    horizon=horizon,
                )
                warm = len(actions)
                arms[arm] = dict(
                    frames=int(frames),
                    post_frames=int(max(0, frames - warm)),
                    error_first=float(np.mean(errors[:10]))
                    if errors else 0.0,
                    error_all=float(np.mean(errors)) if errors else 0.0,
                    prior=prior_info,
                    prediction=prediction,
                )
            rows[episode] = arms
        results[world_seed] = rows
    return dict(
        context_dim=int(context_dim),
        model_seed=int(model_seed),
        warmup=int(warmup),
        candidates=len(candidates),
        results=results,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--model-seed", type=int, default=200)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = run(
        train_steps=args.train_steps, warmup=args.warmup,
        max_frames=args.max_frames, episodes=args.episodes,
        horizon=args.horizon, model_seed=args.model_seed,
        data_seed=(args.data_seed if args.data_seed > 0 else None),
    )
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
