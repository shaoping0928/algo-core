"""Do the generic context and hierarchical memory discover/adapt dynamics?

The world varies two hidden dynamics parameters (gravity and flap impulse)
per world.  The learner sees only observations, actions, next observations
and the world's feedback; it never sees either parameter.  Training covers a
grid of worlds, evaluation uses unseen values, and a mid-episode switch
tests detection plus temporary adaptation.

Arms:
    global        one shared linear model
    context       shared model + online low-rank context
    context+mem   the same, with a hierarchical-memory context prior

Nothing here names gravity or flap in the learner; those names exist only in
the world-side table.
"""
import argparse
import copy
import json

import numpy as np

from context_model import ContextualLinearModel
from hierarchical_memory import HierarchicalMemory
from latent_world import latent
from substrate_generic import LayerPair, LinearObservationModel

LAG = 3
OBS_DIM = 7

# World-side parameter grid.  These values are never shown to the learner;
# they are the hidden factors the probe varies.  Several unrelated factors
# move at once so a context dimension cannot be attributed to one of them by
# construction.
TRAIN_WORLDS = [
    (200, 0.0011, 0.017, 0.0, 0.0100),
    (201, 0.0013, 0.019, 1.5, 0.0085),
    (202, 0.0015, 0.021, 3.0, 0.0115),
    (203, 0.0017, 0.023, 0.0, 0.0085),
    (204, 0.0019, 0.017, 1.5, 0.0115),
    (205, 0.0021, 0.019, 3.0, 0.0100),
    (206, 0.0012, 0.022, 3.0, 0.0110),
    (207, 0.0018, 0.018, 1.5, 0.0090),
]
HELDOUT_WORLDS = [
    (220, 0.0012, 0.020, 0.7, 0.0095),
    (221, 0.0016, 0.018, 2.2, 0.0105),
    (222, 0.0020, 0.022, 0.4, 0.0112),
    (223, 0.0014, 0.024, 2.7, 0.0088),
]


def _behavior_action(env, layer_pair, rng):
    gains = np.abs(layer_pair.h)
    total = float(gains.sum())
    confidence = float(gains.max() / total) if total > 1e-9 else 0.0
    if total > 1e-9 and (layer_pair.t >= 20 or confidence >= 0.25):
        gain = float(layer_pair.h[int(np.argmax(gains))])
        center = float(np.clip(-layer_pair.b / gain, 0.0, 1.0))
    else:
        center = 0.1
    explore = 0.03 + 0.10 * (1.0 - confidence)
    probability = float(np.clip(
        center + rng.uniform(-explore, explore), 0.0, 1.0
    ))
    return int(rng.rand() < probability)


def collect_world(seed, gravity, flap, drag, speed, steps=1200, lag=LAG):
    rng = np.random.RandomState(seed)
    rows = []
    episode = 0
    while len(rows) < steps:
        env = latent[lag](
            seed * 10000 + episode, gravity=gravity, flap=flap,
            drag=drag, speed=speed,
        )
        layer_pair = LayerPair(k_max=8)
        layer_pair.hist = [0.0] * 8
        recent = [0.0] * lag
        while not env.dead and len(rows) < steps:
            obs = np.asarray(env.obs(), dtype=float)
            action = _behavior_action(env, layer_pair, rng)
            history = np.asarray([action] + recent[:lag], dtype=float)
            vy0 = env.vy
            env.step(action)
            next_obs = np.asarray(env.obs(), dtype=float)
            feedback = env.feedback()
            layer_pair.observe(env.vy - vy0, float(action))
            rows.append(dict(
                obs=obs, next_obs=next_obs, history=history,
                signals=np.asarray(
                    [feedback["dopamine"], feedback["pain"]], dtype=float
                ),
                done=bool(env.dead), episode=episode, world=seed,
            ))
            recent = [action] + recent[:-1]
        episode += 1
    return rows


def _flatten(streams):
    observations = np.stack([row["obs"] for stream in streams
                             for row in stream])
    histories = np.stack([row["history"] for stream in streams
                          for row in stream])
    targets = np.stack([row["next_obs"] for stream in streams
                        for row in stream])
    worlds = np.asarray([row["world"] for stream in streams
                         for row in stream])
    return observations, histories, targets, worlds


def _run_episode(env, model, memory, global_model, episode_id,
                 switch_after=None, switch_to=None, max_frames=160,
                 use_memory_prior=True):
    rng = np.random.RandomState(episode_id * 7919 + 17)
    layer_pair = LayerPair(k_max=8)
    layer_pair.hist = [0.0] * 8
    recent = [0.0] * LAG
    errors_global = []
    errors_context = []
    contexts = []
    changed = []
    steps = 0
    prior = None
    prior_precision = 0.0
    if use_memory_prior and memory is not None and memory.episodes:
        query = model._last_episode_context
        if query is not None:
            radius = memory.world_tree.merge_radius
            hit = memory.query_world(query, max_distance=radius)
            if hit is not None:
                prior = hit["leaf"].mean[:model.context_dim]
                prior_precision = (
                    min(hit["count"], 40.0) / 10.0
                    * float(np.exp(-0.5 * (hit["distance"] / radius) ** 2))
                )
    model.begin_episode(prior=prior, prior_precision=prior_precision)
    while not env.dead and steps < max_frames:
        if switch_after is not None and steps == switch_after:
            gravity, flap, drag, speed = switch_to
            env.gravity = gravity
            env.flap = flap
            env.drag = drag
            env.speed = speed
        obs = np.asarray(env.obs(), dtype=float)
        action = _behavior_action(env, layer_pair, rng)
        history = np.asarray([action] + recent[:LAG], dtype=float)
        prediction_global = global_model.predict(
            obs.reshape(1, -1), history.reshape(1, -1)
        )[0]
        prediction_context = model.predict(
            obs.reshape(1, -1), history.reshape(1, -1)
        )[0]
        vy0 = env.vy
        env.step(action)
        next_obs = np.asarray(env.obs(), dtype=float)
        feedback = env.feedback()
        layer_pair.observe(env.vy - vy0, float(action))
        errors_global.append(float(np.linalg.norm(
            next_obs - prediction_global
        )))
        errors_context.append(float(np.linalg.norm(
            next_obs - prediction_context
        )))
        update = model.observe(obs, history, next_obs)
        contexts.append(update["context"].copy())
        changed.append(update["changed"])
        if memory is not None:
            memory.observe(
                observation=obs,
                feedback=np.asarray(
                    [feedback["dopamine"], feedback["pain"]], dtype=float
                ),
                error=errors_context[-1], context=update["context"],
                done=bool(env.dead), episode=episode_id,
            )
        recent = [action] + recent[:-1]
        steps += 1
    mean_context, frame_count = model.end_episode()
    if memory is not None and env.dead:
        memory.end_episode()
    return dict(
        errors_global=errors_global, errors_context=errors_context,
        contexts=contexts, changed=changed, steps=steps,
        mean_context=mean_context, frame_count=frame_count,
    )


def _recovery_step(errors, start, baseline):
    for offset in range(start, max(start, len(errors) - 4)):
        window = float(np.mean(errors[offset:offset + 5]))
        if window <= 1.2 * baseline:
            return offset - start
    return None


def run(seed=0, train_steps=1200, episodes=4, max_frames=60,
        switch_after=12, world_radius=0.1, segment_size=800):
    del seed
    streams = [
        collect_world(world_seed, gravity, flap, drag, speed,
                      steps=train_steps)
        for world_seed, gravity, flap, drag, speed in TRAIN_WORLDS
    ]
    observations, histories, targets, _worlds = _flatten(streams)
    global_model = LinearObservationModel(OBS_DIM, LAG + 1).fit(
        observations, histories, targets
    )
    context_model = ContextualLinearModel(
        OBS_DIM, LAG + 1, context_dim=4, context_ridge=1e-2,
        window=48, change_factor=1.0, segment_size=segment_size,
    ).fit(observations, histories, targets)
    context_dim = (
        int(context_model.basis.shape[0])
        if context_model.basis is not None else 0
    )
    arm_memory = {
        "global": None, "context": None,
        "context+mem": HierarchicalMemory(
            context_dim=max(context_dim, 1), observation_dim=OBS_DIM,
            window=3, salience_z=1.0, phase_split=0.5,
            world_radius=world_radius, world_capacity=8, world_fanout=2,
            keyword_dim=1,
        ),
    }
    arm_models = {
        arm: copy.deepcopy(context_model)
        for arm in ("global", "context", "context+mem")
    }
    results = {}
    hidden_vs_context = []
    for world_index, (world_seed, gravity, flap, drag, speed) in enumerate(
        HELDOUT_WORLDS
    ):
        next_index = (world_index + 1) % len(HELDOUT_WORLDS)
        switch_to = HELDOUT_WORLDS[next_index][1:]
        arm_rows = {}
        for arm in ("global", "context", "context+mem"):
            memory = arm_memory[arm]
            model = arm_models[arm]
            errors_global = []
            errors_context = []
            switches = []
            for episode in range(episodes):
                env = latent[LAG](
                    world_seed * 10000 + episode,
                    gravity=gravity, flap=flap, drag=drag, speed=speed,
                )
                switch = switch_to if episode == 0 else None
                outcome = _run_episode(
                    env, model, memory, global_model,
                    episode_id=world_seed * 100 + episode,
                    switch_after=switch_after if switch else None,
                    switch_to=switch,
                    max_frames=max_frames,
                    use_memory_prior=(arm == "context+mem"),
                )
                errors_global.extend(outcome["errors_global"])
                errors_context.extend(outcome["errors_context"])
                if arm == "context+mem" and outcome["steps"] > 0:
                    hidden_vs_context.append(dict(
                        gravity=gravity, flap=flap, drag=drag, speed=speed,
                        context=[float(value) for value in (
                            outcome["mean_context"]
                        )],
                    ))
                if switch is not None and outcome["steps"] > switch_after:
                    pre = float(np.mean(
                        outcome["errors_context"][max(
                            0, switch_after - 20
                        ):switch_after]
                    ))
                    recovery = _recovery_step(
                        outcome["errors_context"], switch_after, pre
                    )
                    switches.append(dict(
                        pre=pre,
                        post=[
                            float(np.mean(outcome["errors_context"][
                                switch_after + offset:
                                switch_after + offset + 5
                            ]))
                            for offset in (0, 5, 10, 20)
                            if switch_after + offset < len(
                                outcome["errors_context"]
                            )
                        ],
                        recovery=recovery,
                        changes=int(sum(outcome["changed"])),
                    ))
            arm_rows[arm] = dict(
                global_error=float(np.mean(errors_global)),
                context_error=float(np.mean(errors_context)),
                switches=switches,
            )
        results[world_seed] = arm_rows
    memory = arm_memory["context+mem"]
    world_summaries = []
    for node_id, node in memory.world_tree.nodes.items():
        if node.leaf or node.keywords is None:
            continue
        expanded = memory.expand_world(node_id)
        world_summaries.append(dict(
            node=node_id, level=int(node.level),
            count=float(node.count),
            keywords=int(node.keywords.shape[0]),
            residual=float(node.residual),
            reconstruction_error=float(np.mean([
                item["error"] for item in expanded
            ])),
            keyword_persistence=memory.world_keywords(node_id),
        ))
    phase_summaries = []
    for index in range(len(memory.phases)):
        summary = memory.phase_summary(index)
        expansion = memory.expand_phase(index)
        phase_summaries.append(dict(
            phase=index, events=len(memory.phases[index].events),
            keywords=0 if summary["keywords"] is None
            else int(summary["keywords"].shape[0]),
            residual=float(summary["residual"]),
            reconstruction_error=float(np.mean([
                item["error"] for item in expansion["events"]
            ])) if expansion["events"] else 0.0,
        ))
    directions = []
    for index in range(context_dim):
        unit = np.zeros(context_dim)
        unit[index] = 1.0
        directions.append(dict(
            index=index, **memory.persistence(unit),
        ))
    registry = memory.registry.snapshot()
    levels = {
        key: value for key, value in registry.items()
        if key.startswith(("level:", "root"))
    }
    worlds_report = {
        key: value for key, value in registry.items()
        if key.startswith("world:")
    }
    # A diagnostic only: does the inferred context correlate with the
    # hidden parameters?  The learner never uses these correlations.
    correlations = {}
    if hidden_vs_context and context_dim:
        matrix = np.asarray([
            row["context"] for row in hidden_vs_context
        ])
        hidden = {
            "gravity": np.asarray([
                row["gravity"] for row in hidden_vs_context
            ]),
            "flap": np.asarray([row["flap"] for row in hidden_vs_context]),
            "drag": np.asarray([row["drag"] for row in hidden_vs_context]),
            "speed": np.asarray([row["speed"] for row in hidden_vs_context]),
        }
        for index in range(context_dim):
            column = matrix[:, index]
            if float(np.std(column)) <= 1e-9:
                continue
            for name, values in hidden.items():
                if float(np.std(values)) <= 1e-9:
                    continue
                correlations["z%d_%s" % (index, name)] = float(
                    np.corrcoef(column, values)[0, 1]
                )
    return dict(
        training=dict(
            transitions=int(len(observations)),
            worlds=len(TRAIN_WORLDS),
            context_dim=context_dim,
            discovered_groups=int(context_model.discovered_groups),
            threshold=float(context_model.change_threshold or 0.0),
        ),
        results=results,
        memory=memory.snapshot(),
        level_variables=levels,
        world_variables=worlds_report,
        world_summaries=world_summaries,
        phase_summaries=phase_summaries,
        directions=directions,
        hidden_correlations=correlations,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--switch-after", type=int, default=12)
    parser.add_argument("--world-radius", type=float, default=0.1)
    parser.add_argument("--segment-size", type=int, default=800)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = run(
        seed=args.seed, train_steps=args.train_steps,
        episodes=args.episodes, max_frames=args.max_frames,
        switch_after=args.switch_after, world_radius=args.world_radius,
        segment_size=args.segment_size,
    )
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
