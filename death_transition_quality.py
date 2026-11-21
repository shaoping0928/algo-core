"""Measure the quality of the terminal transition judgment.

The planner consumes a composite:

    s --f(s, a)--> s' --T(s')--> death / safe / unknown

This probe separates the two error sources.  It trains the benchmark agent
with the 0/1 death rules attached, collects control-policy evaluation
trajectories, then branches each sampled state in the deterministic world:

- one-step: the true next state, the world-model prediction, and the rule
  and hazard reads on both;
- multi-step: first-death step under fixed continuation policies, true vs
  predicted;
- action contrast: at states where the two actions have different true
  first-death steps, does the predicted first-death rank the saving action
  above the fatal one?

No decision path is changed.

The metric core consumes only generic transition records: state, action,
next state, terminal label, and first-death step.  The benchmark world is
imported only to supply oracle branches; no task vocabulary or geometry
enters the rule learner or the quality metrics.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

import substrate_nodes as sn
from latent_world import latent
from substrate_nodes import LAG, _curiosity_off, evaluate, fit

CONFIG = dict(
    seed=100, steps=1500, epochs=1, max_nodes=1000,
    birth_threshold=0.45, dagger_rounds=1, dagger_frames=5000,
    dagger_cap=5000, dagger_horizon=8, step_stride=2,
    curiosity_weight=0.5, death_rules=True,
    death_rule_max_thresholds=256,
)


def seed_spec(text):
    seeds = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(part))
    return seeds


def encode_state(agent, observation, pending):
    observation = np.asarray(observation, dtype=float).reshape(-1)
    pending = np.asarray(pending, dtype=float).reshape(-1)[:agent.lag]
    if pending.size < agent.lag:
        pending = np.concatenate([
            pending, np.zeros(agent.lag - pending.size)
        ])
    if agent.lag:
        state = np.concatenate([observation, pending[::-1]])
    else:
        state = observation
    return state / agent.scales


def history_for(action, recent):
    values = [float(action)] + list(recent)[:LAG]
    if len(values) < LAG + 1:
        values.extend([0.0] * (LAG + 1 - len(values)))
    return np.asarray(values, dtype=float)


def true_rollout(env, action, policy, horizon):
    """First death step after `action` under a fixed continuation policy."""
    child = copy.deepcopy(env)
    child.step(int(action))
    if child.dead:
        return 0, np.asarray(child.obs(), dtype=float)
    first_obs = np.asarray(child.obs(), dtype=float)
    for step in range(1, horizon):
        child.step(int(policy(step)))
        if child.dead:
            return step, first_obs
    return -1, first_obs


def predicted_rollout(world, observation, recent, action, policy,
                      horizon, agent, rules):
    """Predicted first death step and predicted next state."""
    state = np.asarray(observation, dtype=float).reshape(-1)
    pending = list(recent)[:agent.lag]
    if len(pending) < agent.lag:
        pending.extend([0.0] * (agent.lag - len(pending)))
    history = history_for(action, pending)
    state = np.asarray(
        world.predict(state[None], history[None]), dtype=float
    )[0]
    first_state = state.copy()
    predicted = int(rules.classify(state[None])[0])
    if predicted == 1:
        return 0, first_state
    pending = [float(action)] + pending[:-1]
    for step in range(1, horizon):
        action_now = int(policy(step))
        history = history_for(action_now, pending)
        state = np.asarray(
            world.predict(state[None], history[None]), dtype=float
        )[0]
        predicted = int(rules.classify(state[None])[0])
        if predicted == 1:
            return step, first_state
        pending = [float(action_now)] + pending[:-1]
    return -1, first_state


def counts_rule(prediction, truth):
    prediction = np.asarray(prediction, dtype=int)
    truth = np.asarray(truth, dtype=int)
    answered = prediction != -1
    return dict(
        total=int(len(truth)),
        tp=int(np.sum(answered & (prediction == 1) & (truth == 1))),
        fp=int(np.sum(answered & (prediction == 1) & (truth == 0))),
        tn=int(np.sum(answered & (prediction == 0) & (truth == 0))),
        fn=int(np.sum(answered & (prediction == 0) & (truth == 1))),
        unknown=int(np.sum(~answered)),
    )


def counts_hazard(hazard, truth, threshold=0.5):
    prediction = (np.asarray(hazard) >= threshold).astype(int)
    truth = np.asarray(truth, dtype=int)
    return dict(
        total=int(len(truth)),
        tp=int(np.sum((prediction == 1) & (truth == 1))),
        fp=int(np.sum((prediction == 1) & (truth == 0))),
        tn=int(np.sum((prediction == 0) & (truth == 0))),
        fn=int(np.sum((prediction == 0) & (truth == 1))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    seeds = seed_spec(args.eval_seeds)
    world, agent = fit(**CONFIG)
    rules = agent.death_rules
    agent.death_rule_hard = False

    policies = [
        ("hold", lambda step: 0),
        ("flap", lambda step: 1),
        ("alternate", lambda step: step % 2),
    ]
    rule_true = []
    rule_pred = []
    rule_true_death = []
    rule_pred_death = []
    hazard_true = []
    hazard_pred = []
    one_step_truth = []
    state_errors = []
    first_true = {name: [] for name, _ in policies}
    first_pred = {name: [] for name, _ in policies}
    contrast = {name: dict(total=0, correct=0, ties=0) for name, _ in policies}
    rng = np.random.RandomState(args.seed + 12345)

    for eval_seed in seeds:
        env_seed = int(eval_seed) * 10000
        with _curiosity_off(agent):
            frames, actions, rows = evaluate(
                args.seed, world, agent, cap=5000, horizon=8, lag=LAG,
                step_stride=2, env_seed=env_seed, record=True, episode=0,
            )
        if frames < 2:
            continue
        points = sorted(set(
            list(np.linspace(2, frames - 2, args.branches).astype(int))
            + list(range(max(1, frames - 8), frames))
        ))
        for point in points:
            env = latent[LAG](env_seed)
            for action in actions[:point]:
                if env.dead:
                    break
                env.step(int(action))
            if env.dead:
                continue
            observation = np.asarray(env.obs(), dtype=float)
            recent = list(reversed(actions[max(0, point - LAG):point]))
            for action in (0, 1):
                truth_step, _ = true_rollout(
                    env, action, policies[0][1], args.horizon
                )
                immediate = int(truth_step == 0)
                one_step_truth.append(immediate)
                # True next state (one step).
                child = copy.deepcopy(env)
                child.step(int(action))
                true_next = np.asarray(child.obs(), dtype=float)
                pending_next = [float(action)] + recent[:LAG - 1]
                encoded_true = encode_state(agent, true_next, pending_next)
                hazard_true.append(
                    float(agent.pain_signal_batch(encoded_true[None])[0])
                )
                rule_true.append(int(rules.classify(true_next[None])[0]))
                rule_true_death.append(
                    int(rules.classify_death(true_next[None])[0])
                )
                # Predicted next state (one step through f).
                history = history_for(action, recent)
                predicted_next = np.asarray(
                    world.predict(observation[None], history[None]),
                    dtype=float,
                )[0]
                state_errors.append(predicted_next - true_next)
                encoded_pred = encode_state(
                    agent, predicted_next, pending_next
                )
                hazard_pred.append(
                    float(agent.pain_signal_batch(encoded_pred[None])[0])
                )
                rule_pred.append(
                    int(rules.classify(predicted_next[None])[0])
                )
                rule_pred_death.append(
                    int(rules.classify_death(predicted_next[None])[0])
                )
                # Multi-step first death for each fixed continuation.
                for name, policy in policies:
                    true_step, _ = true_rollout(
                        env, action, policy, args.horizon
                    )
                    pred_step, _ = predicted_rollout(
                        world, observation, recent, action, policy,
                        args.horizon, agent, rules,
                    )
                    first_true[name].append(int(true_step))
                    first_pred[name].append(int(pred_step))

    # Action contrast: compare the two actions at the same branch state.
    # Re-run the same branch collection but keep pairs together.
    paired = {name: [] for name, _ in policies}
    for eval_seed in seeds:
        env_seed = int(eval_seed) * 10000
        with _curiosity_off(agent):
            frames, actions, rows = evaluate(
                args.seed, world, agent, cap=5000, horizon=8, lag=LAG,
                step_stride=2, env_seed=env_seed, record=True, episode=0,
            )
        points = sorted(set(
            list(np.linspace(2, frames - 2, args.branches).astype(int))
            + list(range(max(1, frames - 8), frames))
        ))
        for point in points:
            env = latent[LAG](env_seed)
            for action in actions[:point]:
                if env.dead:
                    break
                env.step(int(action))
            if env.dead:
                continue
            observation = np.asarray(env.obs(), dtype=float)
            recent = list(reversed(actions[max(0, point - LAG):point]))
            for name, policy in policies:
                true_pair = []
                pred_pair = []
                for action in (0, 1):
                    true_step, _ = true_rollout(
                        env, action, policy, args.horizon
                    )
                    pred_step, _ = predicted_rollout(
                        world, observation, recent, action, policy,
                        args.horizon, agent, rules,
                    )
                    true_pair.append(int(true_step))
                    pred_pair.append(int(pred_step))
                paired[name].append((true_pair, pred_pair))

    summary = dict(
        states=len(one_step_truth),
        one_step=dict(
            terminal=int(np.sum(one_step_truth)),
            rule_true=counts_rule(rule_true, one_step_truth),
            rule_pred=counts_rule(rule_pred, one_step_truth),
            rule_true_death_only=counts_rule(
                rule_true_death, one_step_truth
            ),
            rule_pred_death_only=counts_rule(
                rule_pred_death, one_step_truth
            ),
            hazard_true=counts_hazard(hazard_true, one_step_truth),
            hazard_pred=counts_hazard(hazard_pred, one_step_truth),
            state_error_mean=float(np.mean(np.abs(state_errors))),
            state_error_max=float(np.max(np.abs(state_errors))),
            state_error_per_dim=[
                float(value) for value in np.mean(
                    np.abs(state_errors), axis=0
                )
            ],
        ),
        first_death={},
        action_contrast={},
    )
    for name, _ in policies:
        true = np.asarray(first_true[name], dtype=int)
        pred = np.asarray(first_pred[name], dtype=int)
        errors = np.abs(true - pred)
        summary["first_death"][name] = dict(
            total=int(len(true)),
            true_deaths=int(np.sum(true >= 0)),
            predicted_deaths=int(np.sum(pred >= 0)),
            exact=int(np.sum(errors == 0)),
            within_one=int(np.sum(errors <= 1)),
            false_negative=int(np.sum((true >= 0) & (pred < 0))),
            false_positive=int(np.sum((true < 0) & (pred >= 0))),
            mean_error=float(np.mean(errors)),
        )
        pairs = paired[name]
        total = 0
        correct = 0
        ties = 0
        for true_pair, pred_pair in pairs:
            if true_pair[0] == true_pair[1]:
                continue
            saving = int(np.argmax(true_pair))
            fatal = 1 - saving
            if pred_pair[saving] == pred_pair[fatal]:
                ties += 1
                continue
            total += 1
            if pred_pair[saving] > pred_pair[fatal]:
                correct += 1
        summary["action_contrast"][name] = dict(
            comparable=total, correct=correct, ties=ties,
            accuracy=(float(correct) / total if total else 0.0),
        )

    print(json.dumps(summary, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
