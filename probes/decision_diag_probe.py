"""Decision-time diagnosis: is the prediction the planner consumes any good?

Everything here is read-only and measured on the planner's own stream:

1. World-model quality: the planner rolls the learned model forward; compare
   that roll (conditioned on the actions that were actually taken) with the
   observations that followed, at 1/2/4/8 steps, against a persistence
   baseline.  Reported as normalised RMSE and R^2 per dimension.
2. Hazard channel: the risk term the cost actually uses, scored against
   "death within the horizon" by AUC on the same decision states.
3. Stage alphabet: the learned identities' coverage of a held-out stream.
4. Stage table: top-1 agreement of the remembered successor with the one the
   stream took, and the probability it assigned to the truth.
5. Stage risk calibration: predicted two-stage deficit against the realised
   deficit of the stages that followed.

Nothing is trained and nothing is written.
"""

import argparse
import itertools

import numpy as np

from eval_protocol import parse_seed_spec
from milestone_graph import milestone_memory
from substrate_nodes import (
    LAG, _curiosity_off, _encoded_pending, _stage_evidence_for_streams,
    evaluate, fit,
)


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties so the statistic is unbiased.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if (index == len(sorted_scores)
                or sorted_scores[index] != sorted_scores[start]):
            ranks[order[start:index]] = np.mean(
                np.arange(start + 1, index + 1)
            )
            start = index
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def episodes_of(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row.get("episode", 0)), []).append(row)
    return [rows for rows in grouped.values() if len(rows) >= 3]


def roll_error(world, agent, episodes, steps):
    scales = np.asarray(agent.scales[:agent.obs_dim], dtype=float)
    errors, persists, targets, residual = [], [], [], []
    for rows in episodes:
        observations = np.asarray([row["obs"] for row in rows], dtype=float)
        histories = np.asarray([row["history"] for row in rows], dtype=float)
        for start in range(len(rows) - steps):
            state = observations[start].reshape(1, -1)
            for offset in range(steps):
                state = np.asarray(
                    world.predict(state, histories[start + offset].reshape(1, -1)),
                    dtype=float,
                )
            target = observations[start + steps]
            errors.append(((state[0] - target) / scales) ** 2)
            persists.append(((observations[start] - target) / scales) ** 2)
            targets.append(target)
            residual.append(state[0] - target)
    if not errors:
        return None
    targets = np.asarray(targets, dtype=float)
    residual = np.asarray(residual, dtype=float)
    centered = targets - targets.mean(axis=0)
    spread = np.sum(centered ** 2, axis=0)
    squared = np.sum(residual ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - squared / spread
    return dict(
        rmse=float(np.sqrt(np.asarray(errors).mean())),
        persistence=float(np.sqrt(np.asarray(persists).mean())),
        r2=float(np.nanmean(r2)),
    )


def channel_report(world, agent, episodes, horizons):
    rows = [row for episode_rows in episodes for row in episode_rows]
    if not rows:
        return None
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    encoded = np.concatenate(
        [observations, _encoded_pending(rows, agent.lag)], axis=1
    ) / agent.scales
    risk = agent.pain_signal_batch(encoded)
    mask = agent._feature_mask_table(encoded[:, :agent.obs_dim])
    value, support = agent.feature_value_batch(
        encoded, mask, return_support=True
    )
    distances = []
    for episode_rows in episodes:
        last = len(episode_rows) - 1
        distances.extend(last - index for index in range(len(episode_rows)))
    distances = np.asarray(distances, dtype=int)
    per_horizon = {}
    for horizon in horizons:
        labels = (distances <= int(horizon)).astype(int)
        per_horizon[int(horizon)] = dict(
            auc=auc(risk, labels), base=float(labels.mean())
        )
    return dict(
        per_horizon=per_horizon,
        mean=float(np.mean(risk)),
        value_std=float(np.std(value)),
        value_mean=float(np.mean(value)),
        value_support=float(np.mean(support)),
    )


def print_channels(channel, horizons):
    print("  hazard AUC by lead: %s" % "  ".join(
        "h%d %s" % (
            horizon,
            "n/a" if channel["per_horizon"][horizon]["auc"] is None
            else "%.3f" % channel["per_horizon"][horizon]["auc"],
        )
        for horizon in horizons
    ))
    print("  death base rate by lead: %s" % "  ".join(
        "h%d %.2f" % (horizon, channel["per_horizon"][horizon]["base"])
        for horizon in horizons
    ))
    print("  value channel: std %.3f  mean %.3f  support %.2f" % (
        channel["value_std"], channel["value_mean"],
        channel["value_support"],
    ))


def stage_report(world, agent, rows, label):
    episodes = episodes_of(rows)
    if not episodes:
        return None
    evidence = _stage_evidence_for_streams(episodes, world, agent)
    if evidence is None:
        return None
    boundaries_list, _signatures = evidence
    memories = []
    for rows_episode, boundaries in zip(episodes, boundaries_list):
        observations = np.asarray(
            [row["obs"] for row in rows_episode], dtype=float
        )
        memories.append(milestone_memory(
            observations, boundaries=boundaries, frame=agent.milestone_frame,
        ))
    learned = {tuple(int(v) for v in key) for key in agent.milestone_keys or ()}
    keys = [key for memory in memories for key in memory["keys"]]
    keys = [tuple(int(v) for v in key) for key in keys]
    covered = sum(1 for key in keys if key in learned)
    table = {}
    for row in agent.milestone_transitions or ():
        table.setdefault(tuple(int(v) for v in row["key_from"]), []).append(row)
    agree, mass, seen = 0, 0.0, 0
    for memory in memories:
        for index in range(len(memory["keys"]) - 1):
            source = tuple(int(v) for v in memory["keys"][index])
            actual = tuple(int(v) for v in memory["keys"][index + 1])
            options = table.get(source)
            if not options:
                continue
            total = sum(int(option["count"]) for option in options)
            best = max(options, key=lambda option: int(option["count"]))
            if tuple(int(v) for v in best["key_to"]) == actual:
                agree += 1
            share = sum(
                int(option["count"]) for option in options
                if tuple(int(v) for v in option["key_to"]) == actual
            )
            mass += share / float(total)
            seen += 1
    centers = {}
    totals = {}
    for memory in memories:
        for key, length, outcome in zip(
            memory["keys"], memory["durations"], memory["feedback"]
        ):
            stage = tuple(int(v) for v in key)
            totals[stage] = totals.get(stage, 0.0) + float(outcome)
            centers[stage] = centers.get(stage, 0.0) + float(length)
    values = {
        stage: totals[stage] / max(centers[stage], 1.0) for stage in totals
    }
    average = (
        sum(totals.values()) / max(sum(centers.values()), 1.0)
        if totals else 0.0
    )
    predicted, realized = [], []
    for memory in memories:
        for index in range(len(memory["keys"]) - 1):
            key = tuple(int(v) for v in memory["keys"][index])
            following = tuple(int(v) for v in memory["keys"][index + 1])
            predicted.append(float((agent.milestone_risks or {}).get(key, 0.0)))
            realized.append(
                1 if (values.get(following, average) < average) else 0
            )
    return dict(
        label=label,
        stages=len(keys),
        coverage=covered / max(len(keys), 1),
        transitions=seen,
        key_agreement=agree / max(seen, 1),
        key_mass=mass / max(seen, 1),
        risk_auc=auc(predicted, realized) if predicted else None,
        risk_base=float(np.mean(realized)) if realized else 0.0,
        sample=len(predicted),
    )


def channel_table(agent, episodes, leads):
    """AUC of every static association channel, by lead.

    Also decomposes the composed readout into its two steps - the weighted
    sum and the isotonic map - so "the composition loses the signal" can be
    attributed to one of them instead of being a guess.
    """
    rows = [row for episode_rows in episodes for row in episode_rows]
    observations = np.asarray([row["obs"] for row in rows], dtype=float)
    encoded = np.concatenate(
        [observations, _encoded_pending(rows, agent.lag)], axis=1
    ) / agent.scales
    mask = agent._feature_mask_table(encoded[:, :agent.obs_dim])
    channels = agent._hazard_channels(encoded, mask)
    distances = []
    for episode_rows in episodes:
        last = len(episode_rows) - 1
        distances.extend(last - index for index in range(len(episode_rows)))
    distances = np.asarray(distances, dtype=int)
    table = {}
    for name, values in channels.items():
        values = np.asarray(values, dtype=float)
        table[name] = [
            auc(values, (distances <= int(lead)).astype(int))
            for lead in leads
        ]
    composition = agent.hazard_composition
    if composition:
        raw = np.zeros(len(encoded), dtype=float)
        for name, weight in composition.items():
            column = channels.get(name)
            if column is not None and float(weight) > 0.0:
                raw = raw + float(weight) * np.asarray(column, dtype=float)
        table["composed_raw"] = [
            auc(raw, (distances <= int(lead)).astype(int))
            for lead in leads
        ]
        calibration = agent.hazard_calibration
        if calibration is not None:
            mids, rates = calibration
            mapped = np.interp(raw, mids, rates)
            table["composed_iso"] = [
                auc(mapped, (distances <= int(lead)).astype(int))
                for lead in leads
            ]
        table["_weights"] = dict(composition)
    return table


def plan_report(agent, rows, plans, horizon):
    """The search's own endpoint: predicted vs the state that followed.

    Only plans whose full horizon was actually lived are scored; the rest are
    censored by death before the endpoint, and that rate is reported on its
    own - a plan whose endpoint never happens is not a prediction error.
    """
    scales = np.asarray(agent.scales[:agent.obs_dim], dtype=float)
    errors, persists, targets, residuals = [], [], [], []
    censored = 0
    for plan in plans:
        frame = int(plan["frame"])
        if frame + horizon >= len(rows):
            censored += 1
            continue
        actual = np.asarray(rows[frame + horizon]["obs"], dtype=float)
        start = np.asarray(rows[frame]["obs"], dtype=float)
        final = np.asarray(plan["final"], dtype=float)
        errors.append(((final - actual) / scales) ** 2)
        persists.append(((start - actual) / scales) ** 2)
        targets.append(actual)
        residuals.append(final - actual)
    if not errors:
        return None
    targets = np.asarray(targets, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    spread = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
    squared = np.sum(residuals ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - squared / spread
    return dict(
        rmse=float(np.sqrt(np.asarray(errors).mean())),
        persistence=float(np.sqrt(np.asarray(persists).mean())),
        r2=float(np.nanmean(r2)),
        scored=len(errors),
        censored=censored,
    )


def endpoint_readout_report(agent, per_episode, horizon, leads):
    """Is the death readout accurate *on the states the search predicts*?

    For every plan whose endpoint was lived to, read the hazard at that
    predicted state and check whether the episode actually died within a few
    frames of it.  This is the end-to-end question the observed-state AUC
    cannot answer: the planner charges hazard on its own rollout states, which
    may sit off the observed distribution.
    """
    encoded = []
    remaining = []
    for _eval_seed, rows, plans in per_episode:
        for plan in plans:
            frame = int(plan["frame"])
            endpoint = frame + int(horizon)
            if endpoint >= len(rows):
                continue
            actions = np.asarray(plan["actions"], dtype=float).reshape(-1)
            take = actions[-agent.lag:][::-1] if agent.lag else np.zeros(0)
            if len(take) < agent.lag:
                take = np.concatenate([np.zeros(agent.lag - len(take)), take])
            encoded.append(np.concatenate([
                np.asarray(plan["final"], dtype=float), take,
            ]) / agent.scales)
            remaining.append(len(rows) - 1 - endpoint)
    if not encoded:
        return None
    encoded = np.asarray(encoded, dtype=float)
    remaining = np.asarray(remaining, dtype=int)
    agent.hazard_readout = "raw"
    raw = agent.pain_signal_batch(encoded)
    agent.hazard_readout = "calibrated"
    calibrated = agent.pain_signal_batch(encoded)
    table = {}
    for name, values in (("endpoint_raw", raw),
                         ("endpoint_iso", calibrated)):
        table[name] = [
            auc(values, (remaining <= int(lead)).astype(int))
            for lead in leads
        ]
    table["_base"] = [
        float((remaining <= int(lead)).mean()) for lead in leads
    ]
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100")
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stage", default="strength")
    parser.add_argument("--leads", default="1,2,4,8")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    eval_seeds = parse_seed_spec(args.eval_seeds)
    leads = [int(value) for value in args.leads.split(",")]

    for seed in seeds:
        world, agent = fit(
            seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
            birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
            dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
            dagger_frames=5000, dagger_horizon=args.horizon,
            step_stride=2, milestone_stage=args.stage,
        )
        training = episodes_of(agent.milestone_training_rows)
        print("== seed %d: model quality on the training stream ==" % seed)
        for steps in (1, 2, 4, 8):
            row = roll_error(world, agent, training, steps)
            print("  h %d: rmse %.3f  persistence %.3f  r2 %+.3f" % (
                steps, row["rmse"], row["persistence"], row["r2"],
            ))
        channel = channel_report(world, agent, training, leads)
        print_channels(channel, leads)
        table = channel_table(agent, training, leads)
        if table.get("_weights"):
            print("  composition weights: %s" % (table["_weights"],))
        for name, values in sorted(table.items()):
            if name.startswith("_"):
                continue
            print("  static channel %-9s AUC %s" % (
                name,
                "  ".join(
                    "h%d %s" % (lead,
                                "n/a" if value is None else "%.3f" % value)
                    for lead, value in zip(leads, values)
                ),
            ))
        report = stage_report(world, agent, agent.milestone_training_rows,
                              "training")
        if report:
            print("  stage table (in-sample): stages %d  coverage %.2f  "
                  "transitions %d  top1 %.2f  mass %.2f  risk AUC %s "
                  "(base %.2f, n %d)" % (
                      report["stages"], report["coverage"],
                      report["transitions"], report["key_agreement"],
                      report["key_mass"],
                      None if report["risk_auc"] is None
                      else round(report["risk_auc"], 3),
                      report["risk_base"], report["sample"],
                  ))
        print("== seed %d: the planner's own held-out stream ==" % seed)
        with _curiosity_off(agent):
            held_out = []
            per_episode = []
            for eval_seed in eval_seeds:
                plans = []
                _frames, _actions, transitions = evaluate(
                    seed, world, agent, cap=5000, horizon=args.horizon,
                    lag=LAG, step_stride=2,
                    env_seed=int(eval_seed) * 10000, record=True,
                    episode=int(eval_seed),
                    plan_out=plans,
                )
                held_out.extend(transitions)
                per_episode.append((eval_seed, transitions, plans))
        episodes = episodes_of(held_out)
        for steps in (1, 2, 4, 8):
            row = roll_error(world, agent, episodes, steps)
            print("  h %d: rmse %.3f  persistence %.3f  r2 %+.3f" % (
                steps, row["rmse"], row["persistence"], row["r2"],
            ))
        channel = channel_report(world, agent, episodes, leads)
        print_channels(channel, leads)
        table = channel_table(agent, episodes, leads)
        if table.get("_weights"):
            print("  composition weights: %s" % (table["_weights"],))
        for name, values in sorted(table.items()):
            if name.startswith("_"):
                continue
            print("  static channel %-9s AUC %s" % (
                name,
                "  ".join(
                    "h%d %s" % (lead,
                                "n/a" if value is None else "%.3f" % value)
                    for lead, value in zip(leads, values)
                ),
            ))
        for eval_seed, transitions, plans in per_episode:
            report = plan_report(agent, transitions, plans, args.horizon)
            if report is None:
                continue
            print("  eval %3d: search endpoint rmse %.3f  persistence %.3f  "
                  "r2 %+.3f  scored %d  censored %d" % (
                      eval_seed, report["rmse"], report["persistence"],
                      report["r2"], report["scored"], report["censored"],
                  ))
        log = getattr(agent, "dagger_log", ())
        if log and log[-1].get("ledger_report"):
            print("  ledger report: %s" % (log[-1]["ledger_report"],))
        readout = endpoint_readout_report(
            agent, per_episode, args.horizon, leads
        )
        if readout:
            print("  endpoint death base: %s" % "  ".join(
                "h%d %.2f" % (lead, base)
                for lead, base in zip(leads, readout["_base"])
            ))
            for name in ("endpoint_raw", "endpoint_iso"):
                print("  %-13s AUC %s" % (
                    name,
                    "  ".join(
                        "h%d %s" % (
                            lead,
                            "n/a" if value is None else "%.3f" % value,
                        )
                        for lead, value in zip(leads, readout[name])
                    ),
                ))
        report = stage_report(world, agent, held_out, "held-out")
        if report:
            print("  stage alphabet: stages %d  coverage %.2f" % (
                report["stages"], report["coverage"],
            ))
            print("  stage table (held-out): transitions %d  top1 %.2f  "
                  "mass %.2f  risk AUC %s (base %.2f, n %d)" % (
                      report["transitions"], report["key_agreement"],
                      report["key_mass"],
                      None if report["risk_auc"] is None
                      else round(report["risk_auc"], 3),
                      report["risk_base"], report["sample"],
                  ))


if __name__ == "__main__":
    main()
