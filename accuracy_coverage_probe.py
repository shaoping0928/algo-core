"""Episode-held-out accuracy and calibration for stage transitions.

A stage transition is only trustworthy on the subset of data it has support
for.  This probe fits the memory on training episodes, evaluates every
validation transition, and reports support curves plus multiclass Brier/NLL
against an option-conditioned training prior.  The detector and both memories
are fitted on training episodes only. Censored validation outcomes and
unsupported predictions remain explicit rather than being treated as safe.

It also reports the frame hazard's held-out AUC by distance-to-prototype, so
the reliable safety subset is visible next to the transition subset.

Probe-only; no learner path changes.
"""
import argparse
import json
from types import SimpleNamespace

import numpy as np

from stage_experience import StageExperienceBatch
from stage_outcome_memory import StageOutcomeMemory
from substrate_nodes import LAG, _curiosity_off, evaluate, fit


def realized_outcome(stages, index):
    stage = stages[index]
    if stage.terminal:
        return (None, True, False)
    if stage.censored:
        return (None, False, True)
    if index + 1 < len(stages):
        return (stages[index + 1].key, False, False)
    return (None, False, True)


def outcome_key(row):
    return (row.get("destination"), bool(row.get("terminal", False)),
            bool(row.get("censored", False)))


def fit_heldout_memory(rows, lag, folds=5, validation_fold=4,
                       quantile=0.98, min_support=2, min_episodes=2):
    """Fit signal thresholds and outcome counts on train episodes only."""
    agent = SimpleNamespace(lag=int(lag))
    _all_data, train_batch, validation_batch = (
        StageExperienceBatch.fit_with_validation(
            rows, agent, lag=lag, quantile=quantile,
            build_demand=False, feature_source="branch",
            folds=folds, validation_fold=validation_fold,
        )
    )
    train_ids = {
        stage.episode for episode in train_batch.episodes for stage in episode
    }
    validation_ids = {
        stage.episode
        for episode in validation_batch.episodes for stage in episode
    }
    if train_ids.intersection(validation_ids):
        raise AssertionError("episode split leaked between train and validation")
    memory = StageOutcomeMemory(
        min_support=min_support, min_episodes=min_episodes,
    ).fit(train_batch.outcome_episodes())
    prior_memory = StageOutcomeMemory(
        min_support=1, min_episodes=1,
    ).fit(train_batch.outcome_episodes())
    train_rows = [
        row for row in rows if row.get("episode", 0) in train_ids
    ]
    return train_batch, validation_batch, memory, prior_memory, train_rows


def _distribution(prediction):
    return {
        outcome_key(item): float(item["probability"])
        for item in prediction.get("outcomes", ())
    }


def transition_rows(memory, episodes, prior_memory=None):
    rows = []
    for stages in episodes:
        for index, source in enumerate(stages):
            history = tuple(stage.key for stage in stages[:index + 1])
            prediction = memory.predict(history, source.action_trace)
            prior = (
                prior_memory.predict((), source.action_trace)
                if prior_memory is not None else None
            )
            realized = realized_outcome(stages, index)
            row = dict(
                episode=source.episode,
                supported=prediction.get("status") == "supported",
                support=int(prediction.get("support", 0)),
                depth=int(prediction.get("context_depth") or 0),
                episode_support=int(prediction.get("episode_support", 0)),
                realized=realized,
                censored=bool(realized[2]),
                correct=False,
                probability=0.0,
                top1=None,
                distribution=(
                    _distribution(prediction)
                    if prediction.get("status") == "supported" else None
                ),
                prior_distribution=(
                    _distribution(prior)
                    if prior is not None
                    and prior.get("status") == "supported" else None
                ),
            )
            if row["supported"]:
                outcomes = prediction["outcomes"]
                best = max(
                    outcomes,
                    key=lambda item: (
                        float(item["probability"]), repr(outcome_key(item)),
                    ),
                )
                row["top1"] = outcome_key(best)
                row["correct"] = bool(row["top1"] == realized)
                matched = [
                    float(item["probability"]) for item in outcomes
                    if outcome_key(item) == realized
                ]
                row["probability"] = matched[0] if matched else 0.0
            rows.append(row)
    return rows


def apply_option_prior_shrinkage(rows, alpha):
    """Shrink context probabilities toward the same-option training prior.

    ``alpha`` is pseudo-support measured in independent episodes, not frames.
    This is a probe readout only; it never changes the fitted memory.
    """
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("shrinkage alpha must be non-negative")
    output = []
    for row in rows:
        item = dict(row)
        context = row.get("distribution")
        prior = row.get("prior_distribution")
        if context is None or prior is None:
            item["distribution"] = None
            output.append(item)
            continue
        support = max(float(row.get("episode_support", 0)), 0.0)
        context_weight = (
            1.0 if alpha == 0.0
            else support / (support + alpha)
        )
        classes = set(context).union(prior)
        item["distribution"] = {
            outcome: (
                context_weight * float(context.get(outcome, 0.0))
                + (1.0 - context_weight)
                * float(prior.get(outcome, 0.0))
            )
            for outcome in classes
        }
        output.append(item)
    return output


def select_shrinkage_alpha(train_rows, lag, candidates=(0.0, 0.5, 1.0, 2.0,
                                                        4.0, 8.0),
                           folds=4, quantile=0.98, min_support=2,
                           min_episodes=2, min_oof_episodes=20):
    """Choose shrinkage using nested episode-held-out NLL within train rows."""
    candidates = tuple(sorted(set(float(value) for value in candidates)))
    if not candidates or any(value < 0.0 for value in candidates):
        raise ValueError("shrinkage candidates must be non-negative")
    folds = int(folds)
    if folds < 2:
        raise ValueError("shrinkage folds must be at least two")

    oof_rows = []
    completed_folds = 0
    for validation_fold in range(folds):
        try:
            (
                _train_batch, validation_batch, memory, prior_memory, _rows,
            ) = fit_heldout_memory(
                train_rows, lag=lag, folds=folds,
                validation_fold=validation_fold, quantile=quantile,
                min_support=min_support, min_episodes=min_episodes,
            )
        except ValueError:
            continue
        if not validation_batch.episodes:
            continue
        oof_rows.extend(transition_rows(
            memory, validation_batch.episodes,
            prior_memory=prior_memory,
        ))
        completed_folds += 1

    oof_episodes = sorted(
        {row["episode"] for row in oof_rows}, key=repr,
    )
    candidate_rows = []
    for alpha in candidates:
        adjusted = apply_option_prior_shrinkage(oof_rows, alpha)
        metrics = _score_distributions(adjusted, "distribution")
        eligible = metrics["episodes"] >= int(min_oof_episodes)
        candidate_rows.append(dict(
            alpha=float(alpha),
            eligible=bool(eligible and metrics["episode_macro_nll"] is not None),
            nll=metrics["nll"],
            episode_macro_nll=metrics["episode_macro_nll"],
            brier=metrics["brier"],
            episode_macro_brier=metrics["episode_macro_brier"],
            transitions=int(metrics["count"]),
            episodes=int(metrics["episodes"]),
        ))
    eligible_rows = [
        row for row in candidate_rows
        if row["eligible"] and row["episode_macro_nll"] is not None
    ]
    selected = min(
        eligible_rows,
        key=lambda row: (row["episode_macro_nll"], row["alpha"]),
    ) if eligible_rows else None
    return dict(
        selected_alpha=(None if selected is None else selected["alpha"]),
        selection_metric="training_oof_episode_macro_nll",
        minimum_oof_episodes=int(min_oof_episodes),
        oof_episodes=int(len(oof_episodes)),
        oof_episode_ids=oof_episodes,
        completed_folds=int(completed_folds),
        candidates=candidate_rows,
    )


def _multiclass_brier(distribution, realized):
    classes = set(distribution)
    classes.add(realized)
    return float(sum(
        (float(distribution.get(label, 0.0)) - float(label == realized)) ** 2
        for label in classes
    ))


def _score_distributions(rows, distribution_key):
    scored = [
        row for row in rows
        if not row["censored"] and row.get(distribution_key) is not None
    ]
    if not scored:
        return dict(count=0, episodes=0, nll=None, brier=None)
    losses = []
    by_episode = {}
    for row in scored:
        distribution = row[distribution_key]
        probability = max(
            float(distribution.get(row["realized"], 0.0)), 1e-12,
        )
        loss = dict(
            nll=-float(np.log(probability)),
            brier=_multiclass_brier(distribution, row["realized"]),
        )
        losses.append(loss)
        by_episode.setdefault(row["episode"], []).append(loss)
    return dict(
        count=int(len(scored)),
        episodes=int(len(by_episode)),
        nll=float(np.mean([item["nll"] for item in losses])),
        brier=float(np.mean([item["brier"] for item in losses])),
        episode_macro_nll=float(np.mean([
            np.mean([item["nll"] for item in episode_losses])
            for episode_losses in by_episode.values()
        ])),
        episode_macro_brier=float(np.mean([
            np.mean([item["brier"] for item in episode_losses])
            for episode_losses in by_episode.values()
        ])),
    )


def _episode_bootstrap_brier_skill(rows, samples=1000, seed=0):
    paired = {}
    for row in rows:
        if row["censored"]:
            continue
        model = row.get("distribution")
        prior = row.get("prior_distribution")
        if model is None or prior is None:
            continue
        paired.setdefault(row["episode"], []).append((
            _multiclass_brier(model, row["realized"]),
            _multiclass_brier(prior, row["realized"]),
        ))
    if len(paired) < 2:
        return dict(episodes=len(paired), low=None, high=None)
    episode_ids = list(paired)
    episode_losses = np.asarray([
        np.mean(paired[episode], axis=0) for episode in episode_ids
    ], dtype=float)
    rng = np.random.RandomState(int(seed))
    skills = []
    for _ in range(max(int(samples), 1)):
        indices = rng.randint(0, len(episode_ids), size=len(episode_ids))
        model_brier, prior_brier = episode_losses[indices].mean(axis=0)
        if prior_brier > 0.0:
            skills.append(1.0 - float(model_brier) / float(prior_brier))
    if not skills:
        return dict(episodes=len(paired), low=None, high=None)
    return dict(
        episodes=int(len(paired)),
        low=float(np.quantile(skills, 0.025)),
        high=float(np.quantile(skills, 0.975)),
    )


def calibration_report(rows, bootstrap_samples=1000, seed=0,
                       min_eval_episodes=20, min_coverage=0.1):
    """Score supported, uncensored outcomes against an option-only prior."""
    uncensored = [row for row in rows if not row["censored"]]
    supported = [row for row in uncensored if row["supported"]]
    supported_all = [row for row in rows if row["supported"]]
    prior_matched = [
        row for row in supported if row["prior_distribution"] is not None
    ]
    model_scores = _score_distributions(prior_matched, "distribution")
    prior_scores = _score_distributions(prior_matched, "prior_distribution")
    skill = None
    nll_gain = None
    if (model_scores["episode_macro_brier"] is not None
            and prior_scores["episode_macro_brier"] is not None
            and prior_scores["episode_macro_brier"] > 0.0):
        skill = 1.0 - (
            model_scores["episode_macro_brier"]
            / prior_scores["episode_macro_brier"]
        )
    if (model_scores["episode_macro_nll"] is not None
            and prior_scores["episode_macro_nll"] is not None):
        nll_gain = (
            prior_scores["episode_macro_nll"]
            - model_scores["episode_macro_nll"]
        )
    micro_skill = None
    if (model_scores["brier"] is not None
            and prior_scores["brier"] is not None
            and prior_scores["brier"] > 0.0):
        micro_skill = 1.0 - model_scores["brier"] / prior_scores["brier"]
    micro_nll_gain = None
    if model_scores["nll"] is not None and prior_scores["nll"] is not None:
        micro_nll_gain = prior_scores["nll"] - model_scores["nll"]
    ci = _episode_bootstrap_brier_skill(
        prior_matched, samples=bootstrap_samples, seed=seed,
    )
    supported_episode_count = len({
        row["episode"] for row in prior_matched
    })
    predicted_censored_mass = [
        float(sum(
            probability for outcome, probability in row["distribution"].items()
            if outcome[2]
        ))
        for row in supported_all if row["distribution"] is not None
    ]
    supported_coverage = (
        float(len(supported)) / float(len(uncensored)) if uncensored else 0.0
    )
    comparison_coverage = (
        float(len(prior_matched)) / float(len(uncensored))
        if uncensored else 0.0
    )
    required_episodes = max(int(min_eval_episodes), 2)
    enough_episodes = supported_episode_count >= required_episodes
    enough_coverage = comparison_coverage >= float(min_coverage)
    positive_skill = ci["low"] is not None and ci["low"] > 0.0
    improved_nll = nll_gain is not None and nll_gain > 0.0
    reasons = []
    if not enough_episodes:
        reasons.append("insufficient_supported_validation_episodes")
    if not enough_coverage:
        reasons.append("supported_coverage_below_threshold")
    if not positive_skill:
        reasons.append("brier_skill_ci_not_positive")
    if not improved_nll:
        reasons.append("nll_not_better_than_option_prior")
    return dict(
        validation_uncensored=int(len(uncensored)),
        validation_censored=int(len(rows) - len(uncensored)),
        supported_uncensored=int(len(supported)),
        supported_uncensored_episodes=int(supported_episode_count),
        supported_coverage=supported_coverage,
        option_prior_comparison_coverage=comparison_coverage,
        unknown_uncensored=int(len(uncensored) - len(supported)),
        unknown_uncensored_rate=(
            1.0 - supported_coverage if uncensored else 0.0
        ),
        supported_without_option_prior=int(
            len(supported) - len(prior_matched)
        ),
        endpoint_diagnostics=dict(
            supported_rows=int(len(supported_all)),
            observed_censored_fraction=(
                float(sum(row["censored"] for row in supported_all))
                / float(len(supported_all)) if supported_all else None
            ),
            predicted_censored_mass_mean=(
                float(np.mean(predicted_censored_mass))
                if predicted_censored_mass else None
            ),
            observed_terminal_fraction=(
                float(sum(bool(row["realized"][1])
                          for row in supported_all))
                / float(len(supported_all)) if supported_all else None
            ),
        ),
        model=model_scores,
        option_prior=prior_scores,
        brier_skill=skill,
        micro_brier_skill=micro_skill,
        nll_gain=nll_gain,
        micro_nll_gain=micro_nll_gain,
        brier_skill_episode_bootstrap_95ci=ci,
        gate=dict(
            passed=bool(
                enough_episodes and enough_coverage
                and positive_skill and improved_nll
            ),
            minimum_supported_episodes=int(required_episodes),
            minimum_option_prior_comparison_coverage=float(min_coverage),
            reasons=reasons,
        ),
    )


def curve(rows, key, thresholds, total):
    output = []
    realized_counts = {}
    for row in rows:
        realized_counts[row["realized"]] = (
            realized_counts.get(row["realized"], 0) + 1
        )
    base = (
        max(realized_counts.values()) / float(total)
        if realized_counts else 0.0
    )
    for threshold in thresholds:
        subset = [row for row in rows if row[key] >= threshold]
        supported = [row for row in subset if row["supported"]]
        if not supported:
            output.append(dict(
                threshold=float(threshold), count=0,
                coverage=0.0, accuracy=None, nll=None, base=base,
            ))
            continue
        accuracy = float(np.mean([row["correct"] for row in supported]))
        probabilities = np.maximum(
            np.asarray([row["probability"] for row in supported], float),
            1e-12,
        )
        nll = float(-np.log(probabilities).mean())
        output.append(dict(
            threshold=float(threshold),
            count=int(len(supported)),
            coverage=float(len(supported)) / float(total) if total else 0.0,
            accuracy=accuracy,
            nll=nll,
            base=base,
            lift=accuracy - base,
        ))
    return output


def hazard_curve(seed, world, agent, eval_seeds, cap, horizon):
    rows = []
    for eval_seed in eval_seeds:
        with _curiosity_off(agent):
            _length, _actions, transitions = evaluate(
                seed, world, agent, cap=int(cap), horizon=int(horizon),
                lag=LAG, step_stride=2, env_seed=int(eval_seed) * 10000,
                decision="search", beam_width=0, record=True,
            )
        observations = np.asarray(
            [transitions[0]["obs"]]
            + [row["next_obs"] for row in transitions],
            dtype=float,
        )
        done = np.asarray([bool(row.get("done")) for row in transitions])
        for index, row in enumerate(transitions):
            history = np.asarray(row["history"], dtype=float)
            pending = history[1:LAG + 1][::-1]
            obs_dim = int(agent.obs_dim)
            state = np.concatenate([
                observations[index][:obs_dim], pending,
            ]) / np.asarray(agent.scales, dtype=float)
            prototype, distance = agent._nearest(state)
            hazard = float(
                agent.pain_signal_batch(state.reshape(1, -1))[0]
            )
            label = bool(done[index:index + 2].any())
            rows.append((
                float(distance) if prototype is not None else np.nan,
                hazard, label,
            ))
    values = np.asarray([row[0] for row in rows], dtype=float)
    hazards = np.asarray([row[1] for row in rows], dtype=float)
    labels = np.asarray([row[2] for row in rows], dtype=bool)
    finite = np.isfinite(values)
    if not finite.any():
        return []
    quantiles = np.quantile(values[finite], [0.25, 0.5, 0.75, 0.9])
    edges = [0.0] + list(quantiles) + [float(values[finite].max()) + 1e-6]
    output = []
    for index in range(len(edges) - 1):
        mask = (
            finite & (values >= edges[index]) & (values < edges[index + 1])
        )
        if mask.sum() < 10:
            continue
        output.append(dict(
            low=float(edges[index]), high=float(edges[index + 1]),
            samples=int(mask.sum()),
            coverage=float(mask.sum()) / float(len(rows)),
            positives=int(labels[mask].sum()),
            hazard_mean=float(hazards[mask].mean()),
            auc=_auc(hazards[mask], labels[mask]),
        ))
    return output


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    sorted_values = combined[order]
    ranks = np.empty(len(combined), dtype=float)
    start = 0
    while start < len(combined):
        end = start
        while (end + 1 < len(combined)
               and sorted_values[end + 1] == sorted_values[start]):
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return float(
        (ranks[:len(positive)].sum()
         - len(positive) * (len(positive) + 1) / 2.0)
        / (len(positive) * len(negative))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+",
                        default=[120, 121, 122, 123, 124, 125])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--supports", type=int, nargs="+",
                        default=[1, 2, 3, 5, 10])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episode-supports", type=int, nargs="+",
                        default=[1, 2, 3])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fold", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-episodes", type=int, default=2)
    parser.add_argument("--min-eval-episodes", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--shrinkage-alphas", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--shrinkage-folds", type=int, default=4)
    parser.add_argument("--min-oof-episodes", type=int, default=20)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    world, agent = fit(
        args.seed, steps=int(args.steps), coverage=float(args.coverage),
        epochs=int(args.epochs), max_nodes=int(args.max_nodes),
        stage_memory=False,
    )
    (
        train_batch, validation_batch, memory, prior_memory, train_rows,
    ) = fit_heldout_memory(
        agent.milestone_training_rows, lag=agent.lag,
        folds=args.folds, validation_fold=args.validation_fold,
        quantile=args.quantile, min_support=args.min_support,
        min_episodes=args.min_episodes,
    )
    train = list(train_batch.episodes)
    validation = list(validation_batch.episodes)
    rows = transition_rows(memory, validation, prior_memory=prior_memory)
    total = len(rows)
    supported = [row for row in rows if row["supported"]]
    calibration = calibration_report(
        rows, bootstrap_samples=args.bootstrap_samples,
        seed=args.seed, min_eval_episodes=args.min_eval_episodes,
        min_coverage=args.min_coverage,
    )
    shrinkage_selection = select_shrinkage_alpha(
        train_rows, lag=agent.lag,
        candidates=args.shrinkage_alphas,
        folds=args.shrinkage_folds, quantile=args.quantile,
        min_support=args.min_support, min_episodes=args.min_episodes,
        min_oof_episodes=args.min_oof_episodes,
    )
    selected_alpha = shrinkage_selection["selected_alpha"]
    shrunk_validation = None
    if selected_alpha is not None:
        adjusted_rows = apply_option_prior_shrinkage(
            rows, selected_alpha,
        )
        shrunk_validation = calibration_report(
            adjusted_rows, bootstrap_samples=args.bootstrap_samples,
            seed=args.seed, min_eval_episodes=args.min_eval_episodes,
            min_coverage=args.min_coverage,
        )
    shrinkage_summary = dict(shrinkage_selection)
    shrinkage_summary.pop("oof_episode_ids", None)
    calibration["shrinkage"] = dict(
        train_oof=shrinkage_summary,
        outer_validation=shrunk_validation,
    )
    report = dict(
        seed=int(args.seed),
        config=dict(
            steps=int(args.steps), epochs=int(args.epochs),
            max_nodes=int(args.max_nodes), cap=int(args.cap),
            horizon=int(args.horizon),
            eval_seeds=[int(v) for v in args.eval_seeds],
            folds=int(args.folds),
            validation_fold=int(args.validation_fold),
            quantile=float(args.quantile),
            min_support=int(args.min_support),
            min_episodes=int(args.min_episodes),
            min_eval_episodes=int(args.min_eval_episodes),
            min_coverage=float(args.min_coverage),
            bootstrap_samples=int(args.bootstrap_samples),
            shrinkage_alphas=[
                float(value) for value in args.shrinkage_alphas
            ],
            shrinkage_folds=int(args.shrinkage_folds),
            min_oof_episodes=int(args.min_oof_episodes),
        ),
        episodes=dict(
            train=int(len(train)), validation=int(len(validation)),
            train_census=train_batch.snapshot(),
            validation_census=validation_batch.snapshot(),
        ),
        transitions=int(total),
        supported=int(len(supported)),
        supported_coverage=float(len(supported)) / float(total)
        if total else 0.0,
        supported_accuracy=(
            float(np.mean([row["correct"] for row in supported]))
            if supported else None
        ),
        support_curve=curve(rows, "support", args.supports, total),
        depth_curve=curve(rows, "depth", args.depths, total),
        episode_curve=curve(
            rows, "episode_support", args.episode_supports, total,
        ),
        calibration=calibration,
        hazard=hazard_curve(
            args.seed, world, agent, args.eval_seeds,
            args.cap, args.horizon,
        ),
    )
    print(
        "accuracy-coverage: seed %d | train %d / validation %d episodes | "
        "%d transitions, %d supported (%.2f), top1 %.3f"
        % (
            args.seed, len(train), len(validation), total,
            len(supported), report["supported_coverage"],
            report["supported_accuracy"] or 0.0,
        )
    )
    print("  support curve: %s" % (report["support_curve"],))
    print("  depth curve:   %s" % (report["depth_curve"],))
    print("  episode curve: %s" % (report["episode_curve"],))
    shrink_gate = (
        None if shrunk_validation is None
        else shrunk_validation["gate"]["passed"]
    )
    print(
        "  calibration raw NLL gain %s | Brier skill %s | gate %s; "
        "train-OOF alpha %s | shrunk gate %s"
        % (
            calibration["nll_gain"], calibration["brier_skill"],
            calibration["gate"]["passed"], selected_alpha, shrink_gate,
        )
    )
    print("  hazard by distance: %s" % (report["hazard"],))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("report written to %s" % args.report)


if __name__ == "__main__":
    main()
