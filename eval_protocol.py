"""Held-out evaluation helpers shared by the substrate experiments.

The protocol is deliberately task-agnostic: it only parses seed sets and
summarises numbers, never touches the environment.  Training seeds and
evaluation seeds are supplied separately so a run can be scored on worlds it
never trained on.

The summary keeps the per-episode rows, a per-cluster view (one cluster per
trained model) and a bootstrap interval.  With few seeds the interval is a
description of the sample, not a significance test.
"""
import numpy as np


def parse_seed_spec(spec):
    """Parse ``"120-124,130"`` into a sorted unique list of seeds."""
    if spec is None:
        return []
    if isinstance(spec, (list, tuple, np.ndarray)):
        values = [int(value) for value in spec]
    else:
        text = str(spec).strip()
        if not text:
            return []
        values = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                raise ValueError("empty seed in %r" % spec)
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise ValueError("descending seed range %r" % part)
                values.extend(range(start, end + 1))
            else:
                values.append(int(part))
    if any(value < 0 for value in values):
        raise ValueError("seeds must be non-negative")
    return sorted(set(values))


def bootstrap_mean_ci(values, samples=2000, confidence=0.95, seed=0):
    """Percentile bootstrap interval for the mean.

    ``values`` should already be one number per independent cluster, not one
    number per evaluation episode.  Repeating a model over many worlds does
    not make those worlds independent of the model.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if int(samples) < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(
        0, values.size, size=(int(samples), values.size)
    )
    means = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def summarize(values):
    """Mean, median, IQR, range and a bootstrap interval for one sample."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return dict(
            n=0, mean=float("nan"), median=float("nan"),
            q1=float("nan"), q3=float("nan"), iqr=float("nan"),
            min=float("nan"), max=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"),
        )
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    low, high = bootstrap_mean_ci(values)
    return dict(
        n=int(values.size),
        mean=float(values.mean()),
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        min=float(values.min()),
        max=float(values.max()),
        ci_low=low,
        ci_high=high,
    )


def group_values(rows, value="frames", group="train_seed"):
    """Group row values, preserving a stable key order."""
    groups = {}
    for row in rows:
        groups.setdefault(row[group], []).append(row[value])
    return {key: groups[key] for key in sorted(groups)}


def group_summaries(rows, value="frames", group="train_seed"):
    """One summary per group key."""
    return {
        str(key): summarize(values)
        for key, values in group_values(rows, value, group).items()
    }


def cluster_summary(rows, value="frames", cluster="train_seed",
                    samples=2000, seed=0):
    """Overall summary plus a bootstrap interval over cluster means.

    The point estimate is the mean of all evaluation episodes; the interval
    resamples trained models, because episodes produced by the same model are
    not independent evidence about how a different training seed would do.
    """
    values = [float(row[value]) for row in rows]
    groups = group_values(rows, value, cluster)
    means = [float(np.mean(group)) for group in groups.values()]
    result = summarize(values)
    low, high = bootstrap_mean_ci(means, samples=samples, seed=seed)
    result.update(
        clusters=len(groups),
        cluster_mean=float(np.mean(means)) if means else float("nan"),
        cluster_ci_low=low,
        cluster_ci_high=high,
    )
    return result


def paired_delta(left, right, key="train_seed", value="frames"):
    """Per-training-seed mean difference (left - right) for two arms."""
    left_groups = group_values(left, value, key)
    right_groups = group_values(right, value, key)
    keys = sorted(set(left_groups) & set(right_groups))
    deltas = [
        float(np.mean(left_groups[item]) - np.mean(right_groups[item]))
        for item in keys
    ]
    return dict(
        pairs=len(deltas),
        keys=keys,
        deltas=deltas,
        mean=float(np.mean(deltas)) if deltas else float("nan"),
        summary=summarize(deltas),
    )
