"""Time-independent milestones read from the observation stream alone.

A phase reset - one cycle of the world's recurring structure ending and the
next beginning - is a *duration-free* transition unit: it counts progress, not
frames, exactly what a decision layer needs when the same passage can take
half as long or twice as long depending on the world's time scale.

The detector is task-neutral.  It sees only the observation rows and never a
geometry, an action alphabet or a terminal flag.  It works on the leaky
baseline of the observation (the same ``capacitor`` filter as
``node_components``) and calls a reset at the frames where that baseline
steps.

Why those two details matter (measured, 2026-11-22):

* The scale must come from a **high quantile** of the baseline jump, not the
  median.  A reset is rare: its amplitude lives in the 90th-percentile-to-max
  tail, while the median of the jump series is set by the ordinary drift and
  sits an order of magnitude below it.  A median-based threshold lands in the
  middle of the reset's own decay and fires several times per reset.
* The scale must be commensurate across time scales, so it is taken from the
  observed stream rather than fixed.  The ordinary-frame dynamics are
  invariant to the world's time scale while the reset amplitude scales with
  it, so one relative rule covers every scale.
* The scale must be taken **per coordinate**.  A raw max over coordinates is
  dominated by whichever coordinate ramps fastest: an unbounded timer has a
  constant leaky-baseline step, which pins every frame's score at a constant
  and makes the threshold a value nothing can ever exceed.  Measured on this
  task's own stream (2026-11-22), the raw statistic read exactly 1.0000 at
  every single frame and found **zero** resets; dividing each coordinate by
  its own ordinary step is what makes a reset visible at all.
* A **non-maximum suppression** window is required because the leaky baseline
  smears one reset over ``1 / beta`` frames; without it each reset is counted
  three to five times.

An action-driven transient is *not* the problem it looks like: measured, it
perturbs one coordinate about 1.2x the ordinary frame, far below a reset, so
the tail statistic separates the two without ever looking at the action.

One scale constraint is intrinsic.  The baseline lags a steady drift of `d`
per frame by `d / beta`, so a drift contributes a jump of about `d` while a
reset of amplitude `A` contributes about `beta * A`; the rule separates them
when `beta * A > k * d`.  A reset that is small against the stream's own
per-frame drift is not a phase boundary by this definition and will not be
found - which is the intended reading, not a failure.

`milestone_memory` builds the next level up: a count-based memory whose nodes
are milestones and whose edges are the transitions between them.  A milestone
is identified by the quantile bin of the state it starts in, which is
task-neutral and invariant to the world's time scale; its length is recorded
as a distribution rather than assumed.
"""
from typing import Optional

import numpy as np

DEFAULT_BETA = 0.2
DEFAULT_K = 3.0
DEFAULT_QUANTILE = 0.90
DEFAULT_NMS = None  # derived from beta: the baseline smears a reset over 1/beta frames


def _leaky_steps(observations, beta: float = DEFAULT_BETA) -> np.ndarray:
    """Per-frame absolute leaky-baseline step, one column per coordinate.

    The leaky baseline is ``s <- s + beta * (x - s)``, so its step is
    ``beta * (x - s)``.  Frame 0 has no step and reads zero.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2:
        raise ValueError("observations must be a 2-D (frames, dim) array")
    if not 0.0 < float(beta) <= 1.0:
        raise ValueError("beta must be in (0, 1]")
    steps = np.zeros_like(x)
    if len(x) < 2:
        return steps
    beta = float(beta)
    state = x[0].copy()
    for index in range(1, len(x)):
        step = beta * (x[index] - state)
        state = state + step
        steps[index] = np.abs(step)
    return steps


def slow_baseline_jumps(observations, beta: float = DEFAULT_BETA) -> np.ndarray:
    """Per-frame magnitude of the leaky-baseline step, max over dimensions.

    This is the raw, unnormalised magnitude.  It is a faithful read of "how
    far the stream moved this frame", but it is **not** what a boundary is
    measured against - see :func:`boundary_scores` for why.
    """
    return _leaky_steps(observations, beta=beta).max(axis=1)


def boundary_scale(observations, beta: float = DEFAULT_BETA,
                   quantile: float = DEFAULT_QUANTILE) -> np.ndarray:
    """Each coordinate's ordinary step scale: the quantile of its ``|step|``.

    Normalising a coordinate by its own ordinary step is what makes the
    detector blind to a coordinate that merely *ramps*.  An unbounded timer
    has a constant leaky-baseline step, so a raw max over coordinates becomes
    a constant and the threshold is a value nothing can ever exceed: measured
    on this task's own stream (2026-11-22), every frame scored exactly 1.0000
    and **not one reset was found**.  Per-coordinate scale removes that.

    A coordinate with no ordinary motion (flat, or moved only by an isolated
    transient) has a zero quantile; its own largest step is then the only
    scale it has, so a lone transient scores 1 and stays below the threshold.
    """
    x = np.asarray(observations, dtype=float)
    steps = _leaky_steps(x, beta=beta)
    if len(steps) < 2:
        return np.zeros(x.shape[1])
    body = steps[1:]
    scale = np.asarray([
        float(np.quantile(body[:, d], float(quantile)))
        for d in range(body.shape[1])
    ])
    peak = body.max(axis=0)
    return np.where(scale > 0.0, scale, peak)


def boundary_scores(observations, beta: float = DEFAULT_BETA,
                    scale=None) -> np.ndarray:
    """The phase-reset score of each frame: the max over coordinates of
    ``|step| / scale``.

    A reset is a frame where *some* coordinate moves far relative to how that
    coordinate ordinarily moves.  That is dimensionless and comparable across
    coordinates and across time scales, which the raw magnitude is not.
    """
    x = np.asarray(observations, dtype=float)
    if scale is None:
        scale = boundary_scale(x, beta=beta)
    scale = np.asarray(scale, dtype=float).reshape(-1)
    steps = _leaky_steps(x, beta=beta)
    scores = np.zeros(len(x))
    if len(steps) == 0 or scale.size == 0:
        return scores
    usable = scale > 0.0
    if usable.any():
        scores = (steps[:, usable] / scale[usable]).max(axis=1)
    scores[0] = 0.0
    return scores


def pooled_boundary_model(streams, beta: float = DEFAULT_BETA,
                          k: float = DEFAULT_K,
                          quantile: float = DEFAULT_QUANTILE):
    """Calibrate ``(scale, threshold)`` on several streams at once.

    A training run is many episodes.  Episode *seams* are discontinuities that
    are not phase resets, so the streams must not be stitched together and the
    statistics taken over the join: the seam would inflate the high quantile
    and can suppress every real reset.  The steps and scores are therefore
    pooled from the *within-stream* frames only.

    Returns ``dict(beta, scale, threshold)``; a causal consumer freezes it.
    """
    streams = [np.asarray(s, dtype=float) for s in streams
               if np.asarray(s, dtype=float).size]
    streams = [s for s in streams if len(s) >= 2]
    if not streams:
        return dict(beta=float(beta), scale=np.zeros(0),
                    threshold=float("inf"))
    steps = np.concatenate([_leaky_steps(s, beta=beta)[1:] for s in streams],
                           axis=0)
    scale = np.asarray([
        float(np.quantile(steps[:, d], float(quantile)))
        for d in range(steps.shape[1])
    ])
    peak = steps.max(axis=0)
    scale = np.where(scale > 0.0, scale, peak)
    scores = np.concatenate(
        [boundary_scores(s, beta=beta, scale=scale)[1:] for s in streams]
    )
    threshold = (float(k) * float(np.quantile(scores, float(quantile)))
                 if len(scores) else float("inf"))
    return dict(beta=float(beta), scale=scale, threshold=threshold)


def boundary_threshold(observations, beta: float = DEFAULT_BETA,
                       k: float = DEFAULT_K,
                       quantile: float = DEFAULT_QUANTILE,
                       scale=None) -> float:
    """The boundary threshold a stream calibrates: ``k * quantile(scores)``.

    Freezing this value (together with the scale it was read with) lets the
    same rule run causally at decision time, when the whole stream - and
    therefore the quantile - is not available.
    """
    scores = boundary_scores(observations, beta=beta, scale=scale)
    if len(scores) < 3:
        return float("inf")
    return float(k) * float(np.quantile(scores[1:], float(quantile)))


def segment_boundaries(observations, beta: float = DEFAULT_BETA,
                       k: float = DEFAULT_K,
                       quantile: float = DEFAULT_QUANTILE,
                       nms: int = DEFAULT_NMS,
                       threshold=None, scale=None) -> np.ndarray:
    """Indices of the frames where a slow variable resets.

    Boundaries are the local maxima of :func:`boundary_scores` above
    ``k * quantile(scores)``; `nms` is the half-width of the non-maximum
    suppression window.  A constant stream returns no boundaries (the scale is
    zero and nothing exceeds it).  Passing `threshold` (with the `scale` it was
    read against) skips the calibration and applies an offline threshold
    causally.
    """
    if float(k) <= 0.0:
        raise ValueError("k must be positive")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    nms = int(round(1.0 / float(beta))) if nms is None else int(nms)
    if nms < 0:
        raise ValueError("nms must be non-negative")
    scores = boundary_scores(observations, beta=beta, scale=scale)
    if len(scores) < 3:
        return np.zeros(0, dtype=int)
    # The first frame is a cold-start artifact of the baseline; it never
    # participates in the scale or in the selection.
    if threshold is None:
        threshold = float(k) * float(np.quantile(scores[1:], float(quantile)))
    fired = np.flatnonzero(scores > float(threshold))
    if nms == 0 or len(fired) == 0:
        return fired.astype(int)
    keep = []
    for index in fired:
        low = max(0, index - nms)
        high = min(len(scores), index + nms + 1)
        if scores[index] >= scores[low:high].max() - 1e-12:
            keep.append(int(index))
    return np.asarray(keep, dtype=int)


def milestone_statistics(observations, feedback=None,
                         boundaries: Optional[np.ndarray] = None,
                         **segment_kwargs):
    """Duration and accumulated feedback of each stretch between boundaries.

    `feedback` is an optional ``(frames, channels)`` array of the world's own
    feedback channels (the only thing the learner is allowed to read).  The
    stretches tile the whole stream - the leading stretch before the first
    boundary, every stretch between two boundaries, and the trailing stretch
    to the end - so the durations always sum to the number of frames.
    Durations are recorded as a distribution; nothing here assumes a horizon.
    """
    x = np.asarray(observations, dtype=float)
    if boundaries is None:
        boundaries = segment_boundaries(x, **segment_kwargs)
    boundaries = np.asarray(boundaries, dtype=int)
    edges = np.concatenate([[0], boundaries, [len(x)]])
    durations = np.diff(edges)
    result = dict(
        boundaries=boundaries,
        durations=durations.astype(int),
        count=int(len(edges) - 1),
    )
    if feedback is not None:
        values = np.asarray(feedback, dtype=float)
        if len(values) != len(x):
            raise ValueError(
                "feedback must have one row per observation"
            )
        totals = np.zeros((len(edges) - 1, values.shape[1]))
        for index in range(len(edges) - 1):
            totals[index] = values[edges[index]:edges[index + 1]].sum(axis=0)
        result["feedback"] = totals
    return result


def quantile_cuts(observations, bins: int = 2):
    """Per-dimension quantile cut points taken from the stream itself.

    Using the stream's own marginal distribution makes the identity invariant
    to any monotone rescaling of a dimension - including the rescaling a
    change of the world's time scale produces.  Dimensions whose value at the
    boundary never moves carry no levels and contribute no bits.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2:
        raise ValueError("observations must be a 2-D (frames, dim) array")
    bins = int(bins)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    inner = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    if len(inner) == 0:
        return np.zeros((x.shape[1], 0))
    return np.asarray([np.quantile(x[:, d], inner)
                       for d in range(x.shape[1])])


def support_bins(projection, min_support: int = 3, max_bins: int = 6):
    """Largest bin count that still gives every bin ``min_support`` members.

    A fixed two-bin identity makes the transition table and the coverage
    metric nearly vacuous; a fixed four-bin identity fragments the key set.
    The support gate chooses from the data instead: more identity resolution
    only while each identity is still observed often enough to estimate.
    """
    projection = np.asarray(projection, dtype=float).reshape(-1)
    if projection.size == 0:
        return np.zeros(0), 1
    default = np.quantile(projection, [0.5])
    if len(np.unique(projection)) < 2:
        return np.zeros(0), 1
    best_cuts, best_bins = default, 2
    for count in range(2, int(max_bins) + 1):
        inner = np.linspace(0.0, 1.0, count + 1)[1:-1]
        cuts = np.quantile(projection, inner)
        keys = np.searchsorted(cuts, projection)
        counts = np.bincount(keys, minlength=count)
        if len(counts) == count and counts.min() >= int(min_support):
            best_cuts, best_bins = cuts, count
    return np.asarray(best_cuts, dtype=float), int(best_bins)


def boundary_axis(streams, boundaries_list=None, bins=None, axes: int = 2,
                  min_support: int = 3, max_bins: int = 6,
                  clock_r2: float = 0.98, **segment_kwargs):
    """The identity's frame: time-orthogonal directions of the boundary states.

    A per-coordinate bucket is a *product* of per-dimension levels, so it
    behaves like a fingerprint rather than a concept.  Projecting the boundary
    states onto their dominant directions and binning that *scalar* gives a
    reusable concept.

    Two corrections over the first version, both measured (2026-11-22):

    * **The directions are orthogonalised against the clock.**  A coordinate
      whose value progress explains almost perfectly (R² at least
      ``clock_r2``) is a clock, not a concept - the same milestone would get
      different names early and late in an episode.  Only such coordinates
      enter the time direction; a coordinate that merely drifts with time
      still carries the concept and is kept.  This is a property of the
      stream, not of the task: no coordinate is named.
    * **The bin count is chosen by support, not fixed at two.**  Two bins
      make the transition table (at most four edges) and the coverage metric
      vacuous; the support gate raises the resolution only while every bin is
      still observed often enough.

    Returns ``dict(axis, axes, center, sigma, scale, cuts, bins,
    time_corr)``; read states through it with :func:`frame_project` /
    :func:`frame_project_axes` / :func:`frame_keys`.
    """
    streams = [np.asarray(s, dtype=float) for s in streams]
    dims = streams[0].shape[1] if streams else 0
    states = []
    progress = []
    groups = []
    for index, stream in enumerate(streams):
        if boundaries_list is not None:
            boundaries = np.asarray(boundaries_list[index], dtype=int)
        else:
            boundaries = segment_boundaries(stream, **segment_kwargs)
        span = max(len(stream) - 1, 1)
        for boundary in np.asarray(boundaries, dtype=int):
            if 0 <= int(boundary) < len(stream):
                states.append(stream[int(boundary)])
                progress.append(float(boundary) / float(span))
                groups.append(int(index))
    empty = dict(axis=np.zeros(dims), axes=np.zeros((max(int(axes), 1), dims)),
                 center=np.zeros(dims), sigma=np.ones(dims), scale=1.0,
                 cuts=np.zeros(max((2 if bins is None else int(bins)) - 1, 1)),
                 bins=2 if bins is None else int(bins), time_corr=0.0,
                 time_corr_pooled=0.0, time_dir=np.zeros(dims))
    if not states:
        return empty
    states = np.asarray(states, dtype=float)
    progress = np.asarray(progress, dtype=float)
    center = states.mean(axis=0)
    sigma = states.std(axis=0)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    standardized = (states - center) / sigma
    # The clock coordinates: a coordinate whose value is (near-)deterministic
    # in the stream's own progress *within every episode*.  The statistic is
    # per episode and then minimised across episodes, because pooling streams
    # of different lengths makes a true clock (e.g. a frame counter) look
    # nonlinear even though it is exact inside each stream.
    #
    # It is read off the *whole* stream, not off the boundary states: the
    # clock is a property of the state, and a stage segmentation can leave a
    # short episode with one or two boundary states, which made the earlier
    # version silently skip those episodes and stop recognising the clock at
    # all.  The r^2 of a linear fit is invariant to the standardisation, so
    # raw stream frames are the right input.
    groups = np.asarray(groups, dtype=int)
    columns = standardized.shape[1] if standardized.ndim == 2 else 0
    time_dir = np.zeros(columns)
    clock_dims = np.zeros(0, dtype=int)
    if columns:
        r2_min = np.ones(columns)
        evaluated = np.zeros(columns, dtype=bool)
        for stream in streams:
            values = np.linspace(0.0, 1.0, len(stream))
            if len(stream) < 3 or float(np.std(values)) <= 1e-12:
                continue
            block = np.asarray(stream, dtype=float)
            design = np.column_stack([np.ones(len(values)), values])
            coefficients, _residual, _rank, _singular = np.linalg.lstsq(
                design, block, rcond=None
            )
            predicted = design @ coefficients
            residual = block - predicted
            centered = block - block.mean(axis=0)
            total = np.sum(centered ** 2, axis=0)
            explained = np.sum(residual ** 2, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2 = np.where(total > 1e-12, 1.0 - explained / total, 1.0)
            r2_min = np.minimum(r2_min, r2)
            evaluated[:] = True
        clock_dims = np.flatnonzero(evaluated & (r2_min >= float(clock_r2)))
        if clock_dims.size:
            time_dir[clock_dims] = 1.0 / float(np.sqrt(clock_dims.size))
    if len(states) < 2 or not standardized.any():
        raw_axes = np.zeros((max(int(axes), 1), standardized.shape[1]))
        if standardized.shape[1]:
            raw_axes[0, 0] = 1.0
    else:
        # Fit the directions on within-episode deviations.  A boundary cloud
        # pooled across episodes carries an episode-level offset (a policy or
        # world-instance difference); letting that supply the axis would name
        # "which episode this is" instead of "which milestone inside it".
        fit_states = standardized.copy()
        for group in np.unique(groups):
            members = groups == group
            if members.any():
                fit_states[members] -= fit_states[members].mean(axis=0)
        if not fit_states.any():
            fit_states = standardized
        _u, _s, vt = np.linalg.svd(fit_states, full_matrices=False)
        raw_axes = vt[:max(int(axes), 1)]
    directions = []
    for candidate in raw_axes:
        direction = np.asarray(candidate, dtype=float).copy()
        if clock_dims.size:
            direction[clock_dims] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            # This component is the clock itself; the next one carries the
            # concept.  Falling back to a coordinate axis here would put the
            # clock straight back into the identity.
            continue
        direction = direction / norm
        if direction[int(np.argmax(np.abs(direction)))] < 0.0:
            direction = -direction
        directions.append(direction)
        if len(directions) >= max(int(axes), 1):
            break
    if not directions:
        directions = [np.zeros(standardized.shape[1])]
        if standardized.shape[1]:
            directions[0][0] = 1.0
    while len(directions) < max(int(axes), 1):
        fill = np.zeros(standardized.shape[1])
        if fill.size:
            column = len(directions) % fill.size
            fill[column] = 1.0
            if clock_dims.size:
                fill[clock_dims] = 0.0
                norm = float(np.linalg.norm(fill))
                # A coordinate that is the clock projects to nothing; leave
                # the axis empty rather than putting the clock back in.
                fill = fill / norm if norm > 1e-9 else np.zeros_like(fill)
        directions.append(fill)
    directions = np.asarray(directions, dtype=float)
    projection = standardized @ directions[0]
    scale = float(np.std(projection))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    normalized = projection / scale
    if bins is None:
        cuts, chosen_bins = support_bins(
            normalized, min_support=min_support, max_bins=max_bins
        )
    else:
        chosen_bins = int(bins)
        inner = np.linspace(0.0, 1.0, chosen_bins + 1)[1:-1]
        cuts = (np.quantile(normalized, inner)
                if len(inner) else np.zeros(0))
    time_corr = 0.0
    if (len(progress) > 1 and float(np.std(progress)) > 1e-9
            and float(np.std(normalized)) > 1e-9):
        time_corr = float(abs(np.corrcoef(normalized, progress)[0, 1]))
    # The leak that matters is within an episode: a pooled correlation also
    # counts "long episodes look different", which the per-episode centring
    # above already removes from the directions.
    time_corr_within = 0.0
    within_projection = []
    within_progress = []
    for group in np.unique(groups):
        members = groups == group
        if members.sum() < 2:
            continue
        local_projection = normalized[members]
        local_progress = progress[members]
        within_projection.extend(local_projection - local_projection.mean())
        within_progress.extend(local_progress - local_progress.mean())
    if (len(within_projection) > 2
            and float(np.std(within_projection)) > 1e-9
            and float(np.std(within_progress)) > 1e-9):
        time_corr_within = float(abs(np.corrcoef(
            within_projection, within_progress
        )[0, 1]))
    return dict(axis=directions[0], axes=directions, center=center,
                sigma=sigma, scale=scale, cuts=np.asarray(cuts, dtype=float),
                bins=int(chosen_bins), time_corr=time_corr_within,
                time_corr_pooled=time_corr, time_dir=time_dir)


def frame_project(frame, states):
    """Read states through the frame: the identity's scalar coordinate.

    Accepts a single state (returns a float) or a 2-D batch (returns a column).
    """
    x = np.asarray(states, dtype=float)
    standardized = ((x - np.asarray(frame["center"], dtype=float))
                    / np.asarray(frame.get("sigma", np.ones(x.shape[-1])),
                                 dtype=float))
    return (standardized @ np.asarray(frame["axis"], dtype=float)) / float(
        frame["scale"]
    )


def frame_project_axes(frame, states):
    """All identity directions at once: an ``(n, axes)`` projection.

    The first column is :func:`frame_project`; the extra columns carry the
    second (and later) direction, which the goal term uses to tell the two
    sides of an identity apart instead of collapsing them onto one scalar.
    """
    x = np.asarray(states, dtype=float)
    center = np.asarray(frame["center"], dtype=float)
    sigma = np.asarray(
        frame.get("sigma", np.ones(x.shape[-1])), dtype=float
    )
    axes = np.asarray(
        frame.get("axes", np.asarray([frame["axis"]], dtype=float)),
        dtype=float,
    )
    standardized = (x - center) / sigma
    return (standardized @ axes.T) / float(frame["scale"])


def frame_keys(observations, boundaries, frame):
    """The identity of each milestone: the bin of its entry state's projection."""
    x = np.asarray(observations, dtype=float)
    boundaries = np.asarray(boundaries, dtype=int)
    edges = np.concatenate([[0], boundaries, [len(x)]])
    keys = []
    for index in range(len(edges) - 1):
        start = int(edges[index])
        state = x[start] if 0 <= start < len(x) else x[-1]
        value = float(frame_project(frame, state))
        keys.append((int(np.searchsorted(np.asarray(frame["cuts"]), value)),))
    return keys


def boundary_keys(observations, boundaries, bins: int = 2, cuts=None):
    """A symbolic identity per milestone: the boundary state's quantile bin.

    The identity is the state the milestone *starts* in, read through the
    stream's own marginal quantiles.  It is task-neutral (observations only)
    and time-scale invariant by construction.
    """
    x = np.asarray(observations, dtype=float)
    boundaries = np.asarray(boundaries, dtype=int)
    if cuts is None:
        cuts = quantile_cuts(x, bins)
    edges = np.concatenate([[0], boundaries, [len(x)]])
    keys = []
    for index in range(len(edges) - 1):
        start = int(edges[index])
        state = x[start] if 0 <= start < len(x) else x[-1]
        keys.append(tuple(
            int(np.searchsorted(cuts[d], state[d])) for d in range(x.shape[1])
        ))
    return keys


def next_milestone(memory, key):
    """The memory's prediction of what follows `key`.

    This is the top-down half: given the milestone the stream is in, the
    memory names the most frequently observed successor together with its
    recorded duration and feedback.  Returns ``None`` when the key was never
    a transition source, which is how a caller detects an unseen milestone.
    """
    key = tuple(int(value) for value in key)
    best = None
    for row in memory.get("transitions", ()):
        if tuple(row["key_from"]) != key:
            continue
        if best is None or row["count"] > best["count"]:
            best = row
    return best


def milestone_lookahead(table, key, key_feedback=None, steps: int = 2,
                        key_cost=None):
    """Wide, non-frame prediction: the stage distribution ``steps`` ahead.

    The state-level planner sees a fixed number of frames; this walk sees
    *stages*.  Starting from `key`, the transition table's counts spread a
    probability mass over the successors, and the walk is repeated `steps`
    times.  Reported quantities:

    * ``keys`` - the reachable stage identities, as (key, mass) pairs;
    * ``duration`` - expected frames spent along the walk (sum over the
      traversed transitions of their mean stage duration);
    * ``per_frame_feedback`` - expected net feedback per frame along the
      walk, which is the quantity comparable across stages of any length;
    * ``risk`` - the mass that lands on a stage whose own per-frame feedback
      is non-positive (a bad stage), or zero when the caller supplies no
      feedback table;
    * ``cost`` - the expected value of ``key_cost`` over the same frontier, a
      graded version of ``risk`` for consumers that need to tell "slightly
      worse than usual" from "much worse" (the zero-mass case is 0).

    Nothing here is frame-indexed: the horizon is a number of stages, and a
    stage's length is a distribution learned from the stream.
    """
    key = tuple(int(value) for value in key)
    table = list(table or ())
    steps = max(int(steps), 0)
    frontier = {key: 1.0}
    duration = 0.0
    per_frame = 0.0
    mass = 0.0
    expanded = False
    for _ in range(steps):
        following = {}
        advanced = False
        for source, weight in frontier.items():
            rows = [
                row for row in table
                if tuple(int(v) for v in row["key_from"]) == source
            ]
            total = sum(int(row["count"]) for row in rows)
            if total <= 0:
                # A stage with no recorded outgoing transition is where the
                # observed stream stopped: an absorbing end.  Keep its mass
                # in the frontier instead of letting it vanish, or the walk
                # silently stops counting exactly the stages that end badly.
                following[source] = following.get(source, 0.0) + weight
                continue
            advanced = True
            for row in rows:
                share = weight * int(row["count"]) / float(total)
                target = tuple(int(v) for v in row["key_to"])
                following[target] = following.get(target, 0.0) + share
                length = max(float(row["mean_duration"]), 1.0)
                duration += share * float(row["mean_duration"])
                per_frame += share * float(row["mean_feedback"]) / length
                mass += share
        if not following:
            break
        frontier = following
        expanded = expanded or advanced
    risk = 0.0
    if key_feedback:
        risk = float(sum(
            weight for stage, weight in frontier.items()
            if float(key_feedback.get(stage, 0.0)) <= 0.0
        ))
    cost = 0.0
    if key_cost and expanded:
        cost = float(sum(
            weight * float(key_cost.get(stage, 0.0))
            for stage, weight in frontier.items()
        ))
    return dict(
        keys=sorted(frontier.items()) if expanded else [],
        duration=float(duration),
        per_frame_feedback=float(per_frame / mass) if mass > 0.0 else 0.0,
        risk=risk,
        cost=cost,
        stages=steps,
    )


def transition_signatures(observations, predict, lag: int = 3,
                          horizon: int = 6):
    """Action-conditioned rollout signature per frame.

    ``signature[t]`` is the concatenation, over both actions, of the
    observations reached by rolling the learned model forward ``horizon``
    steps with that action held.  Two states are the same *stage* when they
    lead to the same places under the same actions - the transition-based
    identity, not the entry-state projection.

    ``predict(states, history)`` is the world model; nothing about the task
    is read.
    """
    x = np.asarray(observations, dtype=float)
    width = x.shape[1]
    signatures = np.zeros((len(x), 2 * int(horizon) * width))
    history = np.zeros((1, int(lag) + 1))
    for index, state in enumerate(x):
        for action in (0, 1):
            history[0, :] = float(action)
            current = state.reshape(1, -1)
            for step in range(int(horizon)):
                current = np.asarray(predict(current, history), dtype=float)
                start = (action * int(horizon) + step) * width
                signatures[index, start:start + width] = current[0]
    return signatures


def strength_change_scores(strength, signatures):
    """``|Δstrength| * d(signature)`` per frame - the stage-boundary statistic."""
    strength = np.asarray(strength, dtype=float).reshape(-1)
    signatures = np.asarray(signatures, dtype=float)
    if len(strength) != len(signatures):
        raise ValueError("strength and signatures must align")
    score = np.zeros(len(strength))
    if len(strength) < 2:
        return score
    change = np.abs(np.diff(strength, prepend=strength[0]))
    step = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
    score[1:] = change[1:] * step
    return score


def pooled_strength_threshold(score_streams, k: float = 2.0,
                              quantile: float = 0.90):
    """One threshold for every stream: the identity must be a common alphabet.

    Calibrating per stream lets a short life with a quiet score distribution
    accept almost anything as a boundary while a busy one accepts almost
    nothing.  Pooling the within-episode scores and taking ``k`` times their
    ``quantile`` gives every stream the same bar.
    """
    pooled = np.concatenate([
        np.asarray(score, dtype=float).reshape(-1)[1:]
        for score in score_streams if len(score) > 1
    ]) if len(score_streams) else np.zeros(0)
    if pooled.size == 0:
        return float("inf")
    return float(k) * float(np.quantile(pooled, quantile))


def strength_boundaries(strength, signatures, k: float = 3.0,
                        quantile: float = 0.90, nms=None, threshold=None):
    """Stage boundaries: a node's strength rises where the future changes.

    ``strength[t]`` is the learned node's count; ``signatures[t]`` the
    action-conditioned rollout.  The score is ``|Δstrength| * d(signature)``:
    a boundary is where the graph has just been reinforced *and* the states
    that follow are no longer the ones that followed before.  Returns the
    boundary indices, thresholded and non-maximum suppressed exactly like
    :func:`segment_boundaries`.  Pass `threshold` to read several streams
    through one pooled bar (:func:`pooled_strength_threshold`).
    """
    strength = np.asarray(strength, dtype=float).reshape(-1)
    signatures = np.asarray(signatures, dtype=float)
    if len(strength) != len(signatures):
        raise ValueError("strength and signatures must align")
    if len(strength) < 3:
        return np.zeros(0, dtype=int)
    score = strength_change_scores(strength, signatures)
    if nms is None:
        nms = int(round(1.0 / 0.2))
    if threshold is None:
        threshold = float(k) * float(np.quantile(score[1:], quantile))
    fired = np.flatnonzero(score > float(threshold))
    if nms <= 0 or len(fired) == 0:
        return fired.astype(int)
    keep = []
    for index in fired:
        low = max(0, index - int(nms))
        high = min(len(score), index + int(nms) + 1)
        if score[index] >= score[low:high].max() - 1e-12:
            keep.append(int(index))
    return np.asarray(keep, dtype=int)


def segment_prototypes(signatures, boundaries=None):
    """One prototype per stage: the mean signature between two boundaries."""
    x = np.asarray(signatures, dtype=float)
    if boundaries is None:
        boundaries = np.zeros(0, dtype=int)
    boundaries = np.asarray(boundaries, dtype=int)
    edges = np.concatenate([[0], boundaries, [len(x)]])
    return np.asarray([
        x[int(start):int(stop)].mean(axis=0) if stop > start
        else np.zeros(x.shape[1])
        for start, stop in zip(edges[:-1], edges[1:])
    ])


def kmeans_labels(points, k: int, iterations: int = 25):
    """Deterministic k-means (farthest-point init), no external dependency."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 0))
    width = points.shape[1]
    k = int(max(1, min(int(k), len(points))))
    centers = [points[0]]
    for _ in range(1, k):
        distances = np.min(np.linalg.norm(
            points[:, None, :] - np.asarray(centers, dtype=float)[None, :, :],
            axis=2,
        ), axis=1)
        centers.append(points[int(np.argmax(distances))])
    centers = np.asarray(centers, dtype=float).reshape(k, width)
    labels = np.full(len(points), -1, dtype=int)
    for _ in range(int(iterations)):
        distances = np.linalg.norm(
            points[:, None, :] - centers[None, :, :], axis=2
        )
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for index in range(k):
            members = points[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return labels, centers


def silhouette(points, labels, cap: int = 800):
    """Mean silhouette of a labelling; deterministic subsample above `cap`."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    if len(points) > int(cap):
        step = int(np.ceil(len(points) / float(cap)))
        keep = np.arange(0, len(points), step)
        points, labels = points[keep], labels[keep]
        unique = np.unique(labels)
        if len(unique) < 2:
            return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    scores = []
    for index in range(len(points)):
        own = labels == labels[index]
        own[index] = False
        if not own.any():
            continue
        inside = float(distances[index, own].mean())
        outside = min(
            float(distances[index, labels == other].mean())
            for other in unique if other != labels[index]
        )
        scores.append((outside - inside) / max(inside, outside, 1e-12))
    return float(np.mean(scores)) if scores else 0.0


def signature_clusters(prototype_streams, min_support: int = 3,
                       max_clusters: int = 6, min_separation: float = 0.15):
    """Pooled clustering of stage signatures: the identity's common alphabet.

    Every stream's stage prototypes are clustered *together*, so a stage has
    the same name in every episode and at decision time.  The cluster count is
    chosen by two rules that both come from the data: every cluster must be
    observed at least ``min_support`` times (the same support gate the frame's
    bins use), and the split must actually separate - the pooled silhouette
    must reach ``min_separation``, otherwise one identity is the honest answer.
    Without the second rule the support gate alone would happily split a
    tight cloud in two.

    Returns ``dict(centers, k, supports, separation)``.
    """
    streams = [
        np.asarray(stream, dtype=float)
        for stream in prototype_streams if len(stream)
    ]
    if not streams:
        return dict(centers=np.zeros((0, 0)), k=0, supports=[],
                    separation=0.0)
    points = np.vstack(streams)
    best_centers = points.mean(axis=0, keepdims=True)
    best_k = 1
    best_separation = 0.0
    limit = min(int(max_clusters), len(points) // max(int(min_support), 1))
    for count in range(2, int(limit) + 1):
        labels, centers = kmeans_labels(points, count)
        supports = np.bincount(labels, minlength=count)
        if supports.min() < int(min_support):
            continue
        separation = silhouette(points, labels)
        if separation > best_separation + 1e-9:
            best_centers, best_k = centers, count
            best_separation = float(separation)
    if best_k > 1 and best_separation < float(min_separation):
        best_centers = points.mean(axis=0, keepdims=True)
        best_k, best_separation = 1, 0.0
    labels = np.argmin(np.linalg.norm(
        points[:, None, :] - best_centers[None, :, :], axis=2
    ), axis=1)
    supports = np.bincount(labels, minlength=best_k)
    return dict(centers=np.asarray(best_centers, dtype=float), k=int(best_k),
                supports=[int(value) for value in supports],
                separation=float(best_separation))


def signature_keys(prototypes, centers):
    """Read stage prototypes through the shared cluster centers."""
    prototypes = np.asarray(prototypes, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if prototypes.size == 0 or centers.size == 0:
        return []
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    distances = np.linalg.norm(
        prototypes[:, None, :] - centers[None, :, :], axis=2
    )
    return [(int(index),) for index in np.argmin(distances, axis=1)]


def nearest_stage_key(states, keys, frame, state):
    """Which known stage does a state belong to?

    The stage identity is defined by the stage's *future*
    (:func:`signature_keys`), so this map is the inverse question: given only
    a state, the nearest known boundary state names its stage.  Reading the
    stages through the same frame keeps the comparison in the units the
    frame was calibrated on.  Used by any consumer that has a rollout
    endpoint but not the endpoint's own future.
    """
    x = np.asarray(states, dtype=float)
    if x.size == 0:
        return None
    center = np.asarray(frame["center"], dtype=float)
    sigma = np.asarray(frame.get("sigma", np.ones(x.shape[1])), dtype=float)
    point = (np.asarray(state, dtype=float).reshape(-1) - center) / sigma
    distances = np.linalg.norm((x - center) / sigma - point, axis=1)
    return tuple(int(value) for value in keys[int(np.argmin(distances))])


def nearest_stage_risk(states, stage_states, stage_keys, risks, frame,
                       scales=None):
    """How risky is the stage each candidate endpoint lands in?

    The stage table's risk is per stage, so a candidate can only be charged if
    the endpoint is mapped to a stage.  The map is nearest boundary state in
    the identity frame - the inverse of the stage index - and the charge is
    the h-step walk risk of that stage.  Vectorised over candidates.

    ``scales`` converts encoded states (observation / scales) back to the
    units `stage_states` and `frame` are expressed in.
    """
    states = np.asarray(states, dtype=float)
    anchors = np.asarray(stage_states, dtype=float)
    if states.size == 0 or anchors.size == 0:
        return np.zeros(len(states))
    center = np.asarray(frame["center"], dtype=float)
    sigma = np.asarray(frame.get("sigma", np.ones_like(center)), dtype=float)
    if scales is not None:
        unit = np.asarray(scales, dtype=float)
        center, sigma = center / unit, sigma / unit
        anchors, states = anchors / unit, states / unit
    anchors = (anchors - center) / sigma
    queries = (states - center) / sigma
    nearest = np.argmin(np.linalg.norm(
        queries[:, None, :] - anchors[None, :, :], axis=2
    ), axis=1)
    keys = list(stage_keys)
    return np.asarray([
        float(risks.get(tuple(int(value) for value in keys[int(index)]), 0.0))
        for index in nearest
    ])


def signature_stage_risk(signatures, centers, risks):
    """Charge each endpoint the risk of the stage its *own future* lands in.

    For a signature-based stage alphabet this is the consistent map: the
    endpoint's action-conditioned rollout is named by the nearest cluster
    center, exactly as the training stages were named.  No entry-state
    projection is involved.
    """
    signatures = np.asarray(signatures, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if signatures.size == 0 or centers.size == 0:
        return np.zeros(len(signatures))
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    labels = np.argmin(np.linalg.norm(
        signatures[:, None, :] - centers[None, :, :], axis=2
    ), axis=1)
    return np.asarray([
        float(risks.get((int(label),), 0.0)) for label in labels
    ])


def milestone_coverage(keys, learned_keys):
    """How much of a stream's milestone identities a learned memory covers.

    A milestone memory is built from one stream and then applied to another.
    When the two come from different policies they visit different states, so
    the applied identities fall outside the learned set and the memory has
    nothing to say about them.  This reports that fraction - and the unseen
    identities - so a consumer can abstain *visibly* instead of silently
    skipping the frames it cannot speak to.

    Returns ``dict(total, covered, fraction, unseen)``.
    """
    learned = {tuple(int(value) for value in key) for key in learned_keys}
    keys = [tuple(int(value) for value in key)
            for key in keys if key is not None]
    total = len(keys)
    if total == 0:
        return dict(total=0, covered=0, fraction=0.0, unseen=[])
    covered = sum(1 for key in keys if key in learned)
    unseen = sorted({key for key in keys if key not in learned})
    return dict(
        total=total, covered=covered, fraction=covered / total,
        unseen=[list(key) for key in unseen],
    )


class MilestoneTracker:
    """Causal boundary tracker: the offline rule with a frozen threshold.

    The threshold and the quantile cuts are calibrated once on the training
    stream (:func:`boundary_threshold`, :func:`quantile_cuts`) and then reused
    frame by frame, so a decision can know which milestone it is inside
    without seeing the future.  Non-maximum suppression is realised
    causally as a minimum spacing between accepted boundaries.

    ``observe`` returns the key of the current milestone (``None`` before the
    first boundary is seen) and reports whether this frame opened a new one.
    """

    def __init__(self, threshold, frame, beta: float = DEFAULT_BETA,
                 nms: int = DEFAULT_NMS, scale=None):
        if nms is None:
            nms = int(round(1.0 / float(beta)))
        if nms < 0:
            raise ValueError("nms must be non-negative")
        self.threshold = float(threshold)
        self.frame = frame
        self.beta = float(beta)
        self.nms = int(nms)
        # The per-coordinate ordinary step the threshold was read against.
        # Without it the causal score is the raw max over coordinates, which
        # a steadily ramping coordinate pins at a constant - see
        # :func:`boundary_scale`.
        self.scale = None if scale is None else np.asarray(scale, dtype=float)
        self.reset()

    def reset(self):
        self._state = None
        self._frames = 0
        self._last_boundary = -10 ** 9
        self.key = None

    def _score(self, step):
        if self.scale is None or self.scale.size == 0:
            return float(np.max(np.abs(step)))
        usable = self.scale > 0.0
        if not usable.any():
            return 0.0
        return float(np.max(np.abs(step[usable]) / self.scale[usable]))

    def _key_of(self, observation):
        return frame_keys(
            np.asarray(observation, dtype=float).reshape(1, -1),
            np.zeros(0, dtype=int), self.frame,
        )[0]

    def observe(self, observation, forced=False):
        """Advance one frame.  Returns True when a new milestone opened.

        `forced` marks a frame the caller knows is a boundary (for example the
        start of an episode), which opens the first milestone even though no
        jump has been seen yet.
        """
        x = np.asarray(observation, dtype=float)
        opened = False
        if self._state is None:
            self._state = x.copy()
            self._frames = 1
            self.key = self._key_of(x)
            self._last_boundary = 0
            return True
        step = self.beta * (x - self._state)
        self._state = self._state + step
        score = self._score(step)
        self._frames += 1
        if forced or (score > self.threshold
                      and self._frames - self._last_boundary > self.nms):
            self.key = self._key_of(x)
            self._last_boundary = self._frames
            opened = True
        return opened


def milestone_memory(observations, actions=None, feedback=None,
                     boundaries=None, bins: int = 2, cuts=None, frame=None,
                     scale=None, identity=None, **segment_kwargs):
    """Milestone-level transition memory: a count-based concept graph.

    Each stretch between two boundaries is a milestone, identified by the
    quantile bin of the state it starts in (:func:`boundary_keys`).  The
    memory then counts, per transition ``key -> key``, how often it happened,
    how long those milestones lasted and what feedback they carried.

    Nothing is assumed about duration - a milestone's length is recorded as a
    distribution, which is what lets one table describe the same task at any
    time scale.

    Two identity choices were measured on real streams before this one
    (2026-11-22).  The repository's stage rule - identity by *action effect* -
    gives a perfectly time-scale invariant key that is **degenerate**: the
    action effect is a constant of the world (per-dimension cross-milestone
    spread about 0.001 against a mean of 0.2), so every milestone collapses
    onto one key.  A finer quantile bin (3 or 4 levels) is discriminative but
    its key set stops agreeing across time scales (Jaccard 0.44 and 0.26).
    Two levels is the usable point: 16 keys over ~225 milestones at the fast
    scale, and an identical key set (Jaccard 1.00) at twice the time scale.

    ``cuts`` is the frame of reference the identity is read through.  It must
    be *shared* by every stream that is to be compared: those keys are only a
    common alphabet if all of them are read through the same cut points and
    the same boundary scale.  Calibrating per call makes each stream's keys
    incomparable, and leaves a decision-time consumer reading a frame taken
    from a single stream (measured 2026-11-22: 0.25 coverage of the planner's
    stream against a per-call frame, 1.00 for a held-out episode of the same
    policy once the frame is shared).

    ``identity`` overrides the key source with one key per stage, e.g. the
    pooled cluster of the stage's action-conditioned signature
    (:func:`signature_keys`), so the identity is what the stage leads to
    rather than where it happened to start.
    """
    x = np.asarray(observations, dtype=float)
    if boundaries is None:
        boundaries = segment_boundaries(x, scale=scale, **segment_kwargs)
    boundaries = np.asarray(boundaries, dtype=int)
    edges = np.concatenate([[0], boundaries, [len(x)]])
    keys = (list(identity) if identity is not None
            else frame_keys(x, boundaries, frame) if frame is not None
            else boundary_keys(x, boundaries, bins=bins, cuts=cuts))
    action_values = None if actions is None else np.asarray(actions, dtype=float)
    if action_values is not None and len(action_values) != len(x):
        raise ValueError("actions must have one entry per observation")
    durations, prototypes, feedback_totals, action_rates = [], [], [], []
    values = None if feedback is None else np.asarray(feedback, dtype=float)
    for index in range(len(edges) - 1):
        start, stop = int(edges[index]), int(edges[index + 1])
        segment = x[start:stop]
        durations.append(stop - start)
        prototypes.append(
            segment[0] if len(segment) else np.zeros(x.shape[1])
        )
        feedback_totals.append(
            0.0 if values is None else float(values[start:stop].sum())
        )
        if action_values is None:
            action_rates.append(0.0)
        elif stop > start:
            action_rates.append(float(action_values[start:stop].mean()))
        else:
            action_rates.append(0.0)
    durations = np.asarray(durations, dtype=int)
    feedback_totals = np.asarray(feedback_totals, dtype=float)
    action_rates = np.asarray(action_rates, dtype=float)

    transitions = {}
    for index in range(len(keys) - 1):
        entry = transitions.setdefault(
            (keys[index], keys[index + 1]),
            dict(count=0, durations=[], feedback=[]),
        )
        entry["count"] += 1
        entry["durations"].append(int(durations[index + 1]))
        entry["feedback"].append(float(feedback_totals[index + 1]))
    table = []
    for (key_from, key_to), entry in transitions.items():
        table.append(dict(
            count=int(entry["count"]),
            key_from=list(key_from),
            key_to=list(key_to),
            mean_duration=float(np.mean(entry["durations"])),
            mean_feedback=float(np.mean(entry["feedback"])),
        ))
    table.sort(key=lambda row: (-row["count"], row["key_from"], row["key_to"]))
    key_rate = {}
    if action_values is not None:
        # Duration-weighted action rate per milestone identity: the learned
        # "action tendency" of a milestone, a scalar in [0, 1] that carries no
        # position and therefore no time scale.
        totals = {}
        weights = {}
        for index, key in enumerate(keys):
            totals[key] = (totals.get(key, 0.0)
                           + float(action_rates[index]) * durations[index])
            weights[key] = weights.get(key, 0.0) + float(durations[index])
        key_rate = {
            key: totals[key] / weights[key] for key in totals if weights[key]
        }
    return dict(
        boundaries=boundaries,
        keys=keys,
        durations=durations,
        feedback=feedback_totals,
        action_rate=action_rates,
        key_rate=key_rate,
        prototypes=np.asarray(prototypes, dtype=float),
        transitions=table,
    )
