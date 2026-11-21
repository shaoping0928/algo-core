"""Stage graph schema and the offline action-residual segmenter (slice 1).

Design constraints (roundtable, 2026-11-21):

- pure data layer: no planner, no NodeRuntime, no default-path changes;
- task-agnostic: observations, acting actions and their deltas only;
- a stage is a maximal run explained by one action-conditioned local model;
- split only when a BIC-type gain on the *residual* exceeds the penalty;
- a constant stream must return one stage;
- segmentation is gated on the pooled out-of-fold R^2 of the local model
  (standardised fit, alpha=1.0, zero-variance dimensions dropped);
- the local model is one step (k=1) and uses the acting action
  (``history[lag]``): the LOEO probe showed this is the usable form.

Stage identity for cross-episode counting uses the stage's *action effect*
``mean(delta | a=1) - mean(delta | a=0)``; stages with the same effect are
the same stage.  The only downstream gate is
``StageGraph.cross_episode_informative``: transitions that are not
self-loops or END, with enough frames and enough distinct episodes.

Usage (read-only evaluation on cached streams)::

    python stage_graph.py --streams stage_streams.npz --report stages.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ACTING_FALLBACK = 0
MIN_LENGTH = 5
RIDGE_ALPHA = 1.0


@dataclass
class Stage:
    index: int
    episode: int
    start: int
    end: int
    action_means: Dict[int, List[float]] = field(default_factory=dict)
    action_effect: List[float] = field(default_factory=list)
    residual_rss: float = 0.0
    support: int = 0
    action_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return self.end - self.start


@dataclass
class StageTransition:
    source: int          # stage cluster id
    action: int
    target: Optional[int]  # None = END of episode
    frames: int
    episodes: int


@dataclass
class StageGraph:
    stages: List[Stage]
    transitions: List[StageTransition]
    cluster_radius: float
    gate_r2: float
    clusters: int
    stream_frames: int
    episodes: int

    def cross_episode_informative(self, min_support: int = 5,
                                  min_episodes: int = 3) -> int:
        count = 0
        for transition in self.transitions:
            if transition.target is None:
                continue
            if transition.target == transition.source:
                continue
            if (transition.frames >= min_support
                    and transition.episodes >= min_episodes):
                count += 1
        return count

    def summary(self) -> dict:
        return dict(
            episodes=self.episodes,
            frames=self.stream_frames,
            stages=len(self.stages),
            clusters=self.clusters,
            cluster_radius=self.cluster_radius,
            gate_r2=self.gate_r2,
            transitions=len(self.transitions),
            cross3=self.cross_episode_informative(),
            stage_length_p50=float(np.median(
                [stage.frames for stage in self.stages]
            )) if self.stages else 0.0,
        )


def _episode_arrays(rows, lag):
    """Per-episode observations, deltas and acting actions (task-agnostic)."""
    episodes = defaultdict(list)
    for row in rows:
        episodes[int(row["episode"])].append(row)
    result = {}
    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        observations = np.asarray(
            [row["obs"] for row in episode], dtype=float
        )
        deltas = np.asarray(
            [row["next_obs"] for row in episode], dtype=float
        ) - observations
        acting = np.asarray([
            int(row["history"][lag])
            if len(row["history"]) > lag else ACTING_FALLBACK
            for row in episode
        ], dtype=int)
        result[episode_id] = (observations, deltas, acting)
    return result


def _standardise(rows, lag):
    """Global feature/target scaling; zero-variance target dims dropped."""
    episodes = _episode_arrays(rows, lag)
    observations = np.concatenate(
        [value[0] for value in episodes.values()], axis=0
    )
    deltas = np.concatenate(
        [value[1] for value in episodes.values()], axis=0
    )
    state_scale = observations.std(axis=0)
    state_scale = np.where(state_scale > 1e-8, state_scale, 1.0)
    delta_scale = deltas.std(axis=0)
    keep = delta_scale > 1e-8
    return episodes, state_scale, keep


def _design(states, actions, n_actions=2):
    one_hot = np.zeros((len(actions), n_actions), dtype=float)
    one_hot[np.arange(len(actions)), np.clip(actions, 0, n_actions - 1)] = 1.0
    return np.concatenate([states, one_hot], axis=1)


def _ridge_weights(features, target, alpha=RIDGE_ALPHA):
    design = np.concatenate(
        [features, np.ones((len(features), 1))], axis=1
    )
    gram = design.T @ design + alpha * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def _segment_rss(features, target, alpha=RIDGE_ALPHA):
    if len(features) == 0:
        return 0.0
    weights = _ridge_weights(features, target, alpha)
    design = np.concatenate(
        [features, np.ones((len(features), 1))], axis=1
    )
    residual = target - design @ weights
    return float((residual ** 2).sum())


def _bic(rss, count, parameters):
    if count <= 0:
        return 0.0
    return (
        count * np.log(rss / max(count, 1) + 1e-12)
        + parameters * np.log(max(count, 1))
    )


def pooled_oof_r2(features, target, groups, alpha=RIDGE_ALPHA):
    """Pooled leave-one-episode-out R^2 (groups are episodes).

    Per fold: standardise features with the training fold's mean/std, drop
    zero-variance columns, standardise the target the same way, then ridge.
    This is the recipe the diagnostics validated; global scaling leaves a
    few ill-conditioned folds with explosive predictions.
    """
    if target.ndim != 2 or target.shape[1] == 0 or len(groups) == 0:
        return 0.0
    predictions = np.zeros_like(target)
    for group in np.unique(groups):
        train = groups != group
        test = groups == group
        train_features = features[train]
        test_features = features[test]
        mean = train_features.mean(axis=0)
        scale = train_features.std(axis=0)
        keep = scale > 1e-12
        train_features = (train_features[:, keep] - mean[keep]) / scale[keep]
        test_features = (test_features[:, keep] - mean[keep]) / scale[keep]
        train_target = target[train]
        target_mean = train_target.mean(axis=0)
        target_scale = np.maximum(train_target.std(axis=0), 1e-12)
        weights = _ridge_weights(
            train_features,
            (train_target - target_mean) / target_scale,
            alpha,
        )
        design = np.concatenate(
            [test_features, np.ones((int(test.sum()), 1))], axis=1
        )
        predictions[test] = (
            design @ weights
        ) * target_scale + target_mean
    sse = ((target - predictions) ** 2).sum(axis=0)
    sst = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim = np.where(
            sst > 0.0, 1.0 - sse / np.maximum(sst, 1e-12), 0.0
        )
    return float(np.mean(per_dim))


def _segment_episode(features, target, min_length=MIN_LENGTH,
                     penalty=1.0):
    """Greedy BIC splitting on the residual of the action-conditioned model."""
    length = len(features)
    if length < 2 * min_length:
        return [(0, length)]
    boundaries = [0, length]
    stack = [(0, length)]
    parameters = (features.shape[1] + 1) * target.shape[1]
    while stack:
        start, end = stack.pop()
        if end - start < 2 * min_length:
            continue
        parent = _bic(
            _segment_rss(features[start:end], target[start:end]),
            end - start, parameters,
        )
        best_gain = 0.0
        best_cut = None
        for cut in range(start + min_length, end - min_length + 1):
            left = _bic(
                _segment_rss(features[start:cut], target[start:cut]),
                cut - start, parameters,
            )
            right = _bic(
                _segment_rss(features[cut:end], target[cut:end]),
                end - cut, parameters,
            )
            gain = parent - (left + right)
            if gain > best_gain:
                best_gain = gain
                best_cut = cut
        if best_cut is not None and best_gain > penalty:
            boundaries.append(best_cut)
            stack.append((start, best_cut))
            stack.append((best_cut, end))
    boundaries = sorted(set(boundaries))
    return [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]


def _action_effect(deltas, actions):
    one = deltas[actions == 1]
    zero = deltas[actions == 0]
    if len(one) and len(zero):
        return one.mean(axis=0) - zero.mean(axis=0)
    return deltas.mean(axis=0) if len(deltas) else np.zeros(deltas.shape[1])


def _single_link(vectors, radius):
    count = len(vectors)
    if count == 0:
        return []
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(count):
        for second in range(first + 1, count):
            if np.linalg.norm(vectors[first] - vectors[second]) <= radius:
                first_root = find(first)
                second_root = find(second)
                if first_root != second_root:
                    parent[second_root] = first_root
    labels = [find(index) for index in range(count)]
    mapping = {root: index for index, root in enumerate(sorted(set(labels)))}
    return [mapping[label] for label in labels]


def _kmeans(vectors, count, seed=0, iterations=60):
    """Deterministic k-means on the action-effect vectors."""
    vectors = np.asarray(vectors, dtype=float)
    total = len(vectors)
    if total == 0:
        return []
    count = max(1, min(int(count), total))
    rng = np.random.RandomState(seed)
    centers = vectors[rng.choice(total, count, replace=False)].copy()
    labels = np.zeros(total, dtype=int)
    for _ in range(iterations):
        distances = (
            (vectors[:, None, :] - centers[None, :, :]) ** 2
        ).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(count):
            members = vectors[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return [int(label) for label in labels]


def _segment_stream(rows, lag=3, min_length=MIN_LENGTH, penalty=1.0,
                    gate_min_r2=0.0):
    """Per-episode segmentation; returns (stages, gate_r2, episode_count)."""
    episodes, state_scale, keep = _standardise(rows, lag)
    states = []
    targets = []
    groups = []
    for episode_id, (observations, deltas, actions) in episodes.items():
        features = _design(
            observations[:-1] / state_scale, actions[:-1]
        )
        target = deltas[:, keep]
        states.append(features)
        targets.append(target[:-1])
        groups.extend([episode_id] * len(features))
    features = np.concatenate(states, axis=0) if states else np.zeros((0, 0))
    target = np.concatenate(targets, axis=0) if targets else np.zeros((0, 0))
    groups = np.asarray(groups, dtype=int)
    gate_r2 = pooled_oof_r2(features, target, groups) if len(groups) else 0.0

    stages: List[Stage] = []
    for episode_id, (observations, deltas, actions) in episodes.items():
        features = _design(
            observations[:-1] / state_scale, actions[:-1]
        )
        target = deltas[:-1][:, keep]
        if gate_r2 < gate_min_r2 or target.shape[1] == 0:
            found = [(0, len(actions))]
        else:
            found = _segment_episode(
                features, target, min_length=min_length, penalty=penalty
            )
        for start, end in found:
            segment_actions = actions[start:end]
            means = {
                int(action): deltas[start:end][segment_actions == action]
                .mean(axis=0).tolist()
                for action in (0, 1)
                if (segment_actions == action).any()
            }
            stages.append(Stage(
                index=len(stages),
                episode=episode_id,
                start=start,
                end=end,
                action_means=means,
                action_effect=_action_effect(
                    deltas[start:end], segment_actions
                ).tolist(),
                residual_rss=_segment_rss(
                    features[start:end], target[start:end]
                ),
                support=end - start,
                action_counts={
                    int(action): int((segment_actions == action).sum())
                    for action in (0, 1)
                },
            ))
    return stages, float(gate_r2), len(episodes)


def _label_stages(stages, radius=None, clusters=None):
    """Stage identity: fixed-K k-means or single link on action effects."""
    if not stages:
        return [], 0.0
    effects = np.asarray([stage.action_effect for stage in stages],
                         dtype=float)
    if clusters is not None:
        return _kmeans(effects, clusters), 0.0
    if len(effects) > 1:
        pairwise = np.linalg.norm(
            effects[:, None, :] - effects[None, :, :], axis=2
        )
        upper = pairwise[np.triu_indices(len(effects), 1)]
        positive = upper[upper > 1e-12]
        if radius is None:
            # Half the median *positive* distance: the plain median can
            # equal the distance between two distinct action effects (as in
            # a two-regime stream) and would merge them.
            radius = (
                0.5 * float(np.median(positive)) if len(positive) else 0.0
            )
    else:
        radius = 0.0
    return _single_link(effects, radius), float(radius)


def _build_graph(stages, labels, radius, gate_r2, rows, episode_count):
    edge_frames = Counter()
    edge_episodes = defaultdict(set)
    stage_by_episode = defaultdict(list)
    for stage, label in zip(stages, labels):
        stage_by_episode[stage.episode].append((stage, label))
    for episode_stages in stage_by_episode.values():
        for position, (stage, label) in enumerate(episode_stages):
            target_label = (
                episode_stages[position + 1][1]
                if position + 1 < len(episode_stages) else None
            )
            for action in (0, 1):
                frames = stage.action_counts.get(action, 0)
                if frames:
                    key = (label, action, target_label)
                    edge_frames[key] += frames
                    edge_episodes[key].add(stage.episode)
    transitions = [
        StageTransition(
            source=source, action=action, target=target_label,
            frames=frames, episodes=len(edge_episodes[key]),
        )
        for (source, action, target_label), frames in edge_frames.items()
    ]
    return StageGraph(
        stages=stages,
        transitions=transitions,
        cluster_radius=float(radius),
        gate_r2=float(gate_r2),
        clusters=len(set(labels)),
        stream_frames=len(rows),
        episodes=episode_count,
    )


def build_stage_graph(rows, lag=3, min_length=MIN_LENGTH,
                      penalty=1.0, radius=None,
                      gate_min_r2=0.0, clusters=None) -> StageGraph:
    """Segment one real stream into stages and build the transition table."""
    stages, gate_r2, episode_count = _segment_stream(
        rows, lag=lag, min_length=min_length, penalty=penalty,
        gate_min_r2=gate_min_r2,
    )
    labels, radius = _label_stages(
        stages, radius=radius, clusters=clusters
    )
    return _build_graph(
        stages, labels, radius, gate_r2, rows, episode_count
    )


def sweep_vocabulary(rows, ks, lag=3, min_length=MIN_LENGTH, penalty=1.0):
    """Fixed-K vocabulary sweep: real vs action-shuffled cross3 per K."""
    real_stages, real_gate, episodes = _segment_stream(
        rows, lag=lag, min_length=min_length, penalty=penalty
    )
    shuffled_stages, shuffle_gate, _episodes = _segment_stream(
        shuffle_actions(rows, lag=lag), lag=lag,
        min_length=min_length, penalty=penalty,
    )
    result = []
    for k in ks:
        real_labels, real_radius = _label_stages(
            real_stages, clusters=k
        )
        shuffled_labels, _radius = _label_stages(
            shuffled_stages, clusters=k
        )
        real = _build_graph(
            real_stages, real_labels, real_radius, real_gate, rows, episodes
        )
        shuffled = _build_graph(
            shuffled_stages, shuffled_labels, 0.0, shuffle_gate, rows,
            episodes,
        )
        result.append(dict(
            k=int(k),
            real_cross3=real.cross_episode_informative(),
            shuffle_cross3=shuffled.cross_episode_informative(),
            real_transitions=len(real.transitions),
            shuffle_transitions=len(shuffled.transitions),
        ))
    return result


def shuffle_actions(rows, lag=3, seed=12345):
    """Per-episode acting-label permutation (multiset kept)."""
    rng = np.random.RandomState(seed)
    result = []
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode"])].append(row)
    for episode_id in sorted(by_episode):
        episode = by_episode[episode_id]
        actions = [
            int(row["history"][lag])
            if len(row["history"]) > lag else ACTING_FALLBACK
            for row in episode
        ]
        shuffled = rng.permutation(actions)
        for row, action in zip(episode, shuffled):
            copy = dict(row)
            history = list(row["history"])
            if len(history) > lag:
                history[lag] = float(action)
            copy["history"] = np.asarray(history, dtype=float)
            result.append(copy)
    return result


def evaluate_stream(rows, lag=3, min_length=MIN_LENGTH, penalty=1.0):
    real = build_stage_graph(
        rows, lag=lag, min_length=min_length, penalty=penalty
    )
    shuffled = build_stage_graph(
        shuffle_actions(rows, lag=lag), lag=lag,
        min_length=min_length, penalty=penalty,
    )
    return dict(
        real=real.summary(),
        shuffle_actions=shuffled.summary(),
        stages=real.stages,
        transitions=real.transitions,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", default="")
    parser.add_argument("--sweep-k", default="",
                        help="comma-separated fixed vocabulary sizes; "
                             "runs real vs action-shuffled cross3 per K")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if not args.streams:
        raise SystemExit("--streams is required")
    from stage_support_probe import load_streams
    streams = load_streams(args.streams)
    if args.sweep_k:
        ks = [int(value) for value in args.sweep_k.split(",") if value.strip()]
        payload = dict(phases=[])
        print()
        for index, rows in enumerate(streams):
            results = sweep_vocabulary(rows, ks)
            payload["phases"].append(dict(
                phase=index + 1, frames=len(rows), sweep=results,
            ))
            print("== phase %d: %d frames ==" % (index + 1, len(rows)))
            print("    K  real cross3  shuffle cross3  "
                  "real transitions  shuffle transitions")
            for item in results:
                print(
                    "  %3d  %10d  %13d  %16d  %19d"
                    % (item["k"], item["real_cross3"],
                       item["shuffle_cross3"], item["real_transitions"],
                       item["shuffle_transitions"])
                )
        print()
        if args.report:
            with open(args.report, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
        return
    report = dict(phases=[
        dict(phase=index + 1, **evaluate_stream(stream))
        for index, stream in enumerate(streams)
    ])
    print()
    for phase in report["phases"]:
        print(
            "== phase %d: %d frames / %d episodes =="
            % (phase["phase"], phase["real"]["frames"],
               phase["real"]["episodes"])
        )
        for label in ("real", "shuffle_actions"):
            entry = phase[label]
            print(
                "  %-16s stages %d  clusters %d  gate R2 %+.3f  "
                "transitions %d  cross3 %d  length p50 %.0f"
                % (label, entry["stages"], entry["clusters"],
                   entry["gate_r2"], entry["transitions"],
                   entry["cross3"], entry["stage_length_p50"])
            )
    print()
    if args.report:
        payload = dict(phases=[
            dict(
                phase=phase["phase"],
                real=phase["real"],
                shuffle_actions=phase["shuffle_actions"],
                stage_summaries=[
                    dict(index=stage.index, episode=stage.episode,
                         start=stage.start, end=stage.end,
                         support=stage.support,
                         action_counts=stage.action_counts)
                    for stage in phase["stages"]
                ],
            )
            for phase in report["phases"]
        ])
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
