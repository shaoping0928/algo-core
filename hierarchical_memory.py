"""Heterogeneous hierarchical memory: frames -> events -> phases -> worlds.

The levels are not the same operation repeated:

* L0 frame windows are temporal segments of a few transitions;
* L1 events are the salient windows (feedback change, prediction surprise,
  context change, novelty) with a little surrounding context;
* L2 phases are runs of consecutive events while the context stays stable;
* L3 worlds are episode-signature clusters; siblings are different worlds;
* L4 root is the pooled global summary.

A variable is persistent when the same direction carries support across
phases *and* across world siblings, and temporary when its support lives in
one low-level node.  That comparison across memories is the point of the
hierarchy; a flat counter cannot make it.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from memory_tree import MemoryTree
from variable_registry import VariableRegistry


@dataclass
class FrameSummary:
    index: int
    episode: int
    start: int
    end: int
    observation: np.ndarray
    feedback: np.ndarray
    error: float
    context: np.ndarray
    salience: float = 0.0


@dataclass
class EventSummary:
    index: int
    episode: int
    start: int
    end: int
    context: np.ndarray
    feedback: np.ndarray
    salience: float
    frames: List[int] = field(default_factory=list)


@dataclass
class PhaseSummary:
    index: int
    episode: int
    start: int
    end: int
    context_mean: np.ndarray
    context_variance: np.ndarray
    feedback: np.ndarray
    events: List[int] = field(default_factory=list)


@dataclass
class EpisodeSummary:
    episode: int
    context_mean: np.ndarray
    context_variance: np.ndarray
    feedback: np.ndarray
    frames: int
    events: int
    phases: int
    signature: np.ndarray


class HierarchicalMemory:
    """Online frame/event/phase segmentation plus a world-level merge tree."""

    def __init__(self, context_dim: int, observation_dim: int,
                 window: int = 4, salience_z: float = 1.5,
                 phase_split: float = 0.6, max_phase_gap: int = 24,
                 world_fanout: int = 4, world_radius: float = 1.5,
                 world_capacity: int = 24, decay: float = 0.95,
                 base_prior: float = 4.0, keyword_dim: int = 2):
        self.context_dim = int(context_dim)
        self.observation_dim = int(observation_dim)
        self.window = max(int(window), 1)
        self.salience_z = float(salience_z)
        self.phase_split = float(phase_split)
        self.max_phase_gap = max(int(max_phase_gap), 1)
        self.frames: List[FrameSummary] = []
        self.events: List[EventSummary] = []
        self.phases: List[PhaseSummary] = []
        self.episodes: List[EpisodeSummary] = []
        # World signatures: context mean/std, feedback mean, and three
        # scale-free activity ratios.  The dimension is fixed; the tree
        # clusters episodes by their dynamics/phase profile.
        self.signature_dim = self.context_dim * 2 + 5
        self.world_tree = MemoryTree(
            self.signature_dim, max_fanout=world_fanout,
            leaf_capacity=world_capacity, merge_radius=world_radius,
            split_variance=world_radius ** 2, decay=decay,
            keyword_dim=keyword_dim,
        )
        self.registry = VariableRegistry(base_prior=base_prior)
        self._root = self.registry.add(
            "root", "root", np.zeros(self.context_dim),
            support=0.0, timescale="slow", metadata={"level": 4},
        )
        self._level_ids = {}
        for level, name, timescale in (
            (0, "frames", "fast"), (1, "events", "fast"),
            (2, "phases", "slow"), (3, "worlds", "slow"),
        ):
            variable_id = "level:%s" % name
            self.registry.add(
                variable_id, "level", np.zeros(self.context_dim),
                support=0.0, timescale=timescale,
                metadata={"level": level},
            )
            self._level_ids[level] = variable_id
        for level in range(4):
            self.registry.relate(
                self._level_ids[level], self._level_ids.get(level + 1, "root"),
                "part_of",
            )
        self._episode = -1
        self._step = 0
        self._episode_start = 0
        self._episode_event_start = 0
        self._episode_phase_start = 0
        self.episode_leaf: Dict[int, int] = {}
        self._frame_buffer = []
        self._episode_contexts = []
        self._episode_feedback = []
        self._phase_events: List[int] = []
        self._phase_start = 0
        self._last_event_step = 0
        self._last_context = np.zeros(self.context_dim)
        self._last_feedback = np.zeros(2)
        self._salience_history: List[float] = []

    # ---- level construction ----------------------------------------------
    def _salience(self, feedback, error, context):
        feedback_change = float(np.linalg.norm(
            np.asarray(feedback, dtype=float) - self._last_feedback
        ))
        context_change = float(np.linalg.norm(
            np.asarray(context, dtype=float) - self._last_context
        ))
        surprise = 0.0
        if len(self._salience_history) >= 4:
            baseline = float(np.mean(self._salience_history[-16:]))
            spread = float(np.std(self._salience_history[-16:])) + 1e-9
            surprise = max(0.0, (error - baseline) / spread)
        novelty = 0.0
        if self.world_tree.leaves:
            query = self.world_tree.query(self._signature(context))
            if query is not None:
                novelty = max(0.0, query["distance"] - self.world_tree.merge_radius)
        return feedback_change + context_change + surprise + novelty

    def _signature(self, context):
        context = np.asarray(context, dtype=float).reshape(self.context_dim)
        feedback = (
            np.asarray(self._episode_feedback[-1], dtype=float)
            if self._episode_feedback else np.zeros(2)
        )
        return np.concatenate([
            context, np.abs(context), feedback, [0.0, 0.0, 0.0],
        ])[:self.signature_dim]

    def _close_phase(self):
        if not self._phase_events:
            return
        event_rows = [self.events[index] for index in self._phase_events]
        contexts = np.stack([row.context for row in event_rows])
        feedback = np.mean(
            np.stack([row.feedback for row in event_rows]), axis=0
        )
        self.phases.append(PhaseSummary(
            index=len(self.phases), episode=self._episode,
            start=self._phase_start, end=self._last_event_step,
            context_mean=contexts.mean(axis=0),
            context_variance=contexts.var(axis=0),
            feedback=feedback, events=list(self._phase_events),
        ))
        self.registry.update(
            self._level_ids[2], contexts.mean(axis=0), weight=1.0
        )
        self._phase_events = []

    def _make_event(self, frame: FrameSummary):
        self.events.append(EventSummary(
            index=len(self.events), episode=self._episode,
            start=frame.start, end=frame.end,
            context=frame.context.copy(), feedback=frame.feedback.copy(),
            salience=frame.salience, frames=[frame.index],
        ))
        self.registry.update(
            self._level_ids[1], frame.context, weight=1.0
        )
        if not self._phase_events:
            self._phase_start = frame.start
        self._phase_events.append(len(self.events) - 1)
        self._last_event_step = self._step

    def observe(self, observation, feedback, error, context,
                done=False, episode: Optional[int] = None):
        """Feed one transition; level construction happens online."""
        if episode is not None and int(episode) != self._episode:
            if self._episode >= 0:
                self.end_episode()
            self._episode = int(episode)
            self._episode_start = self._step
            self._episode_event_start = len(self.events)
            self._episode_phase_start = len(self.phases)
            self._frame_buffer = []
            self._episode_contexts = []
            self._episode_feedback = []
            self._phase_events = []
            self._last_event_step = self._step
        self._step += 1
        observation = np.asarray(observation, dtype=float).reshape(-1)
        feedback = np.asarray(feedback, dtype=float).reshape(-1)
        context = np.asarray(context, dtype=float).reshape(self.context_dim)
        self._frame_buffer.append((
            observation, feedback, float(error), context.copy(),
        ))
        self._episode_contexts.append(context.copy())
        self._episode_feedback.append(feedback.copy())
        if len(self._frame_buffer) >= self.window:
            block = self._frame_buffer[-self.window:]
            frame = FrameSummary(
                index=len(self.frames), episode=self._episode,
                start=self._step - len(block), end=self._step,
                observation=np.mean(
                    np.stack([item[0] for item in block]), axis=0
                ),
                feedback=np.mean(
                    np.stack([item[1] for item in block]), axis=0
                ),
                error=float(np.mean([item[2] for item in block])),
                context=np.mean(
                    np.stack([item[3] for item in block]), axis=0
                ),
            )
            frame.salience = self._salience(
                frame.feedback, frame.error, frame.context
            )
            threshold = 0.0
            if len(self._salience_history) >= 8:
                values = np.asarray(self._salience_history[-32:])
                threshold = float(
                    values.mean() + self.salience_z * values.std()
                )
            frame.salience = max(frame.salience, 0.0)
            if frame.salience > threshold and threshold > 0.0:
                self._make_event(frame)
            self._salience_history.append(frame.salience)
            self.frames.append(frame)
            self.registry.update(
                self._level_ids[0], frame.context, weight=1.0
            )
            context_delta = float(np.linalg.norm(
                frame.context - self._last_context
            ))
            if self._phase_events and (
                context_delta > self.phase_split
                or self._step - self._last_event_step > self.max_phase_gap
            ):
                self._close_phase()
            self._last_context = frame.context.copy()
            self._last_feedback = frame.feedback.copy()
        if done:
            self.end_episode()
        return None

    def end_episode(self):
        if self._episode < 0:
            return None
        self._close_phase()
        if not self._episode_contexts:
            return None
        contexts = np.stack(self._episode_contexts)
        feedback = np.stack(self._episode_feedback)
        signature = self._signature(contexts.mean(axis=0))
        signature[-3] = len(self._episode_contexts) / 1000.0
        signature[-2] = (
            len(self.events) - self._episode_event_start
        ) / 100.0
        signature[-1] = (
            len(self.phases) - self._episode_phase_start
        ) / 10.0
        episode = EpisodeSummary(
            episode=self._episode,
            context_mean=contexts.mean(axis=0),
            context_variance=contexts.var(axis=0),
            feedback=feedback.mean(axis=0),
            frames=len(self._episode_contexts),
            events=len(self.events) - self._episode_event_start,
            phases=len(self.phases) - self._episode_phase_start,
            signature=signature,
        )
        self.episodes.append(episode)
        leaf_id = self.world_tree.insert(signature, weight=float(len(
            self._episode_contexts
        )))
        self.episode_leaf[int(episode.episode)] = int(leaf_id)
        self._update_registry(episode, leaf_id)
        self._episode = -1
        return episode

    # ---- registry ---------------------------------------------------------
    def _update_registry(self, episode: EpisodeSummary, leaf_id: int):
        world_id = "world:%d" % leaf_id
        if world_id not in self.registry.variables:
            self.registry.add(
                world_id, "world", episode.signature,
                support=episode.frames, variance=float(
                    np.mean(episode.context_variance)
                ),
                timescale="slow",
                metadata={"level": 3, "episode": episode.episode},
            )
            self.registry.relate(world_id, "root", "part_of")
        else:
            self.registry.update(
                world_id, episode.signature, weight=float(episode.frames)
            )
        self.registry.update(
            "root", episode.context_mean, weight=float(episode.frames)
        )
        self.registry.update(
            self._level_ids[3], episode.context_mean,
            weight=float(episode.frames),
        )
        self.registry.register_tree(self.world_tree)
        self.registry.register_summaries(self.world_tree)

    # ---- summaries / outward expansion -----------------------------------
    def phase_summary(self, index: int, keyword_dim: Optional[int] = None):
        """The lossy paragraph summary of one phase: keywords + residual."""
        phase = self.phases[index]
        if not phase.events:
            return dict(mean=phase.context_mean.copy(), keywords=None,
                        weights=None, residual=0.0)
        vectors = np.stack([
            self.events[event].context for event in phase.events
        ])
        weights = np.ones(len(vectors), dtype=float)
        return _summarize_vectors(
            vectors, weights,
            self.world_tree.keyword_dim if keyword_dim is None
            else int(keyword_dim),
        )

    def expand_phase(self, index: int):
        """Regenerate an approximate event sequence from the phase summary."""
        phase = self.phases[index]
        summary = self.phase_summary(index)
        result = []
        for event_id in phase.events:
            event = self.events[event_id]
            if summary["keywords"] is not None:
                code = (
                    event.context - summary["mean"]
                ) @ summary["keywords"].T
                reconstructed = summary["mean"] + code @ summary["keywords"]
            else:
                reconstructed = summary["mean"].copy()
            result.append(dict(
                event=event_id, original=event.context.copy(),
                reconstructed=reconstructed,
                error=float(np.linalg.norm(
                    event.context - reconstructed
                )),
            ))
        return dict(summary=summary, events=result)

    def expand_world(self, node_id: int):
        return self.world_tree.expand(node_id)

    def world_keywords(self, node_id: int):
        return self.world_tree.keyword_persistence(node_id)

    # ---- queries ----------------------------------------------------------
    def _subtree_leaves(self, node_id: int) -> set:
        node = self.world_tree.nodes.get(node_id)
        if node is None:
            return set()
        if node.leaf:
            return {node_id}
        result = set()
        for child in node.children:
            result |= self._subtree_leaves(child)
        return result

    def world_reference_observations(self, node_id: int) -> np.ndarray:
        """Stored frame observations belonging to a world node's episodes."""
        leaves = self._subtree_leaves(node_id)
        episodes = {
            episode.episode for episode in self.episodes
            if self.episode_leaf.get(episode.episode) in leaves
        }
        rows = [
            frame.observation for frame in self.frames
            if frame.episode in episodes
        ]
        if not rows:
            return np.zeros((0, self.observation_dim), dtype=float)
        return np.stack(rows)

    def query_world(self, context_or_signature, max_distance=None):
        signature = np.asarray(context_or_signature, dtype=float)
        if signature.shape[-1] != self.signature_dim:
            signature = self._signature(signature)
        return self.world_tree.query(signature, max_distance=max_distance)

    def persistence(self, direction) -> dict:
        """Compare a direction across levels and world siblings.

        A persistent direction varies across coarse memories (phases,
        worlds) and carries support in many of them; a temporary one is a
        spike in one low-level node.  The level score is the variance ratio
        of the projections times their breadth, so a constant offset (no
        between-memory variation) scores zero and a single spike scores at
        most one over the number of memories.
        """
        direction = np.asarray(direction, dtype=float).reshape(self.context_dim)

        def _level(values):
            if not values:
                return dict(score=0.0, ratio=0.0, breadth=0.0,
                            mean=0.0, variance=0.0, count=0)
            projections = np.stack(values) @ direction
            mean = float(np.mean(projections))
            variance = float(np.var(projections))
            mean_square = float(np.mean(projections ** 2))
            ratio = variance / mean_square if mean_square > 1e-12 else 0.0
            maximum = float(np.max(np.abs(projections)))
            breadth = (
                float(np.mean(np.abs(projections) > 0.25 * maximum))
                if maximum > 1e-12 else 0.0
            )
            return dict(
                score=float(ratio * breadth), ratio=float(ratio),
                breadth=breadth, mean=mean, variance=variance,
                count=len(values),
            )

        levels = [
            _level([frame.context for frame in self.frames]),
            _level([event.context for event in self.events]),
            _level([phase.context_mean for phase in self.phases]),
            _level([episode.context_mean for episode in self.episodes]),
        ]
        weights = (0.1, 0.2, 0.3, 0.4)
        score = sum(
            weight * level["score"]
            for weight, level in zip(weights, levels)
        )
        world = self.world_tree.persistence(np.concatenate([
            direction,
            np.zeros(self.signature_dim - self.context_dim),
        ]))
        return dict(
            score=float(score),
            level_scores=[level["score"] for level in levels],
            level_fractions=[level["breadth"] for level in levels],
            world=world,
            temporary=float(1.0 - score),
        )

    def snapshot(self) -> dict:
        return dict(
            frames=len(self.frames), events=len(self.events),
            phases=len(self.phases), episodes=len(self.episodes),
            world_tree=self.world_tree.snapshot(),
            registry={
                key: value for key, value in self.registry.snapshot().items()
                if key.startswith(("level:", "root", "world:"))
            },
        )


def _summarize_vectors(vectors, weights, keyword_dim: int):
    """Weighted PCA summary used by phase paragraphs and world parents."""
    vectors = np.asarray(vectors, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("vectors must be a non-empty matrix")
    if weights.shape[0] != len(vectors):
        raise ValueError("one weight per vector")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    mean = (vectors * weights[:, None]).sum(axis=0) / total
    centered = vectors - mean
    weighted = centered * np.sqrt(weights)[:, None]
    _u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    keep = min(max(int(keyword_dim), 0), len(vectors), vectors.shape[1])
    if keep <= 0 or singular[0] <= 1e-12:
        return dict(
            mean=mean, keywords=None, weights=None,
            residual=float(((centered ** 2) * weights[:, None]).sum() / total),
        )
    directions = vt[:keep]
    codes = centered @ directions.T
    reconstruction = codes @ directions
    residual = float(
        (((centered - reconstruction) ** 2) * weights[:, None]).sum() / total
    )
    return dict(
        mean=mean, keywords=directions,
        weights=(singular[:keep] ** 2) / total,
        residual=residual,
    )
