"""A task-agnostic hierarchical memory for context vectors.

The structure is a bounded-fanout merge tree, in the spirit of a B-tree but
over metric vectors instead of ordered keys:

* leaves hold episode-level context estimates with pooled sufficient
  statistics and a small reservoir of raw vectors;
* when a level grows past ``max_fanout`` roots, the two closest roots are
  merged into a parent at the next level, so coarser levels hold
  progressively more abstract regimes;
* a leaf whose reservoir is full and whose within-leaf variance is high is
  split by two-means, so granularity is not fixed;
* low-support leaves can be decayed/pruned, which is how a temporary
  fluctuation disappears while a recurring regime survives.

Nothing here knows what a context vector means.  The tree only stores
means, scatter and support; persistence is measured by how much support a
direction carries across distinct memories and levels.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MemoryNode:
    node_id: int
    level: int
    mean: np.ndarray
    count: float
    scatter: np.ndarray
    leaf: bool
    children: List[int] = field(default_factory=list)
    last_seen: int = 0
    inserts: int = 0
    reservoir: List[np.ndarray] = field(default_factory=list)
    # A parent is a lossy paragraph summary of its children: a handful of
    # core directions ("keywords") plus the residual detail it dropped.
    keywords: Optional[np.ndarray] = None
    keyword_weights: Optional[np.ndarray] = None
    residual: float = 0.0

    def variance(self) -> float:
        if self.count <= 0.0:
            return 0.0
        # `scatter` is the per-dimension sum of squared deviations.
        return float(np.sum(self.scatter) / self.count)


class MemoryTree:
    """Bounded-fanout hierarchical clustering of context vectors."""

    def __init__(self, dim: int, max_fanout: int = 4,
                 leaf_capacity: int = 24, merge_radius: float = 0.5,
                 split_variance: float = 1.0, min_count: float = 0.1,
                 decay: float = 0.95, keyword_dim: int = 2):
        self.dim = int(dim)
        self.max_fanout = max(int(max_fanout), 2)
        self.leaf_capacity = max(int(leaf_capacity), 2)
        self.merge_radius = float(merge_radius)
        self.split_variance = float(split_variance)
        self.min_count = float(min_count)
        self.decay_rate = float(decay)
        self.keyword_dim = max(int(keyword_dim), 0)
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.nodes: Dict[int, MemoryNode] = {}
        self.parent: Dict[int, int] = {}
        self.levels: Dict[int, List[int]] = {}
        self.leaves: set = set()
        self._next_id = 0
        self._clock = 0
        self.splits = 0
        self.merges = 0
        self.inserts = 0

    # ---- node helpers -----------------------------------------------------
    def _new_node(self, mean, count=1.0, scatter=None,
                  level=0, leaf=True) -> int:
        node_id = self._next_id
        self._next_id += 1
        mean = np.asarray(mean, dtype=float).reshape(self.dim)
        scatter = (
            np.zeros(self.dim, dtype=float)
            if scatter is None else np.asarray(scatter, dtype=float)
        )
        node = MemoryNode(
            node_id=node_id, level=int(level), mean=mean,
            count=float(count), scatter=scatter, leaf=bool(leaf),
            last_seen=self._clock,
        )
        self.nodes[node_id] = node
        if leaf:
            self.leaves.add(node_id)
        self.levels.setdefault(node.level, []).append(node_id)
        return node_id

    def _absorb(self, node: MemoryNode, vector: np.ndarray,
                weight: float = 1.0) -> None:
        weight = float(weight)
        if weight <= 0.0:
            return
        new_count = node.count + weight
        delta = vector - node.mean
        node.mean = node.mean + delta * (weight / new_count)
        node.scatter = node.scatter + delta * (vector - node.mean) * weight
        node.count = new_count
        node.inserts += 1
        node.last_seen = self._clock

    def _pooled(self, first: MemoryNode, second: MemoryNode):
        total = first.count + second.count
        if total <= 0.0:
            return np.zeros(self.dim), 0.0, np.zeros(self.dim)
        mean = (
            first.mean * first.count + second.mean * second.count
        ) / total
        scatter = (
            first.scatter + second.scatter
            + (first.mean - mean) ** 2 * first.count
            + (second.mean - mean) ** 2 * second.count
        )
        return mean, total, scatter

    def _remove_from_level(self, node: MemoryNode) -> None:
        level = self.levels.get(node.level, [])
        if node.node_id in level:
            level.remove(node.node_id)
        if not level:
            self.levels.pop(node.level, None)

    def _attach(self, parent_id: int, child_id: int) -> None:
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        if child_id not in parent.children:
            parent.children.append(child_id)
        self.parent[child_id] = parent_id
        self._remove_from_level(child)

    def _merge_pair(self, first_id: int, second_id: int) -> int:
        first = self.nodes[first_id]
        second = self.nodes[second_id]
        mean, count, scatter = self._pooled(first, second)
        level = max(first.level, second.level) + 1
        parent_id = self._new_node(
            mean, count=count, scatter=scatter, level=level, leaf=False
        )
        self._attach(parent_id, first_id)
        self._attach(parent_id, second_id)
        self._summarize(parent_id)
        self.merges += 1
        return parent_id

    def _summarize(self, node_id: int) -> None:
        """Build the lossy parent summary: core directions plus residual."""
        node = self.nodes[node_id]
        children = [
            self.nodes[child]
            for child in node.children
            if child in self.nodes and self.nodes[child].count > 0.0
        ]
        if node.leaf or len(children) < 2 or self.keyword_dim <= 0:
            node.keywords = None
            node.keyword_weights = None
            node.residual = 0.0
            return
        means = np.stack([child.mean for child in children])
        weights = np.asarray([child.count for child in children], dtype=float)
        total = float(weights.sum())
        if total <= 0.0:
            node.keywords = None
            node.keyword_weights = None
            node.residual = 0.0
            return
        mean = (means * weights[:, None]).sum(axis=0) / total
        centered = means - mean
        weighted = centered * np.sqrt(weights)[:, None]
        _u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        keep = min(self.keyword_dim, len(children) - 1, self.dim)
        if keep <= 0 or singular[0] <= 1e-12:
            node.keywords = None
            node.keyword_weights = None
            node.residual = float(
                ((centered ** 2) * weights[:, None]).sum() / total
            )
            return
        directions = vt[:keep]
        codes = centered @ directions.T
        reconstruction = codes @ directions
        residual = float(
            (((centered - reconstruction) ** 2) * weights[:, None]).sum()
            / total
        )
        node.keywords = directions
        node.keyword_weights = (singular[:keep] ** 2) / total
        node.residual = residual

    def expand(self, node_id: int) -> List[dict]:
        """Reconstruct the children of a summary from its keywords.

        This is the outward direction: a parent's few core directions are
        enough to regenerate an approximate picture of its children, not
        their exact detail.  The per-child error is exactly the omitted
        residual.
        """
        node = self.nodes[node_id]
        if node.leaf or not node.children:
            return [dict(
                node=node_id, mean=node.mean.copy(), weight=node.count,
                reconstructed=node.mean.copy(), error=0.0,
            )]
        result = []
        for child_id in node.children:
            child = self.nodes.get(child_id)
            if child is None:
                continue
            if node.keywords is not None:
                code = (child.mean - node.mean) @ node.keywords.T
                reconstructed = node.mean + code @ node.keywords
            else:
                reconstructed = node.mean.copy()
            result.append(dict(
                node=child_id, mean=child.mean.copy(),
                weight=child.count, reconstructed=reconstructed,
                error=float(np.linalg.norm(child.mean - reconstructed)),
            ))
        return result

    def keyword_persistence(self, node_id: int,
                            similarity: float = 0.8) -> List[dict]:
        """Compare a parent's keywords with its siblings' keywords.

        A direction that recurs across sibling summaries is a persistent
        concept; one that only exists in this node is a local detail.
        """
        node = self.nodes[node_id]
        if node.keywords is None:
            return []
        parent_id = self.parent.get(node_id)
        if parent_id is not None:
            sibling_ids = [
                child for child in self.nodes[parent_id].children
                if child != node_id
            ]
        else:
            sibling_ids = [
                other for other in self.levels.get(node.level, [])
                if other != node_id
            ]
        siblings = [
            self.nodes[other] for other in sibling_ids
            if other in self.nodes
            and self.nodes[other].keywords is not None
        ]
        result = []
        for index, direction in enumerate(node.keywords):
            agreements = [
                float(np.max(np.abs(sibling.keywords @ direction)))
                for sibling in siblings
            ]
            maximum = max(agreements) if agreements else 0.0
            mean_agreement = (
                float(np.mean(agreements)) if agreements else 0.0
            )
            weight = (
                float(node.keyword_weights[index])
                if node.keyword_weights is not None else 0.0
            )
            result.append(dict(
                index=index, weight=weight,
                similarity=maximum, mean_similarity=mean_agreement,
                persistent=bool(maximum >= similarity),
                siblings=len(agreements),
            ))
        return result

    def _closest_pair(self, node_ids: Sequence[int]) -> Optional[Tuple[int, int]]:
        if len(node_ids) < 2:
            return None
        best = None
        for index, first in enumerate(node_ids):
            first_mean = self.nodes[first].mean
            for second in node_ids[index + 1:]:
                distance = float(np.linalg.norm(
                    first_mean - self.nodes[second].mean
                ))
                if best is None or distance < best[0]:
                    best = (distance, first, second)
        return None if best is None else (best[1], best[2])

    def _rebalance(self) -> None:
        # Merge the closest roots level by level until every level respects
        # the fanout.  A single root per level is fine; it represents the
        # coarsest summary available at that granularity.
        level = 0
        while True:
            roots = list(self.levels.get(level, []))
            if len(roots) <= self.max_fanout:
                if level > max(self.levels, default=0):
                    break
                level += 1
                continue
            pair = self._closest_pair(roots)
            if pair is None:
                break
            self._merge_pair(*pair)
            # Revisit this level: children were removed from it and the
            # parent was appended one level up.
            continue

    # ---- insertion --------------------------------------------------------
    def insert(self, vector, weight: float = 1.0,
               allow_split: bool = True) -> int:
        self._clock += 1
        self.inserts += 1
        vector = np.asarray(vector, dtype=float).reshape(self.dim)
        leaf_id = self._nearest_leaf_id(vector)
        if leaf_id is not None:
            leaf = self.nodes[leaf_id]
            distance = float(np.linalg.norm(vector - leaf.mean))
            if distance <= self.merge_radius:
                self._absorb(leaf, vector, weight)
                self._update_ancestors(leaf_id)
                leaf.reservoir.append(vector.copy())
                if allow_split and len(leaf.reservoir) > self.leaf_capacity:
                    self._split_leaf(leaf_id)
                return leaf_id
        leaf_id = self._new_node(vector, count=weight, level=0, leaf=True)
        self.nodes[leaf_id].reservoir.append(vector.copy())
        self._rebalance()
        return leaf_id

    def _update_ancestors(self, node_id: int) -> None:
        # Ancestor summaries are pooled from children; recompute the path
        # so a later merge sees current statistics.
        current = self.parent.get(node_id)
        while current is not None:
            node = self.nodes[current]
            total = 0.0
            mean = np.zeros(self.dim)
            scatter = np.zeros(self.dim)
            for child_id in node.children:
                child = self.nodes[child_id]
                total += child.count
                mean += child.mean * child.count
            if total > 0.0:
                mean /= total
                for child_id in node.children:
                    child = self.nodes[child_id]
                    scatter += child.scatter + (child.mean - mean) ** 2 * child.count
                node.mean = mean
                node.scatter = scatter
                node.count = total
                self._summarize(current)
            current = self.parent.get(current)

    def _split_leaf(self, leaf_id: int) -> None:
        leaf = self.nodes[leaf_id]
        if leaf_id in self.parent:
            # Splitting a leaf that already belongs to a coarser node would
            # require rewriting the parent's child list; keep the coarser
            # summary and only trim the reservoir instead.
            leaf.reservoir = leaf.reservoir[-self.leaf_capacity // 2:]
            return
        points = np.asarray(leaf.reservoir[-self.leaf_capacity:], dtype=float)
        if len(points) < 4:
            leaf.reservoir = leaf.reservoir[-self.leaf_capacity // 2:]
            return
        spread = float(np.mean(np.linalg.norm(
            points - points.mean(axis=0), axis=1
        )))
        if spread <= self.split_variance:
            leaf.reservoir = leaf.reservoir[-self.leaf_capacity // 2:]
            return
        first, second = _two_means(points)
        if len(first) == 0 or len(second) == 0:
            leaf.reservoir = leaf.reservoir[-self.leaf_capacity // 2:]
            return
        # Split a root leaf into two finer roots; the rebalance step will
        # merge them back into a parent if the level exceeds the fanout.
        self.leaves.discard(leaf_id)
        self._remove_from_level(leaf)
        self.parent.pop(leaf_id, None)
        for group in (first, second):
            mean = group.mean(axis=0)
            scatter = ((group - mean) ** 2).sum(axis=0)
            new_id = self._new_node(
                mean, count=float(len(group)), scatter=scatter,
                level=leaf.level, leaf=True,
            )
            self.nodes[new_id].reservoir = [
                row.copy() for row in group
            ]
            self.nodes[new_id].inserts = len(group)
        del self.nodes[leaf_id]
        self.splits += 1
        self._rebalance()

    # ---- retrieval --------------------------------------------------------
    def _nearest_leaf_id(self, vector) -> Optional[int]:
        if not self.leaves:
            return None
        vector = np.asarray(vector, dtype=float).reshape(self.dim)
        best = None
        for leaf_id in self.leaves:
            leaf = self.nodes.get(leaf_id)
            if leaf is None:
                continue
            distance = float(np.linalg.norm(vector - leaf.mean))
            if best is None or distance < best[0]:
                best = (distance, leaf_id)
        return None if best is None else best[1]

    def query(self, vector, max_distance: Optional[float] = None):
        """Nearest leaf and its coarse-to-fine path."""
        vector = np.asarray(vector, dtype=float).reshape(self.dim)
        leaf_id = self._nearest_leaf_id(vector)
        if leaf_id is None:
            return None
        leaf = self.nodes[leaf_id]
        distance = float(np.linalg.norm(vector - leaf.mean))
        if max_distance is not None and distance > max_distance:
            return None
        path = []
        current = leaf_id
        while current is not None:
            path.append(self.nodes[current])
            current = self.parent.get(current)
        path.reverse()
        return dict(
            leaf=leaf, distance=distance, path=path,
            level=leaf.level, count=leaf.count,
        )

    # ---- persistence ------------------------------------------------------
    def persistence(self, direction, threshold: float = 0.1):
        """How much support a direction carries across distinct memories.

        A persistent direction appears with the same sign in several leaves
        and carries substantial total weight; a temporary one is confined to
        a leaf or two.  The coarse levels are reported separately so a
        direction that only separates low-level leaves can be distinguished
        from one that shapes a whole branch.
        """
        direction = np.asarray(direction, dtype=float).reshape(self.dim)
        total = 0.0
        aligned = 0.0
        leaves = 0
        levels = {}
        for leaf_id in self.leaves:
            leaf = self.nodes.get(leaf_id)
            if leaf is None or leaf.count <= 0.0:
                continue
            projection = float(leaf.mean @ direction)
            total += leaf.count
            if abs(projection) >= threshold:
                leaves += 1
                aligned += leaf.count
        for level, node_ids in self.levels.items():
            weight = 0.0
            mean = 0.0
            for node_id in node_ids:
                node = self.nodes[node_id]
                weight += node.count
                mean += node.count * float(node.mean @ direction)
            if weight > 0.0:
                levels[int(level)] = dict(
                    support=weight, mean=mean / weight,
                )
        return dict(
            support=aligned,
            total=total,
            fraction=(aligned / total if total > 0.0 else 0.0),
            leaves=leaves,
            total_leaves=len(self.leaves),
            levels=levels,
        )

    # ---- forgetting -------------------------------------------------------
    def decay(self) -> None:
        for leaf_id in list(self.leaves):
            leaf = self.nodes.get(leaf_id)
            if leaf is None:
                continue
            leaf.count *= self.decay_rate
            if leaf.count < self.min_count:
                self.leaves.discard(leaf_id)
                self._remove_from_level(leaf)
                self.parent.pop(leaf_id, None)
                del self.nodes[leaf_id]
        # Drop internal nodes with no children after leaf pruning.
        for node_id in list(self.nodes):
            node = self.nodes[node_id]
            if node.leaf:
                continue
            node.children = [
                child for child in node.children
                if child in self.nodes
            ]
            if not node.children:
                self._remove_from_level(node)
                del self.nodes[node_id]
            else:
                self._update_ancestors(node_id)

    def snapshot(self) -> dict:
        summarized = [
            node for node in self.nodes.values()
            if not node.leaf and node.keywords is not None
        ]
        return dict(
            nodes=len(self.nodes), leaves=len(self.leaves),
            levels={level: len(ids) for level, ids in self.levels.items()},
            splits=int(self.splits), merges=int(self.merges),
            inserts=int(self.inserts),
            summaries=len(summarized),
            mean_residual=float(np.mean([
                node.residual for node in summarized
            ])) if summarized else 0.0,
            total_support=float(sum(
                node.count for node in self.nodes.values() if node.leaf
            )),
        )


def _two_means(points: np.ndarray):
    """Deterministic two-means split used only by the leaf splitter."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points, points[:0]
    # Seed with the farthest pair.
    first = 0
    distances = np.linalg.norm(points - points[first], axis=1)
    second = int(np.argmax(distances))
    centers = np.stack([points[first], points[second]])
    labels = None
    for _ in range(6):
        distances = np.linalg.norm(
            points[:, None, :] - centers[None, :, :], axis=2
        )
        labels = np.argmin(distances, axis=1)
        for index in range(2):
            members = points[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    if labels is None:
        return points, points[:0]
    return points[labels == 0], points[labels == 1]
