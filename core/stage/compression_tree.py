"""Task-agnostic hierarchical stage compression tree with self-discovered role slots.

Implements the multi-level summarization and top-down confidence-gated
attention mechanism:
  1. ``SelfDiscoveredRoleExtractor`` learns ``n_roles`` structural role types
     directly from per-step dynamic profiles (relative position in stage,
     salience, local surprise, alignment to stage mean) without any hard-coded
     linguistic labels (such as subject/verb/object).
  2. ``HierarchicalStageTree`` builds a bottom-up tree of ``StageTreeNode``s
     where leaf nodes summarize raw spans and parent nodes summarize groups of
     child nodes into broader states, displacements, and aggregated role slots,
     with an intrinsic summary ``confidence`` derived from child coherence.
  3. ``attend(query_state, conf_threshold)`` performs sparse top-down
     attention: it inspects parent nodes first; if a parent's confidence clears
     ``conf_threshold``, its summary is used directly without touching its
     children; otherwise it drills down selectively into the most relevant
     children to recover fine-grained details.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from summarizer import _fit_deterministic_centroids


@dataclass(frozen=True)
class RoleSlotEntry:
    """One self-discovered key-information slot stored inside a stage tree node."""

    role_id: int
    symbol: int
    salience: float
    confidence: float


@dataclass(frozen=True)
class StageTreeNode:
    """Node in the hierarchical stage compression tree."""

    node_id: int
    level: int
    regime: int
    start: int
    end: int
    duration: int
    entry_state: np.ndarray
    mean_state: np.ndarray
    delta_state: np.ndarray
    slots: Tuple[RoleSlotEntry, ...]
    confidence: float
    children: Tuple["StageTreeNode", ...] = ()


class SelfDiscoveredRoleExtractor:
    """Unsupervised discovery of key-information role slots inside stages."""

    def __init__(self, n_roles: int = 3):
        self.n_roles = max(int(n_roles), 1)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.role_profiles_: Dict[int, Dict[str, float]] = {}

    @staticmethod
    def _span_step_profiles(
        span_tokens: np.ndarray,
        span_surprises: np.ndarray,
        mean_state: np.ndarray,
        embeddings: np.ndarray,
        salience: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-step 4D dynamic role features for non-UNK steps in a span.

        Features: ``[pos_frac, salience, surprise_norm, align_to_mean]``.
        """
        n = len(span_tokens)
        if n == 0:
            return np.zeros((0, 4), dtype=float), np.zeros(0, dtype=int)
        m_norm = float(np.linalg.norm(mean_state))
        rows = []
        valid_idx = []
        denom = float(max(n - 1, 1))
        for i in range(n):
            sym = int(span_tokens[i])
            if sym <= 0 or sym >= len(salience):
                continue
            sal = float(salience[sym])
            if sal <= 1e-6:
                continue
            pos_frac = float(i) / denom
            surp = float(span_surprises[i]) / 10.0
            if m_norm > 1e-6:
                align = float(np.dot(embeddings[sym], mean_state)) / m_norm
            else:
                align = 0.0
            rows.append((pos_frac, sal, surp, align))
            valid_idx.append(i)
        if not rows:
            return np.zeros((0, 4), dtype=float), np.zeros(0, dtype=int)
        return np.asarray(rows, dtype=float), np.asarray(valid_idx, dtype=int)

    def fit_from_spans(
        self,
        spans: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        embeddings: np.ndarray,
        salience: np.ndarray,
    ) -> "SelfDiscoveredRoleExtractor":
        """Learn ``n_roles`` role-slot prototypes purely from observed span dynamics."""
        all_rows = []
        for span_tok, span_surp, mean_st in spans:
            feats, _ = self._span_step_profiles(
                span_tok, span_surp, mean_st, embeddings, salience,
            )
            if len(feats) > 0:
                all_rows.append(feats)
        if not all_rows:
            self.mean_ = np.zeros(4, dtype=float)
            self.scale_ = np.ones(4, dtype=float)
            self.centers_ = np.zeros((self.n_roles, 4), dtype=float)
            return self
        mat = np.vstack(all_rows)
        self.mean_ = mat.mean(axis=0)
        sd = mat.std(axis=0)
        self.scale_ = np.where(sd > 1e-8, sd, 1.0)
        z = (mat - self.mean_) / self.scale_
        centers = _fit_deterministic_centroids(z, self.n_roles, max_iter=30)
        # Canonicalize role order by mean relative position in stage (early -> mid -> late)
        raw_centers = centers * self.scale_[None, :] + self.mean_[None, :]
        order = np.argsort(raw_centers[:, 0])
        self.centers_ = centers[order]
        raw_ordered = raw_centers[order]
        self.role_profiles_ = {
            int(r): dict(
                pos_frac=float(raw_ordered[r, 0]),
                salience=float(raw_ordered[r, 1]),
                surprise_norm=float(raw_ordered[r, 2]),
                mean_align=float(raw_ordered[r, 3]),
            )
            for r in range(len(self.centers_))
        }
        return self

    def extract_slots(
        self,
        span_tokens: np.ndarray,
        span_surprises: np.ndarray,
        mean_state: np.ndarray,
        embeddings: np.ndarray,
        salience: np.ndarray,
    ) -> Tuple[RoleSlotEntry, ...]:
        """Assign each learned role slot to its strongest representative symbol in the span."""
        if self.centers_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit SelfDiscoveredRoleExtractor before extracting slots")
        feats, valid_idx = self._span_step_profiles(
            span_tokens, span_surprises, mean_state, embeddings, salience,
        )
        if len(feats) == 0:
            return ()
        z = (feats - self.mean_) / self.scale_
        c_norm = np.einsum("ij,ij->i", self.centers_, self.centers_)[None, :]
        z_norm = np.einsum("ij,ij->i", z, z)[:, None]
        d2 = z @ (-2.0 * self.centers_.T) + z_norm + c_norm
        np.maximum(d2, 0.0, out=d2)
        role_assign = np.argmin(d2, axis=1)

        slots: List[RoleSlotEntry] = []
        for r in range(len(self.centers_)):
            idxs = np.flatnonzero(role_assign == r)
            if len(idxs) == 0:
                continue
            # Pick symbol maximizing salience * proximity to role prototype
            best_j = None
            best_score = -1.0
            for j in idxs:
                sym = int(span_tokens[int(valid_idx[int(j)])])
                prox = float(np.exp(-0.5 * float(d2[int(j), r])))
                sc = float(salience[sym]) * prox
                if sc > best_score:
                    best_score = sc
                    best_j = int(j)
            if best_j is not None:
                sym = int(span_tokens[int(valid_idx[best_j])])
                prox = float(np.exp(-0.5 * float(d2[best_j, r])))
                slots.append(RoleSlotEntry(
                    role_id=int(r),
                    symbol=sym,
                    salience=float(salience[sym]),
                    confidence=float(np.clip(prox, 0.0, 1.0)),
                ))
        return tuple(slots)


def _merge_child_slots(children: Sequence[StageTreeNode]) -> Tuple[RoleSlotEntry, ...]:
    """Aggregate child role slots into broader parent role slots by salience*confidence."""
    best_by_role: Dict[int, RoleSlotEntry] = {}
    for ch in children:
        for slot in ch.slots:
            prev = best_by_role.get(slot.role_id)
            cur_score = slot.salience * slot.confidence
            if prev is None or cur_score > prev.salience * prev.confidence:
                best_by_role[slot.role_id] = slot
    return tuple(best_by_role[r] for r in sorted(best_by_role))


def _coherence_confidence(children: Sequence[StageTreeNode]) -> float:
    """Measure how tightly a parent summarizes its children (in (0, 1]).

    Combines:
      - child state tightness: ``exp(-mean_L2_dev / scale)``
      - child regime purity: duration-weighted share of the dominant regime
      - mean child confidence (so higher levels naturally carry less detail
        certainty unless their subtree is genuinely homogeneous).
    """
    if len(children) <= 1:
        return float(children[0].confidence) if children else 1.0
    mat = np.vstack([ch.mean_state for ch in children]).astype(float)
    durs = np.asarray([ch.duration for ch in children], dtype=float)
    w = durs / max(float(durs.sum()), 1.0)
    center = (mat * w[:, None]).sum(axis=0)
    devs = np.linalg.norm(mat - center[None, :], axis=1)
    c_norm = max(float(np.linalg.norm(center)), 1e-6)
    rel_spread = float(np.dot(w, devs)) / c_norm
    tightness = float(np.exp(-2.0 * rel_spread))

    reg_dur: Dict[int, float] = {}
    for ch in children:
        reg_dur[ch.regime] = reg_dur.get(ch.regime, 0.0) + float(ch.duration)
    purity = max(reg_dur.values()) / max(float(durs.sum()), 1.0)
    child_conf = float(np.dot(w, np.asarray([ch.confidence for ch in children], dtype=float)))
    return float(np.clip(child_conf * (0.5 * tightness + 0.5 * purity), 0.05, 1.0))


class LearnedDrillDownPolicy:
    """Task-agnostic mental-action policy deciding ``USE_PARENT`` vs ``DRILL_DOWN``.

    Replaces a hand-coded ``conf_threshold`` by learning the net utility
    advantage of drilling down from real feedback:
        ``adv = (reward_drill - reward_parent) - step_cost * extra_nodes``
    conditioned on a 5D internal mental state:
        ``[1.0, node.confidence, query_sim, level_norm, budget_frac]``.
    """

    def __init__(self, step_cost: float = 0.01, lr: float = 0.15, ridge: float = 1e-3):
        self.step_cost = float(step_cost)
        self.lr = float(lr)
        self.ridge = float(ridge)
        # Mild neutral initialization: slightly prefers parent when confidence is high
        self.weights = np.array([0.05, -0.10, 0.0, 0.0, -0.05], dtype=float)
        self.updates = 0
        self.total_drills = 0
        self.total_parent_stops = 0

    @staticmethod
    def features(
        node: StageTreeNode,
        sim: float,
        max_level: int,
        budget_frac: float,
    ) -> np.ndarray:
        lvl_norm = float(node.level) / float(max(max_level, 1))
        return np.asarray(
            [1.0, float(node.confidence), float(sim), lvl_norm, float(budget_frac)],
            dtype=float,
        )

    def predict_drill_advantage(
        self,
        node: StageTreeNode,
        sim: float,
        max_level: int,
        budget_frac: float,
    ) -> float:
        x = self.features(node, sim, max_level, budget_frac)
        return float(np.dot(self.weights, x))

    def should_drill(
        self,
        node: StageTreeNode,
        sim: float,
        max_level: int,
        budget_frac: float,
        *,
        explore_eps: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> bool:
        if not node.children:
            return False
        if explore_eps > 0.0 and rng is not None and float(rng.rand()) < float(explore_eps):
            decision = bool(rng.rand() < 0.5)
        else:
            decision = self.predict_drill_advantage(node, sim, max_level, budget_frac) > 0.0
        if decision:
            self.total_drills += 1
        else:
            self.total_parent_stops += 1
        return decision

    def update(
        self,
        node: StageTreeNode,
        sim: float,
        max_level: int,
        budget_frac: float,
        *,
        reward_parent: float,
        reward_drill: float,
        extra_nodes: int,
    ) -> float:
        """Online gradient update from observed utility gain minus compute cost."""
        x = self.features(node, sim, max_level, budget_frac)
        target_adv = (float(reward_drill) - float(reward_parent)) - self.step_cost * float(extra_nodes)
        pred = float(np.dot(self.weights, x))
        err = target_adv - pred
        step = self.lr / (1.0 + 0.01 * float(self.updates))
        self.weights = (1.0 - step * self.ridge) * self.weights + step * err * x
        self.updates += 1
        return float(err)

    def snapshot(self) -> dict:
        return dict(
            updates=int(self.updates),
            step_cost=float(self.step_cost),
            weights=[round(float(v), 4) for v in self.weights],
            total_drills=int(self.total_drills),
            total_parent_stops=int(self.total_parent_stops),
        )


class HierarchicalStageTree:
    """Multi-level stage compression tree with top-down confidence-gated attention."""

    def __init__(self, roots: Sequence[StageTreeNode], leaves: Sequence[StageTreeNode]):
        self.roots: Tuple[StageTreeNode, ...] = tuple(roots)
        self.leaves: Tuple[StageTreeNode, ...] = tuple(leaves)
        self.max_level: int = max((n.level for n in self.roots), default=0)

    @classmethod
    def build_from_stage_chain(
        cls,
        chain: Sequence,
        tokens: Sequence[int],
        surprises: np.ndarray,
        embeddings: np.ndarray,
        salience: np.ndarray,
        role_extractor: SelfDiscoveredRoleExtractor,
        *,
        branch_factor: int = 4,
    ) -> "HierarchicalStageTree":
        """Recursively compress a Level-0 stage chain into a multi-level tree."""
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        bf = max(int(branch_factor), 2)
        next_id = 0

        # Build Level-0 leaf nodes with self-discovered role slots
        leaves: List[StageTreeNode] = []
        for st in chain:
            s0 = int(st.start)
            s1 = min(int(st.end), len(arr) - 1)
            slots = role_extractor.extract_slots(
                arr[s0 : s1 + 1], surprises[s0 : s1 + 1],
                st.mean_state, embeddings, salience,
            )
            # Leaf confidence: inversely related to surprise and novelty variance
            leaf_conf = float(np.clip(1.0 - 0.5 * float(st.mean_novelty), 0.1, 1.0))
            leaves.append(StageTreeNode(
                node_id=next_id,
                level=0,
                regime=int(st.regime),
                start=s0,
                end=s1,
                duration=int(st.duration),
                entry_state=np.asarray(st.entry_state, dtype=np.float32),
                mean_state=np.asarray(st.mean_state, dtype=np.float32),
                delta_state=np.asarray(st.delta_state, dtype=np.float32),
                slots=slots,
                confidence=leaf_conf,
                children=(),
            ))
            next_id += 1

        if not leaves:
            return cls(roots=(), leaves=())

        current: List[StageTreeNode] = leaves
        level = 1
        while len(current) > bf:
            parents: List[StageTreeNode] = []
            for i in range(0, len(current), bf):
                group = tuple(current[i : i + bf])
                durs = np.asarray([c.duration for c in group], dtype=float)
                w = durs / max(float(durs.sum()), 1.0)
                mean_st = np.sum(
                    np.vstack([c.mean_state for c in group]) * w[:, None],
                    axis=0,
                ).astype(np.float32)
                entry_st = group[0].entry_state
                exit_st = group[-1].entry_state + group[-1].delta_state
                delta_st = (exit_st - entry_st).astype(np.float32)
                # Dominant regime weighted by duration
                reg_dur: Dict[int, float] = {}
                for c in group:
                    reg_dur[c.regime] = reg_dur.get(c.regime, 0.0) + float(c.duration)
                dom_regime = max(reg_dur.items(), key=lambda kv: kv[1])[0]
                parent_conf = _coherence_confidence(group)
                parents.append(StageTreeNode(
                    node_id=next_id,
                    level=level,
                    regime=int(dom_regime),
                    start=group[0].start,
                    end=group[-1].end,
                    duration=int(sum(c.duration for c in group)),
                    entry_state=entry_st,
                    mean_state=mean_st,
                    delta_state=delta_st,
                    slots=_merge_child_slots(group),
                    confidence=parent_conf,
                    children=group,
                ))
                next_id += 1
            current = parents
            level += 1

        return cls(roots=current, leaves=leaves)

    @classmethod
    def build_from_continuous_stages(
        cls,
        stages: Sequence,
        *,
        branch_factor: int = 4,
    ) -> "HierarchicalStageTree":
        """Build the same hierarchical tree directly from ``StageRegimeSummarizer`` stages.

        Works unmodified on any continuous-control trajectory (e.g., Flappy,
        CartPole, MountainCar) with zero text-specific assumptions.
        """
        bf = max(int(branch_factor), 2)
        next_id = 0
        leaves: List[StageTreeNode] = []
        for st in stages:
            mean_st = np.asarray(st.mean_state, dtype=float)
            delta_st = np.asarray(st.delta_state, dtype=float)
            # Self-discovered continuous role slots: top-3 most active state dims
            top_dims = np.argsort(-np.abs(mean_st))[: min(3, len(mean_st))]
            slots = tuple(
                RoleSlotEntry(
                    role_id=int(r_i),
                    symbol=int(d),
                    salience=float(abs(mean_st[int(d)])),
                    confidence=1.0,
                )
                for r_i, d in enumerate(top_dims)
            )
            rel_delta = float(np.linalg.norm(delta_st)) / max(float(np.linalg.norm(mean_st)), 1e-6)
            leaf_conf = float(np.clip(np.exp(-0.5 * rel_delta), 0.1, 1.0))
            leaves.append(StageTreeNode(
                node_id=next_id,
                level=0,
                regime=int(st.regime),
                start=int(st.start),
                end=int(st.end),
                duration=int(st.duration),
                entry_state=np.asarray(st.entry_state, dtype=np.float32),
                mean_state=mean_st.astype(np.float32),
                delta_state=delta_st.astype(np.float32),
                slots=slots,
                confidence=leaf_conf,
                children=(),
            ))
            next_id += 1

        if not leaves:
            return cls(roots=(), leaves=())

        current: List[StageTreeNode] = leaves
        level = 1
        while len(current) > bf:
            parents: List[StageTreeNode] = []
            for i in range(0, len(current), bf):
                group = tuple(current[i : i + bf])
                durs = np.asarray([c.duration for c in group], dtype=float)
                w = durs / max(float(durs.sum()), 1.0)
                mean_st = np.sum(
                    np.vstack([c.mean_state for c in group]) * w[:, None],
                    axis=0,
                ).astype(np.float32)
                entry_st = group[0].entry_state
                exit_st = group[-1].entry_state + group[-1].delta_state
                delta_st = (exit_st - entry_st).astype(np.float32)
                reg_dur: Dict[int, float] = {}
                for c in group:
                    reg_dur[c.regime] = reg_dur.get(c.regime, 0.0) + float(c.duration)
                dom_regime = max(reg_dur.items(), key=lambda kv: kv[1])[0]
                parents.append(StageTreeNode(
                    node_id=next_id,
                    level=level,
                    regime=int(dom_regime),
                    start=group[0].start,
                    end=group[-1].end,
                    duration=int(sum(c.duration for c in group)),
                    entry_state=entry_st,
                    mean_state=mean_st,
                    delta_state=delta_st,
                    slots=_merge_child_slots(group),
                    confidence=_coherence_confidence(group),
                    children=group,
                ))
                next_id += 1
            current = parents
            level += 1
        return cls(roots=current, leaves=leaves)

    @staticmethod
    def _state_similarity(query: np.ndarray, node: StageTreeNode) -> float:
        q = np.asarray(query, dtype=float).reshape(-1)
        m = np.asarray(node.mean_state, dtype=float).reshape(-1)
        qn = float(np.linalg.norm(q))
        mn = float(np.linalg.norm(m))
        if qn <= 1e-8 or mn <= 1e-8:
            return 0.0
        return float(np.dot(q, m) / (qn * mn))

    def attend(
        self,
        query_state: np.ndarray,
        *,
        conf_threshold: float = 0.78,
        rel_threshold: float = 0.35,
        top_k: int = 2,
        policy: Optional[LearnedDrillDownPolicy] = None,
    ) -> Tuple[List[Tuple[StageTreeNode, float]], Dict[str, int]]:
        """Top-down confidence-gated attention over the stage compression tree.

        When ``policy`` (a :class:`LearnedDrillDownPolicy`) is provided, the
        ``USE_PARENT`` vs ``DRILL_DOWN`` decision is made by the learned
        mental-action policy instead of a fixed ``conf_threshold``.
        """
        if not self.roots:
            return [], dict(nodes_visited=0, leaves_total=0, drill_downs=0, parent_hits=0)

        visited = 0
        drill_downs = 0
        parent_hits = 0
        results: List[Tuple[StageTreeNode, float]] = []
        total_leaves = max(len(self.leaves), 1)

        frontier: List[StageTreeNode] = list(self.roots)
        while frontier:
            scored = []
            for node in frontier:
                visited += 1
                sim = self._state_similarity(query_state, node)
                scored.append((sim, node))
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = scored[: max(int(top_k), 1)]

            next_frontier: List[StageTreeNode] = []
            for sim, node in selected:
                if not node.children:
                    results.append((node, float(sim)))
                    continue
                if policy is not None:
                    budget_frac = float(visited) / float(total_leaves)
                    do_drill = policy.should_drill(
                        node, sim, self.max_level, budget_frac,
                    )
                else:
                    do_drill = not (
                        node.confidence >= float(conf_threshold)
                        and sim >= float(rel_threshold)
                    )
                if not do_drill:
                    results.append((node, float(sim)))
                    parent_hits += 1
                else:
                    drill_downs += 1
                    next_frontier.extend(node.children)
            frontier = next_frontier

        results.sort(key=lambda item: item[1], reverse=True)
        stats = dict(
            nodes_visited=visited,
            leaves_total=len(self.leaves),
            drill_downs=drill_downs,
            parent_hits=parent_hits,
        )
        return results, stats
