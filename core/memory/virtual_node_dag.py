"""Task-agnostic Bidirectional Transition-Image Virtual Node DAG.

Models the result set of any local state transition as a first-class *virtual
node* in both directions — completely free of task-specific labels:
  - Forward virtual node ``Post(x) = {y : x -> y}`` (reachable consequences)
  - Backward virtual node ``Pre(y) = {x : x -> y}`` (valid preconditions/causes)

Supports three task-agnostic set-theoretic DAG operations in both directions:
  1. **Containment edges** (``Parent ⊃ Child``): soft subset inclusion builds a
     multi-parent specialization DAG (like Quadrilateral ⊃ Parallelogram ⊃
     Rectangle/Rhombus ⊃ Square).
  2. **Intersection derivation** (``V_A ∩ V_B``): creates shared multi-parent
     child virtual nodes for overlapping transition images.
  3. **Difference purification** (``V_A \\ V_shared``): strips shared generic
     transitions to isolate operator-specific consequences or causes.
  4. **Bidirectional bridge lookup** (``Post(start) ∩ Pre(goal)``): meet-in-
     the-middle search connecting a start state/symbol to a goal.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class VirtualNode:
    """One virtual node representing a transition-image content set."""

    node_id: str
    direction: str  # "forward" (Post) or "backward" (Pre)
    kind: str  # "base", "intersection", or "difference"
    triggers: Tuple[int, ...]
    members: FrozenSet[int]
    member_weights: Dict[int, float] = field(default_factory=dict)
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.members)


class BidirectionalVirtualNodeDAG:
    """Bidirectional (forward Post + backward Pre) transition-image concept DAG."""

    def __init__(
        self,
        vocab_size: int,
        *,
        min_trigger_count: int = 20,
        min_pair_count: int = 3,
        min_lift: float = 2.5,
        max_members: int = 25,
        containment_threshold: float = 0.50,
        min_shared_members: int = 4,
    ):
        self.vocab_size = max(int(vocab_size), 2)
        self.min_trigger_count = max(int(min_trigger_count), 1)
        self.min_pair_count = max(int(min_pair_count), 1)
        self.min_lift = float(min_lift)
        self.max_members = max(int(max_members), 2)
        self.containment_threshold = float(containment_threshold)
        self.min_shared_members = max(int(min_shared_members), 2)

        self.nodes: Dict[str, VirtualNode] = {}
        self.forward_index: Dict[int, str] = {}
        self.backward_index: Dict[int, str] = {}
        self._member_to_forward: Dict[int, Set[str]] = defaultdict(set)
        self._member_to_backward: Dict[int, Set[str]] = defaultdict(set)
        self._fwd_counts: Dict[int, Counter] = defaultdict(Counter)
        self._bwd_counts: Dict[int, Counter] = defaultdict(Counter)
        self._pair_fwd_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self._pair_bwd_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self._global_p: Optional[np.ndarray] = None

    def _register_node(self, node: VirtualNode) -> VirtualNode:
        self.nodes[node.node_id] = node
        idx_map = self._member_to_forward if node.direction == "forward" else self._member_to_backward
        for m in node.members:
            idx_map[int(m)].add(node.node_id)
        return node

    def _build_base_node(
        self,
        trigger: int,
        counts_map: Counter,
        direction: str,
        global_p: np.ndarray,
    ) -> Optional[VirtualNode]:
        n_tot = float(sum(counts_map.values()))
        if n_tot < float(self.min_trigger_count):
            return None
        scored = []
        for target, cnt in counts_map.items():
            if cnt < self.min_pair_count:
                continue
            p_cond = float(cnt) / n_tot
            lift = p_cond / max(float(global_p[target]), 1e-12)
            if lift >= self.min_lift:
                weight = float(cnt) * float(np.log(max(lift, 1.0001)))
                scored.append((weight, int(target), float(lift)))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        top = scored[: self.max_members]
        members = frozenset(t for _w, t, _l in top)
        weights = {t: w for w, t, _l in top}
        prefix = "post" if direction == "forward" else "pre"
        return VirtualNode(
            node_id=f"{prefix}:{trigger}",
            direction=direction,
            kind="base",
            triggers=(int(trigger),),
            members=members,
            member_weights=weights,
        )

    def _link_containment_edges(self, node_ids: Sequence[str]) -> int:
        """Add soft-subset containment edges ``parent ⊃ child`` (strictly acyclic by size)."""
        ordered = sorted(
            (self.nodes[nid] for nid in node_ids),
            key=lambda n: (-n.size, n.node_id),
        )
        added = 0
        for i, parent in enumerate(ordered):
            if parent.size < self.min_shared_members + 1:
                continue
            for child in ordered[i + 1 :]:
                if child.size < self.min_shared_members or child.size >= parent.size:
                    continue
                inter_sz = len(parent.members & child.members)
                if inter_sz < self.min_shared_members:
                    continue
                containment = float(inter_sz) / float(child.size)
                if containment >= self.containment_threshold:
                    parent.children.add(child.node_id)
                    child.parents.add(parent.node_id)
                    added += 1
        return added

    def fit_transitions(
        self,
        pairs: Sequence[Tuple[int, int]],
        *,
        max_triggers_for_dag: int = 300,
    ) -> "BidirectionalVirtualNodeDAG":
        """Build forward (Post) and backward (Pre) virtual nodes and DAG edges from ``(src, dst)`` transitions."""
        counts = np.zeros(self.vocab_size, dtype=float)
        self._fwd_counts = defaultdict(Counter)
        self._bwd_counts = defaultdict(Counter)
        for src, dst in pairs:
            s = int(src)
            d = int(dst)
            if 0 < s < self.vocab_size and 0 < d < self.vocab_size:
                counts[s] += 1.0
                counts[d] += 1.0
                self._fwd_counts[s][d] += 1
                self._bwd_counts[d][s] += 1
        total = float(counts.sum())
        if total <= 0.0:
            return self
        self._global_p = (counts + 1.0) / (total + float(self.vocab_size))

        self.nodes.clear()
        self.forward_index.clear()
        self.backward_index.clear()
        self._member_to_forward.clear()
        self._member_to_backward.clear()

        fwd_ids: List[str] = []
        for s, cmap in self._fwd_counts.items():
            vn = self._build_base_node(s, cmap, "forward", self._global_p)
            if vn is not None:
                self._register_node(vn)
                self.forward_index[s] = vn.node_id
                fwd_ids.append(vn.node_id)

        bwd_ids: List[str] = []
        for d, cmap in self._bwd_counts.items():
            vn = self._build_base_node(d, cmap, "backward", self._global_p)
            if vn is not None:
                self._register_node(vn)
                self.backward_index[d] = vn.node_id
                bwd_ids.append(vn.node_id)

        # Link containment DAG edges among top triggers in each direction
        top_fwd = sorted(
            fwd_ids,
            key=lambda nid: sum(self._fwd_counts[self.nodes[nid].triggers[0]].values()),
            reverse=True,
        )[: max(int(max_triggers_for_dag), 1)]
        top_bwd = sorted(
            bwd_ids,
            key=lambda nid: sum(self._bwd_counts[self.nodes[nid].triggers[0]].values()),
            reverse=True,
        )[: max(int(max_triggers_for_dag), 1)]
        self._link_containment_edges(top_fwd)
        self._link_containment_edges(top_bwd)
        return self

    def fit_sequence(
        self,
        tokens: Sequence[int],
        *,
        max_triggers_for_dag: int = 300,
    ) -> "BidirectionalVirtualNodeDAG":
        """Build bidirectional virtual node DAG + 2-step behavioral statistics from a sequence."""
        arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
        if len(arr) < 2:
            return self
        pairs = list(zip(arr[:-1].tolist(), arr[1:].tolist()))
        self.fit_transitions(pairs, max_triggers_for_dag=max_triggers_for_dag)
        # Collect 2-step behavioral continuations (t, m) -> z and (z -> t, m) for bisimulation splitting
        self._pair_fwd_counts = defaultdict(Counter)
        self._pair_bwd_counts = defaultdict(Counter)
        v = self.vocab_size
        n = len(arr)
        for i in range(n - 2):
            a, b, c = int(arr[i]), int(arr[i + 1]), int(arr[i + 2])
            if 0 < a < v and 0 < b < v and 0 < c < v:
                self._pair_fwd_counts[(a, b)][c] += 1
                self._pair_bwd_counts[(b, c)][a] += 1
        return self

    def refine_virtual_node(
        self,
        node_id: str,
        *,
        max_splits: int = 4,
        min_member_support: int = 4,
        min_split_dissimilarity: float = 0.55,
    ) -> List[VirtualNode]:
        """Automatically split a coarse virtual node into finer child virtual nodes.

        Uses task-agnostic 2-step behavioral signatures (bisimulation):
          - For a forward node ``Post(T)``, each member ``m`` is characterized
            by its downstream continuation distribution ``Post(T, m)`` together
            with its cross-transition equivalence ``Pre(m)``.
          - For a backward node ``Pre(T)``, each member ``m`` is characterized
            by ``Pre(m, T)`` together with ``Post(m)``.
        Members with divergent behavioral signatures are split into distinct
        child virtual nodes under ``node_id`` in the DAG once sufficient data
        has accumulated.
        """
        node = self.nodes.get(node_id)
        if node is None or len(node.triggers) != 1 or node.size < 4:
            return []
        trig = int(node.triggers[0])
        is_fwd = node.direction == "forward"

        # Filter members with enough 2-step observations to justify splitting
        eligible: List[int] = []
        sigs: List[Dict[Tuple[int, int], float]] = []
        for m in sorted(node.members):
            pair_map = self._pair_fwd_counts[(trig, m)] if is_fwd else self._pair_bwd_counts[(m, trig)]
            pair_tot = float(sum(pair_map.values()))
            if pair_tot < float(min_member_support):
                continue
            cross_map = self._bwd_counts[m] if is_fwd else self._fwd_counts[m]
            cross_tot = float(sum(cross_map.values()))
            vec: Dict[Tuple[int, int], float] = {}
            # 2-step conditional behavior (higher weight: distinguishes 吃+不 vs 吃+饭 vs 吃+亏)
            for k_sym, cnt in pair_map.items():
                if self._global_p is not None:
                    lift = (float(cnt) / pair_tot) / max(float(self._global_p[k_sym]), 1e-12)
                    w = float(np.sqrt(float(cnt) / pair_tot) * max(np.log(max(lift, 1.0)), 0.1))
                else:
                    w = float(cnt) / pair_tot
                vec[(0, int(k_sym))] = w
            # Cross-transition neighborhood (Pre(m) for Post(T) members)
            if cross_tot > 0.0:
                for k_sym, cnt in cross_map.most_common(15):
                    if int(k_sym) == trig:
                        continue
                    vec[(1, int(k_sym))] = 0.6 * float(np.sqrt(float(cnt) / cross_tot))
            norm = float(np.sqrt(sum(v * v for v in vec.values())))
            if norm > 1e-8:
                for kk in vec:
                    vec[kk] /= norm
            eligible.append(int(m))
            sigs.append(vec)

        n_el = len(eligible)
        if n_el < 3:
            return []

        # Compute pairwise sparse cosine similarity matrix
        sim_mat = np.eye(n_el, dtype=float)
        for i in range(n_el):
            si = sigs[i]
            for j in range(i + 1, n_el):
                sj = sigs[j]
                dot = sum(v * sj[k] for k, v in si.items() if k in sj)
                sim_mat[i, j] = sim_mat[j, i] = float(dot)

        # Only split if behavioral divergence inside this virtual node is genuine
        off_diag = sim_mat[np.triu_indices(n_el, k=1)]
        if len(off_diag) == 0 or (1.0 - float(off_diag.min())) < float(min_split_dissimilarity):
            return []

        # Deterministic furthest-point medoid seeding + assignment on behavioral cosine distance
        k_sub = min(max(int(max_splits), 2), n_el)
        # Pick medoid seeds maximizing minimum distance (1 - sim)
        medoids = [int(np.argmin(sim_mat.mean(axis=1)))]
        while len(medoids) < k_sub:
            max_sim_to_med = sim_mat[:, medoids].max(axis=1)
            cand = int(np.argmin(max_sim_to_med))
            if (1.0 - float(max_sim_to_med[cand])) < float(min_split_dissimilarity):
                break
            medoids.append(cand)
        if len(medoids) < 2:
            return []

        # Assign every eligible member to its most behaviorally similar medoid
        assign = np.argmax(sim_mat[:, medoids], axis=1)
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx_m, g_id in enumerate(assign):
            groups[int(g_id)].append(eligible[idx_m])

        children: List[VirtualNode] = []
        for sub_idx, (_g, mems) in enumerate(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
            cid = f"{node.node_id}#sub{sub_idx}"
            mset = frozenset(mems)
            mweights = {m: node.member_weights.get(m, 1.0) for m in mems}
            child = VirtualNode(
                node_id=cid,
                direction=node.direction,
                kind="refined",
                triggers=node.triggers,
                members=mset,
                member_weights=mweights,
                parents={node.node_id},
            )
            node.children.add(cid)
            self._register_node(child)
            children.append(child)
        return children

    def derive_intersection(self, node_id_a: str, node_id_b: str) -> Optional[VirtualNode]:
        """Derive a multi-parent intersection child virtual node ``V_A ∩ V_B``."""
        na = self.nodes.get(node_id_a)
        nb = self.nodes.get(node_id_b)
        if na is None or nb is None or na.direction != nb.direction:
            return None
        inter = na.members & nb.members
        if not inter:
            return None
        trigs = tuple(sorted(set(na.triggers + nb.triggers)))
        prefix = "post_inter" if na.direction == "forward" else "pre_inter"
        nid = f"{prefix}:{'_'.join(str(t) for t in trigs)}"
        if nid in self.nodes:
            return self.nodes[nid]
        weights = {
            m: min(na.member_weights.get(m, 1.0), nb.member_weights.get(m, 1.0))
            for m in inter
        }
        child = VirtualNode(
            node_id=nid,
            direction=na.direction,
            kind="intersection",
            triggers=trigs,
            members=inter,
            member_weights=weights,
            parents={na.node_id, nb.node_id},
        )
        na.children.add(nid)
        nb.children.add(nid)
        return self._register_node(child)

    def derive_difference(self, node_id_base: str, node_id_shared: str) -> Optional[VirtualNode]:
        """Derive a purified child virtual node ``V_base \\ V_shared``."""
        nb = self.nodes.get(node_id_base)
        ns = self.nodes.get(node_id_shared)
        if nb is None or ns is None or nb.direction != ns.direction:
            return None
        diff = nb.members - ns.members
        if not diff:
            return None
        nid = f"{nb.node_id}_minus_{ns.node_id}"
        if nid in self.nodes:
            return self.nodes[nid]
        weights = {m: nb.member_weights.get(m, 1.0) for m in diff}
        child = VirtualNode(
            node_id=nid,
            direction=nb.direction,
            kind="difference",
            triggers=nb.triggers,
            members=diff,
            member_weights=weights,
            parents={nb.node_id},
        )
        nb.children.add(nid)
        return self._register_node(child)

    def forward_node(self, symbol: int) -> Optional[VirtualNode]:
        nid = self.forward_index.get(int(symbol))
        return self.nodes.get(nid) if nid is not None else None

    def backward_node(self, symbol: int) -> Optional[VirtualNode]:
        nid = self.backward_index.get(int(symbol))
        return self.nodes.get(nid) if nid is not None else None

    def containing_virtual_nodes(
        self, symbol: int, *, direction: str = "forward",
    ) -> List[VirtualNode]:
        """Return all virtual nodes in ``direction`` that contain ``symbol``."""
        idx_map = self._member_to_forward if direction == "forward" else self._member_to_backward
        nids = idx_map.get(int(symbol), ())
        nodes = [self.nodes[nid] for nid in nids]
        nodes.sort(key=lambda n: (-n.member_weights.get(int(symbol), 0.0), n.node_id))
        return nodes

    def bridge_lookup(self, start: int, goal: int) -> List[Tuple[int, float]]:
        """Bidirectional meet-in-the-middle: ``Post(start) ∩ Pre(goal)``."""
        fn = self.forward_node(start)
        bn = self.backward_node(goal)
        if fn is None or bn is None:
            return []
        inter = fn.members & bn.members
        if not inter:
            return []
        scored = [
            (
                int(mid),
                float(np.sqrt(
                    max(fn.member_weights.get(mid, 1.0), 0.0)
                    * max(bn.member_weights.get(mid, 1.0), 0.0)
                )),
            )
            for mid in inter
        ]
        scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        return scored

    def snapshot(self) -> dict:
        n_fwd = sum(1 for n in self.nodes.values() if n.direction == "forward")
        n_bwd = sum(1 for n in self.nodes.values() if n.direction == "backward")
        n_edges = sum(len(n.children) for n in self.nodes.values())
        return dict(
            total_nodes=len(self.nodes),
            forward_nodes=n_fwd,
            backward_nodes=n_bwd,
            dag_edges=n_edges,
        )
