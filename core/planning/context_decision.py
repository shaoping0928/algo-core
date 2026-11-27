"""Unified Context-Driven Decision Controller.

Connects long-term knowledge reserves with short-term temporary context so that
active context genuinely steers online decisions rather than passively
re-weighting frequency-dominated candidates:
  - Long-term knowledge reserves:
      * ``ClassLevelSequencePolicy`` (abstract class transitions + <=2-gram legality)
      * ``BidirectionalVirtualNodeDAG`` (forward Post, backward Pre, 2-step bridge)
      * Optional ``HierarchicalStageTree`` + ``LearnedDrillDownPolicy``
  - Short-term active context:
      * ``TemporaryContextMemory`` (live node-strength spreading field + episodic cache)
      * Tree-attended role slots from earlier stages of the current episode

Decision flow at each step:
  1. Combine short-term node strengths from ``TemporaryContextMemory`` and
     (when available) top-down ``HierarchicalStageTree.attend()`` slots.
  2. Propose structural candidates from ``ClassLevelSequencePolicy``.
  3. **Active Context Injection & 2-Step Bridge Steering**:
     - Direct injection: high-strength active context nodes that are locally
       legal after ``ctx[-1]`` (via ``local_index`` or ``Post(ctx[-1])``) are
       actively injected into the candidate pool.
     - Goal-directed 2-step bridge: for strongly activated context nodes ``g``,
       ``dag.bridge_lookup(ctx[-1], g)`` identifies bridging symbols ``b``
       connecting ``ctx[-1] -> b -> g``, steering the decision toward active
       context concepts when local entropy is high.
  4. Entropy-aware gating: when local transition entropy is low (mid-compound
     completion), local legality dominates; at high-entropy choice points,
     short-term context node strengths steer the next concept.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from class_policy import ClassLevelSequencePolicy
from compression_tree import HierarchicalStageTree, LearnedDrillDownPolicy
from temp_context import TemporaryContextMemory
from virtual_node_dag import BidirectionalVirtualNodeDAG


class ContextDrivenDecisionController:
    """Steers next-symbol decisions using live short-term context over long-term reserves."""

    def __init__(
        self,
        class_policy: ClassLevelSequencePolicy,
        virtual_dag: BidirectionalVirtualNodeDAG,
        temp_memory: TemporaryContextMemory,
        *,
        stage_tree: Optional[HierarchicalStageTree] = None,
        drill_policy: Optional[LearnedDrillDownPolicy] = None,
        local_weight: float = 0.5,
        semantic_gamma: float = 1.6,
        field_beta: float = 1.0,
        cache_lambda: float = 0.35,
        active_inject_weight: float = 0.22,
        bridge_weight: float = 0.14,
        top_active_targets: int = 16,
        content_class_gate: bool = True,
        lookahead_depth: int = 1,
        lookahead_beam: int = 4,
    ):
        self.class_policy = class_policy
        self.virtual_dag = virtual_dag
        self.temp_memory = temp_memory
        self.stage_tree = stage_tree
        self.drill_policy = drill_policy
        self.local_weight = float(local_weight)
        self.semantic_gamma = float(semantic_gamma)
        self.field_beta = float(field_beta)
        self.cache_lambda = float(cache_lambda)
        self.active_inject_weight = float(active_inject_weight)
        self.bridge_weight = float(bridge_weight)
        self.top_active_targets = max(int(top_active_targets), 4)
        self.content_class_gate = bool(content_class_gate)
        self.lookahead_depth = max(int(lookahead_depth), 1)
        self.lookahead_beam = max(int(lookahead_beam), 2)

    def _local_pair_prob_and_lift(self, prev1: int, target: int) -> Tuple[float, float]:
        post = self.class_policy.local_index._postings.get((-1, int(prev1)))
        if not post:
            return 0.0, 0.0
        tot = self.class_policy.local_index._key_totals.get((-1, int(prev1)), 0.0)
        if tot <= 0.0:
            return 0.0, 0.0
        bucket = post.get(int(target))
        if bucket is None:
            return 0.0, 0.0
        p_loc = max(float(bucket[0]) - 0.5, 0.0) / float(tot)
        gl = self.class_policy.feature_map.global_probs_
        g_p = float(gl[int(target)]) if gl is not None and int(target) < len(gl) else 1e-4
        lift = p_loc / max(g_p, 1e-8)
        return p_loc, lift

    def _collect_active_node_strengths(
        self,
        slow_state: Optional[np.ndarray],
    ) -> Tuple[Dict[int, float], Dict[str, int]]:
        """Merge temporary working-memory field (including root summary) with tree slots."""
        strengths: Dict[int, float] = {}
        tree_stats = dict(nodes_visited=0, leaves_total=0, drill_downs=0, parent_hits=0)

        field = self.temp_memory.effective_field(root_blend=0.45)
        f_max = float(field.max())
        if f_max > 1e-6:
            inv_f = 1.0 / f_max
            k_cap = min(self.top_active_targets, len(field) - 1)
            top_idx = np.argpartition(-field, k_cap)[:k_cap]
            for idx in top_idx:
                i = int(idx)
                val = float(field[i]) * inv_f
                if i > 0 and val > 0.15:
                    strengths[i] = val

        if self.stage_tree is not None and slow_state is not None:
            hits, tree_stats = self.stage_tree.attend(
                slow_state, policy=self.drill_policy, top_k=2,
            )
            for node, sim in hits:
                if sim <= 0.0:
                    continue
                for slot in node.slots:
                    s_id = int(slot.symbol)
                    if s_id > 0:
                        boost = 0.5 * float(sim) * float(slot.confidence)
                        strengths[s_id] = min(strengths.get(s_id, 0.0) + boost, 1.5)

        return strengths, tree_stats

    def decide_distribution(
        self,
        ctx_tokens: Sequence[int],
        slow_state: Optional[np.ndarray] = None,
        *,
        k: int = 8,
    ) -> Tuple[List[int], np.ndarray, Dict[str, float]]:
        """Compute context-driven top-k candidate distribution + telemetry."""
        prev1 = int(ctx_tokens[-1]) if len(ctx_tokens) > 0 else 0
        # 1. Long-term structural proposal (ClassPolicy + <=2-gram legality)
        cand, probs = self.class_policy.propose_next_symbols(
            ctx_tokens,
            slow_state,
            k=max(int(k) * 2, 16),
            top_c=16,
            local_weight=self.local_weight,
            semantic_gamma=self.semantic_gamma,
        )
        # Forward virtual node coherence boost from Post(prev1)
        fnode = self.virtual_dag.forward_node(prev1) if prev1 > 0 else None
        if fnode is not None:
            probs = np.array(
                [float(pr) * (1.25 if int(a) in fnode.members else 1.0) for a, pr in zip(cand, probs)],
                dtype=float,
            )
            p_s = float(probs.sum())
            if p_s > 0.0:
                probs /= p_s

        # Compute normalized local entropy in [0, 1] to gate context steering
        p_clip = np.clip(probs, 1e-12, 1.0)
        ent = float(-np.sum(p_clip * np.log(p_clip))) / max(float(np.log(len(probs))), 1e-6)
        ent_gate = float(np.clip(ent, 0.15, 1.0))

        # 2. Episodic cache + spreading-field modulation on existing pool
        cand, probs = self.temp_memory.apply_to_candidates(
            ctx_tokens, cand, probs,
            field_beta=self.field_beta * ent_gate,
            cache_lambda=self.cache_lambda,
            extra_pool_k=max(int(k) * 2, 16),
        )
        score_map: Dict[int, float] = {int(a): float(p) for a, p in zip(cand, probs) if int(a) > 0}

        # 3. Active Context Injection & Bidirectional 2-Step Bridge Steering
        active_strengths, tree_stats = self._collect_active_node_strengths(slow_state)
        injected = 0
        bridged = 0
        # Content-class gate: activate PMI content injection only when ClassPolicy
        # predicts a content class rather than a confident singleton glue token.
        content_gate = 1.0
        if self.content_class_gate and self.class_policy.n_singletons > 0:
            cids, cprobs = self.class_policy.predict_next_classes(ctx_tokens, top_c=8)
            p_tot = float(cprobs.sum())
            if p_tot > 1e-12:
                top1_is_singleton = int(cids[0]) <= self.class_policy.n_singletons
                top1_share = float(cprobs[0]) / p_tot
                p_content = float(
                    sum(float(cp) for cid, cp in zip(cids, cprobs) if cid > self.class_policy.n_singletons)
                ) / p_tot
                if top1_is_singleton and top1_share >= 0.30:
                    content_gate = 0.0
                else:
                    content_gate = float(np.clip(p_content, 0.0, 1.0))
        if prev1 > 0 and active_strengths:
            top_active = sorted(active_strengths.items(), key=lambda kv: kv[1], reverse=True)[
                : self.top_active_targets
            ]
            w_inj = self.active_inject_weight * ent_gate * content_gate
            w_br = self.bridge_weight * ent_gate * content_gate
            for goal_sym, strength in top_active:
                if goal_sym == prev1:
                    continue
                # 3a. Direct 1-step legal activation scored by PMI lift (avoids high-freq suppression)
                p_loc, lift = self._local_pair_prob_and_lift(prev1, goal_sym)
                in_post = (fnode is not None and goal_sym in fnode.members)
                if (p_loc > 0.0 and lift >= 1.2) or in_post:
                    lift_factor = float(np.log1p(max(lift, 1.0))) / 3.0
                    add_mass = w_inj * float(strength) * min(lift_factor, 1.5)
                    score_map[goal_sym] = score_map.get(goal_sym, 0.0) + add_mass
                    injected += 1
                # 3b. 2-step goal-directed bridge: prev1 -> bridge -> goal_sym
                bridges = self.virtual_dag.bridge_lookup(prev1, goal_sym)
                if bridges:
                    for b_sym, _bw in bridges[:2]:
                        if b_sym > 0 and b_sym != prev1:
                            p_b, lift_b = self._local_pair_prob_and_lift(prev1, b_sym)
                            if p_b > 0.0 and lift_b >= 1.2:
                                lb_factor = float(np.log1p(max(lift_b, 1.0))) / 3.0
                                score_map[b_sym] = score_map.get(b_sym, 0.0) + (
                                    w_br * float(strength) * min(lb_factor, 1.5)
                                )
                                bridged += 1

        ranked = sorted(score_map.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[: max(int(k), 1)]
        acts = [a for a, _ in ranked]
        out_p = np.asarray([p for _, p in ranked], dtype=float)
        out_sum = float(out_p.sum())
        if out_sum > 0.0:
            out_p /= out_sum
        info = dict(
            entropy_gate=ent_gate,
            content_gate=content_gate,
            active_injected=float(injected),
            active_bridged=float(bridged),
            tree_visited=float(tree_stats["nodes_visited"]),
            lookahead_depth=float(self.lookahead_depth),
        )
        return acts, out_p, info

    def decide_with_lookahead(
        self,
        ctx_tokens: Sequence[int],
        slow_state: Optional[np.ndarray] = None,
        *,
        k: int = 8,
        depth: Optional[int] = None,
        beam_width: Optional[int] = None,
    ) -> Tuple[List[int], np.ndarray, Dict[str, float]]:
        """Multi-step lookahead beam decision over ``decide_distribution``.

        When ``depth <= 1``, delegates directly to 1-step ``decide_distribution``.
        When ``depth >= 2``, rolls forward the top ``beam_width`` candidate
        paths for ``depth`` steps and ranks each first action by the best
        cumulative path log-likelihood + active-field continuity.
        """
        d = self.lookahead_depth if depth is None else max(int(depth), 1)
        bw = self.lookahead_beam if beam_width is None else max(int(beam_width), 2)
        acts1, probs1, info = self.decide_distribution(ctx_tokens, slow_state, k=max(int(k), bw))
        if d <= 1 or len(acts1) <= 1:
            return acts1[:k], probs1[:k] / max(float(probs1[:k].sum()), 1e-12), info

        base_ctx = [int(x) for x in ctx_tokens[-6:]]
        eff_field = self.temp_memory.effective_field(root_blend=0.45)
        f_max = max(float(eff_field.max()), 1e-8)

        # Beam entries: (cum_score, first_action, rolled_ctx, rolled_slow)
        beam = []
        for a, p in zip(acts1[:bw], probs1[:bw]):
            h_next = (
                self.class_policy.update_slow_state(slow_state, a)
                if slow_state is not None else None
            )
            f_bonus = 0.15 * float(eff_field[a]) / f_max if a > 0 else 0.0
            beam.append((float(np.log(max(float(p), 1e-9))) + f_bonus, int(a), base_ctx + [int(a)], h_next))

        for _step in range(1, d):
            next_beam = []
            for sc, first_a, r_ctx, r_h in beam:
                a_sub, p_sub, _ = self.decide_distribution(r_ctx, r_h, k=bw)
                for a2, p2 in zip(a_sub[:bw], p_sub[:bw]):
                    h2 = (
                        self.class_policy.update_slow_state(r_h, a2)
                        if r_h is not None else None
                    )
                    f_bonus = 0.15 * float(eff_field[a2]) / f_max if a2 > 0 else 0.0
                    next_beam.append((
                        sc + 0.75 * float(np.log(max(float(p2), 1e-9))) + f_bonus,
                        first_a,
                        r_ctx + [int(a2)],
                        h2,
                    ))
            next_beam.sort(key=lambda item: item[0], reverse=True)
            beam = next_beam[: max(bw * 2, 6)]

        # Aggregate best lookahead path score per first action
        best_by_first: Dict[int, float] = {}
        for sc, first_a, _rc, _rh in beam:
            if first_a not in best_by_first or sc > best_by_first[first_a]:
                best_by_first[first_a] = sc

        # Blend lookahead scores with 1-step tail candidates
        scored_first = []
        min_la = min(best_by_first.values()) if best_by_first else -10.0
        for a, p in zip(acts1, probs1):
            la_s = best_by_first.get(int(a), min_la - 1.0 + float(np.log(max(float(p), 1e-9))))
            scored_first.append((la_s, int(a)))
        scored_first.sort(key=lambda item: item[0], reverse=True)
        top = scored_first[: max(int(k), 1)]
        logits = np.asarray([s for s, _ in top], dtype=float)
        logits -= float(logits.max())
        out_p = np.exp(logits)
        out_p /= max(float(out_p.sum()), 1e-12)
        info["lookahead_depth"] = float(d)
        return [a for _, a in top], out_p, info
