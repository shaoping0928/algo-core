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
from hierarchical_arbitrator import HierarchicalOnlineArbitrator
from recency_kernel import LearnedRecencyLagKernel
from structural_memory import HierarchicalChunkMemory, StructuralFrameMemory
from temp_context import TemporaryContextMemory
from virtual_node_dag import BidirectionalVirtualNodeDAG


class LearnedClassContextGate:
    """Task-agnostic per-class context gate learned strictly from prediction feedback.

    Starts completely uniform (0.5 for every class, zero distinction between
    singletons, function words, or content classes).  For each observed step,
    compares whether short-term context injection helped or hurt the predicted
    class and updates that class's gate weight online.
    """

    def __init__(self, n_classes: int, lr: float = 0.25, init_logit: float = 0.0):
        self.n_classes = max(int(n_classes), 1)
        self.lr = float(lr)
        self.logits = np.full(self.n_classes, float(init_logit), dtype=float)
        self.counts = np.zeros(self.n_classes, dtype=int)

    def class_gate(self, cid: int) -> float:
        i = int(cid)
        if i < 0 or i >= self.n_classes:
            return 0.5
        z = float(np.clip(self.logits[i], -6.0, 6.0))
        return 1.0 / (1.0 + float(np.exp(-z)))

    def gate_for_distribution(
        self, cids: Sequence[int], cprobs: np.ndarray,
    ) -> float:
        p_tot = float(np.sum(cprobs))
        if p_tot <= 1e-12:
            return 0.5
        acc = 0.0
        for cid, cp in zip(cids, cprobs):
            acc += (float(cp) / p_tot) * self.class_gate(int(cid))
        return float(np.clip(acc, 0.0, 1.0))

    def update(self, cid: int, advantage: float) -> None:
        """Update gate logit for ``cid`` from counterfactual advantage ``rew_with - rew_without``."""
        i = int(cid)
        if i < 0 or i >= self.n_classes:
            return
        self.counts[i] += 1
        step = self.lr / (1.0 + 0.02 * float(self.counts[i]))
        self.logits[i] = float(np.clip(self.logits[i] + step * float(advantage), -6.0, 6.0))

    def snapshot(self) -> dict:
        gates = 1.0 / (1.0 + np.exp(-np.clip(self.logits, -6.0, 6.0)))
        seen = self.counts > 0
        return dict(
            seen_classes=int(np.sum(seen)),
            mean_gate=round(float(gates[seen].mean()), 3) if np.any(seen) else 0.5,
            min_gate=round(float(gates[seen].min()), 3) if np.any(seen) else 0.5,
            max_gate=round(float(gates[seen].max()), 3) if np.any(seen) else 0.5,
        )


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
        learned_class_gate: Optional[LearnedClassContextGate] = None,
        lookahead_depth: int = 1,
        lookahead_beam: int = 4,
        lookahead_pmi_alpha: float = 0.18,
        lookahead_anchor_weight: float = 0.25,
        lookahead_stagnation_weight: float = 0.65,
        recency_kernel: Optional[LearnedRecencyLagKernel] = None,
        recency_strength: float = 0.85,
        chunk_memory: Optional[HierarchicalChunkMemory] = None,
        chunk_weight: float = 0.38,
        frame_memory: Optional[StructuralFrameMemory] = None,
        frame_weight: float = 0.18,
        arbitrator: Optional[HierarchicalOnlineArbitrator] = None,
        circuit_diff_weight: float = 0.60,
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
        self.learned_class_gate = (
            learned_class_gate
            if learned_class_gate is not None
            else (
                LearnedClassContextGate(class_policy.n_classes)
                if self.content_class_gate else None
            )
        )
        self.lookahead_depth = max(int(lookahead_depth), 1)
        self.lookahead_beam = max(int(lookahead_beam), 2)
        self.lookahead_pmi_alpha = float(lookahead_pmi_alpha)
        self.lookahead_anchor_weight = float(lookahead_anchor_weight)
        self.lookahead_stagnation_weight = float(lookahead_stagnation_weight)
        self.recency_kernel = recency_kernel
        self.recency_strength = float(recency_strength)
        self.chunk_memory = chunk_memory
        self.chunk_weight = float(chunk_weight)
        self.frame_memory = frame_memory
        self.frame_weight = float(frame_weight)
        self.arbitrator = arbitrator
        self.circuit_diff_weight = float(circuit_diff_weight)

        self._anchor_state: Optional[np.ndarray] = None
        self._tree_root_slots: Dict[int, float] = {}
        if self.stage_tree is not None:
            self.attach_stage_tree(self.stage_tree, drill_policy=self.drill_policy)

    def attach_stage_tree(
        self,
        stage_tree: Optional[HierarchicalStageTree],
        *,
        drill_policy: Optional[LearnedDrillDownPolicy] = None,
    ) -> None:
        """Bind a ``HierarchicalStageTree`` and pin its persistent root summary & role slots."""
        self.stage_tree = stage_tree
        if drill_policy is not None:
            self.drill_policy = drill_policy
        self._anchor_state = None
        self._tree_root_slots = {}
        if stage_tree is None or not stage_tree.roots:
            return
        durs = np.asarray([max(int(n.duration), 1) for n in stage_tree.roots], dtype=float)
        w = durs / max(float(durs.sum()), 1.0)
        self._anchor_state = np.sum(
            np.vstack([np.asarray(n.mean_state, dtype=float) for n in stage_tree.roots]) * w[:, None],
            axis=0,
        )
        slots_acc: Dict[int, float] = {}
        for n in stage_tree.roots:
            for sl in n.slots:
                s_id = int(sl.symbol)
                if s_id > 0:
                    val = float(sl.salience) * float(sl.confidence)
                    if val > slots_acc.get(s_id, 0.0):
                        slots_acc[s_id] = val
        self._tree_root_slots = slots_acc
        if slots_acc:
            self.temp_memory.pin_stage_anchors(slots_acc)

    def pin_anchor_state(self, state: Optional[np.ndarray]) -> None:
        """Pin an episode-level latent anchor state (e.g., end of external prompt)."""
        if state is None:
            return
        arr = np.asarray(state, dtype=float).reshape(-1)
        if np.linalg.norm(arr) > 1e-8:
            self._anchor_state = arr.copy()

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
        b_cnt = float(bucket[0]) if isinstance(bucket, list) else float(bucket)
        p_loc = max(b_cnt - 0.5, 0.0) / float(tot)
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

        if self._tree_root_slots:
            max_rs = max(self._tree_root_slots.values(), default=0.0)
            if max_rs > 1e-8:
                for s_id, raw_v in self._tree_root_slots.items():
                    boost = 0.45 * (float(raw_v) / max_rs)
                    strengths[s_id] = min(strengths.get(s_id, 0.0) + boost, 1.5)

        if self.stage_tree is not None and slow_state is not None:
            q_state = np.asarray(slow_state, dtype=float)
            if self._anchor_state is not None and len(self._anchor_state) == len(q_state):
                q_state = 0.55 * q_state + 0.45 * self._anchor_state
            hits, tree_stats = self.stage_tree.attend(
                q_state, policy=self.drill_policy, top_k=2,
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

    def _compute_expert_maps(
        self,
        ctx_tokens: Sequence[int],
        slow_state: Optional[np.ndarray],
        *,
        k: int = 8,
        recent_history: Optional[Sequence[int]] = None,
    ) -> Tuple[List[Dict[int, float]], float, float, Dict[str, float]]:
        """Compute calibrated probability maps for the 4 orthogonal memory experts."""
        prev1 = int(ctx_tokens[-1]) if len(ctx_tokens) > 0 else 0
        # Expert 0: Base ClassLevelSequencePolicy + Post(prev1)
        cand0, probs0 = self.class_policy.propose_next_symbols(
            ctx_tokens,
            slow_state,
            k=max(int(k) * 2, 16),
            top_c=16,
            local_weight=self.local_weight,
            semantic_gamma=self.semantic_gamma,
        )
        fnode = self.virtual_dag.forward_node(prev1) if prev1 > 0 else None
        if fnode is not None:
            probs0 = np.array(
                [float(pr) * (1.25 if int(a) in fnode.members else 1.0) for a, pr in zip(cand0, probs0)],
                dtype=float,
            )
            p_s = float(probs0.sum())
            if p_s > 0.0:
                probs0 /= p_s
        base_map: Dict[int, float] = {int(a): float(p) for a, p in zip(cand0, probs0) if int(a) > 0}

        p_clip = np.clip(probs0, 1e-12, 1.0)
        ent = float(-np.sum(p_clip * np.log(p_clip))) / max(float(np.log(len(probs0))), 1e-6)
        ent_gate = float(np.clip(ent, 0.15, 1.0))

        # Expert 1: Working Memory (corroborated field + novelty-gated cache + active PMI + 2-step bridge)
        use_nov_cache = bool(self.recency_kernel is not None and self.recency_kernel._fitted)
        cand1, probs1 = self.temp_memory.apply_to_candidates(
            ctx_tokens, cand0, probs0,
            field_beta=self.field_beta * ent_gate,
            cache_lambda=self.cache_lambda,
            extra_pool_k=max(int(k) * 2, 16),
            novelty_gated_cache=use_nov_cache,
        )
        wm_map: Dict[int, float] = {int(a): float(p) for a, p in zip(cand1, probs1) if int(a) > 0}

        active_strengths, tree_stats = self._collect_active_node_strengths(slow_state)
        injected = 0
        bridged = 0
        content_gate = 1.0
        if self.learned_class_gate is not None:
            cids, cprobs = self.class_policy.predict_next_classes(ctx_tokens, top_c=8)
            content_gate = self.learned_class_gate.gate_for_distribution(cids, cprobs)
        if prev1 > 0 and active_strengths:
            top_active = sorted(active_strengths.items(), key=lambda kv: kv[1], reverse=True)[
                : self.top_active_targets
            ]
            w_inj = self.active_inject_weight * ent_gate * content_gate
            w_br = self.bridge_weight * ent_gate * content_gate
            for goal_sym, strength in top_active:
                if goal_sym == prev1:
                    continue
                p_loc, lift = self._local_pair_prob_and_lift(prev1, goal_sym)
                in_post = (fnode is not None and goal_sym in fnode.members)
                if (p_loc > 0.0 and lift >= 1.2) or in_post:
                    lift_factor = float(np.log1p(max(lift, 1.0))) / 3.0
                    add_mass = w_inj * float(strength) * min(lift_factor, 1.5)
                    wm_map[goal_sym] = wm_map.get(goal_sym, 0.0) + add_mass
                    injected += 1
                bridges = self.virtual_dag.bridge_lookup(prev1, goal_sym)
                if bridges:
                    for b_sym, _bw in bridges[:2]:
                        if b_sym > 0 and b_sym != prev1:
                            p_b, lift_b = self._local_pair_prob_and_lift(prev1, b_sym)
                            if p_b > 0.0 and lift_b >= 1.2:
                                lb_factor = float(np.log1p(max(lift_b, 1.0))) / 3.0
                                wm_map[b_sym] = wm_map.get(b_sym, 0.0) + (
                                    w_br * float(strength) * min(lb_factor, 1.5)
                                )
                                bridged += 1

        # Expert 2: Higher-Order Recursive Chunks + Abstract Chunk-Class Grammar + Skip Schemas
        hist_src = (
            recent_history
            if recent_history is not None
            else (self.temp_memory.recent_history if self.temp_memory.recent_history else ctx_tokens)
        )
        chunk_map: Dict[int, float] = {}
        chunk_intra = 0.0
        chunk_inter = 0.0
        chunk_conf = 0.0
        if self.chunk_memory is not None and self.chunk_memory._fitted:
            ch_acts, ch_probs, ch_info = self.chunk_memory.predict_next_atoms(
                hist_src,
                slow_state=slow_state,
                atom_embeddings=self.class_policy.feature_map.embeddings_,
                local_prob_fn=self._local_pair_prob_and_lift,
                k=max(int(k) * 2, 16),
            )
            chunk_intra = float(ch_info.get("intra_mass", 0.0))
            chunk_inter = float(ch_info.get("inter_mass", 0.0))
            chunk_conf = float(ch_info.get("chunk_conf", 0.5))
            for a, p_ch in zip(ch_acts, ch_probs):
                ai = int(a)
                if ai > 0 and float(p_ch) > 0.0:
                    chunk_map[ai] = float(p_ch)

        # Expert 3: Directional Non-Adjacent Frame Register
        frame_map: Dict[int, float] = {}
        frame_hits = 0
        if self.frame_memory is not None and self.frame_memory._fitted:
            closers = self.frame_memory.predict_closers_for_history(hist_src)
            if closers:
                for c_sym, f_str in closers.items():
                    if c_sym <= 0 or c_sym == prev1:
                        continue
                    p_loc, lift = self._local_pair_prob_and_lift(prev1, c_sym) if prev1 > 0 else (0.0, 1.0)
                    if c_sym in wm_map or p_loc > 0.0 or f_str >= 1.5:
                        leg_scale = 0.5 + 0.5 * min(float(lift) / 2.0, 1.5) if p_loc > 0.0 else 0.75
                        frame_map[c_sym] = min(float(f_str) / 3.0, 1.5) * leg_scale
                        frame_hits += 1

        meta = dict(
            entropy=ent,
            entropy_gate=ent_gate,
            content_gate=content_gate,
            active_injected=float(injected),
            active_bridged=float(bridged),
            tree_visited=float(tree_stats["nodes_visited"]),
            lookahead_depth=float(self.lookahead_depth),
            chunk_intra=chunk_intra,
            chunk_inter=chunk_inter,
            chunk_conf=chunk_conf,
            frame_closers=float(frame_hits),
        )
        return [base_map, wm_map, chunk_map, frame_map], ent, ent_gate, meta

    def decide_distribution(
        self,
        ctx_tokens: Sequence[int],
        slow_state: Optional[np.ndarray] = None,
        *,
        k: int = 8,
        recent_history: Optional[Sequence[int]] = None,
        vdiff_override: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray, Dict[str, float]]:
        """Compute context-driven top-k candidate distribution + telemetry."""
        hist_src = (
            recent_history
            if recent_history is not None
            else (self.temp_memory.recent_history if self.temp_memory.recent_history else ctx_tokens)
        )
        expert_maps, ent, ent_gate, info = self._compute_expert_maps(
            ctx_tokens, slow_state, k=k, recent_history=hist_src,
        )
        _base_map, wm_map, chunk_map, frame_map = expert_maps

        if self.arbitrator is not None:
            # Cross-episode hierarchical online arbitration over the 4 orthogonal memory experts
            chunk_conf = float(info.get("chunk_conf", 0.5))
            exp_confs = (1.0, 1.0, 0.25 + 0.75 * chunk_conf, ent_gate)
            score_map, regime, _w = self.arbitrator.blend_experts(
                expert_maps, ent, expert_confidences=exp_confs,
            )
            info["arb_regime"] = float(regime)
            rec_str = self.arbitrator.recency_strength_for_regime(regime)
        else:
            # Standard direct composition path (preserves full backward compatibility)
            score_map = dict(wm_map)
            if chunk_map and self.chunk_weight > 0.0:
                sm_sum = float(sum(score_map.values()))
                if sm_sum > 0.0:
                    inv_sm = 1.0 / sm_sum
                    for a in list(score_map.keys()):
                        score_map[a] *= inv_sm
                chunk_conf = float(info.get("chunk_conf", 0.5))
                cw = float(np.clip(self.chunk_weight * (0.20 + 0.80 * chunk_conf), 0.0, 0.85))
                for a in list(score_map.keys()):
                    score_map[a] *= (1.0 - cw)
                for a, p_ch in chunk_map.items():
                    score_map[a] = score_map.get(a, 0.0) + cw * p_ch
            if frame_map and self.frame_weight > 0.0:
                fw = float(self.frame_weight) * ent_gate
                for c_sym, f_val in frame_map.items():
                    score_map[c_sym] = score_map.get(c_sym, 0.0) + fw * f_val
            rec_str = self.recency_strength

        # 5. Self-learned recency-lag modulation (excitation or inhibition purely from data)
        if self.recency_kernel is not None and rec_str > 0.0:
            score_map = self.recency_kernel.modulate_scores(
                hist_src, score_map, strength=rec_str,
            )

        # 6. Electronic Circuit Differentiator (high-pass V_diff = V_cap - V_int):
        #    propels unexpressed context neighbors (V_diff > 0) and naturally adapts
        #    downward on integrator-saturated nodes (V_diff < 0) without artificial bans.
        v_diff = (
            vdiff_override
            if vdiff_override is not None
            else self.temp_memory.circuit_differentiator()
        )
        if v_diff is not None and self.circuit_diff_weight > 0.0:
            dw = float(self.circuit_diff_weight)
            is_self_mode = (
                vdiff_override is not None
                or getattr(self.temp_memory, "_last_source", "external") == "self"
            )
            eff_dw = dw if is_self_mode else (0.25 * dw)
            for a in list(score_map.keys()):
                if 0 < a < len(v_diff):
                    dv = float(np.clip(v_diff[a], -2.0, 1.2))
                    if dv != 0.0:
                        score_map[a] *= float(np.exp(eff_dw * dv))

        ranked = sorted(score_map.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[: max(int(k), 1)]
        acts = [a for a, _ in ranked]
        out_p = np.asarray([p for _, p in ranked], dtype=float)
        out_sum = float(out_p.sum())
        if out_sum > 0.0:
            out_p /= out_sum
        return acts, out_p, info

    def learn_arbitrator_step(
        self,
        ctx_tokens: Sequence[int],
        slow_state: Optional[np.ndarray],
        actual_target: int,
        *,
        k: int = 8,
        recent_history: Optional[Sequence[int]] = None,
    ) -> bool:
        """Feed real target outcome into ``HierarchicalOnlineArbitrator`` (cross-episode consolidation)."""
        if self.arbitrator is None or int(actual_target) <= 0:
            return False
        tgt = int(actual_target)
        hist_src = (
            recent_history
            if recent_history is not None
            else (self.temp_memory.recent_history if self.temp_memory.recent_history else ctx_tokens)
        )
        expert_maps, ent, _ent_gate, _meta = self._compute_expert_maps(
            ctx_tokens, slow_state, k=k, recent_history=hist_src,
        )
        rec_adv = 0.0
        if self.recency_kernel is not None and self.recency_kernel._fitted:
            blended, _r, _w = self.arbitrator.blend_experts(expert_maps, ent)
            if blended:
                mod_on = self.recency_kernel.modulate_scores(hist_src, blended, strength=0.85)
                sc_off = self.arbitrator._expert_score(blended, tgt, k=k)
                sc_on = self.arbitrator._expert_score(mod_on, tgt, k=k)
                rec_adv = sc_on - sc_off
        return self.arbitrator.record_step(
            expert_maps, ent, tgt, recency_advantage=rec_adv, k=k,
        )

    def learn_rc_step(
        self,
        ctx_tokens: Sequence[int],
        slow_state: Optional[np.ndarray],
        actual_target: int,
        *,
        predicted_top: Optional[int] = None,
        lr: float = 0.12,
    ) -> None:
        """Online feedback adaptation of per-node ``(R, C)`` parameters on the circuit field."""
        cf = getattr(self.temp_memory, "circuit_field", None)
        if cf is None or int(actual_target) <= 0:
            return
        tgt = int(actual_target)
        if predicted_top is None:
            acts, _, _ = self.decide_distribution(ctx_tokens, slow_state, k=4)
            pred = int(acts[0]) if acts else 0
        else:
            pred = int(predicted_top)
        # Measure empirical lag since tgt last appeared in recent history
        hist = self.temp_memory.recent_history
        tgt_lag = -1
        for back in range(1, min(len(hist), 80) + 1):
            if int(hist[-back]) == tgt:
                tgt_lag = back
                break
        cf.learn_rc_feedback(tgt, pred, target_lag=tgt_lag, lr=lr)

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
        base_hist = [
            int(x)
            for x in (
                self.temp_memory.recent_history
                if self.temp_memory.recent_history
                else ctx_tokens
            )[-80:]
        ]
        eff_field = self.temp_memory.effective_field(root_blend=0.45)
        f_max = max(float(eff_field.max()), 1e-8)
        gl_probs = self.class_policy.feature_map.global_probs_
        anchor = self._anchor_state
        anchor_norm = float(np.linalg.norm(anchor)) if anchor is not None else 0.0

        def _step_utility(
            prev_sym: int,
            sym: int,
            prob: float,
            prev_h: Optional[np.ndarray],
            next_h: Optional[np.ndarray],
            branch_vdiff: Optional[np.ndarray] = None,
            branch_vout: Optional[np.ndarray] = None,
        ) -> float:
            # 1. Conditional PMI information progress with floored marginal (1e-3):
            #    penalizes ubiquitous zero-info sinks (g_p ~ 0.05) without over-rewarding
            #    ultra-rare idioms (g_p ~ 1e-4) over regular content symbols.
            ll = float(np.log(max(float(prob), 1e-9)))
            g_p = (
                float(gl_probs[sym])
                if gl_probs is not None and 0 <= sym < len(gl_probs)
                else 1e-3
            )
            pmi_term = ll - self.lookahead_pmi_alpha * float(np.log(max(g_p, 1e-3)))
            # 2. Branch-specific evolved GPU circuit field + active working-memory alignment
            if branch_vout is not None and 0 < sym < len(branch_vout):
                f_bonus = 0.22 * max(float(branch_vout[sym]), 0.0)
            else:
                f_bonus = 0.18 * float(eff_field[sym]) / f_max if sym > 0 else 0.0
            if branch_vdiff is not None and 0 < sym < len(branch_vdiff):
                f_bonus += 0.25 * float(np.clip(branch_vdiff[sym], -1.8, 1.2))
            # 3. Persistent macro-topic anchor alignment
            a_bonus = 0.0
            if anchor is not None and anchor_norm > 1e-8 and next_h is not None and len(next_h) == len(anchor):
                hn = float(np.linalg.norm(next_h))
                if hn > 1e-8:
                    a_bonus = self.lookahead_anchor_weight * float(np.dot(next_h, anchor) / (hn * anchor_norm))
            # 4. Absorbing zero-progress state discount (purely information/state-space based):
            #    penalizes stalling in a self-transition (sym == prev_sym) when latent state barely moves.
            stag_pen = 0.0
            if prev_sym > 0 and sym == prev_sym and self.lookahead_stagnation_weight > 0.0:
                if prev_h is not None and next_h is not None:
                    p_norm = max(float(np.linalg.norm(prev_h)), 1e-6)
                    disp = float(np.linalg.norm(next_h - prev_h)) / p_norm
                else:
                    disp = 0.0
                stag_pen = self.lookahead_stagnation_weight * max(1.0 - min(disp, 1.0), 0.0)
            return pmi_term + f_bonus + a_bonus - stag_pen

        # Optional GPU/CPU parallel multi-branch ElectronicCircuitBeamEngine
        cf = getattr(self.temp_memory, "circuit_field", None)
        beam_eng = None
        vd_cur: Optional[np.ndarray] = None
        vo_cur: Optional[np.ndarray] = None
        keep_cap = max(bw * 2, 6)
        sal_arr = self.temp_memory.salience_
        is_c_arr = getattr(self.temp_memory, "_is_content", None)

        def _sym_sal(sym_id: int) -> float:
            if sym_id <= 0 or (is_c_arr is not None and sym_id < len(is_c_arr) and not bool(is_c_arr[sym_id])):
                return 0.0
            if sal_arr is not None and sym_id < len(sal_arr):
                return float(sal_arr[sym_id])
            return 1.0

        root_vdiff = cf.V_diff[0] if cf is not None else None
        root_vout = cf.V_out[0] if cf is not None else None

        # Beam entries: (cum_score, first_action, rolled_ctx, rolled_hist, rolled_slow, gpu_slot_id)
        beam = []
        prev0 = base_ctx[-1] if base_ctx else 0
        first_acts = [int(a) for a in acts1[:bw]]
        first_probs = [float(p) for p in probs1[:bw]]
        for slot_i, (a, p) in enumerate(zip(first_acts, first_probs)):
            h_next = (
                self.class_policy.update_slow_state(slow_state, a)
                if slow_state is not None else None
            )
            sc0 = _step_utility(prev0, a, p, slow_state, h_next, root_vdiff, root_vout)
            beam.append((sc0, a, base_ctx + [a], base_hist + [a], h_next, slot_i))

        if cf is not None and len(beam) > 0:
            beam_eng = cf.get_beam_engine(max_branches=max(keep_cap, len(beam), 32))
            beam_eng.seed_root(cf.V_cap[0], cf.V_int[0])
            # Single fused GPU kernel evolves all first-step branches simultaneously!
            vd_cur, vo_cur = beam_eng.fork_and_step(
                [0] * len(beam),
                [b_ent[1] for b_ent in beam],
                [_sym_sal(b_ent[1]) for b_ent in beam],
                n_parent_active=1,
            )

        for _step in range(1, d):
            next_beam = []
            n_parents_cur = len(beam)
            for sc, first_a, r_ctx, r_hist, r_h, p_slot in beam:
                b_vd = vd_cur[p_slot] if (vd_cur is not None and p_slot < len(vd_cur)) else None
                b_vo = vo_cur[p_slot] if (vo_cur is not None and p_slot < len(vo_cur)) else None
                a_sub, p_sub, _ = self.decide_distribution(
                    r_ctx, r_h, k=bw, recent_history=r_hist, vdiff_override=b_vd,
                )
                prev_r = r_ctx[-1] if r_ctx else 0
                for a2, p2 in zip(a_sub[:bw], p_sub[:bw]):
                    ai2 = int(a2)
                    h2 = (
                        self.class_policy.update_slow_state(r_h, ai2)
                        if r_h is not None else None
                    )
                    u2 = _step_utility(prev_r, ai2, float(p2), r_h, h2, b_vd, b_vo)
                    next_beam.append((
                        sc + 0.75 * u2,
                        first_a,
                        r_ctx + [ai2],
                        r_hist + [ai2],
                        h2,
                        p_slot,
                        ai2,
                    ))
            next_beam.sort(key=lambda item: item[0], reverse=True)
            top_nb = next_beam[:keep_cap]
            if beam_eng is not None and (_step + 1) < d and top_nb:
                # Single fused GPU launch forks & evolves all surviving child branches for next depth!
                vd_cur, vo_cur = beam_eng.fork_and_step(
                    [item[5] for item in top_nb],
                    [item[6] for item in top_nb],
                    [_sym_sal(item[6]) for item in top_nb],
                    n_parent_active=n_parents_cur,
                )
            beam = [
                (item[0], item[1], item[2], item[3], item[4], new_slot)
                for new_slot, item in enumerate(top_nb)
            ]

        # Aggregate best lookahead path score per first action
        best_by_first: Dict[int, float] = {}
        for sc, first_a, _rc, _rhi, _rh, _sl in beam:
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
        info["beam_circuit_gpu"] = 1.0 if (beam_eng is not None and beam_eng.gpu is not None) else 0.0
        return [a for _, a in top], out_p, info

    def learn_gate_step(
        self,
        ctx_tokens: Sequence[int],
        slow_state: Optional[np.ndarray],
        actual_target: int,
        *,
        k: int = 8,
    ) -> float:
        """Online self-supervised update of ``learned_class_gate`` from real feedback.

        Compares counterfactual prediction quality with injection OFF vs ON
        and updates the predicted class's gate logit strictly from advantage.
        """
        if self.learned_class_gate is None or int(actual_target) <= 0:
            return 0.0
        tgt = int(actual_target)
        cids, _ = self.class_policy.predict_next_classes(ctx_tokens, top_c=1)
        pred_cid = int(cids[0]) if cids else 0

        # Counterfactual OFF vs ON
        orig_inj, orig_br, orig_gate = (
            self.active_inject_weight,
            self.bridge_weight,
            self.learned_class_gate,
        )
        try:
            self.learned_class_gate = None
            self.active_inject_weight = 0.0
            self.bridge_weight = 0.0
            cand_off, _, _ = self.decide_distribution(ctx_tokens, slow_state, k=k)

            self.active_inject_weight = max(orig_inj, 0.25)
            self.bridge_weight = max(orig_br, 0.15)
            cand_on, _, _ = self.decide_distribution(ctx_tokens, slow_state, k=k)
        finally:
            self.active_inject_weight = orig_inj
            self.bridge_weight = orig_br
            self.learned_class_gate = orig_gate

        def _score(cand_list: Sequence[int]) -> float:
            h1 = 1.0 if (cand_list and cand_list[0] == tgt) else 0.0
            h8 = 1.0 if (tgt in cand_list) else 0.0
            return 0.6 * h1 + 0.4 * h8

        adv = _score(cand_on) - _score(cand_off)
        self.learned_class_gate.update(pred_cid, adv)
        return float(adv)
