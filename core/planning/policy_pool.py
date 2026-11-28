"""Task-agnostic Evolutionary Policy Pool: rapid birth, shadow validation, and eviction.

Encourages broad exploration of decision strategies (including deeper lookahead
horizons, content-class gating, and context-steering weights) while keeping the
active pool strictly bounded via continuous shadow validation and LCB/UCB
eviction:
  - **Birth**: periodically spawns mutated/crossed-over candidate policies at
    high cadence (``birth_every`` steps).
  - **Shadow Validation**: evaluates every candidate policy online against the
    same live feedback stream (reward = exact/plausible accuracy minus
    compute-depth cost).
  - **Promotion**: promotes a challenger when its lower confidence bound (LCB)
    exceeds the incumbent's mean (or LCB + margin) after ``min_trials``.
  - **Eviction**: aggressively evicts policies whose upper confidence bound
    (UCB) falls below the incumbent's LCB, or trims lowest-LCB stale policies
    whenever ``len(pool) > max_pool_size``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PolicySpec:
    """Parameter genome of a candidate decision strategy across all memory layers."""

    policy_id: int
    local_weight: float = 0.50
    semantic_gamma: float = 1.60
    field_beta: float = 1.00
    cache_lambda: float = 0.35
    active_inject_weight: float = 0.20
    bridge_weight: float = 0.12
    recency_strength: float = 0.85
    chunk_weight: float = 0.35
    frame_weight: float = 0.15
    content_class_gate: bool = True
    lookahead_depth: int = 1
    parent_id: Optional[int] = None
    born_step: int = 0


def apply_spec_to_controller(spec: PolicySpec, ctrl) -> None:
    """Apply a ``PolicySpec`` genome directly onto a ``ContextDrivenDecisionController``."""
    ctrl.local_weight = float(spec.local_weight)
    ctrl.semantic_gamma = float(spec.semantic_gamma)
    ctrl.field_beta = float(spec.field_beta)
    ctrl.cache_lambda = float(spec.cache_lambda)
    ctrl.active_inject_weight = float(spec.active_inject_weight)
    ctrl.bridge_weight = float(spec.bridge_weight)
    ctrl.recency_strength = float(spec.recency_strength)
    ctrl.chunk_weight = float(spec.chunk_weight)
    ctrl.frame_weight = float(spec.frame_weight)
    ctrl.content_class_gate = bool(spec.content_class_gate)
    ctrl.lookahead_depth = max(int(spec.lookahead_depth), 1)


@dataclass
class PolicyStats:
    """Online validation statistics for a candidate policy."""

    spec: PolicySpec
    trials: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    ewma_reward: float = 0.0
    hits_top1: int = 0
    hits_top8: int = 0

    @property
    def mean(self) -> float:
        return self.reward_sum / float(self.trials) if self.trials > 0 else 0.0

    @property
    def std_err(self) -> float:
        if self.trials < 2:
            return 0.5
        m = self.mean
        var = max(self.reward_sq_sum / float(self.trials) - m * m, 1e-6)
        return math.sqrt(var / float(self.trials))

    def lcb(self, z: float = 1.64) -> float:
        return self.mean - float(z) * self.std_err

    def ucb(self, z: float = 1.64) -> float:
        return self.mean + float(z) * self.std_err

    def record(self, reward: float, *, hit1: bool = False, hit8: bool = False, alpha: float = 0.05) -> None:
        r = float(reward)
        self.trials += 1
        self.reward_sum += r
        self.reward_sq_sum += r * r
        self.ewma_reward = r if self.trials == 1 else (1.0 - alpha) * self.ewma_reward + alpha * r
        if hit1:
            self.hits_top1 += 1
        if hit8:
            self.hits_top8 += 1


class PolicyPopulationManager:
    """Manages high-frequency strategy birth, online validation, promotion, and eviction."""

    def __init__(
        self,
        *,
        max_pool_size: int = 10,
        birth_every: int = 30,
        evict_every: int = 20,
        min_trials_promote: int = 25,
        min_trials_evict: int = 20,
        depth_cost: float = 0.004,
        z_score: float = 1.64,
        max_lookahead_depth: int = 3,
        seed: int = 0,
    ):
        self.max_pool_size = max(int(max_pool_size), 3)
        self.birth_every = max(int(birth_every), 5)
        self.evict_every = max(int(evict_every), 5)
        self.min_trials_promote = max(int(min_trials_promote), 5)
        self.min_trials_evict = max(int(min_trials_evict), 5)
        self.depth_cost = float(depth_cost)
        self.z_score = float(z_score)
        self.max_lookahead_depth = max(int(max_lookahead_depth), 1)
        self.rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

        self._next_id = 1
        self.step_count = 0
        self.total_births = 0
        self.total_promotions = 0
        self.total_evictions = 0
        self.eviction_log: List[Tuple[int, int, str]] = []

        # Seed diverse founder population spanning all structural & working-memory layers
        self.pool: Dict[int, PolicyStats] = {}
        founders = [
            dict(
                active_inject_weight=0.0, bridge_weight=0.0,
                recency_strength=0.0, chunk_weight=0.0, frame_weight=0.0,
                content_class_gate=True, lookahead_depth=1,
            ),
            dict(
                active_inject_weight=0.25, bridge_weight=0.14,
                recency_strength=0.85, chunk_weight=0.0, frame_weight=0.0,
                content_class_gate=True, lookahead_depth=1,
            ),
            dict(
                active_inject_weight=0.25, bridge_weight=0.14,
                recency_strength=0.85, chunk_weight=0.30, frame_weight=0.12,
                content_class_gate=True, lookahead_depth=1,
            ),
            dict(
                active_inject_weight=0.30, bridge_weight=0.18,
                recency_strength=0.65, chunk_weight=0.50, frame_weight=0.22,
                content_class_gate=True, lookahead_depth=min(2, self.max_lookahead_depth),
            ),
        ]
        for cfg in founders:
            spec = PolicySpec(policy_id=self._alloc_id(), born_step=0, **cfg)
            self.pool[spec.policy_id] = PolicyStats(spec=spec)
            self.total_births += 1
        self.active_id: int = 1

    def _alloc_id(self) -> int:
        pid = self._next_id
        self._next_id += 1
        return pid

    @property
    def active_spec(self) -> PolicySpec:
        return self.pool[self.active_id].spec

    def spawn_candidate(self) -> PolicySpec:
        """Mutate / crossover top-performing policies to encourage broader strategies & deeper depth."""
        ranked = sorted(self.pool.values(), key=lambda st: st.lcb(self.z_score), reverse=True)
        parent_a = ranked[0].spec
        parent_b = ranked[int(self.rng.randint(min(len(ranked), 3)))].spec

        def _mut(val_a: float, val_b: float, lo: float, hi: float, scale: float) -> float:
            base = 0.5 * (val_a + val_b) if self.rng.rand() < 0.5 else val_a
            return float(np.clip(base + self.rng.normal(0.0, scale), lo, hi))

        # Actively explore deeper lookahead depths (1..max_lookahead_depth)
        if self.rng.rand() < 0.40:
            depth = int(self.rng.randint(1, self.max_lookahead_depth + 1))
        else:
            depth = int(np.clip(parent_a.lookahead_depth + int(self.rng.choice([-1, 0, 1])), 1, self.max_lookahead_depth))

        gate = (not parent_a.content_class_gate) if self.rng.rand() < 0.25 else parent_a.content_class_gate
        spec = PolicySpec(
            policy_id=self._alloc_id(),
            local_weight=_mut(parent_a.local_weight, parent_b.local_weight, 0.2, 0.85, 0.08),
            semantic_gamma=_mut(parent_a.semantic_gamma, parent_b.semantic_gamma, 0.5, 2.5, 0.25),
            field_beta=_mut(parent_a.field_beta, parent_b.field_beta, 0.2, 1.8, 0.20),
            cache_lambda=_mut(parent_a.cache_lambda, parent_b.cache_lambda, 0.05, 0.55, 0.08),
            active_inject_weight=_mut(parent_a.active_inject_weight, parent_b.active_inject_weight, 0.0, 0.50, 0.08),
            bridge_weight=_mut(parent_a.bridge_weight, parent_b.bridge_weight, 0.0, 0.30, 0.05),
            recency_strength=_mut(parent_a.recency_strength, parent_b.recency_strength, 0.0, 1.40, 0.16),
            chunk_weight=_mut(parent_a.chunk_weight, parent_b.chunk_weight, 0.0, 0.70, 0.10),
            frame_weight=_mut(parent_a.frame_weight, parent_b.frame_weight, 0.0, 0.45, 0.06),
            content_class_gate=gate,
            lookahead_depth=depth,
            parent_id=parent_a.policy_id,
            born_step=self.step_count,
        )
        self.pool[spec.policy_id] = PolicyStats(spec=spec)
        self.total_births += 1
        return spec

    def record_step_validation(
        self,
        eval_fn: Callable[[PolicySpec], Tuple[float, bool, bool]],
    ) -> Dict[str, int]:
        """Shadow-validate all policies in the pool, then trigger birth/promotion/eviction."""
        self.step_count += 1
        for pid, st in list(self.pool.items()):
            raw_reward, hit1, hit8 = eval_fn(st.spec)
            # Net utility accounts for deeper lookahead compute cost
            net_reward = float(raw_reward) - self.depth_cost * float(max(st.spec.lookahead_depth - 1, 0))
            st.record(net_reward, hit1=hit1, hit8=hit8)

        promoted = 0
        evicted = 0
        born = 0

        # 1. Promotion: promote challenger whose LCB beats incumbent's mean
        inc = self.pool[self.active_id]
        if inc.trials >= self.min_trials_promote:
            best_challenger = None
            for pid, st in self.pool.items():
                if pid == self.active_id or st.trials < self.min_trials_promote:
                    continue
                if st.lcb(self.z_score) > inc.mean:
                    if best_challenger is None or st.lcb(self.z_score) > best_challenger.lcb(self.z_score):
                        best_challenger = st
            if best_challenger is not None:
                self.active_id = best_challenger.spec.policy_id
                self.total_promotions += 1
                promoted = 1

        # 2. High-frequency Eviction: remove statistically dominated or excess policies
        if self.step_count % self.evict_every == 0:
            inc = self.pool[self.active_id]
            inc_lcb = inc.lcb(self.z_score)
            to_evict = []
            for pid, st in self.pool.items():
                if pid == self.active_id:
                    continue
                if st.trials >= self.min_trials_evict and st.ucb(self.z_score) < inc_lcb:
                    to_evict.append((pid, "dominated_ucb<inc_lcb"))
            for pid, reason in to_evict:
                if len(self.pool) > 2:
                    del self.pool[pid]
                    self.total_evictions += 1
                    self.eviction_log.append((self.step_count, pid, reason))
                    evicted += 1
            # Capacity cap eviction: drop lowest-LCB matured policies
            while len(self.pool) > self.max_pool_size:
                matured = [
                    st for pid, st in self.pool.items()
                    if pid != self.active_id and st.trials >= self.min_trials_evict
                ]
                candidates = matured if matured else [
                    st for pid, st in self.pool.items() if pid != self.active_id
                ]
                if not candidates:
                    break
                worst = min(candidates, key=lambda st: (st.lcb(self.z_score), st.trials))
                del self.pool[worst.spec.policy_id]
                self.total_evictions += 1
                self.eviction_log.append((self.step_count, worst.spec.policy_id, "capacity_cap"))
                evicted += 1

        # 3. High-frequency Birth: encourage wider strategy & depth exploration
        if self.step_count % self.birth_every == 0:
            self.spawn_candidate()
            born = 1

        return dict(born=born, promoted=promoted, evicted=evicted)

    def snapshot(self) -> dict:
        inc = self.pool[self.active_id]
        return dict(
            steps=int(self.step_count),
            pool_size=len(self.pool),
            total_births=int(self.total_births),
            total_promotions=int(self.total_promotions),
            total_evictions=int(self.total_evictions),
            active_id=int(self.active_id),
            active_depth=int(inc.spec.lookahead_depth),
            active_gate=bool(inc.spec.content_class_gate),
            active_inj=round(float(inc.spec.active_inject_weight), 3),
            active_br=round(float(inc.spec.bridge_weight), 3),
            active_recency=round(float(inc.spec.recency_strength), 3),
            active_chunk=round(float(inc.spec.chunk_weight), 3),
            active_frame=round(float(inc.spec.frame_weight), 3),
            active_local=round(float(inc.spec.local_weight), 3),
            active_mean=round(float(inc.mean), 4),
            active_lcb=round(float(inc.lcb(self.z_score)), 4),
        )
