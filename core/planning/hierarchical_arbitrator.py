"""Cross-Episode Hierarchical Online Arbitrator for Multi-Layer Structural Memory.

Prevents single-episode / single-article overfitting by separating:
  - **Fast intra-episode credit accumulation** across orthogonal memory experts
    (``base_policy``, ``working_memory``, ``chunk_grammar``, ``frame_register``)
    conditioned on data-driven local-entropy regimes (low / mid / high entropy);
  - **Slow cross-episode Mirror-Descent (Hedge) consolidation** at episode
    boundaries, bounding how much any single episode can shift global weights
    and shrinking gently toward balanced consensus.

Completely task-agnostic: operates solely on expert candidate distributions
and target feedback across RL episodes, symbolic streams, or multi-document text.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EXPERT_NAMES: Tuple[str, ...] = (
    "base_policy",
    "working_memory",
    "chunk_grammar",
    "frame_register",
)


class HierarchicalOnlineArbitrator:
    """Entropy-regime-conditioned cross-episode Hedge arbitrator over memory experts."""

    def __init__(
        self,
        *,
        n_regimes: int = 3,
        lr: float = 0.35,
        recency_lr: float = 0.25,
        shrinkage: float = 0.04,
        episode_window: int = 350,
        init_weights: Optional[Sequence[float]] = None,
        init_recency_strength: float = 0.80,
    ):
        self.expert_names = EXPERT_NAMES
        self.n_experts = len(self.expert_names)
        self.n_regimes = max(int(n_regimes), 1)
        self.lr = float(lr)
        self.recency_lr = float(recency_lr)
        self.shrinkage = float(np.clip(shrinkage, 0.0, 0.5))
        self.episode_window = max(int(episode_window), 20)

        if init_weights is not None and len(init_weights) == self.n_experts:
            w0 = np.asarray(init_weights, dtype=float)
            w0 = np.maximum(w0, 1e-3)
            w0 /= w0.sum()
            init_logits = np.log(w0)
        else:
            # Structural consensus prior aligned with L3 composition: working_memory
            # already wraps base_policy, so base_policy acts as a clean regularizer
            # while chunk_grammar + working_memory share primary structural drive.
            w0 = np.asarray([0.06, 0.54, 0.34, 0.06], dtype=float)
            init_logits = np.log(w0)
        self._prior_logits = init_logits.copy()
        self.logits = np.tile(init_logits[None, :], (self.n_regimes, 1))
        # Per-regime learned recency-kernel strength in [0.0, 1.3]
        self.recency_strengths = np.full(self.n_regimes, float(init_recency_strength), dtype=float)

        # Intra-episode accumulators (consolidated strictly at episode boundaries)
        self._ep_util_sum = np.zeros((self.n_regimes, self.n_experts), dtype=float)
        self._ep_util_cnt = np.zeros((self.n_regimes, self.n_experts), dtype=float)
        self._ep_rec_adv_sum = np.zeros(self.n_regimes, dtype=float)
        self._ep_rec_adv_cnt = np.zeros(self.n_regimes, dtype=float)
        self._ep_steps: int = 0
        self.episodes_consolidated: int = 0
        self.total_steps: int = 0

    def entropy_regime(self, normalized_entropy: float) -> int:
        if self.n_regimes == 1:
            return 0
        e = float(normalized_entropy)
        if self.n_regimes == 2:
            return 0 if e < 0.58 else 1
        if e < 0.45:
            return 0
        if e < 0.72:
            return 1
        return 2

    def weights_for_regime(self, regime: int) -> np.ndarray:
        r = min(max(int(regime), 0), self.n_regimes - 1)
        z = self.logits[r] - float(np.max(self.logits[r]))
        w = np.exp(np.clip(z, -8.0, 0.0))
        return w / max(float(w.sum()), 1e-12)

    def recency_strength_for_regime(self, regime: int) -> float:
        r = min(max(int(regime), 0), self.n_regimes - 1)
        return float(np.clip(self.recency_strengths[r], 0.0, 1.35))

    def blend_experts(
        self,
        expert_maps: Sequence[Dict[int, float]],
        normalized_entropy: float,
        *,
        expert_confidences: Optional[Sequence[float]] = None,
        self_mode: bool = False,
        self_wm_scale: float = 0.72,
    ) -> Tuple[Dict[int, float], int, np.ndarray]:
        """Convexly combine active expert probability maps for the current entropy regime.

        When ``self_mode=True`` (open-loop autonomous rollout without external input),
        damps self-referential ``working_memory`` relative to structural progression
        experts (``base_policy``, ``chunk_grammar``, ``frame_register``) so the
        trajectory advances rather than looping on prompt/self echoes.
        """
        r = self.entropy_regime(normalized_entropy)
        w = self.weights_for_regime(r).copy()
        if self_mode and self.n_experts >= 3:
            # Index 1 is working_memory; damp its share so base_policy + chunk_grammar
            # + frame_register drive forward structural progression in open-loop rollouts.
            wm_s = float(np.clip(self_wm_scale, 0.10, 1.0))
            w[1] *= wm_s
            w_tot = float(w.sum())
            if w_tot > 1e-12:
                w /= w_tot
        # Re-normalize weights across experts that produced non-empty distributions
        active_w = np.zeros(self.n_experts, dtype=float)
        norm_maps: List[Dict[int, float]] = []
        for idx, emap in enumerate(expert_maps[: self.n_experts]):
            tot = float(sum(v for v in emap.values() if v > 0.0)) if emap else 0.0
            if tot > 1e-12:
                conf_mul = (
                    float(np.clip(expert_confidences[idx], 0.05, 1.0))
                    if expert_confidences is not None and idx < len(expert_confidences)
                    else 1.0
                )
                active_w[idx] = w[idx] * conf_mul
                inv_t = 1.0 / tot
                norm_maps.append({int(a): float(v) * inv_t for a, v in emap.items() if v > 0.0})
            else:
                norm_maps.append({})

        w_sum = float(active_w.sum())
        if w_sum <= 1e-12:
            return {}, r, w
        active_w /= w_sum

        blended: Dict[int, float] = {}
        for idx, nmap in enumerate(norm_maps):
            wi = float(active_w[idx])
            if wi <= 1e-9 or not nmap:
                continue
            for a, pr in nmap.items():
                blended[a] = blended.get(a, 0.0) + wi * pr
        return blended, r, active_w

    @staticmethod
    def _expert_score(emap: Dict[int, float], target: int, k: int = 8) -> float:
        if not emap or target <= 0:
            return 0.0
        tot = float(sum(v for v in emap.values() if v > 0.0))
        if tot <= 1e-12:
            return 0.0
        ranked = sorted(emap.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[:k]
        h1 = 1.0 if (ranked and ranked[0][0] == target) else 0.0
        h8 = 1.0 if any(a == target for a, _ in ranked) else 0.0
        p_t = float(emap.get(target, 0.0)) / tot
        # Smooth utility combining Top-1, Top-8, and calibrated log-likelihood
        ll_norm = (float(np.log(max(p_t, 1e-4))) - float(np.log(1e-4))) / (-float(np.log(1e-4)))
        return 0.55 * h1 + 0.25 * h8 + 0.20 * ll_norm

    def record_step(
        self,
        expert_maps: Sequence[Dict[int, float]],
        normalized_entropy: float,
        target: int,
        *,
        recency_advantage: float = 0.0,
        expert_confidences: Optional[Sequence[float]] = None,
        k: int = 8,
    ) -> bool:
        """Record one feedback step via marginal mixture-responsibility + LOO credit.

        Instead of grading experts standalone (which biases toward broad experts that
        wrap ``base_policy`` and penalizes sparse high-precision structural experts
        like ``chunk_grammar`` and ``frame_register``), each active expert receives:
          1. Exact Mixture-of-Experts log-likelihood responsibility gradient
             ``log(P_i(y) / P_mix(y))`` (Online Exponentiated Gradient);
          2. Counterfactual Leave-One-Out Top-K advantage ``S(P_mix) - S(P_{-i})``.
        """
        tgt = int(target)
        if tgt <= 0:
            return False
        r = self.entropy_regime(normalized_entropy)
        blended, _, active_w = self.blend_experts(
            expert_maps, normalized_entropy, expert_confidences=expert_confidences,
        )
        if not blended:
            return False

        # Normalize active expert maps once for exact mixture responsibility & LOO
        norm_maps: List[Dict[int, float]] = []
        active_indices: List[int] = []
        for idx, emap in enumerate(expert_maps[: self.n_experts]):
            tot = float(sum(v for v in emap.values() if v > 0.0)) if emap else 0.0
            if tot > 1e-12 and active_w[idx] > 1e-9:
                inv_t = 1.0 / tot
                norm_maps.append({int(a): float(v) * inv_t for a, v in emap.items() if v > 0.0})
                active_indices.append(idx)
            else:
                norm_maps.append({})

        p_mix_y = max(float(blended.get(tgt, 0.0)), 1e-5)
        s_mix = self._expert_score(blended, tgt, k=k)
        n_act = len(active_indices)

        for idx in active_indices:
            nmap = norm_maps[idx]
            wi = float(active_w[idx])
            p_i_y = max(float(nmap.get(tgt, 0.0)), 1e-5)
            # 1. Exact softmax-mixture log-likelihood logit gradient:
            #    d/dz_i log P_mix(y) = a_i * (P_i(y) / P_mix(y) - 1)
            #    Naturally scales by active responsibility a_i so sparse low-confidence
            #    specialists are never unfairly crushed when inactive.
            ll_grad = float(np.clip(wi * ((p_i_y / p_mix_y) - 1.0), -1.0, 1.5))

            # 2. Counterfactual Leave-One-Out Top-K ensemble advantage
            loo_adv = 0.0
            if n_act >= 2:
                rem_w = max(1.0 - wi, 1e-9)
                inv_rem = 1.0 / rem_w
                loo_map: Dict[int, float] = {}
                for a, pv in blended.items():
                    rem_p = pv - wi * nmap.get(a, 0.0)
                    if rem_p > 1e-12:
                        loo_map[a] = rem_p * inv_rem
                s_loo = self._expert_score(loo_map, tgt, k=k)
                loo_adv = float(np.clip(s_mix - s_loo, -1.0, 1.0))

            # Combined exact marginal contribution (log-likelihood gradient + LOO Top-K gain)
            credit = 0.55 * ll_grad + 0.45 * loo_adv
            self._ep_util_sum[r, idx] += credit
            self._ep_util_cnt[r, idx] += 1.0

        if recency_advantage != 0.0:
            self._ep_rec_adv_sum[r] += float(recency_advantage)
            self._ep_rec_adv_cnt[r] += 1.0

        self._ep_steps += 1
        self.total_steps += 1
        if self._ep_steps >= self.episode_window:
            self.consolidate_episode()
            return True
        return False

    def consolidate_episode(self) -> None:
        """Apply one bounded cross-episode Mirror-Descent update across regimes."""
        if self._ep_steps <= 0:
            return
        self.episodes_consolidated += 1
        step_lr = self.lr / (1.0 + 0.08 * float(self.episodes_consolidated))
        rec_step = self.recency_lr / (1.0 + 0.08 * float(self.episodes_consolidated))

        for r in range(self.n_regimes):
            cnts = self._ep_util_cnt[r]
            active = cnts >= 3.0
            if np.any(active):
                means = np.zeros(self.n_experts, dtype=float)
                means[active] = self._ep_util_sum[r, active] / cnts[active]
                avg_m = float(means[active].mean())
                adv = np.zeros(self.n_experts, dtype=float)
                adv[active] = np.clip(means[active] - avg_m, -0.5, 0.5)
                # Bounded Hedge update + gentle cross-episode shrinkage toward consensus prior
                self.logits[r] = (
                    (1.0 - self.shrinkage) * self.logits[r]
                    + self.shrinkage * self._prior_logits
                    + step_lr * adv
                )
                self.logits[r] = np.clip(self.logits[r], -3.0, 3.0)

            if self._ep_rec_adv_cnt[r] >= 3.0:
                mean_ra = float(self._ep_rec_adv_sum[r] / self._ep_rec_adv_cnt[r])
                self.recency_strengths[r] = float(
                    np.clip(self.recency_strengths[r] + rec_step * np.clip(mean_ra, -0.5, 0.5), 0.0, 1.35)
                )

        self._ep_util_sum.fill(0.0)
        self._ep_util_cnt.fill(0.0)
        self._ep_rec_adv_sum.fill(0.0)
        self._ep_rec_adv_cnt.fill(0.0)
        self._ep_steps = 0

    def snapshot(self) -> dict:
        regime_w = {}
        for r in range(self.n_regimes):
            w = self.weights_for_regime(r)
            regime_w[f"R{r}"] = {
                name: round(float(val), 3)
                for name, val in zip(self.expert_names, w)
            }
            regime_w[f"R{r}"]["recency"] = round(self.recency_strength_for_regime(r), 3)
        return dict(
            episodes_consolidated=int(self.episodes_consolidated),
            total_steps=int(self.total_steps),
            regimes=regime_w,
        )
