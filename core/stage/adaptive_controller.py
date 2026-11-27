"""Dual-confidence decaying-threshold controller for adaptive segmented search."""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class AdaptiveSearchController:
    """Strictly task-agnostic dual-confidence controller for adaptive segmented search.

    Combines two causal confidence layers:
      1. **Outer evaluator reliability & mode self-audit**: unlocks clean
         multi-step deepening only after the online stage evaluator clears
         ``meta_lcb`` (causal Wilson-LCB from ``SourceAbstainGate``), and
         continuously audits realized episode lengths against the running
         conservative baseline median via a second ``SourceAbstainGate`` so
         any seed/regime where deeper lookahead underperforms automatically
         reverts to the conservative fallback.
      2. **Inner decaying-threshold plan-state confidence**: releases shallow
         plans immediately when their monotonic path confidence clears
         ``tau_k = tau_0 * (conf_decay ** k)`` (optionally self-calibrated from
         a rolling quantile of Tier-0 confidences), while guaranteeing finite
         termination on the final tier.
    """

    def __init__(
        self,
        conf_threshold: float = 0.72,
        conf_decay: float = 0.80,
        conf_quantile: float = 0.0,
        conf_window: int = 256,
        meta_lcb: float = 0.65,
        stage_warmup: int = 5,
        audit_warmup: int = 8,
        audit_target_lcb: float = 0.25,
        audit_ratio: float = 0.55,
        audit_decay: float = 0.90,
        audit_z: float = 1.0,
    ):
        if not 0.0 <= float(conf_threshold) <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if not 0.0 < float(conf_decay) <= 1.0:
            raise ValueError("conf_decay must be in (0, 1]")
        if not 0.0 <= float(conf_quantile) <= 1.0:
            raise ValueError("conf_quantile must be in [0, 1]")
        if int(conf_window) < 8:
            raise ValueError("conf_window must be at least 8")
        if not 0.0 <= float(meta_lcb) <= 1.0:
            raise ValueError("meta_lcb must be in [0, 1]")
        if int(stage_warmup) < 0:
            raise ValueError("stage_warmup must be non-negative")
        if int(audit_warmup) < 1:
            raise ValueError("audit_warmup must be positive")
        if not 0.0 <= float(audit_target_lcb) <= 1.0:
            raise ValueError("audit_target_lcb must be in [0, 1]")
        if float(audit_ratio) < 0.0:
            raise ValueError("audit_ratio must be non-negative")

        from node_variables import SourceAbstainGate

        self.conf_threshold = float(conf_threshold)
        self.conf_decay = float(conf_decay)
        self.conf_quantile = float(conf_quantile)
        self.conf_window = int(conf_window)
        self.meta_lcb = float(meta_lcb)
        self.stage_warmup = int(stage_warmup)
        self.audit_ratio = float(audit_ratio)
        self.mode_audit_gate = SourceAbstainGate(
            warmup=int(audit_warmup),
            decay=float(audit_decay),
            target_lcb=float(audit_target_lcb),
            floor_weight=0.0,
            z=float(audit_z),
        )
        self._base_episode_lengths: List[float] = []
        self._conf_history: deque = deque(maxlen=self.conf_window)
        self.active: bool = False
        self.base_steps: int = 0
        self.adaptive_steps: int = 0
        self.tier_counts: Dict[int, int] = {}
        self.audit_blocked_episodes: int = 0
        self.episodes_seen: int = 0

    def begin_episode(self, stage_gate=None) -> bool:
        """Decide causally whether the upcoming episode uses adaptive deepening.

        Adaptive deepening is unlocked only when BOTH outer causal gates pass:
          1. ``stage_gate`` confirms the online stage evaluator has predictive
             signal (Wilson-LCB >= ``meta_lcb`` after ``stage_warmup`` episodes).
          2. ``mode_audit_gate`` confirms adaptive episodes are not collapsing
             relative to the running conservative baseline median.
        """
        if self.meta_lcb <= 0.0:
            stage_ok = True
        else:
            stage_ok = (
                stage_gate is not None
                and int(getattr(stage_gate, "raw_trials", 0)) >= self.stage_warmup
                and float(stage_gate.lcb()) >= self.meta_lcb
            )
        audit_ok = True
        if (
            self.mode_audit_gate.target_lcb > 0.0
            and self.mode_audit_gate.raw_trials >= self.mode_audit_gate.warmup
            and self.mode_audit_gate.lcb() < self.mode_audit_gate.target_lcb
        ):
            audit_ok = False
        if stage_ok and not audit_ok:
            self.audit_blocked_episodes += 1
        self.active = bool(stage_ok and audit_ok)
        return self.active

    def record_episode(
        self, frames: int, used_adaptive: Optional[bool] = None,
    ) -> None:
        """Record realized episode survival for the causal mode self-audit.

        Conservative episodes populate the baseline survival reference; adaptive
        episodes are judged against ``audit_ratio * median(base_lengths)`` so
        regressing regimes trigger automatic fallback without future leakage.
        """
        ep_len = float(max(int(frames), 0))
        was_adaptive = self.active if used_adaptive is None else bool(used_adaptive)
        self.episodes_seen += 1
        if not was_adaptive:
            self._base_episode_lengths.append(ep_len)
        elif self._base_episode_lengths:
            base_med = float(np.median(self._base_episode_lengths))
            self.mode_audit_gate.observe(ep_len >= self.audit_ratio * base_med)

    def observe_tier0_confidence(self, conf: float) -> Tuple[float, float]:
        """Record Tier-0 plan confidence and return ``(tau_0, decay)``.

        When ``conf_quantile > 0``, waits for at least 32 Tier-0 samples before
        switching from ``conf_threshold`` to the rolling quantile estimate so
        cold-start variance does not destabilize early thresholds.
        """
        c = float(conf)
        if math.isfinite(c):
            self._conf_history.append(c)
        if self.conf_quantile > 0.0 and len(self._conf_history) >= 32:
            tau0 = float(np.quantile(self._conf_history, self.conf_quantile))
            return tau0, self.conf_decay
        return self.conf_threshold, self.conf_decay

    @staticmethod
    def threshold_for_tier(
        tier_index: int, tau0: float, decay: float, is_final: bool,
    ) -> float:
        """Return decaying release threshold for ``tier_index`` (final always passes)."""
        if is_final:
            return -1.0
        return float(tau0) * (float(decay) ** max(int(tier_index), 0))

    def record_step(
        self, used_adaptive: bool, tier_index: Optional[int] = None,
    ) -> None:
        if used_adaptive:
            self.adaptive_steps += 1
            if tier_index is not None:
                idx = int(tier_index)
                self.tier_counts[idx] = self.tier_counts.get(idx, 0) + 1
        else:
            self.base_steps += 1

    def snapshot(self) -> dict:
        tot_steps = self.base_steps + self.adaptive_steps
        return dict(
            conf_threshold=self.conf_threshold,
            conf_decay=self.conf_decay,
            conf_quantile=self.conf_quantile,
            meta_lcb=self.meta_lcb,
            active=bool(self.active),
            episodes_seen=int(self.episodes_seen),
            base_episodes=len(self._base_episode_lengths),
            audit_blocked_episodes=int(self.audit_blocked_episodes),
            base_steps=int(self.base_steps),
            adaptive_steps=int(self.adaptive_steps),
            adaptive_step_fraction=(
                round(float(self.adaptive_steps) / float(tot_steps), 4)
                if tot_steps > 0 else 0.0
            ),
            tier_counts={int(k): int(v) for k, v in sorted(self.tier_counts.items())},
            mode_audit_gate=self.mode_audit_gate.snapshot(),
        )

