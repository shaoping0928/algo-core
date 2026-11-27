"""Task-agnostic node/state variable abstraction and class-node instantiation.

Provides a first-class, domain-independent mechanism for separating:

  * **Class-Node Slow Templates**: shared invariant structure learned across
    experience (e.g., base transition dynamics or coarse stage-regime trunks),
    registered with ``timescale="slow"`` in :class:`VariableRegistry`.
  * **Instance Fast Variables**: latent context parameters inferred online from
    realized prediction residuals (e.g., constant channel drifts, action-gain
    shifts, or linear slope shifts), registered with ``timescale="fast"`` and
    related via ``"tracks"`` to their owning class template.
  * **Parameterized States & Variable-Bound Instantiation**: binds live states
    and downstream episodic memories to the inferred normalized latent-variable
    signature so different worlds instantiate the same abstract class template
    without overwriting or corrupting each other.

Nothing in this module names or assumes any physical law, coordinate meaning,
or task-specific parameter.  All thresholds (normal step range, residual noise
floor, channel predictability) are derived strictly from training transitions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from variable_registry import VariableRegistry


@dataclass(frozen=True)
class ParameterizedState:
    """An observation paired with its currently inferred latent variable signature."""

    obs: np.ndarray
    variables: np.ndarray
    active_channels: Tuple[int, ...]
    credibility: float


class DynamicsClassNode:
    """Abstract linear dynamics class node with online latent variable slots.

    Wraps a frozen shared dynamics template ``y_base = x @ W_base`` and infers
    per-channel latent variable corrections ``delta_w`` online from realized
    transitions ``(obs, history, next_obs)``.

    Two generic invariance guards keep the class node bit-for-bit faithful to
    the slow template when the world has not changed:
      1. **Coupled-jump transition masking**: if *any* channel exhibits an
         isolated discontinuous reset (``|next_obs[k] - obs[k]| > jump_thresh[k]``),
         the entire transition is excluded from continuous dynamics updates so
         boundary/event resets never corrupt coupled smooth channels.
      2. **Residual-significance dead-zone gate**: a channel's fast variable
         slot activates only when recent clean base residuals significantly
         exceed the training noise floor AND the fitted variable meaningfully
         reduces residual error.  Otherwise the slot stays exact ``0.0``.
    """

    def __init__(
        self,
        base_world,
        train_rows: Sequence[Mapping],
        window: int = 64,
        bias_ridge: float = 0.5,
        action_ridge: float = 2.0,
        state_ridge: float = 12.0,
        deadzone_z: float = 3.5,
        min_gain_ratio: float = 0.30,
        min_clean_samples: int = 4,
        change_patience: int = 3,
        change_factor: float = 4.0,
        registry: Optional[VariableRegistry] = None,
    ):
        if not train_rows:
            raise ValueError("train_rows must not be empty")
        if int(window) < 4:
            raise ValueError("window must be at least 4")
        if float(deadzone_z) <= 0.0:
            raise ValueError("deadzone_z must be positive")
        if not 0.0 <= float(min_gain_ratio) < 1.0:
            raise ValueError("min_gain_ratio must be in [0, 1)")
        if int(min_clean_samples) < 2:
            raise ValueError("min_clean_samples must be at least 2")

        self.base_world = base_world
        self.obs_dim = int(base_world.obs_dim)
        self.action_dim = int(base_world.action_dim)
        self.feat_dim = self.obs_dim + self.action_dim + 1
        self.window = int(window)
        self.deadzone_z = float(deadzone_z)
        self.min_gain_ratio = float(min_gain_ratio)
        self.min_clean_samples = int(min_clean_samples)
        self.change_patience = max(int(change_patience), 1)
        self.change_factor = float(change_factor)

        prior = np.empty(self.feat_dim, dtype=float)
        prior[:self.obs_dim] = float(state_ridge)
        prior[self.obs_dim:self.obs_dim + self.action_dim] = float(action_ridge)
        prior[-1] = float(bias_ridge)
        self._ridge_diag = np.diag(prior)

        # Self-supervised baseline calibration strictly from training rows
        obs = np.asarray(
            [row["obs"] for row in train_rows], dtype=float,
        )[:, :self.obs_dim]
        nxt = np.asarray(
            [row["next_obs"] for row in train_rows], dtype=float,
        )[:, :self.obs_dim]
        hist = np.asarray(
            [row["history"] for row in train_rows], dtype=float,
        )[:, :self.action_dim]

        deltas = np.abs(nxt - obs)
        self._step_p90 = np.quantile(deltas, 0.90, axis=0)
        self._step_med = np.median(deltas, axis=0)
        self._step_scale = np.maximum(self._step_med, 1e-3)
        self._jump_thresh = np.maximum(4.0 * self._step_p90, 0.12)
        clean_train = np.all(deltas <= self._jump_thresh[None, :], axis=1)
        if not np.any(clean_train):
            clean_train = np.ones(len(obs), dtype=bool)

        base_pred = np.asarray(base_world.predict(obs, hist), dtype=float)
        base_res = np.abs(nxt - base_pred)

        self._smooth_dims = np.zeros(self.obs_dim, dtype=bool)
        self._res_scale = np.full(self.obs_dim, 1e-4, dtype=float)
        self._obs_scale = np.maximum(np.std(obs, axis=0), 1e-4)
        for j in range(self.obs_dim):
            rj = base_res[clean_train, j]
            if len(rj) == 0:
                continue
            med = float(np.median(rj))
            mad = float(np.median(np.abs(rj - med)))
            p90 = float(np.quantile(rj, 0.90))
            # Robust residual floor combining median level and MAD spread
            self._res_scale[j] = max(med + 1.4826 * mad, p90 * 0.5, 1e-4)
            if p90 < 0.25 * float(self._obs_scale[j]):
                self._smooth_dims[j] = True

        self.registry = (
            registry if registry is not None else VariableRegistry(base_prior=4.0)
        )
        self._register_class_and_slots()
        self.reset_variables()

    def _register_class_and_slots(self) -> None:
        if "class:dynamics" not in self.registry.variables:
            self.registry.add(
                "class:dynamics",
                "class_template",
                np.asarray(self.base_world.weights, dtype=float).reshape(-1),
                support=float(max(self.obs_dim * 10, 20)),
                variance=0.0,
                timescale="slow",
                metadata=dict(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    smooth_dims=tuple(
                        int(j) for j in range(self.obs_dim) if self._smooth_dims[j]
                    ),
                ),
            )
        for j in range(self.obs_dim):
            vid = f"var:dyn:ch{j}"
            if vid not in self.registry.variables:
                self.registry.add(
                    vid,
                    "world_variable",
                    np.zeros(self.feat_dim, dtype=float),
                    support=0.0,
                    variance=0.0,
                    timescale="fast",
                    metadata=dict(channel=j, active=False),
                )
                self.registry.relate(vid, "class:dynamics", "tracks")

    def reset_variables(self) -> None:
        """Clear fast world-variable slots back to the slow class template."""
        self.delta_w = np.zeros((self.feat_dim, self.obs_dim), dtype=float)
        self.active_mask = np.zeros(self.obs_dim, dtype=bool)
        self._buf_x: deque = deque(maxlen=self.window)
        self._buf_r: deque = deque(maxlen=self.window)
        self._err_ewma: Optional[float] = None
        self._spike_run = 0
        self.changes = 0
        self.updates = 0
        self.skipped_jumps = 0
        for j in range(self.obs_dim):
            var = self.registry.variables[f"var:dyn:ch{j}"]
            var.estimate = np.zeros(self.feat_dim, dtype=float)
            var.support = 0.0
            var.variance = 0.0
            var.metadata["active"] = False

    def _features(
        self, observations: np.ndarray, histories: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [observations, histories, np.ones(observations.shape[:-1] + (1,))],
            axis=-1,
        )

    def predict(self, observations, histories) -> np.ndarray:
        """Predict next observations using slow class template + active variables."""
        obs = np.asarray(observations, dtype=float)
        hist = np.asarray(histories, dtype=float)
        if not np.any(self.active_mask):
            return np.asarray(self.base_world.predict(obs, hist), dtype=float)
        x = self._features(obs, hist)
        return x @ (self.base_world.weights + self.delta_w)

    def observe(self, obs, history, next_obs) -> dict:
        """Ingest one realized transition and update fast latent variable slots.

        Returns a diagnostic dictionary with ``skipped_jump``, ``changed``,
        ``active_channels``, and ``credibility``.
        """
        obs_v = np.asarray(obs, dtype=float).reshape(-1)[:self.obs_dim]
        hist_v = np.asarray(history, dtype=float).reshape(-1)[:self.action_dim]
        nxt_v = np.asarray(next_obs, dtype=float).reshape(-1)[:self.obs_dim]

        step_abs = np.abs(nxt_v - obs_v)
        # Coupled-jump guard: skip the entire frame if ANY channel has a reset spike
        if not np.all(np.isfinite(step_abs)) or np.any(step_abs > self._jump_thresh):
            self.skipped_jumps += 1
            return dict(
                skipped_jump=True,
                changed=False,
                active_channels=self.active_channels(),
                credibility=self.mean_credibility(),
            )

        x = self._features(obs_v.reshape(1, -1), hist_v.reshape(1, -1))[0]
        r_base = nxt_v - x @ self.base_world.weights
        r_cur = nxt_v - x @ (self.base_world.weights + self.delta_w)

        changed = False
        if np.any(self._smooth_dims):
            sm = self._smooth_dims
            norm_err = float(
                np.mean(np.abs(r_cur[sm]) / self._res_scale[sm])
            )
            if self.updates >= 6 and self._err_ewma is not None:
                if norm_err > max(self.change_factor * self._err_ewma, 6.0):
                    self._spike_run += 1
                else:
                    self._spike_run = 0
                    self._err_ewma = 0.85 * self._err_ewma + 0.15 * norm_err
            else:
                self._err_ewma = (
                    norm_err if self._err_ewma is None
                    else 0.7 * self._err_ewma + 0.3 * norm_err
                )
                self._spike_run = 0

        self._buf_x.append(x)
        self._buf_r.append(r_base)

        if self._spike_run >= self.change_patience:
            keep = min(self.change_patience, len(self._buf_x))
            rx = list(self._buf_x)[-keep:]
            rr = list(self._buf_r)[-keep:]
            self._buf_x.clear()
            self._buf_x.extend(rx)
            self._buf_r.clear()
            self._buf_r.extend(rr)
            self._spike_run = 0
            self._err_ewma = None
            self.changes += 1
            changed = True

        n_buf = len(self._buf_x)
        new_dw = np.zeros_like(self.delta_w)
        new_active = np.zeros(self.obs_dim, dtype=bool)
        if n_buf >= self.min_clean_samples:
            X = np.asarray(self._buf_x, dtype=float)
            R = np.asarray(self._buf_r, dtype=float)
            gram = X.T @ X + self._ridge_diag
            cand_dw = np.linalg.solve(gram, X.T @ R)
            pred_r = X @ cand_dw

            for j in range(self.obs_dim):
                if not self._smooth_dims[j]:
                    continue
                base_mae = float(np.mean(np.abs(R[:, j])))
                if base_mae <= self.deadzone_z * float(self._res_scale[j]):
                    continue
                adapt_mae = float(np.mean(np.abs(R[:, j] - pred_r[:, j])))
                if adapt_mae < (1.0 - self.min_gain_ratio) * base_mae:
                    new_dw[:, j] = cand_dw[:, j]
                    new_active[j] = True

        self.delta_w = new_dw
        self.active_mask = new_active
        self.updates += 1

        # Sync fast variable slots in VariableRegistry
        for j in range(self.obs_dim):
            var = self.registry.variables[f"var:dyn:ch{j}"]
            if new_active[j]:
                var.estimate = new_dw[:, j].copy()
                var.support = float(n_buf)
                var.metadata["active"] = True
            else:
                var.estimate = np.zeros(self.feat_dim, dtype=float)
                var.support = 0.0
                var.metadata["active"] = False

        return dict(
            skipped_jump=False,
            changed=bool(changed),
            active_channels=self.active_channels(),
            credibility=self.mean_credibility(),
        )

    def active_channels(self) -> Tuple[int, ...]:
        return tuple(int(j) for j in range(self.obs_dim) if self.active_mask[j])

    def mean_credibility(self) -> float:
        act = self.active_channels()
        if not act:
            return 0.0
        return float(np.mean([
            self.registry.credibility(f"var:dyn:ch{j}") for j in act
        ]))

    def world_signature(self) -> np.ndarray:
        """Compact scale-free latent variable vector describing the active world.

        For each smooth channel ``j``, encodes its credibility-weighted bias
        shift and action-gain shift normalized by training step scale.  When no
        variable slot is active (world matches slow template), returns exact zeros.
        """
        # 2 summary scalars per obs_dim: normalized bias shift & mean action-gain shift
        sig = np.zeros(2 * self.obs_dim, dtype=float)
        for j in range(self.obs_dim):
            if not self.active_mask[j]:
                continue
            cred = self.registry.credibility(f"var:dyn:ch{j}")
            scale = float(self._step_scale[j])
            # Bias shift (intercept column)
            sig[2 * j] = cred * float(self.delta_w[-1, j]) / scale
            # Action-column norm shift
            act_slice = self.delta_w[self.obs_dim:self.obs_dim + self.action_dim, j]
            sig[2 * j + 1] = cred * float(np.sum(act_slice)) / scale
        return sig

    def bind_state(self, obs: np.ndarray) -> ParameterizedState:
        """Wrap raw observation with the currently inferred world variables."""
        return ParameterizedState(
            obs=np.asarray(obs, dtype=float).copy(),
            variables=self.world_signature(),
            active_channels=self.active_channels(),
            credibility=self.mean_credibility(),
        )

    def snapshot(self) -> dict:
        sig = self.world_signature()
        return dict(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            smooth_dims=tuple(
                int(j) for j in range(self.obs_dim) if self._smooth_dims[j]
            ),
            active_channels=self.active_channels(),
            updates=int(self.updates),
            skipped_jumps=int(self.skipped_jumps),
            changes=int(self.changes),
            credibility=round(self.mean_credibility(), 4),
            signature_norm=round(float(np.linalg.norm(sig)), 4),
        )


def variable_similarity(
    sig_a: Optional[np.ndarray],
    sig_b: Optional[np.ndarray],
    sigma: float = 0.75,
) -> float:
    """Gaussian similarity in ``(0, 1]`` between two world-variable signatures.

    Returns ``1.0`` when both signatures are ``None`` or identical (preserving
    exact single-world invariance).
    """
    if sig_a is None or sig_b is None:
        return 1.0
    a = np.asarray(sig_a, dtype=float).reshape(-1)
    b = np.asarray(sig_b, dtype=float).reshape(-1)
    if a.shape != b.shape or len(a) == 0:
        return 1.0
    diff2 = float(np.sum((a - b) ** 2))
    if diff2 <= 1e-12:
        return 1.0
    s2 = max(float(sigma) ** 2, 1e-6)
    return float(np.exp(-0.5 * diff2 / s2))
