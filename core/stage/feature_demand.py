"""Task-agnostic controllable feature demand discovery and online EMA envelope updater."""
from __future__ import annotations

import math
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from summarizer import detect_pure_drift_dims


class ControllableFeatureDemand:
    """Strictly task-agnostic discovery of controllable, value-aligned feature demands.

    Intended exclusively as a heuristic pruning/ordering signal (separate from
    the calibrated final objective):
      1. Strips pure monotonic drift channels via ``detect_pure_drift_dims``.
      2. Measures multi-step action sensitivity through the learned world model
         and gates by both relative effect and signal-to-noise ratio (vs 1-step
         natural transition delta std) so uncontrollable phase/clock channels
         receive zero weight.
      3. Estimates an occupancy-invariant safe-envelope midpoint
         ``0.5 * (q_lo + q_hi)`` on top-quartile surviving training states so
         behavior-policy occupancy skew inside the viable band does not bias
         the target center.
      4. Weights controllable channels by survival-alignment correlation and
         imminent-failure contrast.
    """

    def __init__(
        self,
        min_rel_effect: float = 0.02,
        min_snr: float = 0.20,
        envelope_low_q: float = 0.05,
        envelope_high_q: float = 0.95,
        survival_top_q: float = 0.75,
        imminent_steps: float = 5.0,
        sample_size: int = 128,
        seed: int = 0,
    ):
        self.min_rel_effect = float(min_rel_effect)
        self.min_snr = float(min_snr)
        self.envelope_low_q = float(envelope_low_q)
        self.envelope_high_q = float(envelope_high_q)
        self.survival_top_q = float(survival_top_q)
        self.imminent_steps = float(imminent_steps)
        self.sample_size = int(sample_size)
        self.seed = int(seed)
        self.fitted = False

    def fit_rows(
        self,
        world,
        agent,
        rows: Sequence[Mapping],
        lag: int = 0,
    ) -> "ControllableFeatureDemand":
        """Fit controllable feature weights & envelope centers from training rows."""
        if not rows:
            raise ValueError("rows must not be empty")
        from substrate_generic import history_features

        obs_all = np.asarray([row["obs"] for row in rows], dtype=float)
        nxt_all = np.asarray([row["next_obs"] for row in rows], dtype=float)
        n, d = obs_all.shape
        obs_std = np.std(obs_all, axis=0)
        obs_std = np.where(obs_std > 1e-8, obs_std, 1.0)
        delta_std = np.std(nxt_all - obs_all, axis=0)
        delta_std = np.where(delta_std > 1e-8, delta_std, 1.0)

        by_ep: Dict[Hashable, List[Mapping]] = {}
        for row in rows:
            by_ep.setdefault(row.get("episode", 0), []).append(row)
        ep_seqs = [
            np.asarray([item["obs"] for item in seq], dtype=float)
            for seq in by_ep.values()
            if len(seq) >= 3
        ]
        drift_dims = set(detect_pure_drift_dims(ep_seqs)) if ep_seqs else set()

        # Multi-step action controllability via the learned world model
        rng = np.random.RandomState(self.seed)
        sample_n = min(self.sample_size, n)
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        s0 = obs_all[sample_idx]
        action_space = tuple(getattr(agent, "action_space", (0, 1)))
        roll_h = max(1, int(lag) + 2)
        action_dim = int(lag) + 1
        prefix = np.zeros(int(lag), dtype=float)

        endpoints = []
        for a_val in action_space:
            st = s0.copy()
            seq = np.full((sample_n, roll_h), float(a_val), dtype=float)
            for step in range(roll_h):
                hist = history_features(
                    seq, step + 1, prefix, action_dim=action_dim,
                )
                st = np.asarray(world.predict(st, hist), dtype=float)
            endpoints.append(st[:, :d])
        ep_stack = np.stack(endpoints, axis=0)
        act_std = np.mean(np.std(ep_stack, axis=0), axis=0)
        rel_effect = act_std / obs_std
        snr = act_std / delta_std
        controllable = np.where(
            (rel_effect >= self.min_rel_effect) & (snr >= self.min_snr),
            rel_effect,
            0.0,
        )
        for dd in drift_dims:
            if 0 <= dd < d:
                controllable[dd] = 0.0

        # Survival-envelope target & alignment gain strictly on terminal episodes
        s_term_list = []
        rem_list = []
        for seq in by_ep.values():
            if not seq or not bool(seq[-1].get("done", False)):
                continue
            arr = np.asarray([item["obs"] for item in seq], dtype=float)
            rem = np.arange(len(arr), 0, -1, dtype=float)
            s_term_list.append(arr)
            rem_list.append(rem)

        if s_term_list:
            s_term = np.vstack(s_term_list)
            rem_all = np.concatenate(rem_list)
            y_val = np.log1p(rem_all)
            q_cut = float(np.quantile(y_val, self.survival_top_q))
            safe_states = s_term[y_val >= q_cut]
            if len(safe_states) < 2:
                safe_states = s_term
            q_lo = np.quantile(safe_states, self.envelope_low_q, axis=0)
            q_hi = np.quantile(safe_states, self.envelope_high_q, axis=0)
            env_center = 0.5 * (q_lo + q_hi)
            env_halfwidth = np.maximum(0.5 * (q_hi - q_lo), 1e-4)

            dead_states = s_term[rem_all <= self.imminent_steps]
            align_gain = np.zeros(d, dtype=float)
            fail_contrast = np.zeros(d, dtype=float)
            for j in range(d):
                if controllable[j] <= 0.0:
                    continue
                closeness = -np.abs(s_term[:, j] - env_center[j])
                if np.std(closeness) > 1e-9 and np.std(y_val) > 1e-9:
                    r = float(np.corrcoef(closeness, y_val)[0, 1])
                    if math.isfinite(r):
                        align_gain[j] = max(0.0, r)
                if len(dead_states) > 0:
                    d_dead = float(np.mean(np.abs(dead_states[:, j] - env_center[j])))
                    d_safe = float(np.mean(np.abs(safe_states[:, j] - env_center[j])))
                    fail_contrast[j] = max(
                        0.0, (d_dead - d_safe) / float(env_halfwidth[j])
                    )
            raw_w = np.where(
                controllable > 0.0,
                align_gain * (align_gain + fail_contrast),
                0.0,
            )
        else:
            env_center = np.median(obs_all, axis=0)
            env_halfwidth = np.maximum(obs_std, 1e-4)
            align_gain = np.zeros(d, dtype=float)
            fail_contrast = np.zeros(d, dtype=float)
            raw_w = np.zeros(d, dtype=float)

        total_w = float(raw_w.sum())
        demand_weights = (
            raw_w / total_w if total_w > 1e-9 else np.zeros(d, dtype=float)
        )

        self.obs_dim_ = int(d)
        self.drift_dims_ = tuple(sorted(int(x) for x in drift_dims))
        self.controllable_ = controllable
        self.env_center_ = np.asarray(env_center, dtype=float)
        self.env_halfwidth_ = np.asarray(env_halfwidth, dtype=float)
        self.align_gain_ = align_gain
        self.fail_contrast_ = fail_contrast
        self.demand_weights_ = np.asarray(demand_weights, dtype=float)
        self._inv_scaled_w = self.demand_weights_ / self.env_halfwidth_
        self.fitted = True
        return self

    def distance_batch(self, states: np.ndarray) -> np.ndarray:
        """Weighted normalized L1 distance to learned controllable envelope center."""
        if not self.fitted:
            raise RuntimeError("fit ControllableFeatureDemand first")
        arr = np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if len(arr) == 0:
            return np.zeros(0, dtype=float)
        view = arr[:, :self.obs_dim_]
        return np.abs(view - self.env_center_[None, :]) @ self._inv_scaled_w

    def snapshot(self) -> dict:
        if not self.fitted:
            return dict(fitted=False)
        return dict(
            fitted=True,
            obs_dim=self.obs_dim_,
            drift_dims=self.drift_dims_,
            controllable_dims=tuple(
                int(j) for j in range(self.obs_dim_) if self.controllable_[j] > 0.0
            ),
            demand_weights=tuple(round(float(w), 4) for w in self.demand_weights_),
            env_center=tuple(round(float(c), 4) for c in self.env_center_),
            align_gain=tuple(round(float(r), 4) for r in self.align_gain_),
        )


class OnlineFeatureDemandUpdater:
    """Strictly task-agnostic online EMA updater for :class:`ControllableFeatureDemand`.

    Keeps the learned structural controllability mask fixed while smoothly
    updating the safe-envelope midpoint, halfwidth, and alignment weights from
    a bounded sliding window of realized online episodes.  Non-terminal capped
    episodes contribute to safe survival states without polluting terminal
    failure states.
    """

    def __init__(
        self,
        base_demand: ControllableFeatureDemand,
        update_every: int = 4,
        ema_alpha: float = 0.30,
        max_online_frames: int = 12000,
        min_online_frames: int = 600,
        min_terminal_episodes: int = 0,
        var_sigma: float = 0.75,
        min_var_similarity: float = 0.50,
    ):
        if not base_demand.fitted:
            raise ValueError("base_demand must be fitted")
        if int(update_every) < 1:
            raise ValueError("update_every must be positive")
        if not 0.0 < float(ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if int(max_online_frames) < 1:
            raise ValueError("max_online_frames must be positive")
        if int(min_online_frames) < 1:
            raise ValueError("min_online_frames must be positive")
        if int(min_terminal_episodes) < 0:
            raise ValueError("min_terminal_episodes must be non-negative")
        if float(var_sigma) <= 0.0:
            raise ValueError("var_sigma must be positive")

        self.update_every = int(update_every)
        self.ema_alpha = float(ema_alpha)
        self.max_online_frames = int(max_online_frames)
        self.min_online_frames = int(min_online_frames)
        self.min_terminal_episodes = int(min_terminal_episodes)
        self.var_sigma = float(var_sigma)
        self.min_var_similarity = float(min_var_similarity)

        self.obs_dim_ = int(base_demand.obs_dim_)
        self.controllable_ = np.asarray(base_demand.controllable_, dtype=float).copy()
        self._base_center = np.asarray(base_demand.env_center_, dtype=float).copy()
        self._base_halfwidth = np.asarray(base_demand.env_halfwidth_, dtype=float).copy()
        self._base_weights = np.asarray(base_demand.demand_weights_, dtype=float).copy()
        self.env_center_ = self._base_center.copy()
        self.env_halfwidth_ = self._base_halfwidth.copy()
        self.demand_weights_ = self._base_weights.copy()
        self._inv_scaled_w = self.demand_weights_ / self.env_halfwidth_
        self.envelope_low_q = float(base_demand.envelope_low_q)
        self.envelope_high_q = float(base_demand.envelope_high_q)
        self.survival_top_q = float(base_demand.survival_top_q)
        self.imminent_steps = float(base_demand.imminent_steps)

        self._online_chunks: List[
            Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Optional[np.ndarray]]
        ] = []
        self._online_frames: int = 0
        self.episodes_seen: int = 0
        self.updates: int = 0
        self.active_var_signature: Optional[np.ndarray] = None
        self.world_switches: int = 0

    def distance_batch(self, states: np.ndarray) -> np.ndarray:
        """Weighted normalized L1 distance to the latest online safe-envelope center."""
        arr = np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if len(arr) == 0:
            return np.zeros(0, dtype=float)
        view = arr[:, :self.obs_dim_]
        return np.abs(view - self.env_center_[None, :]) @ self._inv_scaled_w

    def record_episode_rows(
        self,
        ep_rows: Sequence[Mapping],
        var_signature: Optional[np.ndarray] = None,
    ) -> bool:
        """Ingest one realized episode and EMA-update the safe envelope on cadence.

        When ``var_signature`` is provided, scopes updates strictly to episodes
        whose inferred world-variable signature matches within
        ``min_var_similarity``; on a world shift, resets to the shared class
        prior and immediately recalls any matching prior episodes for that world.
        """
        if not ep_rows:
            return False
        arr = np.asarray(
            [row["obs"] for row in ep_rows], dtype=float
        )[:, :self.obs_dim_]
        n = len(arr)
        is_term = bool(ep_rows[-1].get("done", False))
        # Non-terminal (capped) episodes are verified survivors: offset remaining
        # horizon by +n so their states rank in the top quartile without ever
        # being labeled as imminent failures.
        rem = np.arange(n, 0, -1, dtype=float) + (0.0 if is_term else float(n))
        dead = (rem <= self.imminent_steps) if is_term else np.zeros(n, dtype=bool)
        sig_copy = (
            None if var_signature is None
            else np.asarray(var_signature, dtype=float).reshape(-1).copy()
        )

        self._online_chunks.append((arr, rem, dead, is_term, sig_copy))
        self._online_frames += n
        while self._online_frames > self.max_online_frames and len(self._online_chunks) > 1:
            dropped = self._online_chunks.pop(0)
            self._online_frames -= len(dropped[0])
        self.episodes_seen += 1

        shifted = False
        if sig_copy is not None:
            if self.active_var_signature is not None:
                from node_variables import variable_similarity
                shifted = (
                    variable_similarity(
                        sig_copy, self.active_var_signature, sigma=self.var_sigma,
                    )
                    < self.min_var_similarity
                )
            self.active_var_signature = sig_copy

        if shifted:
            # Reset to shared class-node prior before recalling matching world episodes.
            self.env_center_ = self._base_center.copy()
            self.env_halfwidth_ = self._base_halfwidth.copy()
            self.demand_weights_ = self._base_weights.copy()
            self._inv_scaled_w = self.demand_weights_ / self.env_halfwidth_
            self.world_switches += 1

        if not shifted and (self.episodes_seen % self.update_every != 0):
            return False

        if sig_copy is None:
            matching = self._online_chunks
        else:
            from node_variables import variable_similarity
            matching = [
                chunk for chunk in self._online_chunks
                if variable_similarity(sig_copy, chunk[4], sigma=self.var_sigma)
                >= self.min_var_similarity
            ]

        match_frames = sum(len(chunk[0]) for chunk in matching)
        if match_frames < self.min_online_frames:
            return False
        if self.min_terminal_episodes > 0:
            n_term_in_window = sum(1 for chunk in matching if chunk[3])
            if n_term_in_window < self.min_terminal_episodes:
                return False

        s_all = np.vstack([chunk[0] for chunk in matching])
        rem_all = np.concatenate([chunk[1] for chunk in matching])
        dead_mask = np.concatenate([chunk[2] for chunk in matching])

        y_val = np.log1p(rem_all)
        q_cut = float(np.quantile(y_val, self.survival_top_q))
        safe_states = s_all[y_val >= q_cut]
        if len(safe_states) < 16:
            return False
        q_lo = np.quantile(safe_states, self.envelope_low_q, axis=0)
        q_hi = np.quantile(safe_states, self.envelope_high_q, axis=0)
        new_center = 0.5 * (q_lo + q_hi)
        new_hw = np.maximum(0.5 * (q_hi - q_lo), 1e-4)

        # Re-estimate alignment & failure contrast while keeping the structural
        # controllability mask fixed from the initial world-model probe.
        dead_states = s_all[dead_mask]
        d = self.obs_dim_
        align_gain = np.zeros(d, dtype=float)
        fail_contrast = np.zeros(d, dtype=float)
        for j in range(d):
            if self.controllable_[j] <= 0.0:
                continue
            closeness = -np.abs(s_all[:, j] - new_center[j])
            if np.std(closeness) > 1e-9 and np.std(y_val) > 1e-9:
                r = float(np.corrcoef(closeness, y_val)[0, 1])
                if math.isfinite(r):
                    align_gain[j] = max(0.0, r)
            if len(dead_states) > 0:
                d_dead = float(np.mean(np.abs(dead_states[:, j] - new_center[j])))
                d_safe = float(np.mean(np.abs(safe_states[:, j] - new_center[j])))
                fail_contrast[j] = max(0.0, (d_dead - d_safe) / float(new_hw[j]))
        raw_w = np.where(
            self.controllable_ > 0.0,
            align_gain * (align_gain + fail_contrast),
            0.0,
        )
        tot_w = float(raw_w.sum())
        new_w = raw_w / tot_w if tot_w > 1e-9 else self.demand_weights_.copy()

        alpha = self.ema_alpha
        self.env_center_ = (1.0 - alpha) * self.env_center_ + alpha * new_center
        self.env_halfwidth_ = np.maximum(
            (1.0 - alpha) * self.env_halfwidth_ + alpha * new_hw, 1e-4,
        )
        w_mix = (1.0 - alpha) * self.demand_weights_ + alpha * new_w
        w_sum = float(w_mix.sum())
        if w_sum > 1e-9:
            self.demand_weights_ = w_mix / w_sum
        self._inv_scaled_w = self.demand_weights_ / self.env_halfwidth_
        self.updates += 1
        return True

    def snapshot(self) -> dict:
        return dict(
            update_every=int(self.update_every),
            ema_alpha=float(self.ema_alpha),
            max_online_frames=int(self.max_online_frames),
            min_online_frames=int(self.min_online_frames),
            episodes_seen=int(self.episodes_seen),
            online_frames=int(self._online_frames),
            updates=int(self.updates),
            world_switches=int(self.world_switches),
            env_center=tuple(round(float(c), 4) for c in self.env_center_),
            demand_weights=tuple(round(float(w), 4) for w in self.demand_weights_),
        )

