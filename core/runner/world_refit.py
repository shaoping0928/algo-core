"""Residual-gated online world-model refitting (strictly task-agnostic)."""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import numpy as np

from substrate_generic import fit as fit_linear_world


class OnlineWorldRefitGate:
    """Strictly causal, residual-gated online refitter for ``LinearObservationModel``.

    Accumulates realized online transition rows until ``min_frames`` is reached,
    then checks on ``update_every`` episode boundaries whether refitting the
    linear observation model on ``base_rows + online_train`` reduces held-out
    clean (non-jump, non-terminal) validation MSE by at least ``min_rel_gain``.
    Only when the held-out gate passes does it replace the active world model
    on ``base_rows + all_online_rows``, leaving already-accurate world models
    untouched.
    """

    def __init__(
        self,
        base_world,
        base_rows: Sequence[Mapping],
        seed: int = 0,
        min_rel_gain: float = 0.30,
        min_frames: int = 1500,
        update_every: int = 5,
        jump_quantile: float = 0.90,
        jump_factor: float = 4.0,
    ):
        if not base_rows:
            raise ValueError("base_rows must not be empty")
        if not 0.0 <= float(min_rel_gain) < 1.0:
            raise ValueError("min_rel_gain must be in [0, 1)")
        if int(min_frames) < 50:
            raise ValueError("min_frames must be at least 50")
        if int(update_every) < 1:
            raise ValueError("update_every must be positive")

        self.base_world = base_world
        self.base_rows = list(base_rows)
        self.seed = int(seed)
        self.min_rel_gain = float(min_rel_gain)
        self.min_frames = int(min_frames)
        self.update_every = int(update_every)

        obs_tr = np.asarray([r["obs"] for r in self.base_rows], dtype=float)
        nxt_tr = np.asarray([r["next_obs"] for r in self.base_rows], dtype=float)
        self._jump_thresh = np.maximum(
            float(jump_factor)
            * np.quantile(np.abs(nxt_tr - obs_tr), float(jump_quantile), axis=0),
            1e-3,
        )

        self.world = base_world
        self._online_rows: List[Mapping] = []
        self.episodes_seen: int = 0
        self.refitted: bool = False
        self.last_rel_gain: float = 0.0

    def record_episode_rows(self, ep_rows: Sequence[Mapping]) -> bool:
        """Ingest one episode and refit the world model once if held-out error drops."""
        if not ep_rows or self.refitted or self.min_rel_gain <= 0.0:
            return False
        self._online_rows.extend(ep_rows)
        self.episodes_seen += 1
        if (
            len(self._online_rows) < self.min_frames
            or self.episodes_seen % self.update_every != 0
        ):
            return False

        n_split = int(0.70 * len(self._online_rows))
        val_rows = self._online_rows[n_split:]
        tr_rows = self.base_rows + self._online_rows[:n_split]

        obs_v = np.asarray([r["obs"] for r in val_rows], dtype=float)
        nxt_v = np.asarray([r["next_obs"] for r in val_rows], dtype=float)
        hist_v = np.asarray([r["history"] for r in val_rows], dtype=float)
        clean_mask = np.all(
            np.abs(nxt_v - obs_v) <= self._jump_thresh[None, :], axis=1,
        ) & np.asarray([not bool(r.get("done", False)) for r in val_rows], dtype=bool)
        if int(np.sum(clean_mask)) < max(20, min(200, len(val_rows) // 2)):
            return False

        pred_base = self.base_world.predict(obs_v[clean_mask], hist_v[clean_mask])
        mse_base = float(np.mean((pred_base - nxt_v[clean_mask]) ** 2))
        if mse_base <= 1e-10:
            return False

        cand = fit_linear_world(tr_rows, seed=self.seed)[0]
        pred_cand = cand.predict(obs_v[clean_mask], hist_v[clean_mask])
        mse_cand = float(np.mean((pred_cand - nxt_v[clean_mask]) ** 2))
        rel_gain = (mse_base - mse_cand) / mse_base
        self.last_rel_gain = float(rel_gain)
        if rel_gain >= self.min_rel_gain:
            self.world = fit_linear_world(
                self.base_rows + self._online_rows, seed=self.seed,
            )[0]
            self.refitted = True
            return True
        return False

    def combined_rows(self) -> List[Mapping]:
        """Return ``base_rows + online_rows`` accumulated up to refit."""
        return self.base_rows + self._online_rows

    def snapshot(self) -> dict:
        return dict(
            min_rel_gain=float(self.min_rel_gain),
            min_frames=int(self.min_frames),
            update_every=int(self.update_every),
            episodes_seen=int(self.episodes_seen),
            online_frames=len(self._online_rows),
            refitted=bool(self.refitted),
            last_rel_gain=round(float(self.last_rel_gain), 4),
        )
