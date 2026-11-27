"""Trunk-indexed smooth local terminal-failure memory (strictly task-agnostic)."""
from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from summarizer import StageRegimeSummarizer


class TrunkPointFailureMemory:
    """Trunk-indexed smooth local terminal-failure memory (strictly task-agnostic).

    Stores ONLY true world-feedback terminal states (``done=True`` ``next_obs``)
    alongside a bounded background reservoir of surviving states inside each
    frozen coarse trunk leaf of ``StageRegimeSummarizer``.  Quiet trunk leaves
    with zero observed failures stay compressed at zero query cost.  When a
    candidate state falls inside a trunk leaf that has recorded failures (and
    within the summarizer's OOD radius), returns the smooth local Gaussian
    kernel hazard ratio:

        ``k_fail(z) / (k_fail(z) + k_safe(z) + prior)``

    This avoids both precursor-label corridor smear (never marks normal
    pre-terminal flight as hazardous) and hard axis-aligned split cliffs, while
    allowing repeated online blind spots to accumulate calibrated hazard.
    """

    def __init__(
        self,
        summarizer: StageRegimeSummarizer,
        sigma_mult: float = 0.25,
        prior: float = 4.0,
        safe_stride: int = 2,
        max_fail_per_leaf: int = 120,
        max_safe_per_leaf: int = 400,
        var_sigma: float = 0.75,
    ):
        if not summarizer.fitted:
            raise ValueError("summarizer must be fitted")
        if float(sigma_mult) <= 0.0:
            raise ValueError("sigma_mult must be positive")
        if float(prior) < 0.0:
            raise ValueError("prior must be non-negative")
        if int(safe_stride) < 1:
            raise ValueError("safe_stride must be positive")
        if int(max_fail_per_leaf) < 1 or int(max_safe_per_leaf) < 1:
            raise ValueError("per-leaf reservoir limits must be positive")
        if float(var_sigma) <= 0.0:
            raise ValueError("var_sigma must be positive")
        self.summarizer = summarizer
        self.sigma_mult = float(sigma_mult)
        self.prior = float(prior)
        self.safe_stride = int(safe_stride)
        self.max_fail_per_leaf = int(max_fail_per_leaf)
        self.max_safe_per_leaf = int(max_safe_per_leaf)
        self.var_sigma = float(var_sigma)

        self.n_trunk = len(summarizer.centers_)
        self._center_norm = np.einsum(
            "ij,ij->i", summarizer.centers_, summarizer.centers_,
        )[None, :]
        self._center_t_neg2 = np.ascontiguousarray(-2.0 * summarizer.centers_.T)
        self._ood_radius_sq = float(summarizer.ood_radius_) ** 2
        base_scale = float(getattr(summarizer, "kernel_scale_", 1.0))
        self._sigma2 = max(base_scale * (self.sigma_mult ** 2), 1e-4)

        self._fail_lists: List[List[np.ndarray]] = [[] for _ in range(self.n_trunk)]
        self._fail_sigs: List[List[Optional[np.ndarray]]] = [
            [] for _ in range(self.n_trunk)
        ]
        self._safe_lists: List[List[np.ndarray]] = [[] for _ in range(self.n_trunk)]
        self._fail_arrs: List[Optional[np.ndarray]] = [None] * self.n_trunk
        self._safe_arrs: List[Optional[np.ndarray]] = [None] * self.n_trunk
        self.total_failures = 0

    def _route_trunk_batch(
        self, obs_view: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.summarizer._standardize(obs_view)
        z_norm = np.einsum("ij,ij->i", z, z)[:, None]
        d2 = z @ self._center_t_neg2
        d2 += z_norm
        d2 += self._center_norm
        np.maximum(d2, 0.0, out=d2)
        trunk = np.argmin(d2, axis=1)
        min_d2 = d2[np.arange(len(obs_view)), trunk]
        in_rad = np.isfinite(z).all(axis=1) & (min_d2 <= self._ood_radius_sq)
        return z, trunk, in_rad

    def _rebuild_leaf(self, tid: int) -> None:
        fl = self._fail_lists[tid]
        sl = self._safe_lists[tid]
        self._fail_arrs[tid] = np.asarray(fl, dtype=float) if fl else None
        self._safe_arrs[tid] = np.asarray(sl, dtype=float) if sl else None

    def ingest_rows(
        self,
        rows: Sequence[Mapping],
        include_failures: bool = True,
        var_signature: Optional[np.ndarray] = None,
    ) -> int:
        """Ingest realized transition rows into trunk-indexed fail/safe reservoirs.

        Returns the number of terminal failures newly recorded on this call.
        """
        if not rows:
            return 0
        by_ep: Dict[Hashable, List[Mapping]] = {}
        for row in rows:
            by_ep.setdefault(row.get("episode", 0), []).append(row)

        touched = set()
        added_failures = 0
        obs_dim = self.summarizer.obs_dim_
        sig_copy = (
            None if var_signature is None
            else np.asarray(var_signature, dtype=float).reshape(-1).copy()
        )
        for seq in by_ep.values():
            if not seq:
                continue
            length = len(seq)
            terminal = bool(seq[-1].get("done", False))
            obs_list = [
                np.asarray(item["obs"], dtype=float)[:obs_dim] for item in seq
            ]
            if terminal:
                obs_list.append(
                    np.asarray(seq[-1]["next_obs"], dtype=float)[:obs_dim]
                )
            obs_arr = np.asarray(obs_list, dtype=float)
            z_arr, trunk_arr, in_rad = self._route_trunk_batch(obs_arr)

            if terminal and include_failures:
                idx_term = len(obs_list) - 1
                if in_rad[idx_term]:
                    tid = int(trunk_arr[idx_term])
                    self._fail_lists[tid].append(z_arr[idx_term].copy())
                    self._fail_sigs[tid].append(sig_copy)
                    if len(self._fail_lists[tid]) > self.max_fail_per_leaf:
                        self._fail_lists[tid].pop(0)
                        self._fail_sigs[tid].pop(0)
                    touched.add(tid)
                    self.total_failures += 1
                    added_failures += 1

            safe_end = max(0, length - 2) if terminal else length
            for idx in range(0, safe_end, self.safe_stride):
                if not in_rad[idx]:
                    continue
                tid = int(trunk_arr[idx])
                self._safe_lists[tid].append(z_arr[idx].copy())
                if len(self._safe_lists[tid]) > self.max_safe_per_leaf:
                    self._safe_lists[tid].pop(0)
                touched.add(tid)

        for tid in touched:
            self._rebuild_leaf(tid)
        return added_failures

    def hazard_batch(
        self,
        states: np.ndarray,
        var_signature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return localized smooth terminal-hazard ratio in ``[0, 1)``."""
        arr = np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = len(arr)
        if n == 0 or self.total_failures == 0:
            return np.zeros(n, dtype=float)
        z_arr, trunk_arr, in_rad = self._route_trunk_batch(
            arr[:, :self.summarizer.obs_dim_],
        )
        out = np.zeros(n, dtype=float)
        inv_2s2 = -0.5 / self._sigma2
        sim_fn = None
        if var_signature is not None:
            from node_variables import variable_similarity
            sim_fn = variable_similarity
        for tid in np.unique(trunk_arr):
            f_arr = self._fail_arrs[int(tid)]
            if f_arr is None:
                continue
            mask = (trunk_arr == tid) & in_rad
            if not np.any(mask):
                continue
            zq = z_arr[mask]
            df2 = np.sum((zq[:, None, :] - f_arr[None, :, :]) ** 2, axis=2)
            if sim_fn is not None:
                vw = np.asarray(
                    [
                        sim_fn(var_signature, fs, sigma=self.var_sigma)
                        for fs in self._fail_sigs[int(tid)]
                    ],
                    dtype=float,
                )
                kf = (np.exp(inv_2s2 * df2) * vw[None, :]).sum(axis=1)
            else:
                kf = np.exp(inv_2s2 * df2).sum(axis=1)
            s_arr = self._safe_arrs[int(tid)]
            if s_arr is not None:
                ds2 = np.sum((zq[:, None, :] - s_arr[None, :, :]) ** 2, axis=2)
                ks = np.exp(inv_2s2 * ds2).sum(axis=1)
            else:
                ks = np.zeros(len(zq), dtype=float)
            out[mask] = kf / (kf + ks + self.prior)
        return out

    def snapshot(self) -> dict:
        active_leaves = sum(1 for arr in self._fail_arrs if arr is not None)
        return dict(
            sigma_mult=self.sigma_mult,
            prior=self.prior,
            total_failures=int(self.total_failures),
            buffered_failures=int(sum(len(x) for x in self._fail_lists)),
            buffered_safe=int(sum(len(x) for x in self._safe_lists)),
            active_failure_leaves=int(active_leaves),
            compressed_leaves=int(self.n_trunk - active_leaves),
        )

