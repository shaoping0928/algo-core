"""Task-agnostic latent-context linear dynamics.

The model assumes the world can be in one of several dynamical regimes that
share structure but differ in some latent parameters.  Nothing here names a
physical parameter; the only inputs are the same raw transitions every other
learner consumes:

    observation, action history, next observation

Training fits one shared linear map and then compresses the per-world
residual corrections into a small orthonormal basis.  At runtime the context
coefficients are inferred from the most recent real transitions by ridge
least squares on those corrections.  A residual monitor detects a regime
change and drops the stale part of the buffer, so the fast channel can
re-identify the context in a few steps while the shared map stays put.

This is the dynamics-side analogue of the existing prototype birth/pruning
machinery: structure is shared, the per-context state is temporary, and the
change detector decides when the temporary state is no longer valid.
"""
from collections import deque

import numpy as np


class ContextualLinearModel:
    """Shared linear dynamics plus an online low-rank context correction."""

    def __init__(self, obs_dim, action_dim, ridge=1e-8,
                 context_dim=4, context_ridge=1e-2, window=64,
                 change_detection=True, change_patience=3,
                 change_factor=3.0, warmup=4, segment_size=128,
                 discovery="cluster", merge_threshold=0.5):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.ridge = float(ridge)
        self.context_dim = max(int(context_dim), 0)
        self.context_ridge = float(context_ridge)
        self.window = max(int(window), 1)
        self.change_detection = bool(change_detection)
        self.change_patience = max(int(change_patience), 1)
        self.change_factor = float(change_factor)
        self.warmup = max(int(warmup), 0)
        self.segment_size = max(int(segment_size), 1)
        if discovery not in ("segment", "cluster"):
            raise ValueError("discovery must be 'segment' or 'cluster'")
        self.discovery = discovery
        self.merge_threshold = float(merge_threshold)
        self.discovered_groups = 0
        if self.change_factor <= 0.0:
            raise ValueError("change_factor must be positive")
        self.weights = None
        self.basis = None                 # (k, obs_dim, feature_dim)
        self.context = np.zeros(self.context_dim)
        self.change_threshold = None
        self.changes = 0
        self.updates = 0
        self._buffer_x = deque(maxlen=self.window)
        self._buffer_r = deque(maxlen=self.window)
        self._error_ewma = None
        self._change_count = 0
        self._prior = None
        self._prior_precision = 0.0
        self._episode_sum = np.zeros(0)
        self._episode_steps = 0
        self._last_episode_context = None

    # ---- training ---------------------------------------------------------
    def _features(self, observations, histories):
        observations = np.asarray(observations, dtype=float)
        histories = np.asarray(histories, dtype=float)
        if observations.shape[-1] != self.obs_dim:
            raise ValueError("observation dimension mismatch")
        if histories.shape[-1] != self.action_dim:
            raise ValueError("action-history dimension mismatch")
        return np.concatenate(
            [observations, histories,
             np.ones(observations.shape[:-1] + (1,))],
            axis=-1,
        )

    def fit(self, observations, histories, next_observations,
            segments=None, segment_size=None):
        """Fit the shared map and discover the context basis.

        `segments` are observable segment ids (for example episode
        boundaries).  When omitted, the model uses fixed contiguous windows
        of `segment_size` transitions.  Neither is a hidden label: the
        model never learns which parameters a segment has, only which
        transitions the stream says belong together.  The basis comes from
        the spectrum of the per-segment residual corrections, and the
        number of kept directions is selected from that spectrum too.
        """
        x = self._features(observations, histories)
        y = np.asarray(next_observations, dtype=float)
        if len(x) < 2:
            raise ValueError("need at least two transitions")
        gram = x.T @ x + self.ridge * np.eye(x.shape[1])
        self.weights = np.linalg.solve(gram, x.T @ y)
        residuals = y - x @ self.weights
        self.basis = None
        self.context = np.zeros(self.context_dim)
        noise_norms = np.linalg.norm(residuals, axis=1)
        if segments is None:
            window = (
                self.segment_size if segment_size is None
                else max(int(segment_size), 1)
            )
            groups = np.arange(len(x)) // window
        else:
            groups = np.asarray(segments)
        if len(np.unique(groups)) >= 2 and self.context_dim > 0:
            segment_x = []
            segment_r = []
            corrections = []
            within = []
            for group in np.unique(groups):
                mask = groups == group
                xg = x[mask]
                rg = residuals[mask]
                gram_g = xg.T @ xg + self.context_ridge * np.eye(
                    x.shape[1]
                )
                correction = np.linalg.solve(gram_g, xg.T @ rg)
                corrections.append(correction.T)          # (obs, feature)
                segment_x.append(xg)
                segment_r.append(rg)
                after = rg - xg @ correction
                within.extend(np.linalg.norm(after, axis=1))
            stacked = np.stack(corrections)               # (groups, obs, feat)
            if self.discovery == "cluster" and len(stacked) >= 2:
                # Discover regimes from the corrections themselves: merge
                # the segments whose residual corrections are closer than
                # a fraction of the median pairwise distance.  No world
                # label and no contiguity assumption enter this step.
                flat_all = stacked.reshape(len(stacked), -1)
                centroids, assignments = _cluster_corrections(
                    flat_all, self.merge_threshold
                )
                self.discovered_groups = len(centroids)
                stacked = centroids.reshape(
                    len(centroids), self.obs_dim, -1
                )
                within = []
                for index, (xg, rg) in enumerate(
                    zip(segment_x, segment_r)
                ):
                    correction = centroids[assignments[index]].reshape(
                        self.obs_dim, -1
                    )
                    after = rg - xg @ correction.T
                    within.extend(np.linalg.norm(after, axis=1))
            else:
                self.discovered_groups = len(stacked)
            if len(stacked) >= 2:
                stacked = stacked - stacked.mean(axis=0, keepdims=True)
                flat = stacked.reshape(len(stacked), -1)
                _u, singular, vt = np.linalg.svd(
                    flat, full_matrices=False
                )
                if len(singular) and singular[0] > 1e-12:
                    # Keep directions that carry a non-negligible share of
                    # the between-regime variation; the number is
                    # discovered from the residual spectrum.
                    limit = min(
                        self.context_dim, vt.shape[0], len(singular)
                    )
                    significant = int(np.sum(
                        singular[:limit] >= 0.1 * singular[0]
                    ))
                    keep = max(1, significant) if limit > 0 else 0
                    keep = min(keep, limit)
                    if keep > 0:
                        basis = vt[:keep].reshape(
                            keep, self.obs_dim, x.shape[1]
                        )
                        self.basis = basis
                        self.context = np.zeros(keep)
                        noise_norms = np.asarray(within, dtype=float)
        # The change threshold reflects the within-regime noise, not the
        # spread between regimes: a new regime should look surprising even
        # though the pooled residual is large by construction.
        typical = float(np.quantile(noise_norms, 0.95)) if len(
            noise_norms
        ) else 0.0
        self.change_threshold = max(
            self.change_factor * typical, 1e-9
        )
        self.reset_context()
        return self

    # ---- runtime context --------------------------------------------------
    def reset_context(self):
        self.context = np.zeros(
            self.basis.shape[0] if self.basis is not None else 0
        )
        self._buffer_x.clear()
        self._buffer_r.clear()
        self._error_ewma = None
        self._change_count = 0
        self._prior = None
        self._prior_precision = 0.0
        self._episode_sum = np.zeros_like(self.context)
        self._episode_steps = 0

    def begin_episode(self, prior=None, prior_precision: float = 0.0):
        """Start an episode, optionally from a remembered context prior."""
        self.reset_context()
        if prior is not None and self.basis is not None:
            prior = np.asarray(prior, dtype=float).reshape(-1)
            if prior.shape[0] == self.context.shape[0]:
                self.context = prior.copy()
                self._prior = prior.copy()
                self._prior_precision = max(float(prior_precision), 0.0)
        self._episode_sum = np.zeros_like(self.context)

    def end_episode(self):
        """Return the episode's mean context and its length."""
        if self._episode_steps > 0:
            mean = self._episode_sum / float(self._episode_steps)
        else:
            mean = self.context.copy()
        self._last_episode_context = mean.copy()
        return mean, self._episode_steps

    def _solve_context(self):
        if self.basis is None or not len(self._buffer_x):
            return
        k = self.basis.shape[0]
        gram = np.zeros((k, k), dtype=float)
        rhs = np.zeros(k, dtype=float)
        for x, residual in zip(self._buffer_x, self._buffer_r):
            regressor = np.einsum("f,kdf->kd", x, self.basis)
            gram += regressor @ regressor.T
            rhs += regressor @ residual
        gram += self.context_ridge * np.eye(k)
        if self._prior is not None and self._prior_precision > 0.0:
            # The remembered context acts as a pseudo-observation; real
            # transitions override it as their evidence accumulates.
            gram += self._prior_precision * np.eye(k)
            rhs += self._prior_precision * self._prior
        self.context = np.linalg.solve(gram, rhs)

    def observe(self, observation, history, next_observation):
        """Fold one real transition into the fast context.

        Returns the base-model residual norm, whether a regime change was
        declared, and the current context vector.
        """
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        x = self._features(
            np.asarray(observation, dtype=float).reshape(1, -1),
            np.asarray(history, dtype=float).reshape(1, -1),
        )[0]
        residual = (
            np.asarray(next_observation, dtype=float).reshape(-1)
            - x @ self.weights
        )
        error = float(np.linalg.norm(residual))
        changed = False
        if (
            self.change_detection
            and self.updates >= self.warmup
            and self.change_threshold is not None
        ):
            if self._error_ewma is None:
                self._error_ewma = error
            else:
                self._error_ewma = 0.8 * self._error_ewma + 0.2 * error
            if self._error_ewma > self.change_threshold:
                self._change_count += 1
            else:
                self._change_count = 0
        self._buffer_x.append(x)
        self._buffer_r.append(residual)
        if self._change_count >= self.change_patience:
            # Keep the transitions that already belong to the new regime;
            # the stale ones would pull the context back toward the old
            # dynamics.  Four points are enough to seed a small solve.
            keep = min(4, len(self._buffer_x))
            recent_x = list(self._buffer_x)[-keep:]
            recent_r = list(self._buffer_r)[-keep:]
            self._buffer_x.clear()
            self._buffer_x.extend(recent_x)
            self._buffer_r.clear()
            self._buffer_r.extend(recent_r)
            self.context = np.zeros_like(self.context)
            self._change_count = 0
            self._error_ewma = None
            self.changes += 1
            changed = True
        self._solve_context()
        self.updates += 1
        self._episode_sum = self._episode_sum + self.context
        self._episode_steps += 1
        return dict(
            error=error,
            changed=bool(changed),
            context=self.context.copy(),
        )

    def predict(self, observations, histories):
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        x = self._features(observations, histories)
        base = x @ self.weights
        if self.basis is None or not np.any(self.context):
            return base
        correction = np.einsum(
            "nf,kdf,k->nd", x, self.basis, self.context
        )
        return base + correction

def _cluster_corrections(vectors, fraction: float = 0.5):
    """Agglomerate residual corrections closer than a median-distance cut.

    The number of regimes is whatever survives the cut; the caller never
    supplies a world id or a regime count.  Returns centroids and the
    nearest-centroid assignment of every input segment.
    """
    vectors = np.asarray(vectors, dtype=float)
    count = len(vectors)
    if count < 2:
        return vectors.copy(), np.zeros(count, dtype=int)
    differences = vectors[:, None, :] - vectors[None, :, :]
    distances = np.sqrt((differences ** 2).sum(axis=2))
    upper = np.triu_indices(count, 1)
    median = float(np.median(distances[upper])) if len(upper[0]) else 0.0
    threshold = max(0.0, float(fraction) * median)
    clusters = [[index] for index in range(count)]
    centroids = [vectors[index].copy() for index in range(count)]
    counts = [1] * count
    while len(clusters) > 1:
        centers = np.stack(centroids)
        deltas = centers[:, None, :] - centers[None, :, :]
        pair = np.sqrt((deltas ** 2).sum(axis=2))
        np.fill_diagonal(pair, np.inf)
        flat_index = int(np.argmin(pair))
        first, second = divmod(flat_index, len(centers))
        if pair[first, second] > threshold:
            break
        total = counts[first] + counts[second]
        centroids[first] = (
            centroids[first] * counts[first]
            + centroids[second] * counts[second]
        ) / total
        counts[first] = total
        clusters[first].extend(clusters[second])
        del clusters[second]
        del centroids[second]
        del counts[second]
    centers = np.stack(centroids)
    deltas = vectors[:, None, :] - centers[None, :, :]
    assignments = np.argmin((deltas ** 2).sum(axis=2), axis=1)
    return centers, assignments
