"""Task-agnostic latent-context linear dynamics.

The model assumes the world can be in one of several dynamical regimes that
share structure but differ in some latent parameters.  Nothing here names a
parameter (gravity, drag, flap strength, ...); the only inputs are the same
raw transitions every other learner consumes:

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
                 change_factor=3.0, warmup=4):
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
            worlds=None):
        """Fit the shared map and, with multiple worlds, the context basis.

        `worlds` is a per-transition group id used only to estimate the
        per-regime residual corrections.  It is training-time supervision
        of the *grouping*, not of any physical quantity; at runtime the
        context is inferred without it.
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
        if worlds is not None and self.context_dim > 0:
            groups = np.asarray(worlds)
            unique = np.unique(groups)
            if len(unique) >= 2:
                corrections = []
                within = []
                for group in unique:
                    mask = groups == group
                    xg = x[mask]
                    rg = residuals[mask]
                    gram_g = xg.T @ xg + self.context_ridge * np.eye(
                        x.shape[1]
                    )
                    correction = np.linalg.solve(gram_g, xg.T @ rg)
                    corrections.append(correction.T)      # (obs, feature)
                    after = rg - xg @ correction
                    within.extend(np.linalg.norm(after, axis=1))
                stacked = np.stack(corrections)           # (worlds, obs, feat)
                stacked = stacked - stacked.mean(axis=0, keepdims=True)
                flat = stacked.reshape(len(unique), -1)
                _u, singular, vt = np.linalg.svd(
                    flat, full_matrices=False
                )
                if len(singular) and singular[0] > 1e-12:
                    keep = min(self.context_dim, vt.shape[0])
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
