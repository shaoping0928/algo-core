"""Value learning from world-emitted dopamine and pain channels."""
from collections import OrderedDict
from typing import Iterable

import numpy as np


class InputNovelty:
    """Online density estimate for generic substrate input-feature vectors."""

    def __init__(self, feature_dim, precision=6, max_patterns=100000):
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if max_patterns < 1:
            raise ValueError("max_patterns must be positive")
        self.feature_dim = int(feature_dim)
        self.precision = int(precision)
        self.max_patterns = int(max_patterns)
        self.counts = OrderedDict()

    def _key(self, features):
        x = np.asarray(features, dtype=float).reshape(-1)
        if x.shape != (self.feature_dim,):
            raise ValueError("novelty feature dimension mismatch")
        if not np.isfinite(x).all():
            raise ValueError("novelty features must be finite")
        return tuple(np.round(x, self.precision))

    def score(self, features):
        """A count-based information-gain proxy; scoring does not update the density."""
        count = self.counts.get(self._key(features), 0)
        return float(1.0 / np.sqrt(1.0 + count))

    def score_batch(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError("expected a feature matrix with the configured width")
        quantized = np.round(x, self.precision)
        unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
        scores = np.asarray(
            [
                1.0 / np.sqrt(1.0 + self.counts.get(tuple(row), 0))
                for row in unique
            ],
            dtype=float,
        )
        return scores[inverse]

    def observe(self, features):
        key = self._key(features)
        score = float(1.0 / np.sqrt(1.0 + self.counts.get(key, 0)))
        if key not in self.counts and len(self.counts) >= self.max_patterns:
            self.counts.popitem(last=False)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.counts.move_to_end(key)
        return score


class WorldFeedbackValueHead:
    """Learn discounted world feedback values from observed transitions.

    Outputs are ordered (dopamine, pain). The head itself contains no task reward formula;
    it only predicts channels supplied by the world. Planning uses generic valence semantics:
    first avoid pain, then seek dopamine.
    """

    def __init__(self, feature_dim, hidden_dim=32, learning_rate=0.001, discount=0.99,
                 seed=0):
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must be in [0, 1)")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        rng = np.random.RandomState(seed)
        self.w1 = rng.normal(
            0.0, 1.0 / np.sqrt(self.feature_dim),
            size=(self.feature_dim, self.hidden_dim),
        )
        self.b1 = np.zeros(self.hidden_dim, dtype=float)
        self.w2 = np.zeros((self.hidden_dim, 2), dtype=float)
        self.b2 = np.zeros(2, dtype=float)

    def _features(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim < 1 or x.shape[-1] != self.feature_dim:
            raise ValueError(
                "expected %d features, got shape %r" % (self.feature_dim, x.shape)
            )
        return x

    def value(self, features):
        x = self._features(features)
        hidden = np.tanh(x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    @staticmethod
    def preference_scores(channel_values):
        """Return lexicographic costs (pain, -dopamine) for the generic search core."""
        values = np.asarray(channel_values, dtype=float)
        if values.shape[-1] != 2:
            raise ValueError("expected (dopamine, pain) channel values")
        return np.stack([values[..., 1], -values[..., 0]], axis=-1)

    def update(self, features, feedback, next_features, terminal=False):
        x = self._features(features)
        next_x = self._features(next_features)
        if x.ndim != 1 or next_x.ndim != 1:
            raise ValueError("update expects one feature vector per transition")
        signal = np.asarray(feedback, dtype=float).reshape(-1)
        if signal.shape != (2,) or not np.isfinite(signal).all():
            raise ValueError("feedback must contain finite dopamine and pain channels")

        hidden = np.tanh(x @ self.w1 + self.b1)
        current = hidden @ self.w2 + self.b2
        if terminal:
            target = signal
        else:
            target = signal + self.discount * self.value(next_x)
        td_error = target - current

        learning_rate = self.learning_rate / (1.0 + float(x @ x))
        old_w2 = self.w2.copy()
        self.w2 += learning_rate * np.outer(hidden, td_error)
        self.b2 += learning_rate * td_error
        hidden_error = (old_w2 @ td_error) * (1.0 - hidden * hidden)
        self.w1 += learning_rate * np.outer(x, hidden_error)
        self.b1 += learning_rate * hidden_error
        return td_error


def _lag_profile(model):
    gains = np.abs(np.asarray(model.h, dtype=float))
    if not gains.size:
        raise ValueError("the predictor must expose at least one lag coefficient")
    total = float(gains.sum())
    probabilities = (gains / total) if total > 1e-12 else np.full_like(gains, 1.0 / len(gains))
    entropy = 1.0
    if len(probabilities) > 1:
        entropy = float(
            -np.sum(probabilities * np.log(probabilities + 1e-12))
            / np.log(len(probabilities))
        )
    return probabilities, entropy


def substrate_features(
    model,
    events: Iterable[float] = None,
    novelty_score: float = 0.0,
):
    """Encode model health and recent input statistics without task-specific fields."""
    probabilities, entropy = _lag_profile(model)
    history = np.asarray(
        list(model.hist if events is None else events), dtype=float
    ).reshape(-1)
    window = len(model.hist)
    if window:
        history = history[-window:]
        if len(history) < window:
            history = np.pad(history, (window - len(history), 0))
        event_rate = float(history.mean())
        switch_rate = (
            float(np.abs(np.diff(history)).mean()) if window > 1 else 0.0
        )
    else:
        event_rate = switch_rate = 0.0

    return np.concatenate(
        (
            np.asarray(
                [
                    np.log1p(max(float(model.err), 0.0)),
                    abs(float(model.b)),
                    entropy,
                    float(probabilities.max()),
                ]
            ),
            probabilities,
            np.asarray([event_rate, switch_rate, float(novelty_score)]),
        )
    )


def planned_substrate_features(model, sequences, novelty_model=None):
    """Model diagnostics plus the planned recent input pattern for each candidate."""
    sequences = np.asarray(sequences, dtype=float)
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape (candidates, horizon)")
    base = substrate_features(model)
    window = len(model.hist)
    result = np.broadcast_to(base, (len(sequences), len(base))).copy()
    if not window:
        return result

    past = np.asarray(model.hist, dtype=float)[::-1]
    combined = np.concatenate(
        [np.broadcast_to(past, (len(sequences), len(past))), sequences], axis=1
    )
    recent = combined[:, -window:]
    result[:, -3] = recent.mean(axis=1)
    result[:, -2] = (
        np.abs(np.diff(recent, axis=1)).mean(axis=1) if window > 1 else 0.0
    )
    if novelty_model is not None:
        result[:, -1] = novelty_model.score_batch(recent[:, ::-1])
    return result
