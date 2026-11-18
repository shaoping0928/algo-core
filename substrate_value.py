"""Value learning from world-emitted dopamine and pain channels."""
from collections import OrderedDict, deque
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

    def update_return(self, features, target):
        """Supervised update to a Monte-Carlo return target for one state."""
        x = self._features(features)
        if x.ndim != 1:
            raise ValueError("update_return expects one feature vector")
        target = np.asarray(target, dtype=float).reshape(-1)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("return target must contain two finite channels")
        hidden = np.tanh(x @ self.w1 + self.b1)
        current = hidden @ self.w2 + self.b2
        error = target - current
        learning_rate = self.learning_rate / (1.0 + float(x @ x))
        old_w2 = self.w2.copy()
        self.w2 += learning_rate * np.outer(hidden, error)
        self.b2 += learning_rate * error
        hidden_error = (old_w2 @ error) * (1.0 - hidden * hidden)
        self.w1 += learning_rate * np.outer(x, hidden_error)
        self.b1 += learning_rate * hidden_error
        return error

    def update_batch(self, features, feedback, next_features, terminal):
        """Apply the online update to a replay batch; returns the number of samples."""
        features = np.asarray(features, dtype=float)
        feedback = np.asarray(feedback, dtype=float)
        next_features = np.asarray(next_features, dtype=float)
        terminal = np.asarray(terminal, dtype=bool).reshape(-1)
        count = len(features)
        if count == 0:
            return 0
        if feedback.shape != (count, 2):
            raise ValueError("feedback batch must have shape (n, 2)")
        if next_features.shape != features.shape:
            raise ValueError("next_features must match features")
        if terminal.shape != (count,):
            raise ValueError("terminal must have one flag per sample")
        for index in range(count):
            self.update(
                features[index], feedback[index], next_features[index],
                terminal=bool(terminal[index]),
            )
        return count


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


def binary_auc(scores, labels):
    """Rank-based AUC; ties receive the average rank."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same length")
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")

    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    ordered = combined[order]
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop] == ordered[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop

    positive_rank_sum = ranks[:len(positive)].sum()
    return float(
        (positive_rank_sum - len(positive) * (len(positive) + 1) / 2.0)
        / (len(positive) * len(negative))
    )


def value_probe_metrics(records, head, risk_head=None, gap=None):
    """Summarise a held-out set of real transitions without using their labels for training."""
    if not records:
        return {"n": 0}
    features = np.asarray([row["features"] for row in records], dtype=float)
    terminal = np.asarray([row["terminal"] for row in records], dtype=bool)
    values = head.value(features)
    pain, dopamine = values[:, 1], values[:, 0]
    safe = ~terminal
    metrics = {
        "n": len(records),
        "terminal": int(terminal.sum()),
        "safe": int(safe.sum()),
        "pain_safe": float(pain[safe].mean()) if safe.any() else float("nan"),
        "pain_terminal": (
            float(pain[terminal].mean()) if terminal.any() else float("nan")
        ),
        "dopamine_safe": (
            float(dopamine[safe].mean()) if safe.any() else float("nan")
        ),
        "dopamine_terminal": (
            float(dopamine[terminal].mean()) if terminal.any() else float("nan")
        ),
        "pain_auc": binary_auc(pain, terminal),
    }
    if risk_head is not None:
        risk = risk_head.probability(features)
        metrics.update({
            "risk_safe": float(risk[safe].mean()) if safe.any() else float("nan"),
            "risk_terminal": (
                float(risk[terminal].mean()) if terminal.any() else float("nan")
            ),
            "risk_auc": binary_auc(risk, terminal),
            "risk_brier": float(np.mean((risk - terminal.astype(float)) ** 2)),
        })
        near = np.asarray([
            abs(row.get("pipe_dx", 1.0)) < 0.20
            and row.get("pipe_dx", -1.0) > -0.10
            for row in records
        ], dtype=bool)
        metrics["near_pipe_risk_auc"] = (
            binary_auc(risk[near], terminal[near]) if near.any() else float("nan")
        )
        if gap is not None:
            inside = np.asarray([
                abs(row.get("gap_dy", 1.0)) <= gap / 2.0 for row in records
            ], dtype=bool)
            for name, mask in (
                ("near_inside", near & inside),
                ("near_outside", near & ~inside),
            ):
                metrics[name + "_n"] = int(mask.sum())
                metrics[name + "_risk"] = (
                    float(risk[mask].mean()) if mask.any() else float("nan")
                )
                metrics[name + "_terminal_rate"] = (
                    float(terminal[mask].mean()) if mask.any() else float("nan")
                )
    for name, predicate in (
        ("high_y", lambda row: row.get("y", 0.0) > 0.9),
        ("low_y", lambda row: row.get("y", 0.0) < 0.1),
        ("near_pipe", lambda row: abs(row.get("pipe_dx", 1.0)) < 0.08),
    ):
        mask = np.asarray([predicate(row) for row in records], dtype=bool)
        metrics[name + "_n"] = int(mask.sum())
        metrics[name + "_pain"] = (
            float(pain[mask].mean()) if mask.any() else float("nan")
        )
    return metrics


class TransitionReplay:
    """Store real transitions by outcome class and sample without discarding terminal cases."""

    def __init__(self, capacity=20000, seed=0):
        if capacity < 2:
            raise ValueError("capacity must allow at least two transitions")
        half = int(capacity) // 2
        self.alive = deque(maxlen=half)
        self.terminal = deque(maxlen=int(capacity) - half)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.alive) + len(self.terminal)

    def add(self, features, signals, next_features, terminal, metadata=None):
        item = (
            np.asarray(features, dtype=float).reshape(-1),
            np.asarray(signals, dtype=float).reshape(-1),
            np.asarray(next_features, dtype=float).reshape(-1),
            bool(terminal),
            dict(metadata or {}),
        )
        if item[3]:
            self.terminal.append(item)
        else:
            self.alive.append(item)

    def _draw(self, pool, count, replace):
        if count <= 0:
            return []
        indices = self.rng.choice(len(pool), size=count, replace=replace)
        return [pool[int(index)] for index in indices]

    @staticmethod
    def _near_pipe(item):
        metadata = item[4]
        pipe_dx = metadata.get("pipe_dx", 1.0)
        return abs(pipe_dx) < 0.20 and pipe_dx > -0.10

    def _balanced_items(self, batch_size, terminal_fraction):
        terminal_count = min(
            len(self.terminal), int(round(batch_size * terminal_fraction))
        )
        alive_count = min(len(self.alive), batch_size - terminal_count)
        remaining = batch_size - terminal_count - alive_count
        if remaining:
            extra_terminal = min(remaining, len(self.terminal) - terminal_count)
            terminal_count += extra_terminal
            remaining -= extra_terminal
        if remaining:
            alive_count += min(remaining, len(self.alive) - alive_count)

        selected = self._draw(
            self.terminal, terminal_count, terminal_count > len(self.terminal)
        )
        selected += self._draw(
            self.alive, alive_count, alive_count > len(self.alive)
        )
        return selected

    def arrays(self):
        """Return every stored transition as arrays, without sampling."""
        items = list(self.alive) + list(self.terminal)
        if not items:
            return None
        return (
            np.asarray([item[0] for item in items], dtype=float),
            np.asarray([item[1] for item in items], dtype=float),
            np.asarray([item[2] for item in items], dtype=float),
            np.asarray([item[3] for item in items], dtype=bool),
        )

    def sample(self, batch_size, terminal_fraction=0.5,
               near_pipe_fraction=0.0):
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= terminal_fraction <= 1.0:
            raise ValueError("terminal_fraction must be in [0, 1]")
        if not 0.0 <= near_pipe_fraction <= 1.0:
            raise ValueError("near_pipe_fraction must be in [0, 1]")
        if not len(self):
            return None

        near_count = min(
            batch_size, int(round(batch_size * near_pipe_fraction))
        )
        selected = self._balanced_items(
            batch_size - near_count, terminal_fraction
        )
        if near_count:
            near_items = [
                item for pool in (self.alive, self.terminal)
                for item in pool if self._near_pipe(item)
            ]
            if near_items:
                selected += self._draw(
                    near_items, near_count, near_count > len(near_items)
                )
            else:
                selected += self._balanced_items(near_count, terminal_fraction)
        if not selected:
            return None
        order = self.rng.permutation(len(selected))
        selected = [selected[int(index)] for index in order]
        return (
            np.asarray([item[0] for item in selected], dtype=float),
            np.asarray([item[1] for item in selected], dtype=float),
            np.asarray([item[2] for item in selected], dtype=float),
            np.asarray([item[3] for item in selected], dtype=bool),
        )


def fit_value_head(head, replay, updates, batch_size=128, terminal_fraction=0.5):
    """Offline replay training after data collection; does not touch the world model."""
    if updates < 0:
        raise ValueError("updates must be non-negative")
    completed = 0
    for _ in range(int(updates)):
        batch = replay.sample(batch_size, terminal_fraction)
        if batch is None:
            break
        head.update_batch(*batch)
        completed += 1
    return completed


class TerminalRiskHead:
    """Logistic readout of the world's pain label, giving a calibrated terminal risk."""

    def __init__(self, feature_dim, learning_rate=0.05, l2=1e-4,
                 interaction_dim=0):
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.feature_dim = int(feature_dim)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.interaction_dim = min(max(int(interaction_dim), 0), self.feature_dim)
        start = self.feature_dim - self.interaction_dim
        self.interaction_pairs = [
            (i - start, j - start)
            for i in range(start, self.feature_dim)
            for j in range(i + 1, self.feature_dim)
        ]
        self.mean = np.zeros(self.feature_dim, dtype=float)
        self.scale = np.ones(self.feature_dim, dtype=float)
        # Squared terms represent two-sided boundaries; pairwise products let the
        # readout express pipe-distance x gap-offset interactions.
        self.w = np.zeros(
            2 * self.feature_dim + len(self.interaction_pairs), dtype=float
        )
        self.b = 0.0

    def set_normalization(self, features):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("expected a feature matrix with the configured width")
        self.mean = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1.0)

    def _expand(self, features):
        features = np.asarray(features, dtype=float)
        if features.ndim < 1 or features.shape[-1] != self.feature_dim:
            raise ValueError("feature dimension mismatch")
        x = (features - self.mean) / self.scale
        parts = [x, x * x]
        if self.interaction_pairs:
            block = x[..., -self.interaction_dim:]
            parts.append(np.stack(
                [block[..., i] * block[..., j]
                 for i, j in self.interaction_pairs],
                axis=-1,
            ))
        return np.concatenate(parts, axis=-1)

    def logits(self, features):
        return self._expand(features) @ self.w + self.b

    def probability(self, features):
        z = np.clip(self.logits(features), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def update_batch(self, features, labels, learning_rate=None):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float).reshape(-1)
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        if not len(features):
            return 0
        x = self._expand(features)
        probability = self.probability(features)
        error = (probability - labels) / len(labels)
        rate = self.learning_rate if learning_rate is None else float(learning_rate)
        self.w -= rate * (x.T @ error + self.l2 * self.w)
        self.b -= rate * float(error.sum())
        return len(features)


def fit_terminal_risk(head, replay, updates, batch_size=128,
                      terminal_fraction=0.5, near_pipe_fraction=0.0):
    """Train the pain/terminal readout on replay batches with balanced samplers."""
    data = replay.arrays()
    if data is None:
        return 0
    head.set_normalization(data[0])
    completed = 0
    for _ in range(int(updates)):
        batch = replay.sample(
            batch_size, terminal_fraction, near_pipe_fraction
        )
        if batch is None:
            break
        head.update_batch(batch[0], batch[3])
        completed += 1
    return completed


class ReturnReplay:
    """Monte-Carlo state/return pairs from complete real episodes."""

    def __init__(self, capacity=20000, seed=0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.data = deque(maxlen=int(capacity))
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.data)

    def add(self, features, target):
        self.data.append((
            np.asarray(features, dtype=float).reshape(-1),
            np.asarray(target, dtype=float).reshape(-1),
        ))

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.data:
            return None
        replace = batch_size > len(self.data)
        indices = self.rng.choice(len(self.data), size=batch_size, replace=replace)
        selected = [self.data[int(index)] for index in indices]
        return (
            np.asarray([item[0] for item in selected], dtype=float),
            np.asarray([item[1] for item in selected], dtype=float),
        )


def discounted_returns(transitions, discount):
    """Reverse accumulated world feedback for one real episode."""
    returns = []
    running = np.zeros(2, dtype=float)
    for transition in reversed(transitions):
        running = (
            np.asarray(transition["signals"], dtype=float) + discount * running
        )
        returns.append(running.copy())
    returns.reverse()
    return returns


def fit_value_head_returns(head, replay, updates, batch_size=128):
    """Train the value head against Monte-Carlo returns from real episodes."""
    completed = 0
    for _ in range(int(updates)):
        batch = replay.sample(batch_size)
        if batch is None:
            break
        features, targets = batch
        for index in range(len(features)):
            head.update_return(features[index], targets[index])
        completed += 1
    return completed
