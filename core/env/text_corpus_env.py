"""Streaming Chinese/text corpus environment for sparse sequence prediction.

Reads a bounded prefix of a UTF-8 text file (without loading multi-GB files
into memory), builds a Top-V character vocabulary (plus ``<UNK>=0``), and
exposes the standard task-agnostic ``env_factory`` protocol:
  - ``obs()``: fixed-width float vector of normalized recent context tokens
    (zero target leakage) + ``alive_clock``.
  - ``sparse_keys()``: order-sensitive unigram/bigram/trigram keys for sparse
    inverted-index retrieval.
  - ``step(action)``: compares predicted token ``action`` against the true
    next token, returning ``dopamine=1.0`` on exact match and ``pain=0.25``
    (or ``1.0`` on episode end) on mismatch.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import List, Optional, Sequence, Tuple

import numpy as np


def load_corpus_tokens(
    path: str,
    *,
    max_chars: int = 80000,
    vocab_size: int = 256,
    vocab: Optional[dict] = None,
    offset_chars: int = 0,
) -> Tuple[np.ndarray, dict, List[str]]:
    """Read a bounded slice of ``path`` and encode characters into ``[0, vocab_size)``."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if offset_chars > 0:
            f.read(int(offset_chars))
        text = f.read(int(max_chars))
    if not text:
        raise ValueError(f"empty text slice from {path!r}")
    if vocab is None:
        counts = Counter(text)
        keep = max(int(vocab_size) - 1, 1)
        itos = ["<UNK>"] + [ch for ch, _ in counts.most_common(keep)]
        vocab = {ch: i for i, ch in enumerate(itos)}
    else:
        itos = [None] * len(vocab)
        for ch, idx in vocab.items():
            if 0 <= idx < len(itos):
                itos[idx] = ch
        itos = [ch if ch is not None else "<UNK>" for ch in itos]
    tokens = np.asarray([vocab.get(ch, 0) for ch in text], dtype=np.int32)
    return tokens, vocab, itos


class TextCorpusSeqEnv:
    """Next-token prediction environment over a tokenized text stream."""

    def __init__(
        self,
        tokens: np.ndarray,
        vocab_size: int,
        *,
        seed: int = 0,
        ctx_len: int = 4,
        lag: int = 0,
        cap: int = 200,
        start_pos: Optional[int] = None,
    ):
        self.tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        self.vocab_size = max(int(vocab_size), 2)
        self.action_space = tuple(range(self.vocab_size))
        self.seed = int(seed)
        self.ctx_len = max(int(ctx_len), 1)
        self.lag = max(int(lag), 0)
        self.max_t = int(cap)
        self._fixed_start = start_pos
        if len(self.tokens) <= self.ctx_len + 2:
            raise ValueError("token stream too short for ctx_len")
        self.reset()

    def reset(self):
        rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        max_start = max(len(self.tokens) - self.max_t - 2, self.ctx_len + 1)
        if self._fixed_start is not None:
            self.pos = min(max(int(self._fixed_start), self.ctx_len), max_start)
        else:
            self.pos = int(rng.randint(self.ctx_len, max_start + 1))
        self._q = deque([0] * (self.lag + 1), maxlen=self.lag + 1)
        self.t = 0
        self.dead = False
        self.passed = 0
        self._last_match = False
        self._alive = 0.0
        return self.obs()

    def context_tokens(self) -> Tuple[int, ...]:
        start = self.pos - self.ctx_len
        return tuple(int(x) for x in self.tokens[start:self.pos])

    def sparse_keys(self) -> Tuple[Tuple[int, ...], ...]:
        """Order-sensitive unigram, bigram, and trigram keys for sparse lookup."""
        ctx = self.context_tokens()
        keys: List[Tuple[int, ...]] = [(-1, ctx[-1])]
        if len(ctx) >= 2:
            keys.append((-2, ctx[-2]))
            keys.append((12, ctx[-2], ctx[-1]))
        if len(ctx) >= 3:
            keys.append((13, ctx[-3], ctx[-2], ctx[-1]))
        if len(ctx) >= 4:
            keys.append((14, ctx[-4], ctx[-3], ctx[-2], ctx[-1]))
        return tuple(keys)

    def pending(self):
        return list(self._q)[1:]

    def feedback(self) -> dict:
        if self._last_match:
            return dict(dopamine=1.0, pain=0.0)
        return dict(dopamine=0.0, pain=1.0 if self.dead else 0.25)

    def obs(self) -> np.ndarray:
        ctx = self.context_tokens()
        inv_v = 1.0 / float(self.vocab_size)
        feats = [float(tok) * inv_v for tok in ctx]
        feats.append(self._alive)
        return np.asarray(feats, dtype=np.float32)

    def step(self, action):
        self._q.append(int(action))
        eff_action = int(self._q[0]) % self.vocab_size
        target = int(self.tokens[self.pos])
        self._last_match = (eff_action == target)
        if self._last_match:
            self.passed += 1
        self.pos += 1
        self.t += 1
        self._alive += 1.0
        self.dead = bool(self.t >= self.max_t or self.pos >= len(self.tokens) - 1)
        dopamine = 1.0 if self._last_match else 0.0
        return self.obs(), dopamine, self.dead
