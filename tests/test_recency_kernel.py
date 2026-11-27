"""Bidirectional domain-agnostic unit tests for LearnedRecencyLagKernel and multi-cue corroboration.

Verifies zero sign bias and zero language assumptions:
  1. Hold-action / dwell-time stream -> learns POSITIVE lag=1 self-excitation.
  2. Periodic symbolic stream (period 4) -> learns NEGATIVE lag=1 and POSITIVE lag=2..4.
  3. Memoryless i.i.d. stream -> converges near 0.0 (unbiased null).
  4. Non-repeating foraging stream -> learns NEGATIVE short-lag inhibition.
  5. POMDP multi-cue corroboration & high-PMI 2-symbol compound chunks -> disambiguates lone noisy cues.
"""
from __future__ import annotations

import unittest
import numpy as np


class RecencyKernelAndDisambiguationTests(unittest.TestCase):
    def test_hold_action_stream_learns_positive_short_lag_excitation(self):
        from recency_kernel import LearnedRecencyLagKernel

        rng = np.random.RandomState(0)
        seq = []
        for _ in range(300):
            act = int(rng.randint(1, 7))
            dwell = int(rng.randint(5, 10))
            seq.extend([act] * dwell)
        tokens = np.asarray(seq, dtype=np.int32)

        kern = LearnedRecencyLagKernel(vocab_size=10, n_classes=1).fit(tokens)
        # lag=1 bin (index 0) MUST learn strong positive excitation (> 0.6)
        self.assertGreater(float(kern.global_log_ratio_[0]), 0.6)
        mod = kern.modulate_scores([3, 3, 3], {3: 0.5, 4: 0.5})
        self.assertGreater(mod[3], mod[4])

    def test_periodic_stream_learns_negative_lag1_and_positive_period_lag(self):
        from recency_kernel import LearnedRecencyLagKernel

        # Period-4 deterministic cycle over 8 symbols split into two alternating 4-cycles
        seq = [1, 2, 3, 4] * 300 + [5, 6, 7, 8] * 300
        tokens = np.asarray(seq, dtype=np.int32)
        kern = LearnedRecencyLagKernel(vocab_size=12, n_classes=1).fit(tokens)
        # lag=1 never repeats -> negative; lag=4 (bin index 1: 2..4) always recurs -> positive
        self.assertLess(float(kern.global_log_ratio_[0]), -0.5)
        self.assertGreater(float(kern.global_log_ratio_[1]), 0.5)

    def test_iid_null_stream_stays_unbiased_near_zero(self):
        from recency_kernel import LearnedRecencyLagKernel

        rng = np.random.RandomState(42)
        tokens = rng.randint(1, 25, size=25000, dtype=np.int32)
        kern = LearnedRecencyLagKernel(vocab_size=30, n_classes=1).fit(tokens)
        self.assertLess(float(np.max(np.abs(kern.global_log_ratio_))), 0.15)
        self.assertLess(float(np.max(np.abs(kern.trans_log_ratio_))), 0.25)

    def test_refractory_foraging_stream_learns_short_lag_inhibition(self):
        from recency_kernel import LearnedRecencyLagKernel

        rng = np.random.RandomState(7)
        # Agent visits distinct locations without repeating within 8 steps (inhibition of return)
        seq = []
        recent = []
        for _ in range(3000):
            choices = [s for s in range(1, 16) if s not in recent[-8:]]
            nxt = int(rng.choice(choices))
            seq.append(nxt)
            recent.append(nxt)
        tokens = np.asarray(seq, dtype=np.int32)

        kern = LearnedRecencyLagKernel(vocab_size=20, n_classes=1).fit(tokens)
        # lag=1 (bin 0) and lag=2..4 (bin 1) must learn strong negative inhibition
        self.assertLess(float(kern.global_log_ratio_[0]), -0.8)
        self.assertLess(float(kern.global_log_ratio_[1]), -0.8)
        mod = kern.modulate_scores(seq[-20:], {seq[-1]: 0.5, seq[-2]: 0.5, 19: 0.5})
        self.assertGreater(mod[19], mod[seq[-1]])
        self.assertGreater(mod[19], mod[seq[-2]])

    def test_multi_anchor_corroboration_and_compound_chunk_disambiguate_aliasing(self):
        from temp_context import TemporaryContextMemory

        # Context X: ambiguous cue 10 appears with decoy 28..29
        # Context Y: corroborated cues 11, 12, 13 co-occur with true target 25
        # Compound chunk (20, 21) strongly co-occurs with target 26 (not with 28)
        rng = np.random.RandomState(3)
        seq = []
        for _ in range(50):
            seq.extend([1, 2, 10, 28, 29, 1, 2, 3])
            seq.extend([1, 2, 11, 12, 13, 25, 1, 2, 3])
            seq.extend([1, 2, 20, 21, 26, 27, 1, 2, 3])
            # Add mild noise so 1..3 are frequent glue tokens
            seq.extend([1, 2, 3, int(rng.randint(1, 4))])
        tokens = np.asarray(seq, dtype=np.int32)

        tcm = TemporaryContextMemory(
            vocab_size=35, assoc_window=5, top_assoc_per_sym=12, min_cooc=4,
            max_compounds=50, min_compound_pmi=1.5, corroboration_singleton_scale=0.2,
        ).fit_associations(tokens, stride=1)

        # Compound (20, 21) should be auto-discovered purely from PMI
        self.assertIn((20, 21), tcm.assoc_graph_compound_)

        # Multi-cue context: 11, 12, 13 corroborate 25; lone ambiguous cue 10 points to decoy 28
        tcm.reset()
        for sym in (11, 12, 13, 10):
            tcm.observe(sym, source="external")
        field = tcm.effective_field(root_blend=0.0)
        self.assertGreater(float(field[25]), float(field[28]) * 2.0)


if __name__ == "__main__":
    unittest.main()
