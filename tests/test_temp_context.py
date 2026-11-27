"""Unit tests for task-agnostic TemporaryContextMemory (working memory + episodic cache)."""
from __future__ import annotations

import unittest
import numpy as np


class TemporaryContextMemoryTests(unittest.TestCase):
    def test_spreading_activation_and_episodic_cache_boost_context_predictions(self):
        from temp_context import TemporaryContextMemory

        # Two distinct long-range co-occurring domains:
        #   Domain A: symbols 10..15; Domain B: symbols 20..25; glue symbols 1..3
        rng = np.random.RandomState(0)
        seq = []
        for _ in range(80):
            dom = list(range(10, 16)) if rng.rand() < 0.5 else list(range(20, 26))
            for _ in range(12):
                seq.append(int(rng.choice(dom)))
                if rng.rand() < 0.3:
                    seq.append(int(rng.choice([1, 2, 3])))
        tokens = np.asarray(seq, dtype=np.int32)

        tcm = TemporaryContextMemory(
            vocab_size=30, assoc_window=8, top_assoc_per_sym=10, min_cooc=2,
        ).fit_associations(tokens, stride=1)

        # Observe novel episodic sequence [10, 11, 29] inside current episode
        tcm.reset()
        for sym in (10, 11, 29, 10, 11):
            tcm.observe(sym)
        # Spreading field should activate Domain A neighbors (12..15) more than Domain B (20..25)
        dom_a_act = float(tcm._field[12:16].mean())
        dom_b_act = float(tcm._field[20:26].mean())
        self.assertGreater(dom_a_act, dom_b_act)

        # Episodic cache should strongly recall 29 after [10, 11] even if base_probs gave it 0
        cand, probs = tcm.apply_to_candidates(
            [10, 11], candidates=[1, 2, 20], base_probs=np.array([0.5, 0.3, 0.2]),
            field_beta=0.5, cache_lambda=0.4,
        )
        self.assertIn(29, cand)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

        # Reset must completely clear temporary working memory
        tcm.reset()
        self.assertEqual(float(tcm._field.sum()), 0.0)
        self.assertEqual(tcm.episodic_cache_probs([10, 11]), {})


if __name__ == "__main__":
    unittest.main()
