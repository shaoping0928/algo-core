"""Unit tests for ClassLevelSequencePolicy (self-discovered symbol classes & class transitions)."""
from __future__ import annotations

import unittest
import numpy as np


class ClassLevelPolicyTests(unittest.TestCase):
    def test_class_policy_learns_symbol_classes_and_generalizes_transitions(self):
        from class_policy import ClassLevelSequencePolicy

        # Structured class-level grammar: A-class {1,2} -> B-class {3,4} -> C-class {5,6}
        rng = np.random.RandomState(0)
        seq = []
        for _ in range(150):
            seq.append(int(rng.choice([1, 2])))
            seq.append(int(rng.choice([3, 4])))
            seq.append(int(rng.choice([5, 6])))
        tokens = np.asarray(seq, dtype=np.int32)

        pol = ClassLevelSequencePolicy(
            vocab_size=8, n_classes=4, feature_dim=8, window=2,
            class_ctx_len=3, local_ctx_len=2, seed=42,
        ).fit(tokens)

        self.assertEqual(pol.sym_to_class_.shape, (8,))
        cids, cprobs = pol.predict_next_classes([1, 3], top_c=3)
        self.assertEqual(len(cids), 3)
        self.assertGreater(float(cprobs[0]), 0.0)
        # Next symbol after [1, 3] should belong to C-class {5, 6}
        cand, probs = pol.propose_next_symbols([1, 3], k=4, local_weight=0.3)
        self.assertIn(cand[0], (5, 6))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
