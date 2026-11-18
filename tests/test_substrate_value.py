import unittest

import numpy as np

from substrate_value import InputNovelty, WorldFeedbackValueHead


class SubstrateValueTests(unittest.TestCase):
    def test_feedback_head_learns_world_channels(self):
        head = WorldFeedbackValueHead(
            feature_dim=2, hidden_dim=8, learning_rate=0.02, discount=0.0
        )
        dopamine_state = np.array([1.0, 0.0])
        pain_state = np.array([0.0, 1.0])

        for _ in range(200):
            head.update(
                dopamine_state, [1.0, 0.0], dopamine_state, terminal=True
            )
            head.update(pain_state, [0.0, 1.0], pain_state, terminal=True)

        dopamine_value = head.value(dopamine_state)
        pain_value = head.value(pain_state)
        self.assertGreater(dopamine_value[0], dopamine_value[1])
        self.assertGreater(pain_value[1], pain_value[0])

    def test_input_novelty_uses_only_observed_feature_counts(self):
        novelty = InputNovelty(feature_dim=2)
        self.assertEqual(novelty.score([0.0, 1.0]), 1.0)
        self.assertEqual(novelty.observe([0.0, 1.0]), 1.0)
        self.assertAlmostEqual(novelty.score([0.0, 1.0]), 1.0 / np.sqrt(2.0))
        self.assertEqual(novelty.score([1.0, 0.0]), 1.0)

    def test_input_novelty_memory_is_bounded(self):
        novelty = InputNovelty(feature_dim=1, max_patterns=2)
        novelty.observe([0.0])
        novelty.observe([1.0])
        novelty.observe([2.0])
        self.assertEqual(len(novelty.counts), 2)
        self.assertEqual(novelty.score([0.0]), 1.0)

    def test_novelty_is_scored_from_observed_feature_counts(self):
        novelty = InputNovelty(feature_dim=2)
        self.assertEqual(novelty.score([1.0, 0.0]), 1.0)
        self.assertEqual(novelty.observe([1.0, 0.0]), 1.0)
        self.assertLess(novelty.score([1.0, 0.0]), 1.0)
        self.assertEqual(novelty.score([0.0, 1.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
