import unittest

import numpy as np

from substrate_value import (
    InputNovelty,
    ReturnReplay,
    TerminalRiskHead,
    TransitionReplay,
    WorldFeedbackValueHead,
    binary_auc,
    discounted_returns,
    fit_terminal_risk,
    fit_value_head,
    fit_value_head_returns,
    value_probe_metrics,
)


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

    def test_binary_auc_handles_ordering_and_ties(self):
        self.assertAlmostEqual(binary_auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(binary_auc([4, 3, 2, 1], [0, 0, 1, 1]), 0.0)
        self.assertAlmostEqual(binary_auc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5)
        self.assertTrue(np.isnan(binary_auc([1, 2], [0, 0])))

    def test_value_probe_reports_terminal_separation(self):
        class FixedHead:
            def value(self, features):
                return np.column_stack([features[:, 0], features[:, 1]])

        records = [
            {"features": np.array([1.0, 0.1]), "terminal": False,
             "y": 0.5, "pipe_dx": 0.5},
            {"features": np.array([1.0, 0.9]), "terminal": True,
             "y": 1.1, "pipe_dx": 0.02},
        ]
        metrics = value_probe_metrics(records, FixedHead())
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["terminal"], 1)
        self.assertAlmostEqual(metrics["pain_safe"], 0.1)
        self.assertAlmostEqual(metrics["pain_terminal"], 0.9)
        self.assertAlmostEqual(metrics["pain_auc"], 1.0)
        self.assertAlmostEqual(metrics["near_pipe_pain"], 0.9)

    def test_transition_replay_balances_terminal_samples(self):
        replay = TransitionReplay(capacity=40, seed=0)
        for index in range(10):
            replay.add([float(index)], [1.0, 0.0], [float(index)], False)
            replay.add([float(100 + index)], [0.0, 1.0], [float(100 + index)], True)

        batch = replay.sample(batch_size=8, terminal_fraction=0.5)
        self.assertIsNotNone(batch)
        features, feedback, next_features, terminal = batch
        self.assertEqual(len(features), 8)
        self.assertEqual(feedback.shape, (8, 2))
        self.assertEqual(next_features.shape, features.shape)
        self.assertEqual(int(terminal.sum()), 4)

    def test_stream_replay_keeps_every_transition(self):
        replay = TransitionReplay.for_stream(600, seed=0)
        for index in range(600):
            replay.add([float(index)], [1.0, 0.0], [float(index)], False)
        for index in range(600):
            replay.add([float(index)], [0.0, 1.0], [float(index)], True)
        self.assertEqual(len(replay), 1200)
        self.assertEqual(len(replay.alive), 600)
        self.assertEqual(len(replay.terminal), 600)

    def test_terminal_risk_prior_correction_moves_towards_base_rate(self):
        head = TerminalRiskHead(feature_dim=1)
        head.calibrate(base_rate=0.05, sample_rate=0.5)
        probability = head.probability(np.zeros((1, 1)))[0]
        self.assertAlmostEqual(probability, 0.05, places=6)
        head.calibrate(base_rate=0.5, sample_rate=0.5)
        self.assertEqual(head.prior_shift, 0.0)

    def test_terminal_risk_interaction_block_is_selectable(self):
        head = TerminalRiskHead(
            feature_dim=4, interaction_dim=2, interaction_start=1
        )
        expanded = head._expand(np.asarray([[0.0, 1.0, 1.0, 0.0]]))
        self.assertAlmostEqual(float(expanded[0, -1]), 1.0)
        head = TerminalRiskHead(
            feature_dim=4, interaction_dim=2, interaction_start=2
        )
        expanded = head._expand(np.asarray([[0.0, 1.0, 1.0, 0.0]]))
        self.assertAlmostEqual(float(expanded[0, -1]), 0.0)

    def test_fit_value_head_uses_replay_batch(self):
        head = WorldFeedbackValueHead(
            feature_dim=2, hidden_dim=8, learning_rate=0.05, discount=0.0
        )
        replay = TransitionReplay(capacity=20, seed=1)
        for _ in range(10):
            replay.add([1.0, 0.0], [1.0, 0.0], [1.0, 0.0], False)
            replay.add([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], True)
        completed = fit_value_head(
            head, replay, updates=100, batch_size=8, terminal_fraction=0.5
        )
        self.assertEqual(completed, 100)
        states = np.array([[1.0, 0.0], [0.0, 1.0]])
        values = head.value(states)
        self.assertGreater(values[0, 0], values[1, 0])
        self.assertGreater(values[1, 1], values[0, 1])

    def test_terminal_risk_head_learns_pain_label(self):
        replay = TransitionReplay(capacity=200, seed=2)
        for index in range(50):
            safe = np.array([1.0, float(index) / 50.0])
            danger = np.array([-1.0, float(index) / 50.0])
            replay.add(safe, [1.0, 0.0], safe, False)
            replay.add(danger, [0.0, 1.0], danger, True)
        head = TerminalRiskHead(feature_dim=2, learning_rate=0.2)
        completed = fit_terminal_risk(
            head, replay, updates=300, batch_size=32, terminal_fraction=0.5
        )
        self.assertEqual(completed, 300)
        states = np.array([[1.0, 0.5], [-1.0, 0.5]])
        probability = head.probability(states)
        self.assertGreater(probability[1], 0.5)
        self.assertLess(probability[0], 0.5)
        self.assertGreater(binary_auc(probability, [False, True]), 0.5)

    def test_discounted_returns_use_terminal_pain(self):
        transitions = [
            {"signals": np.array([1.0, 0.0]), "terminal": False},
            {"signals": np.array([1.0, 0.0]), "terminal": False},
            {"signals": np.array([0.0, 1.0]), "terminal": True},
        ]
        returns = discounted_returns(transitions, 0.5)
        np.testing.assert_allclose(returns[0], [1.5, 0.25])
        np.testing.assert_allclose(returns[1], [1.0, 0.5])
        np.testing.assert_allclose(returns[2], [0.0, 1.0])

    def test_return_replay_and_value_update(self):
        head = WorldFeedbackValueHead(
            feature_dim=2, hidden_dim=8, learning_rate=0.05, discount=0.0
        )
        replay = ReturnReplay(capacity=20, seed=3)
        for _ in range(10):
            replay.add([1.0, 0.0], [10.0, 0.0])
            replay.add([0.0, 1.0], [0.0, 10.0])
        completed = fit_value_head_returns(
            head, replay, updates=100, batch_size=8
        )
        self.assertEqual(completed, 100)
        values = head.value(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertGreater(values[0, 0], values[1, 0])
        self.assertGreater(values[1, 1], values[0, 1])

    def test_value_probe_accepts_risk_head(self):
        class FixedHead:
            def value(self, features):
                return np.column_stack([features[:, 0], features[:, 1]])

        class FixedRisk:
            def probability(self, features):
                return features[:, 1]

        records = [
            {"features": np.array([1.0, 0.1]), "terminal": False,
             "pipe_dx": 0.1, "gap_dy": 0.05},
            {"features": np.array([1.0, 0.9]), "terminal": True,
             "pipe_dx": 0.1, "gap_dy": 0.5},
        ]
        metrics = value_probe_metrics(
            records, FixedHead(), FixedRisk(), gap=0.34
        )
        self.assertAlmostEqual(metrics["risk_auc"], 1.0)
        self.assertAlmostEqual(metrics["risk_safe"], 0.1)
        self.assertAlmostEqual(metrics["risk_terminal"], 0.9)
        self.assertAlmostEqual(metrics["near_pipe_risk_auc"], 1.0)
        self.assertAlmostEqual(metrics["near_inside_risk"], 0.1)
        self.assertAlmostEqual(metrics["near_outside_risk"], 0.9)

    def test_replay_priority_favours_high_error_transitions(self):
        replay = TransitionReplay(capacity=2000, seed=5)
        for _ in range(500):
            replay.add([0.0], [1.0, 0.0], [0.0], False, priority=1.0)
            replay.add([1.0], [1.0, 0.0], [1.0], False, priority=100.0)
        batch = replay.sample(200, terminal_fraction=0.0, priority_alpha=2.0)
        self.assertIsNotNone(batch)
        features = batch[0][:, 0]
        self.assertGreater(np.sum(features == 1.0), np.sum(features == 0.0))

    def test_return_replay_priority_favours_high_error_targets(self):
        replay = ReturnReplay(capacity=20, seed=6)
        replay.add([0.0], [1.0, 0.0], priority=1.0)
        replay.add([1.0], [2.0, 0.0], priority=100.0)
        features, _targets = replay.sample(200, priority_alpha=2.0)
        self.assertGreater(np.sum(features[:, 0] == 1.0),
                           np.sum(features[:, 0] == 0.0))

    def test_terminal_risk_interaction_features_learn_product_rule(self):
        replay = TransitionReplay(capacity=200, seed=4)
        corners = np.asarray([[-1.0, -1.0], [-1.0, 1.0],
                              [1.0, -1.0], [1.0, 1.0]])
        labels = [False, True, True, False]
        for _ in range(20):
            for features, label in zip(corners, labels):
                replay.add(features, [1.0, 0.0], features, label)
        head = TerminalRiskHead(
            feature_dim=2, learning_rate=0.1, interaction_dim=2
        )
        completed = fit_terminal_risk(
            head, replay, updates=1000, batch_size=32,
            terminal_fraction=0.5,
        )
        self.assertEqual(completed, 1000)
        probability = head.probability(corners)
        self.assertGreater(probability[1], probability[0])
        self.assertGreater(probability[2], probability[0])
        self.assertLess(probability[3], probability[1])
        self.assertLess(probability[3], probability[2])


if __name__ == "__main__":
    unittest.main()
