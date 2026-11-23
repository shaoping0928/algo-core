"""Tests for the B0.6 predicted-vs-true risk readout probe."""

import unittest

import numpy as np

from risk_readout_probe import (
    _terminal_factor, predicted_rollout, summarize, true_rollout,
)


class FakeAgent:
    obs_dim = 1
    scales = np.ones(4)
    risk_error_weight = False
    step_errors = []

    def pain_signal_batch(self, states, mask_table=None):
        return np.asarray(states[:, 0] > 5.0, dtype=float)

    def valence_state_batch(self, states):
        return np.zeros(len(states)), np.zeros(len(states))

    def _feature_mask_table(self, states):
        return None


class FakeWorld:
    def predict(self, states, histories):
        return np.asarray(states, dtype=float) + 1.0


class FakeEnv:
    def __init__(self):
        self.value = 0.0
        self.dead = False

    def obs(self):
        return np.asarray([self.value])

    def step(self, action):
        self.value += 1.0
        return self.obs(), 1.0, False


class RiskReadoutProbeTest(unittest.TestCase):
    def test_terminal_factor_is_one_without_decay(self):
        self.assertAlmostEqual(_terminal_factor(FakeAgent(), 8, 1.0), 1.0)
        self.assertAlmostEqual(
            _terminal_factor(FakeAgent(), 2, 0.5), 0.25
        )

    def test_predicted_rollout_adds_terminal_risk(self):
        agent = FakeAgent()
        result = predicted_rollout(
            FakeEnv(), np.zeros(8, dtype=int), FakeWorld(), agent,
            np.zeros(3), horizon=8, lag=3, stride=2,
        )
        self.assertEqual(result["step_total"], 1.0)
        self.assertEqual(result["terminal"], 1.0)
        self.assertEqual(result["total"], 2.0)
        self.assertEqual(result["terms"], [0.0, 0.0, 0.0, 1.0])

    def test_true_rollout_uses_true_states(self):
        agent = FakeAgent()
        result = true_rollout(
            FakeEnv(), np.zeros(8, dtype=int), agent,
            np.zeros(3), horizon=8, lag=3, stride=2,
        )
        self.assertEqual(result["death"], -1)
        self.assertEqual(result["step_total"], 1.0)
        self.assertEqual(result["terminal"], 1.0)
        self.assertEqual(result["total"], 2.0)

    def test_summarize_counts_step_and_terminal(self):
        base = dict(
            action_regret=0, suffix_regret=2,
            risk_true_correct=True, risk_true_step_correct=True,
            risk_true_terminal_correct=True,
            risk_first_true_correct=True, risk_pred_correct=False,
            risk_pred_terminal_correct=False,
            pred_match_selected=True, pred_match_best=True,
            true_step_margin=0.1, true_total_margin=1.0,
            true_terminal_margin=0.9, first_true_margin=0.0,
            pred_step_margin=0.0, pred_margin=-0.1,
            pred_terminal_margin=-0.1, state_error_mean=0.2,
        )
        summary = summarize([base])
        self.assertEqual(summary["plan_level"], 1)
        self.assertEqual(summary["risk_true_step_correct"], 1)
        self.assertEqual(summary["risk_pred_terminal_correct"], 0)
        self.assertEqual(summary["pred_match"], 1)
        self.assertEqual(summary["verdict"], "world_prediction_drift")


if __name__ == "__main__":
    unittest.main()
