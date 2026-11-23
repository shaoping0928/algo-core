"""Tests for the 6-step vs 8-step executor horizon probe."""

import unittest

from executor_horizon_probe import horizon_config, summarize


class ExecutorHorizonProbeTest(unittest.TestCase):
    def test_config_matches_default_stack(self):
        config = horizon_config(101, 6)
        self.assertEqual(config["seed"], 101)
        self.assertEqual(config["dagger_horizon"], 6)
        self.assertEqual(config["steps"], 1500)
        self.assertEqual(config["epochs"], 1)
        self.assertNotIn("death_rules", config)

    def test_summarize_pairs_and_counts_directions(self):
        rows = [
            dict(train_seed=100, horizon=6, eval_seed=120, frames=150,
                 passes=2, action_rate=0.1),
            dict(train_seed=100, horizon=8, eval_seed=120, frames=140,
                 passes=2, action_rate=0.1),
            dict(train_seed=100, horizon=6, eval_seed=121, frames=100,
                 passes=1, action_rate=0.1),
            dict(train_seed=100, horizon=8, eval_seed=121, frames=120,
                 passes=1, action_rate=0.1),
        ]
        summary = summarize(rows)
        overall = summary["overall"]
        self.assertEqual(overall["n"], 2)
        self.assertAlmostEqual(overall["mean_delta_frames"], -5.0)
        self.assertEqual(overall["improved"], 1)
        self.assertEqual(overall["worsened"], 1)
        self.assertEqual(len(summary["pairs"]), 2)
        self.assertAlmostEqual(
            summary["per_seed"][100]["mean_delta_frames"], -5.0
        )


if __name__ == "__main__":
    unittest.main()
