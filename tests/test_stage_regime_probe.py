"""Smoke tests for stage_regime_probe episode splitting and metrics."""
import unittest

from stage_regime_probe import evaluate_seed, split_rows_by_episode


class StageRegimeProbeTests(unittest.TestCase):
    def test_episode_split_has_zero_leakage(self):
        rows = [
            dict(episode=ep, step=t)
            for ep in range(10) for t in range(3)
        ]
        tr, va = split_rows_by_episode(rows, folds=5, validation_fold=4)
        tr_ids = {r["episode"] for r in tr}
        va_ids = {r["episode"] for r in va}
        self.assertEqual(tr_ids.intersection(va_ids), set())
        self.assertEqual(len(va_ids), 2)

    def test_evaluate_seed_smoke_runs_causally(self):
        res = evaluate_seed(100, steps=1200, n_regimes=8, min_dwell=5)
        self.assertIn(6, res["summarizer"]["drift_dims"])
        self.assertEqual(res["singleton_profiles"], 0)
        self.assertGreater(res["one_step_pool_all"]["n"], 0)
        self.assertIn("k=2", res["multistep_rollout"])


if __name__ == "__main__":
    unittest.main()
