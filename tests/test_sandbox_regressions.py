import unittest

import numpy as np

from substrate_sandbox_search import BIRD_X, SPEED, practice, target_gap
from substrate_value import substrate_features


class SandboxRegressionTests(unittest.TestCase):
    def test_target_gap_follows_the_requested_horizon(self):
        class FakeEnv:
            pipes = [[0.5, 0.4], [1.3, 0.6]]

        self.assertAlmostEqual(target_gap(FakeEnv(), 14), 0.4)
        self.assertAlmostEqual(target_gap(FakeEnv(), 78), 0.6)
        self.assertAlmostEqual(
            target_gap(FakeEnv(), 14), 0.4
        )
        self.assertGreater(BIRD_X + 78 * SPEED, BIRD_X + 14 * SPEED)

    def test_risk_action_dim_is_validated_before_collection(self):
        for dimension in (-1, 5):
            with self.assertRaises(ValueError):
                practice(100, steps=10, value_updates=0, risk_updates=0,
                         risk_action_dim=dimension)

    def test_probe_risk_features_describe_the_next_state(self):
        lp, _lag, _f, _value, _risk, _novelty, probe, _replay, _coverage = (
            practice(100, steps=300, value_updates=0, risk_updates=0,
                     risk_action_dim=0)
        )
        self.assertTrue(probe)
        substrate_dim = len(substrate_features(lp))
        for row in probe[:20]:
            features = np.asarray(row["risk_features"], dtype=float)
            self.assertAlmostEqual(
                float(features[substrate_dim] + 0.5), row["y"], places=6
            )
            self.assertAlmostEqual(
                float(features[substrate_dim + 1] / 10.0), row["vy"], places=6
            )


if __name__ == "__main__":
    unittest.main()
