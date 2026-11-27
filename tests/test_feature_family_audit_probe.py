import unittest

import numpy as np

from feature_family_audit_probe import (
    _relation_features,
    _residual_features,
    _temporal_features,
    split_episode_rows,
)


def row(episode, obs, next_obs=None, history=(0.0, 0.0, 0.0, 0.0)):
    obs = np.asarray(obs, dtype=float)
    return dict(
        episode=episode,
        obs=obs,
        next_obs=(
            obs.copy() if next_obs is None
            else np.asarray(next_obs, dtype=float)
        ),
        history=np.asarray(history, dtype=float),
    )


class IdentityWorld:
    def predict(self, observations, histories):
        return np.asarray(observations, dtype=float)


class FeatureFamilyAuditTests(unittest.TestCase):
    def test_episode_split_keeps_partitions_disjoint(self):
        rows = [
            row(episode, [float(step)])
            for episode in range(10)
            for step in range(2)
        ]

        induction, calibration, test = split_episode_rows(
            rows, outer_folds=5, outer_fold=4,
            inner_folds=4, calibration_fold=3,
        )

        ids = [
            {item["episode"] for item in group}
            for group in (induction, calibration, test)
        ]
        self.assertEqual(ids[0].intersection(ids[1]), set())
        self.assertEqual(ids[0].intersection(ids[2]), set())
        self.assertEqual(ids[1].intersection(ids[2]), set())
        self.assertEqual(ids[2], {4, 9})

    def test_temporal_features_reset_at_episode_boundaries(self):
        rows = [
            row(1, [1.0]),
            row(1, [3.0]),
            row(1, [6.0]),
            row(2, [100.0]),
            row(2, [104.0]),
        ]

        features = _temporal_features(rows)

        np.testing.assert_allclose(features[1][0], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(features[1][1], [2.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(features[1][2], [3.0, 1.0, 1.0, 1.0])
        # The 100-unit episode offset is not treated as a temporal transition.
        np.testing.assert_allclose(features[2][0], [0.0, 0.0, 0.0, 0.0])

    def test_residual_uses_previous_transition_and_marks_missing_prefix(self):
        rows = [
            row(1, [2.0], next_obs=[50.0]),
            row(1, [5.0], next_obs=[60.0]),
            row(2, [100.0], next_obs=[200.0]),
            row(2, [107.0], next_obs=[300.0]),
        ]

        features = _residual_features(rows, IdentityWorld())

        # row 1 compares its current observation with the preceding model
        # prediction; row 0 has no preceding transition in its episode.
        np.testing.assert_allclose(features[1], [[0.0, 0.0], [3.0, 1.0]])
        np.testing.assert_allclose(features[2], [[0.0, 0.0], [7.0, 1.0]])

    def test_relation_features_are_pairwise_and_task_agnostic(self):
        rows = [row(1, [2.0, 5.0, -1.0])]

        features = _relation_features(rows)[1]

        self.assertEqual(features.shape, (1, 3))
        np.testing.assert_allclose(features[0], [-3.0, 3.0, 6.0])


if __name__ == "__main__":
    unittest.main()
