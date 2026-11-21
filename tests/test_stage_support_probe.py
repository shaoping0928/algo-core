"""Regression tests for the stage-support probe's segmentation math."""
import unittest

import numpy as np

from stage_support_probe import (
    segment_cost,
    segment_episode,
    single_link_clusters,
)


class StageSupportProbeTest(unittest.TestCase):
    def test_bic_split_finds_a_piecewise_switch(self):
        # Two regimes with opposite deltas: the split must land on the switch.
        deltas = np.concatenate([
            np.tile([1.0, 0.0], (10, 1)),
            np.tile([-1.0, 0.0], (10, 1)),
        ])
        actions = [index % 2 for index in range(20)]
        segments = segment_episode(deltas, actions, min_length=5)
        starts = [start for start, _end in segments]
        self.assertTrue(
            any(abs(start - 10) <= 1 for start in starts),
            "expected a boundary near the switch, got %s" % starts,
        )

    def test_constant_stream_stays_single_stage(self):
        deltas = np.tile([1.0, 0.0], (20, 1))
        actions = [index % 2 for index in range(20)]
        self.assertEqual(segment_episode(deltas, actions), [(0, 20)])

    def test_cost_prefers_a_split_when_regimes_differ(self):
        deltas = np.concatenate([
            np.tile([1.0], (6, 1)),
            np.tile([-1.0], (6, 1)),
        ])
        actions = [0] * 12
        whole = segment_cost(deltas, actions, 0, 12)
        split = (
            segment_cost(deltas, actions, 0, 6)
            + segment_cost(deltas, actions, 6, 12)
        )
        self.assertGreater(whole - split, 0.0)

    def test_single_link_groups_near_vectors(self):
        vectors = np.asarray([[0.0, 0.0], [0.01, 0.0], [5.0, 5.0]])
        labels = single_link_clusters(vectors, radius=0.5)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])


if __name__ == "__main__":
    unittest.main()
