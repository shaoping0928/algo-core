"""Regression tests for the N-tie probe's tie statistics."""
import unittest

import numpy as np

from n_tie_probe import tie_statistics


class TieStatisticsTest(unittest.TestCase):
    def test_machine_tie_with_different_phi_flips(self):
        scores = np.asarray([1.0, 1.0 + 1e-13, 1.005, 2.0])
        phi = np.asarray([0.1, 0.9, 0.5, 0.9])
        stats = tie_statistics(scores, phi, tolerance=0.01)
        self.assertEqual(stats["exact"]["pairs"], 1)
        self.assertEqual(stats["exact"]["pairs_diff_phi"], 1)
        self.assertTrue(stats["exact"]["flip"])
        # The near band covers the first three candidates.
        self.assertEqual(stats["near"]["pairs"], 3)
        self.assertEqual(stats["near"]["pairs_diff_phi"], 3)
        self.assertTrue(stats["near"]["flip"])

    def test_constant_phi_never_flips(self):
        scores = np.asarray([1.0, 1.001, 1.002, 5.0])
        phi = np.ones(4)
        stats = tie_statistics(scores, phi, tolerance=0.01)
        self.assertEqual(stats["near"]["pairs"], 3)
        self.assertEqual(stats["near"]["pairs_diff_phi"], 0)
        self.assertFalse(stats["near"]["flip"])

    def test_distinct_scores_have_no_ties(self):
        scores = np.asarray([0.0, 0.5, 1.0])
        phi = np.asarray([0.2, 0.8, 0.4])
        stats = tie_statistics(scores, phi, tolerance=0.01)
        self.assertEqual(stats["near"]["pairs"], 0)
        self.assertFalse(stats["near"]["flip"])

    def test_empty_decision(self):
        stats = tie_statistics(np.asarray([]), np.asarray([]), 0.01)
        self.assertEqual(stats["near"]["pairs"], 0)
        self.assertFalse(stats["exact"]["flip"])


if __name__ == "__main__":
    unittest.main()
