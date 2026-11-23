"""Tests for the P1-D.1 event-triggered deepening probe."""

import unittest

from event_deepen_probe import summarize


class EventDeepenProbeTest(unittest.TestCase):
    def test_summarize_pairs_three_arms(self):
        rows = []
        for eval_seed, (f8, f6, fd) in enumerate(
            [(200, 150, 190), (120, 100, 140)], start=120,
        ):
            rows.append(dict(train_seed=100, eval_seed=eval_seed, arm="h8",
                             frames=f8, passes=2, stats={}))
            rows.append(dict(train_seed=100, eval_seed=eval_seed, arm="h6",
                             frames=f6, passes=1, stats={}))
            rows.append(dict(train_seed=100, eval_seed=eval_seed, arm="deep",
                             frames=fd, passes=2, stats={}))
        summary = summarize(rows, [6, 8])
        overall = summary["overall"]
        self.assertEqual(overall["n"], 2)
        self.assertAlmostEqual(overall["mean_frames"]["h8"], 160.0)
        self.assertAlmostEqual(overall["mean_frames"]["deep"], 165.0)
        self.assertAlmostEqual(overall["deep_vs_h8"], 5.0)
        self.assertEqual(overall["deep_better"], 2)
        self.assertAlmostEqual(
            summary["per_seed"][100]["mean_deep"], 165.0
        )


if __name__ == "__main__":
    unittest.main()
