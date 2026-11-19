import unittest

from eval_protocol import (
    bootstrap_mean_ci,
    cluster_summary,
    group_summaries,
    paired_delta,
    parse_seed_spec,
    summarize,
)


class SeedSpecTests(unittest.TestCase):
    def test_ranges_and_lists_are_merged_and_sorted(self):
        self.assertEqual(parse_seed_spec("120-122,121,130"), [120, 121, 122, 130])
        self.assertEqual(parse_seed_spec(""), [])
        self.assertEqual(parse_seed_spec([3, 1, 3]), [1, 3])

    def test_bad_specs_are_rejected(self):
        for spec in ("122-120", "1,,2", "a-b", "-3"):
            with self.assertRaises(ValueError):
                parse_seed_spec(spec)


class SummaryTests(unittest.TestCase):
    def test_summary_keeps_the_sample_and_iqr(self):
        summary = summarize([10, 20, 30, 40])
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["mean"], 25.0)
        self.assertAlmostEqual(summary["median"], 25.0)
        self.assertAlmostEqual(summary["q1"], 17.5)
        self.assertAlmostEqual(summary["q3"], 32.5)
        self.assertAlmostEqual(summary["iqr"], 15.0)

    def test_bootstrap_interval_is_degenerate_for_constants(self):
        low, high = bootstrap_mean_ci([7.0, 7.0, 7.0])
        self.assertEqual((low, high), (7.0, 7.0))

    def test_interval_brackets_the_sample_mean(self):
        low, high = bootstrap_mean_ci([1, 2, 3, 4, 5], samples=200)
        self.assertLessEqual(low, 3.0)
        self.assertGreaterEqual(high, 3.0)

    def test_bootstrap_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            bootstrap_mean_ci([1, 2], samples=0)
        with self.assertRaises(ValueError):
            bootstrap_mean_ci([1, 2], confidence=1.0)

    def test_cluster_summary_resamples_models_not_episodes(self):
        rows = []
        for model, frames in ((100, [10, 10, 10]), (101, [20, 20, 20])):
            for episode, value in enumerate(frames):
                rows.append(dict(
                    train_seed=model, eval_seed=120 + episode, frames=value
                ))
        summary = cluster_summary(rows, cluster="train_seed")
        self.assertEqual(summary["n"], 6)
        self.assertEqual(summary["clusters"], 2)
        self.assertAlmostEqual(summary["mean"], 15.0)
        self.assertAlmostEqual(summary["cluster_mean"], 15.0)

    def test_group_summaries_use_string_keys(self):
        rows = [
            dict(train_seed=100, frames=10),
            dict(train_seed=100, frames=20),
            dict(train_seed=101, frames=30),
        ]
        summary = group_summaries(rows, group="train_seed")
        self.assertEqual(set(summary), {"100", "101"})
        self.assertAlmostEqual(summary["100"]["mean"], 15.0)

    def test_paired_delta_averages_per_model_not_per_episode(self):
        left = [
            dict(train_seed=100, frames=10),
            dict(train_seed=100, frames=20),
            dict(train_seed=101, frames=30),
        ]
        right = [
            dict(train_seed=100, frames=30),
            dict(train_seed=101, frames=40),
        ]
        result = paired_delta(left, right)
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["deltas"], [-15.0, -10.0])
        self.assertAlmostEqual(result["mean"], -12.5)


if __name__ == "__main__":
    unittest.main()
