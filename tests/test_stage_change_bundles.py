"""Tests for learned stage-change bundles over static feature expansions."""
import unittest

import numpy as np

from stage_change_bundles import (
    StageChangeBundles,
    StaticExpansion,
    static_expansion,
    static_expansion_from_snapshots,
)


def expansion(frames, nodes=("a", "b", "c")):
    return StaticExpansion(
        node_ids=tuple(nodes),
        strengths=np.asarray(frames, dtype=float),
        weights=np.ones(len(nodes), dtype=float),
    )


class StaticExpansionTests(unittest.TestCase):
    def test_changes_emit_signed_tokens_above_threshold(self):
        table = expansion([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        changes = table.changes(0.5)
        self.assertEqual(set(changes[0]), {("a", 1), ("b", 1)})
        self.assertEqual(
            set(changes[1]), {("a", -1), ("b", -1), ("c", 1)},
        )

    def test_static_expansion_reads_the_runtime_masks(self):
        class StubAgent:
            _feature_all = ["a", "b"]

            def _feature_mask_table(self, observations):
                return [
                    observations[:, 0] > 0.5,
                    observations[:, 1] > 0.5,
                ]

        observations = np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        table = static_expansion(StubAgent(), observations)
        self.assertEqual(table.node_ids, ("a", "b"))
        np.testing.assert_array_equal(
            table.strengths,
            np.asarray([[0, 0], [1, 0], [1, 1]], dtype=float),
        )

    def test_snapshot_expansion_is_binary(self):
        table = static_expansion_from_snapshots(
            ("a", "b", "c"), [("a", "b"), (), ("c",)],
        )
        np.testing.assert_array_equal(
            table.strengths,
            np.asarray([[1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=float),
        )

    def test_header_validation(self):
        with self.assertRaises(ValueError):
            StaticExpansion(("a",), np.zeros((2, 2)), np.ones(1))
        with self.assertRaises(ValueError):
            StaticExpansion(("a",), np.zeros((2, 1)), np.ones(2))


class StageChangeBundleTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            expansion([
                [0, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
                [1, 1, 0],
                [0, 0, 0],
            ]),
            expansion([
                [0, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ]),
            expansion([
                [0, 0, 0],
                [1, 1, 0],
                [0, 0, 0],
            ]),
        ]

    def test_recurring_co_changes_become_bundles(self):
        model = StageChangeBundles.fit(
            self.episodes, min_support=2, min_bundle_size=2,
            min_lift=0.5,
        )
        learned = {frozenset(bundle.tokens) for bundle in model.bundles}
        self.assertIn(
            frozenset({("a", 1), ("b", 1)}), learned,
        )
        self.assertIn(
            frozenset({("a", -1), ("b", -1)}), learned,
        )
        self.assertFalse(any(
            any(node == "c" for node, _sign in bundle.tokens)
            and any(sign > 0 for node, sign in bundle.tokens
                    if node in ("a", "b"))
            for bundle in model.bundles
        ))
        for bundle in model.bundles:
            self.assertEqual(bundle.lift, 1.0)
            self.assertGreaterEqual(bundle.episode_support, 2)

    def test_key_weights_prefer_rarer_nodes(self):
        model = StageChangeBundles.fit(
            self.episodes, min_support=2, min_bundle_size=2,
            min_lift=0.5,
        )
        self.assertGreater(
            model.key_weights["c"], model.key_weights["a"],
        )

    def test_boundaries_and_intervals(self):
        model = StageChangeBundles.fit(
            self.episodes, min_support=2, min_bundle_size=2,
            min_lift=0.5,
        )
        boundaries = model.boundaries(self.episodes[0])
        self.assertEqual([item.frame for item in boundaries], [1, 2, 4, 5])
        segments = model.segments(self.episodes[0])
        self.assertEqual(
            [(item.start, item.end) for item in segments],
            [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)],
        )
        self.assertEqual(segments[0].duration, 1)
        self.assertEqual(len(segments[0].profile), 3)

    def test_independent_changes_do_not_form_a_bundle(self):
        episodes = [
            expansion([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
            expansion([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        ]
        # a+ and b+ never share a transition once the first episode is split.
        episodes[0] = expansion([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        model = StageChangeBundles.fit(
            episodes, min_support=1, min_bundle_size=2, min_lift=0.5,
        )
        self.assertFalse(any(
            {node for node, _sign in bundle.tokens} >= {"a", "b"}
            for bundle in model.bundles
        ))

    def test_min_lift_filters_weak_co_occurrence(self):
        # b+ fires in four episodes, but a+ joins only three of them.
        episodes = [
            expansion([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
            expansion([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
            expansion([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
            expansion([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            expansion([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        ]
        loose = StageChangeBundles.fit(
            episodes, min_support=2, min_bundle_size=2, min_lift=0.5,
        )
        strict = StageChangeBundles.fit(
            episodes, min_support=2, min_bundle_size=2, min_lift=1.0,
        )
        self.assertTrue(any(
            frozenset(bundle.tokens) == frozenset({("a", 1), ("b", 1)})
            for bundle in loose.bundles
        ))
        self.assertFalse(any(
            frozenset(bundle.tokens) == frozenset({("a", 1), ("b", 1)})
            for bundle in strict.bundles
        ))

    def test_fit_is_deterministic(self):
        first = StageChangeBundles.fit(
            self.episodes, min_support=2, min_bundle_size=2, min_lift=0.5,
        )
        second = StageChangeBundles.fit(
            self.episodes, min_support=2, min_bundle_size=2, min_lift=0.5,
        )
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(
            [bundle.tokens for bundle in first.bundles],
            [bundle.tokens for bundle in second.bundles],
        )


if __name__ == "__main__":
    unittest.main()
