"""Tests for the predictive-equivalence stage tree."""
import unittest

import numpy as np

from predictive_stage_tree import PredictiveStageTree, UNKNOWN
from stage_outcome_memory import StageOccurrence, encode_action_trace


def stage(episode, key, duration, feedback, terminal=False, censored=False):
    return StageOccurrence(
        episode=episode,
        key=(key,),
        entry_state=np.zeros(2),
        exit_state=np.ones(2),
        action_trace=encode_action_trace([0]),
        duration=duration,
        feedback=np.asarray(feedback, dtype=float),
        terminal=terminal,
        censored=censored,
    )


def episodes(prefix, count=2):
    """A/B share outcomes; C has a distinct successor, duration and feedback."""
    result = []
    episode_id = prefix * 100
    for _ in range(count):
        result.append([
            stage(episode_id, "A", 2.0, (1.0, 0.0)),
            stage(episode_id, "D", 3.0, (0.0, 0.0), terminal=True),
        ])
        episode_id += 1
        result.append([
            stage(episode_id, "A", 4.0, (1.0, 0.0)),
            stage(episode_id, "E", 3.0, (0.0, 0.0), terminal=True),
        ])
        episode_id += 1
        result.append([
            stage(episode_id, "B", 2.0, (1.0, 0.0)),
            stage(episode_id, "D", 3.0, (0.0, 0.0), terminal=True),
        ])
        episode_id += 1
        result.append([
            stage(episode_id, "B", 4.0, (1.0, 0.0)),
            stage(episode_id, "E", 3.0, (0.0, 0.0), terminal=True),
        ])
        episode_id += 1
        result.append([
            stage(episode_id, "C", 10.0, (0.0, 5.0)),
            stage(episode_id, "F", 3.0, (0.0, 0.0), terminal=True),
        ])
        episode_id += 1
    return result


class PredictiveStageTreeTests(unittest.TestCase):
    def setUp(self):
        self.tree = PredictiveStageTree.fit(
            episodes(1), episodes(2),
            prior=1.0, tolerance=0.0, min_validation_support=2,
            min_support=1, min_episodes=1,
        )

    def test_predictively_equivalent_types_merge_and_distinct_type_stays(self):
        self.assertGreaterEqual(len(self.tree.merge_log), 1)
        chain_a = self.tree.chain(("A",))
        chain_b = self.tree.chain(("B",))
        chain_c = self.tree.chain(("C",))
        self.assertGreater(len(chain_a), 1)
        self.assertGreater(len(chain_b), 1)
        self.assertEqual(chain_a[0], self.tree.leaf_of[("A",)])
        self.assertNotEqual(chain_a[0], chain_b[0])
        self.assertEqual(chain_a[-1], chain_b[-1])
        # C is not predictively equivalent to A/B in this stream.
        self.assertNotIn(chain_a[-1], chain_c)
        self.assertTrue(all(
            np.isfinite(merge["validation_delta"])
            for merge in self.tree.merge_log
        ))

    def test_prediction_falls_back_to_the_merged_parent(self):
        prediction = self.tree.predict(
            ("A",), min_support=5, min_episodes=1,
        )
        self.assertEqual(prediction["status"], "supported")
        self.assertGreaterEqual(prediction["level"], 1)
        # The parent pools A and B occurrences.
        self.assertGreaterEqual(prediction["support"], 8)
        c_prediction = self.tree.predict(
            ("C",), min_support=1, min_episodes=1,
        )
        self.assertEqual(c_prediction["level"], 0)
        self.assertEqual(c_prediction["support"], 2)

    def test_support_gate_walks_to_the_global_base_profile(self):
        held_out_leaf = self.tree.predict(
            ("A",), min_support=999, min_episodes=999,
        )
        self.assertEqual(held_out_leaf["status"], "supported")
        self.assertEqual(held_out_leaf["level"], "base")
        self.assertTrue(held_out_leaf["known_stage"])

    def test_unknown_stage_uses_the_base_profile_without_a_leaf(self):
        prediction = self.tree.predict(("ZZZ",))
        self.assertEqual(prediction["status"], "supported")
        self.assertEqual(prediction["level"], "base")
        self.assertFalse(prediction["known_stage"])

    def test_large_tolerance_merges_everything(self):
        tree = PredictiveStageTree.fit(
            episodes(3), episodes(4),
            prior=1.0, tolerance=100.0, min_validation_support=1,
            min_support=1, min_episodes=1,
        )
        self.assertEqual(tree.snapshot()["roots"], 1)
        self.assertEqual(tree.chain(("A",))[-1], tree.chain(("C",))[-1])

    def test_querying_does_not_mutate_the_fitted_tree(self):
        before = self.tree.snapshot()
        self.tree.predict(("A",))
        self.tree.predict(("C",))
        self.tree.chain(("B",))
        self.assertEqual(self.tree.snapshot(), before)

    def test_held_out_joint_log_likelihood_is_finite(self):
        value = self.tree.log_likelihood(
            stage(9, "A", 3.0, (1.0, 0.0)),
            (("D",), False, False),
        )
        self.assertIsNotNone(value)
        self.assertTrue(np.isfinite(value))

    def test_unsupported_tree_returns_unknown(self):
        tree = PredictiveStageTree.fit(
            [], [], min_support=2, min_episodes=2,
        )
        prediction = tree.predict(("A",))
        self.assertEqual(prediction["status"], UNKNOWN)
        self.assertFalse(prediction["known_stage"])


if __name__ == "__main__":
    unittest.main()
