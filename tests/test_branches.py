"""Role-specific branch strengths over one shared graph."""
import unittest

from branches import (
    BranchBank,
    CurrentBranch,
    PredictionBranch,
    RecentBranch,
    ValueBranch,
)


class BranchTests(unittest.TestCase):
    def test_current_branch_is_instantaneous(self):
        branch = CurrentBranch()
        branch.activate(["a"], [1.0])
        self.assertEqual(branch.read("a"), 1.0)
        branch.activate(["b"], [0.5])
        self.assertEqual(branch.read("a"), 0.0)
        self.assertEqual(branch.read("b"), 0.5)

    def test_recent_branch_decays_and_accumulates(self):
        branch = RecentBranch(decay=0.5)
        branch.activate(["a"], [1.0])
        self.assertEqual(branch.read("a"), 1.0)
        branch.step()
        self.assertAlmostEqual(branch.read("a"), 0.5)
        branch.activate(["a"], [0.25])
        self.assertAlmostEqual(branch.read("a"), 0.75)
        branch.step()
        self.assertAlmostEqual(branch.read("a"), 0.375)

    def test_prediction_error_is_per_node(self):
        prediction = PredictionBranch()
        current = CurrentBranch()
        prediction.activate(["a", "b"], [1.0, 0.0])
        current.activate(["a", "b"], [0.75, 0.25])
        comparison = prediction.compare(current)
        self.assertAlmostEqual(comparison["mean_error"], 0.25)
        self.assertAlmostEqual(comparison["errors"]["a"], 0.25)
        self.assertAlmostEqual(comparison["errors"]["b"], 0.25)
        self.assertEqual(comparison["overlap"], 2)
        self.assertGreater(comparison["cosine"], 0.0)

    def test_transition_pairs_use_recent_times_current(self):
        bank = BranchBank()
        bank.add(RecentBranch())
        bank.add(CurrentBranch())
        bank.activate("recent", ["a", "b"], [1.0, 0.5])
        bank.activate("current", ["c", "d"], [1.0, 0.2])
        pairs = bank.transition_pairs(action=1, top=4)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(pair[2] == 1 for pair in pairs))
        self.assertTrue(all(pair[3] > 0.0 for pair in pairs))
        self.assertEqual(pairs[0][:2], ("a", "c"))
        self.assertGreaterEqual(pairs[0][3], pairs[-1][3])

    def test_value_branch_moves_slowly_toward_the_target(self):
        branch = ValueBranch(learn_rate=0.5)
        branch.activate(["a"], [1.0])
        self.assertAlmostEqual(branch.read("a"), 0.5)
        branch.activate(["a"], [0.0])
        self.assertAlmostEqual(branch.read("a"), 0.25)
        self.assertEqual(branch.support["a"], 2)

    def test_bank_snapshot_and_unknown_branches(self):
        bank = BranchBank()
        bank.add(CurrentBranch())
        bank.add(RecentBranch())
        bank.activate("current", ["a"], [1.0])
        snapshot = bank.snapshot()
        self.assertIn("current", snapshot)
        self.assertIn("recent", snapshot)
        self.assertEqual(snapshot["current"]["active"], 1)
        with self.assertRaises(KeyError):
            bank.get("missing")
        with self.assertRaises(ValueError):
            bank.add(CurrentBranch())

    def test_branches_share_ids_but_not_strengths(self):
        bank = BranchBank()
        bank.add(CurrentBranch())
        bank.add(PredictionBranch())
        bank.activate("current", ["a"], [1.0])
        bank.activate("prediction", ["a"], [0.25])
        self.assertEqual(bank.get("current").read("a"), 1.0)
        self.assertEqual(bank.get("prediction").read("a"), 0.25)

    def test_runtime_records_current_and_recent_branches(self):
        import numpy as np
        from node_runtime import NodeRuntime

        runtime = NodeRuntime(
            obs_dim=1, lag=0, scales=np.asarray([1.0]),
            birth_threshold=0.01, max_nodes=20, exploration=0.0,
            alpha_v=0.5, alpha_q=0.5, gamma=0.9, lam=0.9,
        )
        first = runtime._create_node(np.asarray([0.0]))
        second = runtime._create_node(np.asarray([1.0]))
        third = runtime._create_node(np.asarray([2.0]))
        runtime._interact(first, second)
        self.assertEqual(runtime.branches.get("current").read(first), 1.0)
        self.assertAlmostEqual(
            runtime.branches.get("recent").read(second), 1.0
        )
        runtime._interact(second, third)
        self.assertEqual(runtime.branches.get("current").read(second), 1.0)
        self.assertEqual(runtime.branches.get("current").read(first), 0.0)
        self.assertLess(
            runtime.branches.get("recent").read(second), 1.0
        )
        self.assertAlmostEqual(
            runtime.branches.get("recent").read(third), 1.0
        )


if __name__ == "__main__":
    unittest.main()
