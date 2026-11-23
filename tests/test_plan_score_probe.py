"""Tests for the B0.5 plan/suffix score diagnostic."""

import unittest
from types import SimpleNamespace

import numpy as np

from plan_score_probe import ablation_cost, diagnose, summarize


class PlanScoreProbeTest(unittest.TestCase):
    def test_ablation_cost_drops_or_keeps_one_column(self):
        scores = np.asarray([[1.0, 0.0, 0.4, 0.3, 0.2, 0.1]])
        self.assertAlmostEqual(
            ablation_cost(scores, ("drop", 2))[0], 0.6
        )
        self.assertAlmostEqual(
            ablation_cost(scores, ("drop", 3, 4))[0], 0.5
        )
        self.assertAlmostEqual(
            ablation_cost(scores, ("only", 2))[0], 0.4
        )

    def test_diagnose_decomposes_suffix_regret(self):
        sequences = np.asarray([
            [0, 0], [0, 1], [1, 0], [1, 1],
        ], dtype=int)
        scores = np.asarray([
            [0.3, 0.0, 0.10, 0.10, 0.05, 0.05],
            [0.5, 0.0, 0.30, 0.10, 0.05, 0.05],
            [0.6, 0.0, 0.30, 0.20, 0.05, 0.05],
            [0.4, 0.0, 0.20, 0.10, 0.05, 0.05],
        ], dtype=float)
        record = dict(
            eval_seed=120, point=64, sequences=sequences, scores=scores,
            feasible=np.ones(4, dtype=bool), true_value=np.asarray(
                [2, 4, 3, 4], dtype=int
            ),
            best_value=4, selected_action=0, oracle_action=1,
            oracle_index=3,
            trace=dict(finite=np.ones(4, dtype=bool)),
            result=SimpleNamespace(actions=sequences[0]),
        )
        row = dict(
            selected_value=2, oracle_best_value=4, regret=2,
            **{"class": "suffix_within_action"},
        )
        result = diagnose(record, row)
        self.assertEqual(result["action_regret"], 0)
        self.assertEqual(result["suffix_regret"], 2)
        self.assertEqual(result["same_best_value"], 4)
        self.assertEqual(result["selected_index"], 0)
        self.assertEqual(result["same_best_index"], 1)
        self.assertGreater(result["same_best_score_gap"], 0.0)
        self.assertEqual(result["same_best_rank"], 1)
        self.assertGreater(
            result["columns"]["risk"]["same_best_minus_selected"], 0.0
        )

    def test_diagnose_decomposes_action_regret(self):
        sequences = np.asarray([
            [0, 0], [1, 0], [1, 1],
        ], dtype=int)
        scores = np.asarray([
            [0.2, 0.0, 0.10, 0.05, 0.03, 0.02],
            [0.3, 0.0, 0.10, 0.10, 0.05, 0.05],
            [0.4, 0.0, 0.20, 0.10, 0.05, 0.05],
        ], dtype=float)
        record = dict(
            eval_seed=120, point=67, sequences=sequences, scores=scores,
            feasible=np.ones(3, dtype=bool), true_value=np.asarray(
                [3, 5, 9], dtype=int
            ),
            best_value=9, selected_action=0, oracle_action=1,
            oracle_index=2,
            trace=dict(finite=np.ones(3, dtype=bool)),
            result=SimpleNamespace(actions=sequences[0]),
        )
        row = dict(
            selected_value=3, oracle_best_value=9, regret=6,
            **{"class": "coverage_support"},
        )
        result = diagnose(record, row)
        self.assertEqual(result["action_regret"], 6)
        self.assertEqual(result["suffix_regret"], 0)

    def test_summarize_counts_plan_and_action_level(self):
        plan = dict(
            action_regret=0, suffix_regret=2, same_best_rank=1,
            global_fixes=["drop_value"], same_action_fixes=["drop_value"],
            columns={
                name: dict(
                    same_best_minus_selected=0.1,
                    oracle_minus_selected=0.1,
                ) for name in ("risk", "value", "curiosity", "q")
            },
            positive_score_shares={
                name: 0.25 for name in ("risk", "value", "curiosity", "q")
            },
        )
        action = dict(
            action_regret=6, suffix_regret=0, same_best_rank=0,
            global_fixes=["drop_value"], same_action_fixes=[],
            columns={
                name: dict(
                    same_best_minus_selected=0.0,
                    oracle_minus_selected=0.1,
                ) for name in ("risk", "value", "curiosity", "q")
            },
            positive_score_shares={
                name: 0.25 for name in ("risk", "value", "curiosity", "q")
            },
        )
        summary = summarize([plan, action])
        self.assertEqual(summary["misses"], 2)
        self.assertEqual(summary["plan_level"], 1)
        self.assertEqual(summary["action_level"], 1)
        self.assertEqual(summary["plan_level_same_action_any_fix"], 1)
        self.assertEqual(summary["global_fix_counts"]["drop_value"], 2)
        self.assertEqual(summary["verdict"], "partial_single_column")
        self.assertEqual(
            summary["active_columns"],
            ["risk", "value", "curiosity", "q"],
        )


if __name__ == "__main__":
    unittest.main()
