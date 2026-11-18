import unittest

import numpy as np

from trajectory_search import beam_search, exhaustive_search


class TrajectorySearchTests(unittest.TestCase):
    @staticmethod
    def add_action(states, sequences, step):
        return states + np.asarray(sequences[:, step], dtype=float).reshape(-1, 1)

    def test_rolls_out_and_selects_minimum_heuristic(self):
        def evaluate_step(states, _actions, _step):
            zeros = np.zeros(len(states))
            return zeros, np.ones(len(states), dtype=bool)

        def evaluate_terminal(states, _sequences):
            score = np.abs(states[:, 0] - 2.0)
            return score, np.ones(len(score), dtype=bool)

        result = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=3,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_terminal=evaluate_terminal,
            batch_size=2,
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.candidate_count, 8)
        self.assertEqual(result.feasible_count, 8)
        self.assertEqual(int(result.actions.sum()), 2)
        self.assertEqual(result.final_state[0], 2.0)

    def test_prefers_feasible_candidate_then_reports_best_effort(self):
        def evaluate_step(states, _actions, _step):
            scores = states[:, 0]
            feasible = states[:, 0] == 0
            return scores, feasible

        result = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=1,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.actions[0], 0.0)

        def no_feasible(states, _actions, _step):
            return states[:, 0], np.zeros(len(states), dtype=bool)

        fallback = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=1,
            transition_batch=self.add_action,
            evaluate_step=no_feasible,
        )
        self.assertFalse(fallback.feasible)
        self.assertEqual(fallback.actions[0], 0.0)

    def test_multiobjective_scores_prioritize_first_objective(self):
        def evaluate_step(_states, sequences, _step):
            action = sequences[:, 0]
            # Action 1 wins the primary objective even though its secondary score is worse.
            scores = np.column_stack(
                [np.where(action == 1, -1.0, 0.0),
                 np.where(action == 1, 100.0, -100.0)]
            )
            return scores, np.ones(len(action), dtype=bool)

        result = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=1,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
        )
        self.assertEqual(result.actions[0], 1.0)
        self.assertEqual(result.score, (-1.0, 100.0))

    def test_beam_search_finds_the_same_short_horizon_plan(self):
        def evaluate_step(states, _actions, _step):
            return np.abs(states[:, 0] - 2.0), np.ones(len(states), dtype=bool)

        result = beam_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=4,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            beam_width=3,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(int(result.actions.sum()), 2)
        self.assertEqual(result.final_state[0], 2.0)
        self.assertGreater(result.candidate_count, 0)

    def test_beam_search_matches_exhaustive_when_width_is_large(self):
        def evaluate_step(states, sequences, step):
            score = np.abs(states[:, 0] - 2.0) + 0.01 * sequences[:, 0]
            return score, np.ones(len(states), dtype=bool)

        kwargs = dict(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=3,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
        )
        exhaustive = exhaustive_search(**kwargs)
        beam = beam_search(**kwargs, beam_width=8)
        np.testing.assert_allclose(beam.actions, exhaustive.actions)
        self.assertAlmostEqual(beam.final_state[0], exhaustive.final_state[0])


if __name__ == "__main__":
    unittest.main()
