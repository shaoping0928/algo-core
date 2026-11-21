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

    def test_death_callback_vetoes_dead_plans_and_keeps_latest_death(self):
        def evaluate_step(states, _actions, _step):
            # Prefer the largest state; without the death callback this
            # would choose the plan that reaches 2.
            return -states[:, 0], np.ones(len(states), dtype=bool)

        def death_by_state(states, _sequences, step):
            return np.where(states[:, 0] >= 1.5, step, -1)

        result = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_death=death_by_state,
            batch_size=2,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.first_death, -1)
        self.assertLess(result.final_state[0], 1.5)

        def death_by_first_action(_states, sequences, step):
            # The first action decides whether death happens at step 0
            # or step 1; the latest death must win.
            return np.where(sequences[:, 0] == 1, 0, 1)

        doomed = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_death=death_by_first_action,
            batch_size=2,
        )
        self.assertFalse(doomed.feasible)
        self.assertEqual(doomed.first_death, 1)
        self.assertEqual(doomed.actions[0], 0.0)

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

    def test_max_combine_penalizes_the_worst_step(self):
        def evaluate_step(_states, sequences, step):
            raised = sequences[:, 0] > 0.5
            if step == 0:
                return np.where(raised, 5.0, 0.0), np.ones(len(sequences), dtype=bool)
            return np.where(raised, 5.0, 10.0), np.ones(len(sequences), dtype=bool)

        kwargs = dict(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
        )
        summed = exhaustive_search(**kwargs)
        maxed = exhaustive_search(**kwargs, score_combine="max")
        self.assertEqual(int(summed.actions[0]), 0)
        self.assertEqual(int(maxed.actions[0]), 1)
        self.assertEqual(maxed.score, 5.0)

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

    def test_death_status_tracks_unknown_and_safe_steps(self):
        def evaluate_step(states, _actions, _step):
            return np.zeros(len(states)), np.ones(len(states), dtype=bool)

        def status(_states, sequences, step):
            # Action 0 is safe then unknown; action 1 is unknown then dead.
            first = sequences[:, 0]
            if step == 0:
                return np.where(first == 0, 0, -1)
            return np.where(first == 0, -1, 1)

        result = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_death_status=status,
            batch_size=4,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.first_death, -1)
        self.assertEqual(result.first_unknown, 1)
        self.assertEqual(result.unknown_count, 1)
        self.assertEqual(result.safe_count, 1)

    def test_unknown_penalty_prefers_fewer_unknown_steps(self):
        def evaluate_step(_states, sequences, _step):
            score = np.where(sequences[:, 0] == 1, -1.0, 0.0)
            return score, np.ones(len(score), dtype=bool)

        def status(_states, sequences, _step):
            return np.where(sequences[:, 0] == 1, -1, 0)

        kwargs = dict(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=1,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_death_status=status,
        )
        unpenalised = exhaustive_search(**kwargs)
        self.assertEqual(unpenalised.actions[0], 1.0)
        self.assertEqual(unpenalised.unknown_count, 1)

        mild = exhaustive_search(**kwargs, unknown_penalty=0.5)
        self.assertEqual(mild.actions[0], 1.0)
        self.assertEqual(mild.score, -1.0)
        self.assertEqual(mild.unknown_count, 1)

        strict = exhaustive_search(**kwargs, unknown_penalty=2.0)
        self.assertEqual(strict.actions[0], 0.0)
        self.assertEqual(strict.score, 0.0)
        self.assertEqual(strict.unknown_count, 0)
        self.assertEqual(strict.safe_count, 1)

    def test_candidate_callback_sees_all_candidates_and_keeps_selection(self):
        def evaluate_step(states, _actions, _step):
            return np.abs(states[:, 0] - 1.0), np.ones(len(states), dtype=bool)

        kwargs = dict(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            batch_size=4,
        )
        batches = []
        traced = exhaustive_search(
            **kwargs, candidate_callback=batches.append
        )
        plain = exhaustive_search(**kwargs)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(len(batch["sequences"]), 4)
        self.assertEqual(batch["first_death"].shape, (4,))
        self.assertEqual(batch["first_unknown"].shape, (4,))
        self.assertEqual(batch["unknown_count"].shape, (4,))
        self.assertEqual(batch["safe_count"].shape, (4,))
        np.testing.assert_allclose(traced.actions, plain.actions)
        np.testing.assert_allclose(traced.final_state, plain.final_state)
        self.assertEqual(traced.score, plain.score)

    def test_status_callback_matches_legacy_death_callback(self):
        def evaluate_step(states, _actions, _step):
            return -states[:, 0], np.ones(len(states), dtype=bool)

        def death_by_state(states, _sequences, step):
            return np.where(states[:, 0] >= 1.5, step, -1)

        def status_by_state(states, _sequences, _step):
            return np.where(states[:, 0] >= 1.5, 1, -1)

        kwargs = dict(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            batch_size=4,
        )
        legacy = exhaustive_search(**kwargs, evaluate_death=death_by_state)
        status = exhaustive_search(
            **kwargs, evaluate_death_status=status_by_state
        )
        np.testing.assert_allclose(status.actions, legacy.actions)
        np.testing.assert_allclose(status.final_state, legacy.final_state)
        self.assertEqual(status.feasible, legacy.feasible)
        self.assertEqual(status.feasible_count, legacy.feasible_count)
        self.assertEqual(status.first_death, legacy.first_death)
        self.assertEqual(status.score, legacy.score)


if __name__ == "__main__":
    unittest.main()
