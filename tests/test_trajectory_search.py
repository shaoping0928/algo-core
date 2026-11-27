import unittest

import numpy as np

from trajectory_search import (
    SearchResult,
    beam_search,
    best_candidate_index,
    chunked_beam_search,
    exhaustive_search,
    merge_candidate_batches,
    override_first_action,
)


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

    def test_chunk_size_one_matches_frame_beam_search(self):
        def evaluate_step(states, sequences, step):
            score = np.abs(states[:, 0] - 2.0)
            return score, np.ones(len(sequences), dtype=bool)

        def evaluate_terminal(states, _sequences):
            score = np.abs(states[:, 0] - 3.0)
            return score, np.ones(len(states), dtype=bool)

        kwargs = dict(
            initial_state=np.asarray([0.0]),
            action_space=(0, 1),
            horizon=5,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_terminal=evaluate_terminal,
            beam_width=3,
        )
        frame = beam_search(**kwargs)
        chunked = chunked_beam_search(**kwargs, chunk_size=1)
        np.testing.assert_array_equal(chunked.actions, frame.actions)
        np.testing.assert_array_equal(chunked.final_state, frame.final_state)
        self.assertEqual(chunked.score, frame.score)
        self.assertEqual(chunked.candidate_count, frame.candidate_count)
        self.assertEqual(chunked.feasible_count, frame.feasible_count)

    def test_chunked_beam_expands_three_frames_then_prunes(self):
        def evaluate_step(states, _sequences, _step):
            return np.zeros(len(states)), np.ones(len(states), dtype=bool)

        trace = []
        result = chunked_beam_search(
            initial_state=np.asarray([0.0]),
            action_space=(0, 1),
            horizon=6,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            beam_width=2,
            chunk_size=3,
            frontier_callback=trace.append,
        )

        self.assertEqual(len(trace), 2)
        self.assertEqual(
            [
                (row["chunk_start"], row["chunk_end"], row["step"])
                for row in trace
            ],
            [(0, 3, 2), (3, 6, 5)],
        )
        self.assertEqual(
            [row["chunk_candidate_count"] for row in trace], [14, 28],
        )
        self.assertEqual(result.candidate_count, 42)
        self.assertTrue(all(
            int(row["retained"].sum()) == 2 for row in trace
        ))

    def test_chunked_rank_credit_stays_separate_from_objective_score(self):
        def evaluate_step(states, _sequences, _step):
            return np.zeros(len(states)), np.ones(len(states), dtype=bool)

        def evaluate_terminal(states, _sequences):
            return np.abs(states[:, 0] - 1.0), np.ones(
                len(states), dtype=bool
            )

        result = chunked_beam_search(
            initial_state=np.asarray([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_terminal=evaluate_terminal,
            beam_width=2,
            chunk_size=2,
            rank_step=lambda states, _sequences, _step: states[:, 0],
        )

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.rank_priority, 2.0)
        np.testing.assert_array_equal(result.actions, [1.0, 0.0])

    def test_chunked_group_quota_preserves_proposals_without_changing_objective(self):
        def evaluate_step(states, _sequences, _step):
            return states[:, 0].copy(), np.ones(
                len(states), dtype=bool,
            )

        def rank_step(_states, sequences, _step):
            groups = tuple(
                "left" if float(actions[0]) == 0.0 else "right"
                for actions in sequences
            )
            priorities = np.asarray([
                10.0 if group == "left" else 0.0
                for group in groups
            ])
            return priorities, groups

        trace = []
        result = chunked_beam_search(
            initial_state=np.asarray([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            beam_width=2,
            chunk_size=2,
            rank_step=rank_step,
            rank_group_quota=1,
            frontier_callback=trace.append,
        )

        final_frontier = trace[-1]
        retained_groups = {
            final_frontier["rank_groups"][index]
            for index in np.flatnonzero(final_frontier["retained"])
        }
        self.assertEqual(retained_groups, {"left", "right"})
        self.assertEqual(result.score, 0.0)
        np.testing.assert_array_equal(result.actions, [0.0, 0.0])
        self.assertEqual(result.candidate_count, 6)

    def test_tie_mode_group_quota_does_not_displace_clear_objective_winner(self):
        def evaluate_step(states, _sequences, _step):
            return states[:, 0].copy(), np.ones(
                len(states), dtype=bool,
            )

        def rank_step(states, _sequences, _step):
            groups = tuple(
                "lower-objective" if state[0] == 0.0 else "higher-objective"
                for state in states
            )
            priorities = np.where(states[:, 0] == 0.0, 0.0, 100.0)
            return priorities, groups

        result = chunked_beam_search(
            initial_state=np.asarray([0.0]),
            action_space=(0, 1),
            horizon=1,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            beam_width=1,
            chunk_size=1,
            rank_step=rank_step,
            rank_mode="tie",
            rank_margin=0.0,
            rank_group_quota=1,
        )

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.final_state[0], 0.0)
        np.testing.assert_array_equal(result.actions, [0.0])

    def test_frontier_callback_is_observational_only(self):
        def evaluate_step(states, _actions, _step):
            return np.abs(states[:, 0] - 2.0), np.ones(
                len(states), dtype=bool
            )

        kwargs = dict(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=3,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            beam_width=2,
            rank_step=lambda states, _sequences, _step: states[:, 0],
        )
        baseline = beam_search(**kwargs)
        trace = []
        observed = beam_search(**kwargs, frontier_callback=trace.append)
        np.testing.assert_array_equal(observed.actions, baseline.actions)
        np.testing.assert_array_equal(
            observed.final_state, baseline.final_state
        )
        self.assertEqual(observed.score, baseline.score)
        self.assertEqual(observed.candidate_count, baseline.candidate_count)
        self.assertEqual(len(trace), kwargs["horizon"])
        for step, batch in enumerate(trace):
            self.assertEqual(batch["step"], step)
            self.assertEqual(
                batch["states"].shape[0],
                batch["sequences"].shape[0],
            )
            self.assertEqual(
                batch["priorities"].shape,
                (batch["states"].shape[0],),
            )
            self.assertEqual(
                int(batch["retained"].sum()),
                len(batch["retained_indices"]),
            )
            np.testing.assert_array_equal(
                np.flatnonzero(batch["retained"]),
                np.sort(batch["retained_indices"]),
            )

    def test_beam_rank_hint_orders_frontier_but_not_final_value(self):
        def evaluate_step(states, _sequences, _step):
            return np.zeros(len(states)), np.ones(len(states), dtype=bool)

        def evaluate_terminal(states, _sequences):
            return np.abs(states[:, 0] - 1.0), np.ones(
                len(states), dtype=bool
            )

        kwargs = dict(
            initial_state=np.asarray([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_terminal=evaluate_terminal,
            beam_width=2,
        )
        baseline = beam_search(**kwargs)
        guided = beam_search(
            **kwargs,
            rank_step=lambda states, _sequences, _step: states[:, 0],
        )
        # The hint explores the high-state route first, but the final choice
        # among retained paths is still made by the true terminal objective.
        np.testing.assert_array_equal(guided.actions, [1.0, 0.0])
        self.assertEqual(guided.final_state[0], 1.0)
        self.assertEqual(guided.score, 0.0)
        self.assertEqual(baseline.score, guided.score)
        self.assertFalse(np.array_equal(baseline.actions, guided.actions))

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

    def test_beam_rank_tie_mode_keeps_a_clearly_better_score(self):
        def evaluate_step(states, sequences, _step):
            # Action 0 has a much worse objective than action 1.
            score = np.where(sequences[:, 0] > 0.5, 0.0, 10.0)
            return score, np.ones(len(states), dtype=bool)

        def evaluate_terminal(states, _sequences):
            return np.zeros(len(states)), np.ones(len(states), dtype=bool)

        kwargs = dict(
            initial_state=np.asarray([0.0]),
            action_space=(0, 1),
            horizon=1,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_terminal=evaluate_terminal,
            beam_width=1,
        )
        hint = lambda _states, sequences, _step: 1.0 - sequences[:, 0]
        ordered = beam_search(**kwargs, rank_step=hint, rank_mode="order")
        tied = beam_search(**kwargs, rank_step=hint, rank_mode="tie")
        self.assertEqual(int(ordered.actions[0]), 0)
        self.assertEqual(int(tied.actions[0]), 1)
        self.assertEqual(float(ordered.score), 10.0)
        self.assertEqual(float(tied.score), 0.0)

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

        batches = []
        result = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_death_status=status,
            candidate_callback=batches.append,
            batch_size=4,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.first_death, -1)
        self.assertEqual(result.first_unknown, 1)
        self.assertEqual(result.unknown_count, 1)
        self.assertEqual(result.safe_count, 1)
        batch = batches[0]
        dead = batch["sequences"][:, 0] == 1
        self.assertTrue(batch["model_feasible"][dead].all())
        self.assertFalse(batch["feasible"][dead].any())

    def test_near_status_is_soft_and_tracked(self):
        def evaluate_step(states, _actions, _step):
            return np.zeros(len(states)), np.ones(len(states), dtype=bool)

        def status(_states, sequences, step):
            first = sequences[:, 0]
            if step == 0:
                return np.where(first == 0, 2, -1)
            return np.where(first == 0, 0, 1)

        batches = []
        result = exhaustive_search(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=2,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_death_status=status,
            candidate_callback=batches.append,
            batch_size=4,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.first_death, -1)
        self.assertEqual(result.first_near, 0)
        self.assertEqual(result.near_count, 1)
        self.assertEqual(result.safe_count, 1)
        batch = batches[0]
        dead = batch["sequences"][:, 0] == 1
        self.assertTrue(batch["model_feasible"][dead].all())
        self.assertFalse(batch["feasible"][dead].any())
        self.assertTrue(batch["near_count"][~dead].all())

    def test_near_penalty_breaks_ties_without_vetoing(self):
        def evaluate_step(states, _sequences, _step):
            return (
                np.zeros(len(states)),
                np.ones(len(states), dtype=bool),
            )

        def status(_states, sequences, _step):
            return np.where(sequences[:, 0] == 0, 2, 0)

        kwargs = dict(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=1,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            evaluate_death_status=status,
        )
        unpenned = exhaustive_search(**kwargs)
        self.assertEqual(unpenned.actions[0], 0.0)
        self.assertTrue(unpenned.feasible)
        penalised = exhaustive_search(**kwargs, near_penalty=1.0)
        self.assertEqual(penalised.actions[0], 1.0)
        self.assertTrue(penalised.feasible)
        self.assertEqual(penalised.near_count, 0)
        with self.assertRaises(ValueError):
            exhaustive_search(
                initial_state=kwargs["initial_state"],
                action_space=kwargs["action_space"],
                horizon=1,
                transition_batch=self.add_action,
                evaluate_step=lambda states, sequences, step: (
                    np.zeros(len(states)),
                    np.ones(len(states), dtype=bool),
                ),
                near_penalty=1.0,
            )

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
        self.assertEqual(batch["model_feasible"].shape, (4,))
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

    @staticmethod
    def candidate_batch():
        count = 4
        return dict(
            sequences=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]],
                                 dtype=float),
            scores=np.asarray([[1.0], [2.0], [3.0], [4.0]]),
            ranking_scores=np.asarray([[1.0], [2.0], [3.0], [4.0]]),
            feasible=np.asarray([True, False, True, True]),
            finite=np.ones(count, dtype=bool),
            first_death=np.asarray([-1, -1, -1, -1]),
            first_unknown=np.zeros(count, dtype=int),
            unknown_count=np.full(count, 2, dtype=int),
            safe_count=np.zeros(count, dtype=int),
            first_near=np.full(count, -1, dtype=int),
            near_count=np.zeros(count, dtype=int),
            first_states=np.arange(8, dtype=float).reshape(4, 2),
            final_states=np.arange(8, dtype=float).reshape(4, 2) + 100.0,
        )

    def test_offline_baseline_matches_the_no_death_search(self):
        def evaluate_step(states, sequences, step):
            score = np.abs(states[:, 0] - 1.0) + 0.1 * sequences[:, 0]
            return score, states[:, 0] < 3.0

        def death_by_state(states, _sequences, step):
            return np.where(states[:, 0] >= 1.5, step, -1)

        kwargs = dict(
            initial_state=np.array([0.0]),
            action_space=(0, 1),
            horizon=3,
            transition_batch=self.add_action,
            evaluate_step=evaluate_step,
            batch_size=8,
        )
        baseline = exhaustive_search(**kwargs)
        batches = []
        _death_aware = exhaustive_search(
            **kwargs, evaluate_death=death_by_state,
            candidate_callback=batches.append,
        )
        merged = merge_candidate_batches(batches)
        index = best_candidate_index(
            merged["scores"], merged["model_feasible"]
        )
        np.testing.assert_allclose(
            merged["sequences"][index], baseline.actions
        )
        self.assertAlmostEqual(
            float(merged["scores"][index][0]),
            float(baseline.score),
        )

    def test_best_candidate_index_matches_the_search_selection(self):
        scores = np.asarray([[1.0, 5.0], [1.0, 4.0], [1.0, 9.0]])
        feasible = np.asarray([True, True, True])
        self.assertEqual(best_candidate_index(scores, feasible), 1)
        self.assertEqual(
            best_candidate_index(scores, np.asarray([True, False, False])),
            0,
        )
        self.assertEqual(
            best_candidate_index(scores, np.zeros(3, dtype=bool)), 1
        )
        with_nan = np.asarray([[np.nan], [2.0], [1.0]])
        self.assertEqual(
            best_candidate_index(with_nan, np.ones(3, dtype=bool)), 2
        )

    def test_merge_candidate_batches_concatenates_every_field(self):
        merged = merge_candidate_batches([
            self.candidate_batch(), self.candidate_batch()
        ])
        self.assertEqual(len(merged["sequences"]), 8)
        self.assertEqual(merged["first_death"].shape, (8,))
        self.assertEqual(merged["final_states"].shape, (8, 2))

    def test_override_first_action_picks_best_feasible_suffix(self):
        result = SearchResult(
            actions=np.asarray([0.0, 0.0]), score=1.0, feasible=True,
            feasible_count=3, candidate_count=4,
            first_state=np.zeros(1), final_state=np.zeros(1),
        )
        overridden = override_first_action(
            result, [self.candidate_batch()], 1
        )
        np.testing.assert_array_equal(overridden.actions, [1.0, 0.0])
        self.assertEqual(overridden.score, 3.0)
        self.assertTrue(overridden.feasible)
        self.assertEqual(overridden.feasible_count, 3)
        self.assertEqual(overridden.candidate_count, 4)
        self.assertEqual(overridden.unknown_count, 2)

    def test_override_first_action_falls_back_when_nothing_is_feasible(self):
        result = SearchResult(
            actions=np.asarray([1.0, 0.0]), score=3.0, feasible=True,
            feasible_count=3, candidate_count=4,
            first_state=np.zeros(1), final_state=np.zeros(1),
        )
        batch = self.candidate_batch()
        batch["feasible"] = np.asarray([True, True, False, False])
        unchanged = override_first_action(result, [batch], 1)
        self.assertIs(unchanged, result)
        missing = override_first_action(result, [batch], 7)
        self.assertIs(missing, result)


if __name__ == "__main__":
    unittest.main()
