"""Tests for the automatic self-supervised stage-experience pipeline."""
import unittest

import numpy as np

from stage_experience import StageExperienceBatch
from stage_outcome_memory import StageOutcomeMemory


OBS_TOKEN = (("observation_change", 0), 1)
FEEDBACK_TOKEN = (("feedback_change", 0), 1)


class FakeAgent:
    """Minimal internal-feature reader; mirrors the runtime's read-only call."""

    def __init__(self, lag=2):
        self.lag = int(lag)
        self.scales = np.ones(2)

    def _branch_active_features(self, obs):
        return ("f_high",) if float(obs[0]) > 5.0 else ("f_low",)


def make_episode(episode, ys, actions, feedback, done=False, lag=2,
                 acting_lag=None):
    """Rows in the recorded schema: history is newest-first, history[lag] acts."""
    acting_lag = int(lag if acting_lag is None else acting_lag)
    rows = []
    for step in range(len(actions)):
        issued = 1 - int(actions[step])
        history = [float(issued)] + [0.0] * (acting_lag - 1)
        history += [float(actions[step])]
        last = step == len(actions) - 1
        rows.append(dict(
            episode=int(episode),
            obs=np.asarray([ys[step], 0.0], dtype=float),
            next_obs=np.asarray([ys[step + 1], 0.0], dtype=float),
            history=np.asarray(history, dtype=float),
            signals=np.asarray(feedback[step], dtype=float),
            terminal=bool(done and last),
            done=bool(done and last),
        ))
    return rows


def training_rows():
    zero = np.zeros(2)
    spike = np.asarray([5.0, 0.0])
    episode_zero = make_episode(
        100,
        ys=[0.0, 6.0, 6.0, 16.0, 16.0, 16.0, 16.0],
        actions=[0, 1, 1, 0, 1, 0],
        feedback=[zero, zero, zero, zero, spike, zero],
        done=False,
    )
    episode_one = make_episode(
        101,
        ys=[0.0, 0.5, 20.5, 20.5, 20.5],
        actions=[1, 0, 0, 1],
        feedback=[zero, zero, zero, zero],
        done=True,
    )
    # No history[lag] slot: the episode must be dropped, never attributed to
    # the newly issued action.
    episode_two = make_episode(
        102, ys=[0.0, 1.0, 2.0], actions=[0, 1],
        feedback=[zero, zero], acting_lag=1,
    )
    # Strip the acting slot entirely, keeping a short history.
    for row in episode_two:
        row["history"] = row["history"][:2]
    return episode_zero + episode_one + episode_two


class StageExperienceTests(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent(lag=2)
        self.batch = StageExperienceBatch.fit(
            training_rows(), self.agent, lag=2, quantile=0.6,
        )

    def test_pipeline_segments_real_episodes_and_reports_census(self):
        census = self.batch.census
        self.assertEqual(census["episodes"], 2)
        self.assertEqual(census["episodes_skipped"], 1)
        self.assertEqual(census["rows"], 10)
        self.assertEqual(census["rows_skipped"], 2)
        self.assertEqual(census["stages"], 5)
        self.assertEqual(census["transitions"], 3)
        self.assertEqual(census["stage_types"], 3)
        self.assertEqual(census["keyed_stages"], 3)
        self.assertEqual(census["empty_key_stages"], 2)
        self.assertEqual(census["demand_samples"], 7)
        self.assertEqual(census["terminated_episodes"], 1)
        self.assertEqual(census["censored_episodes"], 1)
        self.assertEqual(census["feature_ids"], 2)
        self.assertAlmostEqual(census["feature_coverage"], 1.0)
        self.assertGreater(census["signal_channels"], 0)

    def test_stage_boundaries_come_from_generic_change_channels(self):
        first, second = self.batch.episodes
        self.assertEqual(
            [stage.duration for stage in first], [3.0, 2.0, 1.0],
        )
        self.assertEqual(
            [stage.key for stage in first],
            [(), (OBS_TOKEN,), (FEEDBACK_TOKEN,)],
        )
        self.assertEqual(
            [stage.duration for stage in second], [2.0, 2.0],
        )
        self.assertEqual(
            [stage.key for stage in second], [(), (OBS_TOKEN,)],
        )
        self.assertTrue(first[-1].censored)
        self.assertTrue(second[-1].terminal)

    def test_actions_are_attributed_to_the_acting_slot_not_the_issued_one(self):
        first = self.batch.episodes[0]
        self.assertEqual(first[0].action_trace, ((0, 1), (1, 2)))
        self.assertEqual(first[1].action_trace, ((0, 1), (1, 1)))
        self.assertEqual(first[2].action_trace, ((0, 1),))

    def test_demand_samples_pair_each_transition_snapshot_with_next_stage(self):
        samples = self.batch.demand_samples
        self.assertEqual(len(samples), 7)
        destinations = {sample.destination_stage for sample in samples}
        self.assertEqual(destinations, {(OBS_TOKEN,), (FEEDBACK_TOKEN,)})
        first_stage_features = {
            sample.active_features
            for sample in samples
            if sample.stage_history == ((),)
            and sample.episode == 100
        }
        # Entry-only snapshots would see just f_low; the per-transition
        # snapshots must also carry f_high from inside the stage.
        self.assertEqual(first_stage_features, {("f_low",), ("f_high",)})

    def test_transform_reuses_the_frozen_detector(self):
        thresholds = dict(self.batch.detector.thresholds)
        heldout = make_episode(
            300,
            ys=[0.0, 30.0, 30.0, 30.0],
            actions=[1, 1, 0],
            feedback=[
                np.zeros(2), np.asarray([9.0, 0.0]), np.zeros(2),
            ],
            done=True,
        )
        heldout_batch = self.batch.transform(heldout, self.agent)
        self.assertEqual(self.batch.detector.thresholds, thresholds)
        self.assertIs(heldout_batch.detector, self.batch.detector)
        self.assertEqual(heldout_batch.census["episodes"], 1)
        self.assertEqual(
            [stage.key for stage in heldout_batch.episodes[0]],
            [(), (OBS_TOKEN,), (FEEDBACK_TOKEN,)],
        )

    def test_outcome_memory_consumes_the_pipeline_output(self):
        memory = StageOutcomeMemory(min_support=1, min_episodes=1).fit(
            self.batch.outcome_episodes()
        )
        snapshot = memory.snapshot()
        self.assertEqual(snapshot["transition_rows"], 5)
        self.assertGreater(snapshot["contexts"], 1)

    def test_no_usable_episode_is_an_error(self):
        broken = make_episode(
            400, ys=[0.0, 1.0], actions=[0], feedback=[np.zeros(2)],
        )
        broken[0]["history"] = np.asarray([0.0], dtype=float)
        with self.assertRaises(ValueError):
            StageExperienceBatch.fit(broken, self.agent, lag=2)

    def test_terminal_row_before_the_end_drops_the_episode(self):
        zero = np.zeros(2)
        bad = make_episode(
            500, ys=[0.0, 1.0, 2.0], actions=[0, 1],
            feedback=[zero, zero],
        )
        bad[0]["terminal"] = True
        bad[0]["done"] = True
        batch = StageExperienceBatch.fit(
            training_rows() + bad, self.agent, lag=2, quantile=0.6,
        )
        self.assertEqual(batch.census["episodes"], 2)
        self.assertEqual(batch.census["episodes_skipped"], 2)
        self.assertEqual(batch.census["rows_skipped"], 4)

    def test_noncontiguous_episode_ids_are_dropped(self):
        zero = np.zeros(2)
        first = make_episode(
            100, ys=[0.0, 1.0, 2.0], actions=[0, 1],
            feedback=[zero, zero],
        )
        second = make_episode(
            101, ys=[0.0, 1.0, 2.0], actions=[0, 1],
            feedback=[zero, zero],
        )
        clean = make_episode(
            103, ys=[0.0, 1.0, 2.0], actions=[0, 1],
            feedback=[zero, zero],
        )
        rows = []
        for position in range(2):
            rows.append(first[position])
            rows.append(second[position])
        batch = StageExperienceBatch.fit(
            rows + clean, self.agent, lag=2, quantile=0.6,
        )
        self.assertEqual(batch.census["episodes"], 1)
        self.assertEqual(batch.census["episodes_skipped"], 2)
        self.assertEqual(batch.census["rows_skipped"], 4)

    def test_mixed_observation_dimensions_are_dropped(self):
        zero = np.zeros(2)
        bad = make_episode(
            500, ys=[0.0, 1.0, 2.0], actions=[0, 1],
            feedback=[zero, zero],
        )
        bad[1]["obs"] = np.asarray([1.0, 0.0, 0.0])
        clean = make_episode(
            103, ys=[0.0, 1.0, 2.0], actions=[0, 1],
            feedback=[zero, zero],
        )
        batch = StageExperienceBatch.fit(
            bad + clean, self.agent, lag=2, quantile=0.6,
        )
        self.assertEqual(batch.census["episodes"], 1)
        self.assertEqual(batch.census["episodes_skipped"], 1)
        self.assertEqual(batch.census["rows_skipped"], 2)


if __name__ == "__main__":
    unittest.main()
