"""Regression tests for the stage-graph segmenter (slice 1)."""
import unittest

import numpy as np

from stage_graph import (
    _kmeans,
    build_stage_graph,
    evaluate_stream,
    shuffle_actions,
    sweep_vocabulary,
)


def make_rows(episodes=6, length=20, switch=None, noise=0.0, seed=7):
    rng = np.random.RandomState(seed)
    rows = []
    for episode in range(episodes):
        obs = np.zeros(2)
        for step in range(length):
            action = step % 2
            if switch is None:
                delta = np.asarray([0.1, 0.02])
            else:
                sign = 1.0 if step < switch else -1.0
                delta = np.asarray([sign * action, 0.0])
            delta = delta + noise * rng.normal(size=2)
            next_obs = obs + delta
            rows.append(dict(
                obs=obs.copy(),
                next_obs=next_obs.copy(),
                history=np.asarray([0.0, 0.0, 0.0, float(action)]),
                signals=np.asarray([1.0, 0.0]),
                done=step == length - 1,
                episode=episode,
            ))
            obs = next_obs
    return rows


class StageGraphTest(unittest.TestCase):
    def test_constant_stream_stays_single_stage(self):
        rows = make_rows(switch=None, noise=0.0)
        graph = build_stage_graph(rows)
        # One stage per episode, no cross-episode informative transitions.
        self.assertEqual(len(graph.stages), 6)
        self.assertEqual(graph.cross_episode_informative(), 0)

    def test_action_effect_switch_creates_a_boundary(self):
        rows = make_rows(switch=10)
        graph = build_stage_graph(rows)
        self.assertGreater(len(graph.stages), 6)
        boundaries = sorted({
            stage.end for stage in graph.stages
            if stage.end < 20
        })
        self.assertTrue(
            any(abs(boundary - 10) <= 2 for boundary in boundaries),
            "expected a boundary near the action switch, got %s"
            % boundaries,
        )
        self.assertGreater(graph.cross_episode_informative(), 0)

    def test_shuffle_control_does_not_add_structure(self):
        rows = make_rows(switch=10)
        real = build_stage_graph(rows)
        shuffled = build_stage_graph(shuffle_actions(rows))
        self.assertLessEqual(
            shuffled.cross_episode_informative(),
            real.cross_episode_informative(),
        )

    def test_gate_abstains_on_unpredictable_stream(self):
        rng = np.random.RandomState(3)
        rows = []
        for episode in range(8):
            for step in range(20):
                action = step % 2
                obs = rng.normal(size=3)
                rows.append(dict(
                    obs=obs,
                    next_obs=obs + rng.normal(size=3),
                    history=np.asarray(
                        [0.0, 0.0, 0.0, float(action)]
                    ),
                    signals=np.asarray([1.0, 0.0]),
                    done=step == 19,
                    episode=episode,
                ))
        graph = build_stage_graph(rows)
        self.assertLess(graph.gate_r2, 0.0)
        # The gate turns segmentation off: one stage per episode.
        self.assertEqual(len(graph.stages), 8)

    def test_evaluate_stream_reports_both_arms(self):
        rows = make_rows(switch=10)
        result = evaluate_stream(rows)
        self.assertIn("real", result)
        self.assertIn("shuffle_actions", result)
        self.assertGreater(
            result["real"]["stages"], result["shuffle_actions"]["stages"] - 1
        )

    def test_kmeans_separates_two_action_effects(self):
        vectors = np.asarray([[1.0, 0.0]] * 5 + [[-1.0, 0.0]] * 5)
        labels = _kmeans(vectors, 2)
        self.assertEqual(len(set(labels[:5])), 1)
        self.assertEqual(len(set(labels[5:])), 1)
        self.assertNotEqual(labels[0], labels[5])

    def test_sweep_vocabulary_reports_each_k(self):
        rows = make_rows(switch=10)
        result = sweep_vocabulary(rows, [2, 3])
        self.assertEqual([item["k"] for item in result], [2, 3])
        for item in result:
            self.assertIn("real_cross3", item)
            self.assertIn("shuffle_cross3", item)
        # K=2 recovers the two-regime structure on the real stream.
        self.assertGreater(result[0]["real_cross3"], 0)


if __name__ == "__main__":
    unittest.main()
