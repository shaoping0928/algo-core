"""Tests for the continuous-state -> realized-stage resolver."""
import unittest

import numpy as np

from stage_experience import StageExperienceBatch
from stage_state_resolver import StageStateResolver


OBS_TOKEN = (("observation_change", 0), 1)
FEEDBACK_TOKEN = (("feedback_change", 0), 1)


class FakeAgent:
    """Prototype reader with deterministic buckets for the test stream."""

    def __init__(self, lag=2, obs_dim=2, birth_threshold=1.0):
        self.lag = int(lag)
        self.obs_dim = int(obs_dim)
        self.birth_threshold = float(birth_threshold)
        self.scales = np.ones(self.obs_dim + self.lag)

    def _branch_active_features(self, obs):
        return ()

    def _nearest(self, state):
        value = float(state[0])
        if value > 100.0:
            return "p_low", 10.0
        if value < 5.0:
            return "p_low", 0.0
        if value < 15.0:
            return "p_mid", 0.0
        return "p_high", 0.0


class NeverSeenAgent(FakeAgent):
    def _nearest(self, state):
        return "p_new", 0.0


def make_episode(episode, ys, actions, feedback, done=False, lag=2):
    rows = []
    for step in range(len(actions)):
        issued = 1 - int(actions[step])
        history = [float(issued), 0.0, float(actions[step])]
        last = step == len(actions) - 1
        rows.append(dict(
            episode=int(episode),
            obs=np.asarray([ys[step], 0.0], dtype=float),
            next_obs=np.asarray([ys[step + 1], 0.0], dtype=float),
            history=np.asarray(history[:lag + 1], dtype=float),
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
    return episode_zero + episode_one


class StageStateResolverTests(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.rows = training_rows()
        self.experience = StageExperienceBatch.fit(
            self.rows, self.agent, lag=2, quantile=0.6,
        )
        self.resolver = StageStateResolver.fit_rows(
            self.rows, self.agent, experience=self.experience,
            min_support=1, min_episodes=1,
            min_probability=0.5, min_margin=0.0,
        )

    def test_resolves_a_supported_state_to_its_realized_stage(self):
        result = self.resolver.resolve(self.agent, [0.0, 0.0], [0, 0])
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["key"], ())
        self.assertEqual(result["prototype"], "p_low")
        self.assertEqual(result["episode_support"], 2)
        self.assertFalse(result["affects_value"])
        self.assertEqual(
            result["provenance"], "realized_state_stage_cooccurrence"
        )

    def test_episode_support_gate_returns_unknown(self):
        strict = StageStateResolver.fit_rows(
            self.rows, self.agent, experience=self.experience,
            min_support=1, min_episodes=2,
            min_probability=0.5, min_margin=0.0,
        )
        # p_mid only occurs in episode 100.
        result = strict.resolve(self.agent, [6.0, 0.0], [0, 0])
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["key"])
        self.assertEqual(result["episode_support"], 1)
        self.assertEqual(result["reason"], "insufficient_support")

    def test_distribution_and_margin_gate_the_resolution(self):
        result = self.resolver.resolve(self.agent, [16.0, 0.0], [0, 0])
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["key"], (OBS_TOKEN,))
        self.assertAlmostEqual(result["probability"], 0.8)
        self.assertAlmostEqual(result["margin"], 0.6)
        self.assertEqual(len(result["distribution"]), 2)

        ambiguous = StageStateResolver.fit_rows(
            self.rows, self.agent, experience=self.experience,
            min_support=1, min_episodes=1,
            min_probability=0.5, min_margin=0.7,
        )
        result = ambiguous.resolve(self.agent, [16.0, 0.0], [0, 0])
        self.assertEqual(result["status"], "unknown")
        self.assertAlmostEqual(result["probability"], 0.8)
        self.assertEqual(len(result["distribution"]), 2)
        self.assertEqual(result["reason"], "ambiguous")

    def test_state_outside_the_runtime_radius_is_unknown(self):
        result = self.resolver.resolve(self.agent, [200.0, 0.0], [0, 0])
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["prototype"], "p_low")
        self.assertGreater(result["distance"], self.resolver.radius)
        self.assertEqual(result["reason"], "outside_radius")

    def test_unseen_prototype_is_unknown(self):
        result = self.resolver.resolve(
            NeverSeenAgent(), [0.0, 0.0], [0, 0],
        )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["prototype"], "p_new")
        self.assertEqual(result["support"], 0)
        self.assertEqual(result["reason"], "unseen_prototype")

    def test_querying_never_updates_the_fitted_association(self):
        before = self.resolver.snapshot()
        self.resolver.resolve(self.agent, [0.0, 0.0], [0, 0])
        self.resolver.resolve(self.agent, [16.0, 0.0], [0, 0])
        self.assertEqual(self.resolver.snapshot(), before)

    def test_held_out_episodes_can_be_resolved_without_refitting(self):
        heldout = make_episode(
            300,
            ys=[0.0, 0.5, 20.5, 20.5],
            actions=[1, 0, 0],
            feedback=[np.zeros(2), np.zeros(2), np.zeros(2)],
            done=True,
        )
        before = self.resolver.snapshot()
        batch = self.experience.transform(heldout, self.agent)
        transitions = list(batch.stage_transitions(heldout, self.agent))
        self.assertEqual(len(transitions), 3)
        results = self.resolver.resolve_experience(
            heldout, self.agent, batch,
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(self.resolver.snapshot(), before)

    def test_pending_must_match_the_fitted_lag(self):
        with self.assertRaises(ValueError):
            self.resolver.resolve(self.agent, [0.0, 0.0], [0])

    def test_stage_alignment_covers_every_realized_transition(self):
        transitions = list(
            self.experience.stage_transitions(self.rows, self.agent)
        )
        self.assertEqual(len(transitions), len(self.rows))
        self.assertEqual(
            [row.stage.key for row in transitions],
            [()] * 3 + [(OBS_TOKEN,), (OBS_TOKEN,)]
            + [(FEEDBACK_TOKEN,)]
            + [()] * 2 + [(OBS_TOKEN,), (OBS_TOKEN,)],
        )
        self.assertTrue(all(
            len(row.pending) == 2 for row in transitions
        ))


if __name__ == "__main__":
    unittest.main()
