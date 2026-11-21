"""The generative side of a summary: keywords prompt a new approximate flow."""
import unittest

import numpy as np

from context_model import ContextualLinearModel
from generative_expansion import (
    GenerativeExpander,
    nearest_reference_distance,
)
from hierarchical_memory import HierarchicalMemory


def _world(z, basis, base, count, rng):
    obs = rng.normal(0.0, 1.0, (count, 2))
    actions = rng.randint(0, 2, (count, 1)).astype(float)
    x = np.concatenate([obs, actions, np.ones((count, 1))], axis=1)
    delta = np.einsum("k,kdf->df", z, basis)
    target = x @ base + x @ delta.T
    return obs, actions, target


class GenerativeExpansionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(41)
        self.basis = rng.normal(0.0, 0.35, (2, 2, 4))
        self.base = rng.normal(0.0, 0.3, (4, 2))
        observations = []
        actions = []
        targets = []
        worlds = []
        for index in range(4):
            angle = np.pi * index / 2.0
            z = np.asarray([np.cos(angle), np.sin(angle)])
            obs, act, target = _world(
                z, self.basis, self.base, 80, rng
            )
            observations.append(obs)
            actions.append(act)
            targets.append(target)
            worlds.extend([index] * len(obs))
        self.model = ContextualLinearModel(
            obs_dim=2, action_dim=1, context_dim=2,
            context_ridge=1e-2, discovery="segment",
        ).fit(
            np.concatenate(observations), np.concatenate(actions),
            np.concatenate(targets), segments=np.asarray(worlds),
        )

    def test_context_prompt_changes_the_generated_flow(self):
        expander = GenerativeExpander(self.model, horizon=12)
        policy = lambda obs, history: 0
        first = expander.expand(
            np.asarray([1.0, 0.0]), start_obs=np.zeros(2),
            policy=policy,
        )
        second = expander.expand(
            np.asarray([-1.0, 0.0]), start_obs=np.zeros(2),
            policy=policy,
        )
        self.assertGreater(len(first), 0)
        self.assertGreater(len(second), 0)
        self.assertFalse(np.allclose(
            first[-1]["prediction"], second[-1]["prediction"]
        ))
        phases = expander.summarize(first)
        self.assertGreaterEqual(len(phases), 1)
        self.assertTrue(all(
            np.isfinite(phase["observation"]).all() for phase in phases
        ))

    def test_pick_start_and_bounds_come_from_memory(self):
        memory = HierarchicalMemory(
            context_dim=2, observation_dim=2, window=1
        )
        for step in range(8):
            memory.observe(
                observation=np.asarray([step * 0.1, 0.0]),
                feedback=np.asarray([1.0, 0.0]), error=0.01,
                context=np.asarray([step * 0.1, 0.0]),
                done=False, episode=0,
            )
        self.assertTrue(memory.frames)
        expander = GenerativeExpander(
            self.model, memory=memory, horizon=5
        )
        start = expander.pick_start(np.asarray([0.3, 0.0]))
        self.assertEqual(start.shape, (2,))
        low, high = expander.bounds()
        self.assertTrue(np.all(low <= high))
        records = expander.expand(
            np.asarray([0.3, 0.0]), start_obs=start
        )
        self.assertGreater(len(records), 0)

    def test_reference_distance_prefers_the_matching_flow(self):
        reference = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        close = nearest_reference_distance(
            [[0.1, 0.0], [0.9, 0.1]], reference
        )
        far = nearest_reference_distance(
            [[5.0, 5.0], [6.0, 5.0]], reference
        )
        self.assertLess(close, far)


if __name__ == "__main__":
    unittest.main()
