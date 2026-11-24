"""Tests for the soft stage-type endpoint value term."""
import unittest

import numpy as np

from substrate_nodes import make_stage_value_penalty


class StubResolver:
    def resolve(self, _agent, observation, _queue):
        if float(observation[0]) > 0.5:
            return dict(status="supported", key=("a",))
        if float(observation[0]) < -0.5:
            return dict(status="unsupported", key=None)
        return dict(status="supported", key=("b",))


class StubTree:
    def chain(self, key):
        return (0, 7) if key == ("a",) else (1, 9)


class StubAgent:
    obs_dim = 1
    scales = np.asarray([1.0])
    stage_resolver = StubResolver()
    stage_tree = StubTree()
    stage_prototype_roots = {"p0": ("tree", "7")}

    @staticmethod
    def _nearest(_state):
        return "p0", 0.0


class StageValuePenaltyTests(unittest.TestCase):
    def test_penalties_follow_resolved_tree_roots(self):
        penalty = make_stage_value_penalty(
            StubAgent(), {("tree", "7"): 0.1, ("tree", "9"): 0.9},
            prefix=np.zeros(1), lag=1, neutral=0.5,
        )
        states = np.asarray([[1.0], [0.0]])
        sequences = np.asarray([[0.0, 0.0], [0.0, 1.0]])
        values = penalty(states, sequences)
        np.testing.assert_allclose(values, [0.1, 0.9])

    def test_unknown_state_is_neutral(self):
        penalty = make_stage_value_penalty(
            StubAgent(), {("tree", "7"): 0.1}, prefix=np.zeros(1),
            lag=1, neutral=0.25,
        )
        values = penalty(
            np.asarray([[-1.0]]), np.asarray([[0.0, 0.0]]),
        )
        np.testing.assert_allclose(values, [0.25])

    def test_penalty_length_is_checked(self):
        penalty = make_stage_value_penalty(
            StubAgent(), {("tree", "7"): 0.1}, prefix=np.zeros(1), lag=1,
        )
        values = penalty(
            np.asarray([[1.0], [0.0]]),
            np.asarray([[0.0, 0.0], [0.0, 1.0]]),
        )
        self.assertEqual(values.shape, (2,))


if __name__ == "__main__":
    unittest.main()
