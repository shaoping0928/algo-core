"""Generic action-effect statistics."""
import unittest

import numpy as np

from action_effect import ActionEffectLibrary


class ActionEffectTests(unittest.TestCase):
    @staticmethod
    def data():
        states = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ])
        actions = np.asarray([0, 0, 1, 1])
        next_states = states.copy()
        next_states[0, 0] += 1.0
        next_states[1, 0] += 1.0
        next_states[2, 3] -= 2.0
        next_states[3, 3] -= 2.0
        return states, actions, next_states

    def test_fit_records_signed_and_absolute_effects(self):
        states, actions, next_states = self.data()
        library = ActionEffectLibrary().fit(states, actions, next_states)
        effect0 = library.effect(0)
        effect1 = library.effect(1)
        self.assertEqual(effect0.count, 2)
        np.testing.assert_allclose(effect0.mean_delta, [1, 0, 0, 0])
        np.testing.assert_allclose(effect1.mean_delta, [0, 0, 0, -2])
        self.assertAlmostEqual(library.change(0, [0]), 1.0)
        self.assertAlmostEqual(library.change(1, [0]), 0.0)
        self.assertAlmostEqual(library.change(1, [3]), 2.0)

    def test_preferred_uses_requested_dimensions(self):
        states, actions, next_states = self.data()
        library = ActionEffectLibrary().fit(states, actions, next_states)
        self.assertEqual(library.preferred([0]), 0)
        self.assertEqual(library.preferred([3]), 1)
        self.assertEqual(library.ranked([0]), (0, 1))
        self.assertEqual(library.ranked([3]), (1, 0))
        self.assertEqual(library.preferred([0, 3], actions=[1]), 1)
        self.assertEqual(
            library.ranked([0, 3], actions=[0, 1]), (1, 0)
        )

    def test_normalized_change_uses_the_supplied_scale(self):
        states, actions, next_states = self.data()
        library = ActionEffectLibrary().fit(
            states, actions, next_states, scales=[0.5, 1.0, 1.0, 0.5]
        )
        self.assertAlmostEqual(library.normalized_change(0, [0]), 2.0)
        self.assertAlmostEqual(library.normalized_change(1, [3]), 4.0)
        self.assertEqual(library.preferred([0, 3]), 1)

    def test_unknown_action_and_empty_data_raise(self):
        states, actions, next_states = self.data()
        library = ActionEffectLibrary().fit(states, actions, next_states)
        self.assertNotIn(2, library)
        with self.assertRaises(KeyError):
            library.effect(2)
        with self.assertRaises(ValueError):
            ActionEffectLibrary().fit(
                np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2))
            )
        with self.assertRaises(RuntimeError):
            ActionEffectLibrary().preferred()

    def test_describe_lists_each_action(self):
        states, actions, next_states = self.data()
        library = ActionEffectLibrary().fit(states, actions, next_states)
        text = library.describe()
        self.assertIn("action 0", text)
        self.assertIn("action 1", text)


if __name__ == "__main__":
    unittest.main()
