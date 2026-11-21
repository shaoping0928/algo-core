"""The generic context layer: shared dynamics plus a fast latent regime."""
import unittest

import numpy as np

from context_model import ContextualLinearModel


def _world_data(z, basis, base, count, rng, noise=0.01):
    obs = rng.normal(0.0, 1.0, (count, 2))
    actions = rng.randint(0, 2, (count, 1)).astype(float)
    features = np.concatenate(
        [obs, actions, np.ones((count, 1))], axis=1
    )
    delta = np.einsum("k,kdf->df", z, basis)
    target = features @ base + features @ delta.T
    target = target + rng.normal(0.0, noise, target.shape)
    return obs, actions, target


class ContextModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        self.basis = rng.normal(0.0, 0.25, (2, 2, 4))
        self.base = rng.normal(0.0, 0.3, (4, 2))

    def _training_stream(self, rng):
        observations = []
        actions = []
        targets = []
        worlds = []
        for index in range(6):
            angle = 2.0 * np.pi * index / 6.0
            z = np.asarray([np.cos(angle), np.sin(angle)])
            obs, act, target = _world_data(
                z, self.basis, self.base, 120, rng
            )
            observations.append(obs)
            actions.append(act)
            targets.append(target)
            worlds.extend([index] * len(obs))
        return (
            np.concatenate(observations), np.concatenate(actions),
            np.concatenate(targets), np.asarray(worlds),
        )

    def _make_model(self, change_detection=True):
        model = ContextualLinearModel(
            obs_dim=2, action_dim=1, context_dim=2,
            context_ridge=1e-2, window=32,
            change_detection=change_detection, change_factor=4.0,
            discovery="segment",
        )
        rng = np.random.RandomState(11)
        obs, act, target, worlds = self._training_stream(rng)
        model.fit(obs, act, target, segments=worlds)
        return model

    def test_context_adapts_to_an_unseen_regime(self):
        rng = np.random.RandomState(19)
        held = np.asarray([0.65, -0.55])
        obs, act, target = _world_data(
            held, self.basis, self.base, 80, rng
        )
        model = self._make_model()
        self.assertIsNotNone(model.basis)
        global_error = []
        context_error = []
        for step in range(len(obs)):
            prediction = model.predict(obs[step:step + 1], act[step:step + 1])
            global_prediction = (
                model._features(obs[step:step + 1], act[step:step + 1])
                @ model.weights
            )
            global_error.append(float(np.linalg.norm(
                target[step] - global_prediction[0]
            )))
            context_error.append(float(np.linalg.norm(
                target[step] - prediction[0]
            )))
            model.observe(obs[step], act[step], target[step])
        # The first few steps are the shared prior; by the tail the fast
        # context has identified the unseen regime.
        self.assertLess(
            float(np.mean(context_error[-20:])),
            0.5 * float(np.mean(global_error[-20:])),
        )

    def test_change_detection_flags_a_mid_stream_switch(self):
        rng = np.random.RandomState(23)
        first = np.asarray([0.9, 0.1])
        second = np.asarray([-0.7, 0.7])
        obs_a, act_a, target_a = _world_data(
            first, self.basis, self.base, 60, rng
        )
        obs_b, act_b, target_b = _world_data(
            second, self.basis, self.base, 60, rng
        )
        model = self._make_model(change_detection=True)
        for step in range(len(obs_a)):
            model.observe(obs_a[step], act_a[step], target_a[step])
        detected_before = model.changes
        errors = []
        for step in range(len(obs_b)):
            prediction = model.predict(obs_b[step:step + 1], act_b[step:step + 1])
            errors.append(float(np.linalg.norm(target_b[step] - prediction[0])))
            model.observe(obs_b[step], act_b[step], target_b[step])
        # The monitor must attribute the persistent residual to a regime
        # change rather than smoothing it into the stale context.
        self.assertGreater(model.changes, detected_before)
        # And the fast channel must recover inside the new regime without
        # a full retrain: the tail is much closer than the switch window.
        self.assertLess(
            float(np.mean(errors[-15:])),
            float(np.mean(errors[:5])),
        )

    def test_without_change_detection_no_switch_is_declared(self):
        model = self._make_model(change_detection=False)
        rng = np.random.RandomState(29)
        obs, act, target = _world_data(
            np.asarray([0.2, 0.8]), self.basis, self.base, 50, rng
        )
        for step in range(len(obs)):
            model.observe(obs[step], act[step], target[step])
        self.assertEqual(model.changes, 0)

    def test_basis_is_discovered_without_any_group_labels(self):
        rng = np.random.RandomState(31)
        model = ContextualLinearModel(
            obs_dim=2, action_dim=1, context_dim=3,
            context_ridge=1e-2, window=32, segment_size=48,
        )
        obs, act, target, _worlds = self._training_stream(rng)
        model.fit(obs, act, target)
        self.assertIsNotNone(model.basis)
        self.assertGreaterEqual(model.basis.shape[0], 1)
        self.assertLessEqual(model.basis.shape[0], 3)

    def test_apply_prior_keeps_the_observed_warmup_window(self):
        rng = np.random.RandomState(43)
        obs, act, target = _world_data(
            np.asarray([0.4, -0.3]), self.basis, self.base, 8, rng
        )
        model = self._make_model()
        model.begin_episode()
        for step in range(5):
            model.observe(obs[step], act[step], target[step])
        buffered = len(model._buffer_x)
        prior = np.asarray([0.5, -0.5])
        model.apply_prior(prior, precision=1.0)
        self.assertEqual(len(model._buffer_x), buffered)
        np.testing.assert_allclose(model.context, prior)
        model.observe(obs[5], act[5], target[5])
        self.assertGreater(model.updates, 5)


if __name__ == "__main__":
    unittest.main()
