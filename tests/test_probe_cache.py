"""Fitted-model cache for probe arms."""
import os
import tempfile
import unittest

from probe_cache import load_or_fit


class ProbeCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "cache.pkl")

    def tearDown(self):
        self.directory.cleanup()

    def test_matching_config_reuses_the_fit(self):
        calls = []

        def fit_fn(**config):
            calls.append(config)
            return ("world", config["seed"])

        first = load_or_fit(
            {"seed": 100}, self.path, fit_fn=fit_fn
        )
        second = load_or_fit(
            {"seed": 100}, self.path, fit_fn=fit_fn
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(first[2])
        self.assertTrue(second[2])
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_mismatched_config_refits(self):
        calls = []

        def fit_fn(**config):
            calls.append(config)
            return ("world", config["seed"])

        load_or_fit({"seed": 100}, self.path, fit_fn=fit_fn)
        _world, _agent, loaded = load_or_fit(
            {"seed": 101}, self.path, fit_fn=fit_fn
        )
        self.assertEqual(len(calls), 2)
        self.assertFalse(loaded)

    def test_refit_forces_a_new_fit(self):
        calls = []

        def fit_fn(**config):
            calls.append(config)
            return ("world", config["seed"])

        load_or_fit({"seed": 100}, self.path, fit_fn=fit_fn)
        load_or_fit(
            {"seed": 100}, self.path, refit=True, fit_fn=fit_fn
        )
        self.assertEqual(len(calls), 2)

    def test_corrupt_cache_falls_back_to_fit(self):
        calls = []

        def fit_fn(**config):
            calls.append(config)
            return ("world", config["seed"])

        with open(self.path, "wb") as handle:
            handle.write(b"not a pickle")
        _world, _agent, loaded = load_or_fit(
            {"seed": 100}, self.path, fit_fn=fit_fn
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(loaded)

    def test_no_path_always_fits(self):
        calls = []

        def fit_fn(**config):
            calls.append(config)
            return ("world", config["seed"])

        load_or_fit({"seed": 100}, None, fit_fn=fit_fn)
        load_or_fit({"seed": 100}, None, fit_fn=fit_fn)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
