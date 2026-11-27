import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

import flappy as flappy_module
from flappy import Flappy, run, zero_policy


class FlappyTests(unittest.TestCase):
    def test_run_respects_frame_cap(self):
        self.assertEqual(run(zero_policy, seed=0, cap=2), 2)

    def test_environment_uses_its_instance_cap(self):
        env = Flappy(seed=0, max_t=1)
        _, _, done = env.step(0)
        self.assertTrue(done)

    def test_world_emits_dopamine_and_pain_channels(self):
        env = Flappy(seed=0, max_t=2)
        env.step(0)
        self.assertEqual(env.feedback(), {"dopamine": 1.0, "pain": 0.0})
        env.step(0)
        self.assertEqual(env.feedback(), {"dopamine": 0.0, "pain": 1.0})

    def test_non_positive_cap_is_rejected(self):
        with self.assertRaises(ValueError):
            Flappy(seed=0, max_t=0)

    def test_spacing_parameter_stretches_the_passage_interval(self):
        env = Flappy(seed=3, spacing=1.6)
        self.assertAlmostEqual(env.spacing, 1.6)
        self.assertAlmostEqual(
            env.pipes[1][0] - env.pipes[0][0], 1.6, places=9
        )
        default = Flappy(seed=3)
        self.assertAlmostEqual(default.spacing, flappy_module.SPACING)

    def test_non_positive_spacing_is_rejected(self):
        with self.assertRaises(ValueError):
            Flappy(seed=0, spacing=0.0)
        with self.assertRaises(ValueError):
            Flappy(seed=0, spacing=-1.0)

    def test_spacing_leaves_the_first_passage_geometry_unchanged(self):
        # Same seed -> the same gap draws, so the passage the bird must fly
        # through is identical; only the distance to the pipe AFTER it grows.
        # That is exactly the claim that a passage is a time-independent
        # milestone: the objective is the same, only its duration changes.
        near = Flappy(seed=7, spacing=0.8)
        far = Flappy(seed=7, spacing=1.6)
        obs_near, obs_far = near.obs(), far.obs()
        np.testing.assert_allclose(obs_near[:4], obs_far[:4], rtol=1e-6)
        self.assertAlmostEqual(float(obs_far[4] - obs_near[4]), 0.8, places=6)

    def test_pass_counter_records_each_recycled_pipe(self):
        env = Flappy(seed=0, spacing=0.8)
        self.assertEqual(env.passed, 0)
        env.pipes[0][0] = -0.199
        env.step(0)
        self.assertEqual(env.passed, 1)
        self.assertAlmostEqual(
            env.pipes[1][0] - env.pipes[0][0], 0.8, places=9
        )
        env.pipes[0][0] = -0.199
        env.step(0)
        self.assertEqual(env.passed, 2)

    def test_import_does_not_parse_cli_args_or_run_experiment(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "core")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['python', '--unrelated-flag']; import flappy",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_latent_runner_respects_its_frame_cap(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "core")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from latent_world import make_policy, run; "
                    "r = {'G0': -0.0016, 'CVY': 1.0, 'F': 0.02}; "
                    "p = make_policy(r, 0, 12); "
                    "assert run(0, 100, p, 'exact', cap=2)[0] == 2"
                ),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
