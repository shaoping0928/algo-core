import os
from pathlib import Path
import subprocess
import sys
import unittest

from flappy import Flappy, run, zero_policy


class FlappyTests(unittest.TestCase):
    def test_run_respects_frame_cap(self):
        self.assertEqual(run(zero_policy, seed=0, cap=2), 2)

    def test_environment_uses_its_instance_cap(self):
        env = Flappy(seed=0, max_t=1)
        _, _, done = env.step(0)
        self.assertTrue(done)

    def test_non_positive_cap_is_rejected(self):
        with self.assertRaises(ValueError):
            Flappy(seed=0, max_t=0)

    def test_import_does_not_parse_cli_args_or_run_experiment(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['python', '--unrelated-flag']; import flappy",
            ],
            cwd=root,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_latent_runner_respects_its_frame_cap(self):
        root = Path(__file__).resolve().parents[1]
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
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
