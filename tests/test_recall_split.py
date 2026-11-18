import os
from pathlib import Path
import subprocess
import sys
import unittest


class RecallSplitTests(unittest.TestCase):
    def test_cli_flag_adds_and_runs_split_arm(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("OMNI_ARMS", None)
        env.pop("OMNI_TYPES", None)
        env["OMNI_SEED"] = "21"
        result = subprocess.run(
            [sys.executable, "recall_world.py", "30", "64", "1"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all+split", result.stdout)
        self.assertIn("splits 3", result.stdout)


if __name__ == "__main__":
    unittest.main()
