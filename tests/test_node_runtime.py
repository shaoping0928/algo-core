import unittest
from pathlib import Path

import numpy as np

from node_runtime import NodeRuntime
from substrate_nodes import run_seed


class NodeRuntimeTests(unittest.TestCase):
    def make_runtime(self, **overrides):
        options = dict(
            obs_dim=1, lag=0, scales=np.asarray([1.0]),
            birth_threshold=0.01, max_nodes=20, exploration=0.0,
            alpha_v=0.5, alpha_q=0.5, gamma=0.9, lam=0.9,
        )
        options.update(overrides)
        return NodeRuntime(**options)

    def test_action_edges_select_the_lower_pain_action(self):
        runtime = self.make_runtime()
        source = runtime._create_node(np.asarray([0.0]))
        safe = runtime._create_node(np.asarray([1.0]))
        danger = runtime._create_node(np.asarray([2.0]))
        safe_edge = runtime.graph.add_action_edge(source, 0, safe)
        safe_edge.count = 10
        safe_edge.q_pain = 0.1
        safe_edge.q_dopamine = 1.0
        danger_edge = runtime.graph.add_action_edge(source, 1, danger)
        danger_edge.count = 10
        danger_edge.q_pain = 0.9
        danger_edge.q_dopamine = 1.0

        runtime.reset()
        action = runtime.act(np.asarray([0.0]), explore=False)
        self.assertEqual(action, 0)

    def test_pain_credit_reaches_earlier_nodes(self):
        runtime = self.make_runtime(max_nodes=40)
        for _episode in range(60):
            runtime.reset()
            observation = np.asarray([0.0])
            for step in range(3):
                action = runtime.act(observation, explore=False)
                next_observation = np.asarray([float(step + 1)])
                pain = 1.0 if step == 2 else 0.0
                runtime.observe(
                    next_observation, action,
                    [1.0 - pain, pain], done=(step == 2),
                )
                observation = next_observation

        values = [
            runtime.node_value(np.asarray([float(state)]))
            for state in range(4)
        ]
        self.assertGreater(values[0], 0.2)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertGreater(values[3], 0.9)

    def test_episode_returns_reach_earlier_nodes(self):
        runtime = self.make_runtime(max_nodes=40)
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0, 0.0, 0.0, 0.0]),
                signals=np.asarray([1.0, 0.0]), done=False,
                episode=0,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0, 0.0, 0.0, 0.0]),
                signals=np.asarray([1.0, 0.0]), done=False,
                episode=0,
            ),
            dict(
                obs=np.asarray([2.0]), next_obs=np.asarray([3.0]),
                history=np.asarray([0.0, 0.0, 0.0, 0.0]),
                signals=np.asarray([0.0, 1.0]), done=True,
                episode=0,
            ),
        ]
        for _ in range(60):
            runtime.learn_episode(rows)

        values = [
            runtime.node_value(np.asarray([float(state)]))
            for state in range(4)
        ]
        self.assertGreater(values[0], 0.2)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertGreater(values[3], 0.9)

    def test_novel_observations_birth_nodes(self):
        runtime = self.make_runtime(birth_threshold=0.1)
        runtime.reset()
        runtime.act(np.asarray([0.0]), explore=False)
        runtime.observe(
            np.asarray([5.0]), 0, [1.0, 0.0], done=False,
        )
        self.assertGreaterEqual(runtime.snapshot()["nodes"], 2)
        self.assertGreaterEqual(runtime.snapshot()["action_edges"], 1)

    def test_runtime_does_not_use_terminal_as_a_risk_label(self):
        runtime = self.make_runtime()
        runtime.reset()
        runtime.act(np.asarray([0.0]), explore=False)
        # `done` only truncates bootstrapping; pain is still supplied by the
        # valence channel, so a zero-pain terminal step must not raise V.
        runtime.observe(
            np.asarray([0.0]), 0, [0.0, 0.0], done=True,
        )
        self.assertAlmostEqual(runtime.node_value(np.asarray([0.0])), 0.0)

    def test_smoke_run_on_flappy_signals(self):
        row = run_seed((100, 100, 0.1, 100, 200, 0.45, 6, 1, 3))
        self.assertGreater(row["frames"], 0)
        self.assertGreaterEqual(row["graph"]["nodes"], 1)
        self.assertGreaterEqual(row["graph"]["action_edges"], 1)

    def test_core_does_not_embed_task_geometry(self):
        root = Path(__file__).resolve().parents[1]
        forbidden = (
            "BIRD_X", "GAP", "pipes", "target_gap", "near_pipe",
            "crossing", "TerminalRiskHead", "latent_world", "flappy",
        )
        for name in ("node_runtime.py", "node_graph.py"):
            text = (root / name).read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, "%s in %s" % (token, name))
        adapter = (root / "substrate_nodes.py").read_text(encoding="utf-8")
        for token in (
            "BIRD_X", "GAP", "pipes", "target_gap", "near_pipe",
            "crossing", "TerminalRiskHead",
        ):
            self.assertNotIn(token, adapter, "%s in substrate_nodes.py" % token)


if __name__ == "__main__":
    unittest.main()
