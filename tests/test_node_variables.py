"""Unit tests for task-agnostic node/state variable abstraction & class instantiation."""
from pathlib import Path
import unittest

import numpy as np

from node_variables import (
    DynamicsClassNode,
    ParameterizedState,
    variable_similarity,
)
from stage_regime import (
    OnlineRegimeContinuationUpdater,
    StageRegimeSummarizer,
    TrunkPointFailureMemory,
)
from substrate_generic import LinearObservationModel


def _synthetic_linear_world_rows(
    n_episodes=12,
    steps_per_ep=30,
    drift=-0.02,
    gain=0.10,
    seed=0,
    ep_offset=0,
    include_reset_jumps=True,
):
    """Generic 3-channel synthetic world with optional coupled periodic reset jumps.

    Ch 0: position-like integral of Ch 1
    Ch 1: velocity-like channel driven by constant drift + action gain
    Ch 2: periodic ramp with coupled reset jump every 12 steps
    """
    rng = np.random.RandomState(seed)
    rows = []
    for ep in range(n_episodes):
        s = np.asarray([0.0, 0.0, 0.8], dtype=float)
        for t in range(steps_per_ep):
            a = float(1 if (t % 3 == 0) else 0)
            hist = np.asarray([a], dtype=float)
            nxt = s.copy()
            nxt[1] = 0.98 * s[1] + drift + gain * a
            nxt[0] = s[0] + nxt[1]
            if include_reset_jumps and t > 0 and t % 12 == 0:
                nxt[2] = s[2] + 0.85  # Discontinuous reset jump
            else:
                nxt[2] = s[2] - 0.05
            rows.append(dict(
                episode=ep_offset + ep,
                obs=s.copy(),
                next_obs=nxt.copy(),
                history=hist,
                signals=np.asarray([1.0, float(t == steps_per_ep - 1)]),
                done=(t == steps_per_ep - 1),
            ))
            s = nxt
    return rows


class NodeVariablesTests(unittest.TestCase):
    def test_module_is_task_agnostic(self):
        source = (Path(__file__).resolve().parents[1] / "node_variables.py").read_text(
            encoding="utf-8"
        ).lower()
        for forbidden in ("flappy", "bird", "pipe", "gap", "flap", "gravity"):
            self.assertNotIn(forbidden, source)

    def test_dynamics_class_node_deadzone_invariant_on_unchanged_world(self):
        train_rows = _synthetic_linear_world_rows(n_episodes=10, drift=-0.02, gain=0.10, seed=1)
        obs = np.asarray([r["obs"] for r in train_rows], dtype=float)
        hist = np.asarray([r["history"] for r in train_rows], dtype=float)
        nxt = np.asarray([r["next_obs"] for r in train_rows], dtype=float)
        base = LinearObservationModel(3, 1).fit(obs, hist, nxt)

        node = DynamicsClassNode(base, train_rows)
        test_rows = _synthetic_linear_world_rows(n_episodes=2, drift=-0.02, gain=0.10, seed=99)
        for r in test_rows:
            node.observe(r["obs"], r["history"], r["next_obs"])

        # Unchanged world stays strictly inside dead-zone: zero delta, exact base predictions
        self.assertEqual(node.active_channels(), ())
        self.assertEqual(node.mean_credibility(), 0.0)
        np.testing.assert_array_equal(node.world_signature(), np.zeros(6))
        np.testing.assert_array_equal(
            node.predict(obs[:8], hist[:8]),
            base.predict(obs[:8], hist[:8]),
        )
        self.assertGreater(node.skipped_jumps, 0)

    def test_dynamics_class_node_extracts_shifted_latent_variable_and_detects_switch(self):
        train_rows = _synthetic_linear_world_rows(n_episodes=12, drift=-0.02, gain=0.10, seed=2)
        obs = np.asarray([r["obs"] for r in train_rows], dtype=float)
        hist = np.asarray([r["history"] for r in train_rows], dtype=float)
        nxt = np.asarray([r["next_obs"] for r in train_rows], dtype=float)
        base = LinearObservationModel(3, 1).fit(obs, hist, nxt)

        # Test across 5 unseen hidden drift values -> verify near-perfect linear recovery
        drifts = [-0.045, -0.030, -0.020, -0.005, +0.010]
        learned_biases = []
        for d_val in drifts:
            node = DynamicsClassNode(base, train_rows)
            eval_rows = _synthetic_linear_world_rows(
                n_episodes=2, steps_per_ep=30, drift=d_val, gain=0.10, seed=10,
            )
            for r in eval_rows:
                node.observe(r["obs"], r["history"], r["next_obs"])
            learned_biases.append(float(node.delta_w[-1, 1]))
            if abs(d_val - (-0.020)) > 1e-6:
                self.assertIn(1, node.active_channels())
                self.assertGreater(node.mean_credibility(), 0.5)
                ps = node.bind_state(eval_rows[-1]["obs"])
                self.assertIsInstance(ps, ParameterizedState)
                self.assertGreater(float(np.linalg.norm(ps.variables)), 0.0)

        corr = float(np.corrcoef(drifts, learned_biases)[0, 1])
        self.assertGreater(corr, 0.999)

        # Mid-stream sudden switch triggers change detection and recovers quickly
        node = DynamicsClassNode(base, train_rows)
        seg_a = _synthetic_linear_world_rows(
            n_episodes=1, steps_per_ep=25, drift=-0.04, gain=0.10, seed=20,
            include_reset_jumps=False,
        )
        seg_b = _synthetic_linear_world_rows(
            n_episodes=1, steps_per_ep=25, drift=+0.01, gain=0.10, seed=21,
            include_reset_jumps=False,
        )
        for r in seg_a:
            node.observe(r["obs"], r["history"], r["next_obs"])
        sig_a = node.world_signature().copy()
        changes_before = node.changes
        post_errs = []
        for idx, r in enumerate(seg_b):
            pred = node.predict(r["obs"].reshape(1, -1), r["history"].reshape(1, -1))[0]
            if idx >= 8:
                post_errs.append(abs(float(pred[1]) - float(r["next_obs"][1])))
            node.observe(r["obs"], r["history"], r["next_obs"])
        sig_b = node.world_signature().copy()
        self.assertGreater(node.changes, changes_before)
        self.assertLess(float(np.mean(post_errs)), 0.005)
        self.assertLess(variable_similarity(sig_a, sig_b, sigma=0.75), 0.25)

    def test_variable_scoped_failure_memory_and_stage_updater_isolate_worlds(self):
        # Build a tiny summarizer and test that failures recorded under sig_a do NOT
        # fire under distant sig_b, while matching sig_a fires strongly.
        train_rows = _synthetic_linear_world_rows(
            n_episodes=12, steps_per_ep=24, drift=-0.02, gain=0.10, seed=5,
        )
        summ = StageRegimeSummarizer(
            n_regimes=4, min_dwell=3, hysteresis=2, partition="survival_tree",
        ).fit_rows(train_rows, lag=0)
        base_eps = summ.transform_rows(train_rows, lag=0)

        fm = TrunkPointFailureMemory(summ, sigma_mult=0.40, prior=2.0, var_sigma=0.5)
        fm.ingest_rows(train_rows, include_failures=False)
        sig_a = np.asarray([0.0, 0.0, -2.0, 0.0, 0.0, 0.0], dtype=float)
        sig_b = np.asarray([0.0, 0.0, +2.0, 0.0, 0.0, 0.0], dtype=float)

        dead_state = np.asarray(train_rows[5]["obs"], dtype=float)
        for k in range(5):
            ep_rows = [
                dict(
                    episode=500 + k,
                    obs=dead_state - np.asarray([0.01, 0.0, 0.0]),
                    next_obs=dead_state,
                    history=np.asarray([0.0]),
                    signals=np.asarray([0.0, 1.0]),
                    done=True,
                ),
            ]
            fm.ingest_rows(ep_rows, include_failures=True, var_signature=sig_a)

        h_same = float(fm.hazard_batch(dead_state.reshape(1, -1), var_signature=sig_a)[0])
        h_other = float(fm.hazard_batch(dead_state.reshape(1, -1), var_signature=sig_b)[0])
        self.assertGreater(h_same, 0.10)
        self.assertLess(h_other, 0.01)
        self.assertGreater(h_same, 10.0 * max(h_other, 1e-6))

        # OnlineRegimeContinuationUpdater triggers immediate refit on variable shift
        updater = OnlineRegimeContinuationUpdater(
            summ, base_eps, update_every=10, var_sigma=0.5, min_var_similarity=0.35,
            failure_memory=fm, failure_memory_weight=0.25,
        )
        updater.refit(var_signature=sig_a)
        u_before = updater.updates
        # Even though update_every=10, shifting from sig_a -> sig_b forces re-instantiation
        refitted = updater.record_episode_rows(
            train_rows[:24], lag=0, var_signature=sig_b,
        )
        self.assertTrue(refitted)
        self.assertEqual(updater.updates, u_before + 1)

    def test_run_seed_world_variables_opt_in_invariance_and_end_to_end(self):
        from substrate_nodes import run_seed

        common = dict(
            seed=100,
            steps=300,
            coverage=0.1,
            cap=40,
            max_nodes=120,
            epochs=1,
            horizon=6,
            lag=3,
            decision="chained",
            chain_chunk_size=2,
            chain_hops=3,
            chain_beam_width=2,
            chain_demand_weight=0.10,
            regime_value_weight=1.0,
            regime_n_regimes=8,
            regime_partition="survival_tree",
            regime_online_every=2,
            regime_failure_weight=0.25,
            eval_seeds=[120, 121],
            eval_episodes=1,
        )
        off_a = run_seed(dict(common))
        off_b = run_seed(dict(common, world_variables=False))
        self.assertEqual(
            [r["frames"] for r in off_a["evals"]],
            [r["frames"] for r in off_b["evals"]],
        )
        self.assertIsNone(off_a["world_variables"])

        on = run_seed(dict(common, world_variables=True))
        self.assertIsNotNone(on["world_variables"])
        self.assertGreater(on["world_variables"]["updates"], 0)
        self.assertEqual(len(on["evals"]), 2)


if __name__ == "__main__":
    unittest.main()
