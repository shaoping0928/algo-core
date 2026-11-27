"""Cross-task generalization smoke tests on DelayedCartPole (zero core modifications)."""
from __future__ import annotations

import unittest
import numpy as np


class DelayedCartPoleCrossTaskTests(unittest.TestCase):
    def test_delayed_cartpole_lag_and_termination(self):
        from cartpole_env import DelayedCartPole

        env = DelayedCartPole(seed=42, lag=2, cap=200)
        obs0 = env.obs()
        self.assertEqual(obs0.shape, (5,))
        self.assertEqual(len(env.pending()), 2)
        self.assertFalse(env.dead)
        for _ in range(200):
            obs, dop, dead = env.step(1)
            self.assertEqual(obs.shape, (5,))
            if dead:
                self.assertEqual(dop, 0.0)
                self.assertEqual(env.feedback()["pain"], 1.0)
                break
        self.assertTrue(env.dead)

    def test_full_phase12_stack_runs_unmodified_on_delayed_cartpole(self):
        from cartpole_env import make_cartpole_factory
        from stage_regime import (
            AdaptiveSearchController,
            ControllableFeatureDemand,
            OnlineFeatureDemandUpdater,
            OnlineRegimeContinuationUpdater,
            StageRegimeSummarizer,
            TrunkPointFailureMemory,
        )
        from substrate_nodes import adaptive_chained_result, fit

        lag = 2
        env_factory = make_cartpole_factory(lag=lag, cap=80)
        world, agent = fit(
            100, steps=400, coverage=0.1, max_nodes=200,
            birth_threshold=0.45, epochs=1, lag=lag,
            env_factory=env_factory,
        )
        rows = list(agent.milestone_training_rows)
        summ = StageRegimeSummarizer(
            n_regimes=8, min_dwell=3, hysteresis=1, partition="survival_tree",
        ).fit_rows(rows, lag=lag)
        beps = summ.transform_rows(rows, lag=lag)
        dem = ControllableFeatureDemand().fit_rows(world, agent, rows, lag=lag)
        # Controllable dimensions must exclude the monotonic clock channel (dim 4)
        self.assertNotIn(4, dem.snapshot()["controllable_dims"])

        fm = TrunkPointFailureMemory(summ, sigma_mult=0.25, prior=4.0)
        fm.ingest_rows(rows, include_failures=False)
        updater = OnlineRegimeContinuationUpdater(
            summ, beps, horizon_stages=6, mode="lcb", min_action_episodes=2,
            update_every=2, failure_memory=fm, failure_memory_weight=0.25,
            stage_abstain_lcb=0.45,
        )
        cb = updater.as_stage_value_callback()
        ctrl = AdaptiveSearchController(meta_lcb=0.0)
        online_dem = OnlineFeatureDemandUpdater(
            dem, update_every=2, ema_alpha=0.30, min_online_frames=40,
            min_terminal_episodes=1,
        )

        for ep in range(3):
            ctrl.begin_episode(updater.stage_abstain_gate)
            env = env_factory(2000 + ep)
            prefix = np.zeros(lag, dtype=float)
            recent = [0.0] * lag
            ep_rows = []
            while not env.dead and env.t < 40:
                obs = np.asarray(env.obs(), dtype=float)
                res = adaptive_chained_result(
                    env, world, agent, prefix,
                    chunk_size=2, base_hops=3, max_hops=5,
                    beam_width=4, wide_beam_width=6, lag=lag,
                    stage_value=cb, stage_value_weight=1.0,
                    heuristic_demand=online_dem.distance_batch,
                    heuristic_demand_weight=0.10,
                    controller=ctrl,
                )
                a = int(res.actions[0])
                hist = np.asarray([float(a)] + recent[:lag], dtype=float)
                prefix = np.concatenate([[float(a)], prefix[:-1]])
                env.step(a)
                nxt = np.asarray(env.obs(), dtype=float)
                ep_rows.append(dict(
                    episode=5000 + ep, obs=obs, next_obs=nxt, history=hist,
                    signals=np.asarray([0.0, 1.0] if env.dead else [1.0, 0.0], dtype=float),
                    done=bool(env.dead),
                ))
                recent = [float(a)] + recent[:lag - 1]
            ctrl.record_episode(len(ep_rows))
            updater.record_episode_rows(ep_rows, lag=lag)
            online_dem.record_episode_rows(ep_rows)

        self.assertGreater(ctrl.adaptive_steps, 0)
        self.assertEqual(ctrl.episodes_seen, 3)


if __name__ == "__main__":
    unittest.main()
