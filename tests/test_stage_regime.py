"""Tests for task-agnostic stage regime summarization and calibrated transitions."""
import unittest

import numpy as np

from stage_regime import (
    UNKNOWN,
    CalibratedRegimeTransitionModel,
    ControllableFeatureDemand,
    OnlineRegimeContinuationUpdater,
    RegimeContinuationEvaluator,
    StageRegimeSummarizer,
    TrunkPointFailureMemory,
    detect_pure_drift_dims,
)


def _synthetic_controllable_episodes(n_episodes=24, seed=0):
    """Two-regime controllable dynamical world + an unbounded step counter.

    Dim 0: state position in [-1, 1]; action 1 pushes up (+0.18), action 0 drifts down (-0.12).
    Dim 1: periodic ramp (-0.05/step, resets +1.0 every 15 steps) — NOT a pure counter!
    Dim 2: pure monotonic counter (+1.0 every step) — MUST be stripped as pure drift!
    """
    rng = np.random.RandomState(seed)
    episodes = []
    for ep_id in range(n_episodes):
        x = float(rng.uniform(-0.2, 0.2))
        ramp = 0.75
        obs = [[x, ramp, 0.0]]
        actions = []
        feedback = []
        terminal = False
        # Alternate policies across episodes so both low and high profiles occur
        mode = ep_id % 3
        for t in range(36):
            if mode == 0:
                a = 0
            elif mode == 1:
                a = 1 if (t % 2 == 0) else 0
            else:
                a = 1 if (t % 3 == 0) else 0
            actions.append(a)
            x = x + (0.18 if a == 1 else -0.10) + float(rng.normal(0, 0.01))
            ramp = ramp - 0.05
            if ramp < 0.0:
                ramp += 1.0
            obs.append([x, ramp, float(t + 1)])
            if abs(x) > 1.2:
                feedback.append([0.0, 1.0])
                terminal = True
                break
            feedback.append([1.0, 0.0])
        episodes.append(dict(
            episode=ep_id,
            observations=np.asarray(obs, dtype=float),
            actions=tuple(actions),
            feedback=np.asarray(feedback, dtype=float),
            terminal=terminal,
            censored=not terminal,
        ))
    return episodes


class StageRegimeTests(unittest.TestCase):
    def test_detect_pure_drift_dims_strips_counter_but_keeps_resetting_ramp(self):
        episodes = _synthetic_controllable_episodes(n_episodes=12)
        drift = detect_pure_drift_dims([ep["observations"] for ep in episodes])
        self.assertEqual(drift, (2,))

    def test_resolve_is_immune_to_unbounded_counter_extrapolation(self):
        episodes = _synthetic_controllable_episodes(n_episodes=18)
        model = StageRegimeSummarizer(n_regimes=6, min_dwell=4, hysteresis=2)
        model.fit_episodes(episodes)
        self.assertEqual(model.drift_dims_, (2,))

        # Counter value 10_000 is far beyond training (<=36), yet state (0.0, 0.5) is in-distribution
        in_dist = model.resolve(np.asarray([0.0, 0.5, 10000.0]))
        self.assertEqual(in_dist["status"], "supported")
        self.assertTrue(in_dist["in_radius"])

        # True state OOD in non-drift dims must return UNKNOWN
        ood = model.resolve(np.asarray([50.0, 0.5, 5.0]))
        self.assertEqual(ood["status"], UNKNOWN)
        self.assertEqual(ood["reason"], "outside_radius")

    def test_summarizer_fit_is_frozen_during_heldout_segmentation(self):
        episodes = _synthetic_controllable_episodes(n_episodes=16)
        train, heldout = episodes[:12], episodes[12:]
        model = StageRegimeSummarizer(n_regimes=6, min_dwell=4, hysteresis=2)
        model.fit_episodes(train)
        snap_before = model.snapshot()
        centers_before = model.centers_.copy()
        for ep in heldout:
            stages = model.segment_episode(
                episode=ep["episode"],
                observations=ep["observations"],
                actions=ep["actions"],
                feedback=ep["feedback"],
                terminal=ep["terminal"],
                censored=ep["censored"],
            )
            self.assertGreaterEqual(len(stages), 1)
            self.assertEqual(sum(s.duration for s in stages), len(ep["actions"]))
            self.assertEqual(stages[-1].terminal, ep["terminal"])
            self.assertEqual(stages[-1].censored, ep["censored"])
        self.assertEqual(model.snapshot(), snap_before)
        np.testing.assert_array_equal(model.centers_, centers_before)

    def test_calibrated_transition_rollout_conserves_probability_and_shows_control(self):
        episodes = _synthetic_controllable_episodes(n_episodes=30)
        model = StageRegimeSummarizer(n_regimes=6, min_dwell=4, hysteresis=2)
        model.fit_episodes(episodes)
        stage_eps = [
            model.segment_episode(
                episode=ep["episode"],
                observations=ep["observations"],
                actions=ep["actions"],
                feedback=ep["feedback"],
                terminal=ep["terminal"],
                censored=ep["censored"],
            )
            for ep in episodes
        ]
        tm = CalibratedRegimeTransitionModel(max_depth=2, min_episodes=2)
        tm.fit(stage_eps, use_action=True)
        self.assertIn(tm.alpha_, tm.alphas)

        # Probability conservation over k=3 stages (pooled and action-planned)
        start_regime = stage_eps[0][0].regime
        pooled_roll = tm.rollout([start_regime], steps=3, action_profiles=None)
        total_mass = (
            sum(pooled_roll["regime_distribution"].values())
            + pooled_roll["terminal_probability"]
            + pooled_roll["unknown_mass"]
        )
        self.assertAlmostEqual(total_mass, 1.0, places=9)
        self.assertAlmostEqual(
            sum(p for _, p in pooled_roll["elapsed_distribution"]), 1.0, places=9,
        )
        self.assertGreater(pooled_roll["expected_elapsed_frames"], 3.0)

        # Action-plan rollout differentiates idle vs active profiles
        idle_roll = tm.rollout(
            [start_regime], steps=2, action_profiles=[("idle", 0), ("idle", 0)],
        )
        active_roll = tm.rollout(
            [start_regime], steps=2, action_profiles=[("high", 1), ("high", 1)],
        )
        self.assertAlmostEqual(
            sum(idle_roll["regime_distribution"].values())
            + idle_roll["terminal_probability"]
            + idle_roll["unknown_mass"],
            1.0,
            places=9,
        )
        ctrl = tm.controllability_report(model.n_regimes)
        self.assertGreaterEqual(ctrl["controllable_regimes"], 1)
        self.assertNotEqual(
            idle_roll["regime_distribution"], active_roll["regime_distribution"],
        )

    def test_continuation_values_and_batch_evaluator_respect_ood(self):
        episodes = _synthetic_controllable_episodes(n_episodes=30)
        model = StageRegimeSummarizer(n_regimes=6, min_dwell=4, hysteresis=2)
        model.fit_episodes(episodes)
        stage_eps = [
            model.segment_episode(
                episode=ep["episode"],
                observations=ep["observations"],
                actions=ep["actions"],
                feedback=ep["feedback"],
                terminal=ep["terminal"],
                censored=ep["censored"],
            )
            for ep in episodes
        ]
        tm = CalibratedRegimeTransitionModel(max_depth=2, min_episodes=2)
        tm.fit(stage_eps, use_action=True)

        pooled = tm.continuation_values(model.n_regimes, horizon_stages=3, mode="pooled")
        optimal = tm.continuation_values(model.n_regimes, horizon_stages=3, mode="optimal")
        # Optimal survival must be >= pooled survival at every regime
        for s_opt, s_pool in zip(optimal["survival"], pooled["survival"]):
            self.assertGreaterEqual(s_opt + 1e-9, s_pool)
            self.assertTrue(0.0 <= s_pool <= 1.0)
            self.assertTrue(0.0 <= s_opt <= 1.0)

        ev = RegimeContinuationEvaluator(model, optimal, min_regime_episodes=2)
        batch = np.asarray([
            [0.0, 0.5, 5000.0],   # in-distribution despite huge counter
            [99.0, 0.5, 5.0],     # OOD in non-drift dim
        ])
        risks, supported = ev.continuation_risk_batch(batch)
        self.assertTrue(bool(supported[0]))
        self.assertFalse(bool(supported[1]))
        self.assertAlmostEqual(
            float(risks[1]), 1.0 - optimal["baseline_survival"], places=9,
        )
        cb = ev.as_stage_value_callback()
        cb_risks = cb(batch, None)
        np.testing.assert_allclose(cb_risks, risks)

        # Coverage gate abstains (zeros) when coverage < min_decision_coverage
        gated = RegimeContinuationEvaluator(
            model, optimal, min_regime_episodes=2,
            min_decision_coverage=0.75, ood_policy="neutral_mean",
        )
        stats = {}
        gated_cb = gated.as_stage_value_callback(stats=stats)
        # 1 of 2 supported (0.5 < 0.75) -> abstains with zeros
        abstained = gated_cb(batch, None)
        np.testing.assert_array_equal(abstained, np.zeros(2))
        self.assertEqual(stats["abstained"], 1)
        self.assertEqual(stats["active"], 0)
        # 3 of 4 supported (0.75 >= 0.75) -> active, OOD gets supported mean
        batch4 = np.asarray([
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.5, 3.0],
            [99.0, 0.5, 4.0],
        ])
        active_vals = gated_cb(batch4, None)
        self.assertEqual(stats["active"], 1)
        self.assertAlmostEqual(
            float(active_vals[3]), float(np.mean(active_vals[:3])), places=9,
        )

    def test_module_is_task_agnostic(self):
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "core" / "stage_regime.py"
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read().lower()
        for forbidden in ("flappy", "bird_x", "pipe", "gap", "flap", "gravity"):
            self.assertNotIn(forbidden, source)

    def test_run_seed_opt_in_regime_wiring_is_invariant_at_zero(self):
        from substrate_nodes import run_seed

        base_task = dict(
            seed=100, steps=400, coverage=0.1, max_nodes=200,
            birth_threshold=0.45, epochs=1, cap=80, horizon=4, lag=3,
            eval_seeds=[120], eval_episodes=1,
        )
        default_res = run_seed(dict(base_task))
        zero_res = run_seed(dict(base_task, regime_value_weight=0.0))
        self.assertIsNone(default_res["regime_continuation"])
        self.assertIsNone(zero_res["regime_continuation"])
        self.assertEqual(default_res["evals"], zero_res["evals"])

        opt_in_res = run_seed(dict(
            base_task,
            regime_value_weight=0.25,
            regime_n_regimes=6,
            regime_min_dwell=4,
            regime_k_stages=2,
            regime_min_coverage=0.5,
        ))
        rc = opt_in_res["regime_continuation"]
        self.assertIsNotNone(rc)
        self.assertAlmostEqual(rc["weight"], 0.25)
        self.assertGreater(rc["stats"]["decisions"], 0)
        self.assertEqual(rc["evaluator"]["value_key"], "cum_survival")
        self.assertEqual(rc["evaluator"]["soft_top_k"], 4)

    def test_lcb_cumulative_survival_and_soft_kernel_interpolation(self):
        episodes = _synthetic_controllable_episodes(n_episodes=30)
        model = StageRegimeSummarizer(n_regimes=6, min_dwell=4, hysteresis=2)
        model.fit_episodes(episodes)
        self.assertGreater(model.kernel_scale_, 0.0)
        self.assertIn("kernel_scale", model.snapshot())

        stage_eps = [
            model.segment_episode(
                episode=ep["episode"],
                observations=ep["observations"],
                actions=ep["actions"],
                feedback=ep["feedback"],
                terminal=ep["terminal"],
                censored=ep["censored"],
            )
            for ep in episodes
        ]
        tm = CalibratedRegimeTransitionModel(max_depth=2, min_episodes=2)
        tm.fit(stage_eps, use_action=True)

        pooled = tm.continuation_values(
            model.n_regimes, horizon_stages=6, mode="pooled",
        )
        lcb = tm.continuation_values(
            model.n_regimes, horizon_stages=6, mode="lcb", lcb_z=1.0,
        )
        strict_lcb = tm.continuation_values(
            model.n_regimes, horizon_stages=6, mode="lcb", lcb_z=100.0,
        )
        for p, c, s in zip(
            pooled["cum_survival"], lcb["cum_survival"], strict_lcb["cum_survival"]
        ):
            self.assertTrue(0.0 <= p <= 1.0)
            self.assertTrue(0.0 <= c <= 1.0)
            self.assertAlmostEqual(s, p, places=9)

        # Soft kernel evaluator interpolates continuously between regime centers
        # while respecting OOD gating.
        soft_ev = RegimeContinuationEvaluator(
            model, lcb, min_regime_episodes=2,
            min_decision_coverage=0.5, ood_policy="neutral_mean",
            value_key="cum_survival", soft_top_k=4, kernel_tau=0.7,
        )
        # Construct two distinct points moving smoothly between supported centers
        sup_ids = [
            r for r, cnt in enumerate(lcb["regime_support"]) if cnt >= 2
        ]
        self.assertGreaterEqual(len(sup_ids), 2)
        # Map centers back to raw observation space (non-drift dims 0, 1 + counter 2)
        c0 = model.centers_[sup_ids[0]] * model.scale_ + model.mean_
        c1 = model.centers_[sup_ids[1]] * model.scale_ + model.mean_
        p_near = np.array([c0[0], c0[1], 10.0])
        p_mid = np.array([
            0.75 * c0[0] + 0.25 * c1[0],
            0.75 * c0[1] + 0.25 * c1[1],
            10.0,
        ])
        p_ood = np.array([99.0, 99.0, 10.0])
        batch = np.vstack([p_near, p_mid, p_ood])
        risks, supported = soft_ev.continuation_risk_batch(batch)
        self.assertTrue(bool(supported[0]))
        self.assertTrue(bool(supported[1]))
        self.assertFalse(bool(supported[2]))
        # Penalties are finite and OOD receives neutral mean of supported candidates
        penalties, cov, active = soft_ev.decision_penalties(batch)
        self.assertTrue(active)
        self.assertAlmostEqual(cov, 2.0 / 3.0, places=9)
        self.assertTrue(np.isfinite(penalties).all())
        self.assertAlmostEqual(
            float(penalties[2]), float(np.mean(penalties[:2])), places=9,
        )
        # Verify soft-weights reuse parity on NodeRuntime
        from node_runtime import NodeRuntime
        rt = NodeRuntime(obs_dim=2, lag=1, action_space=(0, 1))
        for idx in range(20):
            obs = np.array([idx * 0.05, (idx % 3) * 0.2], dtype=float)
            nxt = obs + 0.01
            rt.learn_transition(obs, idx % 2, nxt, [0.1, float(idx == 19)], done=(idx == 19))
        enc = np.array([[0.1, 0.2, 0.0], [0.5, 0.4, 1.0]], dtype=float)
        sw = rt._soft_weights_batch(enc)
        p1, d1 = rt.valence_state_batch(enc)
        p2, d2 = rt.valence_state_batch(enc, soft_weights=sw)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(d1, d2)
        s1 = rt.pain_signal_batch(enc)
        s2 = rt.pain_signal_batch(enc, soft_weights=sw)
        np.testing.assert_array_equal(s1, s2)

    def test_survival_tree_partition_ignores_collinear_duplicate_dim(self):
        rng = np.random.RandomState(42)
        episodes = []
        for ep in range(30):
            # obs dims:
            #   d0: informative control coordinate (large |d0| dies quickly)
            #   d1: exact collinear duplicate of d0 (d0 + 5.0)
            #   d2: pure independent noise
            #   d3: pure drift counter (stripped automatically)
            long_ep = (ep % 2 == 0)
            length = 45 if long_ep else 12
            d0 = rng.normal(0.0 if long_ep else 2.5, 0.3, size=length + 1)
            d1 = d0 + 5.0
            d2 = rng.uniform(-1.0, 1.0, size=length + 1)
            d3 = np.arange(length + 1, dtype=float)
            obs = np.column_stack([d0, d1, d2, d3])
            actions = [int(ep % 2)] * length
            fb = np.zeros((length, 2), dtype=float)
            fb[:, 0] = 1.0
            fb[-1, 1] = 1.0
            episodes.append(dict(
                episode=ep,
                observations=obs,
                actions=actions,
                feedback=fb,
                terminal=True,
                censored=False,
            ))

        summ = StageRegimeSummarizer(
            n_regimes=6, min_dwell=4, hysteresis=2,
            partition="survival_tree", min_leaf_episodes=3, min_leaf_samples=10,
        ).fit_episodes(episodes)
        snap = summ.snapshot()
        self.assertEqual(snap["partition"], "survival_tree")
        self.assertIn(3, summ.drift_dims_)
        self.assertGreaterEqual(len(summ.centers_), 2)
        self.assertLessEqual(len(summ.centers_), 6)
        # Collinear duplicate d1 should never be chosen over identical d0
        self.assertNotIn(1, summ.split_dims_)
        self.assertIn(0, summ.split_dims_)

        stage_eps = [
            summ.segment_episode(
                episode=ep["episode"],
                observations=ep["observations"],
                actions=ep["actions"],
                feedback=ep["feedback"],
                terminal=ep["terminal"],
                censored=ep["censored"],
            )
            for ep in episodes
        ]
        tm = CalibratedRegimeTransitionModel(max_depth=2, min_episodes=2)
        tm.fit(stage_eps, use_action=True)
        vals = tm.continuation_values(
            len(summ.centers_), horizon_stages=4, mode="lcb", lcb_z=1.0,
        )
        ev = RegimeContinuationEvaluator(
            summ, vals, min_regime_episodes=2,
            min_decision_coverage=0.5, ood_policy="neutral_mean",
            value_key="cum_survival", soft_top_k=4, kernel_tau=0.7,
        )
        safe_pt = np.array([[0.0, 5.0, 0.0, 999.0]])
        risky_pt = np.array([[2.5, 7.5, 0.0, 999.0]])
        r_safe, s_safe = ev.continuation_risk_batch(safe_pt)
        r_risk, s_risk = ev.continuation_risk_batch(risky_pt)
        self.assertTrue(bool(s_safe[0]))
        self.assertTrue(bool(s_risk[0]))
        self.assertLess(float(r_safe[0]), float(r_risk[0]))

    def test_controllable_feature_demand_filters_uncontrollable_and_stays_centered(self):
        # Synthetic world:
        #   d0: action-controllable coordinate; safe band [-1, +1], survival peaks near 0
        #       even when behavior occupancy is skewed toward +0.5 inside [-1, +1]
        #   d1: uncontrollable periodic ramp (action has zero effect)
        #   d2: pure drift counter (stripped automatically)
        class FakeWorld:
            def predict(self, states, history):
                nxt = np.asarray(states, dtype=float).copy()
                act = np.asarray(history, dtype=float)[:, 0]
                nxt[:, 0] = nxt[:, 0] + 0.15 * (act - 0.5)
                nxt[:, 1] = nxt[:, 1] - 0.05
                nxt[:, 2] = nxt[:, 2] + 1.0
                return nxt

        class FakeAgent:
            action_space = (0, 1)

        rng = np.random.RandomState(7)
        rows = []
        for ep in range(30):
            long_ep = (ep % 3 != 0)
            length = 40 if long_ep else 8
            # Skewed occupancy inside safe band [-0.9, +0.9] for long episodes
            if long_ep:
                vals = np.concatenate([
                    rng.uniform(-0.9, 0.9, size=max(4, length // 4)),
                    rng.uniform(0.2, 0.85, size=length + 1 - max(4, length // 4)),
                ])
                rng.shuffle(vals)
            else:
                # Failing episodes deviate outside [-1, +1]
                sign = 1.0 if (ep % 2 == 0) else -1.0
                vals = sign * np.linspace(0.6, 2.2, length + 1)
            for t in range(length):
                obs = np.array([vals[t], 1.0 - 0.05 * (t % 20), float(t)])
                nxt = np.array([vals[t + 1], 1.0 - 0.05 * ((t + 1) % 20), float(t + 1)])
                rows.append(dict(
                    episode=ep,
                    obs=obs,
                    next_obs=nxt,
                    history=np.array([float(t % 2), 0.0]),
                    signals=np.array([1.0, float(t == length - 1)]),
                    done=(t == length - 1),
                ))

        dem = ControllableFeatureDemand().fit_rows(FakeWorld(), FakeAgent(), rows, lag=0)
        snap = dem.snapshot()
        self.assertTrue(snap["fitted"])
        self.assertEqual(snap["controllable_dims"], (0,))
        self.assertAlmostEqual(float(dem.demand_weights_[0]), 1.0, places=6)
        self.assertEqual(float(dem.demand_weights_[1]), 0.0)
        self.assertEqual(float(dem.demand_weights_[2]), 0.0)
        # Occupancy-invariant envelope center stays near 0 despite skewed occupancy
        self.assertLess(abs(float(dem.env_center_[0])), 0.25)
        # Distance ranks centered state strictly better than extreme state
        d_batch = dem.distance_batch(np.array([[0.0, 0.5, 10.0], [1.8, 0.5, 10.0]]))
        self.assertLess(float(d_batch[0]), float(d_batch[1]))

    def test_chained_shallow_search_separates_heuristic_from_final_score(self):
        from substrate_nodes import chained_shallow_result, fit, latent

        world, agent = fit(
            100, steps=400, coverage=0.1, max_nodes=200,
            epochs=1, lag=3,
        )
        env = latent[3](1200000)
        prefix = np.zeros(3, dtype=float)
        # When beam_width >= 2**horizon (no pruning), heuristic_demand_weight
        # MUST NOT change the final objective score at all!
        dem = ControllableFeatureDemand().fit_rows(
            world, agent, agent.milestone_training_rows, lag=3,
        )
        r_no_h = chained_shallow_result(
            env, world, agent, prefix,
            chunk_size=2, n_hops=2, beam_width=16, lag=3,
            heuristic_demand=dem.distance_batch, heuristic_demand_weight=0.0,
        )
        r_with_h = chained_shallow_result(
            env, world, agent, prefix,
            chunk_size=2, n_hops=2, beam_width=16, lag=3,
            heuristic_demand=dem.distance_batch, heuristic_demand_weight=5.0,
        )
        self.assertEqual(r_no_h.score, r_with_h.score)
        np.testing.assert_array_equal(r_no_h.actions, r_with_h.actions)

        # Compact active pruning (beam_width=2) runs cleanly with far fewer candidates
        r_pruned = chained_shallow_result(
            env, world, agent, prefix,
            chunk_size=2, n_hops=3, beam_width=2, lag=3,
            heuristic_demand=dem.distance_batch, heuristic_demand_weight=0.10,
        )
        self.assertEqual(r_pruned.actions.shape, (6,))
        self.assertLess(r_pruned.candidate_count, 64)

    def test_online_regime_updater_keeps_partition_frozen_and_matches_full_refit(self):
        episodes = _synthetic_controllable_episodes(n_episodes=24, seed=7)

        def _ep_to_rows(ep):
            obs = ep["observations"]
            acts = ep["actions"]
            fb = ep["feedback"]
            out = []
            for t, a in enumerate(acts):
                out.append(dict(
                    episode=ep["episode"],
                    obs=obs[t],
                    next_obs=obs[t + 1],
                    history=np.asarray([float(a)], dtype=float),
                    signals=fb[t],
                    done=(ep["terminal"] and t == len(acts) - 1),
                ))
            return out

        base_rows = [r for ep in episodes[:16] for r in _ep_to_rows(ep)]
        online_eps_rows = [_ep_to_rows(ep) for ep in episodes[16:22]]

        summ = StageRegimeSummarizer(
            n_regimes=6, min_dwell=4, hysteresis=2, partition="survival_tree",
        ).fit_rows(base_rows, lag=0)
        centers_before = summ.centers_.copy()
        snap_before = summ.snapshot()
        base_stage_eps = summ.transform_rows(base_rows, lag=0)

        updater = OnlineRegimeContinuationUpdater(
            summ,
            base_stage_eps,
            horizon_stages=4,
            mode="lcb",
            min_action_episodes=2,
            lcb_z=1.0,
            update_every=2,
            max_online_frames=0,
        )
        stats = {}
        cb = updater.as_stage_value_callback(stats=stats)
        probe_states = np.asarray(
            [[0.0, 0.5, 1.0], [0.7, 0.5, 2.0], [-0.7, 0.5, 3.0]], dtype=float,
        )
        p0 = cb(probe_states, None)
        self.assertEqual(stats["decisions"], 1)

        # Empty input is a clean no-op
        self.assertFalse(updater.record_episode_rows([], lag=0))
        self.assertEqual(updater.episodes_seen, 0)
        self.assertEqual(updater.updates, 0)

        # Cadence: refits on episodes 2, 4, 6
        refit_flags = [
            updater.record_episode_rows(ep_rows, lag=0)
            for ep_rows in online_eps_rows
        ]
        self.assertEqual(refit_flags, [False, True, False, True, False, True])
        self.assertEqual(updater.episodes_seen, 6)
        self.assertEqual(updater.updates, 3)

        # Partition stayed strictly frozen
        np.testing.assert_array_equal(summ.centers_, centers_before)
        self.assertEqual(summ.snapshot(), snap_before)

        # Exact equivalence to full batch transform on base_rows + online_rows
        merged_rows = list(base_rows)
        for ep_rows in online_eps_rows:
            merged_rows.extend(ep_rows)
        full_stage_eps = summ.transform_rows(merged_rows, lag=0)
        tm_full = CalibratedRegimeTransitionModel(max_depth=2, min_episodes=2).fit(
            full_stage_eps, use_action=True,
        )
        vals_full = tm_full.continuation_values(
            len(summ.centers_), horizon_stages=4, mode="lcb",
            min_action_episodes=2, lcb_z=1.0,
        )
        ev_full = RegimeContinuationEvaluator(
            summ, vals_full, min_regime_episodes=2, min_decision_coverage=0.80,
            ood_policy="neutral_mean", value_key="cum_survival",
            soft_top_k=4, kernel_tau=0.7,
        )
        p_online = cb(probe_states, None)
        p_full, _cov, _act = ev_full.decision_penalties(probe_states)
        np.testing.assert_allclose(p_online, p_full, atol=1e-12)
        self.assertEqual(stats["decisions"], 2)
        self.assertEqual(p0.shape, (3,))

        # Bounded sliding window evicts oldest episodes while retaining latest
        bounded = OnlineRegimeContinuationUpdater(
            summ, base_stage_eps, update_every=1, max_online_frames=40,
        )
        for ep_rows in online_eps_rows:
            bounded.record_episode_rows(ep_rows, lag=0)
        b_snap = bounded.snapshot()
        self.assertEqual(b_snap["episodes_seen"], 6)
        self.assertEqual(b_snap["updates"], 6)
        self.assertLess(b_snap["online_episodes_buffered"], 6)
        self.assertGreaterEqual(b_snap["online_episodes_buffered"], 1)

    def test_run_seed_fixed_partition_online_regime_wiring(self):
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
            eval_seeds=[120, 121, 122, 123],
            eval_episodes=1,
        )
        frozen_a = run_seed(dict(common))
        frozen_b = run_seed(dict(common, regime_online_every=0))
        self.assertEqual(
            [row["frames"] for row in frozen_a["evals"]],
            [row["frames"] for row in frozen_b["evals"]],
        )
        self.assertIsNone(frozen_a["regime_continuation"]["online"])
        self.assertIsNone(frozen_b["regime_continuation"]["online"])

        online = run_seed(dict(
            common, regime_online_every=2, regime_online_frames=500,
        ))
        rc = online["regime_continuation"]
        self.assertIsNotNone(rc["online"])
        self.assertEqual(rc["online"]["episodes_seen"], 4)
        self.assertEqual(rc["online"]["updates"], 2)
        self.assertEqual(rc["online"]["update_every"], 2)
        self.assertEqual(rc["online"]["max_online_frames"], 500)
        self.assertGreater(rc["online"]["online_frames_buffered"], 0)
        self.assertIsNone(rc["online"]["failure_memory"])
        self.assertEqual(len(online["evals"]), 4)

        # Explicitly passing regime_failure_weight=0.0 stays bit-for-bit identical
        online_zero_fm = run_seed(dict(
            common, regime_online_every=2, regime_online_frames=500,
            regime_failure_weight=0.0,
        ))
        self.assertEqual(
            [row["frames"] for row in online["evals"]],
            [row["frames"] for row in online_zero_fm["evals"]],
        )

        # Enabling opt-in pure-online point-event failure memory runs end-to-end
        online_fm = run_seed(dict(
            common, regime_online_every=2, regime_online_frames=500,
            regime_failure_weight=0.25, regime_failure_sigma=0.25,
            regime_failure_prior=4.0,
        ))
        fm_snap = online_fm["regime_continuation"]["online"]["failure_memory"]
        self.assertIsNotNone(fm_snap)
        self.assertGreater(fm_snap["buffered_safe"], 0)
        self.assertEqual(
            fm_snap["active_failure_leaves"] + fm_snap["compressed_leaves"], 8,
        )

    def test_trunk_point_failure_memory_localizes_terminal_hazard_and_respects_ood(self):
        episodes = _synthetic_controllable_episodes(n_episodes=24, seed=11)

        def _ep_to_rows(ep):
            obs = ep["observations"]
            acts = ep["actions"]
            fb = ep["feedback"]
            return [
                dict(
                    episode=ep["episode"],
                    obs=obs[t],
                    next_obs=obs[t + 1],
                    history=np.asarray([float(a)], dtype=float),
                    signals=fb[t],
                    done=(ep["terminal"] and t == len(acts) - 1),
                )
                for t, a in enumerate(acts)
            ]

        base_rows = [r for ep in episodes[:18] for r in _ep_to_rows(ep)]
        summ = StageRegimeSummarizer(
            n_regimes=6, min_dwell=4, hysteresis=2, partition="survival_tree",
        ).fit_rows(base_rows, lag=0)

        mem = TrunkPointFailureMemory(
            summ, sigma_mult=0.25, prior=4.0, max_fail_per_leaf=5, max_safe_per_leaf=30,
        )
        # Safe-only background seed produces zero hazard everywhere
        added = mem.ingest_rows(base_rows, include_failures=False)
        self.assertEqual(added, 0)
        self.assertEqual(mem.total_failures, 0)
        probe = np.asarray(
            [[0.0, 0.5, 5.0], [1.25, 0.5, 10.0], [50.0, 0.5, 5.0]], dtype=float,
        )
        np.testing.assert_array_equal(mem.hazard_batch(probe), np.zeros(3))

        # Inject 6 identical online terminal events at x=+1.25 (cap per leaf = 5)
        for k in range(6):
            ep_rows = [
                dict(
                    episode=1000 + k,
                    obs=np.asarray([0.0, 0.5, 1.0]),
                    next_obs=np.asarray([0.1, 0.45, 2.0]),
                    history=np.asarray([0.0]),
                    signals=np.asarray([1.0, 0.0]),
                    done=False,
                ),
                dict(
                    episode=1000 + k,
                    obs=np.asarray([1.15, 0.5, 3.0]),
                    next_obs=np.asarray([1.25, 0.5, 4.0]),
                    history=np.asarray([1.0]),
                    signals=np.asarray([0.0, 1.0]),
                    done=True,
                ),
            ]
            mem.ingest_rows(ep_rows, include_failures=True)

        snap = mem.snapshot()
        self.assertEqual(snap["total_failures"], 6)
        self.assertLessEqual(snap["buffered_failures"], 5)

        h = mem.hazard_batch(probe)
        # Safe corridor x=0.0 stays near 0; repeated terminal x=1.25 is clearly elevated; OOD is 0
        self.assertLess(float(h[0]), 0.05)
        self.assertGreater(float(h[1]), 0.40)
        self.assertEqual(float(h[2]), 0.0)

        # Integration with OnlineRegimeContinuationUpdater adds exact weighted hazard
        base_stage_eps = summ.transform_rows(base_rows, lag=0)
        updater_plain = OnlineRegimeContinuationUpdater(
            summ, base_stage_eps, update_every=2,
        )
        updater_fm = OnlineRegimeContinuationUpdater(
            summ, base_stage_eps, update_every=2,
            failure_memory=mem, failure_memory_weight=0.25,
        )
        p_plain = updater_plain.as_stage_value_callback()(probe, None)
        p_fm = updater_fm.as_stage_value_callback()(probe, None)
        np.testing.assert_allclose(p_fm - p_plain, 0.25 * h, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
