import unittest
from pathlib import Path

import numpy as np

from node_features import FeatureRuleBank
from node_graph import MEMBERSHIP
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

    def test_action_probability_flattens_at_zero_margin(self):
        runtime = self.make_runtime()
        scores = {
            0: dict(cost=0.0, uncertainty=0.0, dopamine=0.0),
            1: dict(cost=0.0, uncertainty=0.0, dopamine=0.0),
        }
        self.assertAlmostEqual(
            runtime._action_probability(scores), 0.5, places=6
        )
        scores[1]["cost"] = 1.0
        self.assertLess(runtime._action_probability(scores), 0.01)
        scores[1]["cost"] = -1.0
        self.assertGreater(runtime._action_probability(scores), 0.99)
        scores[1]["cost"] = 0.0
        scores[0]["uncertainty"] = 1.0
        scores[1]["uncertainty"] = 1.0
        self.assertAlmostEqual(
            runtime._action_probability(scores), 0.5, places=6
        )

    def test_act_samples_both_actions_when_preference_is_flat(self):
        runtime = self.make_runtime()
        observed = {
            runtime.act(np.asarray([0.0]), explore=True)
            for _ in range(120)
        }
        self.assertEqual(observed, {0, 1})
        self.assertIn(0, runtime.action_nodes)
        self.assertIn(1, runtime.action_nodes)

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
        self.assertGreater(values[2], values[0])
        self.assertGreater(values[3], 0.9)
        snapshot = runtime.snapshot()
        self.assertGreater(snapshot["associations"], 0)
        self.assertGreater(snapshot["transitions"], 0)
        self.assertGreater(snapshot["memories"], 0)

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
        row = run_seed(dict(
            seed=100, steps=100, coverage=0.1, cap=100, max_nodes=200,
            birth_threshold=0.45, horizon=6, epochs=1, lag=3,
            component_kinds=(), component_min_count=8, max_components=0,
            component_weight=0.5, component_gating=False,
            memory_weight=0.1, association_rate=0.0,
            induce_features=True, precursor_labels=True, pain_horizon=0,
            voi_fraction=0.0, voi_target=3, feature_value_weight=0.02,
            eval_seeds=[120], eval_episodes=1,
        ))
        self.assertGreater(row["frames"], 0)
        self.assertGreaterEqual(row["graph"]["nodes"], 1)
        self.assertGreaterEqual(row["graph"]["action_edges"], 1)
        self.assertEqual(len(row["evals"]), 1)
        self.assertEqual(row["evals"][0]["eval_seed"], 120)
        self.assertEqual(row["evals"][0]["train_seed"], 100)

    def test_planner_trajectories_are_recorded_as_real_transitions(self):
        from substrate_nodes import collect_planner, fit
        world, agent = fit(
            100, steps=80, epochs=1, max_nodes=100, lag=3,
            induce_features=False,
        )
        rows, frames = collect_planner(
            100, world, agent, episodes=1, cap=20, horizon=2, lag=3,
            step_stride=1,
        )
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows), frames[0])
        for row in rows:
            self.assertEqual(row["history"].shape, (4,))
            self.assertEqual(row["signals"].shape, (2,))
            self.assertEqual(row["done"], row["terminal"])
            self.assertGreaterEqual(row["episode"], 50000)

    def test_fit_accepts_closed_loop_rounds(self):
        from substrate_nodes import fit
        world, agent = fit(
            100, steps=80, epochs=1, max_nodes=100, lag=3,
            induce_features=False, dagger_rounds=1, dagger_episodes=1,
            dagger_cap=20, dagger_offset=60000, dagger_horizon=2,
        )
        self.assertIsNotNone(world)
        self.assertGreaterEqual(agent.snapshot()["nodes"], 1)
        self.assertEqual(len(agent.dagger_log), 1)
        self.assertGreater(agent.dagger_log[0]["planner_steps"], 0)

    def test_pain_signal_is_learned_by_association(self):
        runtime = self.make_runtime(max_nodes=40)
        first = runtime._create_node(np.asarray([0.0]))
        second = runtime._create_node(np.asarray([1.0]))
        runtime._interact(first, second, pain=1.0, dopamine=0.0)
        self.assertIn(runtime.pain_node, runtime.graph.nodes)
        self.assertGreater(runtime._pain_association(second), 0.0)
        self.assertGreater(
            runtime.pain_signal_state(np.asarray([1.0])), 0.0
        )

    def test_memory_is_reinforced_and_retrievable_from_members(self):
        runtime = self.make_runtime(max_nodes=40)
        first = runtime._create_node(np.asarray([0.0]))
        second = runtime._create_node(np.asarray([1.0]))
        runtime._interact(first, second, pain=1.0, dopamine=0.0)
        memory_id = next(iter(runtime.memories_for([second])))
        self.assertIsNotNone(memory_id)
        memory = runtime.graph.nodes[memory_id]
        self.assertGreater(memory.trace, 0.0)
        self.assertGreater(memory.value, 0.0)
        found = runtime.memories_for([second])
        self.assertIn(memory_id, found)
        self.assertGreater(found[memory_id], 0.0)
        self.assertGreater(
            runtime.graph.edges[(memory_id, second, MEMBERSHIP)].weight,
            0.0,
        )
        activated = runtime.graph.recall([second], steps=1)
        self.assertGreater(activated[memory_id], 0.0)

    def test_low_support_memory_is_forgotten(self):
        runtime = self.make_runtime(max_nodes=40)
        first = runtime._create_node(np.asarray([0.0]))
        second = runtime._create_node(np.asarray([1.0]))
        runtime._interact(first, second, pain=1.0, dopamine=0.0)
        memory_id = next(iter(runtime.memories_for([second])))
        runtime.graph.nodes[memory_id].trace = 0.0
        runtime.graph.nodes[memory_id].count = 1
        runtime.graph.nodes[memory_id].value = 0.0
        runtime._forget_memories()
        self.assertNotIn(memory_id, runtime.graph.nodes)

    def test_memory_is_created_by_pain_signal(self):
        def rows(count, final_pain):
            result = []
            for index in range(count):
                terminal = index == count - 1 and final_pain
                pain = 1.0 if terminal else 0.0
                result.append(dict(
                    obs=np.asarray([float(index)]),
                    next_obs=np.asarray([float(index + 1)]),
                    history=np.asarray([0.0, 0.0, 0.0, 0.0]),
                    signals=np.asarray([1.0 - pain, pain]),
                    done=terminal, episode=0,
                ))
            return result

        steady = self.make_runtime(max_nodes=40)
        steady.learn_episode(rows(12, False))
        self.assertEqual(steady.snapshot()["memories"], 0)

        surprise = self.make_runtime(max_nodes=40)
        surprise.learn_episode(rows(12, True))
        self.assertGreaterEqual(surprise.snapshot()["memories"], 1)

    def test_future_pain_labels_window_and_truncation(self):
        from node_runtime import future_pain_labels

        rows = []
        # Episode 0: the entered state after row 3 is painful and terminal.
        for index, pain in enumerate([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]):
            rows.append(dict(
                obs=np.asarray([float(index)]),
                next_obs=np.asarray([float(index + 1)]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([1.0 - pain, pain]),
                done=index == 3, episode=0,
            ))
        # A second episode whose first entered state is painful; it must not
        # bleed into the last episode-0 label.
        for index, pain in enumerate([1.0, 0.0, 0.0]):
            rows.append(dict(
                obs=np.asarray([float(index)]),
                next_obs=np.asarray([float(index + 1)]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([1.0 - pain, pain]),
                done=False, episode=1,
            ))
        labels = future_pain_labels(rows, 3)
        self.assertEqual(labels[:6], [0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(labels[5], 0.0)

    def test_feature_pain_learns_the_precursor_label(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 0.2), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        # The initialization uses train counts, not the held-out confidence.
        self.assertAlmostEqual(
            runtime.graph.nodes[feature_id].value, 5.0 / 7.0, places=6
        )
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([1.0, 1.0]), done=True, episode=0,
            ),
        ]
        runtime.learn_episode(rows)
        # lag=1 → window 2: the label of row 0 is pain within 2 entered
        # states (1.0), and the feature is active on the *current* state.
        self.assertGreater(
            runtime.graph.nodes[feature_id].value, 5.0 / 7.0
        )

    def test_feature_action_stats_pair_with_the_acting_action(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([2.0]), next_obs=np.asarray([3.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([1.0, 1.0]), done=True, episode=0,
            ),
        ]
        runtime.learn_episode(rows)
        cells = runtime.feature_action_stats[feature_id]
        # The action acting on the transition s_m -> s_{m+1} is
        # history[lag]: row 0 acts with the pre-episode 0, row 1 acts with
        # the action issued at row 0 (1.0).  The feature is inactive on the
        # final state, so only two cells exist.  Hits carry the precursor
        # label P_2: row 0 sees no death within its window, row 1 sees the
        # terminal row 2.
        self.assertEqual(cells[0], [1, 0.0])
        self.assertEqual(cells[1], [1, 1.0])

    def test_feature_risk_depends_on_the_acting_action(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        prior_value = 5.0 / 7.0
        state_hold = np.asarray([0.0, 0.0])
        state_flap = np.asarray([0.0, 1.0])
        # Without data both pending actions fall back to the scalar value.
        plain = runtime._feature_risk_batch(
            np.vstack([state_hold, state_flap])
        )
        np.testing.assert_allclose(plain, prior_value)
        # With data the same state separates by the acting action.
        runtime.feature_action_stats[feature_id] = {
            0: [10, 0.0], 1: [10, 10.0],
        }
        separated = runtime._feature_risk_batch(
            np.vstack([state_hold, state_flap])
        )
        self.assertLess(separated[0], prior_value)
        self.assertGreater(separated[1], prior_value)
        self.assertGreater(separated[1], separated[0])

    def test_feature_windows_are_per_node(self):
        from node_features import FeatureRule, Primitive
        from node_graph import NodeGraph

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        short = FeatureRule(
            Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        long_rule = FeatureRule(
            Primitive(0, False, "<", 2.5), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        runtime.rules.rules = [short]
        runtime.install_feature_rules()
        short_id = runtime.feature_ids[0]
        runtime.rules_long = FeatureRuleBank()
        runtime.rules_long.rules = [long_rule]
        runtime.install_feature_rules([long_rule], horizon=3)
        long_id = runtime.feature_ids[1]
        self.assertIsNone(runtime.graph.nodes[short_id].pain_horizon)
        self.assertEqual(runtime.graph.nodes[long_id].pain_horizon, 3)
        restored = NodeGraph.from_state(runtime.graph.to_state())
        self.assertIsNone(restored.nodes[short_id].pain_horizon)
        self.assertEqual(restored.nodes[long_id].pain_horizon, 3)

        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([2.0]), next_obs=np.asarray([3.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([1.0, 1.0]), done=True, episode=0,
            ),
        ]
        runtime.learn_episode(rows)
        # The default (lag+1 = 2) window misses the death at the third
        # entered state for row 0, so the short root moves down; the 3-step
        # window sees it, so the long root moves up.
        self.assertLess(
            runtime.graph.nodes[short_id].value, 5.0 / 7.0
        )
        self.assertGreater(
            runtime.graph.nodes[long_id].value, 5.0 / 7.0
        )

    def test_prediction_calibration_uses_the_pre_update_readout(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        runtime.feature_action_stats[feature_id] = {1: [10, 10.0]}
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([1.0, 1.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
        ]
        runtime.learn_episode(rows)
        entry = runtime.feature_calibration[(feature_id, 1)]
        expected = (10.0 + 2.0 * (5.0 / 7.0)) / 12.0
        self.assertEqual(entry[0], 1)
        self.assertAlmostEqual(entry[1], expected, places=9)
        self.assertAlmostEqual(entry[2], 0.0, places=9)
        # The record happened before the update: the cell now includes the
        # benign transition.
        self.assertEqual(runtime.feature_action_stats[feature_id][1],
                         [11, 10.0])
        self.assertTrue(runtime.surprise_buffer)
        report = runtime.prediction_report()
        self.assertGreaterEqual(report["buffer_size"], 1)
        self.assertEqual(len(report["buffer"]), min(
            len(runtime.surprise_buffer), 8))

    def test_prediction_report_flags_miscalibrated_cells(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        # The readout kept predicting disaster; reality kept being benign.
        runtime.feature_calibration[(feature_id, 1)] = [20, 18.0, 4.0, 9.0]
        report = runtime.prediction_report()
        flagged = [
            item for item in report["cells"]
            if item["node"] == feature_id and item["miscalibrated"]
        ]
        self.assertEqual(len(flagged), 1)
        self.assertGreater(flagged[0]["pred"], 0.8)
        self.assertLess(flagged[0]["actual"], 0.3)

    def test_counterfactual_action_targets_the_deficit_cell(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        observation = np.asarray([0.0])
        # No cell data yet: both actions are deficient, action 0 first.
        self.assertEqual(runtime.counterfactual_action(observation), 0)
        # One cell rich, the other empty: the empty one is the target.
        runtime.feature_action_stats[feature_id] = {0: [10, 10.0]}
        self.assertEqual(runtime.counterfactual_action(observation), 1)
        # Both cells rich -> no deficit.
        runtime.feature_action_stats[feature_id] = {
            0: [10, 10.0], 1: [10, 10.0],
        }
        self.assertIsNone(runtime.counterfactual_action(observation))
        # A low prior makes the consequence too small to bother with.
        runtime.feature_action_stats[feature_id] = {0: [10, 10.0]}
        runtime.graph.nodes[feature_id].value = 0.3
        self.assertIsNone(runtime.counterfactual_action(observation))
        # The per-cell test budget stops repeated testing of one cell.
        runtime.graph.nodes[feature_id].value = 5.0 / 7.0
        runtime.voi_tests.clear()
        for _ in range(16):
            self.assertEqual(runtime.counterfactual_action(observation), 1)
        self.assertIsNone(runtime.counterfactual_action(observation))

    def test_collect_applies_the_counterfactual_budget(self):
        from substrate_generic import collect

        propensities = []
        transitions, _ = collect(
            100, 60, coverage=0.1, lag=3, guide=lambda obs: 1, budget=5,
            episode_start=7, propensities=propensities,
        )
        self.assertTrue(transitions)
        self.assertEqual(len(propensities), len(transitions))
        overrides = sum(1 for p in propensities if p == 1.0)
        self.assertEqual(overrides, 5)
        episodes = {row["episode"] for row in transitions}
        self.assertTrue(all(e >= 7 for e in episodes))

    def test_feature_value_learns_the_discounted_return(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40, gamma=0.5
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([2.0]), next_obs=np.asarray([3.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([0.0, 1.0]), done=True, episode=0,
            ),
        ]
        runtime.learn_episode(rows)
        # The feature is active on the first two states.  Monte-Carlo
        # dopamine returns, computed backward: the final transition earns 0
        # (died), the middle one 1, the first one 1 + gamma.
        cells = runtime.feature_value_stats[feature_id]
        self.assertEqual(cells[0], [1, 1.5])   # row 0: 1 + 0.5 * 1
        self.assertEqual(cells[1], [1, 1.0])   # row 1: acting = row0's action
        self.assertGreater(runtime.value_baseline, 0.0)
        # The readout separates the same state by the acting action; both
        # reads shrink toward the running global baseline.
        hold = np.asarray([0.0, 0.0])
        flap = np.asarray([0.0, 1.0])
        values = runtime.feature_value_batch(np.vstack([hold, flap]))
        prior = runtime.feature_value_prior
        baseline = runtime.value_baseline
        self.assertAlmostEqual(
            values[0], (1.5 + prior * baseline) / (1 + prior), places=9
        )
        self.assertAlmostEqual(
            values[1], (1.0 + prior * baseline) / (1 + prior), places=9
        )
        self.assertGreater(values[0], values[1])

    def test_components_attach_to_runtime_nodes(self):
        runtime = self.make_runtime()
        node = runtime._create_node(np.asarray([0.0]))
        component_id = runtime.bind_component(
            node, "difference", smoothing=1.0
        )
        runtime.reset()
        runtime.act(np.asarray([0.0]), explore=False)
        runtime.observe(
            np.asarray([1.0]), 0, [1.0, 0.0], done=False,
        )
        plus = runtime.components.sign_nodes[component_id][1]
        self.assertIn(plus, runtime.graph.nodes)
        self.assertGreater(runtime.graph.nodes[plus].dopamine, 0.0)

    def test_runtime_learns_component_birth_from_valence_variance(self):
        runtime = self.make_runtime(
            component_kinds=("difference",), component_min_count=1,
            max_components=10,
        )
        high = runtime._create_node(np.asarray([0.0]))
        low_one = runtime._create_node(np.asarray([10.0]))
        low_two = runtime._create_node(np.asarray([20.0]))
        for _ in range(4):
            runtime._note_pain(high, 1.0)
            runtime._note_pain(high, 0.0)
            runtime._note_pain(low_one, 0.0)
            runtime._note_pain(low_two, 0.0)
        runtime._maybe_bind_components(high)
        self.assertIn(high, runtime._component_bound)
        self.assertIn("%s:difference" % high, runtime.components.specs)
        runtime._component_step(high)
        self.assertGreaterEqual(
            len(runtime.components.active_sign_nodes(high)), 1
        )

    def test_component_gating_uses_the_signature(self):
        runtime = self.make_runtime(
            component_kinds=("difference",), component_min_count=1,
            max_components=1, component_gating=True,
        )
        runtime.reset()
        node = runtime._create_node(np.asarray([0.0]))
        runtime.bind_component(node, "difference", smoothing=1.0)
        runtime._component_step(node)
        signature = runtime._signature(node)
        runtime.gated_edges[(node, signature, 0)] = {
            "safe": dict(count=10, q_pain=0.1, q_dopamine=1.0)
        }
        runtime.gated_edges[(node, signature, 1)] = {
            "danger": dict(count=10, q_pain=0.9, q_dopamine=1.0)
        }
        _node, scores = runtime.decision(np.asarray([0.0]))
        self.assertLess(scores[0]["pain"], scores[1]["pain"])
        self.assertLess(scores[0]["cost"], scores[1]["cost"])

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
        # The feature grammar must stay task-neutral too: its comments may
        # not lean on the benchmark world's shapes.
        feature_text = (root / "node_features.py").read_text(
            encoding="utf-8"
        )
        for token in ("pipe", "flap", "gap", "boundary"):
            self.assertNotIn(
                token, feature_text, "%s in node_features.py" % token
            )
        adapter = (root / "substrate_nodes.py").read_text(encoding="utf-8")
        for token in (
            "BIRD_X", "GAP", "pipes", "target_gap", "near_pipe",
            "crossing", "TerminalRiskHead",
        ):
            self.assertNotIn(token, adapter, "%s in substrate_nodes.py" % token)


class PlanScoreTests(unittest.TestCase):
    def test_plan_score_matches_search_selection(self):
        from latent_world import latent
        from substrate_nodes import fit, plan_score, policy_result

        world, agent = fit(100, steps=300, epochs=1, max_nodes=200, lag=3)
        env = latent[3](100)
        prefix = np.zeros(3, dtype=float)
        result = policy_result(env, world, agent, prefix, horizon=1, lag=3)
        scores = [
            plan_score(
                env.obs(), world, agent, prefix, [float(action)], 1, lag=3
            )
            for action in (0, 1)
        ]
        best = min(range(2), key=lambda index: scores[index])
        self.assertEqual(int(result.actions[0]), best)
        self.assertAlmostEqual(result.score[0], scores[best][0], places=6)
        self.assertAlmostEqual(result.score[1], scores[best][1], places=6)

    def test_plan_score_matches_search_at_the_lag_point(self):
        """The lag point is the first state a plan action can affect."""
        from latent_world import latent
        from substrate_nodes import fit, plan_score, policy_result

        world, agent = fit(100, steps=300, epochs=1, max_nodes=200, lag=3)
        env = latent[3](100)
        prefix = np.zeros(3, dtype=float)
        result = policy_result(
            env, world, agent, prefix, horizon=5, lag=3,
            step_stride=2, stride_includes_lag=True,
        )
        plans = [
            plan for plan in np.ndindex(*(2,) * 5)
        ]
        scores = [
            plan_score(
                env.obs(), world, agent, prefix, plan, 5, lag=3,
                step_stride=2, stride_includes_lag=True,
            )
            for plan in plans
        ]
        best = min(range(len(plans)), key=lambda index: scores[index])
        np.testing.assert_array_equal(result.actions, plans[best])
        self.assertAlmostEqual(result.score[0], scores[best][0], places=6)
        self.assertAlmostEqual(result.score[1], scores[best][1], places=6)


if __name__ == "__main__":
    unittest.main()
