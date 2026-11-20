import types
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

    def test_planner_collection_honours_a_frame_budget(self):
        from substrate_nodes import collect_planner, fit
        world, agent = fit(
            100, steps=80, epochs=1, max_nodes=100, lag=3,
            induce_features=False,
        )
        rows, frames = collect_planner(
            100, world, agent, episodes=40, cap=10, horizon=2, lag=3,
            step_stride=1, max_frames=20,
        )
        self.assertGreaterEqual(sum(frames), 20)
        # Each episode contributes at most `cap` frames, so the budget is
        # reached well before the 40-reset hard limit.
        self.assertLessEqual(len(frames), 20)
        self.assertEqual(len(rows), sum(frames))

    def test_cell_q_action_decides_without_search(self):
        from latent_world import latent
        from substrate_nodes import cell_q_action, fit
        world, agent = fit(
            100, steps=80, epochs=1, max_nodes=100, lag=3,
            induce_features=False,
        )
        env = latent[3](100)
        prefix = np.zeros(3, dtype=float)
        action = cell_q_action(env, world, agent, prefix, lag=3)
        self.assertIn(action, (0, 1))
        # The pending prefix is untouched by a no-search decision.
        self.assertEqual(list(env.pending()), [0, 0, 0])

    def test_k_step_value_target_matches_the_truncated_return(self):
        from node_runtime import NodeRuntime
        stub = types.SimpleNamespace(gamma=0.5, value_horizon=2)
        steps = [
            (None, 0, "n0", 0.0, 1.0, False, (), (), 0, ()),
            (None, 0, "n1", 0.0, 2.0, False, (), (), 0, ()),
            (None, 0, "n2", 0.0, 4.0, True, (), (), 0, ()),
        ]
        snapshot = {"n1": 8.0}
        # position 0: 1 + 0.5*2 + 0.25*V(n1) = 4.0; the state two steps
        # ahead of step 0 is the next node of step 1.
        self.assertAlmostEqual(
            NodeRuntime._k_step_value_target(stub, steps, 0, snapshot), 4.0
        )
        # position 1: 2 + 0.5*4, no bootstrap (episode ends inside horizon)
        self.assertAlmostEqual(
            NodeRuntime._k_step_value_target(stub, steps, 1, snapshot), 4.0
        )
        # position 2: terminal step, reward only
        self.assertAlmostEqual(
            NodeRuntime._k_step_value_target(stub, steps, 2, snapshot), 4.0
        )

    def test_fit_accepts_a_value_horizon(self):
        from substrate_nodes import fit
        world, agent = fit(
            100, steps=80, epochs=1, max_nodes=100, lag=3,
            induce_features=False, value_horizon=4,
        )
        self.assertEqual(agent.value_horizon, 4)
        self.assertGreaterEqual(agent.snapshot()["nodes"], 1)

    def test_ood_trust_shrinks_readouts_toward_the_baseline(self):
        runtime = self.make_runtime(ood_trust_scale=0.5)
        first = runtime._create_node(np.asarray([0.0]))
        second = runtime._create_node(np.asarray([1.0]))
        runtime.graph.nodes[first].value = 1.0
        runtime.graph.nodes[second].value = 0.0
        states = np.asarray([[4.0]])
        indices, weights, nearest_min = runtime._soft_weights_batch(states)
        self.assertIsNotNone(indices)
        # Distances are squared, matching the soft-weight convention.
        self.assertAlmostEqual(float(nearest_min[0]), 9.0, places=6)
        # Without trust shrinkage the nearest prototype dominates.
        runtime.ood_trust_scale = 0.0
        pain_off, _ = runtime.valence_state_batch(states)
        self.assertLess(float(pain_off[0]), 0.01)
        # With shrinkage the unsupported query reports the population mean.
        runtime.ood_trust_scale = 0.5
        pain_on, _ = runtime.valence_state_batch(states)
        self.assertAlmostEqual(float(pain_on[0]), 0.5, places=6)

    def test_obs_trust_downweights_the_observation_channel(self):
        runtime = self.make_runtime()
        first = runtime._create_node(np.asarray([0.0]))
        second = runtime._create_node(np.asarray([1.0]))
        runtime.graph.nodes[first].value = 1.0
        runtime.graph.nodes[second].value = 0.0
        states = np.asarray([[0.0]])
        pain_full, _ = runtime.valence_state_batch(states)
        runtime.obs_trust = 0.5
        pain_half, _ = runtime.valence_state_batch(states)
        self.assertAlmostEqual(
            float(pain_half[0]), 0.5 * float(pain_full[0]), places=6
        )

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

    def test_future_return_labels_match_the_mc_backward_pass(self):
        from node_runtime import future_return_labels

        rows = []
        # Episode 0: dopamine 1, 1, 1 then a terminal step earning 0; the
        # discounted returns run 1.75, 1.5, 1.0, 0.0 with gamma = 0.5.
        for index in range(4):
            rows.append(dict(
                obs=np.asarray([float(index)]),
                next_obs=np.asarray([float(index + 1)]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([1.0, 0.0]) if index < 3
                else np.asarray([0.0, 1.0]),
                done=index == 3, episode=0,
            ))
        # A second episode whose stream just stops (truncated, no terminal):
        # its only step's return is its own dopamine.
        rows.append(dict(
            obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
            history=np.asarray([0.0, 0.0]),
            signals=np.asarray([2.0, 0.0]), done=False, episode=1,
        ))
        labels = future_return_labels(rows, 0.5)
        self.assertEqual(labels[:4], [1.75, 1.5, 1.0, 0.0])
        self.assertEqual(labels[4], 2.0)

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

    def test_evidence_batch_counts_the_active_cells(self):
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
        # No learned cells yet: reported evidence is zero, not a prior.
        states = np.asarray([[0.0, 0.0], [0.0, 1.0], [5.0, 0.0]])
        np.testing.assert_allclose(runtime.evidence_batch(states), 0.0)
        runtime.feature_action_stats[feature_id] = {
            0: [10, 0.0], 1: [3, 0.0],
        }
        # Either channel supports the state's read: the value cell's count
        # wins where it is larger, the risk cell's where the value one is
        # absent.
        runtime.feature_value_stats[feature_id] = {
            0: (20, 0.0, {1}),
        }
        np.testing.assert_allclose(
            runtime.evidence_batch(states), [20.0, 3.0, 0.0]
        )

    def test_curiosity_term_is_a_potential_difference(self):
        from substrate_nodes import _curiosity_potential, _curiosity_term

        # The potential falls as evidence grows; it stays bounded by 1.
        potential = _curiosity_potential(np.asarray([0.0, 3.0, 99.0]))
        self.assertAlmostEqual(float(potential[0]), 1.0, places=9)
        self.assertLess(float(potential[1]), float(potential[0]))
        self.assertGreater(float(potential[2]), 0.0)
        # A less-supported endpoint lowers the cost (negative term); the
        # same potential at both ends cancels exactly, so no free reward.
        term = _curiosity_term(0.2, np.asarray([0.8]), 1.0, 0.99, 1)
        self.assertLess(float(term[0]), 0.0)
        self.assertAlmostEqual(
            float(_curiosity_term(0.5, np.asarray([0.5]), 1.0, 1.0, 1)[0]),
            0.0, places=9,
        )
        self.assertAlmostEqual(
            float(_curiosity_term(0.2, np.asarray([0.8]), 0.0, 0.99, 1)[0]),
            0.0, places=9,
        )

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

    def test_counterfactual_guide_targets_decision_critical_states(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40,
            voi_min_prior=0.05,
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        observation = np.asarray([0.0])
        # The readout considers the state safe: the threat read (the
        # minimum over both actions' reads) sits below the gate even
        # though the minority cell is starving - a test spent here buys
        # no decision information.  The pre-v2 guide would have tested.
        runtime.graph.nodes[feature_id].value = 0.1
        runtime.feature_action_stats[feature_id] = {
            0: [2, 0.0], 1: [2, 2.0],
        }
        self.assertIsNone(runtime.counterfactual_action(observation))
        # At a dangerous state (the calibrated threat read at 0.5) the
        # same starving cell becomes the target.
        states = np.asarray([[0.0]] * 8)
        pending = np.zeros((8, 1), dtype=float)
        labels = np.asarray([1.0, 0.0] * 4)
        runtime.fit_readout_calibration(states, pending, labels)
        self.assertEqual(runtime.counterfactual_action(observation), 0)
        self.assertEqual(
            runtime.voi_tests[(feature_id, 0)], 1
        )

    def test_counterfactual_starvation_is_relative(self):
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
        runtime.graph.nodes[feature_id].value = 5.0 / 7.0
        # A well-fed cell that still holds less than a quarter of the
        # other action's evidence is starved: the absolute floor alone
        # would let the guide go silent at any real data volume.
        runtime.feature_action_stats[feature_id] = {
            0: [200, 200.0], 1: [30, 30.0],
        }
        self.assertEqual(runtime.counterfactual_action(observation), 1)
        # Balanced cells stay untouched even with a large deficit-free
        # absolute count.
        runtime.voi_tests.clear()
        runtime.feature_action_stats[feature_id] = {
            0: [120, 120.0], 1: [100, 100.0],
        }
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
        self.assertEqual(cells[0][:2], [1, 1.5])   # row 0: 1 + 0.5 * 1
        self.assertEqual(cells[1][:2], [1, 1.0])   # row 1: acting = row0's action
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

    def test_value_channel_installs_and_reads_separately(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40, gamma=0.5
        )
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        ids = runtime.install_feature_rules(channel="value")
        # The value bank owns its roots; the risk-side bookkeeping and the
        # pain-prior initialization stay untouched.
        self.assertEqual(runtime.value_feature_ids, ids)
        self.assertEqual(runtime.feature_ids, [])
        self.assertEqual(runtime.graph.nodes[ids[0]].count, 0)
        # The union readout covers the dedicated root even without any
        # risk-bank roots, and the pain readout stays at zero for them.
        runtime._update_feature_value(ids, 0, 2.0)
        hold = np.asarray([0.0, 0.0])
        flap = np.asarray([0.0, 1.0])
        values = runtime.feature_value_batch(np.vstack([hold, flap]))
        prior = runtime.feature_value_prior
        baseline = runtime.value_baseline
        self.assertAlmostEqual(
            values[0], (2.0 + prior * baseline) / (1.0 + prior), places=9
        )
        self.assertAlmostEqual(values[1], baseline, places=9)
        # No pain-side statistics are accumulated for a value root.
        self.assertEqual(list(runtime.feature_action_stats), [])

    def test_subgraph_cells_separate_or_branches(self):
        from node_features import Composite, FeatureRule, Primitive

        def build(subgraph):
            runtime = self.make_runtime(
                lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40,
                gamma=0.5, subgraph_cells=subgraph,
            )
            runtime.rules.rules = [FeatureRule(
                Composite("or", (
                    Primitive(0, True, "<", 1.0),
                    Primitive(0, True, ">", 3.0),
                )),
                5, 3, 0.6, 2, 1, 0.5, 0.7,
            )]
            runtime.install_feature_rules()
            return runtime

        on = build(True)
        off = build(False)
        root = on.feature_ids[0]
        children = on._feature_children[root]
        self.assertEqual(len(children), 2)
        rows = []
        # Episode 0: low states earn dopamine, no pain anywhere.
        for index, obs in enumerate([0.0, 0.5]):
            rows.append(dict(
                obs=np.asarray([obs]), next_obs=np.asarray([obs + 0.1]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([1.0, 0.0]), done=index == 1,
                episode=0,
            ))
        # Episode 1: high states hurt.
        for index, obs in enumerate([4.0, 4.5]):
            rows.append(dict(
                obs=np.asarray([obs]), next_obs=np.asarray([obs - 0.1]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([0.0, 1.0]), done=index == 1,
                episode=1,
            ))
        for runtime in (on, off):
            for episode in (0, 1):
                runtime.learn_episode(
                    [row for row in rows if row["episode"] == episode]
                )
        low_state = np.asarray([[0.0, 0.0]])   # encoded: obs + pending 0
        high_state = np.asarray([[4.0, 0.0]])
        encoded = np.vstack([low_state, high_state])
        # Pain side: the low branch has the fewest observations for this
        # action, so its read wins over the root's pooled cell - and the
        # low branch has seen only painless states.
        risk_on = on.feature_risk_batch(encoded)
        risk_off = off.feature_risk_batch(encoded)
        self.assertAlmostEqual(risk_on[0], 0.0, places=9)
        self.assertGreater(risk_on[1], 0.5)
        self.assertAlmostEqual(risk_off[0], risk_off[1], places=9)
        # Value side: the branch reads replace the root's pooled average.
        values_on = on.feature_value_batch(encoded)
        values_off = off.feature_value_batch(encoded)
        prior = on.feature_value_prior
        baseline = on.value_baseline
        # Low branch: count 2, sum 2.5.  High branch: count 2, sum 0.
        self.assertAlmostEqual(
            values_on[0], (2.5 + prior * baseline) / (2.0 + prior), places=9
        )
        self.assertAlmostEqual(
            values_on[1], (0.0 + prior * baseline) / (2.0 + prior), places=9
        )
        # The root-only read pools both directions into one number.
        self.assertAlmostEqual(values_off[0], values_off[1], places=9)
        self.assertAlmostEqual(
            values_off[0], (2.5 + prior * baseline) / (4.0 + prior), places=9
        )

    def test_subgraph_branches_inherit_the_root_horizon(self):
        from node_features import Composite, FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40, gamma=0.5,
            subgraph_cells=True,
        )
        rule = FeatureRule(
            Composite("and", (
                Primitive(0, False, "<", 5.0),
                Primitive(0, False, ">", -5.0),
            )),
            5, 3, 0.6, 2, 1, 0.5, 0.7,
        )
        runtime.install_feature_rules([rule], horizon=8)
        root = runtime.feature_ids[0]
        child = runtime._feature_children[root][0]
        rows = []
        # One episode of five rows whose only pain lands on the terminal
        # state.  With the root's window (8) every row's precursor label is
        # 1; with the default window (lag + 1 = 2) only the last two rows
        # would see it.
        for index in range(5):
            pain = 1.0 if index == 4 else 0.0
            rows.append(dict(
                obs=np.asarray([float(index)]),
                next_obs=np.asarray([float(index) + 0.1]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([1.0 - pain, pain]),
                done=index == 4, episode=0,
            ))
        runtime.learn_episode(rows)
        cell = runtime.feature_action_stats.get(child, {}).get(0)
        self.assertEqual(cell, [5, 5.0])
        root_cell = runtime.feature_action_stats.get(root, {}).get(0)
        self.assertEqual(root_cell, [5, 5.0])

    def test_predicate_readouts_drop_the_prototype_channel(self):
        from node_features import FeatureRule, Primitive

        rows = []
        # One episode whose last state is painful; the prototypes at obs
        # 0, 1, 2 learn its association.
        for index, pain in enumerate([0.0, 0.0, 1.0]):
            rows.append(dict(
                obs=np.asarray([float(index)]),
                next_obs=np.asarray([float(index) + 1.0]),
                history=np.asarray([0.0]),
                signals=np.asarray([1.0 - pain, pain]),
                done=index == 2, episode=0,
            ))
        prototype_mode = self.make_runtime()
        prototype_mode.learn_episode(rows)
        learned = prototype_mode.pain_signal_batch(
            np.asarray([[2.0]])
        )[0]
        self.assertGreater(learned, 0.0)

        predicate_mode = self.make_runtime(predicate_readouts=True)
        predicate_mode.learn_episode(rows)
        # Without predicates the honest read is zero, not an extrapolated
        # prototype statistic.
        self.assertAlmostEqual(
            float(predicate_mode.pain_signal_batch(np.asarray([[2.0]]))[0]),
            0.0, places=9,
        )

        # With a predicate installed, its cells carry the read again - and
        # states outside every predicate stay at zero.
        predicate_mode.rules.rules = [
            FeatureRule(
                Primitive(0, True, ">", 1.5), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        predicate_mode.install_feature_rules()
        predicate_mode.learn_episode(rows)
        risk = predicate_mode.pain_signal_batch(
            np.asarray([[2.0], [0.0]])
        )
        self.assertGreater(float(risk[0]), 0.0)
        self.assertAlmostEqual(float(risk[1]), 0.0, places=9)

    def test_pain_contrast_centers_on_the_action_mean(self):
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
        # Action 0 survives the band far more often than action 1.
        runtime._update_feature_action([feature_id], 0, 0.0)
        runtime._update_feature_action([feature_id], 0, 0.0)
        runtime._update_feature_action([feature_id], 1, 1.0)
        hold = np.asarray([0.0, 0.0])    # encoded: obs + pending 0
        flap = np.asarray([0.0, 1.0])
        contrast = runtime.pain_contrast_batch(np.vstack([hold, flap]))
        prior = runtime.feature_action_prior
        base = runtime.graph.nodes[feature_id].value
        read0 = (0.0 + prior * base) / (2.0 + prior)
        read1 = (1.0 + prior * base) / (1.0 + prior)
        mean_read = (read0 + read1) / 2.0
        # The absolute lethality cancels: the reads are centered, so the
        # two actions land symmetrically around zero regardless of base.
        self.assertAlmostEqual(contrast[0], read0 - mean_read, places=9)
        self.assertAlmostEqual(contrast[1], read1 - mean_read, places=9)
        self.assertLess(abs(contrast[0]), abs(read0))
        self.assertGreater(abs(contrast[1] - contrast[0]), 0.0)
        # A state outside every predicate reads zero, not a extrapolation.
        outside = np.asarray([5.0, 0.0])
        self.assertAlmostEqual(
            float(runtime.pain_contrast_batch(outside[None, :])[0]),
            0.0, places=9,
        )

    def test_feature_value_batch_reports_its_evidence(self):
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
        hold = np.asarray([0.0, 0.0])    # encoded: obs + pending 0
        flap = np.asarray([0.0, 1.0])
        # Both actions carry value evidence; action 0 has twice the count.
        runtime._update_feature_value([feature_id], 0, 2.0)
        runtime._update_feature_value([feature_id], 0, 3.0)
        runtime._update_feature_value([feature_id], 1, 1.0)
        encoded = np.vstack([hold, flap])
        values, support = runtime.feature_value_batch(
            encoded, return_support=True
        )
        self.assertEqual(support[0], 2.0)
        self.assertEqual(support[1], 1.0)
        prior = runtime.feature_value_prior
        baseline = runtime.value_baseline
        self.assertAlmostEqual(
            values[0], (5.0 + prior * baseline) / (2.0 + prior), places=9
        )
        # D4: with the trust half-count K, the thin-evidence action's
        # endpoint contribution is scaled down relative to the rich one.
        trust0 = 2.0 / (2.0 + runtime.value_trust_samples)
        trust1 = 1.0 / (1.0 + runtime.value_trust_samples)
        self.assertGreater(trust0, trust1)

    def test_value_cells_track_episode_diversity(self):
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
        hold = np.asarray([0.0, 0.0])
        # Three updates: two from one episode, one from another - the raw
        # count is three but the evidence spans two world instances.
        runtime._update_feature_value([feature_id], 0, 2.0, episode_id=7)
        runtime._update_feature_value([feature_id], 0, 3.0, episode_id=7)
        runtime._update_feature_value([feature_id], 0, 1.0, episode_id=9)
        values, support = runtime.feature_value_batch(
            hold[None, :], return_support=True
        )
        prior = runtime.feature_value_prior
        baseline = runtime.value_baseline
        self.assertAlmostEqual(
            values[0],
            (6.0 + prior * baseline) / (3.0 + prior),
            places=9,
        )
        # Trust counts distinct episodes, not repetitions.
        self.assertEqual(support[0], 2.0)
        cell = runtime.feature_value_stats[feature_id][0]
        self.assertEqual(cell[0], 3)
        self.assertEqual(sorted(cell[2]), [7, 9])

    def test_value_dispersion_labels_cover_both_extremes(self):
        from substrate_nodes import value_dispersion_labels

        returns = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                              6.0, 7.0, 8.0, 9.0], dtype=float)
        labels = value_dispersion_labels(returns, 0.8)
        # Top 20% (returns > 7.2) and bottom 20% (returns < 1.8).
        self.assertEqual(labels.tolist(),
                         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    def test_pain_grading_splits_threat_from_controllable_excess(self):
        from node_features import Composite, FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40, gamma=0.5
        )
        # A short-window rule whose two actions differ a lot, and a
        # long-window rule that must be excluded from the graded read.
        short_rule = FeatureRule(
            Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        long_rule = FeatureRule(
            Primitive(0, True, ">", 0.5), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        runtime.rules.rules = [short_rule, long_rule]
        runtime.install_feature_rules([short_rule])
        runtime.install_feature_rules([long_rule], horizon=8)
        feature_id = runtime.feature_ids[0]
        long_id = runtime.feature_ids[1]
        runtime._update_feature_action([feature_id], 0, 0.0)
        runtime._update_feature_action([feature_id], 0, 0.0)
        runtime._update_feature_action([feature_id], 1, 1.0)
        # The long root must not contaminate the graded read.
        runtime._update_feature_action([long_id], 0, 1.0)
        runtime._update_feature_action([long_id], 1, 1.0)
        hold = np.asarray([1.0, 0.0])
        flap = np.asarray([1.0, 1.0])
        threat, excess = runtime.pain_graded_batch(
            np.vstack([hold, flap])
        )
        prior = runtime.feature_action_prior
        base = runtime.graph.nodes[feature_id].value
        read0 = (0.0 + prior * base) / (2.0 + prior)
        read1 = (1.0 + prior * base) / (1.0 + prior)
        # Threat is the unavoidable share: the better action's read,
        # identical for both pendings.
        self.assertAlmostEqual(threat[0], read0, places=9)
        self.assertAlmostEqual(threat[1], read0, places=9)
        # Excess is what the plan's own action adds on top: zero for the
        # better action, the full contrast for the worse one.
        self.assertAlmostEqual(excess[0], 0.0, places=9)
        self.assertAlmostEqual(excess[1], read1 - read0, places=9)

    def test_readout_calibration_maps_to_empirical_rates(self):
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
        # Prime the cell so every read lands in one high raw bin: the
        # raw read is inflated relative to the labels fed to the fit.
        runtime._update_feature_action([feature_id], 0, 1.0)
        runtime._update_feature_action([feature_id], 0, 1.0)
        hold = np.asarray([1.0, 0.0])
        flap = np.asarray([1.0, 1.0])
        # Labels: half the covered states painful, half not - the
        # calibrated read must land near 0.5 even though the raw read
        # sits at the top of its bin.
        states = np.asarray([[1.0], [1.0], [1.0], [1.0]])
        pending = np.asarray([[0.0], [0.0], [0.0], [0.0]])
        labels = np.asarray([1.0, 1.0, 0.0, 0.0])
        runtime.fit_readout_calibration(
            np.asarray([[1.0], [1.0], [1.0], [1.0]]), pending, labels
        )
        threat, _excess = runtime.pain_graded_batch(
            np.vstack([hold, flap])
        )
        for value in threat:
            self.assertLess(value, 0.95)
            self.assertGreater(value, 0.05)

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

    def test_offline_pending_matches_the_runtime_encoding(self):
        from substrate_nodes import _encoded_pending

        # history is newest-first: [a_t, a_{t-1}, a_{t-2}, a_{t-3}].  The
        # runtime encodes this row's state with the actions issued
        # *before* it, oldest first: [a_{t-3}, a_{t-2}, a_{t-1}].
        rows = [dict(history=np.asarray([3.0, 2.0, 1.0, 0.0]))]
        np.testing.assert_allclose(
            _encoded_pending(rows, 3), [[0.0, 1.0, 2.0]]
        )
        # A short history degrades to what exists, zero-padded on the left.
        rows = [dict(history=np.asarray([2.0, 1.0]))]
        np.testing.assert_allclose(
            _encoded_pending(rows, 3), [[0.0, 0.0, 1.0]]
        )

    def test_graded_read_includes_tagged_short_window_roots(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        rule = FeatureRule(
            Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        runtime.install_feature_rules([rule], horizon=2)
        feature_id = runtime.feature_ids[0]
        runtime._update_feature_action([feature_id], 0, 0.0)
        runtime._update_feature_action([feature_id], 0, 0.0)
        runtime._update_feature_action([feature_id], 1, 1.0)
        hold = np.asarray([1.0, 0.0])
        worse = np.asarray([1.0, 1.0])
        threat, excess = runtime.pain_graded_batch(np.vstack([hold, worse]))
        prior = runtime.feature_action_prior
        base = runtime.graph.nodes[feature_id].value
        read0 = (0.0 + prior * base) / (2.0 + prior)
        read1 = (1.0 + prior * base) / (1.0 + prior)
        # The tag carries the root's own window (lag + 1 = the governable
        # span), so the root must be read: excluding it by tag presence
        # would mute every re-induced short-window root.
        self.assertAlmostEqual(threat[0], read0, places=9)
        self.assertAlmostEqual(threat[1], read0, places=9)
        self.assertAlmostEqual(excess[1], read1 - read0, places=9)

    def test_association_channel_reports_a_rate_not_an_envelope(self):
        runtime = self.make_runtime(
            lag=0, scales=np.asarray([1.0]), max_nodes=40
        )
        node = runtime._create_node(np.asarray([0.0]))
        # Four self-loop transitions, one painful: the read is the shrunk
        # rate, not the monotone envelope 1.0 a pain-only channel would
        # report.  Self-loops are visits and must be counted.
        for pain in (1.0, 0.0, 0.0, 0.0):
            runtime.reset()
            runtime.learn_episode([dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([0.0]),
                history=np.asarray([0.0]),
                signals=np.asarray([1.0, pain]), done=False, episode=0,
            )])
        stats = runtime._association_stats[node]
        self.assertEqual(stats, [4, 1.0])
        base = runtime._association_base()
        self.assertAlmostEqual(base, 0.25, places=9)
        prior = runtime.association_prior
        self.assertAlmostEqual(
            runtime._pain_association(node),
            (1.0 + prior * 0.25) / (4.0 + prior),
            places=9,
        )
        # Clean transitions through the same node pull the read down:
        # negative evidence is information.  The prior keeps a floor
        # under a single pain event, but the rate must decay toward the
        # base rate instead of pinning at the envelope.
        for _ in range(8):
            runtime.reset()
            runtime.learn_episode([dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([0.0]),
                history=np.asarray([0.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=1,
            )])
        decayed = runtime._pain_association(node)
        self.assertLess(decayed, 0.15)
        self.assertLess(
            decayed, (1.0 + prior * 0.25) / (4.0 + prior)
        )
        # A node with no samples reads the global base rate, not zero.
        fresh = runtime._create_node(np.asarray([9.0]))
        self.assertAlmostEqual(
            runtime._pain_association(fresh), runtime._association_base(),
            places=9,
        )

    def test_memory_risk_reads_the_live_rate(self):
        runtime = self.make_runtime(
            lag=0, scales=np.asarray([1.0]), max_nodes=40
        )
        runtime._create_node(np.asarray([0.0]))
        runtime._create_node(np.asarray([3.0]))
        # A pain event creates a memory indexed to the transition.
        runtime.reset()
        runtime.learn_episode([dict(
            obs=np.asarray([0.0]), next_obs=np.asarray([3.0]),
            history=np.asarray([0.0]),
            signals=np.asarray([1.0, 1.0]), done=False, episode=0,
        )])
        memory_ids = [
            node_id for node_id, node in runtime.graph.nodes.items()
            if node.kind == "memory"
        ]
        self.assertEqual(len(memory_ids), 1)
        memory_id = memory_ids[0]
        created = runtime._pain_association(memory_id)
        # The creation snapshot stays frozen, but the channel's read is
        # the live rate: clean transitions through the same region pull
        # it down below the creation-time value.
        for _ in range(8):
            runtime.reset()
            runtime.learn_episode([dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([3.0]),
                history=np.asarray([0.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=1,
            )])
        self.assertLess(runtime._pain_association(memory_id), created)

    def test_action_edges_key_on_the_governing_action(self):
        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([1.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([1.0, 1.0]), done=True, episode=0,
            ),
        ]
        runtime.learn_episode(rows)
        keyed = {}
        for source, _target, action in runtime.graph.action_edges:
            keyed[float(runtime.graph.nodes[source].features[0])] = action
        # Row 0's outcome is governed by history[lag]=0, row 1's by 1 -
        # keying by the just-issued action would attribute each outcome
        # to the other action.
        self.assertEqual(keyed, {0.0: 0, 1.0: 1})

    def test_pain_anchor_is_a_cumulative_mean(self):
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
        node = runtime.graph.nodes[feature_id]
        node.value = 0.0
        node.count = 0
        for _ in range(10):
            runtime._update_feature_pain([feature_id], 0.4)
        # A 1/n mean converges to the label rate; a constant-rate EMA
        # would stall short of it forever and drag every thin cell's
        # read along.
        self.assertAlmostEqual(node.value, 0.4, places=9)

    def test_value_baseline_is_a_cumulative_mean(self):
        runtime = self.make_runtime(max_nodes=40)
        for value in (2.0, 4.0, 6.0):
            runtime._update_feature_value([], 0, value)
        self.assertAlmostEqual(runtime.value_baseline, 4.0, places=9)

    def test_feature_value_support_reports_no_data_without_roots(self):
        runtime = self.make_runtime(max_nodes=40)
        value, support = runtime.feature_value_batch(
            np.asarray([[0.0], [1.0], [2.0]]), return_support=True
        )
        np.testing.assert_allclose(value, runtime.value_baseline)
        # No roots means no evidence: support must read zero, not a bare
        # array that would unpack element-wise at the caller.
        np.testing.assert_allclose(support, 0.0)

    def test_calibration_fits_the_per_read_granularity(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        hot_rule = FeatureRule(
            Primitive(0, True, "<", 1.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        cold_rule = FeatureRule(
            Primitive(0, True, ">", 3.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        runtime.install_feature_rules([hot_rule], horizon=2)
        runtime.install_feature_rules([cold_rule], horizon=2)
        hot_id, cold_id = runtime.feature_ids
        runtime.graph.nodes[hot_id].value = 0.9
        runtime.graph.nodes[cold_id].value = 0.1
        runtime._update_feature_action([hot_id], 0, 1.0)
        runtime._update_feature_action([hot_id], 0, 1.0)
        runtime._update_feature_action([cold_id], 0, 0.0)
        runtime._update_feature_action([cold_id], 0, 0.0)
        # Twelve states under the hot root (all painful) and twelve under
        # the cold root (all clean): the per-read pairs are
        # (0.95, 1.0) and (0.05, 0.0).
        states = np.asarray([[0.5]] * 12 + [[4.0]] * 12)
        pending = np.zeros((24, 1), dtype=float)
        labels = np.asarray([1.0] * 12 + [0.0] * 12)
        runtime.fit_readout_calibration(states, pending, labels)
        self.assertIsNotNone(runtime.readout_calibration)
        threat, _excess = runtime.pain_graded_batch(
            np.asarray([[0.5, 0.0], [4.0, 0.0]])
        )
        # The map is consumed per (root, action) read, so it must have
        # been fitted on those reads: the hot root's read maps near its
        # empirical rate, the cold root's near zero.  A map fitted on the
        # composed threat would calibrate a level no read ever carries.
        self.assertGreater(threat[0], 0.8)
        self.assertLess(threat[1], 0.2)

    def test_nnls_solves_the_non_negative_least_squares(self):
        from node_runtime import _nnls

        # The unconstrained solution here is [1, -1], but clipping it is
        # not the constrained optimum: on the w1 = 0 face the best w0 is
        # 0.5 (residual [0.5, -0.5] with a non-positive gradient on the
        # released column).  The active set must find the boundary point.
        np.testing.assert_allclose(
            _nnls(np.asarray([[1.0, 0.0], [1.0, 1.0]]),
                  np.asarray([1.0, 0.0])),
            [0.5, 0.0], atol=1e-9,
        )
        # An exactly solvable interior case keeps both weights.
        np.testing.assert_allclose(
            _nnls(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                  np.asarray([1.0, 1.0, 2.0])),
            [1.0, 1.0], atol=1e-9,
        )
        # The ridge keeps collinear columns stable and non-negative.
        shrunk = _nnls(
            np.asarray([[1.0, 1.0], [1.0, 1.0]]),
            np.asarray([2.0, 2.0]), ridge=1.0,
        )
        self.assertTrue((shrunk >= 0.0).all())

    def test_hazard_composition_replaces_the_max_combine(self):
        runtime = self.make_runtime()
        readings = {
            "prototype": np.asarray([0.9, 0.1, 0.9, 0.1]),
            "feature": np.asarray([0.9, 0.9, 0.1, 0.1]),
        }
        labels = np.asarray([1.0, 0.0, 0.0, 0.0])
        runtime._hazard_channels = (
            lambda states, mask_table=None: readings
        )
        report = runtime.fit_hazard_composition(
            np.zeros((4, 1)), np.zeros((4, 0)), labels
        )
        weights = runtime.hazard_composition
        # Weights are non-negative, and the fitted composition beats the
        # hand-written max against the same labels: the max's false
        # positive on row 1 (both channels high, label clean) is deflated.
        self.assertTrue(all(weight >= 0.0 for weight in weights.values()))
        self.assertLess(report["composed_mae"], report["maximum_mae"])
        read = runtime.pain_signal_batch(np.zeros((4, 1)))
        composed = np.zeros(4)
        for name, weight in weights.items():
            composed = composed + weight * readings[name]
        mids, rates = runtime.hazard_calibration
        np.testing.assert_allclose(
            read, np.interp(composed, mids, rates), atol=1e-9
        )
        self.assertLess(
            read[1],
            max(readings["prototype"][1], readings["feature"][1]),
        )

    def test_pain_signal_keeps_the_max_without_a_composition(self):
        runtime = self.make_runtime(
            lag=0, scales=np.asarray([1.0]), soft_neighbors=1
        )
        first = runtime._create_node(np.asarray([0.0]))
        runtime._create_node(np.asarray([1.0]))
        # Rate semantics (P0): the prototype channel reads each node's
        # transition pain rate.  The first node saw ten painful
        # transitions; the second has no samples and reads the global
        # base rate (10 pain mass over 20 transitions).
        runtime._association_stats[first] = [10, 10.0]
        runtime._association_global = [20, 10.0]
        read = runtime.pain_signal_batch(np.asarray([[0.0], [1.0]]))
        self.assertAlmostEqual(float(read[0]), 11.0 / 12.0, places=9)
        self.assertAlmostEqual(float(read[1]), 0.5, places=9)

    def test_importance_weighting_scales_the_cell_evidence(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40,
            importance_weighting=True,
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
                propensity=0.9,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([1.0, 1.0]), done=True, episode=0,
                propensity=0.1,
            ),
        ]
        runtime.learn_episode(rows)
        cells = runtime.feature_action_stats[feature_id]
        # Each row's window weight is the clipped product of
        # uniform-target ratios over the governing actions.  Row 0:
        # window = [0.9] -> min(1, 0.5/0.9).  Row 1: window = [0.9, 0.1]
        # -> the product clips the rare action's ratio at one.
        row0 = min(1.0, 0.5 / 0.9)
        self.assertAlmostEqual(cells[0][0], row0, places=9)
        self.assertAlmostEqual(cells[1][0], row0, places=9)
        # The anchor's effective sample size scales the same way, on top
        # of the install seeding (count = rule support = 5).
        node = runtime.graph.nodes[feature_id]
        self.assertAlmostEqual(node.count, 5.0 + 2.0 * row0, places=9)
        # Off: the same rows leave plain integer counts.
        plain = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40,
        )
        plain.rules.rules = runtime.rules.rules
        plain.install_feature_rules()
        plain.learn_episode(rows)
        plain_cells = plain.feature_action_stats[plain.feature_ids[0]]
        self.assertEqual(plain_cells[0][0], 1)
        self.assertEqual(plain_cells[1][0], 1)

    def test_value_cells_bootstrap_from_the_dopamine_channel(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=0, scales=np.asarray([1.0]), max_nodes=40, gamma=0.5
        )
        runtime.value_horizon = 1
        runtime.rules.rules = [
            FeatureRule(
                Primitive(0, True, "<", 0.5), 5, 4, 0.8, 2, 2, 1.0, 0.75
            )
        ]
        runtime.install_feature_rules(channel="value")
        feature_id = runtime.value_feature_ids[0]
        episode_one = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0]),
                signals=np.asarray([2.0, 0.0]), done=False, episode=0,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0]),
                signals=np.asarray([4.0, 1.0]), done=True, episode=0,
            ),
        ]
        runtime.learn_episode(episode_one)
        # The bootstrap snapshot is taken before the backward pass but
        # after the forward pass, so the entered node's immediate
        # dopamine (0.2 * 2.0) is already visible: the first pass's
        # target is 2.0 + 0.5 * 0.4 = 2.2.
        cell = runtime.feature_value_stats[feature_id][0]
        self.assertAlmostEqual(cell[1], 2.2, places=9)
        # Second pass: the middle node now carries a dopamine value.  The
        # bootstrap must read the dopamine channel (dopamine + immediate),
        # not the pain value planted next to it.  The forward pass of the
        # second episode moves the immediate again (0.2 * 2.0).
        middle = None
        for node_id, node in runtime.graph.nodes.items():
            features = getattr(node, "features", None)
            if node.kind == "concept" and features is not None and (
                np.allclose(features, [1.0])
            ):
                middle = node_id
                break
        self.assertIsNotNone(middle)
        runtime.graph.nodes[middle].value = 7.0
        runtime.graph.nodes[middle].dopamine = 6.0
        runtime.graph.nodes[middle].immediate_dopamine = 0.0
        episode_two = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0]),
                signals=np.asarray([2.0, 0.0]), done=False, episode=1,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0]),
                signals=np.asarray([4.0, 1.0]), done=True, episode=1,
            ),
        ]
        runtime.learn_episode(episode_two)
        # Target = 2.0 + 0.5 * (6.0 + 0.4) = 5.2; the pain value 7.0
        # would have produced 5.5.
        self.assertAlmostEqual(cell[1], 2.2 + 5.2, places=9)

    def test_k_step_target_arithmetic(self):
        runtime = self.make_runtime(gamma=0.5)
        runtime.value_horizon = 2
        ahead = runtime._create_node(np.asarray([0.0]))
        steps = [
            (None, 0, "n0", 0.0, 1.0, False, (), (), 0, (), 1.0),
            (None, 0, ahead, 0.0, 1.0, False, (), (), 0, (), 1.0),
            (None, 0, "n2", 0.0, 1.0, True, (), (), 0, (), 1.0),
        ]
        snapshot = {ahead: 8.0}
        # 1 + 0.5*1 + 0.25*8 = 3.5, truncated at the episode end.
        self.assertAlmostEqual(
            runtime._k_step_value_target(steps, 0, snapshot), 3.5
        )
        # Position 2 (terminal): reward only.
        self.assertAlmostEqual(
            runtime._k_step_value_target(steps, 2, snapshot), 1.0
        )

    def test_collect_rows_carry_the_action_propensity(self):
        from substrate_generic import collect

        transitions, _ = collect(101, 30, coverage=0.1, lag=3)
        for row in transitions:
            self.assertTrue(0.0 <= float(row["propensity"]) <= 1.0)

    def test_learn_objective_weight_follows_realized_survival(self):
        from substrate_nodes import learn_objective_weight
        # Hazard and |value share| fall along the stream while the
        # realized remaining survival falls with it too: the hazard-only
        # read ranks survival backwards (a high-cost early decision
        # outlives every later one), and a weight past the hazard/value
        # ratio restores the ordering.
        rows = []
        for index in range(8):
            hazard = float(8 - index)
            rows.append(dict(episode=0, ledger=(hazard, -10.0 * hazard)))
        learned = learn_objective_weight(rows)
        self.assertIsNotNone(learned)
        self.assertGreater(learned, 0.1)
        # Too few decisions cannot rank the candidates.
        self.assertIsNone(learn_objective_weight(rows[:4]))
        # A single distinct outcome cannot rank the candidates: every row
        # in its own episode realizes one remaining frame.
        flat = [
            dict(episode=e, ledger=(1.0, -1.0)) for e in range(12)
        ]
        self.assertIsNone(learn_objective_weight(flat))

    def test_ledger_survival_report_flags_blind_components(self):
        from substrate_nodes import ledger_survival_report

        # The hazard share rises along the stream while the realized
        # remaining survival falls with it: the hazard component ranks
        # survival correctly (more hazard, shorter life), while the value
        # share is a constant - a blind component whose agreement is
        # exactly zero.
        rows = []
        for index in range(10):
            rows.append(dict(
                episode=0,
                ledger=(float(index + 1), -1.0),
            ))
        report = ledger_survival_report(rows, weight=0.05)
        self.assertEqual(report["decisions"], 10)
        self.assertGreater(report["hazard_agreement"], 0.9)
        self.assertAlmostEqual(report["value_agreement"], 0.0, places=9)
        # Too few decisions report nothing.
        self.assertIsNone(ledger_survival_report(rows[:4], weight=0.05))

    def test_planner_rows_carry_the_decision_ledger(self):
        from substrate_nodes import collect_planner, fit

        world, agent = fit(
            100, steps=80, epochs=1, max_nodes=100, lag=3,
            induce_features=False,
        )
        rows, _frames = collect_planner(
            100, world, agent, episodes=1, cap=20, horizon=2, lag=3,
            step_stride=1,
        )
        for row in rows:
            self.assertEqual(len(row["ledger"]), 2)
            hazard, value_share = row["ledger"]
            self.assertTrue(np.isfinite(hazard))
            self.assertTrue(np.isfinite(value_share))
            # Under the default objective the hazard share is
            # non-negative and the value share is a discount.
            self.assertGreaterEqual(hazard, 0.0)
            self.assertLessEqual(value_share, 0.0)

    def test_uninstall_withdraws_root_registrations_only(self):
        from node_features import FeatureRule, Primitive

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        rule = FeatureRule(
            Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        runtime.rules.rules = [rule]
        runtime.install_feature_rules()
        feature_id = runtime.feature_ids[0]
        runtime.uninstall_feature_rules([feature_id])
        self.assertEqual(runtime.feature_ids, [])
        self.assertNotIn(feature_id, runtime._feature_all)
        self.assertNotIn(feature_id, runtime._feature_objects)
        # The materialization key is kept: re-installing the same rule
        # returns the existing node instead of duplicating it.
        runtime.install_feature_rules()
        self.assertEqual(runtime.feature_ids, [feature_id])

    def test_graded_holdout_loss_computes_the_backfit_counterfactual(self):
        from node_features import FeatureRule, Primitive
        from substrate_nodes import _graded_holdout_loss

        runtime = self.make_runtime(
            lag=1, scales=np.asarray([1.0, 1.0]), max_nodes=40
        )
        good_rule = FeatureRule(
            Primitive(0, True, "<", 2.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        bad_rule = FeatureRule(
            Primitive(0, True, ">", 3.0), 5, 4, 0.8, 2, 2, 1.0, 0.75
        )
        runtime.rules.rules = [good_rule, bad_rule]
        runtime.install_feature_rules([good_rule], horizon=2)
        runtime.install_feature_rules([bad_rule], horizon=2)
        good_id, bad_id = runtime.feature_ids
        # The good root reads its clean action slightly above zero (so
        # the fired-scope proxy keeps it); the bad root flags clean
        # states as dangerous.
        runtime.graph.nodes[good_id].value = 0.1
        runtime.graph.nodes[bad_id].value = 0.9
        runtime._update_feature_action([good_id], 0, 0.0)
        runtime._update_feature_action([good_id], 0, 0.0)
        runtime._update_feature_action([good_id], 1, 1.0)
        runtime._update_feature_action([bad_id], 0, 1.0)
        runtime._update_feature_action([bad_id], 0, 1.0)
        runtime._update_feature_action([bad_id], 1, 1.0)
        enc = np.asarray([
            [0.5, 0.0], [0.5, 0.0], [4.0, 0.0], [4.0, 0.0],
        ])
        labels = np.zeros(4)
        mask = np.ones(4, dtype=bool)
        # Good root: threat = min(read0, read1) = (0 + 2*0.1)/4 = 0.05.
        # Bad root: min((2+1.8)/4, (1+1.8)/3) = 0.9333...
        loss_all = _graded_holdout_loss(runtime, enc, labels, mask)
        self.assertAlmostEqual(loss_all, (0.05 + 0.9333333333) / 2, places=6)
        # The exact counterfactual: the same rows read without the
        # generation's roots - the bad root's false alarm is gone.
        loss_without = _graded_holdout_loss(
            runtime, enc, labels, mask, exclude=[bad_id]
        )
        self.assertAlmostEqual(loss_without, 0.05, places=9)
        self.assertLess(loss_without, loss_all)
        # The swap is reversible: the full readout is intact.
        self.assertEqual(runtime.feature_ids, [good_id, bad_id])

    def test_iterative_generations_record_their_backfit(self):
        from substrate_generic import collect
        from substrate_nodes import _train

        transitions, _ = collect(101, 240, coverage=0.1, lag=3)
        world, agent = _train(
            transitions, seed=101, epochs=3, lag=3, max_nodes=200,
            birth_threshold=0.45, component_kinds=(), component_min_count=8,
            max_components=0, component_weight=0.5, component_gating=False,
            memory_weight=0.1, association_rate=0.0, induce_features=True,
            precursor_labels=True, pain_horizon=None, voi_target=3,
            feature_value_weight=0.02, contrast_objective=True,
            iterative_induction=True,
        )
        # Every generation's install meets a held-out backfit test; the
        # log records the two losses and the keep-or-rollback verdict.
        self.assertIsInstance(agent.generation_log, list)
        for entry in agent.generation_log:
            self.assertIn("loss_with", entry)
            self.assertIn("loss_without", entry)
            self.assertIn("rolled_back", entry)

    def test_variance_birth_splits_a_mixed_cell(self):
        runtime = self.make_runtime(
            lag=0, scales=np.asarray([1.0]), max_nodes=40,
            birth_threshold=0.45, component_min_count=2,
        )
        first = runtime._create_node(np.asarray([0.0]))
        runtime._create_node(np.asarray([10.0]))
        runtime._create_node(np.asarray([20.0]))
        # A sharply mixed cell and two stable ones set the population bar
        # below the mixed cell's variance.
        runtime._node_pain_count[first] = 10
        runtime._node_pain_m2[first] = 100.0 * 9.0
        for other in ("concept_2", "concept_3"):
            runtime._node_pain_count[other] = 10
            runtime._node_pain_m2[other] = 1.0 * 9.0
        runtime._refresh_variance_bar()
        # A query well inside the mixed cell's territory but closer than
        # the geometric birth threshold: the refinement birth fires.
        born = runtime._assign(np.asarray([0.3]))
        self.assertNotEqual(born, first)
        self.assertTrue(np.allclose(
            runtime.graph.nodes[born].features, [0.3]
        ))
        # A query at the center does not split.
        same = runtime._assign(np.asarray([0.05]))
        self.assertEqual(same, first)

    def test_utility_pruning_merges_into_the_neighbour(self):
        runtime = self.make_runtime(
            lag=0, scales=np.asarray([1.0]), max_nodes=3,
            birth_threshold=0.01,
        )
        runtime._assign(np.asarray([0.0]))
        runtime._assign(np.asarray([10.0]))
        runtime._assign(np.asarray([20.0]))
        first_id = runtime._prototype_ids[0]
        host_id = runtime._prototype_ids[1]
        # concept_1 sits at value 1.0 next to concept_2 at 0.0 - a cheap
        # near-duplicate.  concept_3 is far away with a distinct outcome:
        # removing it would cost the most, so it must survive.
        runtime.graph.nodes[first_id].value = 1.0
        runtime.graph.nodes[runtime._prototype_ids[2]].value = 5.0
        runtime._association_stats[first_id] = [3, 1.0]
        # An episode boundary: the nodes assigned so far belong to past
        # episodes and become prune candidates again.
        runtime.reset()
        # A fourth distinct state is blocked by the budget: the least
        # useful prototype merges into its nearest neighbour and the
        # birth proceeds, so the cover keeps refining at the cap.
        born = runtime._assign(np.asarray([30.0]))
        self.assertEqual(len(runtime._prototype_ids), 3)
        self.assertNotIn(first_id, runtime.graph.nodes)
        self.assertIn(runtime._prototype_ids[2], runtime.graph.nodes)
        self.assertEqual(
            len(runtime._prototype_matrix), len(runtime._prototype_ids)
        )
        # The victim's evidence is consolidated, not dropped: the merged
        # value is the count-weighted mean (1 and 0 over two visits).
        self.assertEqual(runtime.graph.nodes[host_id].count, 2)
        self.assertAlmostEqual(
            runtime.graph.nodes[host_id].value, 0.5, places=9
        )
        self.assertEqual(runtime._association_stats[host_id], [3, 1.0])
        self.assertTrue(np.allclose(
            runtime.graph.nodes[born].features, [30.0]
        ))


class PlanScoreTests(unittest.TestCase):
    def test_plan_score_matches_search_selection(self):
        from latent_world import latent
        from substrate_nodes import fit, plan_score, policy_result

        world, agent = fit(100, steps=300, epochs=1, max_nodes=200, lag=3)
        env = latent[3](100)
        prefix = np.zeros(3, dtype=float)
        result = policy_result(env, world, agent, prefix, horizon=1, lag=3)
        score = result.score
        # P4: the score carries the objective's ledger - the scalarized
        # column is the sum of its hazard, value and curiosity shares.
        self.assertEqual(len(score), 5)
        self.assertAlmostEqual(
            score[0], score[2] + score[3] + score[4], places=6
        )
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

    def test_plan_score_carries_the_curiosity_share(self):
        from latent_world import latent
        from substrate_nodes import fit, plan_score, policy_result

        world, agent = fit(100, steps=300, epochs=1, max_nodes=200, lag=3)
        agent.curiosity_weight = 0.5
        env = latent[3](100)
        prefix = np.zeros(3, dtype=float)
        result = policy_result(env, world, agent, prefix, horizon=3, lag=3)
        score = result.score
        self.assertEqual(len(score), 5)
        self.assertAlmostEqual(
            score[0], score[2] + score[3] + score[4], places=6
        )
        scores = [
            plan_score(
                env.obs(), world, agent, prefix, plan, 3, lag=3
            )
            for plan in np.ndindex(*(2,) * 3)
        ]
        best = min(range(len(scores)), key=lambda index: scores[index])
        self.assertAlmostEqual(result.score[0], scores[best][0], places=6)

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
