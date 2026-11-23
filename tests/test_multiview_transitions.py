"""Tests for action-aligned multiview transition measures."""

import unittest

import numpy as np

from multiview_transitions import MultiViewTransitionModel, UNKNOWN
from multiview_transition_probe import (
    evaluate_fold,
    _observed_outcome,
    _summarize_level_audit,
)
from predictive_state_quotient import PredictiveStateQuotient
from predictive_state_quotient import PredictiveStateHierarchy
from hierarchical_transitions import HierarchicalTransitionModel
from substrate_nodes import _transition_table


class FakeAgent:
    lag = 1
    obs_dim = 1
    scales = np.ones(2, dtype=float)
    action_space = (0, 1)

    @staticmethod
    def _nearest(state):
        state = np.asarray(state, dtype=float)
        return "s%d" % int(round(float(state[0]))), 0.0


class TransitionTableAlignmentTest(unittest.TestCase):
    def test_legacy_transition_table_keeps_delivered_action_counts(self):
        agent = FakeAgent()
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([0.0, 0.0]), done=False, episode=1,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([0.0, 0.0]), done=True, episode=1,
            ),
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([0.0, 0.0]), done=False, episode=2,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([0.0, 0.0]), done=True, episode=2,
            ),
        ]
        edges, _risk, _default = _transition_table(
            rows, agent, min_support=2,
        )
        edge = edges["s0"]["s1"]
        self.assertEqual(edge["count"], 2)
        # Newly-issued actions are 0 and 1; the actual acting actions are
        # history[lag] = 1 and 0 respectively.
        self.assertEqual(edge["actions"], {0: 1, 1: 1})

    def test_terminal_row_does_not_create_an_outgoing_graph_edge(self):
        agent = FakeAgent()
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([0.0, 0.0]), done=False, episode=7,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([1.0, 1.0]),
                signals=np.asarray([0.0, 1.0]), done=True, episode=7,
            ),
            # Malformed tail row verifies that terminal is still a hard
            # boundary even if an episode identifier is accidentally reused.
            dict(
                obs=np.asarray([2.0]), next_obs=np.asarray([3.0]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([0.0, 0.0]), done=False, episode=7,
            ),
        ]
        edges, _risk, _default = _transition_table(
            rows, agent, min_support=1,
        )
        self.assertIn("s1", edges["s0"])
        self.assertNotIn("s2", edges.get("s1", {}))


class ObservedOutcomeWindowTest(unittest.TestCase):
    def test_terminal_inside_window_is_an_event_not_a_short_window_negative(self):
        rows = [
            {"done": False}, {"done": True}, {"done": False},
        ]
        self.assertEqual(_observed_outcome(rows, 0, 3), (2, True))

    def test_full_alive_window_is_right_censored_at_the_window(self):
        rows = [{"done": False} for _ in range(4)]
        self.assertEqual(_observed_outcome(rows, 0, 3), (3, False))

    def test_short_alive_tail_is_right_censored(self):
        rows = [{"done": False}, {"done": False}]
        self.assertIsNone(_observed_outcome(rows, 0, 3))


class HierarchyLevelAuditTest(unittest.TestCase):
    def test_fold_reports_independent_layer_metrics_and_base_rates(self):
        agent = FakeAgent()

        def episode_rows(episode):
            observations = (0.0, 1.0, 0.0, 1.0)
            return [
                dict(
                    obs=np.asarray([observations[index]]),
                    next_obs=np.asarray([observations[index + 1]])
                    if index + 1 < len(observations)
                    else np.asarray([0.0]),
                    history=np.asarray([0.0, 0.0]),
                    signals=np.asarray([0.0, 0.0]),
                    done=False, episode=episode,
                )
                for index in range(len(observations))
            ]

        train_rows = episode_rows(0) + episode_rows(1)
        test_rows = episode_rows(2)
        report = evaluate_fold(
            agent, train_rows, test_rows, lag=1, windows=(1,),
            max_starts=10, seed=7, hierarchy_groups=(2, 1),
        )
        window = report["windows"]["1"]
        audit = window["hierarchy_level_audit"]
        self.assertEqual(set(audit), {"0", "1", "2"})
        for level in audit.values():
            self.assertEqual(level["eligible"], 4)
            self.assertGreater(level["mapped_targets"], 0)
            self.assertIsNotNone(level["base_rate_log_loss"])
            self.assertIsNotNone(level["log_loss"])
            self.assertIsNotNone(level["fine_node_resolution"])
            self.assertEqual(level["unsupported"], 0)

    def test_unknowns_reduce_coverage_and_are_penalized_against_base_rate(self):
        item = dict(
            eligible=2,
            mapped_targets=2,
            unmapped_targets=0,
            unsupported=1,
            no_live_prediction=0,
            all_correct=[0.0, 1.0],
            all_log_loss=[-np.log(1e-12), 0.2],
            supported_correct=[1.0],
            supported_log_loss=[0.2],
            supported_targets=["q1"],
            fine_resolution=[3.0],
            _targets=["q0", "q1"],
        )
        report = _summarize_level_audit(
            item, {"q0": 3, "q1": 1},
        )
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["unknown_rate"], 0.5)
        self.assertEqual(report["supported_n"], 1)
        self.assertAlmostEqual(report["base_rate_top1"], 0.5)
        self.assertLess(report["log_loss_gain"], 0.0)
        self.assertIsNotNone(report["supported_log_loss_gain"])
        self.assertAlmostEqual(report["fine_node_resolution"], 3.0)

    def test_no_live_prediction_is_not_counted_as_coverage(self):
        item = dict(
            eligible=1,
            mapped_targets=1,
            unmapped_targets=0,
            unsupported=0,
            no_live_prediction=1,
            all_correct=[0.0],
            all_log_loss=[-np.log(1e-12)],
            supported_correct=[],
            supported_log_loss=[],
            supported_targets=[],
            fine_resolution=[],
            _targets=["q0"],
        )
        report = _summarize_level_audit(item, {"q0": 1})
        self.assertEqual(report["coverage"], 0.0)
        self.assertEqual(report["unknown_rate"], 0.0)
        self.assertEqual(report["no_live_prediction"], 1)


class PredictiveStateQuotientTest(unittest.TestCase):
    def test_groups_states_with_matching_observed_transition_profiles(self):
        rows = []
        for state, successor in ((0, 10), (1, 10), (2, 100), (3, 100)):
            for repeat in range(8):
                rows.append(dict(
                    obs=np.asarray([float(state)]),
                    next_obs=np.asarray([float(successor)]),
                    history=np.asarray([float(repeat % 2), 0.0]),
                    signals=np.asarray([0.2, 0.1]),
                    done=False, episode=state * 10 + repeat,
                ))
        quotient = PredictiveStateQuotient.fit_rows(
            rows, FakeAgent(), n_groups=2, seed=3,
        )
        self.assertEqual(quotient.n_groups, 2)
        self.assertEqual(quotient.map_node("s0"), quotient.map_node("s1"))
        self.assertEqual(quotient.map_node("s2"), quotient.map_node("s3"))
        self.assertNotEqual(quotient.map_node("s0"), quotient.map_node("s2"))

    def test_quotient_kernel_uses_coarse_source_and_successor(self):
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([0.1, 0.0]), done=False, episode=1,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([0.1, 0.0]), done=True, episode=1,
            ),
        ]
        node_map = {"s0": "q0", "s1": "q0"}
        model = MultiViewTransitionModel.fit_rows(
            rows, FakeAgent(), lag=1, node_map=node_map,
        )
        transition = model.transition("q0", 1)
        self.assertEqual(transition[0], {"q0": 1.0})
        result = model.predict_views(
            FakeAgent(), observation=[0.0], pending=[1],
            issued_actions=[0], windows=(1,),
        )[1]
        self.assertEqual(result["distribution"], {"q0": 1.0})


class HierarchicalTransitionBackoffTest(unittest.TestCase):
    def test_learned_hierarchy_is_nested(self):
        rows = []
        for state, successor in ((0, 10), (1, 10), (2, 100), (3, 100)):
            for repeat in range(8):
                rows.append(dict(
                    obs=np.asarray([float(state)]),
                    next_obs=np.asarray([float(successor)]),
                    history=np.asarray([float(repeat % 2), 0.0]),
                    signals=np.asarray([0.2, 0.1]),
                    done=False, episode=state * 10 + repeat,
                ))
        hierarchy = PredictiveStateHierarchy.fit_rows(
            rows, FakeAgent(), group_counts=(2, 1), seed=3,
        )
        self.assertEqual(len(hierarchy.parents), 2)
        self.assertEqual(len(hierarchy.level_sizes), 3)
        for fine, labels in hierarchy.fine_to_levels.items():
            self.assertEqual(
                hierarchy.parents[0][labels[0]], labels[1],
            )
            self.assertEqual(
                hierarchy.parents[1][labels[1]], labels[2],
            )

    def test_rollout_uses_parent_only_when_fine_action_edge_is_missing(self):
        agent = FakeAgent()
        rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([0.1, 0.0]), done=False, episode=1,
            ),
            dict(
                obs=np.asarray([2.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([0.0, 0.0]),
                signals=np.asarray([0.0, 1.0]), done=True, episode=2,
            ),
        ]
        hierarchy = PredictiveStateHierarchy(
            fine_to_levels={
                "s0": ("s0", "q0", "root"),
                "s1": ("s1", "q0", "root"),
                "s2": ("s2", "q1", "root"),
            },
            parents=(
                {"s0": "q0", "s1": "q0", "s2": "q1"},
                {"q0": "root", "q1": "root"},
            ),
            level_sizes=(3, 2, 1),
            level_group_counts=(3, 2, 1),
        )
        models = [
            MultiViewTransitionModel.fit_rows(rows, agent, node_map=None),
            MultiViewTransitionModel.fit_rows(
                rows, agent, node_map={
                    "s0": "q0", "s1": "q0", "s2": "q1",
                },
            ),
            MultiViewTransitionModel.fit_rows(
                rows, agent, node_map={
                    "s0": "root", "s1": "root", "s2": "root",
                },
            ),
        ]
        model = HierarchicalTransitionModel(hierarchy, models)
        result = model.predict_views(
            agent, observation=[0.0], pending=[1],
            issued_actions=[0, 0], windows=(1, 2),
        )
        self.assertEqual(result[1]["distribution"], {(0, "s1"): 1.0})
        self.assertAlmostEqual(result[2]["unknown"], 0.0)
        self.assertAlmostEqual(result[2]["terminal"], 1.0)
        self.assertGreater(result[2]["level_usage"][2], 0.0)
        self.assertAlmostEqual(result[1]["resolution"], 1.0)
        self.assertIsNone(result[2]["resolution"])


class MultiViewTransitionTest(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        # history is newest-first: [newly-issued action, action that acts now].
        self.rows = [
            dict(
                obs=np.asarray([0.0]), next_obs=np.asarray([1.0]),
                history=np.asarray([0.0, 1.0]),
                signals=np.asarray([0.5, 0.1]), done=False, episode=1,
            ),
            dict(
                obs=np.asarray([1.0]), next_obs=np.asarray([2.0]),
                history=np.asarray([1.0, 0.0]),
                signals=np.asarray([0.7, 0.3]), done=True, episode=1,
            ),
        ]
        self.model = MultiViewTransitionModel.fit_rows(
            self.rows, self.agent, lag=1,
        )

    def test_fit_uses_the_action_that_actually_caused_the_edge(self):
        self.assertIsNone(self.model.transition("s0", 0))
        transition = self.model.transition("s0", 1)
        self.assertIsNotNone(transition)
        distribution, support = transition
        self.assertEqual(support, 1)
        self.assertEqual(distribution, {"s1": 1.0})

    def test_one_kernel_answers_multiple_windows(self):
        result = self.model.predict_views(
            self.agent, observation=[0.0], pending=[1],
            issued_actions=[0, 1], windows=(1, 2),
        )
        self.assertEqual(result[1]["distribution"], {"s1": 1.0})
        self.assertEqual(result[1]["status"], "supported")
        self.assertEqual(result[2]["distribution"], {})
        self.assertAlmostEqual(result[2]["terminal"], 1.0)
        self.assertEqual(result[2]["status"], "supported")
        self.assertAlmostEqual(result[2]["pain"], 0.4)

    def test_target_queries_return_first_passage_by_window(self):
        result = self.model.predict_views(
            self.agent, observation=[0.0], pending=[1],
            issued_actions=[0, 1], windows=(1, 2),
            target_nodes={"s2"},
        )
        self.assertAlmostEqual(result[1]["target_hit"], 0.0)
        self.assertAlmostEqual(result[2]["target_hit"], 1.0)
        self.assertEqual(result[2]["distribution"], {})

    def test_unsupported_action_becomes_unknown_not_safe(self):
        result = self.model.predict_views(
            self.agent, observation=[0.0], pending=[0],
            issued_actions=[1], windows=(1,),
        )[1]
        self.assertEqual(result["status"], UNKNOWN)
        self.assertAlmostEqual(result["unknown"], 1.0)
        self.assertEqual(result["distribution"], {})

    def test_out_of_support_state_becomes_unknown(self):
        result = self.model.predict_views(
            self.agent, observation=[10.0], pending=[1],
            issued_actions=[0], windows=(1,),
        )[1]
        self.assertEqual(result["status"], UNKNOWN)
        self.assertAlmostEqual(result["unknown"], 1.0)

    def test_zero_window_returns_current_state(self):
        result = self.model.predict_views(
            self.agent, observation=[0.0], pending=[1],
            issued_actions=[], windows=(0,),
        )[0]
        self.assertEqual(result["distribution"], {"s0": 1.0})
        self.assertEqual(result["terminal"], 0.0)


if __name__ == "__main__":
    unittest.main()
