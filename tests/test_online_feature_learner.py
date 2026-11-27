import unittest

import numpy as np

from online_feature_learner import OnlineFeatureLearner
from node_runtime import NodeRuntime


def transition(episode, step, pain, done=False, obs=None):
    if obs is None:
        obs = float(step % 2)
    return dict(
        episode=episode,
        obs=np.asarray([obs], dtype=float),
        next_obs=np.asarray([float(1.0 - obs)], dtype=float),
        pending=np.zeros(0, dtype=float),
        history=np.asarray([0.0], dtype=float),
        action=0,
        signals=np.asarray([0.0, float(pain)], dtype=float),
        done=bool(done),
    )


def feed_balanced_episode(learner, episode, length=12):
    results = []
    for step in range(length):
        pain = float(step % 2)
        results.append(learner.observe_transition(
            transition(
                episode, step, pain,
                done=(step == length - 1),
            )
        ))
    return results


class OnlineFeatureLearnerTests(unittest.TestCase):
    def test_pain_labels_wait_for_their_future_horizon(self):
        learner = OnlineFeatureLearner(
            obs_dim=1, lag=0, label_horizon=3,
            warmup_episodes=20, discovery_interval=5,
            min_model_samples=100,
        )
        learner.observe_transition(transition(0, 0, 0.0))
        learner.observe_transition(transition(0, 1, 1.0))

        self.assertEqual(learner.snapshot()["counters"].get(
            "labels_completed", 0,
        ), 0)
        learner.observe_transition(transition(0, 2, 0.0))
        self.assertEqual(
            learner.snapshot()["counters"]["labels_completed"], 1,
        )

    def test_unfinished_tail_is_censored_not_safe(self):
        learner = OnlineFeatureLearner(
            obs_dim=1, lag=0, label_horizon=4,
            warmup_episodes=20, discovery_interval=5,
            min_model_samples=100,
        )
        learner.observe_transition(transition(0, 0, 0.0))
        learner.observe_transition(transition(0, 1, 0.0))
        learner.observe_transition(transition(1, 0, 0.0))

        counters = learner.snapshot()["counters"]
        self.assertEqual(counters["censored_labels"], 2)
        self.assertEqual(counters.get("labels_completed", 0), 0)

    def test_features_are_proposed_from_past_and_validated_on_future_episodes(self):
        learner = OnlineFeatureLearner(
            obs_dim=1, lag=0, label_horizon=1,
            warmup_episodes=2, discovery_interval=1,
            discovery_window=8, max_episode_samples=32,
            max_features=48, max_features_per_family=12,
            min_discovery_support=2, verification_episodes=4,
            verification_timeout=4, min_verification_support=2,
            monitor_episodes=8, min_model_samples=100, seed=3,
        )

        feed_balanced_episode(learner, 0)
        self.assertEqual(learner.snapshot()["candidates"], 0)
        feed_balanced_episode(learner, 1)
        proposed = learner.snapshot()
        self.assertGreater(proposed["candidates"], 0)
        self.assertEqual(proposed["statuses"].get("verifying"), proposed["candidates"])

        for episode in range(2, 6):
            feed_balanced_episode(learner, episode)
        validated = learner.snapshot()
        self.assertGreater(validated["statuses"].get("validated", 0), 0)

        readout = learner.observe_transition(
            transition(6, 0, 0.0, obs=0.0)
        )
        self.assertEqual(readout["status"], "supported")
        self.assertTrue(readout["features"])

    def test_verification_waits_for_labels_until_its_timeout(self):
        learner = OnlineFeatureLearner(
            obs_dim=1, lag=0, label_horizon=1,
            warmup_episodes=2, discovery_interval=1,
            discovery_window=8, max_episode_samples=32,
            max_features=48, max_features_per_family=12,
            min_discovery_support=2, verification_episodes=4,
            verification_timeout=6, min_verification_support=2,
            monitor_episodes=8, min_model_samples=100, seed=4,
        )
        feed_balanced_episode(learner, 0)
        feed_balanced_episode(learner, 1)
        self.assertGreater(learner.snapshot()["candidates"], 0)
        learner.label_horizon = 4
        learner.discovery_interval = 1000

        for episode in range(2, 6):
            learner.observe_transition(transition(episode, 0, 0.0))
            learner.end_episode(terminal=False)
        self.assertGreater(
            learner.snapshot()["statuses"].get("verifying", 0), 0,
        )

        for episode in range(6, 8):
            learner.observe_transition(transition(episode, 0, 0.0))
            learner.end_episode(terminal=False)
        snapshot = learner.snapshot()
        self.assertEqual(snapshot["statuses"].get("verifying", 0), 0)
        self.assertGreater(snapshot["counters"]["features_rejected"], 0)
        self.assertTrue(any(
            feature["last_reason"] == "verification_support_timeout"
            for feature in snapshot["features"]
        ))

    def test_online_prediction_error_is_prequential(self):
        learner = OnlineFeatureLearner(
            obs_dim=1, lag=0, label_horizon=1,
            warmup_episodes=100, min_model_samples=2, seed=1,
        )
        for step in range(12):
            obs = float(step)
            learner.observe_transition(dict(
                episode=0,
                obs=np.asarray([obs]),
                next_obs=np.asarray([obs + 1.0]),
                pending=np.zeros(0),
                history=np.asarray([0.0]),
                signals=np.asarray([0.0, 0.0]),
                done=False,
            ))
        self.assertEqual(learner.snapshot()["world_model_samples"], 12)
        # The residual channel only becomes available after prior transitions
        # have supplied enough training rows for a prequential prediction.
        self.assertGreater(
            learner.counts["residual_available_transitions"], 0,
        )

    def test_pending_queue_and_history_must_agree(self):
        learner = OnlineFeatureLearner(
            obs_dim=1, lag=2, label_horizon=1,
            warmup_episodes=20, min_model_samples=100,
        )
        result = learner.observe_transition(dict(
            episode=0,
            obs=np.asarray([0.0]),
            next_obs=np.asarray([0.1]),
            pending=np.asarray([0.0, 1.0]),
            history=np.asarray([0.0, 0.0, 1.0]),
            signals=np.asarray([0.0, 0.0]),
            done=False,
        ))
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(
            result["reason"], "pending_history_mismatch",
        )
        self.assertEqual(learner.snapshot()["transitions"], 0)

    def test_node_runtime_streaming_transition_hook_is_opt_in(self):
        runtime = NodeRuntime(
            obs_dim=1, lag=2, scales=np.ones(3),
            birth_threshold=0.1, max_nodes=8, exploration=0.0,
            online_feature_learning=True,
            online_feature_options=dict(
                warmup_episodes=10, discovery_interval=5,
                min_model_samples=100,
            ),
        )
        runtime.learn_transition(
            [0.0], 1, [0.25], [0.0, 0.0], done=False,
        )

        self.assertEqual(
            runtime.online_feature_learner.snapshot()["transitions"], 1,
        )
        self.assertIsNone(
            NodeRuntime(
                obs_dim=1, lag=0, scales=np.ones(1),
                birth_threshold=0.1, max_nodes=8,
            ).online_feature_learner
        )

    def test_live_act_observe_path_updates_online_features(self):
        runtime = NodeRuntime(
            obs_dim=1, lag=2, scales=np.ones(3),
            birth_threshold=0.1, max_nodes=8, exploration=0.0,
            online_feature_learning=True,
            online_feature_options=dict(
                warmup_episodes=10, discovery_interval=5,
                label_horizon=1, min_model_samples=100,
            ),
        )
        action = runtime.act([0.1], explore=False)
        runtime.observe([0.2], action, [0.0, 0.0], done=True)

        snapshot = runtime.online_feature_learner.snapshot()
        self.assertEqual(snapshot["transitions"], 1)
        self.assertEqual(snapshot["episodes"], 1)

    def test_batch_replay_does_not_duplicate_online_episode_support(self):
        runtime = NodeRuntime(
            obs_dim=1, lag=0, scales=np.ones(1),
            birth_threshold=0.1, max_nodes=8,
            online_feature_learning=True,
            online_feature_options=dict(
                warmup_episodes=10, discovery_interval=5,
                min_model_samples=100,
            ),
        )
        rows = [
            transition(0, 0, 0.0),
            transition(0, 1, 1.0, done=True),
        ]
        runtime.learn_episode(rows)
        self.assertEqual(
            runtime.online_feature_learner.snapshot()["transitions"], 2,
        )

        runtime.learn_episode(rows, learn_online=False)
        self.assertEqual(
            runtime.online_feature_learner.snapshot()["transitions"], 2,
        )


if __name__ == "__main__":
    unittest.main()
