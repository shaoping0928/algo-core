"""Tests for the deployable event-warning model and the P1-D.0 trigger."""

import unittest

import numpy as np

from event_warning import (
    OnlineEvents, calibrate_threshold, phi_for, warning_value,
)
from outcome_level_probe import stream_stats, candidate_predicates


def _predicate(name, kind, dim, thr):
    return dict(name=name, kind=kind, dim=dim, thr=thr)


class EventWarningTest(unittest.TestCase):
    def test_online_events_match_offline_crossings(self):
        stats = {0: dict(q10=-0.5, q90=0.5, jump=1.0, spans_zero=True)}
        predicates = [
            _predicate("x0_low", "cross_low", 0, stats[0]["q10"]),
            _predicate("x0_sign", "sign", 0, None),
        ]
        names = ["x0_low", "x0_sign"]
        tracker = OnlineEvents(names, predicates)
        observations = [
            np.asarray([0.0]),    # row 0: no previous -> nothing
            np.asarray([-0.6]),   # row 1: crosses low + sign
            np.asarray([-0.6]),   # row 2: nothing
            np.asarray([0.7]),    # row 3: sign only
        ]
        for obs in observations:
            tracker.update(obs)
        self.assertEqual(tracker.last["x0_low"], 1)
        self.assertEqual(tracker.last["x0_sign"], 3)
        features = tracker.features()
        self.assertAlmostEqual(features[0], 2.0 / 32.0)
        self.assertEqual(tracker.current_row, 3)

    def test_history_features_are_causal_at_decision_time(self):
        stats = {0: dict(q10=-0.5, q90=0.5, jump=1.0, spans_zero=True)}
        predicates = [_predicate("x0_low", "cross_low", 0, -0.5)]
        tracker = OnlineEvents(["x0_low"], predicates)
        for obs in (np.asarray([0.0]), np.asarray([-0.6])):
            tracker.update(obs)
        features = tracker.features()
        self.assertAlmostEqual(features[0], 0.0)
        self.assertAlmostEqual(features[1], 1.0 / 16.0)
        self.assertAlmostEqual(features[2], 1.0)

    def test_phi_uses_model_dims_and_scaling(self):
        model = dict(
            dims=[1], obs_mean=np.asarray([10.0]),
            obs_scale=np.asarray([2.0]),
        )
        phi = phi_for(model, np.asarray([0.5, 14.0]))
        self.assertAlmostEqual(float(phi[0]), 2.0)

    def test_warning_value_survives_feature_block(self):
        # A hand-built arm with a constant hazard: F(w) = 1 - 0.5^w.
        model = dict(
            arms=dict(A1_history=dict(
                coefficients=np.zeros(5),
                mean=np.zeros(4), scale=np.ones(4),
            )),
            l_max=4, n_lags=1,
        )
        value = warning_value(
            model,
            np.zeros(0), np.zeros(0), [], 0.0, window=3,
        )
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_calibrate_threshold_maximises_coverage(self):
        values = np.asarray([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        tau, covered, precision = calibrate_threshold(values, labels, 0.5)
        self.assertEqual(covered, 4)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(tau, 0.3)


if __name__ == "__main__":
    unittest.main()
