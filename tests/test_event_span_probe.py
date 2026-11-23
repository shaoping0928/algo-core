"""Tests for the event-span probe: approach leads, rule warning, verdict."""

import unittest

import numpy as np

from event_span_probe import (
    approach_lead_report, band_auc, rule_warning, verdict_of,
)


def _table(length, done_row, event_rows=()):
    done = np.zeros(length, dtype=bool)
    done[int(done_row)] = True
    return dict(
        obs=np.zeros((length, 1), dtype=float),
        actions=np.zeros(length, dtype=int),
        done=done,
        episode=0,
    )


def _mask(length, rows):
    mask = np.zeros(length, dtype=bool)
    mask[list(rows)] = True
    return mask


class ApproachLeadTest(unittest.TestCase):
    def test_first_and_last_instance(self):
        tables = [_table(11, 10), _table(21, 20)]
        masks = [
            [_mask(11, [2, 6]), _mask(21, [])],
        ]
        report = approach_lead_report(masks, ["x1_high"], tables, 16)
        row = report["x1_high"]
        self.assertEqual(row["n_deaths"], 2)
        self.assertEqual(row["n_with_instance"], 1)
        self.assertAlmostEqual(row["coverage"], 0.5)
        self.assertAlmostEqual(row["first_lead_median"], 8.0)
        self.assertAlmostEqual(row["last_lead_median"], 4.0)
        self.assertAlmostEqual(row["share_first_ge_8"], 0.5)
        self.assertAlmostEqual(row["share_first_ge_12"], 0.0)

    def test_instance_outside_horizon_is_censored(self):
        tables = [_table(41, 40)]
        masks = [[_mask(41, [10])]]
        report = approach_lead_report(
            masks, ["x1_high"], tables, horizon=16,
        )
        self.assertEqual(report["x1_high"]["n_with_instance"], 0)
        self.assertAlmostEqual(report["x1_high"]["coverage"], 0.0)


class RuleWarningTest(unittest.TestCase):
    def test_counts_at_horizon_four(self):
        # Death at row 10; one event at row 2.  Fired rows 2-6; positive
        # rows 6-9 (remaining <= 4).
        tables = [_table(11, 10)]
        masks = [[_mask(11, [2])]]
        rule = rule_warning(masks, [0], tables, 4)
        self.assertEqual(rule["tp"], 1)
        self.assertEqual(rule["fp"], 4)
        self.assertEqual(rule["fn"], 3)
        self.assertEqual(rule["tn"], 2)
        self.assertAlmostEqual(rule["precision"], 0.2)
        self.assertAlmostEqual(rule["recall"], 0.25)
        self.assertAlmostEqual(rule["base"], 0.4)
        self.assertAlmostEqual(rule["lift"], 0.5)

    def test_no_events_fires_never(self):
        tables = [_table(11, 10)]
        masks = [[_mask(11, [])]]
        rule = rule_warning(masks, [0], tables, 8)
        self.assertEqual(rule["tp"], 0)
        self.assertEqual(rule["fp"], 0)
        self.assertEqual(rule["fn"], 8)


class BandAucTest(unittest.TestCase):
    def test_band_keeps_near_death_entries(self):
        values = np.asarray([0.9, 0.8, 0.2, 0.1])
        truth = np.asarray([1.0, 1.0, 0.0, 0.0])
        remaining = np.asarray([2, 3, 10, 11])
        self.assertAlmostEqual(
            band_auc(values, truth, remaining, 6), 1.0,
        )
        remaining = np.asarray([2, 3, 20, 21])
        self.assertIsNone(band_auc(values, truth, remaining, 6))


class VerdictTest(unittest.TestCase):
    def test_verdicts(self):
        base = dict(
            early_lead=True, rule_lift=True, span_auc=True,
            span_history=True,
        )
        self.assertEqual(verdict_of(dict(base)), "predicates_sufficient")
        self.assertEqual(
            verdict_of(dict(base, span_auc=False)),
            "new_predicates_needed",
        )
        self.assertEqual(
            verdict_of(dict(base, early_lead=False)),
            "predicates_too_late",
        )
        self.assertEqual(
            verdict_of(dict(base, span_history=False)),
            "state_only_sufficient",
        )


if __name__ == "__main__":
    unittest.main()
