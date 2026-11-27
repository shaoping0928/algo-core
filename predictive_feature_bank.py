"""Episode-balanced, count-based feature discovery for self-supervised data.

The bank deliberately separates broad candidate generation from held-out
certification. Predicates are fitted on training episodes, ranked on an
independent calibration split, and never read test outcomes. The readout uses
episode-balanced empirical rates; unsupported queries remain UNKNOWN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_QUANTILES = (
    0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90,
    0.95, 0.975, 0.99,
)


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    hash(value)
    return value


@dataclass(frozen=True)
class FeatureAtom:
    """One threshold predicate over a task-agnostic numeric channel."""

    dimension: int
    mode: str
    operator: str
    threshold: float

    def mask(self, values):
        values = np.asarray(values, dtype=float)
        column = values[:, int(self.dimension)]
        if self.mode == "abs":
            column = np.abs(column)
        elif self.mode != "raw":
            raise ValueError("feature mode must be raw or abs")
        if self.operator == "<":
            return column < float(self.threshold)
        if self.operator == ">":
            return column > float(self.threshold)
        raise ValueError("feature operator must be < or >")

    def label(self, names=None):
        name = (
            str(self.dimension) if names is None
            else str(names[int(self.dimension)])
        )
        if self.mode == "abs":
            name = "|" + name + "|"
        return "%s %s %.6g" % (name, self.operator, self.threshold)


@dataclass(frozen=True)
class FeatureExpression:
    """An interpretable atom or shallow AND/OR composition."""

    family: str
    operator: str
    atoms: Tuple[FeatureAtom, ...]

    def mask(self, values):
        if not self.atoms:
            return np.zeros(len(values), dtype=bool)
        masks = [atom.mask(values) for atom in self.atoms]
        if self.operator == "atom":
            return masks[0]
        if self.operator == "and":
            return np.logical_and.reduce(masks)
        if self.operator == "or":
            return np.logical_or.reduce(masks)
        raise ValueError("unsupported feature composition")

    def applicable(self, values, ranges):
        values = np.asarray(values, dtype=float)
        result = np.ones(len(values), dtype=bool)
        for dimension, mode, lower, upper in ranges:
            column = values[:, int(dimension)]
            if mode == "abs":
                column = np.abs(column)
            result &= (column >= float(lower)) & (column <= float(upper))
        return result

    @property
    def complexity(self):
        return len(self.atoms)

    def label(self, names=None):
        labels = [atom.label(names) for atom in self.atoms]
        if self.operator == "atom":
            return labels[0]
        joiner = " & " if self.operator == "and" else " | "
        return "(" + joiner.join(labels) + ")"

    @property
    def feature_id(self):
        atoms = tuple(sorted(
            (
                atom.dimension, atom.mode, atom.operator,
                round(float(atom.threshold), 10),
            )
            for atom in self.atoms
        ))
        return "%s:%s:%r" % (self.family, self.operator, atoms)


@dataclass
class FeatureRecord:
    """A candidate's provenance, independent support and validation state."""

    feature_id: str
    expression: FeatureExpression
    train_rows: int
    train_episode_support: int
    train_inactive_episode_support: int
    train_episode_coverage: float
    train_applicability_coverage: float
    train_gain: float
    train_nll_gain: float
    probability_active: float
    probability_inactive: float
    ranges: Tuple[Tuple[int, str, float, float], ...]
    status: str = "train_supported"
    calibration: Optional[dict] = None
    outer_test: Optional[dict] = None

    def snapshot(self, names=None):
        return dict(
            feature_id=self.feature_id,
            family=self.expression.family,
            label=self.expression.label(names),
            complexity=int(self.expression.complexity),
            train_rows=int(self.train_rows),
            train_episode_support=int(self.train_episode_support),
            train_inactive_episode_support=int(
                self.train_inactive_episode_support
            ),
            train_episode_coverage=float(self.train_episode_coverage),
            train_applicability_coverage=float(
                self.train_applicability_coverage
            ),
            train_gain=float(self.train_gain),
            train_nll_gain=float(self.train_nll_gain),
            probability_active=float(self.probability_active),
            probability_inactive=float(self.probability_inactive),
            ranges=self.ranges,
            status=self.status,
            calibration=self.calibration,
            outer_test=self.outer_test,
        )


class EpisodeBalancedFeatureBank:
    """Propose broad predicates, retain supported features, and abstain OOD.

    Probabilities are episode-balanced empirical rates smoothed toward the
    equally weighted episode base rate. Frames within an episode contribute
    to that episode's rate but do not increase independent episode support.
    """

    def __init__(self, family: str, quantiles=DEFAULT_QUANTILES,
                 min_train_episode_support: int = 3,
                 min_calibration_episode_support: int = 3,
                 min_coverage: float = 0.05,
                 prior_episodes: float = 1.0,
                 max_compose_atoms: int = 16,
                 max_frontier: int = 256,
                 bootstrap_samples: int = 400,
                 seed: int = 0):
        if not family:
            raise ValueError("feature family must be named")
        if int(min_train_episode_support) < 1:
            raise ValueError("minimum train support must be positive")
        if int(min_calibration_episode_support) < 1:
            raise ValueError("minimum calibration support must be positive")
        if not 0.0 <= float(min_coverage) <= 1.0:
            raise ValueError("minimum coverage must be in [0, 1]")
        if float(prior_episodes) < 0.0:
            raise ValueError("prior episode mass must be non-negative")
        if int(max_compose_atoms) < 0 or int(max_frontier) < 1:
            raise ValueError("feature budgets are invalid")
        self.family = str(family)
        self.quantiles = tuple(sorted(set(float(q) for q in quantiles)))
        if not self.quantiles or any(q <= 0.0 or q >= 1.0
                                     for q in self.quantiles):
            raise ValueError("quantiles must lie strictly between 0 and 1")
        self.min_train_episode_support = int(min_train_episode_support)
        self.min_calibration_episode_support = int(
            min_calibration_episode_support
        )
        self.min_coverage = float(min_coverage)
        self.prior_episodes = float(prior_episodes)
        self.max_compose_atoms = int(max_compose_atoms)
        self.max_frontier = int(max_frontier)
        self.bootstrap_samples = int(bootstrap_samples)
        self.seed = int(seed)
        self.records = []
        self.selected_ids = ()
        self.frontier_ids = ()
        self.feature_names = None
        self.input_dimensions = None
        self.base_rate = None
        self.fitted = False

    @staticmethod
    def _validate(values, labels, episode_ids):
        values = np.asarray(values, dtype=float)
        labels = np.asarray(labels, dtype=float).reshape(-1)
        native_ids = [_native(value) for value in episode_ids]
        ids = np.empty(len(native_ids), dtype=object)
        ids[:] = native_ids
        if values.ndim != 2 or len(values) != len(labels):
            raise ValueError("values and labels must be aligned matrices")
        if len(ids) != len(labels):
            raise ValueError("episode IDs must have one entry per row")
        if (not np.isfinite(values).all()
                or not np.isfinite(labels).all()):
            raise ValueError("feature values and labels must be finite")
        if np.any((labels < 0.0) | (labels > 1.0)):
            raise ValueError("labels must be in [0, 1]")
        return values, labels, ids

    @staticmethod
    def _episode_indices(episode_ids):
        indices = {}
        for row_index, episode in enumerate(episode_ids):
            indices.setdefault(episode, []).append(row_index)
        return {
            episode: np.asarray(rows, dtype=int)
            for episode, rows in indices.items()
        }

    @staticmethod
    def _rates(mask, labels, episode_indices, prior_episodes):
        active_rates = []
        inactive_rates = []
        active_support = 0
        inactive_support = 0
        active_rows = int(mask.sum())
        for indices in episode_indices.values():
            local_mask = mask[indices]
            local_labels = labels[indices]
            if local_mask.any():
                active_support += 1
                active_rates.append(float(local_labels[local_mask].mean()))
            if (~local_mask).any():
                inactive_support += 1
                inactive_rates.append(
                    float(local_labels[~local_mask].mean())
                )
        base = float(np.mean([
            float(labels[indices].mean())
            for indices in episode_indices.values()
        ]))

        def smooth(rates):
            if not rates:
                return base
            denominator = float(len(rates)) + float(prior_episodes)
            if denominator <= 0.0:
                return float(np.mean(rates))
            return (
                float(np.sum(rates))
                + float(prior_episodes) * base
            ) / denominator

        return dict(
            base=base,
            active=smooth(active_rates),
            inactive=smooth(inactive_rates),
            active_episode_support=int(active_support),
            inactive_episode_support=int(inactive_support),
            active_rows=active_rows,
            episode_count=int(len(episode_indices)),
            episode_coverage=(
                float(active_support) / float(len(episode_indices))
                if episode_indices else 0.0
            ),
        )

    @staticmethod
    def _predict_mask(mask, active_probability, inactive_probability):
        return np.where(
            mask, float(active_probability), float(inactive_probability),
        )

    @staticmethod
    def _episode_metric_gains(
            prediction, labels, episode_indices, base_rate):
        brier_gains = []
        nll_gains = []
        clipped_base = float(np.clip(base_rate, 1e-6, 1.0 - 1e-6))
        for indices in episode_indices.values():
            truth = labels[indices]
            pred = np.clip(prediction[indices], 1e-6, 1.0 - 1e-6)
            baseline_brier = (truth - float(base_rate)) ** 2
            feature_brier = (truth - pred) ** 2
            baseline_nll = -(
                truth * np.log(clipped_base)
                + (1.0 - truth) * np.log(1.0 - clipped_base)
            )
            feature_nll = -(
                truth * np.log(pred)
                + (1.0 - truth) * np.log(1.0 - pred)
            )
            brier_gains.append(float(
                np.mean(baseline_brier - feature_brier)
            ))
            nll_gains.append(float(
                np.mean(baseline_nll - feature_nll)
            ))
        return (
            np.asarray(brier_gains, dtype=float),
            np.asarray(nll_gains, dtype=float),
        )

    @staticmethod
    def _bootstrap_interval(gains, samples, seed):
        gains = np.asarray(gains, dtype=float)
        if gains.size < 2:
            return None, None
        rng = np.random.RandomState(int(seed))
        indices = rng.randint(
            0, len(gains), size=(max(int(samples), 1), len(gains)),
        )
        means = gains[indices].mean(axis=1)
        return (
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        )

    @staticmethod
    @staticmethod
    def _episode_weighted_quantiles(values, episode_indices, quantiles):
        row_weights = np.zeros(len(values), dtype=float)
        for indices in episode_indices.values():
            row_weights[indices] = 1.0 / float(len(indices))
        order = np.argsort(values, kind="stable")
        sorted_values = np.asarray(values, dtype=float)[order]
        weights = row_weights[order]
        total = float(weights.sum())
        if total <= 0.0:
            return np.quantile(values, quantiles)
        centers = np.cumsum(weights) - 0.5 * weights
        centers /= total
        return np.interp(
            np.asarray(quantiles, dtype=float),
            centers, sorted_values,
        )

    @classmethod
    def _make_atoms(cls, values, family, quantiles, episode_indices):
        atoms = []
        for dimension in range(values.shape[1]):
            for mode in ("raw", "abs"):
                column = values[:, dimension]
                if mode == "abs":
                    column = np.abs(column)
                thresholds = np.unique(cls._episode_weighted_quantiles(
                    column, episode_indices, quantiles,
                )
                )
                for threshold in thresholds:
                    for operator in ("<", ">"):
                        atom = FeatureAtom(
                            dimension=int(dimension),
                            mode=mode,
                            operator=operator,
                            threshold=float(threshold),
                        )
                        mask = atom.mask(values)
                        if mask.any() and (~mask).any():
                            atoms.append((atom, mask))
        return atoms

    def _record_for(self, expression, mask, values, labels,
                    episode_ids, episode_indices):
        if int(mask.sum()) <= 0 or not (~mask).any():
            return None
        rates = self._rates(
            mask, labels, episode_indices, self.prior_episodes,
        )
        if rates["active_episode_support"] < (
                self.min_train_episode_support):
            return None
        if rates["inactive_episode_support"] < (
                self.min_train_episode_support):
            return None
        prediction = self._predict_mask(
            mask, rates["active"], rates["inactive"],
        )
        brier_gains, nll_gains = self._episode_metric_gains(
            prediction, labels, episode_indices, rates["base"],
        )
        channels = sorted({
            (atom.dimension, atom.mode)
            for atom in expression.atoms
        })
        ranges = []
        for dimension, mode in channels:
            column = values[:, int(dimension)]
            if mode == "abs":
                column = np.abs(column)
            ranges.append((
                int(dimension), mode,
                float(np.min(column)), float(np.max(column)),
            ))
        return FeatureRecord(
            feature_id=expression.feature_id,
            expression=expression,
            train_rows=int(mask.sum()),
            train_episode_support=rates["active_episode_support"],
            train_inactive_episode_support=(
                rates["inactive_episode_support"]
            ),
            train_episode_coverage=rates["episode_coverage"],
            train_applicability_coverage=1.0,
            train_gain=(
                float(brier_gains.mean()) if brier_gains.size else 0.0
            ),
            train_nll_gain=(
                float(nll_gains.mean()) if nll_gains.size else 0.0
            ),
            probability_active=float(rates["active"]),
            probability_inactive=float(rates["inactive"]),
            ranges=tuple(ranges),
        )

    def fit(self, values, labels, episode_ids,
            feature_names: Optional[Sequence[str]] = None):
        values, labels, episode_ids = self._validate(
            values, labels, episode_ids,
        )
        if values.shape[1] == 0:
            raise ValueError("feature bank needs at least one dimension")
        if feature_names is not None and len(feature_names) != values.shape[1]:
            raise ValueError("feature_names must match the input width")
        episode_indices = self._episode_indices(episode_ids)
        if len(episode_indices) < self.min_train_episode_support:
            raise ValueError("not enough independent training episodes")
        self.feature_names = (
            None if feature_names is None else tuple(feature_names)
        )
        self.input_dimensions = int(values.shape[1])
        self.base_rate = float(np.mean([
            labels[indices].mean() for indices in episode_indices.values()
        ]))

        atoms = self._make_atoms(
            values, self.family, self.quantiles, episode_indices,
        )
        records = []
        seen_masks = set()
        atom_records = []
        for atom, mask in atoms:
            signature = mask.tobytes()
            if signature in seen_masks:
                continue
            seen_masks.add(signature)
            expression = FeatureExpression(
                self.family, "atom", (atom,),
            )
            record = self._record_for(
                expression, mask, values, labels,
                episode_ids, episode_indices,
            )
            if record is not None:
                records.append(record)
                atom_records.append((record, mask))

        if self.max_compose_atoms > 1 and atom_records:
            atom_records.sort(
                key=lambda item: (
                    item[0].train_gain,
                    item[0].train_episode_support,
                    -item[0].expression.complexity,
                ),
                reverse=True,
            )
            top_atoms = atom_records[:self.max_compose_atoms]
            for first in range(len(top_atoms)):
                record_a, mask_a = top_atoms[first]
                atom_a = record_a.expression.atoms[0]
                for second in range(first + 1, len(top_atoms)):
                    record_b, mask_b = top_atoms[second]
                    atom_b = record_b.expression.atoms[0]
                    if atom_a.dimension == atom_b.dimension:
                        continue
                    expression = FeatureExpression(
                        self.family, "and", (atom_a, atom_b),
                    )
                    mask = mask_a & mask_b
                    signature = mask.tobytes()
                    if signature in seen_masks:
                        continue
                    seen_masks.add(signature)
                    record = self._record_for(
                        expression, mask, values, labels,
                        episode_ids, episode_indices,
                    )
                    if record is not None:
                        records.append(record)

            # Bands are expressed as two generic thresholds on one channel.
            by_channel = {}
            for record, mask in top_atoms:
                atom = record.expression.atoms[0]
                by_channel.setdefault(
                    (atom.dimension, atom.mode), [],
                ).append((atom, mask))
            for channel_atoms in by_channel.values():
                lower = [item for item in channel_atoms
                         if item[0].operator == "<"]
                upper = [item for item in channel_atoms
                         if item[0].operator == ">"]
                for atom_low, mask_low in lower:
                    for atom_high, mask_high in upper:
                        if atom_low.threshold >= atom_high.threshold:
                            continue
                        expression = FeatureExpression(
                            self.family, "or", (atom_low, atom_high),
                        )
                        mask = mask_low | mask_high
                        signature = mask.tobytes()
                        if signature in seen_masks:
                            continue
                        seen_masks.add(signature)
                        record = self._record_for(
                            expression, mask, values, labels,
                            episode_ids, episode_indices,
                        )
                        if record is not None:
                            records.append(record)

        self.records = records
        self.fitted = True
        self.selected_ids = ()
        self.frontier_ids = ()
        return self

    def _record_map(self):
        return {record.feature_id: record for record in self.records}

    @staticmethod
    def _dominates(first, second):
        coverage_a, gain_a = first
        coverage_b, gain_b = second
        return (
            coverage_a >= coverage_b and gain_a >= gain_b
            and (coverage_a > coverage_b or gain_a > gain_b)
        )

    def calibrate(self, values, labels, episode_ids):
        """Rank candidates on calibration episodes; never fit new thresholds."""
        if not self.fitted:
            raise RuntimeError("fit the feature bank before calibration")
        values, labels, episode_ids = self._validate(
            values, labels, episode_ids,
        )
        episode_indices = self._episode_indices(episode_ids)
        if not episode_indices:
            raise ValueError("calibration requires at least one episode")

        eligible = []
        for record in self.records:
            predicate = record.expression.mask(values)
            applicable = record.expression.applicable(
                values, record.ranges,
            )
            mask = predicate & applicable
            inactive = (~predicate) & applicable
            prediction = np.full(
                len(values), float(self.base_rate), dtype=float,
            )
            prediction[applicable] = self._predict_mask(
                predicate[applicable], record.probability_active,
                record.probability_inactive,
            )
            brier_gains, nll_gains = self._episode_metric_gains(
                prediction,
                labels, episode_indices, self.base_rate,
            )
            active_support = sum(
                1 for indices in episode_indices.values()
                if mask[indices].any()
            )
            inactive_support = sum(
                1 for indices in episode_indices.values()
                if inactive[indices].any()
            )
            applicable_support = sum(
                1 for indices in episode_indices.values()
                if applicable[indices].any()
            )
            coverage = (
                float(applicable_support) / float(len(episode_indices))
            )
            can_compete = (
                active_support >= self.min_calibration_episode_support
                and inactive_support >= self.min_calibration_episode_support
                and coverage >= self.min_coverage
                and brier_gains.size >= 2
                and float(brier_gains.mean()) > 0.0
            )
            if can_compete:
                ci_low, ci_high = self._bootstrap_interval(
                    brier_gains, self.bootstrap_samples,
                    self.seed + len(eligible),
                )
            else:
                ci_low, ci_high = None, None
            record.calibration = dict(
                active_rows=int(mask.sum()),
                inactive_rows=int(inactive.sum()),
                applicable_rows=int(applicable.sum()),
                episode_support=int(active_support),
                inactive_episode_support=int(inactive_support),
                applicable_episode_support=int(applicable_support),
                active_episode_coverage=(
                    float(active_support) / float(len(episode_indices))
                ),
                episode_coverage=coverage,
                mean_episode_brier_gain=(
                    float(brier_gains.mean()) if brier_gains.size else 0.0
                ),
                mean_episode_nll_gain=(
                    float(nll_gains.mean()) if nll_gains.size else 0.0
                ),
                episode_gain_ci_low=ci_low,
                episode_gain_ci_high=ci_high,
                episodes=int(len(episode_indices)),
            )
            record.status = "calibration_candidate"
            if can_compete:
                eligible.append(record)

        frontier = []
        for candidate in eligible:
            point = (
                candidate.calibration["episode_coverage"],
                candidate.calibration["mean_episode_brier_gain"],
            )
            if any(self._dominates(
                    (
                        other.calibration["episode_coverage"],
                        other.calibration["mean_episode_brier_gain"],
                    ),
                    point,
            ) for other in eligible if other is not candidate):
                continue
            frontier.append(candidate)
        frontier.sort(
            key=lambda record: (
                record.calibration["episode_gain_ci_low"]
                if record.calibration["episode_gain_ci_low"] is not None
                else -np.inf,
                record.calibration["episode_coverage"],
                record.calibration["mean_episode_brier_gain"],
            ),
            reverse=True,
        )
        frontier = frontier[:self.max_frontier]
        for record in frontier:
            if (record.calibration["episode_gain_ci_low"] is not None
                    and record.calibration["episode_gain_ci_low"] > 0.0
                    and record.calibration["mean_episode_nll_gain"] > 0.0):
                record.status = "calibration_supported"
            else:
                record.status = "provisional"
        self.frontier_ids = tuple(
            record.feature_id for record in frontier
        )
        self.selected_ids = tuple(
            record.feature_id for record in frontier
            if record.status == "calibration_supported"
        )
        return dict(
            candidates=int(len(self.records)),
            calibration_eligible=int(len(eligible)),
            pareto_frontier=int(len(frontier)),
            calibration_supported=int(len(self.selected_ids)),
            selected_ids=self.selected_ids,
            frontier_ids=self.frontier_ids,
            frontier=[
                record.snapshot(self.feature_names) for record in frontier
            ],
        )

    def evaluate(self, values, labels, episode_ids, feature_ids=None):
        """Evaluate already selected candidates on untouched outer episodes."""
        if not self.fitted:
            raise RuntimeError("fit the feature bank before evaluation")
        values, labels, episode_ids = self._validate(
            values, labels, episode_ids,
        )
        episode_indices = self._episode_indices(episode_ids)
        ids = tuple(
            self.selected_ids if feature_ids is None else feature_ids
        )
        records = self._record_map()
        output = []
        for feature_id in ids:
            record = records.get(feature_id)
            if record is None:
                continue
            calibration_status = record.status
            predicate = record.expression.mask(values)
            applicable = record.expression.applicable(
                values, record.ranges,
            )
            mask = predicate & applicable
            inactive = (~predicate) & applicable
            prediction = np.full(
                len(values), float(self.base_rate), dtype=float,
            )
            prediction[applicable] = self._predict_mask(
                predicate[applicable], record.probability_active,
                record.probability_inactive,
            )
            brier_gains, nll_gains = self._episode_metric_gains(
                prediction,
                labels, episode_indices, self.base_rate,
            )
            active_support = sum(
                1 for indices in episode_indices.values()
                if mask[indices].any()
            )
            inactive_support = sum(
                1 for indices in episode_indices.values()
                if inactive[indices].any()
            )
            applicable_support = sum(
                1 for indices in episode_indices.values()
                if applicable[indices].any()
            )
            ci_low, ci_high = self._bootstrap_interval(
                brier_gains, self.bootstrap_samples,
                self.seed + len(output) + 10000,
            )
            metrics = dict(
                active_rows=int(mask.sum()),
                inactive_rows=int(inactive.sum()),
                applicable_rows=int(applicable.sum()),
                episode_support=int(active_support),
                inactive_episode_support=int(inactive_support),
                applicable_episode_support=int(applicable_support),
                active_episode_coverage=(
                    float(active_support) / float(len(episode_indices))
                    if episode_indices else 0.0
                ),
                episode_coverage=(
                    float(applicable_support) / float(len(episode_indices))
                    if episode_indices else 0.0
                ),
                mean_episode_brier_gain=(
                    float(brier_gains.mean())
                    if brier_gains.size else 0.0
                ),
                mean_episode_nll_gain=(
                    float(nll_gains.mean()) if nll_gains.size else 0.0
                ),
                episode_gain_ci_low=ci_low,
                episode_gain_ci_high=ci_high,
                episodes=int(len(episode_indices)),
            )
            record.outer_test = metrics
            test_supported = bool(
                ci_low is not None and ci_low > 0.0
                and metrics["mean_episode_nll_gain"] > 0.0
            )
            if calibration_status == "calibration_supported":
                record.status = (
                    "outer_validated" if test_supported
                    else "outer_not_replicated"
                )
            output.append(dict(
                feature_id=feature_id,
                label=record.expression.label(self.feature_names),
                calibration_status=calibration_status,
                status=record.status,
                outer_test=metrics,
                outer_test_supported=test_supported,
            ))
        return output

    def predict(self, values):
        """Read validated active features; return UNKNOWN when none apply."""
        if not self.fitted:
            raise RuntimeError("fit the feature bank before prediction")
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("values must be a vector or matrix")
        if values.shape[1] != self.input_dimensions:
            raise ValueError("values width differs from fitted feature bank")
        records = self._record_map()
        selected = [
            records[feature_id] for feature_id in self.selected_ids
            if feature_id in records
        ]
        output = []
        for row in values:
            applicable = [
                record for record in selected
                if bool(record.expression.applicable(
                    row.reshape(1, -1), record.ranges,
                )[0])
            ]
            if not applicable:
                output.append(dict(
                    status="unknown", features=(), probability=None,
                ))
                continue
            probabilities = [
                (
                    record.probability_active
                    if bool(record.expression.mask(row.reshape(1, -1))[0])
                    else record.probability_inactive
                )
                for record in applicable
            ]
            output.append(dict(
                status="supported",
                features=tuple(record.feature_id for record in applicable),
                active_features=tuple(
                    record.feature_id for record in applicable
                    if bool(record.expression.mask(row.reshape(1, -1))[0])
                ),
                probability=float(np.mean(probabilities)),
            ))
        return tuple(output)

    def snapshot(self):
        return dict(
            family=self.family,
            dimensions=self.input_dimensions,
            candidates=int(len(self.records)),
            frontier=int(len(self.frontier_ids)),
            selected=int(len(self.selected_ids)),
            min_train_episode_support=self.min_train_episode_support,
            min_calibration_episode_support=(
                self.min_calibration_episode_support
            ),
            min_coverage=self.min_coverage,
            prior_episodes=self.prior_episodes,
        )
