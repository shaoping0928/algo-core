"""Context-bucketed 0/1 death rules.

A single rule set assumes one terminal regime.  If the same coordinates
lead to different terminal outcomes in different contexts, fit one rule
set per context bucket and fall back to the global set for buckets with
too few terminal examples.  The context is a caller-supplied scalar
(residual, distance, uncertainty, ...); this module never interprets it
and never sees task labels.

The output stays 0/1/UNKNOWN: no probability is introduced.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from death_rules import UNKNOWN, DeathRuleSet


class ContextualDeathRuleSet:
    """A global rule set plus sparse per-bucket specialisations.

    ``fit`` learns quantile context cuts from the training contexts, fits
    a ``DeathRuleSet`` per bucket that has at least ``min_positive``
    terminal examples, and keeps the global set for the others.  A state
    is classified by its bucket's set when present, otherwise by the
    global set.  ``refine`` appends a counterexample with its context and
    re-fits, so a new regime can be introduced without task-specific
    geometry.
    """

    def __init__(self, buckets: int = 2, min_positive: int = 2,
                 fallback_global: bool = True,
                 thresholds: Optional[list] = None, **rule_kwargs):
        if buckets < 1:
            raise ValueError("buckets must be positive")
        if min_positive < 1:
            raise ValueError("min_positive must be positive")
        self.buckets = int(buckets)
        self.min_positive = int(min_positive)
        self.fallback_global = bool(fallback_global)
        self.thresholds_ = (
            None if thresholds is None
            else np.asarray(sorted(float(v) for v in thresholds), dtype=float)
        )
        self.rule_kwargs = dict(rule_kwargs)
        self.global_: Optional[DeathRuleSet] = None
        self.bucket_rules_: dict = {}
        self.bucket_stats_: dict = {}
        self.states_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.context_: Optional[np.ndarray] = None
        self.refinement_log: List[dict] = []

    # ---- fitting ----------------------------------------------------------
    def fit(self, states: np.ndarray, labels: np.ndarray,
            context: np.ndarray) -> "ContextualDeathRuleSet":
        states = np.asarray(states, dtype=float)
        labels = np.asarray(labels, dtype=float).reshape(-1)
        context = np.asarray(context, dtype=float).reshape(-1)
        if len(states) != len(labels) or len(states) != len(context):
            raise ValueError("states/labels/context shape mismatch")
        if len(states) == 0:
            raise ValueError("need at least one state")
        self.states_ = states.copy()
        self.labels_ = (labels > 0.5).astype(int)
        self.context_ = context.copy()
        self.global_ = DeathRuleSet(**self.rule_kwargs).fit(
            self.states_, self.labels_
        )
        if self.thresholds_ is None:
            if self.buckets > 1:
                quantiles = np.linspace(
                    0.0, 1.0, self.buckets + 1
                )[1:-1]
                unique = np.unique(context)
                cuts = []
                for quantile in quantiles:
                    value = float(np.quantile(context, quantile))
                    index = int(np.searchsorted(unique, value, side="left"))
                    if index <= 0:
                        if len(unique) > 1:
                            value = float((unique[0] + unique[1]) / 2.0)
                        else:
                            value = float(unique[0])
                    elif index >= len(unique):
                        if len(unique) > 1:
                            value = float((
                                unique[-2] + unique[-1]
                            ) / 2.0)
                    else:
                        value = float((
                            unique[index - 1] + unique[index]
                        ) / 2.0)
                    cuts.append(value)
                self.thresholds_ = np.asarray(
                    sorted(set(cuts)), dtype=float
                )
            else:
                self.thresholds_ = np.asarray([], dtype=float)
        bucket = np.searchsorted(
            self.thresholds_, context, side="left"
        )
        self.bucket_rules_ = {}
        self.bucket_stats_ = {}
        for index in range(self.buckets):
            mask = bucket == index
            positives = int(self.labels_[mask].sum())
            self.bucket_stats_[index] = dict(
                total=int(mask.sum()),
                positives=positives,
                fitted=False,
            )
            if positives < self.min_positive or not mask.any():
                continue
            rule_set = DeathRuleSet(**self.rule_kwargs).fit(
                self.states_[mask], self.labels_[mask]
            )
            self.bucket_rules_[index] = rule_set
            self.bucket_stats_[index]["fitted"] = True
            self.bucket_stats_[index]["coverage"] = rule_set.coverage_
        return self

    def add_examples(self, states: np.ndarray, labels: np.ndarray,
                     context: np.ndarray) -> None:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        labels = np.asarray(labels, dtype=float).reshape(-1)
        context = np.asarray(context, dtype=float).reshape(-1)
        if len(states) != len(labels) or len(states) != len(context):
            raise ValueError("states/labels/context shape mismatch")
        if self.states_ is None:
            self.fit(states, labels, context)
            return
        self.fit(
            np.vstack([self.states_, states]),
            np.concatenate([self.labels_, (labels > 0.5).astype(int)]),
            np.concatenate([self.context_, context]),
        )

    def refine(self, state: np.ndarray, label: int,
               context: float) -> int:
        state = np.asarray(state, dtype=float).reshape(1, -1)
        before = int(self.classify(state, [context])[0])
        self.add_examples(state, [label], [context])
        after = int(self.classify(state, [context])[0])
        self.refinement_log.append(dict(
            label=int(label), context=float(context),
            before=before, after=after,
        ))
        return after

    # ---- reading ----------------------------------------------------------
    def _bucket(self, context: np.ndarray) -> np.ndarray:
        return np.searchsorted(
            self.thresholds_,
            np.asarray(context, dtype=float).reshape(-1),
            side="left",
        )

    def classify(self, states: np.ndarray,
                 context: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        context = np.asarray(context, dtype=float).reshape(-1)
        if len(context) != len(states):
            raise ValueError("one context value per state is required")
        buckets = self._bucket(context)
        result = np.full(len(states), UNKNOWN, dtype=int)
        for index in range(len(states)):
            rule_set = self.bucket_rules_.get(int(buckets[index]))
            if rule_set is None:
                if not self.fallback_global:
                    continue
                rule_set = self.global_
            result[index] = int(rule_set.classify(states[index:index + 1])[0])
        return result

    def classify_death(self, states: np.ndarray,
                       context: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        context = np.asarray(context, dtype=float).reshape(-1)
        if len(context) != len(states):
            raise ValueError("one context value per state is required")
        buckets = self._bucket(context)
        result = np.full(len(states), UNKNOWN, dtype=int)
        for index in range(len(states)):
            rule_set = self.bucket_rules_.get(int(buckets[index]))
            if rule_set is None:
                if not self.fallback_global:
                    continue
                rule_set = self.global_
            result[index] = int(
                rule_set.classify_death(states[index:index + 1])[0]
            )
        return result

    def coverage_report(self) -> dict:
        if self.global_ is None:
            raise RuntimeError("fit must be called before coverage_report")
        global_report = self.global_.coverage_report()
        return dict(
            global_=global_report,
            buckets=self.bucket_stats_,
            thresholds=(
                [float(value) for value in self.thresholds_]
                if self.thresholds_ is not None else []
            ),
        )
