"""Task-agnostic compositional feature rules over input coordinates.

A primitive is a test on one input coordinate (raw or magnitude).  A composite
is an AND/OR of primitives or smaller composites.  Rules are composites that
co-occur with the world pain signal and survive a held-out split.

The layer never knows what the coordinates mean.  It only receives the input
matrix and the pain labels.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

QUANTILES = tuple(np.unique(np.concatenate([
    np.linspace(0.01, 0.99, 49),
    np.asarray([0.001, 0.002, 0.003, 0.005, 0.0075]),
    np.asarray([0.9925, 0.995, 0.997, 0.998, 0.999]),
])))


@dataclass(frozen=True)
class Primitive:
    dim: int
    magnitude: bool
    op: str
    threshold: float

    def mask(self, x: np.ndarray) -> np.ndarray:
        values = np.abs(x[:, self.dim]) if self.magnitude \
            else x[:, self.dim]
        return values < self.threshold if self.op == "<" \
            else values > self.threshold

    def label(self) -> str:
        name = ("|f%d|" if self.magnitude else "f%d") % self.dim
        return "%s %s %.3f" % (name, self.op, self.threshold)

    def complexity(self) -> int:
        return 1


@dataclass(frozen=True)
class Composite:
    op: str
    children: Tuple[object, ...]

    def mask(self, x: np.ndarray) -> np.ndarray:
        if not self.children:
            return np.zeros(len(x), dtype=bool)
        result = self.children[0].mask(x)
        for child in self.children[1:]:
            if self.op == "and":
                result = result & child.mask(x)
            elif self.op == "or":
                result = result | child.mask(x)
            else:
                raise ValueError("unknown composite op %s" % self.op)
        return result

    def label(self) -> str:
        joiner = " & " if self.op == "and" else " | "
        return "(" + joiner.join(child.label() for child in self.children) + ")"

    def complexity(self) -> int:
        return 1 + sum(child.complexity() for child in self.children)


@dataclass
class FeatureRule:
    feature: object
    support: int
    hits: float
    precision: float
    test_support: int
    test_hits: float
    test_precision: float
    confidence: float

    def label(self) -> str:
        return "%s n=%d p=%.3f | test n=%d p=%.3f" % (
            self.feature.label(), self.support, self.precision,
            self.test_support, self.test_precision,
        )


def _stats(mask: np.ndarray, target: np.ndarray) -> dict:
    support = int(mask.sum())
    if support <= 0:
        return dict(support=0, hits=0.0, precision=0.0, z=0.0)
    hits = float(target[mask].sum())
    precision = hits / support
    base = float(target.mean()) if len(target) else 0.0
    variance = max(base * (1.0 - base), 1e-12)
    z = (hits - support * base) / np.sqrt(support * variance)
    return dict(support=support, hits=hits, precision=precision, z=float(z))


def expand_observation(x: np.ndarray) -> np.ndarray:
    """Append every pairwise coordinate difference to the observation.

    The expansion is purely algebraic: for an observation vector
    ``x_0 .. x_{d-1}`` it appends ``x_i - x_j`` for every ``i < j``.  It
    carries no task semantics - the indices are array positions - and it
    gives the threshold grammar coordinates whose values are centred near
    zero and spread over a wider range than the raw absolute coordinates.
    """
    values = np.asarray(x, dtype=float)
    single = values.ndim == 1
    if single:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("observations must be a vector or a matrix")
    width = values.shape[1]
    if width < 2:
        return values[0] if single else values
    parts = [values]
    for first in range(width):
        for second in range(first + 1, width):
            parts.append(
                (values[:, first] - values[:, second])[:, None]
            )
    expanded = np.concatenate(parts, axis=1)
    return expanded[0] if single else expanded


class FeatureRuleBank:
    """Induce and store compositional pain rules.

    Thresholds are selected from the training data, compositions are searched
    over AND/OR, and every candidate is re-checked on held-out states.
    """

    def __init__(self, max_rules: int = 8, min_support: int = 5,
                 max_per_group: int = 8, min_z: float = 2.0,
                 min_lift: float = 2.0, min_coverage: float = 0.0,
                 min_primitive_coverage: float = 0.0,
                 pairwise_features: bool = True,
                 max_candidates: int = 240):
        self.max_rules = int(max_rules)
        self.min_support = int(min_support)
        self.max_per_group = int(max_per_group)
        self.min_z = float(min_z)
        self.min_lift = float(min_lift)
        # Coverage gates (fractions of the validation states): a rule must
        # fire on enough states to be testable, and every conjunct/branch
        # primitive must fire on enough of them that the composite is not
        # dead on arrival.  Precision alone keeps extreme thresholds that
        # only ever see rare planner states.
        self.min_coverage = float(min_coverage)
        self.min_primitive_coverage = float(min_primitive_coverage)
        # Generic coordinate difference expansion (task-agnostic algebra).
        self.pairwise_features = bool(pairwise_features)
        # Budget on the univariate candidate pool that feeds the AND/OR
        # search; the pairwise expansion multiplies the coordinate count,
        # and the composition step is quadratic in this pool.  Selection
        # is by the same univariate z used everywhere else.
        self.max_candidates = int(max_candidates)
        self.rules: List[FeatureRule] = []

    # ---- candidates -------------------------------------------------------
    def _primitives(self, x: np.ndarray, target: np.ndarray,
                    min_support: int = None):
        base = float(target.mean()) if len(target) else 0.0
        min_support = (
            self.min_support if min_support is None else int(min_support)
        )
        positive = target > 0.5
        groups = {}
        for dim in range(x.shape[1]):
            for magnitude in (False, True):
                values = np.abs(x[:, dim]) if magnitude else x[:, dim]
                thresholds = [np.quantile(values, QUANTILES)]
                forced_values = set()
                if positive.any():
                    pain_thresholds = np.quantile(
                        values[positive],
                        np.linspace(0.0, 1.0, 11),
                    )
                    margins = np.maximum(
                        np.abs(pain_thresholds) * 1e-3, 1e-6
                    )
                    expanded = np.concatenate([
                        pain_thresholds,
                        pain_thresholds + margins,
                        pain_thresholds - margins,
                    ])
                    thresholds.append(expanded)
                    forced_values.update(
                        float(value) for value in expanded
                    )
                if (~positive).any():
                    thresholds.append(np.quantile(
                        values[~positive],
                        np.linspace(0.0, 1.0, 11),
                    ))
                thresholds = np.unique(np.concatenate(thresholds))
                for op in ("<", ">"):
                    items = []
                    for threshold in thresholds:
                        predicate = Primitive(
                            dim, magnitude, op, float(threshold)
                        )
                        mask = predicate.mask(x)
                        if mask.all() or not mask.any():
                            continue
                        stats = _stats(mask, target)
                        if stats["support"] < min_support:
                            continue
                        if stats["precision"] <= base:
                            continue
                        items.append((stats["z"], predicate, mask))
                    sorted_by_z = sorted(
                        items, key=lambda item: item[0], reverse=True
                    )
                    selected = list(sorted_by_z[:self.max_per_group])
                    precise = sorted(
                        items,
                        key=lambda item: (
                            float(target[item[2]].mean()),
                            int(item[2].sum()),
                        ),
                        reverse=True,
                    )
                    selected.extend(precise[:self.max_per_group])
                    selected.extend(
                        item for item in items
                        if float(item[1].threshold) in forced_values
                    )
                    unique = {}
                    for item in selected:
                        unique.setdefault(item[2].tobytes(), item)
                    groups[(dim, magnitude, op)] = list(unique.values())
        unique = {}
        for items in groups.values():
            for z, predicate, mask in items:
                unique.setdefault(mask.tobytes(), (z, predicate, mask))
        return list(unique.values())

    # ---- induction --------------------------------------------------------
    def induce(self, train_x, train_y, test_x=None, test_y=None,
               coverage_x=None):
        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float).reshape(-1)
        if train_x.ndim != 2 or len(train_x) != len(train_y):
            raise ValueError("train_x/train_y shape mismatch")
        test_x = train_x if test_x is None else np.asarray(test_x, float)
        test_y = train_y if test_y is None else np.asarray(
            test_y, dtype=float
        ).reshape(-1)
        gate_x = (
            test_x if coverage_x is None
            else np.asarray(coverage_x, dtype=float)
        )
        if self.pairwise_features:
            train_x = expand_observation(train_x)
            test_x = expand_observation(test_x)
            gate_x = expand_observation(gate_x)
        base_train = float(train_y.mean()) if len(train_y) else 0.0
        base_test = float(test_y.mean()) if len(test_y) else 0.0
        positives = train_y > 0.5
        min_support = max(
            2, min(self.min_support, max(2, int(positives.sum()) // 8))
        )

        candidates = self._primitives(
            train_x, train_y, min_support=min_support
        )
        if len(candidates) > self.max_candidates:
            candidates = sorted(
                candidates, key=lambda item: item[0], reverse=True
            )[:self.max_candidates]
        considered = []
        seen = set()
        primitive_rates = {}

        def primitive_rate(predicate):
            if predicate not in primitive_rates:
                primitive_rates[predicate] = (
                    float(predicate.mask(gate_x).mean())
                    if len(gate_x) else 0.0
                )
            return primitive_rates[predicate]

        def primitives_of(feature):
            if isinstance(feature, Primitive):
                return [feature]
            result = []
            for child in getattr(feature, "children", ()):
                result.extend(primitives_of(child))
            return result

        def coverage_ok(feature, test_stats):
            if self.min_coverage > 0.0 and len(gate_x):
                coverage = float(feature.mask(gate_x).mean())
                if coverage < self.min_coverage:
                    return False
            if self.min_primitive_coverage > 0.0:
                for primitive in primitives_of(feature):
                    if primitive_rate(primitive) < self.min_primitive_coverage:
                        return False
            return True

        def consider(feature, train_mask):
            key = train_mask.tobytes()
            if key in seen:
                return
            seen.add(key)
            stats = _stats(train_mask, train_y)
            if stats["support"] < min_support:
                return
            if stats["z"] < self.min_z:
                return
            if stats["precision"] <= base_train * self.min_lift:
                return
            test_mask = feature.mask(test_x)
            test_stats = _stats(test_mask, test_y)
            if not coverage_ok(feature, test_stats):
                return
            if test_stats["support"] <= 0:
                # Rare causes can be absent from the held-out split.  Keep
                # them as low-confidence hypotheses instead of discarding
                # every rule that cannot be verified yet.
                test_precision = stats["precision"]
                confidence = 0.5 * (stats["hits"] + 1.0) / (
                    stats["support"] + 2.0
                )
            else:
                if test_stats["precision"] <= base_test * self.min_lift:
                    return
                test_precision = test_stats["precision"]
                confidence = (test_stats["hits"] + 1.0) / (
                    test_stats["support"] + 2.0
                )
            considered.append((FeatureRule(
                feature=feature,
                support=stats["support"],
                hits=stats["hits"],
                precision=stats["precision"],
                test_support=test_stats["support"],
                test_hits=test_stats["hits"],
                test_precision=test_precision,
                confidence=float(confidence),
            ), train_mask))

        for z, predicate, mask in candidates:
            consider(predicate, mask)

        # AND pairs: both parts may be individually weak, so the full
        # candidate pool is paired, not only the strongest singles.
        for first in range(len(candidates)):
            _, predicate_a, mask_a = candidates[first]
            for second in range(first + 1, len(candidates)):
                _, predicate_b, mask_b = candidates[second]
                feature = Composite("and", (predicate_a, predicate_b))
                consider(feature, mask_a & mask_b)

        # OR is restricted to a single coordinate: an "outside a band"
        # pattern is (x < low) OR (x > high), and the two branches share
        # one dimension.
        for first in range(len(candidates)):
            _, predicate_a, mask_a = candidates[first]
            if not isinstance(predicate_a, Primitive):
                continue
            for second in range(first + 1, len(candidates)):
                _, predicate_b, mask_b = candidates[second]
                if not isinstance(predicate_b, Primitive):
                    continue
                if (predicate_a.dim != predicate_b.dim
                        or predicate_a.magnitude != predicate_b.magnitude):
                    continue
                feature = Composite("or", (predicate_a, predicate_b))
                consider(feature, mask_a | mask_b)

        # AND(primitive, OR(primitive, primitive)) captures the general
        # "aligned with one coordinate AND outside a band on another" shape.
        or_features = []
        for first in range(len(candidates)):
            _, predicate_a, mask_a = candidates[first]
            if not isinstance(predicate_a, Primitive):
                continue
            for second in range(first + 1, len(candidates)):
                _, predicate_b, mask_b = candidates[second]
                if not isinstance(predicate_b, Primitive):
                    continue
                if (predicate_a.dim != predicate_b.dim
                        or predicate_a.magnitude != predicate_b.magnitude):
                    continue
                or_features.append((
                    Composite("or", (predicate_a, predicate_b)),
                    mask_a | mask_b,
                ))
        for feature, or_mask in or_features:
            for z, predicate, mask in candidates:
                combined = Composite("and", (predicate, feature))
                consider(combined, mask & or_mask)

        # Coverage-driven selection: each rule must explain positive events
        # that no already-selected rule explains.  This keeps distinct
        # causes instead of letting one high-precision family fill every
        # slot.
        positives = train_y > 0.5
        covered = np.zeros(len(train_y), dtype=bool)
        selected = []
        selected_masks = []
        remaining = list(considered)
        while remaining and len(selected) < self.max_rules:
            best = None
            best_key = None
            best_index = -1
            for index, (rule, mask) in enumerate(remaining):
                new_hits = int((mask & positives & ~covered).sum())
                if new_hits <= 0:
                    continue
                key = (
                    new_hits * rule.precision,
                    rule.test_precision,
                    -rule.feature.complexity(),
                )
                if best_key is None or key > best_key:
                    best = (rule, mask)
                    best_key = key
                    best_index = index
            if best is None:
                break
            selected.append(best[0])
            selected_masks.append(best[1])
            covered |= best[1] & positives
            remaining.pop(best_index)

        # A precise feature can describe a cause that another broad feature
        # already covers when both fire on the same states.  Fill the
        # remaining slots with precise features that introduce a coordinate
        # not already used by the selected rules.
        selected_dims = set()
        selected_primitives = set()
        for rule in selected:
            selected_dims |= feature_dims(rule.feature)
            selected_primitives |= feature_primitives(rule.feature)
        remaining.sort(key=lambda item: (
            feature_primitives(item[0].feature).isdisjoint(
                selected_primitives
            ),
            bool(feature_dims(item[0].feature) - selected_dims),
            item[0].test_precision,
            item[0].confidence,
            item[0].support,
            -item[0].feature.complexity(),
        ), reverse=True)
        for rule, mask in remaining:
            if len(selected) >= self.max_rules:
                break
            if (rule.precision < 0.7 or rule.test_precision < 0.7
                    or rule.confidence < 0.5):
                continue
            primitives = feature_primitives(rule.feature)
            # Refinements of an already selected rule are not a new cause.
            if not primitives.isdisjoint(selected_primitives):
                continue
            redundant = False
            for other in selected_masks:
                union = int((mask | other).sum())
                if union <= 0:
                    continue
                overlap = int((mask & other).sum()) / union
                if overlap > 0.8:
                    redundant = True
                    break
            if redundant:
                continue
            selected.append(rule)
            selected_masks.append(mask)
            selected_dims |= feature_dims(rule.feature)
            selected_primitives |= primitives
        self.rules = selected
        return self.rules

    # ---- use --------------------------------------------------------------
    def activate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.pairwise_features:
            x = expand_observation(x)
        result = np.zeros(len(x), dtype=float)
        for rule in self.rules:
            result = np.maximum(
                result, rule.confidence * rule.feature.mask(x)
            )
        return result

    def describe(self) -> List[str]:
        return [rule.label() for rule in self.rules]

    # ---- persistence ------------------------------------------------------
    def state_dict(self) -> dict:
        return dict(
            max_rules=self.max_rules,
            min_support=self.min_support,
            min_z=self.min_z,
            min_lift=self.min_lift,
            rules=[
                dict(
                    feature=feature_state(rule.feature),
                    support=rule.support,
                    hits=rule.hits,
                    precision=rule.precision,
                    test_support=rule.test_support,
                    test_hits=rule.test_hits,
                    test_precision=rule.test_precision,
                    confidence=rule.confidence,
                )
                for rule in self.rules
            ],
        )

    def load_state_dict(self, state: dict) -> "FeatureRuleBank":
        self.max_rules = int(state.get("max_rules", self.max_rules))
        self.min_support = int(state.get("min_support", self.min_support))
        self.min_z = float(state.get("min_z", self.min_z))
        self.min_lift = float(state.get("min_lift", self.min_lift))
        self.rules = [
            FeatureRule(
                feature=feature_from_state(data["feature"]),
                support=int(data.get("support", 0)),
                hits=float(data.get("hits", 0.0)),
                precision=float(data.get("precision", 0.0)),
                test_support=int(data.get("test_support", 0)),
                test_hits=float(data.get("test_hits", 0.0)),
                test_precision=float(data.get("test_precision", 0.0)),
                confidence=float(data.get("confidence", 0.0)),
            )
            for data in state.get("rules", [])
        ]
        return self


def feature_dims(feature) -> set:
    if isinstance(feature, Primitive):
        return {feature.dim}
    result = set()
    for child in feature.children:
        result |= feature_dims(child)
    return result


def feature_primitives(feature) -> set:
    if isinstance(feature, Primitive):
        return {(
            feature.dim, feature.magnitude, feature.op,
            round(float(feature.threshold), 12),
        )}
    result = set()
    for child in feature.children:
        result |= feature_primitives(child)
    return result


def feature_state(feature):
    if isinstance(feature, Primitive):
        return dict(
            kind="primitive", dim=feature.dim,
            magnitude=feature.magnitude, op=feature.op,
            threshold=feature.threshold,
        )
    return dict(
        kind="composite", op=feature.op,
        children=[feature_state(child) for child in feature.children],
    )


def feature_from_state(state):
    if state["kind"] == "primitive":
        return Primitive(
            int(state["dim"]), bool(state["magnitude"]),
            str(state["op"]), float(state["threshold"]),
        )
    return Composite(
        str(state["op"]),
        tuple(feature_from_state(child) for child in state["children"]),
    )
