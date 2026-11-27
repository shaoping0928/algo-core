"""NodeRuntime readout mixin: compiled feature cells, isotonic calibration, propagation, and valence/value batch readouts."""
from collections import deque
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from branches import (
    BranchBank,
    CurrentBranch,
    PredictionBranch,
    RecentBranch,
)
from node_components import ComponentBank
from node_features import (
    FeatureRuleBank,
    expand_observation,
    feature_from_state,
    feature_state,
)
from node_graph import (
    ASSOCIATION,
    CONCEPT,
    MEMBERSHIP,
    MEMORY,
    TRANSITION_EDGE,
    NodeGraph,
)
from online_feature_learner import OnlineFeatureLearner


class NodeReadoutMixin:
    """Calibrated hazard/value readout and prototype-graph propagation methods."""

    def _compiled_feature_plan(self):
        """Compile materialized feature rules into vectorized primitive arrays."""
        cached = self._cached_readout("feature_plan")
        if cached is not None:
            return cached
        from node_features import Composite, Primitive

        prim_index = {}
        dims = []
        is_abs = []
        is_lt = []
        thresholds = []

        def compile_obj(obj):
            if isinstance(obj, Primitive):
                key = (obj.dim, obj.magnitude, obj.op, obj.threshold)
                idx = prim_index.get(key)
                if idx is None:
                    idx = len(dims)
                    prim_index[key] = idx
                    dims.append(int(obj.dim))
                    is_abs.append(bool(obj.magnitude))
                    is_lt.append(obj.op == "<")
                    thresholds.append(float(obj.threshold))
                return ("prim", idx)
            if isinstance(obj, Composite):
                if obj.op not in ("and", "or"):
                    return ("fallback", obj)
                return (obj.op, tuple(compile_obj(c) for c in obj.children))
            return ("fallback", obj)

        entries = [
            None if node_id not in self._feature_objects
            else compile_obj(self._feature_objects[node_id])
            for node_id in self._feature_all
        ]
        result = (
            np.asarray(dims, dtype=np.intp),
            np.asarray(is_abs, dtype=bool),
            np.asarray(is_lt, dtype=bool),
            np.asarray(thresholds, dtype=float),
            tuple(entries),
        )
        self._store_readout_cache("feature_plan", result)
        return result

    def _feature_mask_table(self, obs: np.ndarray) -> List:
        """Every node's activation over a batch of raw observations.

        The masks are pure threshold comparisons, so one vectorized pass
        over the batch yields exactly the booleans the per-row sweep
        produced - computed once instead of once per row.  With pairwise
        features on, the observation is expanded once before the pass.
        """
        obs = np.asarray(obs, dtype=float)
        if self.pairwise_features:
            obs = expand_observation(obs)
        dims, is_abs, is_lt, thresholds, entries = self._compiled_feature_plan()
        n = len(obs)
        if dims.size:
            vals = obs[:, dims]
            if is_abs.any():
                vals = np.where(is_abs[None, :], np.abs(vals), vals)
            pmasks = np.where(
                is_lt[None, :],
                vals < thresholds[None, :],
                vals > thresholds[None, :],
            )
        else:
            pmasks = None

        def eval_entry(entry):
            kind, payload = entry
            if kind == "prim":
                return pmasks[:, payload]
            if kind == "and":
                if not payload:
                    return np.zeros(n, dtype=bool)
                out = eval_entry(payload[0])
                for child in payload[1:]:
                    out = out & eval_entry(child)
                return out
            if kind == "or":
                if not payload:
                    return np.zeros(n, dtype=bool)
                out = eval_entry(payload[0])
                for child in payload[1:]:
                    out = out | eval_entry(child)
                return out
            return payload.mask(obs)

        return [
            None if entry is None else eval_entry(entry)
            for entry in entries
        ]

    def _active_feature_roots(self, state) -> List[str]:
        active = set(self._active_feature_ids(state))
        return [
            node_id for node_id in self.feature_ids if node_id in active
        ]

    def _all_value_roots(self) -> List[str]:
        """Roots carrying sequence-level value cells.

        Risk-bank roots always do; dedicated value-channel roots join them
        when installed.  Keeping the risk roots lets the learned value of
        the states the pain labels do cover stay readable, while the value
        bank adds resolution everywhere else.
        """
        roots = list(self.feature_ids)
        for node_id in self.value_feature_ids:
            if node_id not in roots:
                roots.append(node_id)
        return roots

    def _subtree_ids(self, root_id: str) -> List[str]:
        """All materialized descendant nodes below a rule root."""
        descendants = []
        stack = list(self._feature_children.get(root_id, ()))
        while stack:
            node_id = stack.pop()
            descendants.append(node_id)
            stack.extend(self._feature_children.get(node_id, ()))
        return descendants

    def _update_feature_pain(self, feature_ids, pain: float,
                             weight: float = 1.0) -> None:
        pain = float(pain)
        weight = float(weight)
        for node_id in feature_ids:
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            # 1/n cumulative mean of the window labels: the root's
            # marginal pain rate.  A constant-rate EMA would leave a
            # residual noise floor that never decays with more data, and
            # every thin cell inherits that noise through this anchor.
            # `weight` scales the effective sample size - the clipped
            # importance weight when importance weighting is on.
            node.count += weight
            node.value += weight * (pain - node.value) / node.count

    def _update_feature_action(self, feature_ids, action: int,
                               pain: float, weight: float = 1.0) -> None:
        """Accumulate P(pain | feature active, acting action) cells.

        `weight` scales the cell's effective sample size - the clipped
        per-decision importance weight when importance weighting is on,
        plain one otherwise.
        """
        pain = float(pain)
        weight = float(weight)
        for node_id in feature_ids:
            cells = self.feature_action_stats.setdefault(node_id, {})
            cell = cells.setdefault(int(action), [0, 0.0])
            cell[0] += weight
            cell[1] += weight * pain

    def _feature_cell_read(self, node_id: str, acting: int):
        """One root's readout value under one acting action, with support."""
        node = self.graph.nodes.get(node_id)
        if node is None:
            return 0.0, 0
        base = float(node.value)
        cell = self.feature_action_stats.get(node_id, {}).get(int(acting))
        if cell is not None and cell[0] > 0:
            prior = self.feature_action_prior
            count, hits = int(cell[0]), float(cell[1])
            return (hits + prior * base) / (count + prior), count
        return base, 0

    def pain_contrast_batch(self, states: np.ndarray,
                            mask_table: Optional[List] = None) -> np.ndarray:
        """Action contrast of the feature risk cells, per encoded state.

        For every active root the per-action risk reads are taken over the
        whole action space and centered on their mean, so a state's read is
        the quality of the acting action *relative to the other action* and
        the band's absolute lethality cancels - a plan that goes through
        the band no longer pays the toll a drifting plan avoids inside a
        short horizon.  Roots without data contribute a zero contrast (their
        fallback read is identical for every action); rows where no root
        fires read zero.
        """
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0)
        result = np.full(len(states), -np.inf)
        any_fired = np.zeros(len(states), dtype=bool)
        if not self.feature_ids:
            return np.zeros(len(states), dtype=float)
        obs = states[:, :self.obs_dim]
        if mask_table is None:
            mask_table = self._feature_mask_table(obs)
        acting = None
        if states.shape[1] > self.obs_dim:
            acting = np.rint(states[:, self.obs_dim]).astype(int)
        if acting is None:
            return np.zeros(len(states), dtype=float)
        prior = self.feature_action_prior
        for node_id in self.feature_ids:
            root_feature = self._feature_objects.get(node_id)
            if root_feature is None:
                continue
            root_mask = self._table_entry(
                mask_table, node_id, root_feature, obs
            )
            if root_mask is None or not root_mask.any():
                continue
            any_fired |= root_mask
            base = float(self.graph.nodes[node_id].value)
            reads = {}
            for action_key in self.action_space:
                cell = self.feature_action_stats.get(node_id, {}).get(
                    int(action_key)
                )
                if cell is not None and cell[0] > 0:
                    count, hits = int(cell[0]), float(cell[1])
                    reads[action_key] = (
                        hits + prior * base
                    ) / (count + prior)
                else:
                    reads[action_key] = base
            mean_read = float(np.mean(
                [reads[action_key] for action_key in self.action_space]
            ))
            for action_key in self.action_space:
                selected = root_mask & (acting == int(action_key))
                if not selected.any():
                    continue
                candidate = np.full(len(states), -np.inf)
                candidate[selected] = reads[action_key] - mean_read
                result = np.maximum(result, candidate)
        return np.where(any_fired, result, 0.0)

    def fit_readout_calibration(self, states: np.ndarray,
                                pending: np.ndarray,
                                labels: np.ndarray,
                                bins: int = 12) -> None:
        """Fit a monotone calibration for the graded per-action reads.

        The graded composition consumes the map one read at a time - each
        (root, action) cell read is mapped, then min/max over the mapped
        reads is recomposed - so the map must be fitted at that same
        granularity: pairs of (raw per-action read, the realized window
        label of the state whose acting action the read prices).  A map
        fitted on the composed threat would calibrate a quantity the
        readout never consumes, because min/max over mapped reads differs
        from the mapped min/max whenever more than one read enters the
        composition.

        Only the acting action's read pairs with a state's label - the
        realized outcome reveals that action's price, not the other
        actions'.  The fit pools all (root, acting action) pairs over the
        training stream so one monotone map is shared across roots;
        per-root maps would have sparse tails.  Monotonicity preserves the
        min/max algebra of the graded decomposition.  (`bins` is retained
        for API compatibility; the estimator is plain isotonic regression
        over the pooled pairs.)
        """
        states = np.asarray(states, dtype=float)
        pending_all = np.asarray(pending, dtype=float)
        if pending_all.shape[0] != len(states):
            raise ValueError("pending must have one row per state")
        labels = np.asarray(labels, dtype=float)
        if len(states) == 0 or not self.feature_ids:
            return
        enc = np.concatenate([states, pending_all], axis=1)
        obs = enc[:, :self.obs_dim]
        acting = None
        if enc.shape[1] > self.obs_dim:
            acting = np.rint(enc[:, self.obs_dim]).astype(int)
        if acting is None:
            return
        mask_table = self._feature_mask_table(obs)
        prior = self.feature_action_prior
        governable = self.lag + 1
        read_column = []
        target_column = []
        for node_id in self.feature_ids:
            node = self.graph.nodes.get(node_id)
            if node is None or self._node_horizon(node_id) > governable:
                continue
            root_feature = self._feature_objects.get(node_id)
            if root_feature is None:
                continue
            root_mask = self._table_entry(
                mask_table, node_id, root_feature, obs
            )
            if root_mask is None or not root_mask.any():
                continue
            base = float(node.value)
            for action_key in self.action_space:
                cell = self.feature_action_stats.get(node_id, {}).get(
                    int(action_key)
                )
                if cell is not None and cell[0] > 0:
                    count, hits = int(cell[0]), float(cell[1])
                    read = (hits + prior * base) / (count + prior)
                else:
                    read = base
                selected = root_mask & (acting == int(action_key))
                if not selected.any():
                    continue
                read_column.append(np.full(int(selected.sum()), read))
                target_column.append(labels[selected])
        if not read_column:
            return
        reads = np.concatenate(read_column)
        targets = np.concatenate(target_column)
        if len(reads) < 2:
            return
        from node_runtime import _isotonic_map
        self.readout_calibration = _isotonic_map(reads, targets)

    def fit_hazard_composition(self, states: np.ndarray,
                               pending: np.ndarray,
                               labels: np.ndarray,
                               ridge: float = 1e-6) -> dict:
        """Fit the stacked non-negative composition of the hazard channels.

        Breiman stacked regressions with a non-negativity constraint: the
        hand-written max over correlated channel reads has
        ``E[max of correlated estimates]`` as its fixed point -
        structurally above max of the truths - and no consistency loss
        ever tied the composition to the data.  Here the channel columns
        are regressed on the precursor labels in closed form with
        ``w >= 0`` (exact zeros de-duplicate overlapping or inflating
        members), and a monotone map then holds the composed level to the
        empirical rate, the same estimator the graded readout uses.
        Unfitted, `pain_signal_batch` keeps the historical max-combine.

        Returns a small report (per-channel weights, mean absolute error
        of the composed read vs the labels, and the same error for the
        historical max-combine) so the fit's effect stays measurable.
        """
        states = np.asarray(states, dtype=float)
        pending_all = np.asarray(pending, dtype=float)
        if pending_all.shape[0] != len(states):
            raise ValueError("pending must have one row per state")
        labels = np.asarray(labels, dtype=float)
        enc = np.concatenate([states, pending_all], axis=1)
        channels = self._hazard_channels(enc)
        names = sorted(channels)
        if not names:
            return {}
        from node_runtime import _isotonic_map, _nnls
        design = np.column_stack([channels[name] for name in names])
        weights = _nnls(design, labels, ridge=ridge)
        composed = design @ weights
        self.hazard_composition = {
            name: float(weight) for name, weight in zip(names, weights)
        }
        self.hazard_calibration = _isotonic_map(composed, labels)
        mids, rates = self.hazard_calibration
        mapped = np.interp(composed, mids, rates)
        maximum = np.zeros(len(states), dtype=float)
        for name in names:
            maximum = np.maximum(maximum, channels[name])
        return {
            "weights": dict(self.hazard_composition),
            # The composed read as the consumer sees it (stacked, then
            # mapped) against the historical consumer (raw max-combine).
            "composed_mae": float(np.mean(np.abs(mapped - labels))),
            "maximum_mae": float(np.mean(np.abs(maximum - labels))),
            "composed_rate": float(np.mean(mapped)),
            "label_rate": float(np.mean(labels)),
        }

    def _calibrate_threat(self, threat: np.ndarray) -> np.ndarray:
        calibration = getattr(self, "readout_calibration", None)
        if calibration is None:
            return threat
        mids, rates = calibration
        return np.interp(threat, mids, rates)

    def pain_graded_batch(self, states: np.ndarray,
                          mask_table: Optional[List] = None):
        """Grade the risk of each state: real threat vs controllable excess.

        Over the short-window roots - the ones whose label window the
        pending actions actually govern - the per-action risk reads are
        reduced to two numbers per state: the *threat*, ``min over actions
        of P(pain | root, action)``, which no plan through the state can
        avoid and which is therefore charged in full; and the *excess*,
        ``P(pain | root, acting) - threat``, the avoidable share a plan
        takes by committing its own action - the part that counts as a
        controllable risk rather than a real threat.  Roots with a
        long-window horizon are excluded: their labels mix in continuations
        the pending actions do not govern.  States under no root read
        (0, 0).
        """
        states = np.asarray(states, dtype=float)
        n = len(states)
        if n == 0:
            return np.zeros(0), np.zeros(0)
        threat = np.full(n, -np.inf)
        excess = np.zeros(n, dtype=float)
        any_fired = np.zeros(n, dtype=bool)
        if not self.feature_ids:
            return np.zeros(n, dtype=float), np.zeros(n, dtype=float)
        obs = states[:, :self.obs_dim]
        if mask_table is None:
            mask_table = self._feature_mask_table(obs)
        acting = None
        if states.shape[1] > self.obs_dim:
            acting = np.rint(states[:, self.obs_dim]).astype(int)
        if acting is None:
            return np.zeros(n, dtype=float), np.zeros(n, dtype=float)
        prior = self.feature_action_prior
        # The governable span is the pending window: a label window of
        # `lag + 1` states is exactly the span whose governing actions are
        # the pending ones plus the action being decided.  The gate tests
        # the *window length*, not the presence of a horizon tag - a
        # tag-based test would both mute short-window roots that carry a
        # tag and admit untagged roots trained on a longer runtime window.
        governable = self.lag + 1
        for node_id in self.feature_ids:
            node = self.graph.nodes.get(node_id)
            if node is None or self._node_horizon(node_id) > governable:
                # Long-window roots: their labels see beyond the pending
                # window, so their per-action reads are not causal.
                continue
            root_feature = self._feature_objects.get(node_id)
            if root_feature is None:
                continue
            root_mask = self._table_entry(
                mask_table, node_id, root_feature, obs
            )
            if root_mask is None or not root_mask.any():
                continue
            base = float(node.value)
            reads = {}
            for action_key in self.action_space:
                cell = self.feature_action_stats.get(node_id, {}).get(
                    int(action_key)
                )
                if cell is not None and cell[0] > 0:
                    count, hits = int(cell[0]), float(cell[1])
                    reads[action_key] = (
                        hits + prior * base
                    ) / (count + prior)
                else:
                    reads[action_key] = base
            calibration = getattr(self, "readout_calibration", None)
            if calibration is not None:
                mids, rates = calibration
                for action_key in reads:
                    reads[action_key] = float(np.interp(
                        reads[action_key], mids, rates
                    ))
            threat_root = min(reads[action_key]
                              for action_key in self.action_space)
            for action_key in self.action_space:
                selected = root_mask & (acting == int(action_key))
                if not selected.any():
                    continue
                better = selected & (threat_root > threat)
                threat[better] = threat_root
                excess[better] = reads[action_key] - threat_root
                any_fired |= better
        return (
            np.where(any_fired, threat, 0.0),
            np.where(any_fired, excess, 0.0),
        )

    def _record_prediction(self, root_ids, acting: int, actual: float,
                           state) -> None:
        """Record the pre-update readout against the actual outcome.

        The prediction is what the search would see for this state under the
        acting action.  Confidence comes from the (feature, action) cell
        support; confident surprises enter the counterfactual target buffer.
        """
        actual = float(actual)
        for node_id in root_ids:
            prediction, count = self._feature_cell_read(node_id, acting)
            confidence = count / (count + self.feature_action_prior)
            entry = self.feature_calibration.setdefault(
                (node_id, int(acting)), [0, 0.0, 0.0, 0.0]
            )
            entry[0] += 1
            entry[1] += prediction
            entry[2] += actual
            entry[3] += (actual - prediction) ** 2
            score = confidence * abs(actual - prediction)
            if score <= 0.02:
                continue
            self.surprise_buffer.append(dict(
                state=np.asarray(state, dtype=float).copy(),
                node=node_id,
                acting=int(acting),
                prediction=float(prediction),
                actual=actual,
                confidence=float(confidence),
                score=float(score),
            ))
            if len(self.surprise_buffer) > self.surprise_capacity:
                self.surprise_buffer.sort(key=lambda item: -item["score"])
                del self.surprise_buffer[self.surprise_capacity:]

    def counterfactual_action(self, observation) -> Optional[int]:
        """The action to force where the decision matters and is unknown.

        Counterfactual guidance for collection, v2: a state qualifies
        only when the calibrated graded threat read puts the danger rate
        above `voi_threat_min` - a test spent on a state the readout
        considers safe buys no decision information, and the death
        attribution probe showed the planner's failures concentrate
        exactly at high-threat states whose rarely-taken action's cell
        is starved.  Among the feature roots active at such a state,
        prefer the one whose minority-action cell is emptiest while the
        learned prior says the action matters; the forced action is the
        minority one, so that cell gains a real outcome exactly where
        the threat's min-over-actions is otherwise neutralized by the
        anchor.  Returns None when the state is not decision-critical,
        every active root already has enough support for both actions,
        the prior is too low, or the per-cell test budget is spent.
        Purely learned signals - no task geometry.
        """
        if not self.feature_ids:
            return None
        obs = np.asarray(observation, dtype=float).reshape(1, -1)
        if obs.shape[1] < self.obs_dim:
            return None
        obs = obs[:, :self.obs_dim]
        # The graded threat read is a per-state scalar (the min over
        # actions composed over firing roots), so the pending window is
        # irrelevant here and zeros encode fine.
        pending = np.zeros(self.lag, dtype=float)
        encoded = np.concatenate(
            [obs[0] / self.scales[:self.obs_dim], pending]
        ).reshape(1, -1)
        threat, _excess = self.pain_graded_batch(encoded)
        if not np.isfinite(threat[0]):
            return None
        threat_value = float(threat[0])
        if threat_value < self.voi_threat_min:
            return None
        best = None
        for node_id in self.feature_ids:
            feature = self._feature_objects.get(node_id)
            node = self.graph.nodes.get(node_id)
            if feature is None or node is None:
                continue
            if not bool(feature.mask(obs)[0]):
                continue
            base = float(node.value)
            if base < self.voi_min_prior:
                continue
            cells = self.feature_action_stats.get(node_id, {})
            counts = [
                int(cells.get(action, [0, 0.0])[0])
                for action in self.action_space
            ]
            majority = max(counts)
            minority = int(np.argmin(counts))
            # Starvation is relative: an absolute floor keeps tiny-data
            # guides busy, and beyond it a cell counts as starved when it
            # holds less than a quarter of the other action's evidence -
            # the imbalance that neutralizes the min-over-actions at any
            # data volume.
            starved = (
                counts[minority] < self.voi_target
                or counts[minority] * 4 < majority
            )
            if not starved:
                continue
            if self.voi_tests.get((node_id, minority), 0) >= self.voi_cell_cap:
                continue
            deficit = self.voi_target - counts[minority]
            if deficit <= 0:
                deficit = 1
            score = threat_value * deficit
            if best is None or score > best[0]:
                best = (score, minority, node_id)
        if best is None:
            return None
        score, action, node_id = best
        self.voi_tests[(node_id, action)] = (
            self.voi_tests.get((node_id, action), 0) + 1
        )
        return action

    def _k_step_value_target(self, steps, position, snapshot):
        """k-step dopamine return with a bootstrap from the pre-update
        dopamine value of the state `value_horizon` steps ahead.

        The bootstrap reads the graph's dopamine value (the same channel
        the cells estimate) - snapshotting the pain channel here would
        inject a hazard level into a return target.  Truncation at the
        episode end simply sums the rewards that actually happened (the
        same convention as the MC path at a terminal step).
        """
        total = 0.0
        discount = 1.0
        for offset in range(int(self.value_horizon)):
            index = position + offset
            if index >= len(steps):
                return total
            total += discount * float(steps[index][4])
            discount *= self.gamma
        index = position + int(self.value_horizon) - 1
        if index < len(steps):
            total += discount * float(snapshot.get(steps[index][2], 0.0))
        return total

    def _update_feature_value(self, feature_ids, acting: int,
                              value: float,
                              episode_id=None, weight: float = 1.0) -> None:
        """Accumulate E[discounted dopamine return | feature, acting].

        Each cell also tracks how many distinct episodes (world instances)
        contributed to it - repetition inside one episode is not the same
        evidence as coverage across episodes, and the trust readout uses
        the distinct count.  `weight` scales the cell's effective sample
        size (the clipped importance weight when importance weighting is
        on); the diversity tally stays unweighted - it counts distinct
        episodes, not evidence mass.
        """
        value = float(value)
        weight = float(weight)
        for node_id in feature_ids:
            cells = self.feature_value_stats.setdefault(node_id, {})
            cell = cells.setdefault(int(acting), [0, 0.0, {}])
            cell[0] += weight
            cell[1] += weight * value
            if episode_id is not None:
                cell[2][episode_id] = cell[2].get(episode_id, 0) + 1
        # 1/n cumulative mean of the value targets: the anchor converges
        # to the stream's marginal mean return instead of tracking a
        # 50-event recency window whose wander is the same order as the
        # per-action differences it is meant to stabilize.
        self._value_baseline_count += weight
        self.value_baseline += (
            weight * (value - self.value_baseline)
            / self._value_baseline_count
        )

    def feature_value_batch(self, states: np.ndarray,
                            mask_table: Optional[List] = None,
                            return_support: bool = False):
        """Expected discounted dopamine return at each state, in return units.

        The read for a state is the support-shrunk ``(feature, pending[0])``
        value cell of its active feature roots.  With subgraph cells on,
        each active root reads its *most specific active member*: among the
        root and its active branches, the one whose cell for this acting
        action has the fewest observations wins, so an OR over two
        directions reads the branch actually taken instead of the root's
        pooled average while an AND keeps its narrower root.  Every read
        shrinks toward the running global baseline with
        `feature_value_prior` pseudo-counts, and members without data for
        this action fall back to that baseline, so the readout degrades to
        a flat prior exactly where nothing was learned.
        """
        states = np.asarray(states, dtype=float)
        baseline = float(self.value_baseline)
        result = np.full(len(states), baseline)
        roots = self._all_value_roots()
        if self.value_bank_only and self.value_feature_ids:
            # L1: the dedicated bank's predicates tile the decision-critical
            # regions; the risk bank's cells carry drifting-continuation
            # statistics that dilute the mean.
            roots = list(self.value_feature_ids)
        if not roots:
            if return_support:
                # No roots at all: report zero support so every consumer
                # treating the read as evidence-scaled sees "no data",
                # not a bare array that unpacks element-wise.
                return result, np.zeros(len(states), dtype=float)
            return result
        obs = states[:, :self.obs_dim]
        if mask_table is None:
            mask_table = self._feature_mask_table(obs)
        acting = None
        if states.shape[1] > self.obs_dim:
            acting = np.rint(states[:, self.obs_dim]).astype(int)
        prior = self.feature_value_prior
        total = np.zeros(len(states), dtype=float)
        weight = np.zeros(len(states), dtype=float)
        count_sum = np.zeros(len(states), dtype=float)
        diversity_sum = np.zeros(len(states), dtype=float)
        for node_id in roots:
            root_feature = self._feature_objects.get(node_id)
            if root_feature is None:
                continue
            root_mask = self._table_entry(
                mask_table, node_id, root_feature, obs
            )
            if root_mask is None or not root_mask.any():
                continue
            subtree = [node_id]
            if self.subgraph_cells:
                subtree.extend(self._feature_descendants.get(node_id, ()))
            values = np.full(len(states), baseline)
            best_count = np.full(len(states), np.inf)
            for member_id in subtree:
                feature = self._feature_objects.get(member_id)
                if feature is None:
                    continue
                mask = self._table_entry(
                    mask_table, member_id, feature, obs
                )
                if mask is None:
                    continue
                mask = mask if member_id == node_id else mask & root_mask
                if not mask.any() or acting is None:
                    continue
                for action_key, cell in (
                    self.feature_value_stats.get(member_id, {}).items()
                ):
                    count, total_return = cell[0], cell[1]
                    if count <= 0:
                        continue
                    diversity = (
                        len(cell[2]) if len(cell) > 2 and cell[2] else count
                    )
                    selected = (
                        mask & (acting == int(action_key))
                        & (count < best_count)
                    )
                    if selected.any():
                        best_count[selected] = count
                        count_sum[selected] += count
                        diversity_sum[selected] += diversity
                        values[selected] = (
                            total_return + prior * baseline
                        ) / (count + prior)
            total += np.where(root_mask, values, 0.0)
            weight += root_mask.astype(float)
        averaged = np.where(weight > 0.0, total / np.maximum(weight, 1e-9),
                            baseline)
        if not return_support:
            return averaged
        # Mean distinct-episode count behind each state's read: repetition
        # inside one world is not diverse evidence, so the trust signal is
        # the number of world instances that shaped the read (falling back
        # to the raw count for cells that never saw an episode id).
        support = np.where(
            weight > 0.0,
            diversity_sum / np.maximum(weight, 1e-9),
            0.0,
        )
        return averaged, support

    def evidence_batch(self, states: np.ndarray,
                       mask_table: Optional[List] = None) -> np.ndarray:
        """Mean observation count behind each state's active cells.

        A task-agnostic evidence read: for every active feature root it
        looks up that root's ``(feature, acting action)`` cell in the risk
        and value counters and keeps the larger count (either channel
        supports the state's read), then averages over the roots active in
        the state.  States with no active root report zero.  Used only by
        the decision-side curiosity potential - less evidence means more
        novelty - never by the hazard or value readouts.
        """
        states = np.asarray(states, dtype=float)
        result = np.zeros(len(states), dtype=float)
        roots = self._all_value_roots()
        if not roots or states.shape[1] <= self.obs_dim:
            return result
        obs = states[:, :self.obs_dim]
        if mask_table is None:
            mask_table = self._feature_mask_table(obs)
        acting = np.rint(states[:, self.obs_dim]).astype(int)
        total = np.zeros(len(states), dtype=float)
        weight = np.zeros(len(states), dtype=float)
        for node_id in roots:
            feature = self._feature_objects.get(node_id)
            if feature is None:
                continue
            root_mask = self._table_entry(
                mask_table, node_id, feature, obs
            )
            if root_mask is None or not root_mask.any():
                continue
            counts = np.zeros(len(states), dtype=float)
            for action_key, cell in (
                self.feature_action_stats.get(node_id, {}).items()
            ):
                if cell[0] <= 0:
                    continue
                selected = root_mask & (acting == int(action_key))
                counts[selected] = np.maximum(
                    counts[selected], float(cell[0])
                )
            for action_key, cell in (
                self.feature_value_stats.get(node_id, {}).items()
            ):
                if cell[0] <= 0:
                    continue
                selected = root_mask & (acting == int(action_key))
                counts[selected] = np.maximum(
                    counts[selected], float(cell[0])
                )
            total += np.where(root_mask, counts, 0.0)
            weight += root_mask.astype(float)
        return np.where(
            weight > 0.0, total / np.maximum(weight, 1e-9), 0.0
        )

    def prediction_report(self) -> dict:
        """Calibration of the readout: prediction vs actual per cell."""
        cells = []
        for (node_id, acting), (n, pred_sum, actual_sum, sq) in (
            self.feature_calibration.items()
        ):
            if n <= 0:
                continue
            feature = self._feature_objects.get(node_id)
            pred_mean = pred_sum / n
            actual_mean = actual_sum / n
            cells.append(dict(
                node=node_id,
                feature="" if feature is None else feature.label(),
                action=int(acting),
                n=n,
                pred=pred_mean,
                actual=actual_mean,
                brier=sq / n,
                gap=pred_mean - actual_mean,
                miscalibrated=(abs(pred_mean - actual_mean) > 0.15
                               and n >= 10),
            ))
        cells.sort(
            key=lambda item: -abs(item["gap"]) * min(item["n"], 50)
        )
        return dict(
            cells=cells,
            miscalibrated=sum(1 for item in cells if item["miscalibrated"]),
            buffer_size=len(self.surprise_buffer),
            buffer=[
                dict(score=item["score"], acting=item["acting"],
                     prediction=item["prediction"], actual=item["actual"],
                     confidence=item["confidence"])
                for item in sorted(
                    self.surprise_buffer, key=lambda item: -item["score"]
                )[:8]
            ],
        )

    def _activate_features(self, feature_ids) -> None:
        active = set(feature_ids)
        for node_id in self._feature_all:
            node = self.graph.nodes.get(node_id)
            if node is not None:
                node.activate(1.0 if node_id in active else 0.0)

    def feature_risk_batch(self, states: np.ndarray) -> np.ndarray:
        """Public read of the action-conditioned feature risk cells."""
        return self._feature_risk_batch(states)

    def _feature_risk_batch(self, states: np.ndarray,
                            mask_table: Optional[List] = None) -> np.ndarray:
        """Action-conditioned feature risk.

        The readout is ``P(pain within lag+1 steps | feature, acting
        action)``.  The acting action is the first pending coordinate of the
        encoded state, so two plans whose pending actions differ receive
        different risk at the same predicted state.  With subgraph cells on,
        a root's read comes from its *most specific active member*: among
        the root and its active branches, the one whose cell for this
        acting action has the fewest observations wins - an OR over two
        directions reads the branch actually taken instead of the root's
        pooled average, while an AND keeps its narrower root.  Members
        without data for this action fall back to the root's scalar
        precursor value exactly as before; with the flag off the read is
        the root's own statistics only.
        """
        states = np.asarray(states, dtype=float)
        result = np.zeros(len(states), dtype=float)
        if not self.feature_ids:
            return result
        obs = states[:, :self.obs_dim]
        if mask_table is None:
            mask_table = self._feature_mask_table(obs)
        acting = None
        if states.shape[1] > self.obs_dim:
            acting = np.rint(states[:, self.obs_dim]).astype(int)
        prior = self.feature_action_prior
        for node_id in self.feature_ids:
            root_feature = self._feature_objects.get(node_id)
            if root_feature is None:
                continue
            root_mask = self._table_entry(
                mask_table, node_id, root_feature, obs
            )
            if root_mask is None or not root_mask.any():
                continue
            subtree = [node_id]
            if self.subgraph_cells:
                subtree.extend(self._feature_descendants.get(node_id, ()))
            base = float(self.graph.nodes[node_id].value)
            # Fallback first: the root's scalar, as in the root-only read.
            values = np.full(len(states), base)
            best_count = np.full(len(states), np.inf)
            for member_id in subtree:
                feature = self._feature_objects.get(member_id)
                node = self.graph.nodes.get(member_id)
                if feature is None or node is None:
                    continue
                mask = self._table_entry(
                    mask_table, member_id, feature, obs
                )
                if mask is None:
                    continue
                mask = mask if member_id == node_id else mask & root_mask
                if not mask.any():
                    continue
                member_base = float(node.value)
                if acting is None:
                    continue
                for action_key, (count, hits) in (
                    self.feature_action_stats.get(member_id, {}).items()
                ):
                    if count <= 0:
                        continue
                    selected = (
                        mask & (acting == int(action_key))
                        & (count < best_count)
                    )
                    if selected.any():
                        best_count[selected] = count
                        values[selected] = (
                            hits + prior * member_base
                        ) / (count + prior)
            result = np.maximum(result, np.where(root_mask, values, 0.0))
        return result

    def _feature_memory_risk_batch(self, states: np.ndarray,
                                   mask_table: Optional[List] = None
                                   ) -> np.ndarray:
        """Risk from the event memories attached to active feature roots.

        Same result as the historical per-row activation sweep - every
        node's mask is now computed once for the whole batch and each row
        looks the activations up in that table, in the same
        materialization order and with the same memory merge.  The sweep
        used to dominate evaluation time: it recomputed every mask for
        every batch row, and the cost grew with both the batch size and
        the number of materialized sub-features.
        """
        states = np.asarray(states, dtype=float)
        result = np.zeros(len(states), dtype=float)
        if not self.feature_ids:
            return result
        obs = states[:, :self.obs_dim]
        if mask_table is None:
            table = self._feature_mask_table(obs)
        else:
            table = mask_table
        # Per-root memory lists, computed once per batch.  The historical
        # path re-ran the membership/association graph scans for every
        # row; the merge below reproduces its exact order (per root: the
        # node's own membership memories, then its one-hop association
        # reach, merged by max), so the reads are bit-identical.
        root_set = set(self.feature_ids)
        root_columns = []
        for node_id, entry in zip(self._feature_all, table):
            if node_id not in root_set or entry is None:
                continue
            if entry.any():
                root_columns.append((node_id, entry))
        root_memories = {
            node_id: self.memories_for([node_id])
            for node_id, _entry in root_columns
        }
        # Live rates are constant over one readout; one read per memory
        # instead of one per (row, root) hit.
        rate_cache: Dict[str, float] = {}
        for row in range(len(states)):
            feature_ids = [
                node_id
                for node_id, mask in root_columns
                if bool(mask[row])
            ]
            if not feature_ids:
                continue
            merged: Dict[str, float] = {}
            for node_id in feature_ids:
                for memory_id, strength in (
                    root_memories[node_id].items()
                ):
                    merged[memory_id] = max(
                        merged.get(memory_id, 0.0), strength
                    )
            total = 0.0
            weighted = 0.0
            for memory_id, strength in merged.items():
                memory = self.graph.nodes.get(memory_id)
                if memory is None or memory.kind != MEMORY:
                    continue
                total += float(strength)
                # The channel reads the memory's live valence rate - the
                # same association statistics every other node reports -
                # not the creation-time snapshot, which would report a
                # single event's pain forever no matter what later
                # transitions through the same region showed.
                rate = rate_cache.get(memory_id)
                if rate is None:
                    rate = self._pain_association(memory_id)
                    rate_cache[memory_id] = rate
                weighted += float(strength) * rate
            if total > 0.0:
                result[row] = weighted / total
        return result

    # ---- analysis ----------------------------------------------------------
    def prototype_pain_baseline(self) -> float:
        """Population mean of the association-based pain readout."""
        if not self._prototype_ids:
            return 0.0
        values = [
            self._pain_association(node_id)
            for node_id in self._prototype_ids
        ]
        return float(np.mean(values)) if values else 0.0

    def prototype_valence_baselines(self):
        """Population mean of (value + immediate) pain and dopamine."""
        if not self._prototype_ids:
            return 0.0, 0.0
        pain = [
            self.graph.nodes[node_id].value
            + self.graph.nodes[node_id].immediate_pain
            for node_id in self._prototype_ids
        ]
        dopamine = [
            self.graph.nodes[node_id].dopamine
            + self.graph.nodes[node_id].immediate_dopamine
            for node_id in self._prototype_ids
        ]
        return float(np.mean(pain)), float(np.mean(dopamine))

    def _shrink_to_baseline(self, values, nearest_min, baseline):
        """Trust-shrink a prototype readout as the state leaves support.

        ``ood_trust_scale`` is an absolute distance in standardized units;
        the natural setting is the birth threshold, where trust is
        exp(-0.5) ~ 0.61.  With the default 0.0 the readout is unchanged.
        """
        if self.ood_trust_scale <= 0.0:
            return values
        # `nearest_min` is the squared distance, matching the soft-weight
        # convention, so trust = exp(-0.5 * d^2 / scale^2).
        trust = np.exp(
            -0.5
            * np.asarray(nearest_min, dtype=float)
            / (self.ood_trust_scale ** 2)
        )
        return trust * np.asarray(
            values, dtype=float
        ) + (1.0 - trust) * float(baseline)

    # ---- readout cache -----------------------------------------------------
    def _invalidate_readout_cache(self) -> None:
        """Drop the derived readout structures after any learning mutation.

        The prototype adjacency (an O(E) edge pass) and the per-prototype
        columns (an O(P) memory retrieval) depend on the graph state, not on
        the query, yet the planner reads them many times per decision.  A
        cache makes an evaluation phase pay for them once; every learning
        entry point drops it so training readouts never see stale numbers.
        """
        self._readout_cache = None

    def _readout_signature(self):
        """Cheap structural fingerprint of the state the cache depends on.

        Value/weight updates during training are caught by the explicit
        invalidation in `_interact`; this guard additionally catches any
        structural change (births, rule materialisation, pruning) that
        happens outside that funnel, without needing a hook per call site.
        """
        return (
            len(self._prototype_ids),
            len(self.graph.nodes),
            len(self.graph.edges),
            len(self._feature_objects),
        )

    def _store_readout_cache(self, key, value) -> None:
        if self._readout_cache is None:
            self._readout_cache = {}
        self._readout_cache["signature"] = self._readout_signature()
        self._readout_cache[key] = value

    def _cached_readout(self, key):
        cache = self._readout_cache
        if (cache is not None and key in cache
                and cache.get("signature") == self._readout_signature()):
            return cache[key]
        return None

    def _prototype_associations(self):
        """Cached per-prototype association rate.

        ``_pain_association`` is deterministic for a fixed graph, yet the
        readout asks for all ``P`` prototypes on every query within a phase
        (the association rate does not depend on the query).  Caching it turns
        that into one sweep per phase.
        """
        cached = self._cached_readout("associations")
        if cached is not None:
            return cached
        values = np.asarray(
            [self._pain_association(node_id) for node_id in self._prototype_ids],
            dtype=float,
        )
        self._store_readout_cache("associations", values)
        return values

    def _prototype_memory_index(self):
        """Cached CSR of prototype -> (memory, membership weight).

        Returns ``(out_start, out_count, mem_slot, mem_weight, rates)``:
        ``out_start[p]``/``out_count[p]`` delimit prototype ``p``'s reachable
        memories inside the flat ``mem_slot``/``mem_weight`` arrays, and
        ``rates`` holds each listed memory's own association rate.

        A single incidence pass resolves every node's membership weights once
        and reuses them for each prototype that associates with it, instead of
        rebuilding a dictionary per prototype (or per row) as ``memories_for``
        does.  A ``(prototype, memory)`` pair keeps the maximum weight over
        the paths reaching it, which is exactly the value ``memories_for``
        returns.  Cached until the runtime learns.
        """
        cached = self._cached_readout("memory_index")
        if cached is not None:
            return cached
        prototypes = self._prototype_ids
        count = len(prototypes)
        membership: Dict[str, Dict[str, float]] = {}

        def weights_of(node_id: str) -> Dict[str, float]:
            weights = membership.get(node_id)
            if weights is None:
                weights = {}
                for key in self.graph.incoming.get(node_id, ()):
                    if key[2] != MEMBERSHIP:
                        continue
                    value = float(self.graph.edges[key].weight)
                    if value > weights.get(key[0], 0.0):
                        weights[key[0]] = value
                membership[node_id] = weights
            return weights

        src: List[int] = []
        slot: List[int] = []
        weight: List[float] = []
        rates: List[float] = []
        slots: Dict[str, int] = {}
        for row, node_id in enumerate(prototypes):
            reachable: Dict[str, float] = {}
            for memory_id, value in weights_of(node_id).items():
                reachable[memory_id] = value
            # Associations are one-hop retrieval paths to memories of related
            # nodes; a single edge is registered under both endpoints, so it
            # is followed from whichever end is a prototype.
            for key in self.graph.outgoing.get(node_id, ()):
                if key[2] != ASSOCIATION:
                    continue
                edge = self.graph.edges[key]
                other = (
                    edge.target if edge.source == node_id else edge.source
                )
                for memory_id, value in weights_of(other).items():
                    if value > reachable.get(memory_id, 0.0):
                        reachable[memory_id] = value
            for memory_id, value in reachable.items():
                if value <= 0.0:
                    continue
                index = slots.get(memory_id)
                if index is None:
                    index = len(rates)
                    slots[memory_id] = index
                    rates.append(self._pain_association(memory_id))
                src.append(row)
                slot.append(index)
                weight.append(value)
        src_arr = np.asarray(src, dtype=np.int64)
        out_count = np.bincount(src_arr, minlength=count).astype(np.int64)
        out_start = np.concatenate([
            np.zeros(1, dtype=np.int64), np.cumsum(out_count)[:-1]
        ])
        result = (
            out_start,
            out_count,
            np.asarray(slot, dtype=np.int64),
            np.asarray(weight, dtype=float),
            np.asarray(rates, dtype=float),
        )
        self._store_readout_cache("memory_index", result)
        return result

    def _prototype_columns(self):
        """Cached per-prototype columns: (pain, dopamine, hazard, assoc).

        ``hazard`` folds each prototype's own association rate and the rates
        of the memories it binds into one number, so the relaxed readout is a
        single matmul instead of a per-row dictionary walk.  The memory
        weights come from ``_prototype_memory_index``'s shared incidence pass
        rather than one ``memories_for`` call per prototype.
        """
        cached = self._cached_readout("columns")
        if cached is not None:
            return cached
        prototypes = self._prototype_ids
        count = len(prototypes)
        nodes = [self.graph.nodes[node_id] for node_id in prototypes]
        pain = np.asarray(
            [node.value + node.immediate_pain for node in nodes], dtype=float
        )
        dopamine = np.asarray(
            [node.dopamine + node.immediate_dopamine for node in nodes],
            dtype=float,
        )
        associations = self._prototype_associations()
        hazard = associations.copy()
        mass = np.ones(count, dtype=float)
        if count:
            out_start, out_count, mem_slot, mem_weight, rates = (
                self._prototype_memory_index()
            )
            if mem_slot.size:
                rows = np.repeat(
                    np.arange(count, dtype=np.int64), out_count
                )
                mass += np.bincount(
                    rows, weights=mem_weight, minlength=count
                )
                hazard += np.bincount(
                    rows, weights=mem_weight * rates[mem_slot],
                    minlength=count,
                )
        hazard = hazard / np.where(mass > 0.0, mass, 1.0)
        columns = (pain, dopamine, hazard, associations)
        self._store_readout_cache("columns", columns)
        return columns

    def _prototype_transition(self):
        """Sparse row-normalised transition operator over prototypes.

        Returns ``(out_start, out_count, edge_dst, edge_weight)``: a CSR view
        grouped by *source* prototype, so ``out_start[c]``/``out_count[c]``
        delimit the outgoing edges of prototype ``c`` inside the flat
        ``edge_dst``/``edge_weight`` arrays.  Grouping by source is what makes
        a relaxation step cheap: the seed's support is only ``soft_neighbors``
        wide per row, so the step expands just those entries along their
        outgoing edges instead of touching a dense ``batch x P`` block.  The
        graph is sparse while ``P`` reaches a few thousand, so a dense
        operator would make every step O(batch * P) rather than
        O(batch * soft_neighbors * outdegree).

        Association edges are added in both directions.  A prototype with no
        outgoing edge keeps an identity self-loop, so every source has at
        least one outgoing edge and mass is conserved.  Cached until the
        runtime learns.
        """
        cached = self._cached_readout("transition")
        if cached is not None:
            return cached
        count = len(self._prototype_ids)
        empty_int = np.zeros(0, dtype=np.int64)
        if count == 0:
            result = (empty_int, empty_int, empty_int, np.zeros(0))
            self._store_readout_cache("transition", result)
            return result
        index = {
            node_id: row for row, node_id in enumerate(self._prototype_ids)
        }
        src: List[int] = []
        dst: List[int] = []
        weight: List[float] = []
        want = self.readout_propagation_edges
        for edge in self.graph.edges.values():
            if edge.kind == TRANSITION_EDGE:
                if want == "association":
                    continue
            elif edge.kind == ASSOCIATION:
                if want == "transition":
                    continue
            else:
                continue
            source = index.get(edge.source)
            target = index.get(edge.target)
            if source is None or target is None:
                continue
            value = float(edge.weight)
            if value <= 0.0:
                continue
            src.append(source)
            dst.append(target)
            weight.append(value)
            if edge.kind == ASSOCIATION:
                src.append(target)
                dst.append(source)
                weight.append(value)
        src_arr = np.asarray(src, dtype=np.int64)
        dst_arr = np.asarray(dst, dtype=np.int64)
        weight_arr = np.asarray(weight, dtype=float)
        present = np.zeros(count, dtype=bool)
        present[src_arr] = True
        missing = np.flatnonzero(~present)
        if missing.size:
            src_arr = np.concatenate([src_arr, missing])
            dst_arr = np.concatenate([dst_arr, missing])
            weight_arr = np.concatenate([weight_arr, np.ones(missing.size)])
        row_sum = np.bincount(src_arr, weights=weight_arr, minlength=count)
        weight_arr = weight_arr / row_sum[src_arr]
        order = np.argsort(src_arr, kind="stable")
        src_arr = src_arr[order]
        dst_arr = dst_arr[order]
        weight_arr = weight_arr[order]
        out_count = np.bincount(src_arr, minlength=count).astype(np.int64)
        out_start = np.concatenate([
            np.zeros(1, dtype=np.int64), np.cumsum(out_count)[:-1]
        ])
        result = (out_start, out_count, dst_arr, weight_arr)
        self._store_readout_cache("transition", result)
        return result

    def _prototype_hazard_from_weights(self, indices, weights) -> np.ndarray:
        """Vectorised per-row prototype hazard for the geometric readout.

        The historical implementation walked each row's ``soft_neighbors`` in
        Python and rebuilt a memory dictionary per row, which is the readout's
        dominant cost (hundreds of thousands of ``memories_for`` calls per
        evaluation).  This computes the same quantity - the association rate
        of the row's neighbouring prototypes, folded with the rates of the
        memories those prototypes bind - with one ragged expand plus a
        ``maximum`` reduce over the ``(row, memory)`` pairs, so the union and
        the max-weighted strength match ``memories_for`` exactly, while the
        per-row Python loop disappears.
        """
        indices = np.asarray(indices, dtype=np.int64)
        weights = np.asarray(weights, dtype=float)
        batch = len(indices)
        associations = self._prototype_associations()
        total = weights.sum(axis=1)
        weighted = (weights * associations[indices]).sum(axis=1)
        _start, out_count, mem_slot, mem_weight, rates = (
            self._prototype_memory_index()
        )
        if mem_slot.size:
            n_mem = int(rates.size)
            flat = indices.reshape(-1)
            degree = out_count[flat]
            edge = self._ragged_edge_indices(degree, _start, flat)
            if edge.size:
                pair_rows = np.repeat(
                    np.arange(batch, dtype=np.int64), indices.shape[1]
                )
                keys = np.repeat(pair_rows, degree) * n_mem + mem_slot[edge]
                order = np.argsort(keys, kind="stable")
                keys = keys[order]
                strength = mem_weight[edge][order]
                bounds = np.flatnonzero(np.concatenate(
                    ([True], keys[1:] != keys[:-1])
                ))
                unique = keys[bounds]
                best = np.maximum.reduceat(strength, bounds)
                unique_rows = unique // n_mem
                total = total + np.bincount(
                    unique_rows, weights=best, minlength=batch
                )
                weighted = weighted + np.bincount(
                    unique_rows, weights=best * rates[unique % n_mem],
                    minlength=batch,
                )
        return np.where(
            total > 0.0,
            weighted / np.where(total > 0.0, total, 1.0),
            0.0,
        )

    @staticmethod
    def _ragged_edge_indices(degree, start, index):
        """Positions in a CSR edge list after expanding ``index`` by ``degree``.

        For a CSR whose row ``i`` occupies ``start[i] .. start[i]+count[i]``,
        this maps each entry of ``index`` to the run of ``degree`` consecutive
        edge positions it owns, laid out back to back - the flat gather a
        ragged expansion needs.
        """
        total = int(degree.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        group = np.cumsum(degree) - degree
        return (
            np.repeat(start[index], degree)
            + np.arange(total, dtype=np.int64)
            - np.repeat(group, degree)
        )

    def _relax_step(self, rows, cols, vals, gain, count, operator):
        """One ``q <- (1-g)*q + g*(q @ A)`` step on sparse triplets.

        ``(rows, cols, vals)`` are the nonzero entries of the current mass.
        Each entry is repeated for every outgoing edge of its prototype (a
        ragged gather), the flow is scaled by ``g``, the retained mass by
        ``1-g``, and the two lists are coalesced in a single ``np.unique``
        pass so duplicate ``(row, prototype)`` pairs sum instead of
        overwriting each other.
        """
        out_start, out_count, edge_dst, edge_w = operator
        degree = out_count[cols]
        edge = self._ragged_edge_indices(degree, out_start, cols)
        if edge.size == 0:
            return rows, cols, vals
        keys = np.concatenate([
            rows * count + cols,
            np.repeat(rows, degree) * count + edge_dst[edge],
        ])
        merged = np.concatenate([
            (1.0 - gain) * vals,
            gain * np.repeat(vals, degree) * edge_w[edge],
        ])
        unique, inverse = np.unique(keys, return_inverse=True)
        coalesced = np.bincount(
            inverse.reshape(-1), weights=merged, minlength=unique.size
        )
        keep = coalesced != 0.0
        unique = unique[keep]
        coalesced = coalesced[keep]
        return unique // count, unique % count, coalesced

    def _propagated_readout(self, indices, weights, columns):
        """Contract the edge-relaxed seed weights with per-prototype columns.

        ``indices``/``weights`` are the geometric seed
        (``_soft_weights_batch``) and ``columns`` is ``(P,)`` or ``(P, m)``;
        the return is ``(batch,)`` or ``(batch, m)``.  Mass is relaxed
        ``steps`` times by ``q <- (1-g)*q + g*(q @ A)`` and then contracted,
        so the readout reflects the topological neighbourhood of each row
        instead of just its geometric neighbours.

        The relaxation never materialises a dense ``batch x P`` block: the
        seed has ``soft_neighbors`` entries per row and each step expands
        them along the outgoing edges, so the work follows the support.  The
        contraction is a ``np.bincount`` per output column, i.e. C-level
        NumPy with no per-row Python loop.
        """
        columns = np.asarray(columns, dtype=float)
        flat = columns.ndim == 1
        if flat:
            columns = columns[:, None]
        width = columns.shape[1]
        count = columns.shape[0]
        batch = len(indices)
        if batch == 0:
            out = np.zeros((0, width), dtype=float)
            return out[:, 0] if flat else out
        idx = np.asarray(indices, dtype=np.int64)
        rows = np.repeat(
            np.arange(batch, dtype=np.int64), idx.shape[1]
        )
        cols = idx.reshape(-1)
        vals = np.asarray(weights, dtype=float).reshape(-1)
        gain = self.readout_propagation_gain
        if gain > 0.0 and cols.size:
            operator = self._prototype_transition()
            if operator[2].size:
                for _ in range(self.readout_propagation_steps):
                    rows, cols, vals = self._relax_step(
                        rows, cols, vals, gain, count, operator
                    )
        if rows.size:
            total = np.bincount(rows, weights=vals, minlength=batch)
            vals = vals / np.where(total[rows] > 0.0, total[rows], 1.0)
        contributions = vals[:, None] * columns[cols]
        out = np.zeros((batch, width), dtype=float)
        for column in range(width):
            out[:, column] = np.bincount(
                rows, weights=contributions[:, column], minlength=batch
            )
        return out[:, 0] if flat else out

    def _prototype_geometry(self):
        """Cached prototype squared norms and -2 * prototypes.T for fast BLAS."""
        cached = self._cached_readout("proto_geom")
        if cached is not None:
            return cached
        prototypes = self._prototype_matrix
        proto_norm = np.einsum("ij,ij->i", prototypes, prototypes)[None, :]
        proto_t_neg2 = np.ascontiguousarray(-2.0 * prototypes.T)
        geom = (proto_norm, proto_t_neg2)
        self._store_readout_cache("proto_geom", geom)
        return geom

    def _soft_weights_batch(self, states: np.ndarray):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.scales):
            raise ValueError("states must be a matrix with the runtime width")
        if not self._prototype_ids:
            return None, None, None
        # Squared distances via cached -2*P^T matmul and in-place norm adds.
        proto_norm, proto_t_neg2 = self._prototype_geometry()
        state_norm = np.einsum("ij,ij->i", states, states)[:, None]
        distances = states @ proto_t_neg2
        distances += state_norm
        distances += proto_norm
        np.maximum(distances, 0.0, out=distances)
        count = min(self.soft_neighbors, distances.shape[1])
        indices = np.argpartition(distances, count - 1, axis=1)[:, :count]
        nearest = np.take_along_axis(distances, indices, axis=1)
        nearest_min = nearest.min(axis=1)
        inv_2sigma2 = -0.5 / (self.soft_sigma * self.soft_sigma)
        weights = np.exp(inv_2sigma2 * nearest)
        total = weights.sum(axis=1, keepdims=True)
        weights /= np.where(total > 0.0, total, 1.0)
        return indices, weights, nearest_min

    def _prototype_memory_attribute_vectors(self, attribute: str):
        """Cached per-prototype weighted memory sums and masses for ``attribute``."""
        cache_key = ("memory_attr", attribute)
        cached = self._cached_readout(cache_key)
        if cached is not None:
            return cached
        n_proto = len(self._prototype_ids)
        attr_sum = np.zeros(n_proto, dtype=float)
        mass_sum = np.zeros(n_proto, dtype=float)
        for p_idx, node_id in enumerate(self._prototype_ids):
            memories = self.memories_for([node_id])
            for memory_id, strength in memories.items():
                memory = self.graph.nodes.get(memory_id)
                if memory is None or memory.kind != MEMORY:
                    continue
                s = float(strength)
                if s <= 0.0:
                    continue
                attr_sum[p_idx] += s * getattr(memory, attribute)
                mass_sum[p_idx] += s
        pair = (attr_sum, mass_sum)
        self._store_readout_cache(cache_key, pair)
        return pair

    def _memory_values_batch(self, indices, weights, attribute: str,
                             cache: Optional[Dict[str, Dict[str, float]]] = None):
        if len(indices) == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        attr_sum, mass_sum = self._prototype_memory_attribute_vectors(attribute)
        sums = (weights * attr_sum[indices]).sum(axis=1)
        mass = (weights * mass_sum[indices]).sum(axis=1)
        return sums, mass

    def valence_state_batch(self, states: np.ndarray, soft_weights=None):
        """Soft-neighbour pain/dopamine values for encoded states."""
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0), np.zeros(0)
        if soft_weights is None:
            indices, weights, nearest_min = self._soft_weights_batch(states)
        else:
            indices, weights, nearest_min = soft_weights
        if indices is None:
            length = len(np.asarray(states))
            return np.zeros(length), np.zeros(length)
        pain_column, dopamine_column, _hazard, _assoc = (
            self._prototype_columns()
        )
        if self.readout_propagation_gain > 0.0:
            channels = self._propagated_readout(
                indices, weights,
                np.stack([pain_column, dopamine_column], axis=1),
            )
            pain = channels[:, 0]
            dopamine = channels[:, 1]
        else:
            pain = (weights * pain_column[indices]).sum(axis=1)
            dopamine = (weights * dopamine_column[indices]).sum(axis=1)
        # Both valence channels read the same memory neighbourhoods; one
        # retrieval pass per prototype serves both instead of rescanning
        # the membership/association graph for each attribute.
        memory_cache: Dict[str, Dict[str, float]] = {}
        for attribute, values in (
            ("value", pain), ("dopamine", dopamine)
        ):
            memory_sums, memory_mass = self._memory_values_batch(
                indices, weights, attribute, cache=memory_cache
            )
            present = memory_mass > 0.0
            if present.any():
                values[present] = (
                    (1.0 - self.memory_weight) * values[present]
                    + self.memory_weight
                    * memory_sums[present] / memory_mass[present]
                )
        if self.ood_trust_scale > 0.0:
            baseline_pain, baseline_dopamine = (
                self.prototype_valence_baselines()
            )
            pain = self._shrink_to_baseline(
                pain, nearest_min, baseline=baseline_pain
            )
            dopamine = self._shrink_to_baseline(
                dopamine, nearest_min, baseline=baseline_dopamine
            )
        # Terminal pain shares the observation channel's trust; dopamine is
        # only a tie-break in the search objective and is left at full trust.
        pain = pain * self.obs_trust
        return pain, dopamine

    def _memory_value_from_weights(self, weights, attribute: str = "value"):
        total = 0.0
        weighted = 0.0
        for node_id, weight in weights:
            for memory_id, strength in self.memories_for([node_id]).items():
                mass = float(weight) * float(strength)
                if mass <= 0.0:
                    continue
                memory = self.graph.nodes.get(memory_id)
                if memory is None or memory.kind != MEMORY:
                    continue
                total += mass
                weighted += mass * getattr(memory, attribute)
        if total <= 0.0:
            return None
        return weighted / total

    def immediate_risk_batch(self, states: np.ndarray) -> np.ndarray:
        """Weighted immediate pain of soft-neighbour states."""
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0)
        indices, weights, nearest_min = self._soft_weights_batch(states)
        if indices is None:
            return np.zeros(len(states))
        values = np.asarray([
            self.graph.nodes[self._prototype_ids[int(index)]].immediate_pain
            for index in indices.ravel()
        ]).reshape(indices.shape)
        result = (weights * values).sum(axis=1)
        if self.ood_trust_scale > 0.0:
            baseline = float(np.mean([
                self.graph.nodes[node_id].immediate_pain
                for node_id in self._prototype_ids
            ])) if self._prototype_ids else 0.0
            result = self._shrink_to_baseline(
                result, nearest_min, baseline=baseline
            )
        return result

    def value_state(self, state: np.ndarray) -> float:
        weights = self._neighbor_weights(np.asarray(state, dtype=float))
        if not weights:
            return 0.0
        value = float(sum(
            weight * (
                self.graph.nodes[node_id].value
                + self.graph.nodes[node_id].immediate_pain
            )
            for node_id, weight in weights
        ))
        memory_value = self._memory_value_from_weights(weights, "value")
        if memory_value is not None:
            value = (
                (1.0 - self.memory_weight) * value
                + self.memory_weight * memory_value
            )
        return value

    def dopamine_state(self, state: np.ndarray) -> float:
        weights = self._neighbor_weights(np.asarray(state, dtype=float))
        return float(sum(
            weight * (
                self.graph.nodes[node_id].dopamine
                + self.graph.nodes[node_id].immediate_dopamine
            )
            for node_id, weight in weights
        )) if weights else 0.0

    def node_value(self, observation) -> float:
        state = self.encode(observation)
        node_id, _distance = self._nearest(state)
        if node_id is None:
            return 0.0
        node = self.graph.nodes[node_id]
        return float(node.value + node.immediate_pain)

    def node_dopamine(self, observation) -> float:
        state = self.encode(observation)
        node_id, _distance = self._nearest(state)
        if node_id is None:
            return 0.0
        node = self.graph.nodes[node_id]
        return float(node.dopamine + node.immediate_dopamine)

