"""Runtime that learns from nodes instead of a hand-labeled risk target.

The only human-supplied inputs are:

    observation, action, valence (dopamine/pain), episode boundary

There is no `terminal` risk label, no gap/collision predicate, and no task
geometry.  The runtime decides which situations precede pain by assigning
observations to concept nodes and learning action-conditioned transition values
with temporal credit assignment.

The graph is used at runtime, not as an offline lookup table:

* observations are assigned to nodes (with online births for novel states);
* every transition creates an action edge ``(node, action) -> node``;
* node values learn discounted future pain/dopamine with TD(lambda) traces;
* action edges learn the one-step expected valence of taking an action;
* decisions compare the learned action-edge values plus an uncertainty bonus.

The implementation deliberately keeps a small, explicit interface.  It is a
prototype, not a replacement for the existing search baselines.
"""
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


def future_pain_labels(rows, window: int) -> List[float]:
    """Precursor pain labels: ``P_k(t) = max pain over the next k states``.

    `rows` are transition dicts ordered in time (``signals[1]`` is the pain of
    the *entered* state, ``done`` marks an episode boundary).  With
    ``window = k`` the label of row ``t`` is the largest pain observed at
    states ``t+1 .. t+k``, truncated at the first terminal step or episode
    end.  With ``window = 1`` this is the plain entered-state pain.
    """
    rows = list(rows)
    pains = []
    dones = []
    episodes = []
    for row in rows:
        signals = np.asarray(row["signals"], dtype=float).reshape(-1)
        pains.append(float(signals[1]))
        dones.append(bool(row["done"]))
        episodes.append(row.get("episode"))
    span = max(int(window), 1)
    labels = []
    for index in range(len(rows)):
        value = 0.0
        for offset in range(span):
            position = index + offset
            if position >= len(rows) or episodes[position] != episodes[index]:
                break
            value = max(value, pains[position])
            if dones[position]:
                break
        labels.append(value)
    return labels


def future_return_labels(rows, gamma: float) -> List[float]:
    """Monte-Carlo discounted dopamine return following each transition.

    The label of row ``t`` is ``sum gamma^j dopamine_{t+j}`` over the rest
    of its episode, truncated at the first terminal step - exactly the
    quantity the backward pass feeds the sequence-level value cells, so a
    predicate selected on these labels tiles the states whose value cells
    actually carry data.  `rows` are transition dicts ordered in time.
    """
    rows = list(rows)
    dopamines = []
    dones = []
    episodes = []
    for row in rows:
        signals = np.asarray(row["signals"], dtype=float).reshape(-1)
        dopamines.append(float(signals[0]))
        dones.append(bool(row["done"]))
        episodes.append(row.get("episode"))
    discount_base = float(gamma)
    labels = []
    for index in range(len(rows)):
        value = 0.0
        discount = 1.0
        for position in range(index, len(rows)):
            if episodes[position] != episodes[index]:
                break
            value += discount * dopamines[position]
            discount *= discount_base
            if dones[position]:
                break
        labels.append(value)
    return labels


def _isotonic_map(reads: np.ndarray,
                  targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the monotone (isotonic) map from raw reads to empirical rates.

    Pool adjacent violators over the sorted (read, label) pairs - the
    classic isotonic regression.  Each block carries its mean read
    (centroid) and mean label (rate), and the returned map interpolates
    between centroids, so every read value in between maps to an
    empirical rate: no quantile bin is involved, and an untested read
    level cannot inherit an empty bin's carried plateau.  Clamping nodes
    are pinned just outside the finite range so ``numpy.interp`` holds
    the end rates in the tails.
    """
    reads = np.asarray(reads, dtype=float)
    targets = np.asarray(targets, dtype=float)
    blocks = []
    for index in np.argsort(reads, kind="stable"):
        blocks.append([float(reads[index]), float(targets[index]), 1.0])
        while len(blocks) >= 2 and (
            blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]
        ):
            top = blocks.pop()
            below = blocks.pop()
            blocks.append([
                below[0] + top[0], below[1] + top[1], below[2] + top[2],
            ])
    block_reads = np.asarray(
        [total_read / weight for total_read, _total, weight in blocks],
        dtype=float,
    )
    block_rates = np.asarray(
        [total_label / weight for _total_read, total_label, weight in blocks],
        dtype=float,
    )
    span = max(float(block_reads.max() - block_reads.min()), 1e-9)
    mids = np.concatenate([
        [block_reads[0] - span - 1.0], block_reads,
        [block_reads[-1] + span + 1.0],
    ])
    rates = np.concatenate([
        [block_rates[0]], block_rates, [block_rates[-1]],
    ])
    return mids, rates


def _nnls(design: np.ndarray, target: np.ndarray,
          ridge: float = 0.0, tol: float = 1e-10,
          max_sweeps: int = 200) -> np.ndarray:
    """Lawson-Hanson active-set non-negative least squares.

    Solves ``min ||A w - b||`` subject to ``w >= 0``.  With `ridge > 0`
    the normal problem is Tikhonov-augmented (``sqrt(ridge) * I`` rows
    against a zero target), which keeps the active-set solves stable when
    channel columns are collinear and shrinks the solution toward zero -
    the classical regularized stacking form.  Column counts here are a
    handful, so the implementation is the textbook loop over
    ``numpy.linalg.lstsq`` on the passive set.
    """
    a = np.asarray(design, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.ndim != 2 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("design must be (rows, columns), target (rows,)")
    columns = a.shape[1]
    if columns == 0:
        return np.zeros(0)
    if ridge > 0.0:
        a = np.vstack([a, np.sqrt(float(ridge)) * np.eye(columns)])
        b = np.concatenate([b, np.zeros(columns)])
    weights = np.zeros(columns)
    passive: List[int] = []
    active = list(range(columns))
    for _ in range(max_sweeps):
        if not active:
            break
        gradient = a.T @ (b - a @ weights)
        best = max(active, key=lambda index: gradient[index])
        if gradient[best] <= tol:
            break
        passive.append(best)
        active.remove(best)
        while True:
            columns_active = a[:, passive]
            solution, *_ = np.linalg.lstsq(
                columns_active, b, rcond=None
            )
            if (solution >= -tol).all():
                weights = np.zeros(columns)
                for position, index in enumerate(passive):
                    weights[index] = max(float(solution[position]), 0.0)
                break
            # Step back along the line from the current weights to the
            # unconstrained solution until the first component hits zero,
            # then release that column back to the active set.
            step = 1.0
            for position, index in enumerate(passive):
                if solution[position] <= 0.0:
                    current = weights[index]
                    gap = current - solution[position]
                    if gap > tol:
                        step = min(step, current / gap)
            weights = weights + step * (
                _expand(solution, passive, columns) - weights
            )
            released = [
                index for index in passive if weights[index] <= tol
            ]
            if not released:
                # Degenerate step (numerically flat): release the most
                # negative component to guarantee progress.
                worst = min(
                    range(len(passive)),
                    key=lambda position: solution[position],
                )
                released = [passive[worst]]
            for index in released:
                weights[index] = 0.0
                passive.remove(index)
                active.append(index)
    return weights


def _expand(solution: np.ndarray, passive: List[int],
            columns: int) -> np.ndarray:
    full = np.zeros(columns)
    for position, index in enumerate(passive):
        full[index] = solution[position]
    return full



from node_learning import NodeLearningMixin
from node_readout import NodeReadoutMixin


class NodeRuntime(NodeLearningMixin, NodeReadoutMixin):
    """Online concept graph with action-conditioned valence learning."""

    def __init__(
        self,
        obs_dim,
        action_space: Iterable[int] = (0, 1),
        lag: int = 0,
        gamma: float = 0.99,
        lam: float = 0.9,
        alpha_v: float = 0.05,
        alpha_q: float = 0.1,
        alpha_now: float = 0.2,
        alpha_mc: float = 0.2,
        birth_threshold: float = 0.45,
        max_nodes: int = 5000,
        exploration: float = 0.1,
        uncertainty_weight: float = 0.05,
        unknown_penalty: float = 0.5,
        soft_neighbors: int = 5,
        soft_sigma: float = 0.75,
        component_kinds: Sequence[str] = (),
        component_min_count: int = 8,
        max_components: int = 128,
        component_threshold: float = 0.2,
        component_smoothing: float = 0.5,
        component_weight: float = 0.5,
        component_gating: bool = False,
        graph_rate: float = 0.1,
        memory_decay: float = 0.9,
        memory_min_strength: float = 0.02,
        memory_weight: float = 0.1,
        association_rate: float = 0.0,
        valence_rate: float = 0.2,
        feature_rate: float = 0.05,
        feature_action_prior: float = 2.0,
        association_prior: float = 2.0,
        importance_weighting: bool = False,
        variance_births: bool = True,
        utility_pruning: bool = True,
        pain_horizon: Optional[int] = None,
        point_labels: bool = False,
        memory_credit: bool = False,
        rule_min_coverage: float = 0.0,
        rule_min_primitive_coverage: float = 0.0,
        pairwise_features: bool = True,
        voi_target: int = 3,
        voi_cell_cap: int = 16,
        voi_min_prior: float = 0.5,
        voi_threat_min: float = 0.25,
        feature_value_prior: float = 4.0,
        feature_value_weight: float = 0.02,
        subgraph_cells: bool = False,
        predicate_readouts: bool = False,
        event_memory_risk: bool = True,
        contrast_objective: bool = False,
        value_trust_samples: float = 150.0,
        control_risk_weight: float = 0.0,
        curiosity_weight: float = 0.0,
        value_bank_only: bool = False,
        value_horizon: Optional[int] = None,
        pain_gamma: Optional[float] = None,
        ood_trust_scale: float = 0.0,
        obs_trust: float = 1.0,
        action_temperature: float = 0.1,
        action_uncertainty: float = 0.5,
        branch_transitions: bool = False,
        action_edge_weight: float = 0.0,
        action_edge_trust_samples: float = 50.0,
        seed: int = 0,
        scales: Optional[Sequence[float]] = None,
        readout_propagation_gain: float = 0.0,
        readout_propagation_steps: int = 1,
        readout_propagation_edges: str = "transition",
        online_feature_learning: bool = False,
        online_feature_options: Optional[dict] = None,
    ):
        if int(obs_dim) < 1:
            raise ValueError("obs_dim must be positive")
        if int(lag) < 0:
            raise ValueError("lag must be non-negative")
        if not 0.0 <= float(gamma) < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if not 0.0 <= float(lam) <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        if float(alpha_v) <= 0.0 or float(alpha_q) <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < float(alpha_now) <= 1.0:
            raise ValueError("alpha_now must be in (0, 1]")
        if not 0.0 < float(alpha_mc) <= 1.0:
            raise ValueError("alpha_mc must be in (0, 1]")
        if float(birth_threshold) <= 0.0:
            raise ValueError("birth_threshold must be positive")
        if float(value_trust_samples) < 0.0:
            raise ValueError("value_trust_samples must be non-negative")
        if float(curiosity_weight) < 0.0:
            raise ValueError("curiosity_weight must be non-negative")
        if float(readout_propagation_gain) < 0.0:
            raise ValueError("readout_propagation_gain must be non-negative")
        if int(readout_propagation_steps) < 1:
            raise ValueError("readout_propagation_steps must be positive")
        if str(readout_propagation_edges) not in (
            "transition", "association", "both"
        ):
            raise ValueError(
                "readout_propagation_edges must be transition/association/both"
            )
        if int(max_nodes) < 1:
            raise ValueError("max_nodes must be positive")
        if not 0.0 <= float(exploration) <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        if float(uncertainty_weight) < 0.0:
            raise ValueError("uncertainty_weight must be non-negative")
        if float(unknown_penalty) < 0.0:
            raise ValueError("unknown_penalty must be non-negative")
        if int(soft_neighbors) < 1:
            raise ValueError("soft_neighbors must be positive")
        if float(soft_sigma) <= 0.0:
            raise ValueError("soft_sigma must be positive")
        if int(component_min_count) < 1:
            raise ValueError("component_min_count must be positive")
        if int(max_components) < 0:
            raise ValueError("max_components must be non-negative")
        if not 0.0 < float(component_smoothing) <= 1.0:
            raise ValueError("component_smoothing must be in (0, 1]")
        if float(component_weight) < 0.0:
            raise ValueError("component_weight must be non-negative")
        if not 0.0 < float(graph_rate) <= 1.0:
            raise ValueError("graph_rate must be in (0, 1]")
        if not 0.0 < float(memory_decay) <= 1.0:
            raise ValueError("memory_decay must be in (0, 1]")
        if float(memory_min_strength) < 0.0:
            raise ValueError("memory_min_strength must be non-negative")
        if not 0.0 <= float(memory_weight) <= 1.0:
            raise ValueError("memory_weight must be in [0, 1]")
        if not 0.0 <= float(association_rate) <= 1.0:
            raise ValueError("association_rate must be in [0, 1]")
        if not 0.0 < float(valence_rate) <= 1.0:
            raise ValueError("valence_rate must be in (0, 1]")
        if not 0.0 < float(feature_rate) <= 1.0:
            raise ValueError("feature_rate must be in (0, 1]")
        if float(action_temperature) < 0.0:
            raise ValueError("action_temperature must be non-negative")
        if float(feature_action_prior) < 0.0:
            raise ValueError("feature_action_prior must be non-negative")
        if float(association_prior) < 0.0:
            raise ValueError("association_prior must be non-negative")
        if pain_horizon is not None and int(pain_horizon) < 1:
            raise ValueError("pain_horizon must be positive")
        if int(voi_target) < 1:
            raise ValueError("voi_target must be positive")
        if int(voi_cell_cap) < 1:
            raise ValueError("voi_cell_cap must be positive")
        if not 0.0 <= float(voi_min_prior) <= 1.0:
            raise ValueError("voi_min_prior must be in [0, 1]")
        if not 0.0 <= float(voi_threat_min) <= 1.0:
            raise ValueError("voi_threat_min must be in [0, 1]")
        if float(feature_value_prior) < 0.0:
            raise ValueError("feature_value_prior must be non-negative")
        if float(feature_value_weight) < 0.0:
            raise ValueError("feature_value_weight must be non-negative")
        if value_horizon is not None and int(value_horizon) < 1:
            raise ValueError("value_horizon must be positive")
        if float(ood_trust_scale) < 0.0:
            raise ValueError("ood_trust_scale must be non-negative")
        if not 0.0 <= float(obs_trust) <= 1.0:
            raise ValueError("obs_trust must be in [0, 1]")
        if float(action_uncertainty) < 0.0:
            raise ValueError("action_uncertainty must be non-negative")
        actions = tuple(int(action) for action in action_space)
        if not actions:
            raise ValueError("action_space must not be empty")

        self.obs_dim = int(obs_dim)
        self.action_space = actions
        self.lag = int(lag)
        self.gamma = float(gamma)
        # The pain value's own discount.  The shared gamma is a dopamine
        # (long survival) horizon; the pain channel needs a discount that
        # matches how far ahead a death is actually predictable, otherwise
        # the return-to-go is nearly flat across the window the planner
        # searches.  None keeps the historical behaviour.
        self.pain_gamma = (
            float(pain_gamma) if pain_gamma is not None else float(gamma)
        )
        if not 0.0 < self.pain_gamma <= 1.0:
            raise ValueError("pain_gamma must be in (0, 1]")
        self.lam = float(lam)
        self.alpha_v = float(alpha_v)
        self.alpha_q = float(alpha_q)
        self.alpha_now = float(alpha_now)
        self.alpha_mc = float(alpha_mc)
        self.birth_threshold = float(birth_threshold)
        self.max_nodes = int(max_nodes)
        self.exploration = float(exploration)
        self.uncertainty_weight = float(uncertainty_weight)
        self.unknown_penalty = float(unknown_penalty)
        self.soft_neighbors = int(soft_neighbors)
        self.soft_sigma = float(soft_sigma)
        self.component_kinds = tuple(str(kind) for kind in component_kinds)
        self.component_min_count = int(component_min_count)
        self.max_components = int(max_components)
        self.component_threshold = float(component_threshold)
        self.component_smoothing = float(component_smoothing)
        self.component_weight = float(component_weight)
        self.component_gating = bool(component_gating)
        self.graph_rate = float(graph_rate)
        self.memory_decay = float(memory_decay)
        self.memory_min_strength = float(memory_min_strength)
        self.memory_weight = float(memory_weight)
        self.association_rate = float(association_rate)
        self.valence_rate = float(valence_rate)
        # Retained for API compatibility: the feature pain anchor is now a
        # 1/n cumulative mean (see `_update_feature_pain`), so no EMA rate
        # drives it anymore.
        self.feature_rate = float(feature_rate)
        self.feature_action_prior = float(feature_action_prior)
        # Pseudo-counts shrinking a node's association-channel pain rate
        # toward the global base rate while its own sample count is thin.
        self.association_prior = float(association_prior)
        # Per-node valence rate statistics: how many transitions ran
        # through each node and how much pain they carried.  Both clean
        # and painful transitions are samples - pain-only evidence would
        # make the channel a monotone "ever adjacent to pain" envelope
        # instead of a base rate.
        self._association_stats = {}
        self._association_global = [0, 0.0]
        # Per-decision clipped importance weighting (P1): when on, each
        # transition's window label is weighted by the clipped product of
        # uniform-target propensity ratios over the window's governing
        # actions, so cells estimate their conditional rates on a
        # state-action mix closer to uniform instead of the behavior
        # policy's occupancy.  The clip min(1, target/b) bounds the
        # variance (Retrace-style); rows without a propensity read one
        # and stay unweighted.
        self.importance_weighting = bool(importance_weighting)
        self.importance_target = 1.0 / len(self.action_space)
        # Refinement births (P3): a cell whose valence variance stays
        # above the population bar holds a mixture of futures; a query
        # far enough inside its territory births a new center there.
        self.variance_births = bool(variance_births)
        # Utility pruning (P3): when the node budget blocks a birth, the
        # least useful prototype is removed and its statistics merge into
        # its nearest surviving neighbour, so growth keeps tracking the
        # distribution instead of freezing at the first pass.
        self.utility_pruning = bool(utility_pruning)
        # Population variance bar for refinement births, recomputed once
        # per episode (an O(nodes) pass per row would dominate training).
        self._variance_bar = None
        # Prototype ids the running episode already assigned: a prune
        # must never remove a node the episode's backward pass still
        # references, so victims come from the rest of the cover.
        self._episode_nodes: set = set()
        self.action_temperature = float(action_temperature)
        self.action_uncertainty = float(action_uncertainty)
        # Per (feature root, acting action) pain statistics.  The acting
        # action of the transition s_m -> s_{m+1} is the one issued `lag`
        # steps earlier (`history[lag]`), which is also the first pending
        # coordinate of the encoded state at decision time.
        self.feature_action_stats = {}
        # Decision-time readout predictions vs actual precursor outcomes,
        # per (feature root, acting action), plus a bounded buffer of the
        # most confident surprises (the counterfactual target list).
        self.feature_calibration = {}
        self.surprise_buffer = []
        self.surprise_capacity = 64
        # Counterfactual guidance: how many samples a (feature, action) cell
        # should have before the guide stops testing it, how many guide
        # tests one cell may receive, and the prior above which a root's
        # consequence counts as large enough to test.
        self.voi_target = int(voi_target)
        self.voi_cell_cap = int(voi_cell_cap)
        self.voi_min_prior = float(voi_min_prior)
        # Counterfactual tests fire only where the calibrated threat read
        # says the danger rate is at least this high: decision-critical
        # states are where a rarely-taken action's real outcome changes
        # what the planner can see.
        self.voi_threat_min = float(voi_threat_min)
        self.voi_tests = {}
        # Sequence-level value: per (feature, acting action) cells learning
        # the discounted dopamine return of the paths that followed them,
        # shrunk toward a running global baseline.  `feature_value_weight`
        # is how much one unit of this value trades against one unit of
        # accumulated hazard in the scalarized search objective.
        self.feature_value_stats = {}
        self.feature_value_prior = float(feature_value_prior)
        self.feature_value_weight = float(feature_value_weight)
        self.value_horizon = (
            None if value_horizon is None else int(value_horizon)
        )
        # Prototype readouts shrink toward the population baseline as the
        # query state leaves the supported region: statistics memorized on
        # other worlds must not be reported with full confidence.
        self.ood_trust_scale = float(ood_trust_scale)
        # Observation-anchored statistics (prototypes and their memories) are
        # down-weighted relative to induced, verified feature channels; 1.0
        # keeps the historical max-combination behaviour.
        self.obs_trust = float(obs_trust)
        # Readout propagation: relax the soft-neighbour seed weights over the
        # learned graph edges before reading values/rates, so the readout
        # averages the *topological* neighbourhood instead of the Euclidean
        # one.  Gain 0 (default) keeps the historical geometric readout
        # bit-for-bit; the relaxation only runs when the gain is positive.
        self.readout_propagation_gain = float(readout_propagation_gain)
        self.readout_propagation_steps = int(readout_propagation_steps)
        self.readout_propagation_edges = str(readout_propagation_edges)
        # Derived readout structures (prototype adjacency + per-prototype
        # columns).  Dropped whenever the runtime learns, so an evaluation
        # phase builds them once and every query reuses them.
        self._readout_cache: Optional[dict] = None
        self.value_baseline = 0.0
        # Retained for API compatibility: the baseline is a 1/n cumulative
        # mean of the value targets (see `_update_feature_value`), not a
        # constant-rate EMA.
        self.value_baseline_rate = 0.02
        self._value_baseline_count = 0
        # Precursor label window: how many entered states ahead a pain event
        # still labels the current state.  Defaults to lag + 1, the first
        # state an action issued now can physically affect.
        self.pain_horizon = (
            int(pain_horizon) if pain_horizon is not None else int(self.lag) + 1
        )
        # Point-event labels: the only pain the world emits is the feedback
        # of the state just entered.  In this mode every root trains on that
        # single next-state pain (the transition's own outcome), not on a
        # hand-chosen precursor window; `_node_horizon` is pinned to one so
        # the graded readout keeps treating all roots as governable.
        self.point_labels = bool(point_labels)
        # Retrospective memory credit: a pain event also corrects the
        # memories that were active just before it, and links its event
        # memory back to that preceding context.  The pain stays a point
        # event - no state label is rewritten - but the eligibility trace
        # (gamma*lambda, the same constants as the TD path) decides how
        # much of it reaches an earlier memory.
        self.memory_credit = bool(memory_credit)
        self.branch_transitions = bool(branch_transitions)
        self.action_edge_weight = max(float(action_edge_weight), 0.0)
        self.action_edge_trust_samples = max(
            float(action_edge_trust_samples), 0.0
        )
        self._memory_traces: Dict[str, float] = {}
        # Diagnostic counters: how many pain events corrected an earlier
        # memory, how much eligibility mass moved, and how many context
        # links the event added.  Read-only; the probe script reports them.
        self.memory_credit_stats = {
            "events": 0,
            "corrected": 0,
            "weight": 0.0,
            "links": 0,
            "corrected_ids": set(),
        }
        # Resource bound on how many preceding context nodes one event
        # memory links to; not a semantic window.
        self.memory_context_limit = 32
        self.gated_edges: Dict[
            Tuple[str, Tuple[int, ...], int], Dict[str, dict]
        ] = {}
        self.last_signature: Tuple[int, ...] = ()
        self.rng = np.random.RandomState(seed)

        state_dim = self.obs_dim + self.lag
        if scales is None:
            scales = np.ones(state_dim, dtype=float)
        scales = np.asarray(scales, dtype=float).reshape(-1)
        if scales.shape != (state_dim,):
            raise ValueError(
                "scales must have length obs_dim + lag (%d)" % state_dim
            )
        if not np.isfinite(scales).all() or (scales <= 0.0).any():
            raise ValueError("scales must be positive and finite")
        self.scales = scales

        self.graph = NodeGraph()
        # Read-only branch record: the same graph, separate strength states
        # for the current state and the recent trace.  Decision paths still
        # read the graph fields; branches are available for the transition,
        # prediction and memory layers without changing behaviour.
        self.branches = BranchBank(self.graph)
        self.branches.add(CurrentBranch(graph=self.graph))
        self.branches.add(RecentBranch(graph=self.graph, decay=0.8))
        self.prediction_branch = self.branches.add(
            PredictionBranch(graph=self.graph)
        )
        # Per-node precision of the planner's one-step prediction: when a
        # root was predicted active, how often was it actually active after
        # the real transition?  Used as the prediction-credibility channel.
        self.prediction_stats: Dict[str, List[float]] = {}
        self._last_prediction: set = set()
        self.components = ComponentBank(
            self.graph,
            threshold=self.component_threshold,
            smoothing=self.component_smoothing,
        )
        self.pain_node = self.graph.add_node(
            CONCEPT, "valence:pain", role="valence"
        )
        self.dopamine_node = self.graph.add_node(
            CONCEPT, "valence:dopamine", role="valence"
        )
        self.action_nodes = {
            int(action): self.graph.add_node(
                CONCEPT, "action:%d" % int(action), role="action"
            )
            for action in self.action_space
        }
        self.last_action_probs = (1.0, 0.0)
        self.rules = FeatureRuleBank(
            pairwise_features=pairwise_features,
            min_coverage=rule_min_coverage,
            min_primitive_coverage=rule_min_primitive_coverage,
        )
        self.pairwise_features = bool(pairwise_features)
        # Optional second bank induced on a longer precursor window; roots
        # installed from it carry their own `pain_horizon` tag.
        self.rules_long = None
        # Optional bank induced from high-return coverage instead of pain
        # coverage: its roots join the sequence-level value readout, giving
        # the value channel predicates where the pain labels have no reach.
        self.rules_value = None
        self.feature_ids: List[str] = []
        # Roots installed from the value bank.  Empty until such a bank is
        # installed; the value readout then covers the risk-bank roots only.
        self.value_feature_ids: List[str] = []
        # D2: branch-level cells.  When on, every node of a rule's subgraph
        # (root and its materialized descendants) trains its own statistics
        # and the readout comes from the most specific active member, so an
        # OR over two directions keeps them separated instead of averaging
        # both into one root cell.  Measured inert on the survival
        # benchmark (six seeds, two readout forms, bit-identical runs) -
        # the planner's plan margins are wider than the read refinements -
        # so it is kept available but off by default.
        self.subgraph_cells = bool(subgraph_cells)
        # D3: when on, the risk readout drops the prototype association
        # channel entirely - prototypes stay as graph structure and event
        # memory, and the generalization load moves to the coordinate
        # predicates.  States outside every predicate read zero hazard
        # instead of an extrapolated prototype statistic.
        self.predicate_readouts = bool(predicate_readouts)
        # The feature-keyed event memories also ride the risk readout
        # through the max-combine; each death memory keeps its predicate's
        # whole band flagged as hazardous.  This switch removes that
        # channel to measure how much of the large-data degradation it
        # carries.
        self.event_memory_risk = bool(event_memory_risk)
        # F1: path-relative search objective.  When on, the per-step cost
        # is the action contrast (per-state relative action quality) and
        # the endpoint cost is the value term alone, so the fixed toll of
        # traversing a genuinely hazardous band no longer penalizes the
        # plans that must go through it.
        self.contrast_objective = bool(contrast_objective)
        # D4: evidence-proportional trust.  In the contrast objective the
        # endpoint value term is scaled by cbar/(cbar + K) with cbar the
        # mean observation count behind that state's value read - a thin
        # value cell speaks softly, a well-sampled one at full weight - so
        # the influence of the value channel grows with the evidence
        # behind it instead of being a fixed hyperparameter.
        self.value_trust_samples = float(value_trust_samples)
        # Threat grading: under the contrast objective the step cost is the
        # real threat (the unavoidable share) plus this weight times the
        # controllable excess.  Zero keeps the controllable risk free to
        # take - entering a risky but action-dependent band is judged by
        # the value ledger, not by the hazard readout.
        self.control_risk_weight = float(control_risk_weight)
        # Decision-side curiosity: the planner's cost is shaped by a
        # potential over the evidence behind a state's active cells
        # (higher potential = less evidence = more novel).  It is applied
        # as a potential difference, so the shaped objective keeps the same
        # optimum as the unshaped one (Ng-Harada-Russell); this is not the
        # collection-side VOI override, whose two forms both tested
        # negative.  Zero disables the term.
        self.curiosity_weight = float(curiosity_weight)
        # L1: when on, the value readout averages only over the dedicated
        # value-bank roots.  The union readout lets the risk bank's cells
        # (learned mostly from drifting continuations) dilute the value
        # signal everywhere - the before/after audit measured the dilution
        # at -4 to -14 against the data's own return table.
        self.value_bank_only = bool(value_bank_only)
        # Stacked hazard composition (P2): when fitted, `pain_signal_batch`
        # reports a non-negative least-squares composition of its channels
        # instead of the hand-written max, followed by a monotone map to
        # the empirical danger rate.  Unfitted (None), the historical
        # max-combination runs unchanged.
        self.hazard_composition: Optional[Dict[str, float]] = None
        self.hazard_calibration: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._feature_all: List[str] = []
        self._feature_descendants: Dict[str, List[str]] = {}
        self._feature_objects: Dict[str, object] = {}
        self._feature_children: Dict[str, List[str]] = {}
        self._feature_keys: Dict[str, str] = {}
        self._feature_counter = 0
        self._prototype_ids: List[str] = []
        self._prototype_matrix = np.zeros((0, state_dim), dtype=float)
        self._traces: Dict[str, List[float]] = {}
        self.pending = self._zero_pending()
        self.last_node: Optional[str] = None
        self.last_action: Optional[int] = None
        self.last_state: Optional[np.ndarray] = None
        self.last_scores: Dict[int, dict] = {}
        self.last_sources: List[str] = []
        self._component_bound = set()
        self._node_pain_count = {}
        self._node_pain_mean = {}
        self._node_pain_m2 = {}
        self._memory_counter = 0
        self.online_feature_learner = None
        if online_feature_learning:
            self.enable_online_feature_learning(
                **({} if online_feature_options is None
                   else dict(online_feature_options))
            )

    def enable_online_feature_learning(self, **options):
        """Enable default-off streaming feature discovery on real feedback."""
        self.online_feature_learner = OnlineFeatureLearner(
            obs_dim=self.obs_dim, lag=self.lag, scales=self.scales,
            **options
        )
        return self.online_feature_learner

    def observe_online_transition(self, transition):
        """Feed one real transition to the optional online feature learner."""
        learner = self.online_feature_learner
        if learner is None:
            return None
        return learner.observe_transition(transition)

    # ---- encoding ----------------------------------------------------------
    def _zero_pending(self):
        if self.lag == 0:
            return deque(maxlen=0)
        return deque([0] * self.lag, maxlen=self.lag)

    def encode(self, observation) -> np.ndarray:
        observation = np.asarray(observation, dtype=float).reshape(-1)
        if observation.shape != (self.obs_dim,):
            raise ValueError(
                "expected observation with %d values" % self.obs_dim
            )
        if not np.isfinite(observation).all():
            raise ValueError("observation must be finite")
        if self.lag:
            state = np.concatenate(
                [observation, np.asarray(list(self.pending), dtype=float)]
            )
        else:
            state = observation
        return state / self.scales

    # ---- node assignment ---------------------------------------------------
    def _nearest(self, state: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._prototype_ids:
            return None, float("inf")
        distances = np.linalg.norm(self._prototype_matrix - state, axis=1)
        index = int(np.argmin(distances))
        return self._prototype_ids[index], float(distances[index])


    def top_nodes(self, limit: int = 10) -> List[dict]:
        nodes = sorted(
            self.graph.nodes.values(),
            key=lambda node: node.value + node.immediate_pain,
            reverse=True,
        )
        return [
            dict(
                id=node.id,
                count=node.count,
                pain=node.value + node.immediate_pain,
                dopamine=node.dopamine + node.immediate_dopamine,
                value=node.value,
                pain_now=node.immediate_pain,
                dopamine_now=node.immediate_dopamine,
                features=(
                    None if node.features is None
                    else [float(value) for value in node.features]
                ),
            )
            for node in nodes[:limit]
        ]

    def top_edges(self, limit: int = 10) -> List[dict]:
        edges = sorted(
            self.graph.action_edges.values(),
            key=lambda edge: edge.q_pain,
            reverse=True,
        )
        return [
            dict(
                source=edge.source,
                action=edge.action,
                target=edge.target,
                count=edge.count,
                pain=edge.q_pain,
                dopamine=edge.q_dopamine,
            )
            for edge in edges[:limit]
        ]

    def snapshot(self) -> dict:
        return dict(
            nodes=len(self.graph.nodes),
            action_edges=len(self.graph.action_edges),
            associations=sum(
                edge.kind == ASSOCIATION
                for edge in self.graph.edges.values()
            ),
            transitions=sum(
                edge.kind == TRANSITION_EDGE
                for edge in self.graph.edges.values()
            ),
            memories=sum(
                node.kind == MEMORY
                for node in self.graph.nodes.values()
            ),
            prototypes=len(self._prototype_ids),
            rules=len(self.rules.rules),
            features=len(self.feature_ids),
            feature_nodes=len(self._feature_all),
            components=len(self.components.specs),
            component_states=sum(
                node.role == "component_state"
                for node in self.graph.nodes.values()
            ),
        )

