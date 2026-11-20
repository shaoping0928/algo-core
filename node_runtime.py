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

from node_components import ComponentBank
from node_features import (
    FeatureRuleBank,
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


class NodeRuntime:
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
        ood_trust_scale: float = 0.0,
        obs_trust: float = 1.0,
        action_temperature: float = 0.1,
        action_uncertainty: float = 0.5,
        seed: int = 0,
        scales: Optional[Sequence[float]] = None,
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
        self.rules = FeatureRuleBank()
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

    def _create_node(self, state: np.ndarray) -> str:
        node_id = self.graph.add_node(
            CONCEPT, features=np.asarray(state, dtype=float).copy()
        )
        self._prototype_ids.append(node_id)
        row = np.asarray(state, dtype=float).reshape(1, -1)
        if self._prototype_matrix.size == 0:
            self._prototype_matrix = row.copy()
        else:
            self._prototype_matrix = np.vstack([self._prototype_matrix, row])
        return node_id

    def _assign(self, state: np.ndarray, create: bool = True,
                count: bool = True) -> str:
        node_id, distance = self._nearest(state)
        if node_id is None:
            node_id = self._create_node(state)
        elif create and distance > self.birth_threshold:
            if len(self._prototype_ids) < self.max_nodes:
                node_id = self._create_node(state)
            elif self.utility_pruning and len(self._prototype_ids) > 1:
                # The budget blocks the birth: consolidate the least
                # useful prototype into its neighbour first (GNG-U), so
                # the cover keeps refining instead of absorbing far
                # states into stale cells forever.  Nodes the running
                # episode still references are not candidates.
                self._prune_prototypes(exclude=self._episode_nodes)
                if len(self._prototype_ids) < self.max_nodes:
                    node_id = self._create_node(state)
        elif create and self.variance_births and self._variance_split(
            node_id, distance
        ):
            # Refinement birth: the nearest cell holds a statistical
            # mixture and the query sits far enough inside its territory
            # to deserve its own statistics.
            if len(self._prototype_ids) < self.max_nodes:
                node_id = self._create_node(state)
            elif self.utility_pruning and len(self._prototype_ids) > 1:
                self._prune_prototypes(exclude=self._episode_nodes)
                if len(self._prototype_ids) < self.max_nodes:
                    node_id = self._create_node(state)
        if count:
            self.graph.nodes[node_id].count += 1
        self._episode_nodes.add(node_id)
        return node_id

    def _refresh_variance_bar(self) -> None:
        """Population-relative valence-variance bar, once per episode."""
        variances = []
        for node_id, count in self._node_pain_count.items():
            if count < self.component_min_count:
                continue
            variances.append(
                self._node_pain_m2.get(node_id, 0.0) / max(count - 1, 1)
            )
        if len(variances) > 1:
            self._variance_bar = float(
                np.median(variances) + np.std(variances)
            )
        else:
            self._variance_bar = None

    def _variance_split(self, node_id: str, distance: float) -> bool:
        """Refinement-birth test for the nearest cell.

        The cell needs enough samples for its variance to mean anything,
        the variance must sit above the same population-relative bar the
        component binder uses, and the query must sit well inside the
        cell's territory - a split at the periphery gives the mixture's
        outer states their own center while the core keeps its history.
        """
        if self._variance_bar is None:
            return False
        count = self._node_pain_count.get(node_id, 0)
        if count < self.component_min_count:
            return False
        variance = self._node_pain_m2.get(node_id, 0.0) / max(count - 1, 1)
        if variance <= self._variance_bar:
            return False
        return distance > 0.5 * self.birth_threshold

    def _prune_prototypes(self, exclude=()) -> Optional[str]:
        """Remove the least useful prototype, merging it into its neighbour.

        Utility follows the growing-neural-gas principle - the error a
        removal would induce: a node's visit count times the squared
        distance to its nearest neighbour plus the squared gap between
        their valence values.  A rare cell far from every other cell with
        a distinct outcome is expensive to delete and survives; a
        near-duplicate of its neighbour is cheap and merges first, its
        statistics carried over as count-weighted means so no evidence is
        lost, only consolidated.  `exclude` names nodes that must not be
        removed - the running episode's own assignments, whose backward
        pass still references them.
        """
        count_all = len(self._prototype_ids)
        if count_all <= 1:
            return None
        matrix = self._prototype_matrix
        state_norm = np.einsum("ij,ij->i", matrix, matrix)[:, None]
        proto_norm = np.einsum("ij,ij->i", matrix, matrix)[None, :]
        distances = state_norm + proto_norm - 2.0 * (matrix @ matrix.T)
        np.fill_diagonal(distances, np.inf)
        nearest_index = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(count_all), nearest_index]
        excluded = set(exclude)
        utilities = np.full(count_all, np.inf)
        for index, node_id in enumerate(self._prototype_ids):
            if node_id in excluded:
                continue
            node = self.graph.nodes[node_id]
            other = self.graph.nodes[self._prototype_ids[nearest_index[index]]]
            gap = node.value - other.value
            utilities[index] = max(float(node.count), 1.0) * (
                nearest_distance[index] ** 2 + gap * gap
            )
        victim_index = int(np.argmin(utilities))
        if not np.isfinite(utilities[victim_index]):
            # Every candidate is protected (a single-node episode at the
            # cap): the birth is absorbed into the nearest cell.
            return None
        victim_id = self._prototype_ids[victim_index]
        host_id = self._prototype_ids[int(nearest_index[victim_index])]
        victim = self.graph.nodes[victim_id]
        host = self.graph.nodes[host_id]
        victim_count = max(float(victim.count), 1.0)
        host_count = max(float(host.count), 1.0)
        total = victim_count + host_count
        blend = victim_count / total
        host.value += blend * (victim.value - host.value)
        host.dopamine += blend * (victim.dopamine - host.dopamine)
        host.immediate_pain += blend * (
            victim.immediate_pain - host.immediate_pain
        )
        host.immediate_dopamine += blend * (
            victim.immediate_dopamine - host.immediate_dopamine
        )
        host.count += victim.count
        victim_stats = self._association_stats.pop(victim_id, None)
        if victim_stats is not None:
            host_stats = self._association_stats.setdefault(
                host_id, [0, 0.0]
            )
            host_stats[0] += victim_stats[0]
            host_stats[1] += victim_stats[1]
        v_count = self._node_pain_count.pop(victim_id, None)
        if v_count:
            h_count = self._node_pain_count.get(host_id, 0)
            v_mean = self._node_pain_mean.get(victim_id, 0.0)
            h_mean = self._node_pain_mean.get(host_id, 0.0)
            v_m2 = self._node_pain_m2.get(victim_id, 0.0)
            h_m2 = self._node_pain_m2.get(host_id, 0.0)
            merged_count = h_count + v_count
            if merged_count > 0:
                delta = v_mean - h_mean
                merged_m2 = (
                    h_m2 + v_m2
                    + delta * delta * h_count * v_count / merged_count
                )
                self._node_pain_count[host_id] = merged_count
                self._node_pain_mean[host_id] = (
                    (h_count * h_mean + v_count * v_mean) / merged_count
                )
                self._node_pain_m2[host_id] = max(merged_m2, 0.0)
        self._component_bound.discard(victim_id)
        self.graph._remove_node(victim_id)
        self._prototype_ids.pop(victim_index)
        self._prototype_matrix = np.delete(
            self._prototype_matrix, victim_index, axis=0
        )
        return victim_id

    def _note_pain(self, node_id: str, pain: float) -> None:
        pain = float(pain)
        self._node_pain_count[node_id] = (
            self._node_pain_count.get(node_id, 0) + 1
        )
        count = self._node_pain_count[node_id]
        mean = self._node_pain_mean.get(node_id, 0.0)
        delta = pain - mean
        mean += delta / count
        self._node_pain_mean[node_id] = mean
        self._node_pain_m2[node_id] = (
            self._node_pain_m2.get(node_id, 0.0)
            + delta * (pain - mean)
        )
        self._maybe_bind_components(node_id)

    def _maybe_bind_components(self, node_id: str) -> None:
        if not self.component_kinds or self.max_components == 0:
            return
        if node_id in self._component_bound:
            return
        count = self._node_pain_count.get(node_id, 0)
        if count < self.component_min_count:
            return
        if len(self.components.specs) >= self.max_components:
            return
        variance = self._node_pain_m2.get(node_id, 0.0) / max(
            count - 1, 1
        )
        variances = [
            self._node_pain_m2[other] / max(
                self._node_pain_count[other] - 1, 1
            )
            for other in self._node_pain_count
            if self._node_pain_count[other] >= self.component_min_count
        ]
        if len(variances) > 1:
            threshold = float(np.median(variances) + np.std(variances))
        else:
            threshold = float("inf")
        # The decision to instantiate a component is learned from the
        # node's valence variance relative to the population of visited nodes.
        if variance <= threshold:
            return
        for kind in self.component_kinds:
            if len(self.components.specs) >= self.max_components:
                break
            component_id = "%s:%s" % (node_id, kind)
            if component_id in self.components.specs:
                continue
            self.components.bind(
                node_id, kind=kind, component_id=component_id,
                threshold=self.component_threshold,
                smoothing=self.component_smoothing,
            )
        self._component_bound.add(node_id)

    # ---- runtime components ------------------------------------------------
    def bind_component(self, base_node: str, kind: str = "difference",
                       params: Sequence[float] = (),
                       component_id: Optional[str] = None,
                       threshold: Optional[float] = None,
                       hysteresis: Optional[float] = None,
                       smoothing: Optional[float] = None) -> str:
        """Bind a numeric component; its +/0/- states become concept nodes."""
        return self.components.bind(
            base_node, kind=kind, params=params,
            component_id=component_id, threshold=threshold,
            hysteresis=hysteresis, smoothing=smoothing,
        )

    def _component_step(self, node_id: str,
                        valence: Optional[np.ndarray] = None) -> None:
        if not self.components.specs:
            return
        self.components.step({node_id: 1.0})
        if valence is not None:
            self.components.observe_valence(
                float(valence[1]), float(valence[0])
            )

    def _new_memory(self) -> str:
        self._memory_counter += 1
        return self.graph.add_memory(
            [], "memory_runtime_%d" % self._memory_counter
        )

    def _interact(self, current: str, next_node: str, pain: float = 0.0,
                  dopamine: float = 0.0,
                  current_features: Iterable[str] = (),
                  next_features: Iterable[str] = (),
                  learn_memory_credit: bool = False) -> None:
        """Create association, causal transition and signal-driven memory."""
        pain = float(pain)
        dopamine = float(dopamine)
        current_features = list(dict.fromkeys(current_features))
        next_features = list(dict.fromkeys(next_features))
        if current == next_node:
            # A self-loop is still a visit: the per-node valence rate must
            # see every transition through the node, even the ones that
            # cannot learn a structural edge.
            self._note_valence([current], pain)
            if learn_memory_credit:
                self._advance_memory_traces(
                    [current] + current_features, (), None, pain
                )
            return
        self.graph.activate([current, next_node], [1.0, 1.0])
        self.graph.observe_association(
            [current, next_node], rate=self.graph_rate
        )
        edge = self.graph.observe_transition(
            current, next_node, rate=self.graph_rate
        )
        memories = self.memories_for(
            [current, next_node] + current_features + next_features
        )
        # Valence rate statistics: every transition through a node - clean
        # or painful - is a sample of its local pain rate.  Pain-only
        # evidence would make the channel a monotone envelope ("has this
        # neighborhood ever been adjacent to pain") instead of a rate.
        self._note_valence(
            [current, next_node] + list(memories), pain
        )
        created_memory = None
        if pain > 0.0 and not memories:
            created_memory = self._new_memory()
            memory = self.graph.nodes[created_memory]
            memory.value = pain
            memory.dopamine = dopamine
            memory.count = 1
            self.graph.add_membership(created_memory, current)
            self.graph.add_membership(created_memory, next_node)
            if edge.edge_node is not None:
                self.graph.add_membership(
                    created_memory, edge.edge_node
                )
            for feature_id in dict.fromkeys(
                current_features + next_features
            ):
                self.graph.add_membership(created_memory, feature_id)
            self._note_valence([created_memory], pain)
        if pain > 0.0:
            # Features are installed after the first epoch.  Later pain
            # events reuse the existing memories, so index those memories
            # with the active features here instead of only at creation.
            for memory_id in memories:
                for feature_id in dict.fromkeys(
                    current_features + next_features
                ):
                    self.graph.add_membership(memory_id, feature_id)
        self._reinforce_memories(
            [current, next_node] + current_features + next_features
        )
        extra = [current, next_node]
        if created_memory is not None:
            extra.append(created_memory)
        self._learn_pain_association(pain, extra)
        if learn_memory_credit:
            self._advance_memory_traces(
                [current, next_node] + current_features + next_features,
                list(memories), created_memory, pain,
            )

    def memories_for(self, node_ids: Iterable[str]) -> Dict[str, float]:
        """Find memory nodes through membership edges of active nodes."""
        result: Dict[str, float] = {}
        for node_id in dict.fromkeys(node_ids):
            for key in self.graph.incoming.get(node_id, []):
                if key[2] != MEMBERSHIP:
                    continue
                edge = self.graph.edges[key]
                result[key[0]] = max(
                    result.get(key[0], 0.0), float(edge.weight)
                )
            # Associations are one-hop retrieval paths to memories of
            # related nodes; membership still carries the strength.
            for key in self.graph.outgoing.get(node_id, []):
                if key[2] != ASSOCIATION:
                    continue
                edge = self.graph.edges[key]
                other = (
                    edge.target if edge.source == node_id
                    else edge.source
                )
                for member_key in self.graph.incoming.get(other, []):
                    if member_key[2] != MEMBERSHIP:
                        continue
                    member_edge = self.graph.edges[member_key]
                    result[member_key[0]] = max(
                        result.get(member_key[0], 0.0),
                        float(member_edge.weight),
                    )
        return result

    def _reinforce_memories(self, node_ids: Iterable[str]) -> None:
        active = set(node_ids)
        for memory_id in self.memories_for(node_ids):
            memory = self.graph.nodes.get(memory_id)
            if memory is None or memory.kind != MEMORY:
                continue
            self.graph.activate([memory_id], [1.0])
            for key in self.graph.outgoing.get(memory_id, []):
                if key[2] != MEMBERSHIP:
                    continue
                edge = self.graph.edges[key]
                if key[1] not in active:
                    continue
                edge.weight += self.graph_rate * (1.0 - edge.weight)
                edge.eligibility = (
                    (1.0 - self.graph_rate) * edge.eligibility
                    + self.graph_rate
                )

    def _propagate_association_values(self) -> None:
        """Blend concept values along learned association weights."""
        if self.association_rate <= 0.0:
            return
        concept_ids = {
            node_id for node_id, node in self.graph.nodes.items()
            if node.kind == CONCEPT and node.role != "component_state"
        }
        if not concept_ids:
            return
        values = {
            node_id: self.graph.nodes[node_id].value
            for node_id in concept_ids
        }
        delta = {node_id: 0.0 for node_id in concept_ids}
        for edge in self.graph.edges.values():
            if edge.kind != ASSOCIATION:
                continue
            if edge.source not in values or edge.target not in values:
                continue
            weight = float(edge.weight)
            if weight <= 0.0:
                continue
            delta[edge.source] += weight * (
                values[edge.target] - values[edge.source]
            )
            delta[edge.target] += weight * (
                values[edge.source] - values[edge.target]
            )
        for node_id, change in delta.items():
            self.graph.nodes[node_id].value += (
                self.association_rate * change
            )

    def _forget_memories(self) -> None:
        memories = [
            memory for memory in self.graph.nodes.values()
            if memory.kind == MEMORY
        ]
        if memories:
            median_value = float(np.median([
                memory.value for memory in memories
            ]))
        else:
            median_value = 0.0
        for memory in memories:
            memory.trace *= self.memory_decay
            for key in self.graph.outgoing.get(memory.id, []):
                if key[2] == MEMBERSHIP:
                    self.graph.edges[key].weight *= self.memory_decay
            if (memory.trace < self.memory_min_strength
                    and memory.count <= 1
                    and (memory.value < median_value
                         or memory.value <= 0.0)):
                self.graph._remove_node(memory.id)

    def _prune_components(self) -> None:
        if len(self.components.specs) < 2:
            return
        removed = self.components.prune()
        if removed:
            self.gated_edges.clear()
            for component_id in removed:
                self._component_bound.discard(
                    component_id.rsplit(":", 1)[0]
                )

    def _note_valence(self, node_ids, pain: float) -> None:
        """Record one transition's outcome as a sample of each node's rate.

        `pain` is the entered state's valence mass.  Clean transitions are
        samples too: without them a node's read would be the frequency of
        pain among pain-adjacent events, not among its transitions.
        """
        pain = float(pain)
        self._association_global[0] += 1
        self._association_global[1] += pain
        for node_id in dict.fromkeys(node_ids):
            stats = self._association_stats.setdefault(node_id, [0, 0.0])
            stats[0] += 1
            stats[1] += pain

    def _note_valence_weighted(self, node_ids, pain: float,
                               weight: float) -> None:
        """Record a fractional outcome sample (retrospective credit).

        An eligibility trace gives the pain event a decayed weight at an
        earlier memory; this adds that fractional observation to the same
        per-node rate statistics a real transition would, so the memory's
        risk read is corrected without ever labelling a flight state as
        painful.
        """
        weight = float(weight)
        if weight <= 0.0:
            return
        pain = float(pain)
        self._association_global[0] += weight
        self._association_global[1] += weight * pain
        for node_id in dict.fromkeys(node_ids):
            stats = self._association_stats.setdefault(node_id, [0.0, 0.0])
            stats[0] += weight
            stats[1] += weight * pain

    def _advance_memory_traces(self, active, memories, created_memory,
                               pain: float) -> None:
        """Advance the episode's memory eligibility and apply the pain event.

        Two corrections happen on a real pain event, both driven only by
        that event and by the episode's own eligibility trace:

        * the values (association rates) of memories that were active
          earlier are corrected by a fractional sample weighted by
          ``(gamma*lambda)^distance``;
        * the event memory is linked back to the preceding context, so a
          similar lead-in retrieves it before the collision state itself.

        No state label is rewritten: the pain stays on the transition that
        produced it.  The context link list is capped by
        ``memory_context_limit`` purely as a resource bound.
        """
        decay = self.gamma * self.lam
        traces = self._memory_traces
        for node_id in list(traces):
            traces[node_id] *= decay
            if traces[node_id] < 1e-4:
                del traces[node_id]
        for node_id in dict.fromkeys(list(active) + list(memories)):
            traces[node_id] = traces.get(node_id, 0.0) + 1.0
        if created_memory is not None:
            traces[created_memory] = traces.get(created_memory, 0.0) + 1.0
        if pain <= 0.0:
            return
        stats = self.memory_credit_stats
        stats["events"] += 1
        current = set(active) | set(memories)
        if created_memory is not None:
            current.add(created_memory)
        for node_id, eligibility in list(traces.items()):
            if node_id in current:
                continue
            node = self.graph.nodes.get(node_id)
            if node is None or node.kind != MEMORY:
                continue
            weight = min(1.0, float(eligibility))
            self._note_valence_weighted([node_id], pain, weight)
            stats["corrected"] += 1
            stats["weight"] += weight
            stats["corrected_ids"].add(node_id)
        target = created_memory
        if target is None and memories:
            target = max(memories, key=lambda mid: traces.get(mid, 0.0))
        if target is None:
            return
        context = [
            (node_id, float(eligibility))
            for node_id, eligibility in traces.items()
            if node_id != target and eligibility >= 0.05
            and self.graph.nodes.get(node_id) is not None
            and self.graph.nodes[node_id].kind != MEMORY
        ]
        context.sort(key=lambda item: -item[1])
        chosen = context[:self.memory_context_limit]
        for member_id, eligibility in chosen:
            edge = self.graph.add_membership(target, member_id)
            edge.weight = max(edge.weight, min(1.0, eligibility))
        stats["links"] += len(chosen)

    def _association_base(self) -> float:
        """Global pain rate per transition, a 1/n cumulative mean."""
        count, hits = self._association_global
        if count <= 0:
            return 0.0
        return hits / count

    def _learn_pain_association(self, pain: float,
                                extra_nodes: Iterable[str] = ()) -> None:
        pain = float(pain)
        if pain <= 0.0:
            return
        self.graph.activate([self.pain_node], [pain])
        nodes = list(dict.fromkeys(list(extra_nodes)))
        for node_id in nodes:
            if node_id == self.pain_node:
                continue
            self.graph.observe_association(
                [node_id, self.pain_node], rate=self.valence_rate
            )

    def _pain_association(self, node_id: str) -> float:
        """Empirical pain rate of a node's transitions, shrunk to the base rate.

        The read is ``(pain mass + prior * base) / (transitions + prior)``
        over every transition that ran through the node, clean ones
        included, so it estimates ``E[pain at the entered state |
        transition through the node]`` - a rate whose uncertainty shrinks
        at 1/count.  A node with no samples reads the global base rate,
        the neutral hazard prior for unseen regions.
        """
        prior = self.association_prior
        base = self._association_base()
        stats = self._association_stats.get(node_id)
        if stats is None or stats[0] <= 0:
            return float(base)
        count, hits = stats
        return (hits + prior * base) / (count + prior)

    def pain_signal_from_weights(self, weights) -> float:
        total = 0.0
        weighted = 0.0
        node_ids = []
        for node_id, weight in weights:
            node_ids.append(node_id)
            total += float(weight)
            weighted += float(weight) * self._pain_association(node_id)
        for memory_id, strength in self.memories_for(node_ids).items():
            total += float(strength)
            weighted += float(strength) * self._pain_association(memory_id)
        if total <= 0.0:
            return 0.0
        return weighted / total

    def pain_signal_state(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        learned = self.pain_signal_from_weights(
            self._neighbor_weights(state)
        )
        row = state.reshape(1, -1)
        if self.feature_ids:
            learned = max(
                learned,
                float(self._feature_risk_batch(row)[0]),
                float(self._feature_memory_risk_batch(row)[0]),
            )
        elif self.rules.rules:
            learned = max(
                learned,
                float(self.rules.activate(row[:, :self.obs_dim])[0]),
            )
        return learned

    def _hazard_channels(self, states: np.ndarray,
                         mask_table: Optional[List] = None
                         ) -> Dict[str, np.ndarray]:
        """The individual hazard readouts the signal composes.

        Each channel reports its own estimate; `pain_signal_batch` either
        max-combines them (the historical hand-written composition) or,
        when the stacked composition is fitted, takes the learned
        non-negative weighted sum.  Keeping the channels in one place
        guarantees the fit and the consumers always see the same columns.
        """
        states = np.asarray(states, dtype=float)
        channels: Dict[str, np.ndarray] = {}
        if not self.predicate_readouts:
            indices, weights, nearest_min = self._soft_weights_batch(states)
            prototype = np.zeros(len(states), dtype=float)
            if indices is not None:
                for row in range(len(states)):
                    total = 0.0
                    weighted = 0.0
                    node_ids = []
                    for column, index in enumerate(indices[row]):
                        node_id = self._prototype_ids[int(index)]
                        node_weight = float(weights[row, column])
                        node_ids.append(node_id)
                        total += node_weight
                        weighted += node_weight * self._pain_association(
                            node_id
                        )
                    for memory_id, strength in self.memories_for(
                        node_ids
                    ).items():
                        total += float(strength)
                        weighted += float(strength) * self._pain_association(
                            memory_id
                        )
                    prototype[row] = weighted / total if total > 0.0 else 0.0
                prototype = self._shrink_to_baseline(
                    prototype, nearest_min,
                    baseline=self.prototype_pain_baseline(),
                )
                prototype = prototype * self.obs_trust
            channels["prototype"] = prototype
        if self.feature_ids:
            if mask_table is None:
                mask_table = self._feature_mask_table(
                    states[:, :self.obs_dim]
                )
            channels["feature"] = self._feature_risk_batch(
                states, mask_table
            )
            if self.event_memory_risk:
                channels["memory"] = self._feature_memory_risk_batch(
                    states, mask_table
                )
        elif self.rules.rules:
            channels["rules"] = self.rules.activate(
                states[:, :self.obs_dim]
            )
        return channels

    def pain_signal_batch(self, states: np.ndarray,
                          mask_table: Optional[List] = None) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0)
        channels = self._hazard_channels(states, mask_table)
        composition = self.hazard_composition
        if composition:
            # P2 stacked composition: the non-negative least-squares
            # weights were fitted to the precursor labels, so the read is
            # the channels' own regression estimate instead of the
            # hand-written max (whose fixed point is E[max of correlated
            # estimates], structurally above max of the truths).  The
            # monotone map then holds the composed level to the empirical
            # rate.
            read = np.zeros(len(states), dtype=float)
            for name, weight in composition.items():
                column = channels.get(name)
                if column is not None and weight > 0.0:
                    read = read + float(weight) * column
            calibration = self.hazard_calibration
            if calibration is not None:
                mids, rates = calibration
                read = np.interp(read, mids, rates)
            return read
        result = np.zeros(len(states), dtype=float)
        if "prototype" in channels:
            result = np.maximum(result, channels["prototype"])
        for name in ("feature", "memory", "rules"):
            if name in channels:
                result = np.maximum(result, channels[name])
        return result

    def _signature(self, node_id: str) -> Tuple[int, ...]:        return tuple(
            self.components.previous_sign[component_id]
            for component_id, spec in self.components.specs.items()
            if spec.base_node == node_id
        )

    def _update_gated(self, node_id: str, signature: Tuple[int, ...],
                      action: int, next_node: str, target_pain: float,
                      target_dopamine: float) -> None:
        if not self.component_gating or not signature:
            return
        targets = self.gated_edges.setdefault(
            (node_id, signature, int(action)), {}
        )
        edge = targets.setdefault(
            next_node, dict(count=0, q_pain=0.0, q_dopamine=0.0)
        )
        edge["count"] += 1
        edge["q_pain"] += self.alpha_q * (
            target_pain - edge["q_pain"]
        )
        edge["q_dopamine"] += self.alpha_q * (
            target_dopamine - edge["q_dopamine"]
        )

    # ---- action values -----------------------------------------------------
    def _neighbor_weights(self, state: np.ndarray):
        """Distance-weighted node mixture used for decisions.

        A hard nearest-node lookup leaves novel states with no outgoing edges.
        Soft neighbours let runtime decisions inherit values from nearby learned
        nodes while keeping the training update tied to the best-matching node.
        """
        if not self._prototype_ids:
            return []
        distances = np.linalg.norm(self._prototype_matrix - state, axis=1)
        count = min(self.soft_neighbors, len(distances))
        order = np.argsort(distances)[:count]
        weights = np.exp(
            -0.5 * (distances[order] / self.soft_sigma) ** 2
        )
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return []
        return [
            (self._prototype_ids[int(index)], float(weight / total))
            for index, weight in zip(order, weights)
        ]

    def _score_actions(self, node_id: str,
                       state: Optional[np.ndarray] = None,
                       sources: Optional[Iterable[str]] = None,
                       signature: Tuple[int, ...] = ()) -> Dict[int, dict]:
        node = self.graph.nodes[node_id]
        if state is None:
            state = (
                node.features if node.features is not None
                else np.zeros(len(self.scales), dtype=float)
            )
        neighbors = self._neighbor_weights(state)
        source_ids = list(dict.fromkeys(sources or ()))
        scores: Dict[int, dict] = {}
        for action in self.action_space:
            gated = (
                self.gated_edges.get((node_id, signature, action))
                if self.component_gating and signature else None
            )
            if gated:
                total = float(sum(item["count"] for item in gated.values()))
                if total > 0.0:
                    pain = sum(
                        item["count"] * item["q_pain"]
                        for item in gated.values()
                    ) / total
                    dopamine = sum(
                        item["count"] * item["q_dopamine"]
                        for item in gated.values()
                    ) / total
                    uncertainty = sum(
                        item["count"] * (1.0 / np.sqrt(item["count"] + 1.0))
                        for item in gated.values()
                    ) / total
                    scores[action] = dict(
                        pain=pain,
                        dopamine=dopamine,
                        uncertainty=uncertainty,
                        count=int(total),
                        cost=pain + self.uncertainty_weight * uncertainty,
                    )
                    continue
            support = 0.0
            pain_sum = 0.0
            dopamine_sum = 0.0
            uncertainty_sum = 0.0
            visits = 0
            for neighbor_id, weight in neighbors:
                for edge in self.graph.action_edges_from(neighbor_id, action):
                    mass = weight * edge.count
                    support += mass
                    pain_sum += mass * edge.q_pain
                    dopamine_sum += mass * edge.q_dopamine
                    uncertainty_sum += mass * (
                        1.0 / np.sqrt(edge.count + 1.0)
                    )
                    visits += edge.count
            for source in source_ids:
                for edge in self.graph.action_edges_from(source, action):
                    mass = self.component_weight * edge.count
                    support += mass
                    pain_sum += mass * edge.q_pain
                    dopamine_sum += mass * edge.q_dopamine
                    uncertainty_sum += mass * (
                        1.0 / np.sqrt(edge.count + 1.0)
                    )
                    visits += edge.count
            if support <= 0.0:
                # Unknown actions are optimistic: preserve the node's current
                # expected pain and add a fixed uncertainty penalty.
                prior_pain = node.value if node.count > 1 else 0.5
                prior_dopamine = node.dopamine if node.count > 1 else 0.0
                scores[action] = dict(
                    pain=prior_pain,
                    dopamine=prior_dopamine,
                    uncertainty=1.0,
                    count=0,
                    cost=prior_pain + self.unknown_penalty,
                )
                continue
            pain = pain_sum / support
            dopamine = dopamine_sum / support
            uncertainty = uncertainty_sum / support
            scores[action] = dict(
                pain=pain,
                dopamine=dopamine,
                uncertainty=uncertainty,
                count=int(visits),
                cost=pain + self.uncertainty_weight * uncertainty,
            )
        return scores

    def decision(self, observation) -> Tuple[str, Dict[int, dict]]:
        """Score actions for an observation without changing the runtime state."""
        state = self.encode(observation)
        node_id = self._assign(state, create=False, count=False)
        sources = self.components.active_sign_nodes(node_id)
        sources = sources + self._active_feature_roots(state)
        signature = self._signature(node_id)
        return node_id, self._score_actions(
            node_id, state, sources=sources, signature=signature
        )

    def _action_margin(self, scores) -> float:
        actions = list(self.action_space)
        if len(actions) < 2:
            return 0.0
        return float(
            scores[actions[0]]["cost"] - scores[actions[1]]["cost"]
        )

    def _action_probability(self, scores) -> float:
        """Probability of the second action from preference and confidence."""
        actions = list(self.action_space)
        if len(actions) < 2:
            return 1.0
        margin = self._action_margin(scores)
        uncertainty = 0.5 * (
            float(scores[actions[0]]["uncertainty"])
            + float(scores[actions[1]]["uncertainty"])
        )
        temperature = (
            self.action_temperature
            + self.action_uncertainty * max(0.0, uncertainty)
        )
        if temperature <= 0.0:
            return 1.0 if margin > 0.0 else 0.0
        return float(1.0 / (1.0 + np.exp(-margin / temperature)))

    def _set_action_nodes(self, margin: float) -> None:
        actions = list(self.action_space)
        if len(actions) < 2:
            return
        self.graph.activate([self.action_nodes[actions[0]]], [-margin])
        self.graph.activate([self.action_nodes[actions[1]]], [margin])

    def act(self, observation, explore: bool = True) -> int:
        state = self.encode(observation)
        node_id = self._assign(state, create=True, count=True)
        self._component_step(node_id)
        sources = self.components.active_sign_nodes(node_id)
        sources = sources + self._active_feature_roots(state)
        signature = self._signature(node_id)
        scores = self._score_actions(
            node_id, state, sources=sources, signature=signature
        )
        actions = list(self.action_space)
        margin = self._action_margin(scores)
        probability = self._action_probability(scores)
        self._set_action_nodes(margin)
        self.last_action_probs = (1.0 - probability, probability)
        if explore and len(actions) >= 2:
            action = (
                actions[1] if self.rng.rand() < probability
                else actions[0]
            )
        else:
            action = min(
                scores,
                key=lambda item: (
                    scores[item]["cost"],
                    -scores[item]["dopamine"],
                    item,
                ),
            )
        self.last_node = node_id
        self.last_action = action
        self.last_state = state
        self.last_scores = scores
        self.last_sources = [node_id] + sources
        self.last_signature = signature
        self.pending.append(action)
        return action

    # ---- learning ----------------------------------------------------------
    def _update_transition_value(self, current: str, next_node: str,
                                 target_pain: float,
                                 target_dopamine: float) -> None:
        edge_node = self.graph.edge_node(current, next_node)
        if edge_node is None:
            return
        node = self.graph.nodes[edge_node]
        node.value += self.alpha_q * (target_pain - node.value)
        node.dopamine += self.alpha_q * (
            target_dopamine - node.dopamine
        )

    def _update_action_edges(self, sources: Iterable[str], action: int,
                             next_node: str, target_pain: float,
                             target_dopamine: float) -> None:
        for source in dict.fromkeys(sources):
            edge = self.graph.add_action_edge(source, action, next_node)
            edge.prediction_error = target_pain - edge.q_pain
            edge.count += 1
            edge.q_pain += self.alpha_q * (target_pain - edge.q_pain)
            edge.q_dopamine += (
                self.alpha_q * (target_dopamine - edge.q_dopamine)
            )

    def _update(self, current: str, action: int, next_node: str,
                target_pain: float, target_dopamine: float,
                immediate_pain: float, immediate_dopamine: float,
                sources: Optional[Iterable[str]] = None) -> None:
        self._update_action_edges(
            sources or (current,), action, next_node,
            target_pain, target_dopamine,
        )
        self._update_transition_value(
            current, next_node, target_pain, target_dopamine
        )

        node = self.graph.nodes[current]
        delta_pain = target_pain - node.value
        delta_dopamine = target_dopamine - node.dopamine

        # The world's valence belongs to the state the transition entered.  Keep
        # an immediate component so a terminal endpoint is not scored as V=0.
        entered = self.graph.nodes[next_node]
        entered.immediate_pain += self.alpha_now * (
            immediate_pain - entered.immediate_pain
        )
        entered.immediate_dopamine += self.alpha_now * (
            immediate_dopamine - entered.immediate_dopamine
        )
        self._note_pain(next_node, immediate_pain)

        decay = self.gamma * self.lam
        for node_id, trace in list(self._traces.items()):
            trace[0] *= decay
            trace[1] *= decay
            if abs(trace[0]) < 1e-8 and abs(trace[1]) < 1e-8:
                del self._traces[node_id]
        trace = self._traces.setdefault(current, [0.0, 0.0])
        trace[0] += 1.0
        trace[1] += 1.0

        for node_id, (pain_trace, dopamine_trace) in self._traces.items():
            traced = self.graph.nodes[node_id]
            traced.value += self.alpha_v * delta_pain * pain_trace
            traced.dopamine += (
                self.alpha_v * delta_dopamine * dopamine_trace
            )

    def observe(self, observation, action: int, valence,
                done: bool = False, learn: bool = True) -> None:
        """Consume the real transition caused by the last `act`.

        `valence` is ``[dopamine, pain]`` from the world.  `done` only marks the
        episode boundary for bootstrapping and reset; it is not a risk label.
        """
        if self.last_node is None:
            # No state/action pair has been recorded yet.
            if done:
                self.reset()
            return
        valence = np.asarray(valence, dtype=float).reshape(-1)
        if valence.shape != (2,) or not np.isfinite(valence).all():
            raise ValueError("valence must contain finite dopamine and pain")
        if not learn:
            if done:
                self.reset()
            return

        next_state = self.encode(observation)
        next_node = self._assign(next_state, create=True, count=True)
        current_active = (
            self._active_feature_ids(self.last_state)
            if self.last_state is not None else []
        )
        current_features = [
            node_id for node_id in current_active
            if node_id in self.feature_ids
        ]
        next_active = self._active_feature_ids(next_state)
        next_features = [
            node_id for node_id in next_active
            if node_id in self.feature_ids
        ]
        self._activate_features(next_active)
        self._interact(
            self.last_node, next_node,
            float(valence[1]), float(valence[0]),
            current_features=current_features,
            next_features=next_features,
        )
        self._update_feature_pain(next_active, float(valence[1]))
        pain = float(valence[1])
        dopamine = float(valence[0])
        next_value = self.graph.nodes[next_node]
        target_pain = pain + (0.0 if done else self.gamma * next_value.value)
        target_dopamine = dopamine + (
            0.0 if done else self.gamma * next_value.dopamine
        )
        self._update(
            self.last_node, int(action), next_node,
            target_pain, target_dopamine, pain, dopamine,
            sources=self.last_sources,
        )
        self._update_gated(
            self.last_node, self.last_signature, int(action), next_node,
            target_pain, target_dopamine,
        )
        self._component_step(next_node, valence)
        self.last_node = None
        self.last_action = None
        self.last_state = None
        self.last_sources = []
        self.last_signature = ()
        if done:
            self.reset()

    def reset(self) -> None:
        """Reset episode-local state; node values and edges are retained."""
        self.pending = self._zero_pending()
        self.last_node = None
        self.last_action = None
        self.last_state = None
        self.last_scores = {}
        self.last_sources = []
        self.last_signature = ()
        self._traces.clear()
        self._memory_traces.clear()
        self._episode_nodes = set()

    def learn_transition(self, observation, action, next_observation,
                         valence, done: bool = False) -> None:
        """Learn one real transition without choosing the action first.

        This is the offline-replay entry point.  `done` only truncates
        bootstrapping and resets the episode; it is not a risk label.
        """
        action = int(action)
        if action not in self.action_space:
            raise ValueError("action is outside the action space")
        valence = np.asarray(valence, dtype=float).reshape(-1)
        if valence.shape != (2,) or not np.isfinite(valence).all():
            raise ValueError("valence must contain finite dopamine and pain")
        state = self.encode(observation)
        current = self._assign(state, create=True, count=True)
        self._component_step(current)
        current_active = self._active_feature_ids(state)
        self._activate_features(current_active)
        current_features = [
            node_id for node_id in current_active
            if node_id in self.feature_ids
        ]
        sources = (
            [current]
            + self.components.active_sign_nodes(current)
            + current_features
        )
        signature = self._signature(current)
        self.pending.append(action)
        next_state = self.encode(next_observation)
        next_node = self._assign(next_state, create=True, count=True)
        next_active = self._active_feature_ids(next_state)
        self._activate_features(next_active)
        next_features = [
            node_id for node_id in next_active
            if node_id in self.feature_ids
        ]
        self._interact(
            current, next_node, float(valence[1]), float(valence[0]),
            current_features=current_features,
            next_features=next_features,
        )
        self._update_feature_pain(next_active, float(valence[1]))
        pain = float(valence[1])
        dopamine = float(valence[0])
        next_value = self.graph.nodes[next_node]
        target_pain = pain + (0.0 if done else self.gamma * next_value.value)
        target_dopamine = dopamine + (
            0.0 if done else self.gamma * next_value.dopamine
        )
        self._update(
            current, action, next_node,
            target_pain, target_dopamine, pain, dopamine,
            sources=sources,
        )
        self._component_step(next_node, valence)
        if done:
            self.reset()

    def learn_episode(self, rows) -> None:
        """Monte-Carlo node/edge values for one real episode.

        `rows` are transition dicts with ``obs``, ``next_obs``, ``history``,
        ``signals`` and ``done`` (an episode boundary, not a risk label).
        Values are accumulated backward from the world's pain signal.
        """
        rows = list(rows)
        if not rows:
            return
        self.reset()
        # Every node's activation over this episode is computed once: the
        # per-row sweeps used to re-evaluate all masks for both the current
        # and the next state of every row.
        scale = self.scales[:self.obs_dim]
        obs_table = self._feature_mask_table(
            np.asarray(
                [np.asarray(row["obs"], dtype=float) / scale for row in rows],
                dtype=float,
            )
        )
        next_table = self._feature_mask_table(
            np.asarray(
                [
                    np.asarray(row["next_obs"], dtype=float) / scale
                    for row in rows
                ],
                dtype=float,
            )
        )
        # Label regime: point-event pain (default) trains every root on the
        # entered state's own feedback; the precursor-window path (opt-in)
        # trains each root on P(pain within its configured window).
        horizons = {self._node_horizon(fid) for fid in self.feature_ids}
        horizons.add(self.pain_horizon)
        if self.point_labels:
            # The world's pain is the feedback of the entered state; with
            # no window that single value is every root's label.  Nothing
            # is spread back onto states where the world reported normal
            # flight.
            point = [
                float(np.asarray(row["signals"], dtype=float)[1] > 0.5)
                for row in rows
            ]
            labels_by_horizon = {
                horizon: point for horizon in sorted(horizons)
            }
        else:
            labels_by_horizon = {
                horizon: future_pain_labels(rows, horizon)
                for horizon in sorted(horizons)
            }
        # The refinement-birth bar is population-relative; one O(nodes)
        # pass per episode keeps it fresh without dominating the loop.
        self._refresh_variance_bar()
        value_roots_all = self._all_value_roots()
        steps = []
        # Per-row propensities within the episode: the window's clipped
        # importance weight needs the propensities of the governing
        # actions, which are the current row's action and the ones issued
        # up to `lag` rows earlier.  Cleared at episode boundaries.
        recent_propensities: List[float] = []
        for index, row in enumerate(rows):
            action = int(row["history"][0])
            if action not in self.action_space:
                raise ValueError("action is outside the action space")
            signals = np.asarray(row["signals"], dtype=float).reshape(-1)
            if signals.shape != (2,) or not np.isfinite(signals).all():
                raise ValueError("signals must contain finite valence")
            done = bool(row["done"])
            # Per-decision clipped importance weight toward the uniform
            # target over the window's governing actions (the current
            # row's propensity plus up to `lag` earlier ones).  Rows
            # without a stored propensity read one - unweighted.
            row_propensity = float(row.get("propensity", 1.0))
            row_weight = 1.0
            if self.importance_weighting:
                row_weight = 1.0
                for propensity in (
                    recent_propensities[-self.lag:] + [row_propensity]
                ):
                    behavior = max(propensity, 1e-9)
                    row_weight *= min(
                        1.0, self.importance_target / behavior
                    )
            recent_propensities.append(row_propensity)
            state = self.encode(row["obs"])
            current = self._assign(state, create=True, count=True)
            self._component_step(current)
            current_active = [
                node_id
                for node_id, mask in zip(self._feature_all, obs_table)
                if mask is not None and bool(mask[index])
            ]
            self._activate_features(current_active)
            current_features = [
                node_id for node_id in current_active
                if node_id in self.feature_ids
            ]
            sources = (
                [current]
                + self.components.active_sign_nodes(current)
                + current_features
            )
            signature = self._signature(current)
            self.pending.append(action)
            next_state = self.encode(row["next_obs"])
            next_node = self._assign(next_state, create=True, count=True)
            next_active = [
                node_id
                for node_id, mask in zip(self._feature_all, next_table)
                if mask is not None and bool(mask[index])
            ]
            self._activate_features(next_active)
            next_features = [
                node_id for node_id in next_active
                if node_id in self.feature_ids
            ]
            self._interact(
                current, next_node,
                float(signals[1]), float(signals[0]),
                current_features=current_features,
                next_features=next_features,
                learn_memory_credit=self.memory_credit,
            )
            active_by_horizon = {}
            active_set = set(current_active)
            if self.subgraph_cells:
                # D2: each active root trains its own window's labels, and
                # every active descendant joins its root's group - a branch
                # inherits the window of the rule it belongs to.  Shared
                # sub-features are deduplicated so one sample updates a
                # node once.
                for root_id in current_features:
                    group = active_by_horizon.setdefault(
                        self._node_horizon(root_id), []
                    )
                    group.append(root_id)
                    group.extend(
                        node_id
                        for node_id in self._feature_descendants.get(
                            root_id, ()
                        )
                        if node_id in active_set
                    )
                for horizon, group in active_by_horizon.items():
                    active_by_horizon[horizon] = list(dict.fromkeys(group))
            else:
                for fid in current_active:
                    active_by_horizon.setdefault(
                        self._node_horizon(fid), []
                    ).append(fid)
            # Prediction-error loop: record what the readout predicts for
            # this state under the acting action *before* any update, then
            # compare it with the actual precursor label.  Surprises with
            # enough confidence land in the counterfactual target buffer.
            history_row = np.asarray(row["history"], dtype=float).reshape(-1)
            acting = None
            if history_row.size > self.lag:
                acting = int(round(float(history_row[self.lag])))
            if acting is not None and self.feature_ids:
                for horizon, horizon_ids in active_by_horizon.items():
                    self._record_prediction(
                        [fid for fid in horizon_ids
                         if fid in self.feature_ids],
                        acting, labels_by_horizon[horizon][index],
                        state,
                    )
            for horizon, horizon_ids in active_by_horizon.items():
                self._update_feature_pain(
                    horizon_ids, labels_by_horizon[horizon][index],
                    weight=row_weight,
                )
            # Action-conditioned statistics: the action acting on the
            # transition s_m -> s_{m+1} was issued `lag` steps earlier
            # (`history[lag]`) and is what the search controls per plan.
            if acting is not None:
                for horizon, horizon_ids in active_by_horizon.items():
                    self._update_feature_action(
                        horizon_ids,
                        acting,
                        labels_by_horizon[horizon][index],
                        weight=row_weight,
                    )
            entered = self.graph.nodes[next_node]
            entered.immediate_pain += self.alpha_now * (
                signals[1] - entered.immediate_pain
            )
            entered.immediate_dopamine += self.alpha_now * (
                signals[0] - entered.immediate_dopamine
            )
            self._note_pain(next_node, float(signals[1]))
            self._component_step(next_node, signals)
            value_roots = [
                fid for fid in current_active if fid in value_roots_all
            ]
            if self.subgraph_cells:
                for root_id in value_roots:
                    value_roots.extend(
                        node_id
                        for node_id in self._feature_descendants.get(
                            root_id, ()
                        )
                        if node_id in active_set
                    )
                value_roots = list(dict.fromkeys(value_roots))
            steps.append((
                current, action, next_node, float(signals[1]),
                float(signals[0]), done, sources, signature,
                acting, value_roots, row_weight,
            ))
            if done:
                self.reset()
                recent_propensities.clear()

        running_pain = 0.0
        running_dopamine = 0.0
        # Short-horizon value attribution bootstraps from the pre-update
        # dopamine value of the state `value_horizon` steps ahead, so the
        # k-step target keeps the return scale of the MC path while
        # crediting only the actions inside the horizon.  The snapshot
        # reads the dopamine channel - the same quantity the cells
        # estimate - not the pain channel.
        dopamine_snapshot = (
            {key: node.dopamine + node.immediate_dopamine
             for key, node in self.graph.nodes.items()}
            if self.value_horizon else None
        )
        for position in range(len(steps) - 1, -1, -1):
            (current, action, next_node, pain, dopamine,
             terminal, sources, signature, acting,
             value_roots, row_weight) = steps[position]
            return_pain = pain + (
                0.0 if terminal else self.gamma * running_pain
            )
            return_dopamine = dopamine + (
                0.0 if terminal else self.gamma * running_dopamine
            )
            node = self.graph.nodes[current]
            node.value += self.alpha_mc * (
                return_pain - node.immediate_pain - node.value
            )
            node.dopamine += self.alpha_mc * (
                return_dopamine - node.immediate_dopamine - node.dopamine
            )
            # The transition's outcome is governed by the action issued
            # `lag` steps earlier (`acting`), not by the one just issued -
            # keying the Q table by the just-issued action would attribute
            # the outcome of one action to another.
            self._update_action_edges(
                sources,
                acting if acting is not None else action,
                next_node, return_pain, return_dopamine,
            )
            self._update_transition_value(
                current, next_node, return_pain, return_dopamine
            )
            self._update_gated(
                current, signature,
                acting if acting is not None else action,
                next_node, return_pain, return_dopamine,
            )
            if acting is not None and value_roots:
                # Sequence-level value: a (feature, acting action) cell
                # learns the discounted dopamine return of the paths that
                # actually followed it, so "aligned for a long survival"
                # becomes readable where the pain hazard only says "local".
                target = (
                    self._k_step_value_target(
                        steps, position, dopamine_snapshot
                    )
                    if dopamine_snapshot is not None else return_dopamine
                )
                self._update_feature_value(
                    value_roots, acting, target, rows[0].get("episode"),
                    weight=row_weight,
                )
            if terminal:
                entered = self.graph.nodes[next_node]
                entered.value += self.alpha_mc * (
                    pain - entered.immediate_pain - entered.value
                )
            running_pain = return_pain
            running_dopamine = return_dopamine
        self._propagate_association_values()
        self._forget_memories()
        self._prune_components()
        self.reset()

    def induce_rules(self, train_states, train_pain,
                     test_states=None, test_pain=None):
        """Induce compositional precursor rules over encoded states.

        Only the observation part is used.  With precursor labels the target
        is ``pain within lag + 1 steps`` of the state itself, so a rule
        describes a situation that *precedes* death while an action can
        still change the outcome.  Lag/action coordinates remain available
        to the node graph.
        """
        train_states = np.asarray(train_states, dtype=float)
        if train_states.ndim != 2 or train_states.shape[1] < self.obs_dim:
            raise ValueError(
                "states must have at least obs_dim columns"
            )
        test_obs = None
        if test_states is not None:
            test_states = np.asarray(test_states, dtype=float)
            if test_states.ndim != 2 or test_states.shape[1] < self.obs_dim:
                raise ValueError(
                    "test states must have at least obs_dim columns"
                )
            test_obs = test_states[:, :self.obs_dim]
        return self.rules.induce(
            train_states[:, :self.obs_dim], train_pain,
            test_obs, test_pain,
        )

    def _materialize_feature(self, feature, role: str) -> str:
        spec = feature_state(feature)
        key = json.dumps(spec, sort_keys=True)
        existing = self._feature_keys.get(key)
        if existing is not None:
            return existing
        self._feature_counter += 1
        node_id = self.graph.add_node(
            CONCEPT, "feature_%d" % self._feature_counter,
            role=role, predicate=spec,
        )
        node = self.graph.nodes[node_id]
        node.beta = self.graph_rate
        self._feature_all.append(node_id)
        self._feature_objects[node_id] = feature
        self._feature_keys[key] = node_id
        children = getattr(feature, "children", None)
        if children:
            child_ids = []
            for child in children:
                child_role = (
                    "composite" if getattr(child, "children", None)
                    else "primitive"
                )
                child_id = self._materialize_feature(child, child_role)
                self.graph.add_derivation(node_id, child_id)
                child_ids.append(child_id)
            self._feature_children[node_id] = child_ids
        return node_id

    def install_feature_rules(self, rules=None,
                              horizon: Optional[int] = None,
                              channel: str = "pain") -> List[str]:
        """Materialize induced rules as a shared feature graph.

        Primitive and composite nodes are shared by every rule that uses the
        same sub-feature; derivation edges point from a composite to its
        children.  Root nodes keep the learned P(pain | feature) in ``value``
        and, when `horizon` is given, the precursor window their value and
        action cells are trained with.

        `channel` selects which readout trains on the installed roots:
        "pain" roots carry the precursor statistics, "value" roots join the
        sequence-level value readout.  Sub-feature nodes are shared between
        channels - only root bookkeeping differs.
        """
        if channel not in ("pain", "value"):
            raise ValueError("unknown feature channel %r" % channel)
        rules = self.rules.rules if rules is None else list(rules)
        for rule in rules:
            node_id = self._materialize_feature(rule.feature, "feature")
            self._feature_descendants[node_id] = self._subtree_ids(node_id)
            node = self.graph.nodes[node_id]
            if channel == "value":
                if node_id not in self.value_feature_ids:
                    self.value_feature_ids.append(node_id)
                continue
            # Initialize from train counts only: the held-out split that
            # selected the rule must not also seed its predictor value.
            prior = (float(rule.hits) + 1.0) / (float(rule.support) + 2.0)
            node.value = max(float(node.value), prior)
            node.count = max(node.count, int(rule.support))
            node.pain_horizon = (
                int(horizon) if horizon is not None else None
            )
            if node_id not in self.feature_ids:
                self.feature_ids.append(node_id)
        if channel == "value":
            return list(self.value_feature_ids)
        return list(self.feature_ids)

    def uninstall_feature_rules(self, node_ids) -> None:
        """Remove root registrations added by a rejected structure stage.

        The materialized predicate nodes and any shared sub-features stay
        in place (another live root may use them); only the root's
        participation in the readouts and training groups is withdrawn.
        The materialization key is kept, so re-installing the same rule
        later returns the existing node instead of duplicating it.
        """
        for node_id in list(node_ids):
            if node_id in self.feature_ids:
                self.feature_ids.remove(node_id)
            if node_id in self._feature_all:
                self._feature_all.remove(node_id)
            self._feature_descendants.pop(node_id, None)
            self._feature_objects.pop(node_id, None)
            self._feature_children.pop(node_id, None)

    def _node_horizon(self, node_id: str) -> int:
        if self.point_labels:
            # Point-event mode has no window: every root predicts the pain
            # of the state the transition entered.
            return 1
        node = self.graph.nodes.get(node_id)
        if node is not None and node.pain_horizon is not None:
            return int(node.pain_horizon)
        return self.pain_horizon

    def _active_feature_ids(self, state) -> List[str]:
        if not self._feature_all:
            return []
        obs = np.asarray(state, dtype=float).reshape(-1)[:self.obs_dim]
        row = obs.reshape(1, -1)
        return [
            node_id for node_id in self._feature_all
            if node_id in self._feature_objects
            and bool(self._feature_objects[node_id].mask(row)[0])
        ]

    def _table_entry(self, mask_table, node_id, feature, obs):
        """A node's mask from a shared table, or computed on the spot."""
        if mask_table is None:
            return feature.mask(obs)
        return mask_table[self._feature_all.index(node_id)]

    def _feature_mask_table(self, obs: np.ndarray) -> List:
        """Every node's activation over a batch of raw observations.

        The masks are pure threshold comparisons, so one vectorized pass
        over the batch yields exactly the booleans the per-row sweep
        produced - computed once instead of once per row.
        """
        return [
            None if node_id not in self._feature_objects
            else self._feature_objects[node_id].mask(obs)
            for node_id in self._feature_all
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
        root_memories = {}
        for node_id in self.feature_ids:
            if mask_table is None:
                feature = self._feature_objects.get(node_id)
                if feature is None:
                    continue
                entry = feature.mask(obs)
            else:
                entry = mask_table[self._feature_all.index(node_id)]
            if entry is not None and entry.any():
                root_memories[node_id] = self.memories_for([node_id])
        for row in range(len(states)):
            feature_ids = [
                node_id
                for node_id, mask in zip(self._feature_all, table)
                if mask is not None and bool(mask[row])
                and node_id in self.feature_ids
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
                weighted += float(strength) * self._pain_association(
                    memory_id
                )
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

    def _soft_weights_batch(self, states: np.ndarray):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.scales):
            raise ValueError("states must be a matrix with the runtime width")
        if not self._prototype_ids:
            return None, None, None
        # Squared distances via matrix multiplication; this bounds memory to
        # (states x prototypes) instead of materialising states x prototypes x dim.
        prototypes = self._prototype_matrix
        state_norm = np.einsum("ij,ij->i", states, states)[:, None]
        proto_norm = np.einsum("ij,ij->i", prototypes, prototypes)[None, :]
        distances = (
            state_norm + proto_norm - 2.0 * (states @ prototypes.T)
        )
        np.maximum(distances, 0.0, out=distances)
        count = min(self.soft_neighbors, distances.shape[1])
        indices = np.argpartition(distances, count - 1, axis=1)[:, :count]
        nearest = np.take_along_axis(distances, indices, axis=1)
        nearest_min = nearest.min(axis=1)
        weights = np.exp(
            -0.5 * nearest / (self.soft_sigma * self.soft_sigma)
        )
        total = weights.sum(axis=1, keepdims=True)
        weights = weights / np.where(total > 0.0, total, 1.0)
        return indices, weights, nearest_min

    def _memory_values_batch(self, indices, weights, attribute: str):
        count = len(indices)
        sums = np.zeros(count, dtype=float)
        mass = np.zeros(count, dtype=float)
        for row in range(count):
            for column, index in enumerate(indices[row]):
                node_id = self._prototype_ids[int(index)]
                for memory_id, strength in self.memories_for(
                    [node_id]
                ).items():
                    memory = self.graph.nodes.get(memory_id)
                    if memory is None or memory.kind != MEMORY:
                        continue
                    weight = float(weights[row, column]) * float(strength)
                    if weight <= 0.0:
                        continue
                    sums[row] += weight * getattr(memory, attribute)
                    mass[row] += weight
        return sums, mass

    def valence_state_batch(self, states: np.ndarray):
        """Soft-neighbour pain/dopamine values for encoded states."""
        states = np.asarray(states, dtype=float)
        if len(states) == 0:
            return np.zeros(0), np.zeros(0)
        indices, weights, nearest_min = self._soft_weights_batch(states)
        if indices is None:
            length = len(np.asarray(states))
            return np.zeros(length), np.zeros(length)
        nodes = [
            self.graph.nodes[self._prototype_ids[int(index)]]
            for index in indices.ravel()
        ]
        pain_values = np.asarray([
            node.value + node.immediate_pain for node in nodes
        ]).reshape(indices.shape)
        dopamine_values = np.asarray([
            node.dopamine + node.immediate_dopamine for node in nodes
        ]).reshape(indices.shape)
        pain = (weights * pain_values).sum(axis=1)
        dopamine = (weights * dopamine_values).sum(axis=1)
        for attribute, values in (
            ("value", pain), ("dopamine", dopamine)
        ):
            memory_sums, memory_mass = self._memory_values_batch(
                indices, weights, attribute
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
