"""Command-line argument parsing, validation, multi-worker dispatch, and reporting."""
from __future__ import annotations

import argparse
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
import json
import numpy as np

from eval_protocol import (
    cluster_summary,
    group_summaries,
    parse_seed_spec,
)
from latent_world import SEEDS


def main():
    """Parse CLI flags, validate options, run parallel seed evaluation, and print reports."""
    from substrate_nodes import DEFAULT_JOBS, FREE_THREADED, LAG, run_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--cap", type=int, default=6000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--birth-threshold", type=float, default=0.45)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lag", type=int, default=LAG)
    parser.add_argument("--components", default="",
                        help="comma-separated runtime component kinds")
    parser.add_argument("--component-min-count", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument("--component-weight", type=float, default=0.5)
    parser.add_argument("--component-gating", action="store_true")
    parser.add_argument("--memory-weight", type=float, default=0.1)
    parser.add_argument("--association-rate", type=float, default=0.0)
    parser.add_argument("--no-features", action="store_true",
                        help="disable the compositional feature layer")
    parser.add_argument("--instant-labels", action="store_true",
                        help="deprecated alias: point-event pain is now the "
                             "default, so this flag is accepted but has no "
                             "effect")
    parser.add_argument("--precursor-labels", action="store_true",
                        help="opt into the historical precursor window "
                             "(P(pain within k), k = --pain-horizon); it "
                             "assigns pain to normal-flight states before a "
                             "collision, which is not world feedback")
    parser.add_argument("--pain-horizon", type=int, default=0,
                        help="precursor label window in entered states "
                             "(only with --precursor-labels); "
                             "0 uses the default lag + 1")
    parser.add_argument("--voi-fraction", type=float, default=0.0,
                        help="fraction of collection steps guided by the "
                             "counterfactual budget (two-stage collect)")
    parser.add_argument("--voi-target", type=int, default=3,
                        help="how many samples a (feature, action) cell "
                             "should have before the guide stops testing it")
    parser.add_argument("--voi-threat-min", type=float, default=0.25,
                        help="minimum calibrated threat rate a state must "
                             "read before the counterfactual guide tests "
                             "an action there (decision-critical targeting)")
    parser.add_argument("--feature-value-weight", type=float, default=0.02,
                        help="how much sequence-level value trades against "
                             "accumulated hazard in the search objective")
    parser.add_argument("--risk-weight", type=float, default=1.0,
                        help="scale on the accumulated hazard column "
                             "(1.0 = current behaviour)")
    parser.add_argument("--risk-decay", type=float, default=1.0,
                        help="per-step decay on the hazard column: later "
                             "predicted steps count less (1.0 = off)")
    parser.add_argument("--risk-error-weight", action="store_true",
                        help="down-weight the hazard read at steps where "
                             "the world model's measured rollout error is "
                             "large (default off)")
    parser.add_argument("--memory-feature-backfill", action="store_true",
                        help="after induction, link event memories to the "
                             "feature roots active at their members (the "
                             "memory risk channel is otherwise empty under "
                             "epochs=1)")
    parser.add_argument("--eval-seeds", default="",
                        help="held-out seed spec, e.g. 120-139; empty keeps "
                             "the legacy self-evaluation")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="episodes per evaluation seed")
    parser.add_argument("--step-stride", type=int, default=2,
                        help="query every Nth predicted state in search")
    parser.add_argument("--stride-includes-lag", action="store_true",
                        help="also query step == lag, the first predicted "
                             "state the plan action can affect")
    parser.add_argument("--decision", choices=("search", "cellq", "stage", "chained"),
                        default="search",
                        help="search = full MPC planning; cellq = read the "
                             "learned (feature, action) cells directly; "
                             "stage = score block-wise constant action "
                             "schedules; chained = chain short 2-3 frame "
                             "active-pruning searches")
    parser.add_argument("--chain-chunk-size", type=int, default=2,
                        help="frames per hop in --decision chained")
    parser.add_argument("--chain-hops", type=int, default=3,
                        help="number of chained shallow hops in --decision chained")
    parser.add_argument("--chain-beam-width", type=int, default=2,
                        help="frontier states retained per hop in --decision chained")
    parser.add_argument("--chain-demand-weight", type=float, default=0.10,
                        help="weight on learned controllable feature demand "
                             "(heuristic pruning ONLY, separate from final objective)")
    parser.add_argument("--chain-demand-online-every", type=int, default=0,
                        help="online EMA refit cadence (episodes) for "
                             "ControllableFeatureDemand safe envelope "
                             "(0 = frozen offline demand, recommended 4)")
    parser.add_argument("--chain-demand-ema", type=float, default=0.30,
                        help="EMA smoothing factor in (0, 1] for online "
                             "ControllableFeatureDemand updates")
    parser.add_argument("--chain-demand-online-frames", type=int, default=12000,
                        help="sliding frame window for online demand updates")
    parser.add_argument("--chain-demand-min-frames", type=int, default=600,
                        help="minimum online frames before first demand update")
    parser.add_argument("--chain-demand-rel-pairs", type=int, default=0,
                        help="max self-supervised relational difference pairs "
                             "(x_i - x_j) added to ControllableFeatureDemand "
                             "(0 = raw dims only, recommended 2)")
    parser.add_argument("--chain-demand-min-terminal", type=int, default=0,
                        help="minimum terminal episodes in sliding window before "
                             "online demand EMA updates (0 = no gate, recommended 12)")
    parser.add_argument("--chain-adaptive", action="store_true",
                        help="enable dual-confidence decaying-threshold adaptive "
                             "segmented deepening under --decision chained "
                             "(combine with --regime-online-every 2 "
                             "--regime-abstain-lcb 0.45 for causal outer gating)")
    parser.add_argument("--chain-max-hops", type=int, default=5,
                        help="maximum segmented hops for --chain-adaptive deepening")
    parser.add_argument("--chain-wide-beam-width", type=int, default=6,
                        help="beam width on escalated tiers under --chain-adaptive")
    parser.add_argument("--chain-conf-threshold", type=float, default=0.72,
                        help="Tier-0 monotonic plan confidence release threshold "
                             "under --chain-adaptive")
    parser.add_argument("--chain-conf-decay", type=float, default=0.80,
                        help="per-tier threshold decay factor under --chain-adaptive")
    parser.add_argument("--chain-meta-lcb", type=float, default=0.65,
                        help="minimum online stage Wilson-LCB before unlocking "
                             "clean multi-step deepening (0 = unconditional)")
    parser.add_argument("--chain-conf-quantile", type=float, default=0.0,
                        help="optional rolling quantile (>0) to self-calibrate "
                             "Tier-0 confidence threshold under --chain-adaptive")
    parser.add_argument("--stage-blocks", type=int, default=4,
                        help="number of action blocks a stage-level "
                             "(--decision stage) schedule holds")
    parser.add_argument("--stage-block-frames", type=int, default=4,
                        help="frames each stage-level action block lasts")
    parser.add_argument("--beam-width", type=int, default=0,
                        help="beam width for the searched decision: 0 keeps "
                             "the exhaustive 2**horizon enumeration (bitwise "
                             "unchanged); positive values switch to beam "
                             "search, which keeps per-frame action timing at "
                             "a bounded candidate count")
    parser.add_argument("--eval-spacing", type=float, default=0.0,
                        help="pipe spacing of the evaluation worlds (0 = the "
                             "world default).  Raising it stretches the same "
                             "task over more frames, which is how the "
                             "time-scale invariance of a decision policy is "
                             "measured; training worlds are unaffected")
    parser.add_argument("--milestone-weight", type=float, default=0.0,
                        help="weight of the milestone shaping term: the "
                             "rollout endpoint is pulled toward the boundary "
                             "states the agent's own stream started "
                             "milestones in.  0 (default) leaves the "
                             "objective bitwise unchanged")
    parser.add_argument("--milestone-mode",
                        choices=("off", "goal", "rate", "plan", "track"),
                        default="off",
                        help="how the milestone term attaches: 'off' is "
                             "inert; 'goal' pulls a rollout endpoint toward "
                             "the modal boundary states of the agent's own "
                             "stream; 'rate' honours, per milestone identity, "
                             "the duration-weighted action rate the behavior "
                             "policy realized there - an epoch-level action "
                             "commitment that carries no position and no "
                             "time scale; 'plan' charges each candidate the "
                             "expected feedback deficit of the stage its own "
                             "endpoint lands in (the two-stage walk of the "
                             "learned transition table); the frame horizon "
                             "is unchanged")
    parser.add_argument("--milestone-stage",
                        choices=("geometry", "strength", "signature"),
                        default="strength",
                        help="how stages are cut: 'geometry' uses the "
                             "observation stream's boundary statistic; "
                             "'strength' uses the learned graph (a node's "
                             "strength rises where the action-conditioned "
                             "future changes) and keeps the frame identity; "
                             "'signature' additionally names each stage by "
                             "the pooled cluster of its future")
    parser.add_argument("--milestone-boundary-k", type=float, default=3.0,
                        help="stage-boundary threshold in pooled score "
                             "quantiles; the same statistical rule (three "
                             "quantiles above the 90th) the geometric "
                             "segmenter uses")
    parser.add_argument("--hazard-readout",
                        choices=("calibrated", "raw"), default="calibrated",
                        help="what the planner reads as hazard: 'calibrated' "
                             "is the isotonic rate (a level; on a rare-event "
                             "target it collapses most of the score range "
                             "into one plateau), 'raw' is the weighted "
                             "channel sum (a ranking, uncalibrated)")
    parser.add_argument("--value-readout",
                        choices=("cells", "association"), default="cells",
                        help="what the planner reads as endpoint value: "
                             "'cells' is the per-feature cell (empty at "
                             "epochs=1); 'association' is the counting "
                             "readout - the weighted realised quality of the "
                             "nodes the stream met after this one")
    parser.add_argument("--veto-fraction", type=float, default=0.0,
                        help="exclude the worst fraction of candidate "
                             "endpoints by the hazard readout within each "
                             "decision (0 = off); the cutoff is a "
                             "within-decision quantile, so only the ranking "
                             "is used")
    parser.add_argument("--value-horizon", type=int, default=0,
                        help="k-step bootstrap target for the sequence-value "
                             "cells; 0 keeps the full-episode MC return")
    parser.add_argument("--ood-trust-scale", type=float, default=0.0,
                        help="shrink prototype readouts toward the population "
                             "baseline beyond this distance (0 = off; the "
                             "birth threshold is a natural setting)")
    parser.add_argument("--obs-trust", type=float, default=1.0,
                        help="weight of the observation-anchored prototype/"
                             "memory channel against induced feature channels")
    parser.add_argument("--feature-action-prior", type=float, default=2.0,
                        help="pseudo-counts shrinking pain cells toward the "
                             "feature scalar; lower = more trust in verified "
                             "action-conditioned cells")
    parser.add_argument("--feature-value-prior", type=float, default=4.0,
                        help="pseudo-counts shrinking value cells toward the "
                             "baseline; lower = more trust in verified "
                             "success cells")
    parser.add_argument("--no-value-features", action="store_true",
                        help="keep the value readout on the pain-coverage "
                             "roots instead of adding a dedicated "
                             "high-return bank")
    parser.add_argument("--value-quantile", type=float, default=0.8,
                        help="return-distribution quantile that defines the "
                             "high-return labels for value-channel "
                             "induction")
    parser.add_argument("--subgraph-cells", action="store_true",
                        help="train and read feature statistics on rule "
                             "branches (most specific active member wins); "
                             "measured inert on survival, off by default")
    parser.add_argument("--predicate-readouts", action="store_true",
                        help="drop the prototype channel from the risk "
                             "readout; prototypes stay as event memory")
    parser.add_argument("--no-event-memory-risk", action="store_true",
                        help="drop the feature-keyed event-memory readout "
                             "from the risk max-combine")
    parser.add_argument("--contrast-objective", action="store_true",
                        help="F1: per-step cost is the action contrast and "
                             "the endpoint cost is the value term alone, "
                             "so genuinely hazardous bands no longer "
                             "penalize the plans that must cross them")
    parser.add_argument("--value-trust-samples", type=float, default=150.0,
                        help="D4: mean value-cell sample count at which the "
                             "value channel earns half weight; 0 disables "
                             "the trust scaling")
    parser.add_argument("--value-dispersion-labels", action="store_true",
                        help="induce the value bank on both return extremes "
                             "(top and bottom quantiles) so its predicates "
                             "tile the decision-critical regions")
    parser.add_argument("--control-risk-weight", type=float, default=0.0,
                        help="threat grading: weight of the controllable "
                             "excess (the avoidable share a plan takes by "
                             "its own action) on top of the real threat; "
                             "0 keeps controllable risk free to take")
    parser.add_argument("--value-bank-only", action="store_true",
                        help="read the value channel from the dedicated "
                             "value-bank roots only, without the risk "
                             "bank's drifting-continuation cells")
    parser.add_argument("--curiosity-weight", type=float, default=0.5,
                        help="decision-side curiosity: the maximum absolute "
                             "share the potential-difference term may take "
                             "in the objective (phi in (0,1] so the term is "
                             "bounded by +/-w); it shapes training-time "
                             "execution only - evaluation always runs with "
                             "w=0. 0 disables it")
    parser.add_argument("--memory-credit", action="store_true",
                        help="on a pain event, correct earlier memories' "
                             "risk rates by an eligibility trace and link "
                             "the event memory back to the preceding "
                             "context; the pain itself stays a point event")
    parser.add_argument("--rule-min-coverage", type=float, default=0.0,
                        help="induction gate: minimum fraction of the "
                             "validation states a candidate rule must "
                             "activate on (0 = off)")
    parser.add_argument("--rule-min-primitive-coverage", type=float,
                        default=0.0,
                        help="induction gate: every conjunct/branch "
                             "primitive must activate on at least this "
                             "fraction of the validation states "
                             "(0 = off)")
    parser.add_argument("--rule-validation-seed", type=int, default=0,
                        help="cross-world activation check: collect this "
                             "many frames from an independent world seed "
                             "and use its states (no labels) for the "
                             "coverage gates; 0 = use the training split")
    parser.add_argument("--rule-validation-frames", type=int, default=600,
                        help="frames collected from --rule-validation-seed")
    parser.add_argument("--pairwise-features", dest="pairwise_features",
                        action="store_true",
                        help="add generic pairwise coordinate differences "
                             "to the induction grammar (six-seed A/B null; "
                             "opt-in)")
    parser.add_argument("--no-pairwise-features", dest="pairwise_features",
                        action="store_false",
                        help="deprecated alias; pairwise differences are "
                             "off by default")
    parser.add_argument("--feature-backfit", action="store_true",
                        help="replay the stream with features only after "
                             "induction: feature cells, value cells and the "
                             "prediction ledger learn; prototypes, edges and "
                             "memories keep their single-pass statistics")
    parser.set_defaults(pairwise_features=False)
    parser.add_argument("--branch-transitions", action="store_true",
                        help="add decayed recent branch sources to "
                             "action-edge evidence (structure-only: the "
                             "planner does not read action-edge Q yet)")
    parser.add_argument("--action-edge-weight", type=float, default=0.0,
                        help="weight of the normalized action-edge Q term "
                             "in the planner cost (0 = the Q read is off)")
    parser.add_argument("--action-edge-trust", type=float, default=50.0,
                        help="evidence pseudo-count for trusting the "
                             "action-edge Q read")
    parser.add_argument("--death-rules", action="store_true",
                        help="train a 0/1 death-condition rule set and use "
                             "it as a hard planner constraint; off by "
                             "default")
    parser.add_argument("--death-rule-compose", action="store_true",
                        help="allow nested AND/OR/XOR/NOT composition in "
                             "the death-rule learner")
    parser.add_argument("--death-rule-thresholds", type=int, default=256,
                        help="candidate threshold budget per coordinate")
    parser.add_argument("--death-rule-max-rules", type=int, default=24,
                        help="maximum death/safe rules per class")
    parser.add_argument("--death-rule-max-literals", type=int, default=5,
                        help="maximum literals in one conjunctive rule")
    parser.add_argument("--death-rule-margin", type=float, default=0.0,
                        help="boundary margin as a fraction of each "
                             "coordinate's observed range; states inside "
                             "the margin are UNKNOWN, not safe")
    parser.add_argument("--death-rule-envelope", action="store_true",
                        help="treat states inside the boundary margin as "
                             "death for the planner's hard constraint "
                             "(model-error-aware envelope); off by default")
    parser.add_argument("--readout-propagation", type=float, default=0.0,
                        help="relax the readout's soft-neighbour seed "
                             "weights over learned graph edges before "
                             "reading values/rates, so the readout reflects "
                             "the topological neighbourhood instead of the "
                             "Euclidean one (0 = off, the geometric readout)")
    parser.add_argument("--readout-propagation-steps", type=int, default=1,
                        help="relaxation steps for --readout-propagation")
    parser.add_argument("--readout-propagation-edges",
                        choices=("transition", "association", "both"),
                        default="transition",
                        help="which learned edges carry the readout "
                             "relaxation")
    parser.add_argument("--iterative-induction", action="store_true",
                        help="F5: re-induce gap banks every epoch, with "
                             "the previous generation's missed positives "
                             "as the next generation's targets")
    parser.add_argument("--no-compose-readout", dest="compose_readout",
                        action="store_false",
                        help="keep the hand-written max-combination in "
                             "pain_signal_batch instead of the stacked "
                             "non-negative composition fitted to the "
                             "precursor labels")
    parser.add_argument("--importance-weighting", action="store_true",
                        help="weight each window label by the clipped "
                             "product of uniform-target propensity "
                             "ratios over the window's governing "
                             "actions, rebalancing the cells' "
                             "state-action mix away from the behavior "
                             "policy's occupancy")
    parser.add_argument("--no-learn-objective", dest="learn_objective",
                        action="store_false",
                        help="keep the hand-set value weight in the "
                             "planning objective instead of selecting "
                             "it from the planner's executed decisions "
                             "by rank agreement with realized survival")
    parser.add_argument("--report", default="",
                        help="write per-episode rows and summaries as JSON")
    parser.add_argument("--dagger-rounds", type=int, default=0,
                        help="closed-loop rounds: record the planner's own "
                             "executed trajectories and retrain")
    parser.add_argument("--dagger-episodes", type=int, default=2,
                        help="planner episodes recorded per DAgger round")
    parser.add_argument("--dagger-cap", type=int, default=1200,
                        help="frame cap per recorded planner episode")
    parser.add_argument("--dagger-offset", type=int, default=50000,
                        help="episode-id offset for recorded planner data")
    parser.add_argument("--dagger-horizon", type=int, default=0,
                        help="search horizon while recording planner data; "
                             "0 uses --horizon")
    parser.add_argument("--dagger-frames", type=int, default=0,
                        help="on-policy frame budget per DAgger round: stop "
                             "recording once this many executed frames were "
                             "collected (0 = use --dagger-episodes as-is)")
    parser.add_argument("--stage-memory", action="store_true",
                        help="build the self-supervised stage experience from "
                             "the realized stream (read-only; planner unwired)")
    parser.add_argument("--online-feature-learning", action="store_true",
                        help="update the default-off online feature candidate "
                             "bank from each real transition; it does not "
                             "change planner decisions")
    parser.add_argument("--stage-memory-quantile", type=float, default=0.98,
                        help="upper-tail quantile for the generic key-signal "
                             "detector's channels")
    parser.add_argument("--stage-feature-source",
                        choices=("branch", "prototype", "hybrid"),
                        default="branch",
                        help="read-only feature source for stage demand "
                             "samples: induced roots, nearest prototype, or "
                             "both namespaced together")
    parser.add_argument("--regime-value-weight", type=float, default=0.0,
                        help="opt-in weight on calibrated stage-regime "
                             "continuation risk at search endpoints (0 = off)")
    parser.add_argument("--regime-n-regimes", type=int, default=16,
                        help="number of self-supervised state regimes when "
                             "--regime-value-weight > 0")
    parser.add_argument("--regime-partition",
                        choices=("survival_tree", "kmeans"),
                        default="survival_tree",
                        help="state regime partitioning method when "
                             "--regime-value-weight > 0")
    parser.add_argument("--regime-k-stages", type=int, default=8,
                        help="continuation stage horizon when "
                             "--regime-value-weight > 0")
    parser.add_argument("--regime-mode",
                        choices=("lcb", "pooled", "optimal"),
                        default="lcb",
                        help="continuation Bellman mode when "
                             "--regime-value-weight > 0")
    parser.add_argument("--regime-lcb-z", type=float, default=1.0,
                        help="LCB confidence penalty z when "
                             "--regime-mode lcb")
    parser.add_argument("--regime-value-key",
                        choices=("cum_survival", "survival"),
                        default="cum_survival",
                        help="continuation value statistic when "
                             "--regime-value-weight > 0")
    parser.add_argument("--regime-soft-top-k", type=int, default=4,
                        help="number of nearest supported regimes for "
                             "kernel interpolation (1 = hard lookup)")
    parser.add_argument("--regime-kernel-tau", type=float, default=0.7,
                        help="bandwidth multiplier on training inter-regime "
                             "scale for soft kernel interpolation")
    parser.add_argument("--regime-min-coverage", type=float, default=0.80,
                        help="minimum fraction of supported candidate "
                             "endpoints required to apply regime continuation")
    parser.add_argument("--regime-ood-policy",
                        choices=("baseline", "neutral_mean", "conservative"),
                        default="neutral_mean",
                        help="OOD fallback policy for active regime decisions")
    parser.add_argument("--regime-online-every", type=int, default=0,
                        help="fixed-partition online regime transition/LCB "
                             "refit cadence in evaluation episodes (0 = off, "
                             "frozen evaluation; requires --regime-value-weight > 0)")
    parser.add_argument("--regime-online-frames", type=int, default=12000,
                        help="sliding tail frame budget kept for online "
                             "regime transition updates alongside bootstrap "
                             "rows (0 = unbounded)")
    parser.add_argument("--regime-failure-weight", type=float, default=0.0,
                        help="opt-in weight on trunk-indexed pure-online "
                             "point-event failure memory (0 = off; requires "
                             "--regime-value-weight > 0 and --regime-online-every > 0)")
    parser.add_argument("--regime-failure-sigma", type=float, default=0.25,
                        help="local Gaussian bandwidth multiplier for online "
                             "point-event failure memory")
    parser.add_argument("--regime-failure-prior", type=float, default=4.0,
                        help="evidence strength prior for online point-event "
                             "failure memory")
    parser.add_argument("--regime-abstain-lcb", type=float, default=0.0,
                        help="opt-in Wilson-LCB threshold for causal post-episode "
                             "self-monitoring of stage continuation calibration "
                             "(0 = off; e.g. 0.45 dampens inverted stage models)")
    parser.add_argument("--world-variables", action="store_true",
                        help="opt-in task-agnostic DynamicsClassNode latent "
                             "world-variable extraction and variable-scoped "
                             "class-node memory instantiation (default off)")
    parser.add_argument("--world-var-reset-episode", action="store_true",
                        help="reset fast world-variable slots at the start of "
                             "each evaluation episode (requires --world-variables)")
    parser.add_argument("--world-var-sigma", type=float, default=0.75,
                        help="similarity bandwidth in normalized variable space "
                             "for variable-scoped class-node instantiation")
    parser.add_argument("--world-var-abstain-lcb", type=float, default=0.0,
                        help="opt-in Wilson-LCB threshold for causal 1-step "
                             "out-of-sample self-monitoring of world-variable "
                             "predictions (0 = off; e.g. 0.55)")
    parser.add_argument("--benchmark", action="store_true",
                        help="standard ruler: 1500 offline frames replayed "
                             "once, then one 5000-frame free-control round "
                             "and retrain; evaluation cap 5000 frames")
    parser.add_argument("--fast", action="store_true",
                        help="small development preset: 1500 steps, 3 epochs, "
                             "1000 nodes, 400-frame evaluation")
    args = parser.parse_args()
    if args.fast:
        args.steps = 1500
        args.epochs = 3
        args.max_nodes = 1000
        args.cap = 400
        args.horizon = 8
    if args.benchmark:
        # The benchmark fixes both halves of the ruler: the offline leg is
        # the fast bootstrap (1500 frames replayed once - no multi-epoch
        # overfitting) and the online leg is one free-control round of 5000
        # executed frames.  The evaluation cap matches the online budget,
        # so a model that survives is not censored.
        args.steps = 1500
        args.epochs = 1
        args.max_nodes = 1000
        args.horizon = 8
        args.cap = 5000
        args.dagger_cap = 5000
        if args.dagger_rounds == 0:
            args.dagger_rounds = 1
        if args.dagger_frames == 0:
            args.dagger_frames = 5000
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.coverage <= 1.0:
        raise ValueError("--coverage must be in [0, 1]")
    if args.cap < 1:
        raise ValueError("--cap must be positive")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.stage_blocks < 1:
        raise ValueError("--stage-blocks must be positive")
    if args.stage_block_frames < 1:
        raise ValueError("--stage-block-frames must be positive")
    if args.beam_width < 0:
        raise ValueError("--beam-width must be non-negative")
    if args.eval_spacing < 0.0:
        raise ValueError("--eval-spacing must be non-negative")
    if args.milestone_weight < 0.0:
        raise ValueError("--milestone-weight must be non-negative")
    if args.milestone_boundary_k <= 0.0:
        raise ValueError("--milestone-boundary-k must be positive")
    if not (0.0 <= args.veto_fraction < 1.0):
        raise ValueError("--veto-fraction must be in [0, 1)")
    if args.max_nodes < 1:
        raise ValueError("--max-nodes must be positive")
    if args.birth_threshold <= 0.0:
        raise ValueError("--birth-threshold must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not 0.5 < args.stage_memory_quantile < 1.0:
        raise ValueError("--stage-memory-quantile must be in (0.5, 1)")
    if args.regime_value_weight < 0.0:
        raise ValueError("--regime-value-weight must be non-negative")
    if args.regime_n_regimes < 1:
        raise ValueError("--regime-n-regimes must be positive")
    if args.regime_k_stages < 1:
        raise ValueError("--regime-k-stages must be positive")
    if args.regime_lcb_z < 0.0:
        raise ValueError("--regime-lcb-z must be non-negative")
    if args.regime_soft_top_k < 1:
        raise ValueError("--regime-soft-top-k must be positive")
    if args.regime_kernel_tau <= 0.0:
        raise ValueError("--regime-kernel-tau must be positive")
    if not 0.0 <= args.regime_min_coverage <= 1.0:
        raise ValueError("--regime-min-coverage must be in [0, 1]")
    if args.regime_online_every < 0:
        raise ValueError("--regime-online-every must be non-negative")
    if args.regime_online_frames < 0:
        raise ValueError("--regime-online-frames must be non-negative")
    if args.regime_failure_weight < 0.0:
        raise ValueError("--regime-failure-weight must be non-negative")
    if args.regime_failure_sigma <= 0.0:
        raise ValueError("--regime-failure-sigma must be positive")
    if args.regime_failure_prior < 0.0:
        raise ValueError("--regime-failure-prior must be non-negative")
    if args.world_var_sigma <= 0.0:
        raise ValueError("--world-var-sigma must be positive")
    if not 0.0 <= args.regime_abstain_lcb <= 1.0:
        raise ValueError("--regime-abstain-lcb must be in [0, 1]")
    if not 0.0 <= args.world_var_abstain_lcb <= 1.0:
        raise ValueError("--world-var-abstain-lcb must be in [0, 1]")
    if args.chain_max_hops < args.chain_hops:
        raise ValueError("--chain-max-hops must be >= --chain-hops")
    if args.chain_wide_beam_width < 1:
        raise ValueError("--chain-wide-beam-width must be positive")
    if not 0.0 <= args.chain_conf_threshold <= 1.0:
        raise ValueError("--chain-conf-threshold must be in [0, 1]")
    if not 0.0 < args.chain_conf_decay <= 1.0:
        raise ValueError("--chain-conf-decay must be in (0, 1]")
    if not 0.0 <= args.chain_meta_lcb <= 1.0:
        raise ValueError("--chain-meta-lcb must be in [0, 1]")
    if not 0.0 <= args.chain_conf_quantile <= 1.0:
        raise ValueError("--chain-conf-quantile must be in [0, 1]")
    if args.chain_demand_online_every < 0:
        raise ValueError("--chain-demand-online-every must be non-negative")
    if not 0.0 < args.chain_demand_ema <= 1.0:
        raise ValueError("--chain-demand-ema must be in (0, 1]")
    if args.chain_demand_online_frames < 1:
        raise ValueError("--chain-demand-online-frames must be positive")
    if args.chain_demand_min_frames < 1:
        raise ValueError("--chain-demand-min-frames must be positive")
    if args.chain_demand_rel_pairs < 0:
        raise ValueError("--chain-demand-rel-pairs must be non-negative")
    if args.chain_demand_min_terminal < 0:
        raise ValueError("--chain-demand-min-terminal must be non-negative")
    if args.lag < 0:
        raise ValueError("--lag must be non-negative")
    if args.pain_horizon < 0:
        raise ValueError("--pain-horizon must be non-negative")
    if not 0.0 <= args.voi_fraction < 1.0:
        raise ValueError("--voi-fraction must be in [0, 1)")
    if args.voi_target < 1:
        raise ValueError("--voi-target must be positive")
    if args.feature_value_weight < 0.0:
        raise ValueError("--feature-value-weight must be non-negative")
    if args.curiosity_weight < 0.0:
        raise ValueError("--curiosity-weight must be non-negative")
    if not 0.0 <= args.rule_min_coverage <= 1.0:
        raise ValueError("--rule-min-coverage must be in [0, 1]")
    if not 0.0 <= args.rule_min_primitive_coverage <= 1.0:
        raise ValueError("--rule-min-primitive-coverage must be in [0, 1]")
    if args.rule_validation_seed < 0:
        raise ValueError("--rule-validation-seed must be non-negative")
    if args.rule_validation_frames < 1:
        raise ValueError("--rule-validation-frames must be positive")
    if not 0.0 < args.value_quantile < 1.0:
        raise ValueError("--value-quantile must be in (0, 1)")
    if args.component_min_count < 1:
        raise ValueError("--component-min-count must be positive")
    if args.max_components < 0:
        raise ValueError("--max-components must be non-negative")
    if args.component_weight < 0.0:
        raise ValueError("--component-weight must be non-negative")
    if not 0.0 <= args.memory_weight <= 1.0:
        raise ValueError("--memory-weight must be in [0, 1]")
    if not 0.0 <= args.association_rate <= 1.0:
        raise ValueError("--association-rate must be in [0, 1]")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be positive")
    if args.step_stride < 1:
        raise ValueError("--step-stride must be positive")
    if args.dagger_rounds < 0:
        raise ValueError("--dagger-rounds must be non-negative")
    if args.dagger_rounds and args.dagger_episodes < 1:
        raise ValueError("--dagger-episodes must be positive")
    if args.dagger_cap < 1:
        raise ValueError("--dagger-cap must be positive")
    if args.dagger_offset < 0:
        raise ValueError("--dagger-offset must be non-negative")
    if args.dagger_horizon < 0:
        raise ValueError("--dagger-horizon must be non-negative")
    if args.dagger_horizon == 0:
        args.dagger_horizon = args.horizon
    if args.dagger_frames < 0:
        raise ValueError("--dagger-frames must be non-negative")
    if args.dagger_frames > 0 and args.dagger_rounds < 1:
        raise ValueError("--dagger-frames requires --dagger-rounds")
    if args.value_horizon < 0:
        raise ValueError("--value-horizon must be non-negative")
    if args.ood_trust_scale < 0.0:
        raise ValueError("--ood-trust-scale must be non-negative")
    if not 0.0 <= args.obs_trust <= 1.0:
        raise ValueError("--obs-trust must be in [0, 1]")
    if args.feature_action_prior < 0.0:
        raise ValueError("--feature-action-prior must be non-negative")
    if args.feature_value_prior < 0.0:
        raise ValueError("--feature-value-prior must be non-negative")
    component_kinds = tuple(
        kind.strip() for kind in args.components.split(",") if kind.strip()
    )
    eval_seeds = parse_seed_spec(args.eval_seeds)

    seeds = [seed for seed in SEEDS if seed >= args.seed][:args.seeds]
    if len(seeds) < args.seeds:
        seeds = list(range(args.seed, args.seed + args.seeds))
    if eval_seeds:
        overlap = sorted(set(seeds) & set(eval_seeds))
        if overlap:
            print(
                "warning: evaluation seeds overlap training seeds: %s"
                % ",".join(str(seed) for seed in overlap)
            )
    else:
        print(
            "warning: --eval-seeds is empty; using legacy self-evaluation "
            "(training seed == evaluation seed)"
        )
    tasks = [
        dict(
            seed=seed,
            steps=args.steps,
            coverage=args.coverage,
            cap=args.cap,
            max_nodes=args.max_nodes,
            birth_threshold=args.birth_threshold,
            horizon=args.horizon,
            epochs=args.epochs,
            lag=args.lag,
            component_kinds=component_kinds,
            component_min_count=args.component_min_count,
            max_components=args.max_components,
            component_weight=args.component_weight,
            component_gating=args.component_gating,
            memory_weight=args.memory_weight,
            association_rate=args.association_rate,
            induce_features=not args.no_features,
            precursor_labels=args.precursor_labels,
            pain_horizon=args.pain_horizon,
            voi_fraction=args.voi_fraction,
            voi_target=args.voi_target,
            voi_threat_min=args.voi_threat_min,
            feature_value_weight=args.feature_value_weight,
            step_stride=args.step_stride,
            stride_includes_lag=args.stride_includes_lag,
            decision=args.decision,
            stage_blocks=args.stage_blocks,
            stage_block_frames=args.stage_block_frames,
            beam_width=args.beam_width,
            eval_spacing=args.eval_spacing,
            milestone_weight=args.milestone_weight,
            milestone_mode=args.milestone_mode,
                veto_fraction=args.veto_fraction,
            value_horizon=args.value_horizon,
            ood_trust_scale=args.ood_trust_scale,
            obs_trust=args.obs_trust,
            feature_action_prior=args.feature_action_prior,
            feature_value_prior=args.feature_value_prior,
            value_features=not args.no_value_features,
            value_label_quantile=args.value_quantile,
            subgraph_cells=args.subgraph_cells,
            predicate_readouts=args.predicate_readouts,
            event_memory_risk=not args.no_event_memory_risk,
            contrast_objective=args.contrast_objective,
            value_trust_samples=args.value_trust_samples,
            value_dispersion=args.value_dispersion_labels,
            control_risk_weight=args.control_risk_weight,
            value_bank_only=args.value_bank_only,
            iterative_induction=args.iterative_induction,
            compose_readout=args.compose_readout,
            importance_weighting=args.importance_weighting,
            learn_objective=args.learn_objective,
            curiosity_weight=args.curiosity_weight,
            memory_credit=args.memory_credit,
            rule_min_coverage=args.rule_min_coverage,
            rule_min_primitive_coverage=args.rule_min_primitive_coverage,
            rule_validation_seed=args.rule_validation_seed,
            rule_validation_frames=args.rule_validation_frames,
            pairwise_features=args.pairwise_features,
            branch_transitions=args.branch_transitions,
            feature_backfit=args.feature_backfit,
            risk_weight=args.risk_weight,
            risk_decay=args.risk_decay,
            risk_error_weight=args.risk_error_weight,
            memory_feature_backfill=args.memory_feature_backfill,
            action_edge_weight=args.action_edge_weight,
            action_edge_trust_samples=args.action_edge_trust,
            death_rules=args.death_rules,
            death_rule_compose=args.death_rule_compose,
            death_rule_max_thresholds=args.death_rule_thresholds,
            death_rule_max_rules=args.death_rule_max_rules,
            death_rule_max_literals=args.death_rule_max_literals,
            death_rule_boundary_margin=args.death_rule_margin,
            death_rule_envelope=args.death_rule_envelope,
            readout_propagation_gain=args.readout_propagation,
            readout_propagation_steps=args.readout_propagation_steps,
            readout_propagation_edges=args.readout_propagation_edges,
            milestone_stage=args.milestone_stage,
            milestone_boundary_k=args.milestone_boundary_k,
            hazard_readout=args.hazard_readout,
            value_readout=args.value_readout,
            eval_seeds=eval_seeds,
            eval_episodes=args.eval_episodes,
            dagger_rounds=args.dagger_rounds,
            dagger_episodes=args.dagger_episodes,
            dagger_cap=args.dagger_cap,
            dagger_offset=args.dagger_offset,
            dagger_horizon=args.dagger_horizon,
            dagger_frames=args.dagger_frames,
            stage_memory=args.stage_memory,
            stage_memory_quantile=args.stage_memory_quantile,
            stage_memory_feature_source=args.stage_feature_source,
            online_feature_learning=args.online_feature_learning,
            regime_value_weight=args.regime_value_weight,
            regime_n_regimes=args.regime_n_regimes,
            regime_partition=args.regime_partition,
            regime_k_stages=args.regime_k_stages,
            regime_mode=args.regime_mode,
            regime_lcb_z=args.regime_lcb_z,
            regime_value_key=args.regime_value_key,
            regime_soft_top_k=args.regime_soft_top_k,
            regime_kernel_tau=args.regime_kernel_tau,
            regime_min_coverage=args.regime_min_coverage,
            regime_ood_policy=args.regime_ood_policy,
            regime_online_every=args.regime_online_every,
            regime_online_frames=args.regime_online_frames,
            regime_failure_weight=args.regime_failure_weight,
            regime_failure_sigma=args.regime_failure_sigma,
            regime_failure_prior=args.regime_failure_prior,
            regime_abstain_lcb=args.regime_abstain_lcb,
            world_variables=args.world_variables,
            world_var_reset_episode=args.world_var_reset_episode,
            world_var_sigma=args.world_var_sigma,
            world_var_abstain_lcb=args.world_var_abstain_lcb,
            chain_chunk_size=args.chain_chunk_size,
            chain_hops=args.chain_hops,
            chain_beam_width=args.chain_beam_width,
            chain_demand_weight=args.chain_demand_weight,
            chain_demand_online_every=args.chain_demand_online_every,
            chain_demand_ema=args.chain_demand_ema,
            chain_demand_online_frames=args.chain_demand_online_frames,
            chain_demand_min_frames=args.chain_demand_min_frames,
            chain_demand_rel_pairs=args.chain_demand_rel_pairs,
            chain_demand_min_terminal=args.chain_demand_min_terminal,
            chain_adaptive=args.chain_adaptive,
            chain_max_hops=args.chain_max_hops,
            chain_wide_beam_width=args.chain_wide_beam_width,
            chain_conf_threshold=args.chain_conf_threshold,
            chain_conf_decay=args.chain_conf_decay,
            chain_meta_lcb=args.chain_meta_lcb,
            chain_conf_quantile=args.chain_conf_quantile,
        )
        for seed in seeds
    ]
    def _drain(futures):
        """Collect worker results, streaming one line per finished model."""
        rows = []
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            frames = [item["frames"] for item in row["evals"]]
            print(
                " model %3d done: held-out mean %.1f over %d worlds"
                % (row["seed"], float(np.mean(frames)), len(frames)),
                flush=True,
            )
        return rows

    if args.jobs == 1:
        rows = []
        for task in tasks:
            row = run_seed(task)
            frames = [item["frames"] for item in row["evals"]]
            print(
                " model %3d done: held-out mean %.1f over %d worlds"
                % (row["seed"], float(np.mean(frames)), len(frames)),
                flush=True,
            )
            rows.append(row)
    elif FREE_THREADED:
        # The free-threaded build runs seed-level Python and numpy work in
        # parallel without pickling graphs between workers.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows = _drain([
                pool.submit(run_seed, task) for task in tasks
            ])
    else:
        # A GIL build still gets true parallelism from independent seeds; the
        # payload per seed is small (plain dicts and lists).
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = _drain([
                pool.submit(run_seed, task) for task in tasks
            ])
    rows.sort(key=lambda row: row["seed"])

    print("train  concepts  graph_nodes  action_edges  components")
    for row in rows:
        graph = row["graph"]
        print(" %4d  %8d  %11d  %12d  %10d"
              % (row["seed"], graph["prototypes"], graph["nodes"],
                 graph["action_edges"], graph["components"]))

    for row in rows:
        if row["dagger"]:
            parts = [
                "r%d +%d steps (%s frames)"
                % (item["round"], item["planner_steps"],
                   "/".join(str(value) for value in item["planner_frames"]))
                for item in row["dagger"]
            ]
            print(" dagger seed %d: %s" % (row["seed"], ", ".join(parts)))

    eval_rows = [row for result in rows for row in result["evals"]]
    print("\ntrain  eval  ep  env_seed  frames  passes  action_rate")
    for row in eval_rows:
        print(" %5d  %4d  %2d  %8d  %6d  %6d  %10.3f"
              % (row["train_seed"], row["eval_seed"], row["episode"],
                 row["env_seed"], row["frames"], row["passes"],
                 row["action_rate"]))

    per_train = group_summaries(eval_rows, group="train_seed")
    per_eval = group_summaries(eval_rows, group="eval_seed")
    overall = cluster_summary(eval_rows, cluster="train_seed")
    per_train_passes = group_summaries(
        eval_rows, value="passes", group="train_seed"
    )
    overall_passes = cluster_summary(
        eval_rows, value="passes", cluster="train_seed"
    )
    print("\nframes per training seed (mean over evaluation worlds)")
    for key, summary in per_train.items():
        print(" train %4s  n=%2d  mean %6.1f  median %6.1f  range %d-%d"
              % (key, summary["n"], summary["mean"], summary["median"],
                 summary["min"], summary["max"]))
    print("frames per evaluation seed (mean over training seeds)")
    for key, summary in per_eval.items():
        print(" eval  %4s  n=%2d  mean %6.1f  median %6.1f"
              % (key, summary["n"], summary["mean"], summary["median"]))
    print("\nframes overall: n=%d mean %.1f median %.1f IQR %.1f [%.1f, %.1f] "
          "range %d-%d"
          % (overall["n"], overall["mean"], overall["median"],
             overall["iqr"], overall["q1"], overall["q3"],
             overall["min"], overall["max"]))
    print("bootstrap 95%% CI over %d training seeds: [%.1f, %.1f]"
          % (overall["clusters"], overall["cluster_ci_low"],
             overall["cluster_ci_high"]))
    print("\nprogress milestones overall: n=%d mean %.2f median %.1f "
          "range %d-%d  (share of episodes with progress: %.3f)"
          % (overall_passes["n"], overall_passes["mean"],
             overall_passes["median"], int(overall_passes["min"]),
             int(overall_passes["max"]),
             float(np.mean([row["passes"] > 0 for row in eval_rows]))))
    print("bootstrap 95%% CI over %d training seeds: [%.2f, %.2f]"
          % (overall_passes["clusters"], overall_passes["cluster_ci_low"],
             overall_passes["cluster_ci_high"]))
    coverages = [row["milestone_coverage"] for row in eval_rows
                 if row.get("milestone_coverage") is not None]
    if coverages:
        # The milestone memory is learned on the behaviour stream and applied
        # to the planner's; this is how much of the planner's own stream it
        # can actually speak to.  Low coverage is a distribution shift, not a
        # weight to tune.
        coverages = np.asarray(coverages, dtype=float)
        print("milestone memory coverage of the planner's stream: "
              "mean %.3f median %.3f min %.3f  (fraction of frames whose "
              "identity the memory had seen)"
              % (coverages.mean(), float(np.median(coverages)),
                 coverages.min()))
    if args.stage_memory:
        census = rows[0].get("stage_memory")
        if census:
            # Read-only diagnostic: how many realized episodes produced
            # stages, how many stages carry a signal key (the rest share the
            # empty root key), and how often the internal feature masks fired.
            print(
                "stage experience (seed %d): %d/%d episodes, %d stages "
                "(%.2f/episode), keyed %.2f, feature coverage %.2f, "
                "%d demand samples"
                % (
                    rows[0]["seed"], census["episodes"],
                    census["episodes"] + census["episodes_skipped"],
                    census["stages"], census["mean_stages_per_episode"],
                    (
                        float(census["keyed_stages"]) / census["stages"]
                        if census["stages"] else 0.0
                    ),
                    census["feature_coverage"], census["demand_samples"],
                )
            )
        resolver = rows[0].get("stage_resolver")
        if resolver:
            print(
                "stage resolver (seed %d): %d prototypes, %d stage types, "
                "%d training rows, radius %.3f, gates support>=%d / "
                "episodes>=%d / p>=%.2f / margin>=%.2f"
                % (
                    rows[0]["seed"], resolver["prototypes"],
                    resolver["stage_types"], resolver["training_rows"],
                    resolver["radius"], resolver["min_support"],
                    resolver["min_episodes"], resolver["min_probability"],
                    resolver["min_margin"],
                )
            )
        selector = rows[0].get("stage_demand")
        if selector:
            readout = selector["readout"]
            print(
                "stage demand selector (seed %d): readout samples %d, "
                "target mean %+.3f std %.3f, min outcome episodes %d, "
                "planner unwired"
                % (
                    rows[0]["seed"], readout["samples"],
                    readout["target_mean"], readout["target_std"],
                    selector["min_outcome_episodes"],
                )
            )
        heuristic = rows[0].get("stage_demand_heuristic")
        if heuristic:
            print(
                "stage demand heuristic (seed %d): %d samples, "
                "%d context-feature pairs, %d context outcomes, "
                "feature source %s"
                % (
                    rows[0]["seed"], heuristic["samples"],
                    heuristic["context_feature_pairs"],
                    heuristic["context_outcomes"],
                    rows[0].get("stage_memory", {}).get("feature_source"),
                )
            )
        tree = rows[0].get("stage_tree")
        if tree:
            print(
                "stage tree (seed %d): %d types, %d merges, %d roots, "
                "max level %d"
                % (
                    rows[0]["seed"], tree["stage_types"], tree["merges"],
                    tree["roots"], tree["max_level"],
                )
            )
        tree_heuristic = rows[0].get("stage_tree_heuristic")
        if tree_heuristic:
            print(
                "stage tree heuristic (seed %d): %d samples, "
                "%d context-feature pairs, %d context outcomes"
                % (
                    rows[0]["seed"], tree_heuristic["samples"],
                    tree_heuristic["context_feature_pairs"],
                    tree_heuristic["context_outcomes"],
                )
            )
    if args.regime_value_weight > 0.0 and args.regime_online_every > 0:
        rc = rows[0].get("regime_continuation") or {}
        online = rc.get("online")
        stats = rc.get("stats") or {}
        if online:
            decisions = int(stats.get("decisions", 0))
            active_rate = (
                float(stats.get("active", 0)) / decisions
                if decisions else 0.0
            )
            fm = online.get("failure_memory")
            fm_note = (
                ", fail_mem=%d deaths (%d/%d leaves)"
                % (
                    fm["total_failures"],
                    fm["active_failure_leaves"],
                    fm["active_failure_leaves"] + fm["compressed_leaves"],
                )
                if fm else ""
            )
            print(
                "regime online (seed %d): %d updates over %d eval episodes "
                "(every %d ep, buffered %d ep / %d frames, active %.3f%s)"
                % (
                    rows[0]["seed"],
                    online["updates"],
                    online["episodes_seen"],
                    online["update_every"],
                    online["online_episodes_buffered"],
                    online["online_frames_buffered"],
                    active_rate,
                    fm_note,
                )
            )
    if args.world_variables:
        wv = rows[0].get("world_variables")
        if wv:
            print(
                "world variables (seed %d): active_channels=%s, updates=%d, "
                "skipped_jumps=%d, changes=%d, cred=%.3f, sig_norm=%.3f"
                % (
                    rows[0]["seed"],
                    wv["active_channels"],
                    wv["updates"],
                    wv["skipped_jumps"],
                    wv["changes"],
                    wv["credibility"],
                    wv["signature_norm"],
                )
            )
    if overall["clusters"] < 5:
        print("warning: fewer than 5 trained models; the interval is "
              "descriptive, not a significance test")

    if args.report:
        report = dict(
            config=dict(
                training_seeds=seeds,
                eval_seeds=eval_seeds,
                eval_episodes=args.eval_episodes,
                steps=args.steps,
                coverage=args.coverage,
                cap=args.cap,
                horizon=args.horizon,
                epochs=args.epochs,
                max_nodes=args.max_nodes,
                lag=args.lag,
                step_stride=args.step_stride,
                stride_includes_lag=args.stride_includes_lag,
                decision=args.decision,
                stage_blocks=args.stage_blocks,
                stage_block_frames=args.stage_block_frames,
                beam_width=args.beam_width,
                milestone_weight=args.milestone_weight,
                milestone_mode=args.milestone_mode,
                veto_fraction=args.veto_fraction,
                eval_spacing=args.eval_spacing,
                value_horizon=args.value_horizon,
                ood_trust_scale=args.ood_trust_scale,
                obs_trust=args.obs_trust,
                feature_action_prior=args.feature_action_prior,
                feature_value_prior=args.feature_value_prior,
                feature_value_weight=args.feature_value_weight,
                features=not args.no_features,
                precursor_labels=args.precursor_labels,
                value_features=not args.no_value_features,
                value_label_quantile=args.value_quantile,
                subgraph_cells=args.subgraph_cells,
                predicate_readouts=args.predicate_readouts,
                event_memory_risk=not args.no_event_memory_risk,
                contrast_objective=args.contrast_objective,
                value_trust_samples=args.value_trust_samples,
                value_dispersion=args.value_dispersion_labels,
                control_risk_weight=args.control_risk_weight,
                value_bank_only=args.value_bank_only,
                iterative_induction=args.iterative_induction,
                self_evaluation=not bool(eval_seeds),
                dagger_rounds=args.dagger_rounds,
                dagger_episodes=args.dagger_episodes,
                dagger_cap=args.dagger_cap,
                dagger_offset=args.dagger_offset,
                dagger_horizon=args.dagger_horizon,
                stage_memory=args.stage_memory,
                stage_memory_quantile=args.stage_memory_quantile,
                stage_feature_source=args.stage_feature_source,
                regime_value_weight=args.regime_value_weight,
                regime_online_every=args.regime_online_every,
                regime_online_frames=args.regime_online_frames,
                regime_failure_weight=args.regime_failure_weight,
                regime_failure_sigma=args.regime_failure_sigma,
                regime_failure_prior=args.regime_failure_prior,
                regime_abstain_lcb=args.regime_abstain_lcb,
                world_variables=args.world_variables,
                world_var_reset_episode=args.world_var_reset_episode,
                world_var_sigma=args.world_var_sigma,
                world_var_abstain_lcb=args.world_var_abstain_lcb,
                chain_adaptive=args.chain_adaptive,
                chain_max_hops=args.chain_max_hops,
                chain_wide_beam_width=args.chain_wide_beam_width,
                chain_conf_threshold=args.chain_conf_threshold,
                chain_conf_decay=args.chain_conf_decay,
                chain_meta_lcb=args.chain_meta_lcb,
                chain_conf_quantile=args.chain_conf_quantile,
                chain_demand_online_every=args.chain_demand_online_every,
                chain_demand_ema=args.chain_demand_ema,
                chain_demand_online_frames=args.chain_demand_online_frames,
                chain_demand_min_frames=args.chain_demand_min_frames,
                chain_demand_rel_pairs=args.chain_demand_rel_pairs,
                chain_demand_min_terminal=args.chain_demand_min_terminal,
            ),
            rows=eval_rows,
            milestones=[
                dict(seed=row["seed"], summary=row["milestone"])
                for row in rows
            ],
            dagger=[
                dict(seed=row["seed"], rounds=row["dagger"])
                for row in rows
            ],
            stage_memory=[
                dict(seed=row["seed"], census=row["stage_memory"])
                for row in rows
                if row.get("stage_memory") is not None
            ],
            stage_resolver=[
                dict(seed=row["seed"], census=row["stage_resolver"])
                for row in rows
                if row.get("stage_resolver") is not None
            ],
            stage_demand=[
                dict(seed=row["seed"], census=row["stage_demand"])
                for row in rows
                if row.get("stage_demand") is not None
            ],
            stage_demand_heuristic=[
                dict(seed=row["seed"], census=row["stage_demand_heuristic"])
                for row in rows
                if row.get("stage_demand_heuristic") is not None
            ],
            stage_tree=[
                dict(seed=row["seed"], census=row["stage_tree"])
                for row in rows
                if row.get("stage_tree") is not None
            ],
            stage_tree_heuristic=[
                dict(seed=row["seed"], census=row["stage_tree_heuristic"])
                for row in rows
                if row.get("stage_tree_heuristic") is not None
            ],
            per_train_seed=per_train,
            per_eval_seed=per_eval,
            overall=overall,
            per_train_seed_passes=per_train_passes,
            overall_passes=overall_passes,
        )
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("wrote %s" % args.report)

    print("\nlearned high-pain nodes (seed %d):" % rows[0]["seed"])
    for node in rows[0]["top_nodes"]:
        features = node["features"]
        preview = (
            "n/a" if features is None
            else "[" + ", ".join("%+.2f" % value for value in features[:6]) + "]"
        )
        print("  %-8s count=%4d pain=%.3f dopamine=%.2f state=%s"
              % (node["id"], node["count"], node["pain"],
                 node["dopamine"], preview))

    print("\nlearned high-pain action edges (seed %d):" % rows[0]["seed"])
    for edge in rows[0]["top_edges"]:
        print("  %-8s -a%d-> %-8s count=%4d pain=%.3f dopamine=%.2f"
              % (edge["source"], edge["action"], edge["target"],
                 edge["count"], edge["pain"], edge["dopamine"]))

