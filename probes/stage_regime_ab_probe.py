"""Paired held-out benchmark A/B for opt-in regime continuation value.

Evaluates baseline (``stage_value_weight=0.0``) against calibrated regime
continuation risk (``RegimeContinuationEvaluator`` fitted strictly on each
seed's training transitions) across multiple training seeds and held-out worlds.

Read-only probe; does not modify default training or decision paths.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("core", "modules", "probes"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage_regime import (
    CalibratedRegimeTransitionModel,
    ControllableFeatureDemand,
    RegimeContinuationEvaluator,
    StageRegimeSummarizer,
)
from substrate_nodes import LAG, _curiosity_off, evaluate, fit

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()
DEFAULT_JOBS = min(16, os.cpu_count() or 1) if FREE_THREADED else 1


def _warm_agent_readout_cache(agent):
    """Populate read-only caches once before multi-threaded evaluation."""
    if hasattr(agent, "_prototype_geometry"):
        agent._prototype_geometry()
    if hasattr(agent, "_prototype_associations"):
        agent._prototype_associations()
    if hasattr(agent, "_prototype_memory_index"):
        agent._prototype_memory_index()
    if hasattr(agent, "_prototype_columns"):
        agent._prototype_columns()
    if hasattr(agent, "_compiled_feature_plan"):
        agent._compiled_feature_plan()
    if hasattr(agent, "_prototype_memory_attribute_vectors"):
        agent._prototype_memory_attribute_vectors("value")
        agent._prototype_memory_attribute_vectors("dopamine")


def _eval_clone(agent):
    """Thread-local shallow clone of ``agent`` with curiosity disabled."""
    clone = copy.copy(agent)
    clone.curiosity_weight = 0.0
    if isinstance(getattr(agent, "_readout_cache", None), dict):
        clone._readout_cache = dict(agent._readout_cache)
    return clone


def _load_or_fit_model(seed: int, cache_prefix: str = "b0"):
    candidates = (
        os.path.join(str(_ROOT), "results", "models", f"{cache_prefix}-{seed}.pkl"),
        f"{cache_prefix}-{seed}.pkl",
    )
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as handle:
                bundle = pickle.load(handle)
            return bundle["world"], bundle["agent"]
    return fit(
        seed, steps=1500, coverage=0.1, max_nodes=1000, epochs=1,
        dagger_rounds=1, dagger_frames=5000, dagger_cap=5000,
        dagger_horizon=8,
    )


def _eval_one_episode(world, agent, eval_seed, horizon, cap,
                      evaluator=None, stage_value_weight=0.0,
                      decision="search", chain_chunk_size=2, chain_hops=3,
                      chain_beam_width=2, heuristic_demand=None,
                      heuristic_demand_weight=0.0):
    worker_agent = _eval_clone(agent)
    cb_stats = {}
    cb = (
        evaluator.as_stage_value_callback(stats=cb_stats)
        if evaluator is not None and float(stage_value_weight) > 0.0
        else None
    )
    metrics = {}
    frames, _actions, _rec = evaluate(
        int(eval_seed) * 10000, world, worker_agent,
        cap=int(cap), horizon=int(horizon),
        decision=decision,
        stage_value=cb,
        stage_value_weight=float(stage_value_weight),
        chain_chunk_size=int(chain_chunk_size),
        chain_hops=int(chain_hops),
        chain_beam_width=int(chain_beam_width),
        heuristic_demand=heuristic_demand,
        heuristic_demand_weight=float(heuristic_demand_weight),
        metrics_out=metrics,
    )
    return (
        int(frames),
        int(metrics.get("passes", 0)),
        int(cb_stats.get("decisions", 0)),
        int(cb_stats.get("active", 0)),
    )


def evaluate_arm(world, agent, eval_seeds, horizon, cap,
                 stage_value=None, stage_value_weight=0.0):
    frames_list = []
    passes_list = []
    with _curiosity_off(agent):
        for es in eval_seeds:
            metrics = {}
            frames, _actions, _rec = evaluate(
                int(es) * 10000, world, agent,
                cap=int(cap), horizon=int(horizon),
                stage_value=stage_value,
                stage_value_weight=float(stage_value_weight),
                metrics_out=metrics,
            )
            frames_list.append(int(frames))
            passes_list.append(int(metrics.get("passes", 0)))
    return frames_list, passes_list


def run_ab(seeds, eval_seeds, horizons, weights, mode="lcb",
           n_regimes=16, min_dwell=6, hysteresis=2,
           continuation_stages=8, cap=1200, cache_prefix="b0",
           min_decision_coverage=0.80, ood_policy="neutral_mean",
           value_key="cum_survival", soft_top_k=4, kernel_tau=0.7,
           lcb_z=1.0, min_action_episodes=4, partition="survival_tree",
           decision="search", chain_chunk_size=2, chain_hops=3,
           chain_beam_width=2, chain_demand_weight=0.10,
           jobs: int = DEFAULT_JOBS):
    t_start = time.perf_counter()
    jobs = max(1, int(jobs))

    def _prepare_seed(seed):
        world, agent = _load_or_fit_model(seed, cache_prefix=cache_prefix)
        _warm_agent_readout_cache(agent)
        rows = agent.milestone_training_rows
        summ = StageRegimeSummarizer(
            n_regimes=n_regimes, min_dwell=min_dwell, hysteresis=hysteresis,
            partition=partition,
        ).fit_rows(rows, lag=LAG)
        tr_eps = summ.transform_rows(rows, lag=LAG)
        tm = CalibratedRegimeTransitionModel(
            max_depth=2, min_episodes=2,
        ).fit(tr_eps, use_action=True)
        vals = tm.continuation_values(
            len(summ.centers_), horizon_stages=continuation_stages, mode=mode,
            min_action_episodes=min_action_episodes, lcb_z=lcb_z,
        )
        evaluator = RegimeContinuationEvaluator(
            summ, vals, min_regime_episodes=2,
            min_decision_coverage=min_decision_coverage,
            ood_policy=ood_policy,
            value_key=value_key,
            soft_top_k=soft_top_k,
            kernel_tau=kernel_tau,
        )
        demand = (
            ControllableFeatureDemand().fit_rows(world, agent, rows, lag=LAG)
            if decision == "chained" and float(chain_demand_weight) > 0.0
            else None
        )
        return seed, world, agent, evaluator, demand

    models = {}
    evaluators = {}
    demands = {}
    if jobs > 1 and len(seeds) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(seeds))) as pool:
            prepared = list(pool.map(_prepare_seed, seeds))
    else:
        prepared = [_prepare_seed(s) for s in seeds]
    for seed, world, agent, evaluator, demand in prepared:
        models[seed] = (world, agent)
        evaluators[seed] = evaluator
        demands[seed] = demand
    t_fit = time.perf_counter() - t_start

    def _run_grid(horizon, weight, arm_decision="search"):
        tasks = [
            (seed, es) for seed in seeds for es in eval_seeds
        ]

        def _worker(pair):
            seed, es = pair
            world, agent = models[seed]
            ev = evaluators[seed] if float(weight) > 0.0 else None
            dem = (
                demands[seed].distance_batch
                if arm_decision == "chained" and demands[seed] is not None
                else None
            )
            return _eval_one_episode(
                world, agent, es, horizon, cap,
                evaluator=ev, stage_value_weight=weight,
                decision=arm_decision,
                chain_chunk_size=chain_chunk_size,
                chain_hops=chain_hops,
                chain_beam_width=chain_beam_width,
                heuristic_demand=dem,
                heuristic_demand_weight=(
                    chain_demand_weight if dem is not None else 0.0
                ),
            )

        if jobs > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
                outs = list(pool.map(_worker, tasks))
        else:
            outs = [_worker(pair) for pair in tasks]

        n_eval = len(eval_seeds)
        by_seed_frames = {}
        by_seed_passes = {}
        tot_dec, tot_act = 0, 0
        for idx, seed in enumerate(seeds):
            chunk = outs[idx * n_eval:(idx + 1) * n_eval]
            by_seed_frames[seed] = [c[0] for c in chunk]
            by_seed_passes[seed] = [c[1] for c in chunk]
            tot_dec += sum(c[2] for c in chunk)
            tot_act += sum(c[3] for c in chunk)
        return by_seed_frames, by_seed_passes, tot_dec, tot_act

    summary = {}
    for horizon in horizons:
        t_h = time.perf_counter()
        print(f"\n{'=' * 74}")
        print(f"PAIRED A/B: horizon={horizon}  part={partition}  mode={mode}(z={lcb_z})  val={value_key}  "
              f"k_stages={continuation_stages}  soft_k={soft_top_k}(tau={kernel_tau})  "
              f"min_cov={min_decision_coverage:.2f}  ood={ood_policy}  jobs={jobs}  "
              f"seeds={seeds}  eval={eval_seeds[0]}-{eval_seeds[-1]} (n={len(eval_seeds)}/seed)  cap={cap}")
        print(f"{'=' * 74}")
        bf_map, bp_map, _, _ = _run_grid(horizon, 0.0)
        baselines = {s: (bf_map[s], bp_map[s]) for s in seeds}
        all_bf = [f for s in seeds for f in baselines[s][0]]
        all_bp = [p for s in seeds for p in baselines[s][1]]
        print(
            f"  control (w=0.00): mean={np.mean(all_bf):.1f} median={np.median(all_bf):.1f} "
            f"passes={sum(all_bp)} | per-seed={[round(float(np.mean(baselines[s][0])), 1) for s in seeds]}"
        )
        horizon_res = dict(
            control_mean=float(np.mean(all_bf)),
            control_median=float(np.median(all_bf)),
            control_passes=int(sum(all_bp)),
            per_seed_control={int(s): baselines[s][0] for s in seeds},
            arms={},
        )
        for weight in weights:
            arm_frames, arm_passes, tot_dec, tot_act = _run_grid(
                horizon, weight, arm_decision=decision,
            )
            seed_deltas = []
            wins, ties, losses = 0, 0, 0
            for seed in seeds:
                af = arm_frames[seed]
                bf = baselines[seed][0]
                seed_deltas.append(float(np.mean(af) - np.mean(bf)))
                for a, b in zip(af, bf):
                    if a > b:
                        wins += 1
                    elif a < b:
                        losses += 1
                    else:
                        ties += 1
            all_af = [f for s in seeds for f in arm_frames[s]]
            all_ap = [p for s in seeds for p in arm_passes[s]]
            delta = float(np.mean(all_af) - np.mean(all_bf))
            pos_seeds = sum(1 for d in seed_deltas if d > 0)
            act_rate = tot_act / max(tot_dec, 1)
            print(
                f"  regime  (w={weight:.2f}): mean={np.mean(all_af):.1f} median={np.median(all_af):.1f} "
                f"delta={delta:+.1f} passes={sum(all_ap)} ({sum(all_ap)-sum(all_bp):+d}) "
                f"W/T/L={wins}/{ties}/{losses} pos_seeds={pos_seeds}/{len(seeds)} "
                f"active={act_rate:.2f} | seed_deltas={[round(d, 1) for d in seed_deltas]}"
            )
            horizon_res["arms"][str(weight)] = dict(
                mean=float(np.mean(all_af)),
                median=float(np.median(all_af)),
                delta=delta,
                passes=int(sum(all_ap)),
                wins=wins, ties=ties, losses=losses,
                pos_seeds=pos_seeds,
                active_rate=act_rate,
                seed_deltas=seed_deltas,
            )
        elapsed_h = time.perf_counter() - t_h
        print(f"  [elapsed: fit={t_fit:.2f}s, horizon_eval={elapsed_h:.2f}s]")
        summary[f"h{horizon}"] = horizon_res
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Paired held-out benchmark A/B for regime continuation value."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--eval-start", type=int, default=120)
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.50, 1.00])
    parser.add_argument("--n-regimes", type=int, default=16)
    parser.add_argument("--partition", choices=("survival_tree", "kmeans"), default="survival_tree")
    parser.add_argument("--decision", choices=("search", "chained"), default="search")
    parser.add_argument("--chain-chunk-size", type=int, default=2)
    parser.add_argument("--chain-hops", type=int, default=3)
    parser.add_argument("--chain-beam-width", type=int, default=2)
    parser.add_argument("--chain-demand-weight", type=float, default=0.10)
    parser.add_argument("--mode", choices=("pooled", "optimal", "lcb"), default="lcb")
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--k-stages", type=int, default=8)
    parser.add_argument("--value-key", choices=("survival", "cum_survival"), default="cum_survival")
    parser.add_argument("--soft-top-k", type=int, default=4)
    parser.add_argument("--kernel-tau", type=float, default=0.7)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument(
        "--ood-policy",
        choices=("baseline", "neutral_mean", "conservative"),
        default="neutral_mean",
    )
    parser.add_argument("--cap", type=int, default=1200)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    eval_seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    summary = run_ab(
        seeds=args.seeds,
        eval_seeds=eval_seeds,
        horizons=args.horizons,
        weights=args.weights,
        mode=args.mode,
        n_regimes=args.n_regimes,
        continuation_stages=args.k_stages,
        cap=args.cap,
        min_decision_coverage=args.min_coverage,
        ood_policy=args.ood_policy,
        value_key=args.value_key,
        soft_top_k=args.soft_top_k,
        kernel_tau=args.kernel_tau,
        lcb_z=args.lcb_z,
        partition=args.partition,
        decision=args.decision,
        chain_chunk_size=args.chain_chunk_size,
        chain_hops=args.chain_hops,
        chain_beam_width=args.chain_beam_width,
        chain_demand_weight=args.chain_demand_weight,
        jobs=args.jobs,
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
